//===- AIERegDefUseTracker.cpp - Track Register Live Ranges --------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
//
// This file implements tracking and analysis of register live ranges in a
// MachineBasicBlock. The tracker performs the following:
// - Identifies register definitions and uses that form live ranges
// - Merges aliasing register accesses into unified live ranges
// - Filters out unsafe ranges (tied operands, live-in/out, implicit uses)
// - Computes appropriate register classes for each live range
// - Optionally replaces physical registers with virtual registers for testing
//
//===----------------------------------------------------------------------===//

#include "AIERegDefUseTracker.h"
#include "AIEBaseInstrInfo.h"
#include "AIEBaseRegisterInfo.h"
#include "Utils/AIEMachineInstrPrint.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aie-reg-liverange"

using namespace llvm;

namespace {

/// Check if a register overlaps with a RegisterMaskPair (live-in/out entry).
/// Currently uses conservative full-register overlap; lane mask support can
/// be added later.
bool overlapsRMP(MCRegister Reg, const MachineBasicBlock::RegisterMaskPair &RMP,
                 const TargetRegisterInfo *TRI) {
  return TRI->regsOverlap(Reg, RMP.PhysReg);
}

} // end anonymous namespace

void RegLiveRange::dumpBrief(const TargetRegisterInfo *TRI) const {
  StringRef Name =
      (BaseReg != MCRegister::NoRegister) ? TRI->getName(BaseReg) : "unknown";

  dbgs() << "  - LR#" << ID << " Base=" << Name << " defs=" << getNumDefs()
         << " uses=" << getNumUses();

  if (IsReserved) {
    dbgs() << " [RESERVED]";
  }

  // Print first def if available
  if (!Defs.empty()) {
    const MachineInstr *MI = Defs[0].getOperand()->getParent();
    assert(MI && "Def operand must have a parent instruction");
    dbgs() << " firstDef: " << AIE::NoDebug(*MI);
  }

  dbgs() << "\n";
}

static cl::opt<std::string> ExcludeLiveRangesByRegClass(
    "aie-exclude-liveranges-by-regclass", cl::Hidden, cl::init(""),
    cl::desc("[AIE] Exclude live ranges of the specified register class name. "
             "Empty string means no filtering."));

static cl::opt<bool> AddUnusedCallerSavedRegs(
    "aie-add-unused-caller-saved-regs", cl::Hidden, cl::init(false),
    cl::desc("[AIE] Add unused caller-saved registers to the available "
             "register pool for pipelining. Only safe when loops with calls "
             "are excluded from pipelining."));

RegLiveRangeTracker::RegLiveRangeTracker(MachineBasicBlock &MBB)
    : MF(MBB.getParent()), TRI(MF->getSubtarget().getRegisterInfo()),
      TII(static_cast<const AIEBaseInstrInfo *>(
          MF->getSubtarget().getInstrInfo())) {
  assert(MF && "MachineFunction cannot be null");
  assert(TRI && "TargetRegisterInfo cannot be null");
  assert(TII && "TargetInstrInfo cannot be null");
}

void RegLiveRange::addDef(MachineOperand *DefOp, unsigned SubRegIdx) {
  Defs.emplace_back(DefOp, SubRegIdx);
}

void RegLiveRange::addUse(MachineOperand *UseOp, unsigned SubRegIdx) {
  Uses.emplace_back(UseOp, SubRegIdx);
}

/// Get the sub-register index if AccessReg is a sub-register of BaseReg
/// Returns 0 if AccessReg is not a sub-register of BaseReg
unsigned RegLiveRangeTracker::getSubRegIndex(MCRegister AccessReg,
                                             MCRegister BaseReg) const {
  if (AccessReg == BaseReg)
    return 0;

  // Check if AccessReg is a sub-register of BaseReg
  for (MCSubRegIndexIterator SubRegIdxIt(BaseReg, TRI); SubRegIdxIt.isValid();
       ++SubRegIdxIt) {
    if (SubRegIdxIt.getSubReg() == AccessReg) {
      return SubRegIdxIt.getSubRegIndex();
    }
  }

  return 0;
}

bool RegLiveRangeTracker::overlapsAnyInSet(
    MCRegister Reg, const DenseSet<MCRegister> &RegSet) const {
  for (MCRegister R : RegSet) {
    if (TRI->regsOverlap(Reg, R))
      return true;
  }
  return false;
}

bool RegLiveRangeTracker::startsWithDefInBlock(const RegLiveRange &LR) const {
  if (LR.getNumDefs() == 0)
    return false;

  // Find the earliest instruction index among all operands
  unsigned EarliestIdx = UINT_MAX;
  bool EarliestIsDef = false;

  for (const auto &Def : LR.defs()) {
    const MachineInstr *MI = Def.getOperand()->getParent();
    const auto It = InstrOrder.find(MI);
    if (It != InstrOrder.end() && It->second < EarliestIdx) {
      EarliestIdx = It->second;
      EarliestIsDef = true;
    }
  }

  for (const auto &Use : LR.uses()) {
    const MachineInstr *MI = Use.getOperand()->getParent();
    const auto It = InstrOrder.find(MI);
    if (It != InstrOrder.end() && It->second < EarliestIdx) {
      EarliestIdx = It->second;
      EarliestIsDef = false;
    }
  }

  return EarliestIsDef;
}

bool RegLiveRangeTracker::isFullyDefined(
    const RegLiveRange &LR, const DenseMap<MCRegister, int> &LiveRegs) const {
  // A live range is fully defined if its base register does not overlap
  // with any register still in LiveRegs. If it overlaps, it means some
  // part of the register is still live from before the block (incomplete def).
  return !llvm::any_of(LiveRegs, [&](const auto &Entry) {
    return TRI->regsOverlap(LR.BaseReg, Entry.first);
  });
}

bool RegLiveRangeTracker::hasTiedOperands(const RegLiveRange &LR) const {
  assert(TII);

  // Check if any operand in this live range is tied
  for (const auto &Def : LR.defs()) {
    MachineOperand *MO = Def.getOperand();
    if (MO->isTied())
      return true;

    MachineInstr *MI = MO->getParent();
    assert(MI);

    // Get the operand index for this def
    unsigned OpIdx = MO->getOperandNo();

    // Check AIE-specific tied register info
    const auto TiedInfo = TII->getTiedRegInfo(*MI);
    for (const auto &TiedSet : TiedInfo) {
      // Check if this operand is in the destination operands of a tied set
      for (const auto &DstOp : TiedSet.DstOps) {
        if (DstOp.OpIdx == OpIdx)
          return true;
      }
      // Check if this operand is in the source operands of a tied set
      for (const auto &SrcOp : TiedSet.SrcOps) {
        if (SrcOp.OpIdx == OpIdx)
          return true;
      }
    }

    const MCRegister R = MO->getReg().asMCReg();
    const int DefIdx = MI->findRegisterDefOperandIdx(R, TRI);
    if (DefIdx >= 0 && MI->isRegTiedToUseOperand(DefIdx))
      return true;
  }

  // Also check uses for tied operands
  for (const auto &Use : LR.uses()) {
    MachineOperand *MO = Use.getOperand();

    // Get the operand index for this use
    unsigned OpIdx = MO->getOperandNo();

    MachineInstr *MI = Use.getOperand()->getParent();
    assert(MI);

    // Check AIE-specific tied register info
    const auto TiedInfo = TII->getTiedRegInfo(*MI);
    for (const auto &TiedSet : TiedInfo) {
      // Check if this operand is in the source operands of a tied set
      for (const auto &SrcOp : TiedSet.SrcOps) {
        if (SrcOp.OpIdx == OpIdx)
          return true;
      }
    }
  }

  return false;
}

void RegLiveRangeTracker::pruneByFullCoverage() {
  LLVM_DEBUG(dbgs() << "\nPrune by full coverage: " << LiveRanges.size()
                    << " ranges before pruning\n");

  // We run this in a fixed point loop, since pruning a range may uncover ranges
  // that were previously covered by it.
  bool Changed = true;
  while (Changed) {
    Changed = false;

    // Build coverage map from current LiveRanges
    DenseSet<MachineOperand *> CoveredOps;
    for (const RegLiveRange &LR : LiveRanges) {
      for (const auto &R : LR.operands()) {
        CoveredOps.insert(R.getOperand());
      }
    }

    // Check if there are any uncovered operands that alias with this LR's
    // registers
    auto HasUncoveredAlias = [&](const DenseSet<MCRegister> &LRRegs,
                                 MCRegister *SampleUncovered = nullptr) {
      for (MachineOperand *MO : AllPhysRegOperands) {
        if (!CoveredOps.contains(MO)) {
          MCRegister UncoveredReg = MO->getReg().asMCReg();
          // Check if this uncovered operand aliases with any register in this
          // LR
          for (const MCRegister LRReg : LRRegs) {
            if (TRI->regsOverlap(UncoveredReg, LRReg)) {
              if (SampleUncovered)
                *SampleUncovered = UncoveredReg;
              return true;
            }
          }
        }
      }
      return false;
    };

    // For each live range, check if ALL operands of its register group are
    // covered
    SmallVector<RegLiveRange, 16> NewLiveRanges;
    for (const RegLiveRange &LR : LiveRanges) {
      // Collect all registers used in this live range
      DenseSet<MCRegister> LRRegs;
      for (const auto &R : LR.operands()) {
        LRRegs.insert(R.getOperand()->getReg().asMCReg());
      }

      MCRegister SampleUncovered = MCRegister::NoRegister;
      if (!HasUncoveredAlias(LRRegs, &SampleUncovered)) {
        NewLiveRanges.push_back(LR);
      } else {
        LLVM_DEBUG({
          dbgs() << "Reject: pruned by full coverage";
          if (SampleUncovered != MCRegister::NoRegister)
            dbgs() << " (uncovered alias " << TRI->getName(SampleUncovered)
                   << ")";
          dbgs() << ": ";
          LR.dumpBrief(TRI);
        });
        Changed = true;
      }
    }

    LiveRanges = std::move(NewLiveRanges);
  }

  LLVM_DEBUG(dbgs() << "After pruning: " << LiveRanges.size() << " ranges\n");

#ifndef NDEBUG
  // Verify that all remaining operands are covered
  DenseSet<MachineOperand *> FinalCoveredOps;
  for (const RegLiveRange &LR : LiveRanges) {
    for (const auto &R : LR.operands()) {
      FinalCoveredOps.insert(R.getOperand());
    }
  }

  for (MachineOperand *MO : AllPhysRegOperands) {
    if (!FinalCoveredOps.contains(MO)) {
      const MCRegister U = MO->getReg().asMCReg();
      // Verify no LR overlaps with this uncovered operand
      for (const RegLiveRange &LR : LiveRanges) {
        for (const auto &R : LR.operands()) {
          assert(!TRI->regsOverlap(U, R.getOperand()->getReg().asMCReg()) &&
                 "Uncovered operand overlaps with kept live range!");
        }
      }
    }
  }
#endif
}

void RegLiveRangeTracker::mergeAliasingLiveRanges(
    unsigned DefLRIdx, MCRegister DefReg, DenseMap<MCRegister, int> &LiveRegs,
    DenseMap<MachineOperand *, unsigned> &OperandToLiveRange) {

  // Collect all aliasing live registers and their live ranges
  SmallVector<std::pair<MCRegister, int>, 8> AliasingLiveRegs;
  for (const auto &[LiveReg, LiveLRIdx] : LiveRegs) {
    if (TRI->regsOverlap(DefReg, LiveReg)) {
      AliasingLiveRegs.push_back({LiveReg, LiveLRIdx});
    }
  }

  if (AliasingLiveRegs.empty())
    return;

  // Collect all unique live range indices to merge (including the def's).
  // Skip NoLiveRange sentinels as they don't have actual ranges yet.
  DenseSet<unsigned> ToMerge;
  ToMerge.insert(DefLRIdx);
  for (const auto &[LiveReg, LRIdx] : AliasingLiveRegs) {
    if (LRIdx != RegLiveRange::NoLiveRange) {
      ToMerge.insert(static_cast<unsigned>(LRIdx));
    }
  }

  // Find the base register (largest among all involved registers)
  MCRegister BaseReg = DefReg;
  for (const auto &[LiveReg, LRIdx] : AliasingLiveRegs) {
    if (getSubRegIndex(BaseReg, LiveReg) != 0) {
      // LiveReg is larger than current base
      BaseReg = LiveReg;
    }
  }

  // Use DefLRIdx as the target for merging
  const unsigned MergedLRIdx = DefLRIdx;
  LiveRanges[MergedLRIdx].BaseReg = BaseReg;

  // Rebuild the def's live range with correct SubRegIdx
  RegLiveRange NewMergedLR;
  NewMergedLR.ID = LiveRanges[MergedLRIdx].ID; // Preserve the ID
  NewMergedLR.BaseReg = BaseReg;
  NewMergedLR.RegisterClass = LiveRanges[MergedLRIdx].RegisterClass;

  // Propagate reserved status: if any merged range is reserved, the result is
  // reserved.
  bool IsReserved = LiveRanges[MergedLRIdx].isReserved();
  for (const auto &[LiveReg, LRIdx] : AliasingLiveRegs) {
    if (LRIdx != RegLiveRange::NoLiveRange && LiveRanges[LRIdx].isReserved()) {
      IsReserved = true;
      break;
    }
  }

  // Also check if any subreg of the merged base register is live-out.
  // Live-out registers are marked with NoLiveRange sentinel in LiveRegs.
  if (!IsReserved) {
    for (MCSubRegIterator SubIt(BaseReg, TRI, /*IncludeSelf=*/true);
         SubIt.isValid(); ++SubIt) {
      auto It = LiveRegs.find(*SubIt);
      if (It != LiveRegs.end() && It->second == RegLiveRange::NoLiveRange) {
        IsReserved = true;
        break;
      }
    }
  }

  NewMergedLR.setIsReserved(IsReserved);
  for (const auto &DefInfo : LiveRanges[MergedLRIdx].defs()) {
    const MCRegister DefRegister = DefInfo.getOperand()->getReg().asMCReg();
    NewMergedLR.addDef(DefInfo.getOperand(),
                       getSubRegIndex(DefRegister, BaseReg));
  }
  for (const auto &UseInfo : LiveRanges[MergedLRIdx].uses()) {
    const MCRegister UseReg = UseInfo.getOperand()->getReg().asMCReg();
    NewMergedLR.addUse(UseInfo.getOperand(), getSubRegIndex(UseReg, BaseReg));
  }

  // Merge all other live ranges into the new merged range
  for (unsigned LRIdx : ToMerge) {
    if (LRIdx != MergedLRIdx) {
      // Add all operands from this range with correct SubRegIdx
      for (const auto &DefInfo : LiveRanges[LRIdx].defs()) {
        const MCRegister DefRegister = DefInfo.getOperand()->getReg().asMCReg();
        NewMergedLR.addDef(DefInfo.getOperand(),
                           getSubRegIndex(DefRegister, BaseReg));
      }
      for (const auto &UseInfo : LiveRanges[LRIdx].uses()) {
        const MCRegister UseReg = UseInfo.getOperand()->getReg().asMCReg();
        NewMergedLR.addUse(UseInfo.getOperand(),
                           getSubRegIndex(UseReg, BaseReg));
      }

      // Clear the merged range
      LiveRanges[LRIdx] = RegLiveRange();

      // Update all LiveRegs entries that pointed to the merged range
      for (auto &[LiveReg, LiveLRIdx] : LiveRegs) {
        if (LiveLRIdx == static_cast<int>(LRIdx)) {
          LiveLRIdx = static_cast<int>(MergedLRIdx);
        }
      }

      // Update OperandToLiveRange
      for (auto &Entry : OperandToLiveRange) {
        if (Entry.second == LRIdx) {
          Entry.second = MergedLRIdx;
        }
      }
    }
  }

  // Replace the merged live range with the new one
  LiveRanges[MergedLRIdx] = std::move(NewMergedLR);

  // Remove fully redefined registers from LiveRegs
  for (auto &[LiveReg, _] : AliasingLiveRegs) {
    if (DefReg == LiveReg || getSubRegIndex(LiveReg, DefReg) != 0) {
      LiveRegs.erase(LiveReg);
    }
  }

  // Also check if this def, combined with other defs in the merged range,
  // fully defines a super-register. If so, remove the super-register from
  // LiveRegs.
  if (MergedLRIdx < LiveRanges.size()) {
    RegLiveRange &MergedLR = LiveRanges[MergedLRIdx];
    const MCRegister MergedBaseReg = LiveRanges[MergedLRIdx].BaseReg;

    // Collect all defined sub-registers and compute their combined lane mask
    LaneBitmask DefinedLanes = LaneBitmask::getNone();
    for (const auto &DefInfo : MergedLR.defs()) {
      const MCRegister DefRegister = DefInfo.getOperand()->getReg().asMCReg();
      if (DefRegister == MergedBaseReg) {
        // Full register defined - covers all lanes
        DefinedLanes = LaneBitmask::getAll();
        break;
      }
      const unsigned SubIdx = getSubRegIndex(DefRegister, MergedBaseReg);
      if (SubIdx != 0) {
        // Add this sub-register's lanes to the defined lanes
        DefinedLanes |= TRI->getSubRegIndexLaneMask(SubIdx);
      }
    }

    // Check if the defined sub-registers fully cover any super-register
    // We need to recursively collect all sub-registers that are defined
    DenseSet<MCRegister> AllDefinedRegs;
    for (const auto &DefInfo : MergedLR.defs()) {
      const MCRegister DefRegister = DefInfo.getOperand()->getReg().asMCReg();
      AllDefinedRegs.insert(DefRegister);
      // Also add all sub-registers of this defined register
      for (MCSubRegIterator SubIt(DefRegister, TRI, /*IncludeSelf=*/false);
           SubIt.isValid(); ++SubIt) {
        AllDefinedRegs.insert(*SubIt);
      }
    }

    // Now check if any super-register of BaseReg is fully covered
    // Start with BaseReg itself and check all its super-registers
    SmallVector<MCRegister, 4> RegsToCheck;
    RegsToCheck.push_back(MergedBaseReg);
    for (MCSuperRegIterator SuperIt(MergedBaseReg, TRI); SuperIt.isValid();
         ++SuperIt) {
      RegsToCheck.push_back(*SuperIt);
    }

    // Check if all sub-registers of Reg are in AllDefinedRegs
    auto FullyCovered = [&](MCRegister Reg) {
      for (MCSubRegIterator SubIt(Reg, TRI, /*IncludeSelf=*/false);
           SubIt.isValid(); ++SubIt) {
        if (!AllDefinedRegs.count(*SubIt)) {
          return false;
        }
      }
      return true;
    };

    for (const MCRegister CheckReg : RegsToCheck) {
      // If this register is fully covered, remove it from LiveRegs
      if (FullyCovered(CheckReg)) {
        LiveRegs.erase(CheckReg);
        // Also remove any super-registers of CheckReg
        for (MCSuperRegIterator SuperIt(CheckReg, TRI); SuperIt.isValid();
             ++SuperIt) {
          LiveRegs.erase(*SuperIt);
        }
      }
    }
  }
}

DenseSet<MCRegister> RegLiveRangeTracker::collectReservedBaseRegs() const {
  DenseSet<MCRegister> ReservedRegs;
  for (const RegLiveRange &LR : LiveRanges) {
    if (LR.isReserved()) {
      ReservedRegs.insert(LR.BaseReg);
    }
  }
  return ReservedRegs;
}

void RegLiveRangeTracker::computeAvailableFromLiveRanges(
    const DenseSet<MCRegister> &ReservedRegs) {

  // Lambda to check if a register overlaps with any reserved register.
  auto OverlapsReserved = [&](MCRegister Reg) {
    return llvm::any_of(ReservedRegs, [&](MCRegister Reserved) {
      return TRI->regsOverlap(Reg, Reserved);
    });
  };

  // Build AvailablePhysRegs from non-reserved ranges, excluding any
  // register that overlaps with a reserved register.
  AvailablePhysRegs.clear();
  for (const RegLiveRange &LR : LiveRanges) {
    assert(LR.RegisterClass && "Live range must have a valid register class");
    assert(LR.BaseReg != MCRegister::NoRegister &&
           "Live range must have a base register");
    assert(LR.BaseReg.isPhysical() && "BaseReg must be a physical register");

    // Skip if this range is reserved.
    if (LR.isReserved()) {
      continue;
    }

    // Skip if base register overlaps with any reserved register.
    // Sub-registers are contained within the base, so if the base doesn't
    // overlap with reserved, neither will any sub-register.
    if (OverlapsReserved(LR.BaseReg)) {
      continue;
    }

    // Add base register and all its sub-registers.
    AvailablePhysRegs.insert(LR.BaseReg);
    for (MCSubRegIterator SubIt(LR.BaseReg, TRI, /*IncludeSelf=*/false);
         SubIt.isValid(); ++SubIt) {
      AvailablePhysRegs.insert(*SubIt);
    }
  }
}

void RegLiveRangeTracker::deriveSuperRegsFromSubRegs() {
  // If all sub-registers of a super-register are available, add the
  // super-register as well. This avoids repeated computation in PostRegAlloc.
  SmallVector<MCRegister, 32> RegsToCheck(AvailablePhysRegs.begin(),
                                          AvailablePhysRegs.end());
  for (MCRegister AvailReg : RegsToCheck) {
    for (MCSuperRegIterator SuperIt(AvailReg, TRI, /*IncludeSelf=*/false);
         SuperIt.isValid(); ++SuperIt) {
      const MCRegister SuperReg = *SuperIt;

      // Skip if already available.
      if (AvailablePhysRegs.count(SuperReg))
        continue;

      // Check if all sub-registers of SuperReg are available.
      bool AllSubregsAvailable = true;
      unsigned SubregCount = 0;
      for (MCSubRegIterator SubIt(SuperReg, TRI, /*IncludeSelf=*/false);
           SubIt.isValid(); ++SubIt) {
        ++SubregCount;
        if (!AvailablePhysRegs.count(*SubIt)) {
          AllSubregsAvailable = false;
          break;
        }
      }

      // If we have at least 2 sub-registers and all are available,
      // add this super-register.
      if (AllSubregsAvailable && SubregCount >= 2) {
        AvailablePhysRegs.insert(SuperReg);
      }
    }
  }
}

void RegLiveRangeTracker::addUnusedCallerSavedRegs(
    MachineBasicBlock &MBB, const DenseSet<MCRegister> &ImplicitRegs,
    const DenseSet<MCRegister> &ReservedRegs) {

  // This feature is controlled by a command-line option because it changes
  // the available register pool, which can affect register allocation results.
  if (!AddUnusedCallerSavedRegs)
    return;

  // Augment AvailablePhysRegs with caller-saved registers that are completely
  // unused in this block. Since pipelining excludes loops with calls, these
  // registers are safe to use as additional allocation candidates.
  //
  // A caller-saved register is safe to add if:
  // 1. It is allocatable (not reserved by the target)
  // 2. It belongs to a register class used by at least one live range
  // 3. It does not overlap with any register used in the block (explicit ops)
  // 4. It does not overlap with any register used implicitly
  // 5. It does not overlap with any live-in register (respecting lane masks)
  // 6. It does not overlap with any live-out register (respecting lane masks)
  // 7. It does not overlap with any reserved live range

  // Collect the set of register classes used by live ranges.
  SmallPtrSet<const TargetRegisterClass *, 8> UsedRegClasses;
  for (const RegLiveRange &LR : LiveRanges) {
    if (LR.RegisterClass) {
      UsedRegClasses.insert(LR.RegisterClass);
    }
  }

  // If no live ranges have register classes, nothing to add.
  if (UsedRegClasses.empty())
    return;

  const auto *AIERII = static_cast<const AIEBaseRegisterInfo *>(TRI);

  // Get the call-preserved mask. clobbersPhysReg returns true for caller-saved
  // registers (those NOT preserved across calls).
  const uint32_t *PreservedMask =
      AIERII->getCallPreservedMask(*MF, CallingConv::C);
  const BitVector AllocatableRegs = TRI->getAllocatableSet(*MF);

  // Generic lambda to check if a register overlaps with any register in a
  // range. Works with any range that yields MCRegister.
  auto OverlapsAny = [this](MCRegister Reg, auto &&Range) {
    return llvm::any_of(Range,
                        [&](MCRegister R) { return TRI->regsOverlap(Reg, R); });
  };

  // Generic lambda to check if a register overlaps with any RegisterMaskPair
  // in a range. Works with MBB.liveins() and MBB.liveouts().
  auto OverlapsAnyRMP = [this](MCRegister Reg, auto &&Range) {
    return llvm::any_of(Range,
                        [&](const MachineBasicBlock::RegisterMaskPair &RMP) {
                          return overlapsRMP(Reg, RMP, TRI);
                        });
  };

  // Helper to check if Reg is caller-saved (clobbered by calls).
  auto IsCallerSaved = [PreservedMask](MCRegister Reg) {
    return MachineOperand::clobbersPhysReg(PreservedMask, Reg);
  };

  // Transformer for AllPhysRegOperands to yield MCRegister.
  auto ToReg = [](const MachineOperand *MO) { return MO->getReg().asMCReg(); };

  // Iterate over allocatable registers and add unused caller-saved ones.
  unsigned NumUnusedCallerSavedAdded = 0;
  for (unsigned RegIdx = 0, E = TRI->getNumRegs(); RegIdx < E; ++RegIdx) {
    const MCRegister Reg = MCRegister::from(RegIdx);

    // Skip if already available.
    if (AvailablePhysRegs.count(Reg))
      continue;

    // Must be allocatable.
    if (!AllocatableRegs.test(RegIdx))
      continue;

    // Must be caller-saved (clobbered by calls).
    if (!IsCallerSaved(Reg))
      continue;

    // Must belong to at least one register class used by live ranges.
    bool BelongsToUsedClass = llvm::any_of(
        UsedRegClasses, [Reg](auto *RC) { return RC->contains(Reg); });
    if (!BelongsToUsedClass)
      continue;

    // Must not overlap with any explicitly used register in the block.
    if (OverlapsAny(Reg, llvm::map_range(AllPhysRegOperands, ToReg)))
      continue;

    // Must not overlap with any implicit register.
    if (OverlapsAny(Reg, ImplicitRegs))
      continue;

    // Must not overlap with any live-in register (respecting lane masks).
    if (OverlapsAnyRMP(Reg, MBB.liveins()))
      continue;

    // Must not overlap with any live-out register (respecting lane masks).
    if (OverlapsAnyRMP(Reg, MBB.liveouts()))
      continue;

    // Must not overlap with any reserved base register.
    if (OverlapsAny(Reg, ReservedRegs))
      continue;

    // This register is safe to use as an additional allocation candidate.
    AvailablePhysRegs.insert(Reg);
    ++NumUnusedCallerSavedAdded;

    LLVM_DEBUG(dbgs() << "Added unused caller-saved register: "
                      << TRI->getName(Reg) << "\n");
  }

  LLVM_DEBUG(dbgs() << "Added " << NumUnusedCallerSavedAdded
                    << " unused caller-saved registers to available set\n");
}

void RegLiveRangeTracker::markScarceRanges() {
  // Mark live ranges as scarce if they have exactly 1 available register.
  for (RegLiveRange &LR : LiveRanges) {
    const TargetRegisterClass *RC = LR.getRegisterClass();
    if (!RC) {
      continue;
    }

    unsigned AvailableCount = 0;
    for (MCPhysReg PhysReg : *RC) {
      if (AvailablePhysRegs.count(PhysReg)) {
        ++AvailableCount;
        if (AvailableCount > 1) {
          break;
        }
      }
    }

    LR.setIsScarce(AvailableCount == 1);
  }
}

void RegLiveRangeTracker::analyze(MachineBasicBlock &MBB,
                                  ArrayRef<MachineInstr *> SemanticOrder) {
  assert(!SemanticOrder.empty() && "SemanticOrder must be provided - MBB order "
                                   "is unreliable after scheduling");
  clear();

  // Build instruction order map from semantic order
  // Also track implicit registers to invalidate overlapping explicit ranges
  DenseSet<MCRegister> ImplicitRegs;
  unsigned InstrIdx = 0;
  for (MachineInstr *MI : SemanticOrder) {
    InstrOrder[MI] = InstrIdx++;

    for (MachineOperand &MO : MI->operands()) {
      if (!MO.isReg() || !MO.getReg().isPhysical()) {
        continue;
      }
      if (MO.isImplicit()) {
        // Track implicit registers - we won't create live ranges for these
        // but will use them to invalidate explicit ranges
        const MCRegister Reg = MO.getReg().asMCReg();

        // Add all aliases
        for (MCRegAliasIterator AI(Reg, TRI, /*IncludeSelf=*/true);
             AI.isValid(); ++AI) {
          MCRegister Alias = *AI;
          ImplicitRegs.insert(Alias);
        }
      } else {
        AllPhysRegOperands.push_back(&MO);
      }
    }
  }

  // Track live registers (backward pass).
  // Map from register to its current live range index (signed).
  // Use NoLiveRange as sentinel for live-out registers not yet associated with
  // a range.
  DenseMap<MCRegister, int> LiveRegs;

  // Initialize with live-out registers using NoLiveRange as sentinel.
  for (const auto &RMP : MBB.liveouts()) {
    LiveRegs[RMP.PhysReg] = RegLiveRange::NoLiveRange;
  }

  // Map from operand to live range index
  DenseMap<MachineOperand *, unsigned> OperandToLiveRange;

  // Lambda to create or find a live range for a register.
  auto GetOrCreateLiveRange = [&](MCRegister Reg,
                                  MachineOperand *MO) -> unsigned {
    bool IsReserved = false;

    // Check if this register or an aliasing register is already live.
    auto It = llvm::find_if(LiveRegs, [Reg, TRI = TRI](const auto &Entry) {
      return TRI->regsOverlap(Reg, Entry.first);
    });

    if (It != LiveRegs.end()) {
      const int LRIdx = It->second;

      if (LRIdx == RegLiveRange::NoLiveRange) {
        // Found a live-out register (NoLiveRange sentinel).
        // Mark the new range as reserved.
        IsReserved = true;
      } else {
        // Found an aliasing live register with an actual live range.
        assert(LRIdx >= 0 && "LRIdx must be valid");
        OperandToLiveRange[MO] = LRIdx;

        // Update base register for this live range if needed.
        MCRegister CurrentBase = LiveRanges[LRIdx].BaseReg;
        if (CurrentBase == MCRegister::NoRegister) {
          // No base yet, use current register.
          LiveRanges[LRIdx].BaseReg = Reg;
        } else {
          // Check if we need to update to a larger base register.
          assert(CurrentBase.isPhysical() && "CurrentBase must be physical");
          assert(Reg.isPhysical() && "Reg must be physical");
          if (getSubRegIndex(Reg, CurrentBase) == 0 &&
              getSubRegIndex(CurrentBase, Reg) != 0) {
            // Reg is larger than current base.
            LiveRanges[LRIdx].BaseReg = Reg;
          }
        }

        return LRIdx;
      }
    }

    // Create a new live range.
    const unsigned NewLRIdx = LiveRanges.size();
    LiveRanges.emplace_back();
    LiveRanges[NewLRIdx].ID = NextLiveRangeID++;
    LiveRanges[NewLRIdx].BaseReg = Reg;
    LiveRanges[NewLRIdx].setIsReserved(IsReserved);
    LiveRegs[Reg] = static_cast<int>(NewLRIdx);
    OperandToLiveRange[MO] = NewLRIdx;
    return NewLRIdx;
  };

  // Process instructions in reverse semantic order (backward pass)
  for (MachineInstr *MI : llvm::reverse(SemanticOrder)) {

    // In backward pass: process uses first (they start liveness), then defs
    // (they kill liveness)

    // First process uses - they start liveness.
    for (MachineOperand &MO : MI->uses()) {
      if (!MO.isReg() || !MO.getReg().isPhysical() || MO.isImplicit())
        continue;

      const MCRegister Reg = MO.getReg().asMCReg();
      const unsigned LRIdx = GetOrCreateLiveRange(Reg, &MO);

      // Add use to the live range with SubRegIdx relative to base.
      const MCRegister CurrentBase = LiveRanges[LRIdx].BaseReg;
      const unsigned SubRegIdx = getSubRegIndex(Reg, CurrentBase);
      LiveRanges[LRIdx].addUse(&MO, SubRegIdx);
    }

    // Then process defs - they kill liveness.
    for (MachineOperand &MO : MI->defs()) {
      if (!MO.isReg() || !MO.getReg().isPhysical() || MO.isImplicit())
        continue;

      const MCRegister Reg = MO.getReg().asMCReg();
      const unsigned DefLRIdx = GetOrCreateLiveRange(Reg, &MO);

      // Add def to the live range with SubRegIdx relative to base.
      const MCRegister CurrentBase = LiveRanges[DefLRIdx].BaseReg;
      const unsigned SubRegIdx = getSubRegIndex(Reg, CurrentBase);
      LiveRanges[DefLRIdx].addDef(&MO, SubRegIdx);

      // Merge with any aliasing live ranges.
      mergeAliasingLiveRanges(DefLRIdx, Reg, LiveRegs, OperandToLiveRange);
    }
  }

  // First-stage safety filtering
  LLVM_DEBUG({ dump("CANDIDATE LIVE RANGES\n"); });
  LLVM_DEBUG(dbgs() << "\nFirst-stage filtering: " << LiveRanges.size()
                    << " candidate ranges\n");
  SmallVector<RegLiveRange, 16> SafeRanges;
  for (const RegLiveRange &LR : LiveRanges) {

    // Skip invalid/cleared ranges from merging
    if (LR.getID() < 0)
      continue;

    // Filter out live ranges whose base register is not fully defined.
    // This uses the same check as during the backward scan to determine
    // if a new live range should be created.
    if (!isFullyDefined(LR, LiveRegs)) {
      LLVM_DEBUG({
        dbgs() << "Reject: base register not fully defined in block: ";
        LR.dumpBrief(TRI);
      });
      continue;
    }

    // Must have at least 1 def (use-only ranges indicate live-in)
    if (LR.getNumDefs() == 0) {
      LLVM_DEBUG({
        dbgs() << "Reject: no defs: ";
        LR.dumpBrief(TRI);
      });
      continue;
    }

    // Filter out any live range that uses an implicit register
    auto UsesImplicitReg = [&ImplicitRegs](const RegOperandInfo &OperInfo) {
      const MCRegister Reg = OperInfo.getOperand()->getReg().asMCReg();
      return ImplicitRegs.count(Reg) > 0;
    };

    if (llvm::any_of(LR.operands(), UsesImplicitReg)) {
      LLVM_DEBUG({
        dbgs() << "Reject: uses implicit register ";
        for (const auto &OI : LR.operands()) {
          MCRegister R = OI.getOperand()->getReg().asMCReg();
          if (ImplicitRegs.count(R)) {
            dbgs() << TRI->getName(R) << " ";
            break;
          }
        }
        dbgs() << ": ";
        LR.dumpBrief(TRI);
      });
      continue;
    }

    // Must start with a def in the block (not use-before-def)
    if (!startsWithDefInBlock(LR)) {
      LLVM_DEBUG({
        dbgs() << "Reject: doesn't start with def (use-before-def): ";
        LR.dumpBrief(TRI);
      });
      continue;
    }

    // Reject tied operands
    if (hasTiedOperands(LR)) {
      LLVM_DEBUG({
        dbgs() << "Reject: has tied operands: ";
        LR.dumpBrief(TRI);
      });
      continue;
    }

    // Note: We don't check killedBeforeEndOfBlock because:
    // 1. Live-out is already filtered by isCarriedByLiveInOut check
    // 2. We want to allow def-only ranges (garbage bin registers)

    LLVM_DEBUG({
      dbgs() << "Keep: ";
      LR.dumpBrief(TRI);
    });
    SafeRanges.push_back(LR);
  }

  LLVM_DEBUG(dbgs() << "After first-stage: " << SafeRanges.size()
                    << " safe ranges\n");

  LiveRanges = std::move(SafeRanges);

  // Compute register classes and apply filtering.
  LLVM_DEBUG(dbgs() << "\nRegister class computation and filtering\n");
  SmallVector<RegLiveRange, 16> ValidRanges;
  for (RegLiveRange &LR : LiveRanges) {
    computeRegisterClass(LR);

    // Filter out ranges with no valid register class.
    if (!LR.RegisterClass) {
      LLVM_DEBUG({
        dbgs() << "Reject: no valid register class: ";
        LR.dumpBrief(TRI);
      });
      continue;
    }

    // Apply register class filtering if specified.
    if (!ExcludeLiveRangesByRegClass.empty() &&
        StringRef(TRI->getRegClassName(LR.RegisterClass)) ==
            ExcludeLiveRangesByRegClass) {
      LLVM_DEBUG({
        dbgs() << "Reject: excluded register class "
               << TRI->getRegClassName(LR.RegisterClass) << ": ";
        LR.dumpBrief(TRI);
      });
      continue;
    }

    ValidRanges.push_back(std::move(LR));
  }
  LiveRanges = std::move(ValidRanges);

  LLVM_DEBUG(dbgs() << "After register class filtering: " << LiveRanges.size()
                    << " ranges\n");

  // Second-stage full coverage pruning.
  // This happens AFTER register class filtering.
  pruneByFullCoverage();

  // Compute and cache available physical registers.
  const DenseSet<MCRegister> ReservedRegs = collectReservedBaseRegs();
  computeAvailableFromLiveRanges(ReservedRegs);
  deriveSuperRegsFromSubRegs();

  addUnusedCallerSavedRegs(MBB, ImplicitRegs, ReservedRegs);
  markScarceRanges();

  // Compute and cache the most promising scarce range set.
  MostPromisingScarceRanges = findMostPromisingScarceRanges(AvailablePhysRegs);
}

void RegLiveRangeTracker::computeRegisterClass(RegLiveRange &LR) const {
  if (LR.BaseReg == MCRegister::NoRegister)
    return;

  // Start with nullptr, representing the universe of all register classes.
  // Intersection with nullptr is identity: intersect(nullptr, X) = X
  const TargetRegisterClass *CommonRC = nullptr;

  // Process all operands (defs and uses) to compute register class constraints
  for (const auto &OpInfo : LR.operands()) {
    MachineInstr *MI = OpInfo.getOperand()->getParent();
    const unsigned OpIdx = OpInfo.getOperand()->getOperandNo();

    // Get the register class constraint for this operand
    const TargetRegisterClass *OpRC =
        MI->getRegClassConstraint(OpIdx, TII, TRI);

    if (OpRC) {
      // Account for subregister access
      if (OpInfo.getSubRegIdx() != 0) {
        // Get the class that can be used with this subreg index
        OpRC = TRI->getSubClassWithSubReg(OpRC, OpInfo.getSubRegIdx());
      }

      if (OpRC) {
        // Intersect: nullptr is identity, otherwise find common subclass
        if (!CommonRC) {
          CommonRC = OpRC;
        } else {
          CommonRC = TRI->getCommonSubClass(CommonRC, OpRC);
          if (!CommonRC) {
            // No common class possible - this live range is illegal
            LR.RegisterClass = nullptr;
            return;
          }
        }
      }
    }
  }

  // If no operand constraints were found, fall back to minimal class
  if (!CommonRC) {
    CommonRC = TRI->getMinimalPhysRegClass(LR.BaseReg);
    assert(CommonRC && "Physical register must have a register class");
  }

  LR.RegisterClass = CommonRC;

  // Populate AdmissibleRegs from RegisterClass.
  // This is initially equivalent to the RC membership, but can be further
  // constrained later by per-LR requirements (e.g., bypass constraints).
  LR.AdmissibleRegs.clear();
  if (CommonRC) {
    for (MCPhysReg Reg : *CommonRC) {
      LR.AdmissibleRegs.insert(Reg);
    }
  }
}

void RegLiveRangeTracker::virtualizeFilteredPhysRegs(OverlapPolicy Policy) {
  assert(!RegistersVirtualized && "Registers are already virtualized");

  MachineRegisterInfo &MRI = MF->getRegInfo();

  // Clear the NoVRegs property.
  MF->getProperties().reset(MachineFunctionProperties::Property::NoVRegs);

  // Build the set of RESERVED base registers.
  DenseSet<MCRegister> ReservedBases;
  for (const RegLiveRange &LR : LiveRanges) {
    if (LR.isReserved()) {
      ReservedBases.insert(LR.BaseReg);
    }
  }

  // Create and rewrite virtual registers. Live ranges are created in reverse,
  // so we run this loop in reverse order to make the dumps more intuitive.
  for (RegLiveRange &LR : reverse(LiveRanges)) {
    // The analysis should have filtered out any live ranges without a valid
    // register class.
    assert(LR.RegisterClass && "Live range must have a valid register class");

    // The analysis should have assigned a base register to every live range.
    assert(LR.BaseReg != MCRegister::NoRegister &&
           "Live range must have a base register");

    // Never virtualize RESERVED ranges themselves.
    if (LR.isReserved()) {
      continue;
    }

    // Apply the overlap policy.
    if (Policy == OverlapPolicy::DisallowOverlapWithReservedBase) {
      // Check if this LR's base register overlaps any RESERVED base.
      bool OverlapsReserved = false;
      for (MCRegister ReservedBase : ReservedBases) {
        if (TRI->regsOverlap(LR.BaseReg, ReservedBase)) {
          OverlapsReserved = true;
          break;
        }
      }
      if (OverlapsReserved) {
        // Skip virtualization for this range.
        continue;
      }
    }
    // If Policy == AllowOverlapWithReservedBase, we proceed to virtualize.

    // Create a virtual register for this live range.
    const Register VReg = MRI.createVirtualRegister(LR.RegisterClass);

    // Store the VReg in the LiveRange for later mapping.
    LR.setVReg(VReg);

    // Replace all operands in this live range with the virtual register.
    const auto RewriteOperand = [VReg](const RegOperandInfo &Info) {
      MachineOperand *MO = Info.getOperand();
      MO->setReg(VReg);
      MO->setSubReg(Info.getSubRegIdx());
    };

    // Rewrite all operands.
    for (const auto &OpInfo : LR.operands()) {
      RewriteOperand(OpInfo);
    }
  }

  // Mark as virtualized even if no live ranges were virtualized.
  RegistersVirtualized = true;
}

void RegLiveRangeTracker::rewriteToPhysRegs(
    const DenseMap<Register, MCRegister> &VRegToPhysMap) {
  assert(RegistersVirtualized && "Registers are not virtualized");

  MachineRegisterInfo &MRI = MF->getRegInfo();

  for (const RegLiveRange &LR : LiveRanges) {
    const Register VReg = LR.getVReg();

    // Skip live ranges that were not virtualized (partial virtualization).
    if (!VReg.isValid()) {
      continue;
    }

    // Look up the physical register for this virtual register.
    auto It = VRegToPhysMap.find(VReg);
    assert(It != VRegToPhysMap.end() &&
           "VReg must have a mapping in VRegToPhysMap");

    const MCRegister PhysReg = It->second;

    // Rewrite all operands in this live range to the physical register.
    for (const auto &OpInfo : LR.operands()) {
      MachineOperand *MO = OpInfo.getOperand();
      if (MO->getReg() == VReg) {
        // Compute the actual physical register considering subregs.
        Register FinalReg = PhysReg;
        if (OpInfo.getSubRegIdx() != 0) {
          FinalReg = TRI->getSubReg(PhysReg, OpInfo.getSubRegIdx());
          assert(FinalReg && "Invalid subregister index for physical register");
        }
        MO->setReg(FinalReg);
        MO->setSubReg(0);
      }
    }
  }

  // Clear virtual registers from MRI and restore NoVRegs property.
  MRI.clearVirtRegs();
  MF->getProperties().set(MachineFunctionProperties::Property::NoVRegs);

  // Mark as no longer virtualized.
  RegistersVirtualized = false;

  LLVM_DEBUG(dbgs() << "Rewritten virtual registers to physical registers\n");
}

void RegLiveRangeTracker::restoreOriginalPhysRegs() {
  // Build the mapping from VRegs to their original PhysRegs
  DenseMap<Register, MCRegister> VRegToPhysMap;
  for (const RegLiveRange &LR : LiveRanges) {
    if (LR.getVReg().isValid()) {
      VRegToPhysMap[LR.getVReg()] = LR.getBaseReg();
    }
  }

  // Use the general rewrite method
  rewriteToPhysRegs(VRegToPhysMap);
  LLVM_DEBUG(dbgs() << "Restored original physical registers\n");
}

bool RegLiveRangeTracker::areRegistersVirtualized() const {
  return RegistersVirtualized;
}

void RegLiveRangeTracker::filterByRegisterAvailability() {
  // Lambda to check if a live range has only one choice of physical register.
  auto HasNoChoice = [&](const RegLiveRange &LR) -> bool {
    // By this point, all live ranges should have a register class.
    assert(LR.RegisterClass && "Live range must have a register class");

    // Count how many physical registers from this register class are available.
    unsigned AvailableCount = 0;
    for (MCPhysReg PhysReg : *LR.RegisterClass) {
      if (AvailablePhysRegs.count(PhysReg)) {
        AvailableCount++;
        // If we find at least 2, this live range has choices.
        if (AvailableCount > 1) {
          return false;
        }
      }
    }

    // Has no choice if 0 or 1 available registers.
    return true;
  };

  // Build a new list of live ranges, excluding those with no choice.
  SmallVector<RegLiveRange, 16> FilteredLiveRanges;

  for (const RegLiveRange &LR : LiveRanges) {
    // Skip live ranges that have no choice of physical register.
    if (HasNoChoice(LR)) {
      LLVM_DEBUG(dbgs() << "Filtering out live range for "
                        << TRI->getName(LR.BaseReg)
                        << " - no alternative physical registers\n");
      continue;
    }

    // This live range has choices, keep it.
    FilteredLiveRanges.push_back(LR);
  }

  // Replace the live ranges with the filtered set.
  LiveRanges = std::move(FilteredLiveRanges);

  LLVM_DEBUG(dbgs() << "Register availability filtering complete: "
                    << LiveRanges.size() << " live ranges remaining\n");
}

void RegLiveRangeTracker::clear() {
  // Clear all containers.
  LiveRanges.clear();
  AllPhysRegOperands.clear();
  InstrOrder.clear();

  // Reset the virtualization flag.
  RegistersVirtualized = false;

  // Reset the ID counter.
  NextLiveRangeID = 0;

  // Note: MF, TRI, and TII are not cleared as they are set in the constructor
  // and represent the context in which this tracker operates.
}

void RegLiveRangeTracker::dump(const char *Header) const {
  if (Header) {
    dbgs() << Header;
  }
  dbgs() << "================================\n";
  dbgs() << "Total live ranges: " << LiveRanges.size() << "\n\n";

  // Create a sorted index array to ensure deterministic output
  SmallVector<size_t, 16> SortedIndices;
  for (size_t LRIdx = 0; LRIdx < LiveRanges.size(); ++LRIdx) {
    SortedIndices.push_back(LRIdx);
  }

  // Sort by base register ID first, then by first def instruction pointer
  // This ensures a stable, deterministic order
  llvm::sort(SortedIndices, [this](size_t A, size_t B) {
    const RegLiveRange &LRA = LiveRanges[A];
    const RegLiveRange &LRB = LiveRanges[B];

    // First sort by base register ID
    if (LRA.getBaseReg() != LRB.getBaseReg()) {
      return LRA.getBaseReg() < LRB.getBaseReg();
    }

    // Then by first def instruction address (if any)
    if (!LRA.defs().empty() && !LRB.defs().empty()) {
      const MachineInstr *MIA = LRA.defs().begin()->getOperand()->getParent();
      const MachineInstr *MIB = LRB.defs().begin()->getOperand()->getParent();
      if (MIA != MIB) {
        // Use instruction order if available
        auto ItA = InstrOrder.find(MIA);
        auto ItB = InstrOrder.find(MIB);
        if (ItA != InstrOrder.end() && ItB != InstrOrder.end()) {
          return ItA->second < ItB->second;
        }
      }
    }

    // Finally by original index for stability
    return A < B;
  });

  for (size_t SortedIdx = 0; SortedIdx < SortedIndices.size(); ++SortedIdx) {
    const size_t LRIdx = SortedIndices[SortedIdx];
    const RegLiveRange &LR = LiveRanges[LRIdx];

    // Skip invalid/cleared ranges
    if (LR.getID() < 0)
      continue;

    // Use the stored base register
    const MCRegister BaseReg = LR.getBaseReg();
    StringRef PrimaryReg = "unknown";
    if (BaseReg != MCRegister::NoRegister) {
      PrimaryReg = TRI->getName(BaseReg);
    }

    dbgs() << "Live Range #" << LR.getID() << " for " << PrimaryReg;
    if (LR.isReserved()) {
      dbgs() << " [RESERVED]";
    }
    dbgs() << ":\n";

    dbgs() << "  Definitions (" << LR.getNumDefs() << "):\n";
    size_t DefIdx = 0;
    for (const RegOperandInfo &DefInfo : LR.defs()) {
      dbgs() << "    [" << DefIdx++ << "] ";
      Register Reg = DefInfo.getOperand()->getReg();
      if (Reg.isPhysical()) {
        dbgs() << "Register: " << TRI->getName(Reg);
      } else {
        dbgs() << "Register: %vreg" << Reg.virtRegIndex();
      }
      if (DefInfo.getSubRegIdx() != 0) {
        dbgs() << " (SubRegIdx: " << DefInfo.getSubRegIdx() << ")";
      }
      dbgs() << " ";
      if (MachineInstr *DefInstr = DefInfo.getOperand()->getParent()) {
        dbgs() << AIE::NoDebug(*DefInstr) << "\n";
      } else {
        dbgs() << "<orphaned operand>\n";
      }
    }

    dbgs() << "  Uses (" << LR.getNumUses() << "):\n";
    size_t UseIdx = 0;
    for (const RegOperandInfo &UseInfo : LR.uses()) {
      dbgs() << "    [" << UseIdx++ << "] ";
      Register Reg = UseInfo.getOperand()->getReg();
      if (Reg.isPhysical()) {
        dbgs() << "Register: " << TRI->getName(Reg);
      } else {
        dbgs() << "Register: %vreg" << Reg.virtRegIndex();
      }
      if (UseInfo.getSubRegIdx() != 0) {
        dbgs() << " (SubRegIdx: " << UseInfo.getSubRegIdx() << ")";
      }
      dbgs() << " ";
      if (MachineInstr *UseInstr = UseInfo.getOperand()->getParent()) {
        dbgs() << AIE::NoDebug(*UseInstr) << "\n";
      } else {
        dbgs() << "<orphaned operand>\n";
      }
    }
    dbgs() << "\n";
  }

  // Dump available physical registers if live ranges exist.
  if (!LiveRanges.empty()) {
    DenseSet<MCRegister> AvailablePhysRegs = getAvailablePhysRegs();
    dbgs() << "Available Physical Registers for Reallocation:\n";
    dbgs() << "==============================================\n";
    SmallVector<MCRegister, 32> SortedRegs(AvailablePhysRegs.begin(),
                                           AvailablePhysRegs.end());
    llvm::sort(SortedRegs);
    for (MCRegister Reg : SortedRegs) {
      // MCRegister should always be physical, but check to be safe.
      if (Reg.isPhysical()) {
        dbgs() << "  " << TRI->getName(Reg) << "\n";
      }
    }
    dbgs() << "Total: " << AvailablePhysRegs.size() << " registers\n\n";
  }

  // Emit end marker if header was provided
  if (Header) {
    dbgs() << "=== END " << Header;
  }
}

std::vector<const RegLiveRange *>
RegLiveRangeTracker::findMostPromisingScarceRanges(
    const DenseSet<MCRegister> &AvailablePhysRegs) const {

  // Group live ranges by base register (not register class).
  // This ensures we only get ranges for the same physical register.
  DenseMap<MCRegister, std::vector<const RegLiveRange *>> RangesByBaseReg;

  for (const auto &LR : LiveRanges) {
    // Only consider ranges that are marked as scarce.
    if (!LR.isScarce()) {
      continue;
    }

    const MCRegister BaseReg = LR.getBaseReg();
    assert(BaseReg != MCRegister::NoRegister &&
           "LiveRange must have a BaseReg after analysis");

    RangesByBaseReg[BaseReg].push_back(&LR);
  }

  // Helper to check if a set of ranges has overlapping instructions.
  auto HasOverlap = [](const std::vector<const RegLiveRange *> &Ranges) {
    DenseSet<const MachineInstr *> SeenInstrs;
    for (const RegLiveRange *LR : Ranges) {
      for (const auto &Info : LR->operands()) {
        if (!SeenInstrs.insert(Info.getOperand()->getParent()).second) {
          return true;
        }
      }
    }
    return false;
  };

  // Find the largest non-overlapping set with actual competition.
  std::vector<const RegLiveRange *> LargestSet;
  for (const auto &Entry : RangesByBaseReg) {
    const auto &Ranges = Entry.second;

    if (Ranges.size() > 1 && !HasOverlap(Ranges) &&
        Ranges.size() > LargestSet.size()) {
      LargestSet = Ranges;
    }
  }

  return LargestSet;
}
