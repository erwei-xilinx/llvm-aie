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
#include "llvm/MC/LaneBitmask.h"
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

void RegLiveRange::mergeFrom(const RegLiveRange &Other,
                             const TargetRegisterInfo *TRI) {
  // Helper to compute sub-register index.
  auto GetSubRegIdx = [TRI](MCRegister AccessReg,
                            MCRegister NewBaseReg) -> unsigned {
    if (AccessReg == NewBaseReg)
      return 0;
    for (MCSubRegIndexIterator SubRegIdxIt(NewBaseReg, TRI);
         SubRegIdxIt.isValid(); ++SubRegIdxIt) {
      if (SubRegIdxIt.getSubReg() == AccessReg) {
        return SubRegIdxIt.getSubRegIndex();
      }
    }
    return 0;
  };

  // Helper to check if Reg1 is a sub-register of Reg2 (Reg2 is larger).
  auto IsSubReg = [TRI](MCRegister Reg1, MCRegister Reg2) -> bool {
    for (MCSubRegIndexIterator SubRegIdxIt(Reg2, TRI); SubRegIdxIt.isValid();
         ++SubRegIdxIt) {
      if (SubRegIdxIt.getSubReg() == Reg1) {
        return true;
      }
    }
    return false;
  };

  // Helper to check if a candidate register contains all operand registers.
  // A register R "contains" an operand register OR if OR == R or OR is a
  // sub-register of R.
  auto ContainsAllOperands =
      [&IsSubReg](MCRegister Candidate,
                  ArrayRef<MCRegister> OperandRegs) -> bool {
    for (MCRegister OpReg : OperandRegs) {
      if (OpReg != Candidate && !IsSubReg(OpReg, Candidate)) {
        return false;
      }
    }
    return true;
  };

  // Collect all operand registers from both ranges.
  SmallVector<MCRegister, 8> AllOperandRegs;
  for (const auto &DefInfo : Defs) {
    AllOperandRegs.push_back(DefInfo.getOperand()->getReg().asMCReg());
  }
  for (const auto &UseInfo : Uses) {
    AllOperandRegs.push_back(UseInfo.getOperand()->getReg().asMCReg());
  }
  for (const auto &DefInfo : Other.Defs) {
    AllOperandRegs.push_back(DefInfo.getOperand()->getReg().asMCReg());
  }
  for (const auto &UseInfo : Other.Uses) {
    AllOperandRegs.push_back(UseInfo.getOperand()->getReg().asMCReg());
  }

  // Compute the new base register: the smallest register that contains all
  // operand registers. Start with the current base registers as candidates.
  MCRegister NewBaseReg = BaseReg;
  if (NewBaseReg == MCRegister::NoRegister) {
    NewBaseReg = Other.BaseReg;
  } else if (Other.BaseReg != MCRegister::NoRegister) {
    // Check if we need to update to a larger base register.
    if (IsSubReg(NewBaseReg, Other.BaseReg)) {
      NewBaseReg = Other.BaseReg;
    }
  }

  // If the current NewBaseReg doesn't contain all operands (e.g., sibling
  // registers like cml4 and cmh4), find the smallest common super-register.
  if (NewBaseReg != MCRegister::NoRegister &&
      !ContainsAllOperands(NewBaseReg, AllOperandRegs)) {
    // Search for the smallest super-register that contains all operands.
    // We iterate through super-registers of NewBaseReg in ascending order
    // (MCSuperRegIterator yields them from smallest to largest).
    for (MCSuperRegIterator SuperIt(NewBaseReg, TRI); SuperIt.isValid();
         ++SuperIt) {
      if (ContainsAllOperands(*SuperIt, AllOperandRegs)) {
        NewBaseReg = *SuperIt;
        break;
      }
    }
  }

  // Re-add existing operands with updated sub-register indices if base
  // changed.
  if (NewBaseReg != BaseReg) {
    SmallVector<RegOperandInfo, 4> OldDefs = std::move(Defs);
    SmallVector<RegOperandInfo, 4> OldUses = std::move(Uses);
    Defs.clear();
    Uses.clear();

    for (const auto &DefInfo : OldDefs) {
      const MCRegister DefReg = DefInfo.getOperand()->getReg().asMCReg();
      Defs.emplace_back(DefInfo.getOperand(), GetSubRegIdx(DefReg, NewBaseReg));
    }
    for (const auto &UseInfo : OldUses) {
      const MCRegister UseReg = UseInfo.getOperand()->getReg().asMCReg();
      Uses.emplace_back(UseInfo.getOperand(), GetSubRegIdx(UseReg, NewBaseReg));
    }

    BaseReg = NewBaseReg;
  }

  // Merge defs from Other with computed sub-register indices.
  for (const auto &DefInfo : Other.defs()) {
    const MCRegister DefReg = DefInfo.getOperand()->getReg().asMCReg();
    Defs.emplace_back(DefInfo.getOperand(), GetSubRegIdx(DefReg, NewBaseReg));
  }

  // Merge uses from Other with computed sub-register indices.
  for (const auto &UseInfo : Other.uses()) {
    const MCRegister UseReg = UseInfo.getOperand()->getReg().asMCReg();
    Uses.emplace_back(UseInfo.getOperand(), GetSubRegIdx(UseReg, NewBaseReg));
  }

  // Propagate reserved status: if Other is reserved, this becomes reserved.
  if (Other.IsReserved) {
    IsReserved = true;
  }
}

void RegLiveRange::expandBaseToInclude(MCRegister ExtReg,
                                       const TargetRegisterInfo *TRI) {
  if (ExtReg == MCRegister::NoRegister)
    return;

  // Helper to compute sub-register index.
  auto GetSubRegIdx = [TRI](MCRegister AccessReg,
                            MCRegister NewBaseReg) -> unsigned {
    if (AccessReg == NewBaseReg)
      return 0;
    for (MCSubRegIndexIterator SubRegIdxIt(NewBaseReg, TRI);
         SubRegIdxIt.isValid(); ++SubRegIdxIt) {
      if (SubRegIdxIt.getSubReg() == AccessReg) {
        return SubRegIdxIt.getSubRegIndex();
      }
    }
    return 0;
  };

  // Helper to check if Reg1 is a sub-register of Reg2 (Reg2 is larger).
  auto IsSubReg = [TRI](MCRegister Reg1, MCRegister Reg2) -> bool {
    for (MCSubRegIndexIterator SubRegIdxIt(Reg2, TRI); SubRegIdxIt.isValid();
         ++SubRegIdxIt) {
      if (SubRegIdxIt.getSubReg() == Reg1) {
        return true;
      }
    }
    return false;
  };

  // If BaseReg is not set, just use ExtReg.
  if (BaseReg == MCRegister::NoRegister) {
    BaseReg = ExtReg;
    return;
  }

  // If ExtReg is already contained by BaseReg, nothing to do.
  if (ExtReg == BaseReg || IsSubReg(ExtReg, BaseReg))
    return;

  // If BaseReg is contained by ExtReg, upgrade to ExtReg.
  if (IsSubReg(BaseReg, ExtReg)) {
    // Recompute SubRegIdx for existing operands.
    SmallVector<RegOperandInfo, 4> OldDefs = std::move(Defs);
    SmallVector<RegOperandInfo, 4> OldUses = std::move(Uses);
    Defs.clear();
    Uses.clear();

    for (const auto &DefInfo : OldDefs) {
      const MCRegister DefReg = DefInfo.getOperand()->getReg().asMCReg();
      Defs.emplace_back(DefInfo.getOperand(), GetSubRegIdx(DefReg, ExtReg));
    }
    for (const auto &UseInfo : OldUses) {
      const MCRegister UseReg = UseInfo.getOperand()->getReg().asMCReg();
      Uses.emplace_back(UseInfo.getOperand(), GetSubRegIdx(UseReg, ExtReg));
    }

    BaseReg = ExtReg;
    return;
  }

  // Neither is a subreg of the other - find the smallest common super-register.
  // Collect all operand registers plus ExtReg.
  SmallVector<MCRegister, 8> AllRegs;
  AllRegs.push_back(ExtReg);
  for (const auto &DefInfo : Defs) {
    AllRegs.push_back(DefInfo.getOperand()->getReg().asMCReg());
  }
  for (const auto &UseInfo : Uses) {
    AllRegs.push_back(UseInfo.getOperand()->getReg().asMCReg());
  }

  // Helper to check if a candidate register contains all registers.
  auto ContainsAll = [&IsSubReg](MCRegister Candidate,
                                 ArrayRef<MCRegister> Regs) -> bool {
    for (MCRegister R : Regs) {
      if (R != Candidate && !IsSubReg(R, Candidate)) {
        return false;
      }
    }
    return true;
  };

  // Search for the smallest super-register that contains all.
  MCRegister NewBaseReg = BaseReg;
  for (MCSuperRegIterator SuperIt(BaseReg, TRI); SuperIt.isValid(); ++SuperIt) {
    if (ContainsAll(*SuperIt, AllRegs)) {
      NewBaseReg = *SuperIt;
      break;
    }
  }

  // Recompute SubRegIdx for existing operands.
  if (NewBaseReg != BaseReg) {
    SmallVector<RegOperandInfo, 4> OldDefs = std::move(Defs);
    SmallVector<RegOperandInfo, 4> OldUses = std::move(Uses);
    Defs.clear();
    Uses.clear();

    for (const auto &DefInfo : OldDefs) {
      const MCRegister DefReg = DefInfo.getOperand()->getReg().asMCReg();
      Defs.emplace_back(DefInfo.getOperand(), GetSubRegIdx(DefReg, NewBaseReg));
    }
    for (const auto &UseInfo : OldUses) {
      const MCRegister UseReg = UseInfo.getOperand()->getReg().asMCReg();
      Uses.emplace_back(UseInfo.getOperand(), GetSubRegIdx(UseReg, NewBaseReg));
    }

    BaseReg = NewBaseReg;
  }
}

void RegLiveRange::clear() {
  Defs.clear();
  Uses.clear();
  BaseReg = MCRegister::NoRegister;
  RegisterClass = nullptr;
  AdmissibleRegs.clear();
  VReg = Register();
  IsScarce = false;
  IsReserved = false;
  ID = -1;
}

/// Get the sub-register index if AccessReg is a sub-register of BaseReg.
/// Returns 0 if AccessReg is not a sub-register of BaseReg.
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

bool RegLiveRangeTracker::isFullyDefined(
    const RegLiveRange &LR,
    const DenseMap<MCRegister, LaneBitmask> &LocalLiveLaneMasks,
    const MachineBasicBlock &MBB) const {
  // A live range is fully defined if its algorithm-local live lanemasks
  // do not intersect with the live-in set of the block.
  //
  // This is more precise than just checking register overlap: it allows
  // ranges where the live lanes are disjoint from the live-in lanes.
  //
  // Importantly, this can discriminate between a truly undefined register
  // (which is not in the live-in set and is safe to virtualize) and a
  // register that was defined outside of the loop (which is in the live-in
  // set and should be rejected because changing it would affect loop-carried
  // values).

  // Check each register in LocalLiveLaneMasks that overlaps with the base
  // register.
  for (const auto &[LiveReg, LocalLanes] : LocalLiveLaneMasks) {
    if (!TRI->regsOverlap(LR.getBaseReg(), LiveReg))
      continue;

    // Found an overlapping register with non-zero live lanes.
    // Check if these lanes intersect with the live-in set.
    for (const auto &LiveIn : MBB.liveins()) {
      if (!TRI->regsOverlap(LiveReg, LiveIn.PhysReg))
        continue;

      // Check if the algorithm-local live lanes intersect with the live-in
      // lanes.
      if ((LocalLanes & LiveIn.LaneMask).any()) {
        return false;
      }
    }
  }

  return true;
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
    unsigned DefLRIdx, MCRegister DefReg,
    DenseMap<MCRegister, std::pair<int, LaneBitmask>> &LiveRegs,
    DenseMap<MachineOperand *, unsigned> &OperandToLiveRange) {

  // Helper to check if a def register's lanes overlap with a live register's
  // current lanes. This is critical for separating live ranges: after x10 is
  // defined, any y5 (containing x10) should only have x11's lanes live, and a
  // subsequent x10 def should NOT merge into that y5 range.
  auto LanesOverlap = [this](MCRegister DefR, MCRegister LiveR,
                             LaneBitmask LiveLanes) -> bool {
    // If registers are equal, check if any lanes are live.
    if (DefR == LiveR)
      return LiveLanes.any();

    // Check if DefR is a subreg of LiveR.
    for (MCSubRegIndexIterator SubIdxIt(LiveR, TRI); SubIdxIt.isValid();
         ++SubIdxIt) {
      if (SubIdxIt.getSubReg() == DefR) {
        // DefR is a subreg of LiveR - check if DefR's lanes are live.
        const LaneBitmask DefLanes =
            TRI->getSubRegIndexLaneMask(SubIdxIt.getSubRegIndex());
        return (LiveLanes & DefLanes).any();
      }
    }

    // Check if LiveR is a subreg of DefR.
    for (MCSubRegIndexIterator SubIdxIt(DefR, TRI); SubIdxIt.isValid();
         ++SubIdxIt) {
      if (SubIdxIt.getSubReg() == LiveR) {
        // LiveR is a subreg of DefR - if any lanes of LiveR are live,
        // they overlap with DefR.
        return LiveLanes.any();
      }
    }

    // Registers overlap but no subreg relationship - conservatively treat
    // as overlapping if any lanes are live.
    return LiveLanes.any();
  };

  // Collect all aliasing live registers and their live ranges.
  // Only include registers where the lanes actually overlap.
  SmallVector<std::pair<MCRegister, int>, 8> AliasingLiveRegs;
  for (const auto &[LiveReg, Info] : LiveRegs) {
    if (TRI->regsOverlap(DefReg, LiveReg) &&
        LanesOverlap(DefReg, LiveReg, Info.second)) {
      AliasingLiveRegs.push_back({LiveReg, Info.first});
    }
  }

  if (AliasingLiveRegs.empty())
    return;

  // Collect all unique live range indices to merge (excluding NoLiveRange
  // sentinels which represent live-out registers without actual ranges).
  SmallVector<unsigned, 4> ToMerge;
  for (const auto &[LiveReg, LRIdx] : AliasingLiveRegs) {
    if (LRIdx != RegLiveRange::NoLiveRange) {
      // Check if we already have this index.
      if (llvm::find(ToMerge, static_cast<unsigned>(LRIdx)) == ToMerge.end() &&
          static_cast<unsigned>(LRIdx) != DefLRIdx) {
        ToMerge.push_back(static_cast<unsigned>(LRIdx));
      }
    }
  }

  // Compute reserved status before merging.
  // Check if any aliasing live register is a live-out sentinel.
  bool IsReservedFromLiveOut = false;
  for (const auto &[LiveReg, LRIdx] : AliasingLiveRegs) {
    if (LRIdx == RegLiveRange::NoLiveRange) {
      IsReservedFromLiveOut = true;
      break;
    }
  }

  // Also check if any subreg of DefReg is live-out.
  if (!IsReservedFromLiveOut) {
    for (MCSubRegIterator SubIt(DefReg, TRI, /*IncludeSelf=*/true);
         SubIt.isValid(); ++SubIt) {
      auto It = LiveRegs.find(*SubIt);
      if (It != LiveRegs.end() &&
          It->second.first == RegLiveRange::NoLiveRange) {
        IsReservedFromLiveOut = true;
        break;
      }
    }
  }

  // Get the target live range and update its reserved status.
  RegLiveRange &TargetLR = LiveRanges[DefLRIdx];
  if (IsReservedFromLiveOut) {
    TargetLR.setIsReserved(true);
  }

  // Expand TargetLR's base to include any external registers from
  // AliasingLiveRegs that don't have actual live ranges (live-out sentinels).
  // These registers affect the base register size but have no operands.
  for (const auto &[LiveReg, LRIdx] : AliasingLiveRegs) {
    if (LRIdx == RegLiveRange::NoLiveRange) {
      TargetLR.expandBaseToInclude(LiveReg, TRI);
    }
  }

  // Incrementally merge all other live ranges into the target.
  // The enhanced mergeFrom() automatically computes the smallest common
  // super-register that contains all operands from both ranges.
  for (unsigned LRIdx : ToMerge) {
    TargetLR.mergeFrom(LiveRanges[LRIdx], TRI);

    // Clear the source range (mark as invalid).
    LiveRanges[LRIdx].clear();

    // Update all LiveRegs entries that pointed to the merged range.
    for (auto &[LiveReg, Info] : LiveRegs) {
      if (Info.first == static_cast<int>(LRIdx)) {
        Info.first = static_cast<int>(DefLRIdx);
      }
    }

    // Update OperandToLiveRange.
    for (auto &Entry : OperandToLiveRange) {
      if (Entry.second == LRIdx) {
        Entry.second = DefLRIdx;
      }
    }
  }

  // Remove fully redefined registers from LiveRegs.
  for (const auto &[LiveReg, LRIdx] : AliasingLiveRegs) {
    if (DefReg == LiveReg || getSubRegIndex(LiveReg, DefReg) != 0) {
      LiveRegs.erase(LiveReg);
    }
  }

  // Update lane masks for partially redefined super-registers.
  // When DefReg is a subreg of LiveReg, the def kills DefReg's lanes within
  // LiveReg. This is critical for separating live ranges: after x10 is defined,
  // any y5 (containing x10) should only have x11's lanes live, not x10's.
  for (const auto &[LiveReg, OrigLRIdx] : AliasingLiveRegs) {
    // Skip if already erased (fully redefined).
    auto LiveIt = LiveRegs.find(LiveReg);
    if (LiveIt == LiveRegs.end())
      continue;

    // Check if DefReg is a subreg of LiveReg (DefReg partially kills LiveReg).
    const unsigned SubRegIdx = getSubRegIndex(DefReg, LiveReg);
    if (SubRegIdx != 0) {
      // DefReg is a subreg of LiveReg - update LiveReg's lane mask.
      const LaneBitmask DefLanes = TRI->getSubRegIndexLaneMask(SubRegIdx);
      LiveIt->second.second &= ~DefLanes;

      // If no lanes remain live, remove the entry entirely.
      if (LiveIt->second.second.none()) {
        LiveRegs.erase(LiveIt);
      }
    }
  }

  // Check if this def, combined with other defs in the merged range,
  // fully defines a super-register. If so, remove the super-register from
  // LiveRegs.
  const MCRegister MergedBaseReg = TargetLR.getBaseReg();

  // Collect all defined sub-registers.
  DenseSet<MCRegister> AllDefinedRegs;
  for (const auto &DefInfo : TargetLR.defs()) {
    const MCRegister DefRegister = DefInfo.getOperand()->getReg().asMCReg();
    AllDefinedRegs.insert(DefRegister);
    // Also add all sub-registers of this defined register.
    for (MCSubRegIterator SubIt(DefRegister, TRI, /*IncludeSelf=*/false);
         SubIt.isValid(); ++SubIt) {
      AllDefinedRegs.insert(*SubIt);
    }
  }

  // Check if all sub-registers of a register are defined.
  auto FullyCovered = [&](MCRegister Reg) {
    for (MCSubRegIterator SubIt(Reg, TRI, /*IncludeSelf=*/false);
         SubIt.isValid(); ++SubIt) {
      if (!AllDefinedRegs.count(*SubIt)) {
        return false;
      }
    }
    return true;
  };

  // Check BaseReg and its super-registers.
  SmallVector<MCRegister, 4> RegsToCheck;
  RegsToCheck.push_back(MergedBaseReg);
  for (MCSuperRegIterator SuperIt(MergedBaseReg, TRI); SuperIt.isValid();
       ++SuperIt) {
    RegsToCheck.push_back(*SuperIt);
  }

  for (const MCRegister CheckReg : RegsToCheck) {
    if (FullyCovered(CheckReg)) {
      LiveRegs.erase(CheckReg);
      for (MCSuperRegIterator SuperIt(CheckReg, TRI); SuperIt.isValid();
           ++SuperIt) {
        LiveRegs.erase(*SuperIt);
      }
    }
  }
}

DenseSet<MCRegister> RegLiveRangeTracker::collectReservedBaseRegs() const {
  DenseSet<MCRegister> ReservedRegs;
  for (const RegLiveRange &LR : LiveRanges) {
    if (LR.isReserved()) {
      ReservedRegs.insert(LR.getBaseReg());
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
    assert(LR.getRegisterClass() &&
           "Live range must have a valid register class");
    assert(LR.getBaseReg() != MCRegister::NoRegister &&
           "Live range must have a base register");
    assert(LR.getBaseReg().isPhysical() &&
           "BaseReg must be a physical register");

    // Skip if this range is reserved.
    if (LR.isReserved()) {
      continue;
    }

    // Skip if base register overlaps with any reserved register.
    // Sub-registers are contained within the base, so if the base doesn't
    // overlap with reserved, neither will any sub-register.
    if (OverlapsReserved(LR.getBaseReg())) {
      continue;
    }

    // Add base register and all its sub-registers.
    AvailablePhysRegs.insert(LR.getBaseReg());
    for (MCSubRegIterator SubIt(LR.getBaseReg(), TRI, /*IncludeSelf=*/false);
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
    if (LR.getRegisterClass()) {
      UsedRegClasses.insert(LR.getRegisterClass());
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

//===----------------------------------------------------------------------===//
// Analyze helper methods (decomposition of analyze())
//===----------------------------------------------------------------------===//

void RegLiveRangeTracker::buildInstructionOrderAndCollectOperands(
    ArrayRef<MachineInstr *> SemanticOrder, LivenessScanState &State) {
  unsigned InstrIdx = 0;
  for (MachineInstr *MI : SemanticOrder) {
    InstrOrder[MI] = InstrIdx++;

    for (MachineOperand &MO : MI->operands()) {
      if (!MO.isReg() || !MO.getReg().isPhysical()) {
        continue;
      }
      if (MO.isImplicit()) {
        // Track implicit registers - we won't create live ranges for these
        // but will use them to invalidate explicit ranges.
        const MCRegister Reg = MO.getReg().asMCReg();

        // Add all aliases.
        for (MCRegAliasIterator AI(Reg, TRI, /*IncludeSelf=*/true);
             AI.isValid(); ++AI) {
          State.ImplicitRegs.insert(*AI);
        }
      } else {
        AllPhysRegOperands.push_back(&MO);
      }
    }
  }
}

void RegLiveRangeTracker::initLiveRegsFromLiveOuts(const MachineBasicBlock &MBB,
                                                   LivenessScanState &State) {
  // Initialize with live-out registers using NoLiveRange as sentinel and their
  // lane masks.
  for (const auto &RMP : MBB.liveouts()) {
    State.LiveRegs[RMP.PhysReg] = {RegLiveRange::NoLiveRange, RMP.LaneMask};
  }
}

unsigned RegLiveRangeTracker::getOrCreateLiveRangeForOperand(
    MCRegister Reg, MachineOperand *MO, LivenessScanState &State) {
  bool IsReserved = false;

  // Check if this register or an aliasing register is already live.
  // We need to find an entry where the lanes actually overlap, not just
  // the registers.  This is critical for separating live ranges: after
  // x10 is defined, any y5 (containing x10) should only have x11's lanes
  // live, and a subsequent x10 access should NOT merge into that y5 range.
  auto It = llvm::find_if(State.LiveRegs, [Reg, TRI = TRI](const auto &Entry) {
    if (!TRI->regsOverlap(Reg, Entry.first))
      return false;

    // Registers overlap - now check if lanes overlap.
    const MCRegister LiveReg = Entry.first;
    const LaneBitmask LiveLanes = Entry.second.second;

    // If LiveReg equals Reg, check if any lanes are live.
    if (LiveReg == Reg)
      return LiveLanes.any();

    // Check if Reg is a subreg of LiveReg.
    for (MCSubRegIndexIterator SubIdxIt(LiveReg, TRI); SubIdxIt.isValid();
         ++SubIdxIt) {
      if (SubIdxIt.getSubReg() == Reg) {
        // Reg is a subreg of LiveReg - check if Reg's lanes are live.
        const LaneBitmask RegLanes =
            TRI->getSubRegIndexLaneMask(SubIdxIt.getSubRegIndex());
        return (LiveLanes & RegLanes).any();
      }
    }

    // Check if LiveReg is a subreg of Reg.
    for (MCSubRegIndexIterator SubIdxIt(Reg, TRI); SubIdxIt.isValid();
         ++SubIdxIt) {
      if (SubIdxIt.getSubReg() == LiveReg) {
        // LiveReg is a subreg of Reg - if any lanes of LiveReg are live,
        // they overlap with Reg.
        return LiveLanes.any();
      }
    }

    // Registers overlap but no subreg relationship - conservatively treat
    // as overlapping if any lanes are live.
    return LiveLanes.any();
  });

  if (It != State.LiveRegs.end()) {
    const int LRIdx = It->second.first;

    if (LRIdx == RegLiveRange::NoLiveRange) {
      // Found a live-out register (NoLiveRange sentinel).
      // Mark the new range as reserved.
      IsReserved = true;
    } else {
      // Found an aliasing live register with an actual live range.
      assert(LRIdx >= 0 && "LRIdx must be valid");
      State.OperandToLiveRange[MO] = LRIdx;

      // Update base register for this live range if needed.
      MCRegister CurrentBase = LiveRanges[LRIdx].getBaseReg();
      if (CurrentBase == MCRegister::NoRegister) {
        // No base yet - expand base to include this register.
        LiveRanges[LRIdx].expandBaseToInclude(Reg, TRI);
      } else {
        // Check if we need to update to a larger base register.
        assert(CurrentBase.isPhysical() && "CurrentBase must be physical");
        assert(Reg.isPhysical() && "Reg must be physical");
        if (getSubRegIndex(Reg, CurrentBase) == 0 &&
            getSubRegIndex(CurrentBase, Reg) != 0) {
          // Reg is larger than current base - update BaseReg and recompute
          // SubRegIdx for all existing operands.
          LiveRanges[LRIdx].expandBaseToInclude(Reg, TRI);
        }
      }

      return LRIdx;
    }
  }

  // Create a new live range.
  const unsigned NewLRIdx = LiveRanges.size();
  LiveRanges.emplace_back(NextLiveRangeID++, Reg, IsReserved);
  State.LiveRegs[Reg] = {static_cast<int>(NewLRIdx), LaneBitmask::getAll()};
  State.OperandToLiveRange[MO] = NewLRIdx;
  return NewLRIdx;
}

void RegLiveRangeTracker::processDefsInInstruction(MachineInstr &MI,
                                                   LivenessScanState &State) {
  for (MachineOperand &MO : MI.defs()) {
    if (!MO.isReg() || !MO.getReg().isPhysical() || MO.isImplicit())
      continue;

    const MCRegister Reg = MO.getReg().asMCReg();
    const unsigned DefLRIdx = getOrCreateLiveRangeForOperand(Reg, &MO, State);

    // Add def to the live range with SubRegIdx relative to base.
    const MCRegister CurrentBase = LiveRanges[DefLRIdx].getBaseReg();
    const unsigned SubRegIdx = getSubRegIndex(Reg, CurrentBase);
    LiveRanges[DefLRIdx].addDef(&MO, SubRegIdx);

    // Merge with any aliasing live ranges.
    mergeAliasingLiveRanges(DefLRIdx, Reg, State.LiveRegs,
                            State.OperandToLiveRange);
  }
}

void RegLiveRangeTracker::processUsesInInstruction(MachineInstr &MI,
                                                   LivenessScanState &State) {
  for (MachineOperand &MO : MI.uses()) {
    if (!MO.isReg() || !MO.getReg().isPhysical() || MO.isImplicit())
      continue;

    const MCRegister Reg = MO.getReg().asMCReg();
    const unsigned LRIdx = getOrCreateLiveRangeForOperand(Reg, &MO, State);

    // Add use to the live range with SubRegIdx relative to base.
    const MCRegister CurrentBase = LiveRanges[LRIdx].getBaseReg();
    const unsigned SubRegIdx = getSubRegIndex(Reg, CurrentBase);
    LiveRanges[LRIdx].addUse(&MO, SubRegIdx);
  }
}

void RegLiveRangeTracker::performLivenessScan(
    ArrayRef<MachineInstr *> SemanticOrder, LivenessScanState &State) {
  // Process instructions in reverse semantic order (backward pass).
  for (MachineInstr *MI : llvm::reverse(SemanticOrder)) {
    // In backward pass: process defs first (they kill liveness), then uses
    // (they start liveness). This order is critical for read-modify-write
    // instructions where the same register is both read and written.
    // The def terminates the current live range, and the use starts a new one.
    processDefsInInstruction(*MI, State);
    processUsesInInstruction(*MI, State);
  }
}

void RegLiveRangeTracker::applySafetyFiltering(
    const MachineBasicBlock &MBB, const LivenessScanState &State,
    const DenseMap<MCRegister, LaneBitmask> &LocalLiveLaneMasks) {
  LLVM_DEBUG({ dump("CANDIDATE LIVE RANGES\n"); });
  LLVM_DEBUG(dbgs() << "\nFirst-stage filtering: " << LiveRanges.size()
                    << " candidate ranges\n");

  SmallVector<RegLiveRange, 16> SafeRanges;
  for (const RegLiveRange &LR : LiveRanges) {
    // Skip invalid/cleared ranges from merging.
    if (LR.getID() < 0)
      continue;

    // Filter out live ranges whose base register is not fully defined.
    // This checks that the range doesn't read from live-in values, which
    // would make it unsafe to virtualize (we'd be changing loop-carried
    // values). This also implicitly handles use-before-def cases.
    if (!isFullyDefined(LR, LocalLiveLaneMasks, MBB)) {
      LLVM_DEBUG({
        dbgs() << "Reject: base register not fully defined in block: ";
        LR.dumpBrief(TRI);
      });
      continue;
    }

    // Filter out any live range that uses an implicit register.
    auto UsesImplicitReg = [&State](const RegOperandInfo &OperInfo) {
      const MCRegister Reg = OperInfo.getOperand()->getReg().asMCReg();
      return State.ImplicitRegs.count(Reg) > 0;
    };

    if (llvm::any_of(LR.operands(), UsesImplicitReg)) {
      LLVM_DEBUG({
        dbgs() << "Reject: uses implicit register ";
        for (const auto &OI : LR.operands()) {
          MCRegister R = OI.getOperand()->getReg().asMCReg();
          if (State.ImplicitRegs.count(R)) {
            dbgs() << TRI->getName(R) << " ";
            break;
          }
        }
        dbgs() << ": ";
        LR.dumpBrief(TRI);
      });
      continue;
    }

    // Reject tied operands.
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
}

void RegLiveRangeTracker::computeRegisterClassesAndFilter() {
  LLVM_DEBUG(dbgs() << "\nRegister class computation and filtering\n");

  SmallVector<RegLiveRange, 16> ValidRanges;
  for (RegLiveRange &LR : LiveRanges) {
    computeRegisterClass(LR);

    // Filter out ranges with no valid register class.
    if (!LR.getRegisterClass()) {
      LLVM_DEBUG({
        dbgs() << "Reject: no valid register class: ";
        LR.dumpBrief(TRI);
      });
      continue;
    }

    // Apply register class filtering if specified.
    if (!ExcludeLiveRangesByRegClass.empty() &&
        StringRef(TRI->getRegClassName(LR.getRegisterClass())) ==
            ExcludeLiveRangesByRegClass) {
      LLVM_DEBUG({
        dbgs() << "Reject: excluded register class "
               << TRI->getRegClassName(LR.getRegisterClass()) << ": ";
        LR.dumpBrief(TRI);
      });
      continue;
    }

    ValidRanges.push_back(std::move(LR));
  }
  LiveRanges = std::move(ValidRanges);

  LLVM_DEBUG(dbgs() << "After register class filtering: " << LiveRanges.size()
                    << " ranges\n");
}

void RegLiveRangeTracker::finalizeAvailabilityAndScarcity(
    MachineBasicBlock &MBB, const LivenessScanState &State) {
  // Second-stage full coverage pruning.
  // This happens AFTER register class filtering.
  pruneByFullCoverage();

  // Compute and cache available physical registers.
  const DenseSet<MCRegister> ReservedRegs = collectReservedBaseRegs();
  computeAvailableFromLiveRanges(ReservedRegs);
  deriveSuperRegsFromSubRegs();

  addUnusedCallerSavedRegs(MBB, State.ImplicitRegs, ReservedRegs);
  markScarceRanges();

  // Compute and cache the most promising scarce range set.
  MostPromisingScarceRanges = findMostPromisingScarceRanges(AvailablePhysRegs);
}

void RegLiveRangeTracker::analyze(MachineBasicBlock &MBB,
                                  ArrayRef<MachineInstr *> SemanticOrder) {
  assert(!SemanticOrder.empty() && "SemanticOrder must be provided - MBB order "
                                   "is unreliable after scheduling");
  clear();

  // Initialize state for liveness scan.
  LivenessScanState State;

  // Build instruction order map and collect operands.
  buildInstructionOrderAndCollectOperands(SemanticOrder, State);

  // Initialize live registers from live-outs.
  initLiveRegsFromLiveOuts(MBB, State);

  // Perform the liveness scan to build live ranges.
  performLivenessScan(SemanticOrder, State);

  // Extract lane masks from LiveRegs for the isFullyDefined check.
  DenseMap<MCRegister, LaneBitmask> LocalLiveLaneMasks;
  for (const auto &[Reg, Info] : State.LiveRegs) {
    LocalLiveLaneMasks[Reg] = Info.second;
  }

  // Apply first-stage safety filtering.
  applySafetyFiltering(MBB, State, LocalLiveLaneMasks);

  // Compute register classes and apply filtering.
  computeRegisterClassesAndFilter();

  // Finalize availability and scarcity.
  finalizeAvailabilityAndScarcity(MBB, State);
}

void RegLiveRange::setRegisterClass(const TargetRegisterClass *RC) {
  RegisterClass = RC;

  // Populate AdmissibleRegs from RegisterClass.
  // This is initially equivalent to the RC membership, but can be further
  // constrained later by per-LR requirements (e.g., bypass constraints).
  AdmissibleRegs.clear();
  if (RC) {
    for (MCPhysReg Reg : *RC) {
      AdmissibleRegs.insert(Reg);
    }
  }
}

void RegLiveRangeTracker::computeRegisterClass(RegLiveRange &LR) const {
  if (LR.getBaseReg() == MCRegister::NoRegister)
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
            // No common class possible - this live range is illegal.
            LR.setRegisterClass(nullptr);
            return;
          }
        }
      }
    }
  }

  // If no operand constraints were found, fall back to minimal class.
  if (!CommonRC) {
    CommonRC = TRI->getMinimalPhysRegClass(LR.getBaseReg());
    assert(CommonRC && "Physical register must have a register class");
  }

  LR.setRegisterClass(CommonRC);
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
      ReservedBases.insert(LR.getBaseReg());
    }
  }

  // Create and rewrite virtual registers. Live ranges are created in reverse,
  // so we run this loop in reverse order to make the dumps more intuitive.
  for (RegLiveRange &LR : reverse(LiveRanges)) {
    // The analysis should have filtered out any live ranges without a valid
    // register class.
    assert(LR.getRegisterClass() &&
           "Live range must have a valid register class");

    // The analysis should have assigned a base register to every live range.
    assert(LR.getBaseReg() != MCRegister::NoRegister &&
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
        if (TRI->regsOverlap(LR.getBaseReg(), ReservedBase)) {
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
    const Register VReg = MRI.createVirtualRegister(LR.getRegisterClass());

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
    assert(LR.getRegisterClass() && "Live range must have a register class");

    // Count how many physical registers from this register class are available.
    unsigned AvailableCount = 0;
    for (MCPhysReg PhysReg : *LR.getRegisterClass()) {
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
                        << TRI->getName(LR.getBaseReg())
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
