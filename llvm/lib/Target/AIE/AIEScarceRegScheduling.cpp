//===- AIEScarceRegScheduling.cpp - Scarce Register Scheduling Strategy --===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// This file implements a PostPipelinerStrategy that prioritizes scheduling
// decisions based on scarce register pressure.
//===----------------------------------------------------------------------===//

#include "AIEScarceRegScheduling.h"
#include "AIERegDefUseTracker.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/TargetSchedule.h"

#define DEBUG_TYPE "scarce-reg-sched"

namespace llvm::AIE {

ScarceRange::ScarceRange(const RegLiveRange &LR, const ScheduleDAGInstrs &DAG)
    : LiveRange(LR) {
  // Collect all unique MachineInstr pointers from defs and uses.
  DenseSet<const MachineInstr *> UniqueInstrs;

  for (const auto &DefInfo : LR.defs()) {
    MachineOperand *const DefOp = DefInfo.getOperand();
    assert(DefOp && "DefOp should be valid");
    MachineInstr *const DefMI = DefOp->getParent();
    assert(DefMI && "Every operand should have a parent MachineInstr");
    UniqueInstrs.insert(DefMI);
  }

  for (const auto &UseInfo : LR.uses()) {
    MachineOperand *const UseOp = UseInfo.getOperand();
    assert(UseOp && "UseOp should be valid");
    MachineInstr *const UseMI = UseOp->getParent();
    assert(UseMI && "Every operand should have a parent MachineInstr");
    UniqueInstrs.insert(UseMI);
  }

  // Iterate over all SUnits and collect those whose instruction is in the set.
  // This handles the case where multiple SUnits reference the same instruction.
  // We only need the first (representative) SUnit for each instruction.
  for (const auto &SU : DAG.SUnits) {
    const MachineInstr *const MI = SU.getInstr();
    assert(MI && "Every SUnit should have a MachineInstr");
    if (UniqueInstrs.count(MI)) {
      Members.push_back(SU.NodeNum);
      // Early break when we've found all unique instructions.
      if (Members.size() == UniqueInstrs.size()) {
        break;
      }
    }
  }

  // Members are in SUnit order, which is deterministic.
}

ScarceRegScheduling::ScarceRegScheduling(ScheduleDAGInstrs &DAG,
                                         ScheduleInfo &Info,
                                         RegLiveRangeTracker &RegTracker,
                                         int II)
    : PostPipelinerStrategy(DAG, Info, /*LatestBias=*/0),
      RegTracker(RegTracker), II(II) {}

BurstMostUrgentStrategy::BurstMostUrgentStrategy(
    ScheduleDAGInstrs &DAG, ScheduleInfo &Info,
    const std::vector<ScarceRange> &ScarceRanges, int LatestBias)
    : PostPipelinerStrategy(DAG, Info, LatestBias), ScarceRanges(ScarceRanges),
      CurrentSet(0) {

  assert(!ScarceRanges.empty() &&
         "BurstMostUrgentStrategy requires at least one scarce range");

  // Build a set to track which SUnits are part of scarce ranges.
  const size_t NumSUnits = Info.NInstr;
  SmallVector<bool, 64> IsScarceRangeMember(NumSUnits, false);
  for (const auto &Range : ScarceRanges) {
    for (int MemberIdx : Range.Members) {
      assert(MemberIdx >= 0 && static_cast<size_t>(MemberIdx) < NumSUnits &&
             "Scarce range member index out of bounds");
      IsScarceRangeMember[MemberIdx] = true;
    }
  }

  // Precompute predecessors and members for each range (in original order).
  Predecessors.reserve(ScarceRanges.size());
  Members.reserve(ScarceRanges.size());

  for (const auto &Range : ScarceRanges) {
    // Collect non-scarce predecessors for this range.
    SmallVector<int, 4> RangePredecessors;
    for (int MemberIdx : Range.Members) {
      const auto &MemberNode = Info[MemberIdx];
      for (int AncestorIdx : MemberNode.Ancestors) {
        // Only include non-scarce ancestors.
        if (static_cast<size_t>(AncestorIdx) < IsScarceRangeMember.size() &&
            !IsScarceRangeMember[AncestorIdx]) {
          // Avoid duplicates.
          if (std::find(RangePredecessors.begin(), RangePredecessors.end(),
                        AncestorIdx) == RangePredecessors.end()) {
            RangePredecessors.push_back(AncestorIdx);
          }
        }
      }
    }

    Predecessors.push_back(std::move(RangePredecessors));
    Members.push_back(Range.Members);
  }

  // Pre-size OrderedMembers (will be populated by init()).
  OrderedMembers.resize(ScarceRanges.size() * 2);
}

void BurstMostUrgentStrategy::init(const SmallVector<int, 4> &RangeOrder) {
  assert(RangeOrder.size() == ScarceRanges.size() &&
         "RangeOrder must have the same size as ScarceRanges");

  // Reset state.
  CurrentSet = 0;

  // Build OrderedMembers by interleaving predecessors and members in the given
  // order.
  for (size_t I = 0; I < RangeOrder.size(); ++I) {
    const int RangeIdx = RangeOrder[I];
    OrderedMembers[2 * I] = Predecessors[RangeIdx];
    OrderedMembers[2 * I + 1] = Members[RangeIdx];
  }
}

bool BurstMostUrgentStrategy::better(const SUnit &A, const SUnit &B) {
  const int AIdx = A.NodeNum;
  const int BIdx = B.NodeNum;

  // Check if either is in the current set.
  if (CurrentSet < OrderedMembers.size()) {
    const auto &CurrentMembers = OrderedMembers[CurrentSet];
    const bool AInSet = std::find(CurrentMembers.begin(), CurrentMembers.end(),
                                  AIdx) != CurrentMembers.end();
    const bool BInSet = std::find(CurrentMembers.begin(), CurrentMembers.end(),
                                  BIdx) != CurrentMembers.end();

    // Prefer members of the current set.
    if (AInSet != BInSet) {
      return AInSet;
    }
  }

  // Default: prefer earlier earliest.
  return Info[AIdx].Earliest < Info[BIdx].Earliest;
}

void BurstMostUrgentStrategy::selected(const SUnit &N) {
  // Check if we've completed the current set.
  if (CurrentSet < OrderedMembers.size()) {
    const auto &CurrentMembers = OrderedMembers[CurrentSet];

    // Check if all members of the current set are scheduled.
    const bool AllMembersScheduled =
        llvm::all_of(CurrentMembers, [this](int MemberIdx) {
          return Info[MemberIdx].Scheduled;
        });

    // If all members are scheduled, advance to the next set.
    if (AllMembersScheduled) {
      ++CurrentSet;
      LLVM_DEBUG(dbgs() << format("Completed set %zu, advancing to %zu\n",
                                  CurrentSet - 1, CurrentSet));

      // If we just completed a members set (odd index), simulate
      // anti-dependences.
      if ((CurrentSet - 1) % 2 == 1) {
        const size_t BurstIdx = (CurrentSet - 1) / 2;
        const int RangeIdx =
            (BurstIdx < ScarceRanges.size()) ? static_cast<int>(BurstIdx) : -1;
        if (RangeIdx >= 0) {
          simulateAntiDependences(RangeIdx);
        }
      }
    }
  }
}

void BurstMostUrgentStrategy::simulateAntiDependences(int CompletedRangeIdx) {
  const auto &CompletedRange = ScarceRanges[CompletedRangeIdx];
  const auto *const SchedModel = DAG.getSchedModel();

  LLVM_DEBUG(dbgs() << format("Simulating anti-dependences for range %d\n",
                              CompletedRangeIdx));

  // For each Use in the completed range's LiveRange.
  for (const auto &UseInfo : CompletedRange.LiveRange.uses()) {
    MachineOperand *const UseOp = UseInfo.getOperand();
    assert(UseOp && "UseOp should be valid");
    MachineInstr *const UseMI = UseOp->getParent();
    assert(UseMI && "Every operand should have a parent MachineInstr");

    const unsigned UseOpIdx = UseOp->getOperandNo();

    // Find the corresponding SUnit index.
    int UseSUIdx = -1;
    for (const int MemberIdx : CompletedRange.Members) {
      if (DAG.SUnits[MemberIdx].getInstr() == UseMI) {
        UseSUIdx = MemberIdx;
        break;
      }
    }
    assert(UseSUIdx >= 0 && "Use instruction should be in completed range");

    const int UseCycle = Info[UseSUIdx].Cycle;

    // For each subsequent range.
    for (size_t LaterRangeIdx = CompletedRangeIdx + 1;
         LaterRangeIdx < ScarceRanges.size(); ++LaterRangeIdx) {
      const auto &LaterRange = ScarceRanges[LaterRangeIdx];

      // For each Def in the later range's LiveRange.
      for (const auto &DefInfo : LaterRange.LiveRange.defs()) {
        MachineOperand *const DefOp = DefInfo.getOperand();
        assert(DefOp && "DefOp should be valid");
        MachineInstr *const DefMI = DefOp->getParent();
        assert(DefMI && "Every operand should have a parent MachineInstr");

        const unsigned DefOpIdx = DefOp->getOperandNo();

        // Find the corresponding SUnit index.
        int DefSUIdx = -1;
        for (const int MemberIdx : LaterRange.Members) {
          if (DAG.SUnits[MemberIdx].getInstr() == DefMI) {
            DefSUIdx = MemberIdx;
            break;
          }
        }
        assert(DefSUIdx >= 0 && "Def instruction should be in later range");

        // Compute the anti-dependence latency.
        const unsigned Latency =
            SchedModel->computeOperandLatency(UseMI, UseOpIdx, DefMI, DefOpIdx);

        // Update Earliest[Def] = max(Earliest[Def], Cycle[Use] + L).
        const int NewEarliest = UseCycle + static_cast<int>(Latency);
        Info[DefSUIdx].Earliest =
            std::max(Info[DefSUIdx].Earliest, NewEarliest);
      }
    }
  }
}

void buildScarceRangeMapping(const std::vector<ScarceRange> &Ranges,
                             const ScheduleInfo &Info,
                             std::vector<int> &RangeOfSUnit) {
  RangeOfSUnit.assign(Info.NInstr, -1);

  for (size_t RangeIdx = 0; RangeIdx < Ranges.size(); ++RangeIdx) {
    const auto &Range = Ranges[RangeIdx];
    for (int MemberIdx : Range.Members) {
      assert(MemberIdx >= 0 && MemberIdx < Info.NInstr &&
             "Scarce range member index out of bounds");
      assert(RangeOfSUnit[MemberIdx] == -1 &&
             "SUnit cannot belong to multiple scarce ranges");
      RangeOfSUnit[MemberIdx] = RangeIdx;
    }
  }
}

void buildScarceDAG(std::vector<ScarceRange> &Ranges, const ScheduleInfo &Info,
                    const ScheduleDAGInstrs &DAG) {
  // Build the mapping from SUnit to range index.
  std::vector<int> RangeOfSUnit;
  buildScarceRangeMapping(Ranges, Info, RangeOfSUnit);

  // Populate PredRanges for each range using direct predecessors from the DAG.
  for (size_t RangeIdx = 0; RangeIdx < Ranges.size(); ++RangeIdx) {
    auto &Range = Ranges[RangeIdx];
    Range.PredRanges.clear();

    // Use a small set to deduplicate predecessor ranges.
    SmallVector<int, 4> PredSet;

    // For each member of this range.
    for (int MemberIdx : Range.Members) {
      assert(MemberIdx >= 0 && MemberIdx < Info.NInstr &&
             "Scarce range member index out of bounds");

      const auto &SU = DAG.SUnits[MemberIdx];

      // For each direct predecessor of this member.
      for (const auto &PredEdge : SU.Preds) {
        const SUnit *PredSU = PredEdge.getSUnit();
        if (!PredSU || PredSU->isBoundaryNode()) {
          continue;
        }

        const int PredIdx = PredSU->NodeNum;
        const int PredRange = RangeOfSUnit[PredIdx];

        // If the predecessor is in a different scarce range, record the edge.
        if (PredRange != -1 && PredRange != static_cast<int>(RangeIdx)) {
          // Add to PredSet if not already present.
          if (std::find(PredSet.begin(), PredSet.end(), PredRange) ==
              PredSet.end()) {
            PredSet.push_back(PredRange);
          }
        }
      }
    }

    // Copy deduplicated predecessors to PredRanges.
    Range.PredRanges = PredSet;
  }
}

bool checkAcyclic(const std::vector<ScarceRange> &Ranges) {
  const size_t K = Ranges.size();

  // Compute indegrees (PredRanges.size() for each range).
  SmallVector<unsigned, 4> Indegree;
  Indegree.reserve(K);
  for (const auto &Range : Ranges) {
    Indegree.push_back(Range.PredRanges.size());
  }

  // Kahn's algorithm: process ranges with indegree 0.
  SmallVector<int, 4> Ready;
  for (size_t I = 0; I < K; ++I) {
    if (Indegree[I] == 0) {
      Ready.push_back(I);
    }
  }

  unsigned ProcessedCount = 0;
  while (!Ready.empty()) {
    const int Current = Ready.pop_back_val();
    ++ProcessedCount;

    // For each range that has Current as a predecessor, decrement indegree.
    for (size_t J = 0; J < K; ++J) {
      const auto &Range = Ranges[J];
      if (std::find(Range.PredRanges.begin(), Range.PredRanges.end(),
                    Current) != Range.PredRanges.end()) {
        --Indegree[J];
        if (Indegree[J] == 0) {
          Ready.push_back(J);
        }
      }
    }
  }

  // If we processed all ranges, the DAG is acyclic.
  return ProcessedCount == K;
}

bool enumerateRangeOrders(
    const std::vector<ScarceRange> &Ranges,
    llvm::function_ref<bool(const SmallVector<int, 4> &Order)> OnOrder) {

  const size_t K = Ranges.size();

  // Track which ranges have been placed in the current order.
  SmallVector<bool, 4> Placed(K, false);

  // Current partial order being built.
  SmallVector<int, 4> Order;
  Order.reserve(K);

  // Recursive DFS to enumerate linear extensions.
  const auto Enumerate = [&](auto &EnumerateRef) -> bool {
    // Base case: complete order found.
    if (Order.size() == K) {
      LLVM_DEBUG(dbgs() << "\nEntering burst scheduling with order ";
                 for (auto Ord : Order) { dbgs() << Ord << ", "; } dbgs()
                 << "\n";);
      return OnOrder(Order);
    }

    // Find ready ranges (all predecessors are in Order).
    for (size_t RangeIdx = 0; RangeIdx < K; ++RangeIdx) {
      if (Placed[RangeIdx]) {
        continue;
      }

      const auto &Range = Ranges[RangeIdx];

      // Check if all predecessors are placed.
      const bool AllPredsPlaced = llvm::all_of(
          Range.PredRanges, [&Placed](int PredIdx) { return Placed[PredIdx]; });

      if (AllPredsPlaced) {
        // This range is ready; add it to the order and recurse.

        Order.push_back(RangeIdx);
        Placed[RangeIdx] = true;

        if (EnumerateRef(EnumerateRef)) {
          return true;
        }

        // Backtrack.
        Placed[RangeIdx] = false;
        Order.pop_back();
      }
    }

    return false;
  };

  LLVM_DEBUG(dbgs() << "Enumerating scarce ranges\n");

  return Enumerate(Enumerate);
}

} // namespace llvm::AIE
