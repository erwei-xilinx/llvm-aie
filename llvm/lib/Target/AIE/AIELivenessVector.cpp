//===- AIELivenessVector.cpp - Liveness vector implementation ------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
//
// This file implements a vector-like container for liveness information that
// provides safe out-of-range access and common operations.
//
//===----------------------------------------------------------------------===//

#include "AIELivenessVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>

using namespace llvm;

namespace llvm {
namespace AIE {

bool Liveness::conflictsWith(const Liveness &Other) const {
  // Check register file lane conflicts.
  if ((Lanes & Other.Lanes).any()) {
    return true;
  }

  // Check bypass conflicts: read in one, write in other (same class).
  for (unsigned ReadClass : BypassReads) {
    if (llvm::is_contained(Other.BypassWrites, ReadClass)) {
      return true;
    }
  }
  for (unsigned WriteClass : BypassWrites) {
    if (llvm::is_contained(Other.BypassReads, WriteClass)) {
      return true;
    }
  }

  // Check bypass vs register file conflicts.
  // If one has bypass activity and the other has register lanes, they
  // conflict because they share the same register address.
  const bool ThisHasBypass = !BypassReads.empty() || !BypassWrites.empty();
  const bool OtherHasBypass =
      !Other.BypassReads.empty() || !Other.BypassWrites.empty();

  if (ThisHasBypass && Other.Lanes.any()) {
    return true;
  }
  if (OtherHasBypass && Lanes.any()) {
    return true;
  }

  return false;
}

LivenessVector::LivenessVector(size_t Size) : Elements(Size) {}

LivenessVector::LivenessVector(size_t Size, LaneBitmask InitialValue)
    : Elements(Size, Liveness(InitialValue)) {}

size_t LivenessVector::size() const { return Elements.size(); }

bool LivenessVector::empty() const { return Elements.empty(); }

Liveness &LivenessVector::operator[](size_t Index) {
  assert(Index < Elements.size() && "Index out of range");
  return Elements[Index];
}

const Liveness &LivenessVector::operator[](size_t Index) const {
  assert(Index < Elements.size() && "Index out of range");
  return Elements[Index];
}

Liveness LivenessVector::at(size_t Index) const {
  if (Index >= Elements.size()) {
    return Liveness();
  }
  return Elements[Index];
}

const SmallVector<Liveness, 8> &LivenessVector::getElements() const {
  return Elements;
}

LivenessVector &LivenessVector::operator|=(const LivenessVector &Other) {
  // Determine the maximum size needed
  const size_t MaxSize = std::max(Elements.size(), Other.Elements.size());

  // Extend this vector if needed
  if (MaxSize > Elements.size()) {
    Elements.resize(MaxSize);
  }

  // Union using at() which returns empty for out-of-bounds
  for (size_t I = 0; I < MaxSize; ++I) {
    Elements[I] |= Other.at(I);
  }
  return *this;
}

LivenessVector &LivenessVector::operator&=(const LivenessVector &Other) {
  // Use at() which returns empty for out-of-bounds
  for (size_t I = 0; I < Elements.size(); ++I) {
    Elements[I] &= Other.at(I);
  }
  return *this;
}

LivenessVector &LivenessVector::operator-=(const LivenessVector &Other) {
  // Use at() which returns empty for out-of-bounds
  for (size_t I = 0; I < Elements.size(); ++I) {
    Elements[I] -= Other.at(I);
  }
  return *this;
}

LivenessVector LivenessVector::operator|(const LivenessVector &Other) const {
  LivenessVector Result = *this;
  Result |= Other;
  return Result;
}

LivenessVector LivenessVector::operator&(const LivenessVector &Other) const {
  LivenessVector Result = *this;
  Result &= Other;
  return Result;
}

LivenessVector LivenessVector::operator-(const LivenessVector &Other) const {
  LivenessVector Result = *this;
  Result -= Other;
  return Result;
}

bool LivenessVector::overlaps(const LivenessVector &Other) const {
  const size_t MinSize = std::min(Elements.size(), Other.Elements.size());
  for (size_t I = 0; I < MinSize; ++I) {
    if (Elements[I].conflictsWith(Other.Elements[I])) {
      return true;
    }
  }
  return false;
}

bool LivenessVector::any() const {
  return llvm::any_of(Elements, [](const Liveness &L) { return L.any(); });
}

bool LivenessVector::none() const {
  return llvm::none_of(Elements, [](const Liveness &L) { return L.any(); });
}

void LivenessVector::dump() const {
  print(dbgs());
  dbgs() << '\n';
}

void LivenessVector::print(raw_ostream &OS) const {
  OS << "[";
  for (size_t I = 0; I < Elements.size(); ++I) {
    if (I > 0)
      OS << ", ";
    OS << PrintLaneMask(Elements[I].getLanes());
  }
  OS << "]";
}

} // namespace AIE
} // namespace llvm
