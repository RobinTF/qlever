//  Copyright 2026, University of Freiburg,
//  Chair of Algorithms and Data Structures.
//  Author: Robin Textor-Falconi <textorr@informatik.uni-freiburg.de>

#ifndef QLEVER_SRC_INDEX_LABELEDBLANKNODES_H
#define QLEVER_SRC_INDEX_LABELEDBLANKNODES_H

#include <absl/container/flat_hash_map.h>
#include <absl/hash/hash.h>

#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "global/Id.h"
#include "util/Exception.h"

// The blank nodes that `BNODE` with an argument has created, keyed by a pair of
// a `uint64_t` that identifies the solution (e.g. the bits of a solution token,
// or a row index) and the argument (the label). `BNODE` has to return the same
// blank node for the same label within one solution, see
// `BlankNodeExpression.cpp` for details.
class LabeledBlankNodes {
  using Key = std::pair<uint64_t, std::string>;
  using KeyView = std::pair<uint64_t, std::string_view>;

  // Hash and equality that also accept a `KeyView`, so that lookups don't have
  // to allocate a `std::string`.
  struct Hash {
    using is_transparent = void;
    size_t operator()(const KeyView& key) const {
      return absl::HashOf(key.first, key.second);
    }
  };
  struct Equal {
    using is_transparent = void;
    bool operator()(const KeyView& a, const KeyView& b) const { return a == b; }
  };

  absl::flat_hash_map<Key, Id, Hash, Equal> blankNodes_;

 public:
  // Return the blank node for the pair of `key` and `label` if there is one.
  std::optional<Id> find(uint64_t key, std::string_view label) const {
    auto it = blankNodes_.find(KeyView{key, label});
    if (it == blankNodes_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  // Store `blankNode` for the pair of `key` and `label`, which must not have a
  // blank node yet.
  void insert(uint64_t key, std::string_view label, Id blankNode) {
    bool isNew =
        blankNodes_.try_emplace(Key{key, std::string{label}}, blankNode).second;
    AD_CORRECTNESS_CHECK(isNew);
  }

  // Return the blank node for the pair of `key` and `label`. If there is none
  // yet, create it by calling `createBlankNode` and store it.
  template <typename CreateBlankNode>
  Id getOrCreate(uint64_t key, std::string_view label,
                 const CreateBlankNode& createBlankNode) {
    if (auto blankNode = find(key, label)) {
      return blankNode.value();
    }
    Id blankNode = createBlankNode();
    insert(key, label, blankNode);
    return blankNode;
  }

  size_t size() const { return blankNodes_.size(); }
};

#endif  // QLEVER_SRC_INDEX_LABELEDBLANKNODES_H
