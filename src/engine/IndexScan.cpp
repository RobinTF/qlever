// Copyright 2015, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Björn Buchhold (buchhold@informatik.uni-freiburg.de)

#include "engine/IndexScan.h"

#include <absl/container/inlined_vector.h>
#include <absl/strings/str_join.h>

#include <sstream>
#include <string>

#include "engine/QueryExecutionTree.h"
#include "index/IndexImpl.h"
#include "parser/ParsedQuery.h"

using std::string;
using LazyScanMetadata = CompressedRelationReader::LazyScanMetadata;

// _____________________________________________________________________________
// Return the number of `Variables` given the `TripleComponent` values for
// `subject_`, `predicate` and `object`.
static size_t getNumberOfVariables(const TripleComponent& subject,
                                   const TripleComponent& predicate,
                                   const TripleComponent& object) {
  return static_cast<size_t>(subject.isVariable()) +
         static_cast<size_t>(predicate.isVariable()) +
         static_cast<size_t>(object.isVariable());
}

// _____________________________________________________________________________
IndexScan::IndexScan(QueryExecutionContext* qec, Permutation::Enum permutation,
                     const SparqlTripleSimple& triple, Graphs graphsToFilter,
                     std::optional<ScanSpecAndBlocks> scanSpecAndBlocks)
    : Operation(qec),
      permutation_(permutation),
      subject_(triple.s_),
      predicate_(triple.p_),
      object_(triple.o_),
      graphsToFilter_{std::move(graphsToFilter)},
      scanSpecAndBlocks_{
          std::move(scanSpecAndBlocks).value_or(getScanSpecAndBlocks())},
      scanSpecAndBlocksIsPrefiltered_{scanSpecAndBlocks.has_value()},
      numVariables_(getNumberOfVariables(subject_, predicate_, object_)) {
  // We previously had `nullptr`s here in unit tests. This is no longer
  // necessary nor allowed.
  AD_CONTRACT_CHECK(qec != nullptr);
  for (auto& [idx, variable] : triple.additionalScanColumns_) {
    additionalColumns_.push_back(idx);
    additionalVariables_.push_back(variable);
  }
  std::tie(sizeEstimateIsExact_, sizeEstimate_) = computeSizeEstimate();

  // Check the following invariant: All the variables must be at the end of the
  // permuted triple. For example in the PSO permutation, either only the O, or
  // the S and O, or all three of P, S, O, or none of them can be variables, all
  // other combinations are not supported.
  auto permutedTriple = getPermutedTriple();
  for (size_t i = 0; i < 3 - numVariables_; ++i) {
    AD_CONTRACT_CHECK(!permutedTriple.at(i)->isVariable());
  }
  for (size_t i = 3 - numVariables_; i < permutedTriple.size(); ++i) {
    AD_CONTRACT_CHECK(permutedTriple.at(i)->isVariable());
  }
}

// _____________________________________________________________________________
IndexScan::IndexScan(QueryExecutionContext* qec, Permutation::Enum permutation,
                     const TripleComponent& s, const TripleComponent& p,
                     const TripleComponent& o,
                     std::vector<ColumnIndex> additionalColumns,
                     std::vector<Variable> additionalVariables,
                     Graphs graphsToFilter, ScanSpecAndBlocks scanSpecAndBlocks,
                     bool scanSpecAndBlocksIsPrefiltered, VarsToKeep varsToKeep)
    : Operation(qec),
      permutation_(permutation),
      subject_(s),
      predicate_(p),
      object_(o),
      graphsToFilter_(std::move(graphsToFilter)),
      scanSpecAndBlocks_(std::move(scanSpecAndBlocks)),
      scanSpecAndBlocksIsPrefiltered_(scanSpecAndBlocksIsPrefiltered),
      numVariables_(getNumberOfVariables(subject_, predicate_, object_)),
      additionalColumns_(std::move(additionalColumns)),
      additionalVariables_(std::move(additionalVariables)),
      varsToKeep_{std::move(varsToKeep)} {
  std::tie(sizeEstimateIsExact_, sizeEstimate_) = computeSizeEstimate();
  determineMultiplicities();
}

// _____________________________________________________________________________
string IndexScan::getCacheKeyImpl() const {
  std::ostringstream os;
  auto permutationString = Permutation::toString(permutation_);

  if (numVariables_ == 3) {
    os << "SCAN FOR FULL INDEX " << permutationString;

  } else {
    os << "SCAN " << permutationString << " with ";
    auto addKey = [&os, &permutationString, this](size_t idx) {
      auto keyString = permutationString.at(idx);
      const auto& key = getPermutedTriple().at(idx)->toRdfLiteral();
      os << keyString << " = \"" << key << "\"";
    };
    for (size_t i = 0; i < 3 - numVariables_; ++i) {
      addKey(i);
      os << ", ";
    }
  }
  if (!additionalColumns_.empty()) {
    os << " Additional Columns: ";
    os << absl::StrJoin(additionalColumns(), " ");
  }
  if (graphsToFilter_.has_value()) {
    // The graphs are stored as a hash set, but we need a deterministic order.
    std::vector<std::string> graphIdVec;
    ql::ranges::transform(graphsToFilter_.value(),
                          std::back_inserter(graphIdVec),
                          &TripleComponent::toRdfLiteral);
    ql::ranges::sort(graphIdVec);
    os << "\nFiltered by Graphs:";
    os << absl::StrJoin(graphIdVec, " ");
  }

  if (varsToKeep_.has_value()) {
    os << "column subset " << absl::StrJoin(getSubsetForStrippedColumns(), ",");
  }
  return std::move(os).str();
}

// _____________________________________________________________________________
bool IndexScan::canResultBeCachedImpl() const {
  return !scanSpecAndBlocksIsPrefiltered_;
};

// _____________________________________________________________________________
string IndexScan::getDescriptor() const {
  return "IndexScan " + subject_.toString() + " " + predicate_.toString() +
         " " + object_.toString();
}

// _____________________________________________________________________________
size_t IndexScan::getResultWidth() const {
  if (varsToKeep_.has_value()) {
    return varsToKeep_.value().size();
  }
  return numVariables_ + additionalVariables_.size();
}

// _____________________________________________________________________________
std::vector<ColumnIndex> IndexScan::resultSortedOn() const {
  std::vector<ColumnIndex> result;
  for (auto i : ad_utility::integerRange(ColumnIndex{numVariables_})) {
    result.push_back(i);
  }
  for (size_t i = 0; i < additionalColumns_.size(); ++i) {
    if (additionalColumns_.at(i) == ADDITIONAL_COLUMN_GRAPH_ID) {
      result.push_back(numVariables_ + i);
    }
  }

  if (varsToKeep_.has_value()) {
    auto permutation = getSubsetForStrippedColumns();
    for (auto it = result.begin(); it != result.end(); ++it) {
      if (!ad_utility::contains(permutation, *it)) {
        result.erase(it, result.end());
        return result;
      }
    }
  }
  return result;
}

// _____________________________________________________________________________
std::optional<std::shared_ptr<QueryExecutionTree>>
IndexScan::setPrefilterGetUpdatedQueryExecutionTree(
    const std::vector<PrefilterVariablePair>& prefilterVariablePairs) const {
  if (!getLimitOffset().isUnconstrained() ||
      scanSpecAndBlocks_.sizeBlockMetadata_ == 0) {
    return std::nullopt;
  }

  auto optSortedVarColIdxPair =
      getSortedVariableAndMetadataColumnIndexForPrefiltering();
  if (!optSortedVarColIdxPair.has_value()) {
    return std::nullopt;
  }

  const auto& [sortedVar, colIdx] = optSortedVarColIdxPair.value();
  auto it =
      ql::ranges::find(prefilterVariablePairs, sortedVar, ad_utility::second);
  if (it != prefilterVariablePairs.end()) {
    const auto& vocab = getIndex().getVocab();
    // If the `BlockMetadataRanges` were previously prefiltered, AND-merge
    // the previous `BlockMetadataRanges` with the `BlockMetadataRanges`
    // retrieved via the newly passed prefilter. This corresponds logically to a
    // conjunction over the prefilters applied for this `IndexScan`.
    const auto& blockMetadataRanges =
        prefilterExpressions::detail::logicalOps::getIntersectionOfBlockRanges(
            it->first->evaluate(
                vocab, getScanSpecAndBlocks().getBlockMetadataSpan(), colIdx),
            scanSpecAndBlocks_.blockMetadata_);

    return makeCopyWithPrefilteredScanSpecAndBlocks(
        {scanSpecAndBlocks_.scanSpec_, blockMetadataRanges});
  }
  return std::nullopt;
}

// _____________________________________________________________________________
VariableToColumnMap IndexScan::computeVariableToColumnMap() const {
  VariableToColumnMap variableToColumnMap;
  auto isContained = [&](const Variable& var) {
    return !varsToKeep_.has_value() || varsToKeep_.value().contains(var);
  };
  auto addCol = [&isContained, &variableToColumnMap,
                 nextColIdx = ColumnIndex{0}](const Variable& var) mutable {
    if (!isContained(var)) {
      return;
    }
    // All the columns of an index scan only contain defined values.
    variableToColumnMap[var] = makeAlwaysDefinedColumn(nextColIdx);
    ++nextColIdx;
  };

  for (const TripleComponent* const ptr : getPermutedTriple()) {
    if (ptr->isVariable()) {
      addCol(ptr->getVariable());
    }
  }
  ql::ranges::for_each(additionalVariables_, addCol);
  return variableToColumnMap;
}

//______________________________________________________________________________
std::shared_ptr<QueryExecutionTree>
IndexScan::makeCopyWithPrefilteredScanSpecAndBlocks(
    ScanSpecAndBlocks scanSpecAndBlocks) const {
  return ad_utility::makeExecutionTree<IndexScan>(
      getExecutionContext(), permutation_, subject_, predicate_, object_,
      additionalColumns_, additionalVariables_, graphsToFilter_,
      std::move(scanSpecAndBlocks), true, varsToKeep_);
}

// _____________________________________________________________________________
Result::Generator IndexScan::chunkedIndexScan() const {
  for (IdTable& idTable : getLazyScan()) {
    co_yield {std::move(idTable), LocalVocab{}};
  }
}

// _____________________________________________________________________________
IdTable IndexScan::materializedIndexScan() const {
  IdTable idTable = getScanPermutation().scan(
      scanSpecAndBlocks_, additionalColumns(), cancellationHandle_,
      locatedTriplesSnapshot(), getLimitOffset());
  LOG(DEBUG) << "IndexScan result computation done.\n";
  checkCancellation();
  idTable = makeApplyColumnSubset()(std::move(idTable));
  AD_CORRECTNESS_CHECK(idTable.numColumns() == getResultWidth());
  return idTable;
}

// _____________________________________________________________________________
Result IndexScan::computeResult(bool requestLaziness) {
  LOG(DEBUG) << "IndexScan result computation...\n";
  if (requestLaziness) {
    return {chunkedIndexScan(), resultSortedOn()};
  }
  return {materializedIndexScan(), getResultSortedOn(), LocalVocab{}};
}

// _____________________________________________________________________________
const Permutation& IndexScan::getScanPermutation() const {
  return getIndex().getImpl().getPermutation(permutation_);
}

// _____________________________________________________________________________
std::pair<bool, size_t> IndexScan::computeSizeEstimate() const {
  AD_CORRECTNESS_CHECK(_executionContext);
  auto [lower, upper] = getScanPermutation().getSizeEstimateForScan(
      scanSpecAndBlocks_, locatedTriplesSnapshot());
  return {lower == upper, std::midpoint(lower, upper)};
}

// _____________________________________________________________________________
size_t IndexScan::getExactSize() const {
  AD_CORRECTNESS_CHECK(_executionContext);
  return getScanPermutation().getResultSizeOfScan(scanSpecAndBlocks_,
                                                  locatedTriplesSnapshot());
}

// _____________________________________________________________________________
size_t IndexScan::getCostEstimate() {
  // If we have a limit present, we only have to read the first
  // `limit + offset` elements.
  return getLimitOffset().upperBound(getSizeEstimateBeforeLimit());
}

// _____________________________________________________________________________
void IndexScan::determineMultiplicities() {
  multiplicity_ = [this]() -> std::vector<float> {
    const auto& idx = getIndex();
    if (numVariables_ == 0) {
      return {};
    } else if (numVariables_ == 1) {
      // There are no duplicate triples in RDF and two elements are fixed.
      return {1.0f};
    } else if (numVariables_ == 2) {
      return idx.getMultiplicities(*getPermutedTriple()[0], permutation_,
                                   locatedTriplesSnapshot());
    } else {
      AD_CORRECTNESS_CHECK(numVariables_ == 3);
      return idx.getMultiplicities(permutation_);
    }
  }();
  multiplicity_.resize(multiplicity_.size() + additionalColumns_.size(), 1.0f);

  if (varsToKeep_.has_value()) {
    std::vector<float> actualMultiplicites;
    for (size_t column : getSubsetForStrippedColumns()) {
      actualMultiplicites.push_back(multiplicity_.at(column));
    }
    multiplicity_ = std::move(actualMultiplicites);
  }
  AD_CONTRACT_CHECK(multiplicity_.size() == getResultWidth());
}

// _____________________________________________________________________________
std::array<const TripleComponent* const, 3> IndexScan::getPermutedTriple()
    const {
  std::array<const TripleComponent* const, 3> triple{&subject_, &predicate_,
                                                     &object_};
  // TODO<joka921> This place has to be changed once we have a permutation
  // that is primarily sorted by G (the graph id).
  return Permutation::toKeyOrder(permutation_).permuteTriple(triple);
}

// _____________________________________________________________________________
ScanSpecification IndexScan::getScanSpecification() const {
  const IndexImpl& index = getIndex().getImpl();
  return getScanSpecificationTc().toScanSpecification(index);
}

// _____________________________________________________________________________
ScanSpecificationAsTripleComponent IndexScan::getScanSpecificationTc() const {
  auto permutedTriple = getPermutedTriple();
  return {*permutedTriple[0], *permutedTriple[1], *permutedTriple[2],
          graphsToFilter_};
}

// _____________________________________________________________________________
std::optional<std::pair<Variable, ColumnIndex>>
IndexScan::getSortedVariableAndMetadataColumnIndexForPrefiltering() const {
  if (numVariables_ < 1) {
    return std::nullopt;
  }
  const auto& permutedTriple = getPermutedTriple();
  size_t colIdx = 3 - numVariables_;
  const auto& tripleComp = permutedTriple.at(colIdx);
  AD_CORRECTNESS_CHECK(tripleComp->isVariable());
  return std::make_pair(tripleComp->getVariable(), colIdx);
}

// ___________________________________________________________________________
Permutation::ScanSpecAndBlocks IndexScan::getScanSpecAndBlocks() const {
  return getScanPermutation().getScanSpecAndBlocks(getScanSpecification(),
                                                   locatedTriplesSnapshot());
}

// _____________________________________________________________________________
Permutation::IdTableGenerator IndexScan::getLazyScan(
    std::optional<std::vector<CompressedBlockMetadata>> blocks) const {
  // If there is a LIMIT or OFFSET clause that constrains the scan
  // (which can happen with an explicit subquery), we cannot use the prefiltered
  // blocks, as we currently have no mechanism to include limits and offsets
  // into the prefiltering (`std::nullopt` means `scan all blocks`).
  auto filteredBlocks =
      getLimitOffset().isUnconstrained() ? std::move(blocks) : std::nullopt;
  auto lazyScanAllCols = getScanPermutation().lazyScan(
      scanSpecAndBlocks_, filteredBlocks, additionalColumns(),
      cancellationHandle_, locatedTriplesSnapshot(), getLimitOffset());
  auto& detailsRef = co_await cppcoro::getDetails;
  lazyScanAllCols.setDetailsPointer(&detailsRef);
  auto applySubset = makeApplyColumnSubset();

  for (auto& table : lazyScanAllCols) {
    co_yield applySubset(std::move(table));
  }
};

// _____________________________________________________________________________
std::optional<Permutation::MetadataAndBlocks> IndexScan::getMetadataForScan()
    const {
  return getScanPermutation().getMetadataAndBlocks(scanSpecAndBlocks_,
                                                   locatedTriplesSnapshot());
};

// _____________________________________________________________________________
std::array<Permutation::IdTableGenerator, 2>
IndexScan::lazyScanForJoinOfTwoScans(const IndexScan& s1, const IndexScan& s2) {
  AD_CONTRACT_CHECK(s1.numVariables_ <= 3 && s2.numVariables_ <= 3);
  AD_CONTRACT_CHECK(s1.numVariables_ >= 1 && s2.numVariables_ >= 1);
  // This function only works for single column joins. This means that the first
  // variable of both scans must be equal, but all other variables of the scans
  // (if present) must be different.
  const auto& getFirstVariable = [](const IndexScan& scan) {
    auto numVars = scan.numVariables();
    AD_CORRECTNESS_CHECK(numVars <= 3);
    size_t indexOfFirstVar = 3 - numVars;
    ad_utility::HashSet<Variable> otherVars;
    for (size_t i = indexOfFirstVar + 1; i < 3; ++i) {
      const auto& el = *scan.getPermutedTriple()[i];
      if (el.isVariable()) {
        otherVars.insert(el.getVariable());
      }
    }
    return std::pair{*scan.getPermutedTriple()[3 - numVars],
                     std::move(otherVars)};
  };

  auto [first1, other1] = getFirstVariable(s1);
  auto [first2, other2] = getFirstVariable(s2);
  AD_CONTRACT_CHECK(first1 == first2);

  size_t numTotal = other1.size() + other2.size();
  for (auto& var : other1) {
    other2.insert(var);
  }
  AD_CONTRACT_CHECK(other2.size() == numTotal);

  auto metaBlocks1 = s1.getMetadataForScan();
  auto metaBlocks2 = s2.getMetadataForScan();

  if (!metaBlocks1.has_value() || !metaBlocks2.has_value()) {
    return {{}};
  }
  auto [blocks1, blocks2] = CompressedRelationReader::getBlocksForJoin(
      metaBlocks1.value(), metaBlocks2.value());

  std::array result{s1.getLazyScan(blocks1), s2.getLazyScan(blocks2)};
  result[0].details().numBlocksAll_ = metaBlocks1.value().sizeBlockMetadata_;
  result[1].details().numBlocksAll_ = metaBlocks2.value().sizeBlockMetadata_;
  return result;
}

// _____________________________________________________________________________
Permutation::IdTableGenerator IndexScan::lazyScanForJoinOfColumnWithScan(
    ql::span<const Id> joinColumn) const {
  AD_EXPENSIVE_CHECK(ql::ranges::is_sorted(joinColumn));
  AD_CORRECTNESS_CHECK(numVariables_ <= 3 && numVariables_ > 0);
  AD_CONTRACT_CHECK(joinColumn.empty() || !joinColumn[0].isUndefined());

  auto metaBlocks = getMetadataForScan();
  if (!metaBlocks.has_value()) {
    return {};
  }
  auto blocks = CompressedRelationReader::getBlocksForJoin(joinColumn,
                                                           metaBlocks.value());
  auto result = getLazyScan(std::move(blocks.matchingBlocks_));
  result.details().numBlocksAll_ = metaBlocks.value().sizeBlockMetadata_;
  return result;
}

// _____________________________________________________________________________
void IndexScan::updateRuntimeInfoForLazyScan(const LazyScanMetadata& metadata) {
  updateRuntimeInformationWhenOptimizedOut(
      RuntimeInformation::Status::lazilyMaterialized);
  auto& rti = runtimeInfo();
  rti.numRows_ = metadata.numElementsYielded_;
  rti.totalTime_ = metadata.blockingTime_;
  rti.addDetail("num-blocks-read", metadata.numBlocksRead_);
  rti.addDetail("num-blocks-all", metadata.numBlocksAll_);
  rti.addDetail("num-elements-read", metadata.numElementsRead_);

  // Add more details, but only if the respective value is non-zero.
  auto updateIfPositive = [&rti](const auto& value, const std::string& key) {
    if (value > 0) {
      rti.addDetail(key, value);
    }
  };
  updateIfPositive(metadata.numBlocksSkippedBecauseOfGraph_,
                   "num-blocks-skipped-graph");
  updateIfPositive(metadata.numBlocksPostprocessed_,
                   "num-blocks-postprocessed");
  updateIfPositive(metadata.numBlocksWithUpdate_, "num-blocks-with-update");
}

// Store a Generator and its corresponding iterator as well as unconsumed values
// resulting from the generator.
struct IndexScan::SharedGeneratorState {
  // The generator that yields the tables to be joined with the index scan.
  Result::LazyResult generator_;
  // The column index of the join column in the tables yielded by the generator.
  ColumnIndex joinColumn_;
  // Metadata and blocks of this index scan.
  Permutation::MetadataAndBlocks metaBlocks_;
  // The iterator of the generator that is currently being consumed.
  std::optional<Result::LazyResult::iterator> iterator_ = std::nullopt;
  // Values returned by the generator that have not been re-yielded yet.
  // Typically we expect only 3 or less values to be prefetched (this is an
  // implementation detail of `BlockZipperJoinImpl`).
  using PrefetchStorage = absl::InlinedVector<Result::IdTableVocabPair, 3>;
  PrefetchStorage prefetchedValues_{};
  // Metadata of blocks that still need to be read.
  std::vector<CompressedBlockMetadata> pendingBlocks_{};
  // The index of the last matching block that was found using the join column.
  std::optional<size_t> lastBlockIndex_ = std::nullopt;
  // Indicates if the generator has yielded any undefined values.
  bool hasUndef_ = false;
  // Indicates if the generator has been fully consumed.
  bool doneFetching_ = false;

  // Advance the `iterator` to the next non-empty table. Set `hasUndef_` to true
  // if the first table is undefined. Also set `doneFetching_` if the generator
  // has been fully consumed.
  void advanceInputToNextNonEmptyTable() {
    bool firstStep = !iterator_.has_value();
    if (iterator_.has_value()) {
      ++iterator_.value();
    } else {
      iterator_ = generator_.begin();
    }
    auto& iterator = iterator_.value();
    while (iterator != generator_.end()) {
      if (!iterator->idTable_.empty()) {
        break;
      }
      ++iterator;
    }
    doneFetching_ = iterator_ == generator_.end();
    // Set the undef flag if the first table is undefined.
    if (firstStep) {
      hasUndef_ =
          !doneFetching_ && iterator->idTable_.at(0, joinColumn_).isUndefined();
    }
  }

  // Consume the next non-empty table from the generator and calculate the next
  // matching blocks from the index scan. This function guarantees that after
  // it returns, both `prefetchedValues` and `pendingBlocks` contain at least
  // one element.
  void fetch() {
    while (prefetchedValues_.empty() || pendingBlocks_.empty()) {
      advanceInputToNextNonEmptyTable();
      if (doneFetching_) {
        return;
      }
      auto& idTable = iterator_.value()->idTable_;
      auto joinColumn = idTable.getColumn(joinColumn_);
      AD_EXPENSIVE_CHECK(ql::ranges::is_sorted(joinColumn));
      AD_CORRECTNESS_CHECK(!joinColumn.empty());
      // Skip processing for undef case, it will be handled differently
      if (hasUndef_) {
        return;
      }
      AD_CORRECTNESS_CHECK(!joinColumn[0].isUndefined());
      auto [newBlocks, numBlocksCompletelyHandled] =
          CompressedRelationReader::getBlocksForJoin(joinColumn, metaBlocks_);
      // The first `numBlocksCompletelyHandled` are either contained in
      // `newBlocks` or can never match any entry that is larger than the
      // entries in `joinColumn` and thus can be ignored from now on.
      metaBlocks_.removePrefix(numBlocksCompletelyHandled);
      if (newBlocks.empty()) {
        // The current input table matches no blocks, so we don't have to yield
        // it.
        continue;
      }
      prefetchedValues_.push_back(std::move(*iterator_.value()));
      // Find first value that differs from the last one that was used to find
      // matching blocks.
      auto startIterator =
          lastBlockIndex_.has_value()
              ? ql::ranges::upper_bound(newBlocks, lastBlockIndex_.value(), {},
                                        &CompressedBlockMetadata::blockIndex_)
              : newBlocks.begin();
      lastBlockIndex_ = newBlocks.back().blockIndex_;
      ql::ranges::move(startIterator, newBlocks.end(),
                       std::back_inserter(pendingBlocks_));
    }
  }

  // Check if there are any undefined values yielded by the original generator.
  // If the generator hasn't been started to get consumed, this will start it.
  bool hasUndef() {
    if (!iterator_.has_value()) {
      fetch();
    }
    return hasUndef_;
  }
};

// _____________________________________________________________________________
Result::Generator IndexScan::createPrefilteredJoinSide(
    std::shared_ptr<SharedGeneratorState> innerState) {
  if (innerState->hasUndef()) {
    AD_CORRECTNESS_CHECK(innerState->prefetchedValues_.empty());
    for (auto& value : ql::ranges::subrange{innerState->iterator_.value(),
                                            innerState->generator_.end()}) {
      co_yield value;
    }
    co_return;
  }
  auto& prefetchedValues = innerState->prefetchedValues_;
  while (true) {
    if (prefetchedValues.empty()) {
      if (innerState->doneFetching_) {
        co_return;
      }
      innerState->fetch();
      AD_CORRECTNESS_CHECK(!prefetchedValues.empty() ||
                           innerState->doneFetching_);
    }
    // Make a defensive copy of the values to avoid modification during
    // iteration when yielding.
    auto copy = std::move(prefetchedValues);
    // Moving out does not necessarily clear the values, so we do it explicitly.
    prefetchedValues.clear();
    for (auto& value : copy) {
      co_yield value;
    }
  }
}

// _____________________________________________________________________________
Result::Generator IndexScan::createPrefilteredIndexScanSide(
    std::shared_ptr<SharedGeneratorState> innerState) {
  if (innerState->hasUndef()) {
    for (auto& pair : chunkedIndexScan()) {
      co_yield pair;
    }
    co_return;
  }
  LazyScanMetadata metadata;
  auto& pendingBlocks = innerState->pendingBlocks_;
  while (true) {
    if (pendingBlocks.empty()) {
      if (innerState->doneFetching_) {
        metadata.numBlocksAll_ = innerState->metaBlocks_.sizeBlockMetadata_;
        updateRuntimeInfoForLazyScan(metadata);
        co_return;
      }
      innerState->fetch();
    }
    auto scan = getLazyScan(std::move(pendingBlocks));
    AD_CORRECTNESS_CHECK(pendingBlocks.empty());
    for (IdTable& idTable : scan) {
      co_yield {std::move(idTable), LocalVocab{}};
    }
    metadata.aggregate(scan.details());
  }
}

// _____________________________________________________________________________
std::pair<Result::Generator, Result::Generator> IndexScan::prefilterTables(
    Result::LazyResult input, ColumnIndex joinColumn) {
  AD_CORRECTNESS_CHECK(numVariables_ <= 3 && numVariables_ > 0);
  auto metaBlocks = getMetadataForScan();

  if (!metaBlocks.has_value()) {
    return {Result::Generator{}, Result::Generator{}};
  }
  auto state = std::make_shared<SharedGeneratorState>(
      std::move(input), joinColumn, std::move(metaBlocks.value()));
  return {createPrefilteredJoinSide(state),
          createPrefilteredIndexScanSide(state)};
}

// _____________________________________________________________________________
std::unique_ptr<Operation> IndexScan::cloneImpl() const {
  return std::make_unique<IndexScan>(
      _executionContext, permutation_, subject_, predicate_, object_,
      additionalColumns_, additionalVariables_, graphsToFilter_,
      scanSpecAndBlocks_, scanSpecAndBlocksIsPrefiltered_, varsToKeep_);
}

// _____________________________________________________________________________
bool IndexScan::columnOriginatesFromGraphOrUndef(
    const Variable& variable) const {
  AD_CONTRACT_CHECK(getExternallyVisibleVariableColumns().contains(variable));
  return variable == subject_ || variable == predicate_ || variable == object_;
}

// _____________________________________________________________________________
std::optional<std::shared_ptr<QueryExecutionTree>>
IndexScan::makeTreeWithStrippedColumns(
    const std::set<Variable>& variables) const {
  ad_utility::HashSet<Variable> newVariables;
  for (const auto& [var, _] : getExternallyVisibleVariableColumns()) {
    if (variables.contains(var)) {
      newVariables.insert(var);
    }
  }

  return ad_utility::makeExecutionTree<IndexScan>(
      _executionContext, permutation_, subject_, predicate_, object_,
      additionalColumns_, additionalVariables_, graphsToFilter_,
      scanSpecAndBlocks_, scanSpecAndBlocksIsPrefiltered_,
      VarsToKeep{std::move(newVariables)});
}

// _____________________________________________________________________________
std::vector<ColumnIndex> IndexScan::getSubsetForStrippedColumns() const {
  AD_CORRECTNESS_CHECK(varsToKeep_.has_value());
  const auto& v = varsToKeep_.value();
  std::vector<ColumnIndex> result;
  size_t idx = 0;
  for (const auto& el : getPermutedTriple()) {
    if (el->isVariable()) {
      if (v.contains(el->getVariable())) {
        result.push_back(idx);
      }
      ++idx;
    }
  }
  for (const auto& var : additionalVariables_) {
    if (v.contains(var)) {
      result.push_back(idx);
    }
    ++idx;
  }
  return result;
}
