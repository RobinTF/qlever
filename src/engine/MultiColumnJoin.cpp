// Copyright 2018 - 2025, University of Freiburg
// Chair of Algorithms and Data Structures
// Authors: Florian Kramer [2018 - 2020]
//          Johannes Kalmbach <kalmbach@cs.uni-freiburg.de>

#include "engine/MultiColumnJoin.h"

#include "engine/AddCombinedRowToTable.h"
#include "engine/CallFixedSize.h"
#include "engine/Engine.h"
#include "engine/IndexScan.h"
#include "engine/JoinHelpers.h"
#include "util/JoinAlgorithms/JoinAlgorithms.h"
#include "util/JoinAlgorithms/JoinColumnMapping.h"
#include "util/Timer.h"

using std::endl;
using std::string;

// _____________________________________________________________________________
MultiColumnJoin::MultiColumnJoin(QueryExecutionContext* qec,
                                 std::shared_ptr<QueryExecutionTree> t1,
                                 std::shared_ptr<QueryExecutionTree> t2,
                                 bool allowSwappingChildrenOnlyForTesting)
    : Operation{qec} {
  // Make sure subtrees are ordered so that identical queries can be identified.
  if (allowSwappingChildrenOnlyForTesting &&
      t1->getCacheKey() > t2->getCacheKey()) {
    std::swap(t1, t2);
  }
  std::tie(_left, _right, _joinColumns) =
      QueryExecutionTree::getSortedSubtreesAndJoinColumns(std::move(t1),
                                                          std::move(t2));
}

// _____________________________________________________________________________
string MultiColumnJoin::getCacheKeyImpl() const {
  std::ostringstream os;
  os << "MULTI_COLUMN_JOIN\n" << _left->getCacheKey() << " ";
  os << "join-columns: [";
  for (size_t i = 0; i < _joinColumns.size(); i++) {
    os << _joinColumns[i][0] << (i < _joinColumns.size() - 1 ? " & " : "");
  };
  os << "]\n";
  os << "|X|\n" << _right->getCacheKey() << " ";
  os << "join-columns: [";
  for (size_t i = 0; i < _joinColumns.size(); i++) {
    os << _joinColumns[i][1] << (i < _joinColumns.size() - 1 ? " & " : "");
  };
  os << "]";
  return std::move(os).str();
}

// _____________________________________________________________________________
string MultiColumnJoin::getDescriptor() const {
  std::string joinVars = "";
  for (auto jc : _joinColumns) {
    joinVars +=
        _left->getVariableAndInfoByColumnIndex(jc[0]).first.name() + " ";
  }
  return "MultiColumnJoin on " + joinVars;
}

// _____________________________________________________________________________
Result MultiColumnJoin::computeResult([[maybe_unused]] bool requestLaziness) {
  AD_LOG_DEBUG << "MultiColumnJoin result computation..." << endl;

  if (_left->knownEmptyResult() || _right->knownEmptyResult()) {
    _left->getRootOperation()->updateRuntimeInformationWhenOptimizedOut();
    _right->getRootOperation()->updateRuntimeInformationWhenOptimizedOut();
    IdTable emptyTable{getResultWidth(), allocator()};
    return {std::move(emptyTable), resultSortedOn(), LocalVocab{}};
  }

  // Check if exactly one of the children is an IndexScan
  auto leftIndexScan =
      std::dynamic_pointer_cast<IndexScan>(_left->getRootOperation());
  auto rightIndexScan =
      std::dynamic_pointer_cast<IndexScan>(_right->getRootOperation());

  // If left is an IndexScan and right is not, evaluate right first
  if (leftIndexScan && !rightIndexScan) {
    auto rightResult = _right->getResult();
    checkCancellation();

    if (rightResult->isFullyMaterialized() && rightResult->idTable().empty()) {
      _left->getRootOperation()->updateRuntimeInformationWhenOptimizedOut();
      IdTable emptyTable{getResultWidth(), allocator()};
      return {std::move(emptyTable), resultSortedOn(), LocalVocab{}};
    }

    // Apply optimization if right is fully materialized
    if (rightResult->isFullyMaterialized()) {
      return computeResultForIndexScanAndIdTable<true>(
          requestLaziness, std::move(rightResult), leftIndexScan);
    }

    // If right is not fully materialized, fall through to standard join
    auto leftResult = _left->getResult();
    checkCancellation();

    AD_LOG_DEBUG << "MultiColumnJoin subresult computation done." << std::endl;
    AD_LOG_DEBUG << "Computing a multi column join between results of size "
                 << leftResult->idTable().size() << " and "
                 << rightResult->idTable().size() << endl;

    IdTable idTable{getExecutionContext()->getAllocator()};
    idTable.setNumColumns(getResultWidth());
    AD_CONTRACT_CHECK(idTable.numColumns() >= _joinColumns.size());

    computeMultiColumnJoin(leftResult->idTable(), rightResult->idTable(),
                           _joinColumns, &idTable);
    checkCancellation();

    AD_LOG_DEBUG << "MultiColumnJoin result computation done" << endl;
    return {std::move(idTable), resultSortedOn(),
            Result::getMergedLocalVocab(*leftResult, *rightResult)};
  }

  // Get left result (either it's not an IndexScan, or right is an IndexScan)
  auto leftResult = _left->getResult();
  checkCancellation();

  if (leftResult->isFullyMaterialized() && leftResult->idTable().empty()) {
    _right->getRootOperation()->updateRuntimeInformationWhenOptimizedOut();
    IdTable emptyTable{getResultWidth(), allocator()};
    return {std::move(emptyTable), resultSortedOn(), LocalVocab{}};
  }

  // If right is an IndexScan and left is not, apply optimization if left is
  // fully materialized
  if (rightIndexScan && !leftIndexScan && leftResult->isFullyMaterialized()) {
    return computeResultForIndexScanAndIdTable<false>(
        requestLaziness, std::move(leftResult), rightIndexScan);
  }

  // Get right result for standard join
  const auto rightResult = _right->getResult();
  checkCancellation();

  AD_LOG_DEBUG << "MultiColumnJoin subresult computation done." << std::endl;

  AD_LOG_DEBUG << "Computing a multi column join between results of size "
               << leftResult->idTable().size() << " and "
               << rightResult->idTable().size() << endl;

  IdTable idTable{getExecutionContext()->getAllocator()};
  idTable.setNumColumns(getResultWidth());

  AD_CONTRACT_CHECK(idTable.numColumns() >= _joinColumns.size());

  computeMultiColumnJoin(leftResult->idTable(), rightResult->idTable(),
                         _joinColumns, &idTable);

  checkCancellation();

  AD_LOG_DEBUG << "MultiColumnJoin result computation done" << endl;
  // If only one of the two operands has a non-empty local vocabulary, share
  // with that one (otherwise, throws an exception).
  return {std::move(idTable), resultSortedOn(),
          Result::getMergedLocalVocab(*leftResult, *rightResult)};
}

// _____________________________________________________________________________
VariableToColumnMap MultiColumnJoin::computeVariableToColumnMap() const {
  return makeVarToColMapForJoinOperation(
      _left->getVariableColumns(), _right->getVariableColumns(), _joinColumns,
      BinOpType::Join, _left->getResultWidth());
}

// _____________________________________________________________________________
size_t MultiColumnJoin::getResultWidth() const {
  size_t res =
      _left->getResultWidth() + _right->getResultWidth() - _joinColumns.size();
  AD_CONTRACT_CHECK(res > 0);
  return res;
}

// _____________________________________________________________________________
std::vector<ColumnIndex> MultiColumnJoin::resultSortedOn() const {
  std::vector<ColumnIndex> sortedOn;
  // The result is sorted on all join columns from the left subtree.
  for (const auto& a : _joinColumns) {
    sortedOn.push_back(a[0]);
  }
  return sortedOn;
}

// _____________________________________________________________________________
float MultiColumnJoin::getMultiplicity(size_t col) {
  if (!_multiplicitiesComputed) {
    computeSizeEstimateAndMultiplicities();
  }
  return _multiplicities[col];
}

// _____________________________________________________________________________
uint64_t MultiColumnJoin::getSizeEstimateBeforeLimit() {
  if (!_multiplicitiesComputed) {
    computeSizeEstimateAndMultiplicities();
  }
  return _sizeEstimate;
}

// _____________________________________________________________________________
size_t MultiColumnJoin::getCostEstimate() {
  size_t costEstimate = getSizeEstimateBeforeLimit() +
                        _left->getSizeEstimate() + _right->getSizeEstimate();
  // This join is slower than a normal join, due to
  // its increased complexity
  costEstimate *= 2;
  // Make the join 7% more expensive per join column
  costEstimate *= (1 + (_joinColumns.size() - 1) * 0.07);
  return _left->getCostEstimate() + _right->getCostEstimate() + costEstimate;
}

// _____________________________________________________________________________
void MultiColumnJoin::computeSizeEstimateAndMultiplicities() {
  // The number of distinct entries in the result is at most the minimum of
  // the numbers of distinct entries in all join columns.
  // The multiplicity in the result is approximated by the product of the
  // maximum of the multiplicities of each side.

  // compute the minimum number of distinct elements in the join columns
  size_t numDistinctLeft = std::numeric_limits<size_t>::max();
  size_t numDistinctRight = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < _joinColumns.size(); i++) {
    size_t dl = std::max(1.0f, _left->getSizeEstimate() /
                                   _left->getMultiplicity(_joinColumns[i][0]));
    size_t dr = std::max(1.0f, _right->getSizeEstimate() /
                                   _right->getMultiplicity(_joinColumns[i][1]));
    numDistinctLeft = std::min(numDistinctLeft, dl);
    numDistinctRight = std::min(numDistinctRight, dr);
  }
  size_t numDistinctResult = std::min(numDistinctLeft, numDistinctRight);

  // compute an estimate for the results multiplicity
  float multLeft = std::numeric_limits<float>::max();
  float multRight = std::numeric_limits<float>::max();
  for (size_t i = 0; i < _joinColumns.size(); i++) {
    multLeft = std::min(multLeft, _left->getMultiplicity(_joinColumns[i][0]));
    multRight =
        std::min(multRight, _right->getMultiplicity(_joinColumns[i][1]));
  }
  float multResult = multLeft * multRight;

  _sizeEstimate = multResult * numDistinctResult;
  // Don't estimate 0 since then some parent operations
  // (in particular joins) using isKnownEmpty() will
  // will assume the size to be exactly zero
  _sizeEstimate += 1;

  // compute estimates for the multiplicities of the result columns
  _multiplicities.clear();

  for (size_t i = 0; i < _left->getResultWidth(); i++) {
    float mult = _left->getMultiplicity(i) * (multResult / multLeft);
    _multiplicities.push_back(mult);
  }

  for (size_t i = 0; i < _right->getResultWidth(); i++) {
    bool isJcl = false;
    for (size_t j = 0; j < _joinColumns.size(); j++) {
      if (_joinColumns[j][1] == i) {
        isJcl = true;
        break;
      }
    }
    if (isJcl) {
      continue;
    }
    float mult = _right->getMultiplicity(i) * (multResult / multRight);
    _multiplicities.push_back(mult);
  }
  _multiplicitiesComputed = true;
}

// _______________________________________________________________________
void MultiColumnJoin::computeMultiColumnJoin(
    const IdTable& left, const IdTable& right,
    const std::vector<std::array<ColumnIndex, 2>>& joinColumns,
    IdTable* result) {
  // check for trivial cases
  if (left.empty() || right.empty()) {
    return;
  }

  ad_utility::JoinColumnMapping joinColumnData{joinColumns, left.numColumns(),
                                               right.numColumns()};

  IdTableView<0> leftJoinColumns =
      left.asColumnSubsetView(joinColumnData.jcsLeft());
  IdTableView<0> rightJoinColumns =
      right.asColumnSubsetView(joinColumnData.jcsRight());

  auto leftPermuted = left.asColumnSubsetView(joinColumnData.permutationLeft());
  auto rightPermuted =
      right.asColumnSubsetView(joinColumnData.permutationRight());

  auto rowAdder = ad_utility::AddCombinedRowToIdTable(
      joinColumns.size(), leftPermuted, rightPermuted, std::move(*result),
      cancellationHandle_);
  auto addRow = [&rowAdder, beginLeft = leftJoinColumns.begin(),
                 beginRight = rightJoinColumns.begin()](const auto& itLeft,
                                                        const auto& itRight) {
    rowAdder.addRow(itLeft - beginLeft, itRight - beginRight);
  };

  // Compute `isCheap`, which is true iff there are no UNDEF values in the join
  // columns (in which case we can use a simpler and cheaper join algorithm).
  //
  // TODO<joka921> This is the most common case. There are many other cases
  // where the generic `zipperJoinWithUndef` can be optimized. We will those
  // for a later PR.
  bool isCheap = ql::ranges::none_of(joinColumns, [&](const auto& jcs) {
    auto [leftCol, rightCol] = jcs;
    return (ql::ranges::any_of(right.getColumn(rightCol), &Id::isUndefined)) ||
           (ql::ranges::any_of(left.getColumn(leftCol), &Id::isUndefined));
  });

  auto checkCancellationLambda = [this] { checkCancellation(); };

  const size_t numOutOfOrder = [&]() {
    if (isCheap) {
      return ad_utility::zipperJoinWithUndef(
          leftJoinColumns, rightJoinColumns,
          ql::ranges::lexicographical_compare, addRow, ad_utility::noop,
          ad_utility::noop, ad_utility::noop, checkCancellationLambda);
    } else {
      return ad_utility::zipperJoinWithUndef(
          leftJoinColumns, rightJoinColumns,
          ql::ranges::lexicographical_compare, addRow,
          ad_utility::findSmallerUndefRanges,
          ad_utility::findSmallerUndefRanges, ad_utility::noop,
          checkCancellationLambda);
    }
  }();
  *result = std::move(rowAdder).resultTable();
  // If there were UNDEF values in the input, the result might be out of
  // order. Sort it, because this operation promises a sorted result in its
  // `resultSortedOn()` member function.
  // TODO<joka921> We only have to do this if the sorting is required (merge the
  // other PR first).
  if (numOutOfOrder > 0) {
    std::vector<ColumnIndex> cols;
    for (size_t i = 0; i < joinColumns.size(); ++i) {
      cols.push_back(i);
    }
    checkCancellation();
    Engine::sort(*result, cols);
  }

  // The result that `zipperJoinWithUndef` produces has a different order of
  // columns than expected, permute them. See the documentation of
  // `JoinColumnMapping` for details.
  result->setColumnSubset(joinColumnData.permutationResult());
  checkCancellation();
}

// _____________________________________________________________________________
std::unique_ptr<Operation> MultiColumnJoin::cloneImpl() const {
  auto copy = std::make_unique<MultiColumnJoin>(*this);
  copy->_left = _left->clone();
  copy->_right = _right->clone();
  return copy;
}

// _____________________________________________________________________________
bool MultiColumnJoin::columnOriginatesFromGraphOrUndef(
    const Variable& variable) const {
  AD_CONTRACT_CHECK(getExternallyVisibleVariableColumns().contains(variable));
  // For the join columns we don't union the elements, we intersect them so we
  // can have a more efficient implementation.
  if (_left->getVariableColumnOrNullopt(variable).has_value() &&
      _right->getVariableColumnOrNullopt(variable).has_value()) {
    using namespace qlever::joinHelpers;
    return doesJoinProduceGuaranteedGraphValuesOrUndef(_left, _right, variable);
  }
  return Operation::columnOriginatesFromGraphOrUndef(variable);
}

// _____________________________________________________________________________
ad_utility::AddCombinedRowToIdTable MultiColumnJoin::makeRowAdder(
    std::function<void(IdTable&, LocalVocab&)> callback) const {
  return ad_utility::AddCombinedRowToIdTable{
      _joinColumns.size(),
      IdTable{getResultWidth(), allocator()},
      cancellationHandle_,
      true,  // keepJoinColumn is always true for MultiColumnJoin
      qlever::joinHelpers::CHUNK_SIZE,
      std::move(callback)};
}

// _____________________________________________________________________________
template <bool idTableIsRightInput>
Result MultiColumnJoin::computeResultForIndexScanAndIdTable(
    bool requestLaziness, std::shared_ptr<const Result> resultWithIdTable,
    std::shared_ptr<IndexScan> scan) const {
  using namespace qlever::joinHelpers;

  // For multi-column joins, all join columns must be at the beginning after
  // permutation (indices 0, 1, 2, ..., n-1 where n = number of join columns).
  for (size_t i = 0; i < _joinColumns.size(); ++i) {
    auto expectedCol = static_cast<ColumnIndex>(i);
    auto actualCol =
        idTableIsRightInput ? _joinColumns[i][1] : _joinColumns[i][0];
    AD_CORRECTNESS_CHECK(actualCol == expectedCol);
  }

  ad_utility::JoinColumnMapping joinColMap{
      _joinColumns, _left->getResultWidth(), _right->getResultWidth()};
  auto resultPermutation = joinColMap.permutationResult();

  auto action = [this, scan = std::move(scan),
                 resultWithIdTable = std::move(resultWithIdTable),
                 joinColMap = std::move(joinColMap)](
                    std::function<void(IdTable&, LocalVocab&)> yieldTable) {
    const IdTable& idTable = resultWithIdTable->idTable();
    auto rowAdder = makeRowAdder(std::move(yieldTable));

    // Create a permuted view containing all columns in the right order
    auto permutedView = idTable.asColumnSubsetView(
        idTableIsRightInput ? joinColMap.permutationRight()
                            : joinColMap.permutationLeft());

    // For block filtering (conservative, using only first column)
    auto firstColForFiltering = ad_utility::IdTableAndFirstCol{
        permutedView, resultWithIdTable->getCopyOfLocalVocab()};

    ad_utility::Timer timer{ad_utility::timer::Timer::InitialStatus::Started};

    // Check if any of the join columns have UNDEF values
    bool idTableHasUndef = false;
    if (!idTable.empty()) {
      for (size_t i = 0; i < _joinColumns.size(); ++i) {
        auto col =
            idTableIsRightInput ? _joinColumns[i][1] : _joinColumns[i][0];
        if (idTable.at(0, col).isUndefined()) {
          idTableHasUndef = true;
          break;
        }
      }
    }

    std::optional<std::shared_ptr<const Result>> indexScanResult = std::nullopt;

    auto rightBlocks = [&scan, idTableHasUndef, &firstColForFiltering,
                        &indexScanResult]()
        -> std::variant<LazyInputView, GeneratorWithDetails> {
      if (idTableHasUndef) {
        indexScanResult =
            scan->getResult(false, ComputationMode::LAZY_IF_SUPPORTED);
        AD_CORRECTNESS_CHECK(!indexScanResult.value()->isFullyMaterialized());
        return convertGenerator(indexScanResult.value()->idTables());
      } else {
        // Use the first column of the join for block filtering
        // (conservative optimization for multi-column joins)
        auto rightBlocksInternal =
            scan->lazyScanForJoinOfColumnWithScan(firstColForFiltering.col());
        return convertGenerator(std::move(rightBlocksInternal));
      }
    }();

    runtimeInfo().addDetail("time-for-filtering-blocks", timer.msecs());

    // For multi-column joins, we use std::less{} for conservative comparison
    // on the first column only (block filtering), and let
    // AddCombinedRowToIdTable verify all join columns match when adding rows
    auto doJoin = [&rowAdder](auto& left, auto& right) mutable {
      // Use std::less{} to compare only the first column (conservative for
      // multi-column)
      ad_utility::zipperJoinForBlocksWithPotentialUndef(left, right,
                                                        std::less{}, rowAdder);
    };

    auto blockForIdTable = std::array{std::move(firstColForFiltering)};
    std::visit(
        [&doJoin, &blockForIdTable](auto& blocks) {
          if constexpr (idTableIsRightInput) {
            doJoin(blocks, blockForIdTable);
          } else {
            doJoin(blockForIdTable, blocks);
          }
        },
        rightBlocks);

    if (std::holds_alternative<GeneratorWithDetails>(rightBlocks)) {
      scan->updateRuntimeInfoForLazyScan(
          std::get<GeneratorWithDetails>(rightBlocks).details());
    }

    auto localVocab = std::move(rowAdder.localVocab());
    return Result::IdTableVocabPair{std::move(rowAdder).resultTable(),
                                    std::move(localVocab)};
  };

  if (requestLaziness) {
    return {runLazyJoinAndConvertToGenerator(std::move(action),
                                             std::move(resultPermutation)),
            resultSortedOn()};
  } else {
    auto [idTable, localVocab] = action(ad_utility::noop);
    applyPermutation(idTable, resultPermutation);
    return {std::move(idTable), resultSortedOn(), std::move(localVocab)};
  }
}

// Explicit template instantiations
template Result MultiColumnJoin::computeResultForIndexScanAndIdTable<true>(
    bool, std::shared_ptr<const Result>, std::shared_ptr<IndexScan>) const;
template Result MultiColumnJoin::computeResultForIndexScanAndIdTable<false>(
    bool, std::shared_ptr<const Result>, std::shared_ptr<IndexScan>) const;
