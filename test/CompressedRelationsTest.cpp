//  Copyright 2023, University of Freiburg,
//                  Chair of Algorithms and Data Structures.
//  Author: Johannes Kalmbach <kalmbach@cs.uni-freiburg.de>

#include <gtest/gtest.h>

#include "./util/GTestHelpers.h"
#include "./util/IdTableHelpers.h"
#include "index/CompressedRelation.h"
#include "index/IndexImpl.h"
#include "util/IndexTestHelpers.h"
#include "util/OnDestructionDontThrowDuringStackUnwinding.h"
#include "util/Serializer/ByteBufferSerializer.h"
#include "util/SourceLocation.h"

namespace {

using ad_utility::source_location;

const LocatedTriplesPerBlock emptyLocatedTriples{};

// Return an `ID` of type `VocabIndex` from `index`. Assert that `index`
// is `>= 0`.
Id V(int64_t index) {
  AD_CONTRACT_CHECK(index >= 0);
  return Id::makeFromVocabIndex(VocabIndex::make(index));
}

// Retrieve the corresponding `BlockMetadataRanges` value for the
// given`CompressedBlockMetadata` vector
auto getBlockMetadataRangesfromVec =
    [](const std::vector<CompressedBlockMetadata>& blockMetadata)
    -> BlockMetadataRanges {
  BlockMetadataSpan blockMetadataSpan(blockMetadata);
  const size_t numBlocks = blockMetadataSpan.size();
  if (numBlocks < 3) {
    return {{blockMetadataSpan.begin(), blockMetadataSpan.end()}};
  }
  // If the BlockMetadataSpan contains more than three block values, split it
  // into three random subspans.
  BlockMetadataRanges ranges;
  ad_utility::SlowRandomIntGenerator<size_t> gen(0, numBlocks);
  auto split1 = gen();
  auto split2 = gen();
  if (split2 < split1) {
    std::swap(split1, split2);
  }
  auto begin = blockMetadataSpan.begin();
  // Add the three susbspans defined by the first and second splitting point.
  ranges.emplace_back(begin, begin + split1);
  ranges.emplace_back(begin + split1, begin + split2);
  ranges.emplace_back(begin + split2, begin + numBlocks);
  return ranges;
};

// A default graph IRI that is used in test cases where we don't care about the
// graph.
const Id g = V(1234059);

// A representation of a relation, consisting of the constant `col0_` element
// as well as the 2D-vector for the other two columns. `col1And2_` must be
// sorted lexicographically.
using RowInput = std::vector<int>;
struct RelationInput {
  int col0_;
  std::vector<RowInput> col1And2_;
};

template <typename Inner>
size_t getNumColumns(const std::vector<Inner>& input) {
  if (input.empty()) {
    return 2;
  }
  auto result = input.at(0).size();
  AD_CONTRACT_CHECK(ql::ranges::all_of(
      input, [result](const auto& vec) { return vec.size() == result; }));
  return result;
}

size_t getNumColumns(const std::vector<RelationInput>& vec) {
  if (vec.empty()) {
    return 2;
  }
  auto result = getNumColumns(vec.at(0).col1And2_);
  AD_CONTRACT_CHECK(ql::ranges::all_of(vec, [&result](const auto& relation) {
    return getNumColumns(relation.col1And2_) == result;
  }));
  return result;
}

// Check that `expected` and `actual` have the same contents. The `int`s in
// expected are converted to `Id`s of type `VocabIndex` using the `V`-function
// before the comparison.
void checkThatTablesAreEqual(const auto& expected, const IdTable& actual,
                             source_location l = source_location::current()) {
  auto trace = generateLocationTrace(l);

  VectorTable exp;
  for (const auto& row : expected) {
    exp.emplace_back();
    for (auto& el : row) {
      exp.back().push_back(el);
    }
  }
  EXPECT_THAT(actual, matchesIdTableFromVector(exp));
}

// If the `inputs` have no graph column (because the corresponding tests don't
// care about named graphs), add a constant dummy graph column, such that the
// assertions inside `CompressedRelation.cpp` (which always expect a graph
// column) work.
auto addGraphColumnIfNecessary(std::vector<RelationInput>& inputs) {
  size_t numColumns = getNumColumns(inputs) + 1;
  if (numColumns == 3) {
    ++numColumns;
    for (auto& input : inputs) {
      for (auto& row : input.col1And2_) {
        row.push_back(103496581);
      }
    }
  }
}
}  // namespace

// Write the given `inputs` (of type `RelationInput`) to a compressed
// permutation that is stored at the given `filename`.  Return the created
// metadata for the blocks and large relations.
// Note: This function can't be declared in the anonymous namespace, because it
// has to be a `friend` of the `CompressedRelationWriter` class. We therefore
// give it a rather long name.
template <typename T>
std::pair<std::vector<CompressedBlockMetadata>,
          std::vector<CompressedRelationMetadata>>
compressedRelationTestWriteCompressedRelations(
    T inputs, std::string filename, ad_utility::MemorySize blocksize) {
  // First check the invariants of the `inputs`. They must be sorted by the
  // `col0_` and for each of the `inputs` the `col1And2_` must also be sorted.
  AD_CONTRACT_CHECK(ql::ranges::is_sorted(
      inputs, {}, [](const RelationInput& r) { return r.col0_; }));
  AD_CONTRACT_CHECK(ql::ranges::all_of(inputs, [](const RelationInput& r) {
    return ql::ranges::is_sorted(r.col1And2_, [](const auto& a, const auto& b) {
      return ql::ranges::lexicographical_compare(a, b);
    });
  }));

  // First create the on-disk permutation.
  addGraphColumnIfNecessary(inputs);
  size_t numColumns = getNumColumns(inputs) + 1;
  AD_CORRECTNESS_CHECK(numColumns >= 4);
  CompressedRelationWriter writer{numColumns, ad_utility::File{filename, "w"},
                                  blocksize};
  std::vector<CompressedRelationMetadata> metaData;
  {
    size_t i = 0;
    for (const auto& input : inputs) {
      std::string bufferFilename =
          filename + ".buffers." + std::to_string(i) + ".dat";
      IdTable buffer{numColumns, ad_utility::makeUnlimitedAllocator<Id>()};
      size_t numBlocks = 0;

      auto addBlock = [&]() {
        if (buffer.empty()) {
          return;
        }
        writer.addBlockForLargeRelation(
            V(input.col0_), std::make_shared<IdTable>(std::move(buffer)));
        buffer.clear();
        ++numBlocks;
      };
      for (const auto& arr : input.col1And2_) {
        std::vector row{V(input.col0_)};
        ql::ranges::transform(arr, std::back_inserter(row), V);
        buffer.push_back(row);
        if (buffer.numRows() > writer.blocksize()) {
          addBlock();
        }
      }
      if (numBlocks > 0 || buffer.numRows() > 0.8 * writer.blocksize()) {
        addBlock();
        // The last argument is the number of distinct elements in `col1`. We
        // store a dummy value here that we can check later.
        metaData.push_back(writer.finishLargeRelation(i + 1));
      } else {
        metaData.push_back(writer.addSmallRelation(V(input.col0_), i + 1,
                                                   buffer.asStaticView<0>()));
      }
      buffer.clear();
      numBlocks = 0;
      ++i;
    }
  }
  auto blocks = std::move(writer).getFinishedBlocks();
  // Test the serialization of the blocks and the metaData.
  ad_utility::serialization::ByteBufferWriteSerializer w;
  w << metaData;
  w << blocks;
  metaData.clear();
  blocks.clear();
  ad_utility::serialization::ByteBufferReadSerializer r{std::move(w).data()};
  r >> metaData;
  r >> blocks;

  EXPECT_EQ(metaData.size(), inputs.size());

  for (size_t i : ad_utility::integerRange(blocks.size())) {
    EXPECT_EQ(blocks.at(i).blockIndex_, i);
  }

  return {std::move(blocks), std::move(metaData)};
}

namespace {
// Create a safe cleanup object, that automatically tries to delete the file at
// the given `filename` when it is destroyed. This is used to delete the
// persistent index files that are created for these tests.
auto makeCleanup(std::string filename) {
  return ad_utility::makeOnDestructionDontThrowDuringStackUnwinding(
      [filename = std::move(filename)] { ad_utility::deleteFile(filename); });
}

// From the `inputs` delete each triple with probability `locatedProbab` and
// add it to a vector of `IdTriple`s which can then be used to build a
// `LocatedTriples` object. Return the remaining triples and the (not-yet)
// located triples.
std::tuple<std::vector<RelationInput>, std::vector<IdTriple<>>>
makeLocatedTriplesFromPartOfInput(float locatedProbab,
                                  const std::vector<RelationInput>& inputs) {
  std::vector<IdTriple<>> locatedTriples;
  std::vector<RelationInput> result;
  ad_utility::RandomDoubleGenerator randomGenerator(0.0, 1.0);
  auto gen = [&randomGenerator, &locatedProbab]() {
    auto r = randomGenerator();
    return locatedProbab == 1.0f || r < locatedProbab;
  };

  auto addLocated = [&locatedTriples](Id col0, const auto& otherCols) {
    locatedTriples.push_back(IdTriple<>{
        {col0, V(otherCols.at(0)), V(otherCols.at(1)), V(otherCols.at(2))}});
  };

  for (const auto& input : inputs) {
    auto col0 = V(input.col0_);
    result.emplace_back(input.col0_);
    auto& row = result.back().col1And2_;
    for (const auto& otherCols : input.col1And2_) {
      AD_CORRECTNESS_CHECK(otherCols.size() >= 3);
      auto isLocated = gen();
      if (isLocated) {
        addLocated(col0, otherCols);
      } else {
        row.push_back(otherCols);
      }
    }
    if (row.empty()) {
      result.pop_back();
    }
  }
  return {std::move(result), std::move(locatedTriples)};
}

// Write the relations specified by the `inputs` to a compressed permutation at
// `filename`. Return the created metadata for blocks and large relations, as
// well as a `CompressedRelationReader`. These are exactly the datastructures
// that are required to test the `CompressedRelationReader` class.
auto writeAndOpenRelations(const std::vector<RelationInput>& inputs,
                           std::string filename,
                           ad_utility::MemorySize blocksize) {
  auto [blocks, metaData] = compressedRelationTestWriteCompressedRelations(
      inputs, filename, blocksize);
  auto reader = [&]() {
    return std::make_unique<CompressedRelationReader>(
        ad_utility::makeUnlimitedAllocator<Id>(),
        ad_utility::File{filename, "r"});
  };
  return std::tuple{std::move(blocks), std::move(metaData), reader()};
}

// Run a set of tests on a permutation that is defined by the `inputs`. The
// `inputs` must be ordered wrt the `col0_`. `testCaseName` is used to create
// a unique name for the required temporary files and for the implicit cache
// of the `CompressedRelationMetaData`. `blocksize` is the size of the blocks
// in which the permutation will be compressed and stored on disk.
void testCompressedRelations(const auto& inputsOriginalBeforeCopy,
                             std::string testCaseName,
                             ad_utility::MemorySize blocksize,
                             float locatedTriplesProbability = 0.5) {
  using ScanSpecAndBlocks = CompressedRelationReader::ScanSpecAndBlocks;
  auto inputs = inputsOriginalBeforeCopy;
  addGraphColumnIfNecessary(inputs);
  auto [inputsWithoutLocated, locatedTriplesInput] =
      makeLocatedTriplesFromPartOfInput(locatedTriplesProbability, inputs);
  DeltaTriples deltaTriples{ad_utility::testing::getQec()->getIndex()};
  auto filename = testCaseName + ".dat";
  auto cleanup = makeCleanup(filename);
  auto [blocksOriginal, metaData, readerPtr] =
      writeAndOpenRelations(inputsWithoutLocated, filename, blocksize);
  auto handle = std::make_shared<ad_utility::CancellationHandle<>>();
  // deltaTriples.insertTriples(handle, std::move(locatedTriplesInput));
  // auto locatedTriples =
  // deltaTriples.getLocatedTriplesPerBlock(Permutation::SPO);
  auto locatedTriples = LocatedTriplesPerBlock{};
  auto loc = LocatedTriple::locateTriplesInPermutation(
      locatedTriplesInput, blocksOriginal, {0, 1, 2, 3}, true, handle);
  locatedTriples.add(loc);
  locatedTriples.setOriginalMetadata(blocksOriginal);
  locatedTriples.updateAugmentedMetadata();
  auto blocks =
      getBlockMetadataRangesfromVec(locatedTriples.getAugmentedMetadata());

  auto& reader = *readerPtr;

  auto cancellationHandle =
      std::make_shared<ad_utility::CancellationHandle<>>();
  // Check the contents of the metadata.

  // TODO<C++23> `ql::ranges::to<vector>`.
  std::vector<ColumnIndex> additionalColumns;
  ql::ranges::copy(ql::views::iota(3ul, getNumColumns(inputs) + 1),
                   std::back_inserter(additionalColumns));
  auto getMetadata = [&, &metaData = metaData](size_t i) {
    Id col0 = V(inputs[i].col0_);
    auto it = ql::ranges::lower_bound(metaData, col0, {},
                                      &CompressedRelationMetadata::col0Id_);
    if (it != metaData.end() && it->col0Id_ == col0) {
      return *it;
    }
    return reader
        .getMetadataForSmallRelation(
            ScanSpecAndBlocks{
                ScanSpecification{col0, std::nullopt, std::nullopt}, blocks},
            col0, locatedTriples)
        .value();
  };
  for (size_t i = 0; i < inputs.size(); ++i) {
    // The metadata does not include the located triples, so we can only test it
    // if there are no located triples.
    if (locatedTriplesProbability == 0) {
      const auto& m = getMetadata(i);
      ASSERT_EQ(V(inputs[i].col0_), m.col0Id_);
      ASSERT_EQ(inputs[i].col1And2_.size(), m.numRows_);
      //  The number of distinct elements in `col1` was passed in as `i + 1` for
      //  testing purposes, so this is the expected multiplicity.
      ASSERT_FLOAT_EQ(m.numRows_ / static_cast<float>(i + 1),
                      m.multiplicityCol1_);
    }

    // Scan for all distinct `col0` and check that we get the expected result.
    ScanSpecification scanSpec{V(inputs[i].col0_), std::nullopt, std::nullopt};
    IdTable table =
        reader.scan(ScanSpecAndBlocks{scanSpec, blocks}, additionalColumns,
                    cancellationHandle, locatedTriples);
    const auto& col1And2 = inputs[i].col1And2_;
    checkThatTablesAreEqual(col1And2, table);
    table.clear();
    // Check that the scans also work with various values for LIMIT and OFFSET.
    std::vector<LimitOffsetClause> limitOffsetClauses{
        {std::nullopt, 5}, {5, 0}, {std::nullopt, 12}, {12, 0}, {7, 5}};
    for (const auto& limitOffset : limitOffsetClauses) {
      IdTable table =
          reader.scan(ScanSpecAndBlocks{scanSpec, blocks}, additionalColumns,
                      cancellationHandle, locatedTriples, limitOffset);
      auto col1And2 = inputs[i].col1And2_;
      col1And2.resize(limitOffset.upperBound(col1And2.size()));
      col1And2.erase(
          col1And2.begin(),
          col1And2.begin() + limitOffset.actualOffset(col1And2.size()));
      checkThatTablesAreEqual(col1And2, table);
    }
    for (const auto& block : reader.lazyScan(
             scanSpec,
             CompressedRelationReader::convertBlockMetadataRangesToVector(
                 CompressedRelationReader::getRelevantBlocks(scanSpec, blocks)),
             additionalColumns, cancellationHandle, locatedTriples)) {
      table.insertAtEnd(block);
    }
    checkThatTablesAreEqual(col1And2, table);

    // Check for all distinct combinations of `(col0, col1)` and check that
    // we get the expected result.
    // TODO<joka921>, C++23 use views::chunk_by
    int lastCol1Id = col1And2[0][0];
    std::vector<std::array<int, 1>> col3;

    auto scanAndCheck = [&]() {
      ScanSpecification scanSpec{V(inputs[i].col0_), V(lastCol1Id),
                                 std::nullopt};
      auto size = reader.getResultSizeOfScan(
          ScanSpecAndBlocks{scanSpec, blocks}, locatedTriples);
      IdTable tableWidthOne = reader.scan(ScanSpecAndBlocks{scanSpec, blocks},
                                          Permutation::ColumnIndicesRef{},
                                          cancellationHandle, locatedTriples);
      ASSERT_EQ(tableWidthOne.numColumns(), 1);
      EXPECT_EQ(size, tableWidthOne.numRows());
      checkThatTablesAreEqual(col3, tableWidthOne);
      tableWidthOne.clear();
      for (const auto& block : reader.lazyScan(
               scanSpec,
               CompressedRelationReader::convertBlockMetadataRangesToVector(
                   CompressedRelationReader::getRelevantBlocks(scanSpec,
                                                               blocks)),
               Permutation::ColumnIndices{}, cancellationHandle,
               locatedTriples)) {
        tableWidthOne.insertAtEnd(block);
      }
      checkThatTablesAreEqual(col3, tableWidthOne);
    };
    for (size_t j = 0; j < col1And2.size(); ++j) {
      if (col1And2[j][0] == lastCol1Id) {
        col3.push_back({col1And2[j][1]});
        continue;
      }
      scanAndCheck();
      lastCol1Id = col1And2[j][0];
      col3.clear();
      col3.push_back({col1And2[j][1]});
    }
    // Don't forget the last block.
    scanAndCheck();
  }
}

// Run `testCompressedRelations` (see above) for the given `inputs` and
// `testCaseName`, but with a set of different `blocksizes` (small and medium
// size, powers of two and odd), to find subtle rounding bugs when creating the
// blocks.
void testWithDifferentBlockSizes(const std::vector<RelationInput>& inputs,
                                 std::string testCaseName,
                                 float locatedTriplesProbability = 0.5) {
  testCompressedRelations(inputs, testCaseName, 19_B,
                          locatedTriplesProbability);
  testCompressedRelations(inputs, testCaseName, 237_B,
                          locatedTriplesProbability);
  testCompressedRelations(inputs, testCaseName, 4096_B,
                          locatedTriplesProbability);
}
}  // namespace

// Test for very small relations many of which are stored in the same block.
TEST(CompressedRelationWriter, SmallRelations) {
  std::vector<RelationInput> inputs;
  for (int i = 1; i < 200; ++i) {
    inputs.push_back(
        RelationInput{i, {{i - 1, i + 1}, {i - 1, i + 2}, {i, i - 1}}});
  }
  testWithDifferentBlockSizes(inputs, "smallRelations");
}

// Internal matchers for the following two tests.
namespace {
// A matcher for a `PermutedTriple`. The `int`s are converted to VocabIds.
auto matchPermutedTriple = [](int a, int b, int c) {
  using P = CompressedBlockMetadata::PermutedTriple;
  return AllOf(AD_FIELD(P, col0Id_, V(a)), AD_FIELD(P, col1Id_, V(b)),
               AD_FIELD(P, col2Id_, V(c)));
};

// A matcher for a `FirstAndLastTriple object` where  `(a, b, c)` is the first
// triple, and `(d, e, f)` is the last triple.
auto matchFirstAndLastTriple = [](int a, int b, int c, int d, int e, int f) {
  using F =
      CompressedRelationReader::ScanSpecAndBlocksAndBounds::FirstAndLastTriple;
  return Optional(
      AllOf(AD_FIELD(F, firstTriple_, matchPermutedTriple(a, b, c)),
            AD_FIELD(F, lastTriple_, matchPermutedTriple(d, e, f))));
};
}  // namespace
// _____________________________________________________________________________
TEST(CompressedRelationWriter, getFirstAndLastTriple) {
  using namespace ::testing;
  // Write some triples, and prepare an index
  std::vector<RelationInput> inputs;
  // A dummy graph ID.
  int g2 = 120349;
  for (int i = 1; i < 200; ++i) {
    inputs.push_back(RelationInput{
        i, {{i - 1, i + 1, g2}, {i - 1, i + 2, g2}, {i + 1, i - 1, g2}}});
  }

  auto filename = "getFirstAndLastTriple.dat";
  auto [blocks, metaData, readerPtr] =
      writeAndOpenRelations(inputs, filename, 40_B);
  auto blockMetadata = getBlockMetadataRangesfromVec(blocks);

  // Test that the result of calling `getFirstAndLastTriple` for the index from
  // above with the given `ScanSpecification` matches the given `matcher`.
  using Loc = ad_utility::source_location;
  auto testFirstAndLastBlock = [&](ScanSpecification spec, auto matcher,
                                   const LocatedTriplesPerBlock& =
                                       emptyLocatedTriples,
                                   Loc loc = Loc::current()) {
    auto trace = generateLocationTrace(loc);
    auto firstAndLastTriple = readerPtr->getFirstAndLastTriple(
        {spec, blockMetadata}, emptyLocatedTriples);
    EXPECT_THAT(firstAndLastTriple, matcher);
  };

  // Test for scans with nonempty results with 0, 1, 2, and 3 variables.
  testFirstAndLastBlock({std::nullopt, std::nullopt, std::nullopt},
                        matchFirstAndLastTriple(1, 0, 2, 199, 200, 198));
  testFirstAndLastBlock({V(3), std::nullopt, std::nullopt},
                        matchFirstAndLastTriple(3, 2, 4, 3, 4, 2));
  testFirstAndLastBlock({V(4), V(3), std::nullopt},
                        matchFirstAndLastTriple(4, 3, 5, 4, 3, 6));
  testFirstAndLastBlock({V(5), V(4), V(6)},
                        matchFirstAndLastTriple(5, 4, 6, 5, 4, 6));

  // For this scan there is no matching block.
  testFirstAndLastBlock({V(200), std::nullopt, std::nullopt},
                        ::testing::Eq(std::nullopt));
  // For this scan there is a matching block, but the scan would still be empty.
  testFirstAndLastBlock({V(3), V(3), std::nullopt},
                        ::testing::Eq(std::nullopt));
}

// _____________________________________________________________________________
TEST(CompressedRelationWriter, getFirstAndLastTripleWithUpdates) {
  // A dummy graph ID.

  // Set up a permutation with three triple: (1, 2, 3) (1, 3, 4) (1, 4, 5), all
  // in the same graph. Each triple will be stored in its own block, to make the
  // manual playing with locatedTriples easier.
  int g2 = 120349;
  std::vector<RelationInput> inputs;
  inputs.push_back(RelationInput{1, {{2, 3, g2}, {3, 4, g2}, {4, 5, g2}}});

  auto filename = "getFirstAndLastTriple2.dat";
  auto [blocks, metaData, readerPtr] =
      writeAndOpenRelations(inputs, filename, 0_B);

  // Set up located triples that delete the first triple. This has the
  // consequence that the first triple of the relation `1` actually lies in the
  // second block of that relation, because the first block is completely empty
  // after the update.
  LocatedTriplesPerBlock locatedTriples;
  std::vector<LocatedTriple> deleteTriples;
  deleteTriples.emplace_back(
      LocatedTriple{0, IdTriple{{V(1), V(2), V(3), V(g2)}}, false});
  locatedTriples.setOriginalMetadata(blocks);
  locatedTriples.add(deleteTriples);

  // Test infrastructure.
  using Loc = ad_utility::source_location;
  auto testFirstAndLastBlock = [&](ScanSpecification spec, auto matcher,
                                   Loc loc = Loc::current()) {
    auto trace = generateLocationTrace(loc);
    auto blockMetadata =
        getBlockMetadataRangesfromVec(locatedTriples.getAugmentedMetadata());
    auto firstAndLastTriple =
        readerPtr->getFirstAndLastTriple({spec, blockMetadata}, locatedTriples);
    EXPECT_THAT(firstAndLastTriple, matcher);
  };

  // The first triple has been deleted, so the second and third triple are
  // `first` and `last` respectively.
  testFirstAndLastBlock({V(1), std::nullopt, std::nullopt},
                        matchFirstAndLastTriple(1, 3, 4, 1, 4, 5));

  // Also delete the last triple (1, 4, 5). Now the `first` and `last` triple
  // both are the original middle triple (1, 3, 4), and which are in the same
  // block.
  deleteTriples.clear();
  deleteTriples.emplace_back(
      LocatedTriple{2, IdTriple{{V(1), V(4), V(5), V(g2)}}, false});
  locatedTriples.add(deleteTriples);
  testFirstAndLastBlock({V(1), std::nullopt, std::nullopt},
                        matchFirstAndLastTriple(1, 3, 4, 1, 3, 4));
}

// Test for larger relations that span over several blocks. There are no
// duplicates in the `col1`, so a combination of `(col0, col1)` will be stored
// in a single block.
TEST(CompressedRelationWriter, LargeRelationsDistinctCol1) {
  std::vector<RelationInput> inputs;
  for (int i = 1; i < 6; ++i) {
    std::vector<RowInput> col1And2;
    for (int j = 0; j < 200; ++j) {
      col1And2.push_back({i * j, i * j + 3});
    }
    inputs.push_back(RelationInput{i * 17, std::move(col1And2)});
  }
  testWithDifferentBlockSizes(inputs, "largeRelationsDistinctCol1");
}

// Test for larger relations that span over several blocks. There are many
// duplicates in the `col1`, so a combination of `(col0, col1)` will also be
// stored in several blocks.
TEST(CompressedRelationWriter, LargeRelationsDuplicatesCol1) {
  std::vector<RelationInput> inputs;
  for (int i = 1; i < 6; ++i) {
    std::vector<RowInput> col1And2;
    for (int j = 0; j < 200; ++j) {
      col1And2.push_back({i * 12, i * j + 3});
    }
    inputs.push_back(RelationInput{i * 17, std::move(col1And2)});
  }
  testWithDifferentBlockSizes(inputs, "largeRelationsDuplicatesCol1");
}

// Test a permutation that consists of relations of different sizes and
// characteristics by combining the characteristics of the three test cases
// above.
TEST(CompressedRelationWriter, MixedSizes) {
  std::vector<RelationInput> inputs;
  for (int y = 0; y < 3; ++y) {
    // First some large relations with many duplicates in `col1`.
    for (int i = 1; i < 6; ++i) {
      std::vector<RowInput> col1And2;
      for (int j = 0; j < 50; ++j) {
        col1And2.push_back({i * 12, i * j + 3});
      }
      inputs.push_back(RelationInput{i + (y * 300), std::move(col1And2)});
    }

    // Then some small relations
    for (int i = 9; i < 50; ++i) {
      inputs.push_back(RelationInput{
          i + (y * 300), {{i - 1, i + 1}, {i - 1, i + 2}, {i, i - 1}}});
    }

    // Finally some large relations with few duplicates in `col1`.
    for (int i = 205; i < 221; ++i) {
      std::vector<RowInput> col1And2;
      for (int j = 0; j < 80; ++j) {
        col1And2.push_back({i * j + y, i * j + 3});
      }
      inputs.push_back(RelationInput{i + (y * 300), std::move(col1And2)});
    }
  }
  testWithDifferentBlockSizes(inputs, "mixedSizes");
}

TEST(CompressedRelationWriter, AdditionalColumns) {
  std::vector<RelationInput> inputs;
  for (int y = 0; y < 3; ++y) {
    // First some large relations with many duplicates in `col1`.
    for (int i = 1; i < 6; ++i) {
      std::vector<RowInput> col1And2;
      for (int j = 0; j < 50; ++j) {
        col1And2.push_back({i * 12, i * j + 3});
      }
      inputs.push_back(RelationInput{i + (y * 300), std::move(col1And2)});
    }

    // Then some small relations
    for (int i = 9; i < 50; ++i) {
      inputs.push_back(RelationInput{
          i + (y * 300), {{i - 1, i + 1}, {i - 1, i + 2}, {i, i - 1}}});
    }

    // Finally some large relations with few duplicates in `col1`.
    for (int i = 205; i < 221; ++i) {
      std::vector<RowInput> col1And2;
      for (int j = 0; j < 80; ++j) {
        col1And2.push_back({i * j + y, i * j + 3});
      }
      inputs.push_back(RelationInput{i + (y * 300), std::move(col1And2)});
    }
  }

  // Add two separate columns.
  for (auto& relation : inputs) {
    for (auto& row : relation.col1And2_) {
      row.push_back(row.at(0) + 42);
      row.push_back(row.at(1) * 42);
    }
  }
  // The additional columns don't yet work properly with located triples /
  // SPARQL UPDATE, so we have to disable the
  testWithDifferentBlockSizes(inputs, "mixedSizes", 0.0);
}

TEST(CompressedRelationWriter, MultiplicityCornerCases) {
  ASSERT_EQ(1.0f, CompressedRelationWriter::computeMultiplicity(12, 12));

  constexpr static size_t veryLarge = 1111111111111111;
  constexpr static size_t plusOne = veryLarge + 1;
  ASSERT_EQ(1.0f, static_cast<float>(plusOne) / static_cast<float>(veryLarge));
  ASSERT_NE(1.0f,
            CompressedRelationWriter::computeMultiplicity(plusOne, veryLarge));
}

TEST(CompressedRelationMetadata, GettersAndSetters) {
  CompressedRelationMetadata m;
  m.setCol1Multiplicity(2.0f);
  ASSERT_FLOAT_EQ(2.0f, m.getCol1Multiplicity());
  ASSERT_FLOAT_EQ(2.0f, m.multiplicityCol1_);
  m.setCol2Multiplicity(1.0f);
  ASSERT_FLOAT_EQ(1.0f, m.multiplicityCol2_);
  ASSERT_FLOAT_EQ(1.0f, m.getCol2Multiplicity());
  ASSERT_FALSE(m.isFunctional());
  m.setCol1Multiplicity(1.0f);
  ASSERT_TRUE(m.isFunctional());
  m.numRows_ = 43;
  ASSERT_EQ(43, m.numRows_);
}

TEST(CompressedRelationReader, getBlocksForJoinWithColumn) {
  using SpecBlocksBounds = CompressedRelationReader::ScanSpecAndBlocksAndBounds;
  CompressedBlockMetadata block1{
      {{}, 0, {V(16), V(0), V(0), g}, {V(38), V(4), V(12), g}, {}, false}, 0};
  CompressedBlockMetadata block2{
      {{}, 0, {V(42), V(3), V(0), g}, {V(42), V(4), V(12), g}, {}, false}, 1};
  CompressedBlockMetadata block3{
      {{}, 0, {V(42), V(4), V(13), g}, {V(42), V(6), V(9), g}, {}, false}, 2};

  // We are only interested in blocks with a col0 of `42`.
  CompressedRelationMetadata relation;
  relation.col0Id_ = V(42);
  auto scanSpec =
      ScanSpecification{relation.col0Id_, std::nullopt, std::nullopt};

  std::vector<CompressedBlockMetadata> blocks{block1, block2, block3};
  std::optional<SpecBlocksBounds> metadataAndBlocks;
  metadataAndBlocks.emplace(
      SpecBlocksBounds{{scanSpec, getBlockMetadataRangesfromVec(blocks)},
                       {{V(42), V(3), V(0), g}, {V(42), V(6), V(9), g}}});

  auto test = [&metadataAndBlocks](
                  const std::vector<Id>& joinColumn,
                  const std::vector<CompressedBlockMetadata>& expectedBlocks,
                  size_t numHandledBlocksExpected,
                  source_location l = source_location::current()) {
    auto t = generateLocationTrace(l);
    auto [result, numHandledBlocks] =
        CompressedRelationReader::getBlocksForJoin(joinColumn,
                                                   *metadataAndBlocks);
    EXPECT_THAT(result, ::testing::ElementsAreArray(expectedBlocks));
    EXPECT_EQ(numHandledBlocks, numHandledBlocksExpected);
  };
  // We have fixed the `col0Id` to be 42. The col1/2Ids of the matching blocks
  // are as follows (starting at `block2`)
  // [(3, 0)-(4, 12)], [(4, 13)-(6, 9)]

  // Tests for a fixed col0Id, so the join is on the middle column.
  test({}, {}, 0);
  // All values smaller than the smallest block.
  test({V(1), V(2)}, {}, 0);
  // None of the values matches a block, but the largest value is larger than
  // the largest block.
  test({V(1), V(2), V(7)}, {}, 2);

  // Largest value contained in the first block.
  test({V(3)}, {block2}, 1);

  // Although only `block2` matches, we have also completely handled `block3`
  // since `V(29)` is larger than the largest value in `block3`, we thus have
  // two blocks handled.
  test({V(1), V(3), V(17), V(29)}, {block2}, 2);
  test({V(2), V(3), V(4), V(5)}, {block2, block3}, 2);
  test({V(4)}, {block2, block3}, 2);
  test({V(6)}, {block3}, 2);

  // Test with a fixed col1Id. We now join on the last column, the first column
  // is fixed (42), and the second column is also fixed (4).
  scanSpec.setCol1Id(V(4));
  metadataAndBlocks.emplace(
      SpecBlocksBounds{{scanSpec, getBlockMetadataRangesfromVec(blocks)},
                       {{V(42), V(4), V(11), g}, {V(42), V(4), V(738), g}}});
  test({V(11), V(27), V(30)}, {block2, block3}, 2);
  test({V(12)}, {block2}, 1);
  test({V(13)}, {block3}, 2);

  // Test empty blocks edge case
  metadataAndBlocks.emplace(SpecBlocksBounds{{scanSpec, {}}, {}});
  test({V(1)}, {}, 0);
}

TEST(CompressedRelationReader, getBlocksForJoin) {
  using SpecBlocksBounds = CompressedRelationReader::ScanSpecAndBlocksAndBounds;
  CompressedBlockMetadata block1{
      {{}, 0, {V(16), V(0), V(0), g}, {V(38), V(4), V(12), g}, {}, false}, 0};
  CompressedBlockMetadata block2{
      {{}, 0, {V(42), V(3), V(0), g}, {V(42), V(4), V(12), g}, {}, false}, 1};
  CompressedBlockMetadata block3{
      {{}, 0, {V(42), V(5), V(13), g}, {V(42), V(8), V(9), g}, {}, false}, 2};
  CompressedBlockMetadata block4{
      {{}, 0, {V(42), V(8), V(16), g}, {V(42), V(20), V(9), g}, {}, false}, 3};
  CompressedBlockMetadata block5{
      {{}, 0, {V(42), V(20), V(16), g}, {V(42), V(20), V(63), g}, {}, false},
      4};

  // We are only interested in blocks with a col0 of `42`.
  CompressedRelationMetadata relation;
  relation.col0Id_ = V(42);

  std::vector<CompressedBlockMetadata> blocks{block1, block2, block3, block4,
                                              block5};
  auto bRanges = getBlockMetadataRangesfromVec(blocks);
  auto scanSpec =
      ScanSpecification{relation.col0Id_, std::nullopt, std::nullopt};
  std::optional<SpecBlocksBounds> metadataAndBlocks;
  metadataAndBlocks.emplace(SpecBlocksBounds{
      {scanSpec, bRanges}, {{V(42), V(3), V(0), g}, {V(42), V(20), V(63), g}}});

  CompressedBlockMetadata blockB1{
      {{}, 0, {V(16), V(0), V(0), g}, {V(38), V(4), V(12), g}, {}, false}, 0};
  CompressedBlockMetadata blockB2{
      {{}, 0, {V(47), V(3), V(0), g}, {V(47), V(6), V(12), g}, {}, false}, 1};
  CompressedBlockMetadata blockB3{
      {{}, 0, {V(47), V(7), V(13), g}, {V(47), V(9), V(9), g}, {}, false}, 2};
  CompressedBlockMetadata blockB4{
      {{}, 0, {V(47), V(38), V(7), g}, {V(47), V(38), V(8), g}, {}, false}, 3};
  CompressedBlockMetadata blockB5{
      {{}, 0, {V(47), V(38), V(9), g}, {V(47), V(38), V(12), g}, {}, false}, 4};
  CompressedBlockMetadata blockB6{
      {{}, 0, {V(47), V(38), V(13), g}, {V(47), V(38), V(15), g}, {}, false},
      5};

  // We are only interested in blocks with a col0 of `42`.
  CompressedRelationMetadata relationB;
  relationB.col0Id_ = V(47);

  std::vector<CompressedBlockMetadata> blocksB{blockB1, blockB2, blockB3,
                                               blockB4, blockB5, blockB6};
  auto bRangesB = getBlockMetadataRangesfromVec(blocksB);
  std::optional<SpecBlocksBounds> metadataAndBlocksB;
  auto scanSpecB = ScanSpecification{V(47), std::nullopt, std::nullopt};
  metadataAndBlocksB.emplace(
      SpecBlocksBounds{{scanSpecB, bRangesB},
                       {{V(47), V(3), V(0), g}, {V(47), V(38), V(15), g}}});

  auto test = [&metadataAndBlocks, &metadataAndBlocksB](
                  const std::array<std::vector<CompressedBlockMetadata>, 2>&
                      expectedBlocks,
                  source_location l = source_location::current()) {
    auto t = generateLocationTrace(l);
    auto result = CompressedRelationReader::getBlocksForJoin(
        *metadataAndBlocks, *metadataAndBlocksB);
    EXPECT_THAT(result[0], ::testing::ElementsAreArray(expectedBlocks[0]));
    EXPECT_THAT(result[1], ::testing::ElementsAreArray(expectedBlocks[1]));

    result = CompressedRelationReader::getBlocksForJoin(*metadataAndBlocksB,
                                                        *metadataAndBlocks);
    EXPECT_THAT(result[1], ::testing::ElementsAreArray(expectedBlocks[0]));
    EXPECT_THAT(result[0], ::testing::ElementsAreArray(expectedBlocks[1]));
  };

  // We have fixed the `col0Id` to be 42 for the left input and 47 for the right
  // input. The col1/2Ids of the blocks that have this col0Id are as follows:

  // (starting at `block2`.
  // [(3, 0)- (4, 12)], [(5, 13) - (8, 9)], [(8, 16) - (20, 9)], [(20, 16) -
  // (20, 63)]

  // Starting at `blockB2`.
  // [(3, 0)-(6, 12)], [(7, 13)-(9, 9)], [(38, 7)-(38, 8)], [(38, 9)-(38, 12)],
  // [(38, 13)-(38, 15)]

  // Test for only the `col0Id` fixed.
  test({std::vector{block2, block3, block4}, std::vector{blockB2, blockB3}});
  // Test with a fixed col1Id on both sides. We now join on the last column.
  scanSpec.setCol1Id(V(20));
  metadataAndBlocks.emplace(
      SpecBlocksBounds{{scanSpec, bRanges},
                       {{V(42), V(20), V(5), g}, {V(42), V(20), V(63), g}}});
  scanSpecB.setCol1Id(V(38));
  metadataAndBlocksB.emplace(
      SpecBlocksBounds{{scanSpecB, bRangesB},
                       {{V(47), V(38), V(5), g}, {V(47), V(38), V(15), g}}});
  test({std::vector{block4}, std::vector{blockB4, blockB5}});

  // Fix only the col1Id of the left input.
  scanSpec.setCol1Id(V(4));
  metadataAndBlocks.emplace(SpecBlocksBounds{
      {scanSpec, bRanges}, {{V(42), V(4), V(8), g}, {V(42), V(4), V(12), g}}});
  scanSpecB.setCol1Id(std::nullopt);
  metadataAndBlocksB.emplace(
      SpecBlocksBounds{{scanSpecB, bRangesB},
                       {{V(47), V(38), V(5), g}, {V(47), V(38), V(15), g}}});
  test({std::vector{block2}, std::vector{blockB3}});

  // Fix only the col1Id of the right input.
  scanSpec.setCol1Id(std::nullopt);
  metadataAndBlocks.emplace(SpecBlocksBounds{
      {scanSpec, bRanges}, {{V(42), V(4), V(8), g}, {V(42), V(4), V(12), g}}});
  scanSpecB.setCol1Id(V(7));
  metadataAndBlocksB.emplace(
      SpecBlocksBounds{{scanSpecB, bRangesB},
                       {{V(47), V(7), V(13), g}, {V(47), V(7), V(58), g}}});
  test({std::vector{block4, block5}, std::vector{blockB3}});
}

TEST(CompressedRelationReader, PermutedTripleToString) {
  auto tr =
      CompressedBlockMetadata::PermutedTriple{V(12), V(13), V(27), V(12345)};
  std::stringstream str;
  str << tr;
  ASSERT_EQ(str.str(), "Triple: V:12 V:13 V:27 V:12345\n");
}

TEST(CompressedRelationReader, filterDuplicatesAndGraphs) {
  auto table = makeIdTableFromVector({{3}, {4}, {5}});
  CompressedBlockMetadata metadata{
      {{}, 0, {V(16), V(0), V(0), g}, {V(38), V(4), V(12), g}, {}, false}, 0};
  using Filter = CompressedRelationReader::FilterDuplicatesAndGraphs;
  ScanSpecification::Graphs graphs = std::nullopt;
  Filter f{graphs, 43, false};
  EXPECT_FALSE(f.postprocessBlock(table, metadata));
  EXPECT_THAT(table, matchesIdTableFromVector({{3}, {4}, {5}}));

  table = makeIdTableFromVector({{3}, {3}, {5}});
  metadata.containsDuplicatesWithDifferentGraphs_ = true;
  EXPECT_TRUE(f.postprocessBlock(table, metadata));
  EXPECT_THAT(table, matchesIdTableFromVector({{3}, {5}}));

  // Keep the graph column (the last column), hence there are no duplicates,
  // but keep only the entries from graphs `1` and `2`.
  table = makeIdTableFromVector({{3, 1}, {3, 2}, {5, 3}});
  graphs.emplace();
  graphs->insert(ValueId::makeFromVocabIndex(VocabIndex::make(1)));
  graphs->insert(ValueId::makeFromVocabIndex(VocabIndex::make(2)));
  f = Filter{graphs, 1, false};
  EXPECT_TRUE(f.postprocessBlock(table, metadata));
  EXPECT_THAT(table, matchesIdTableFromVector({{3, 1}, {3, 2}}));

  // The metadata knows that there is only a single block contained, so we don't
  // need to filter anything. We additionally test the deletion of the graph
  // column in this test.
  metadata.graphInfo_.emplace();
  metadata.graphInfo_->push_back(V(1));
  metadata.containsDuplicatesWithDifferentGraphs_ = false;
  f.deleteGraphColumn_ = true;
  table = makeIdTableFromVector({{3, 1}, {4, 1}, {5, 1}});
  EXPECT_FALSE(f.postprocessBlock(table, metadata));
  EXPECT_THAT(table, matchesIdTableFromVector({{3}, {4}, {5}}));
}

TEST(CompressedRelationReader, makeCanBeSkippedForBlock) {
  CompressedBlockMetadata metadata{
      {{}, 0, {V(16), V(0), V(0), g}, {V(38), V(4), V(12), g}, {}, false}, 0};

  auto filter = CompressedRelationReader::FilterDuplicatesAndGraphs{
      std::nullopt, 0, false};
  auto& graphs = filter.desiredGraphs_;
  // No information about the contained blocks, and no graph filter specified,
  // so we cannot skip.
  EXPECT_FALSE(filter.canBlockBeSkipped(metadata));

  // The graph info says that the block only contains the graph `1`, but we
  // don't filter by graphs, so it can't be skipped.
  metadata.graphInfo_.emplace();
  metadata.graphInfo_->push_back(V(1));
  EXPECT_FALSE(filter.canBlockBeSkipped(metadata));

  // The graph info says that the block only contains the graph `1`, and we in
  // fact want the graphs `1` and `3`, so it can't be skipped.
  graphs.emplace();
  graphs->insert(V(1));
  graphs->insert(V(3));
  EXPECT_FALSE(filter.canBlockBeSkipped(metadata));

  // The block contains graph `1`, but we only want graph `3`, so the block can
  // be skipped.
  graphs->erase(V(1));
  EXPECT_TRUE(filter.canBlockBeSkipped(metadata));

  // The block metadata contains no information on the contained graphs, but we
  // only want graph `3`, so the block can't be skipped.
  metadata.graphInfo_.reset();
  EXPECT_FALSE(filter.canBlockBeSkipped(metadata));
}

TEST(CompressedRelationReader, getResultSizeImpl) {
  using ScanSpecAndBlocks = CompressedRelationReader::ScanSpecAndBlocks;
  auto index = ad_utility::testing::makeTestIndex("getResultSizeImpl", "");
  DeltaTriplesManager& deltaTriplesManager = index.deltaTriplesManager();
  deltaTriplesManager.modify<void>([](DeltaTriples& dt) {
    auto handle = std::make_shared<ad_utility::CancellationHandle<>>();
    dt.insertTriples(handle, {IdTriple{{V(0), V(1), V(2), V(3)}},
                              IdTriple{{V(0), V(4), V(5), V(3)}}});
  });
  auto sharedLocatedTriplesSnapshot = deltaTriplesManager.getCurrentSnapshot();
  const auto& locatedTriplesSnapshot = *sharedLocatedTriplesSnapshot;
  auto& impl = index.getImpl();
  auto expectResultSizes = [&impl, &locatedTriplesSnapshot](
                               Permutation::Enum p,
                               const ScanSpecification& scanSpec, size_t lower,
                               size_t upper, size_t exact,
                               ad_utility::source_location sourceLocation =
                                   ad_utility::source_location::current()) {
    auto loc = generateLocationTrace(sourceLocation);
    auto& perm = impl.getPermutation(p);
    auto& reader = perm.reader();
    auto augmentedBlocks =
        perm.getAugmentedMetadataForPermutation(locatedTriplesSnapshot);
    auto& ltpb = locatedTriplesSnapshot.getLocatedTriplesForPermutation(
        perm.permutation());
    auto [actual_lower, actual_upper] = reader.getSizeEstimateForScan(
        ScanSpecAndBlocks{scanSpec, augmentedBlocks}, ltpb);
    EXPECT_THAT(actual_lower, testing::Eq(lower));
    EXPECT_THAT(actual_upper, testing::Eq(upper));
    auto actual_exact = reader.getResultSizeOfScan(
        ScanSpecAndBlocks{scanSpec, augmentedBlocks}, ltpb);
    EXPECT_THAT(actual_exact, testing::Eq(exact));
  };
  // The Scans request all triples of the one and only block.
  for (auto perm : Permutation::ALL) {
    expectResultSizes(perm, {std::nullopt, std::nullopt, std::nullopt}, 0, 2,
                      2);
  }
  expectResultSizes(Permutation::SPO, {V(0), std::nullopt, std::nullopt}, 0, 2,
                    2);
  // Not all triples of the block are requested. The size estimate is truncated
  // by a factor which is a RuntimeParameter.
  expectResultSizes(Permutation::PSO, {V(1), std::nullopt, std::nullopt}, 0, 1,
                    1);
  expectResultSizes(Permutation::PSO, {V(1), V(5), std::nullopt}, 0, 1, 0);
}

// Test the correct setting of the metadata for the contained graphs.
TEST(CompressedRelationWriter, graphInfoInBlockMetadata) {
  std::vector<RelationInput> inputs;
  for (int i = 1;
       static_cast<size_t>(i) < 10 * MAX_NUM_GRAPHS_STORED_IN_BLOCK_METADATA;
       ++i) {
    inputs.push_back(RelationInput{
        i, {{i - 1, i + 1, 42}, {i - 1, i + 2, 43}, {i, i - 1, 43}}});
  }
  using namespace ::testing;
  {
    auto [blocks, metadata, reader] =
        writeAndOpenRelations(inputs, "graphInfo1", 100_MB);
    EXPECT_EQ(blocks.size(), 1);
    EXPECT_FALSE(blocks.at(0).containsDuplicatesWithDifferentGraphs_);
    EXPECT_THAT(blocks.at(0).graphInfo_,
                Optional(UnorderedElementsAre(V(42), V(43))));
  }

  // Now make sure that there are too many different graphs in the block, such
  // that we won't have the graph info in the metadata.
  for (size_t i = 0; i < inputs.size(); ++i) {
    inputs.at(i).col1And2_.at(0).at(2) = i;
  }
  {
    auto [blocks, metadata, reader] =
        writeAndOpenRelations(inputs, "graphInfo1", 100_MB);
    EXPECT_EQ(blocks.size(), 1);
    EXPECT_FALSE(blocks.at(0).containsDuplicatesWithDifferentGraphs_);
    AD_EXPECT_NULLOPT(blocks.at(0).graphInfo_);
  }

  // There is a duplicate triple (3, 1, 3) that appears in both graphs 0 and 1
  inputs.clear();
  inputs.push_back(RelationInput{3, {{1, 2, 0}, {1, 3, 0}, {1, 3, 1}}});

  {
    auto [blocks, metadata, reader] =
        writeAndOpenRelations(inputs, "graphInfo1", 100_MB);
    EXPECT_EQ(blocks.size(), 1);
    EXPECT_TRUE(blocks.at(0).containsDuplicatesWithDifferentGraphs_);
    EXPECT_THAT(blocks.at(0).graphInfo_,
                Optional(UnorderedElementsAre(V(0), V(1))));
  }
}

// Test the correct setting of the metadata for the contained graphs.
TEST(CompressedRelationWriter, scanWithGraphs) {
  using ScanSpecAndBlocks = CompressedRelationReader::ScanSpecAndBlocks;
  std::vector<RelationInput> inputs;
  inputs.push_back(RelationInput{42,
                                 {{3, 4, 0},
                                  {3, 4, 1},
                                  {7, 4, 0},
                                  {8, 4, 0},
                                  {8, 5, 0},
                                  {8, 5, 1},
                                  {9, 4, 1},
                                  {9, 5, 1}}});
  using namespace ::testing;
  for (auto blocksize : std::array{8_B, 16_B, 32_B, 64_B, 128_B}) {
    auto [blocks, metadata, reader] =
        writeAndOpenRelations(inputs, "scanWithGraphs", blocksize);
    ad_utility::HashSet<Id> graphs{V(0)};
    ScanSpecification spec{V(42), std::nullopt, std::nullopt, {}, graphs};
    auto handle = std::make_shared<ad_utility::CancellationHandle<>>();
    auto res = reader->scan(
        ScanSpecAndBlocks{spec, getBlockMetadataRangesfromVec(blocks)}, {},
        handle, emptyLocatedTriples);
    EXPECT_THAT(res,
                matchesIdTableFromVector({{3, 4}, {7, 4}, {8, 4}, {8, 5}}));

    graphs.clear();
    graphs.insert(V(1));
    spec = ScanSpecification{V(42), std::nullopt, std::nullopt, {}, graphs};
    res = reader->scan(
        ScanSpecAndBlocks{spec, getBlockMetadataRangesfromVec(blocks)}, {},
        handle, emptyLocatedTriples);
    EXPECT_THAT(res,
                matchesIdTableFromVector({{3, 4}, {8, 5}, {9, 4}, {9, 5}}));
  }
}
