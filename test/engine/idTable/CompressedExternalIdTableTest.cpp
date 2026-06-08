// Copyright 2023 - 2025 The QLever Authors, in particular:
//
// 2023 - 2025 Johannes Kalmbach <kalmbach@cs.uni-freiburg.de>, UFR
//
// UFR = University of Freiburg, Chair of Algorithms and Data Structures

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../../util/AllocatorTestHelpers.h"
#include "../../util/GTestHelpers.h"
#include "../../util/IdTableHelpers.h"
#include "engine/idTable/CompressedExternalIdTable.h"
#include "index/ConstantsIndexBuilding.h"
#include "index/ExternalSortFunctors.h"
#include "util/ConstexprUtils.h"

using ad_utility::source_location;
using namespace ad_utility::memory_literals;

namespace {

static constexpr size_t NUM_COLS = NumColumnsIndexBuilding;

// From a `generator` that yields  `IdTable`s, create a single `IdTable` that is
// the concatenation of all the yielded tables.
auto idTableFromBlockGenerator = [](auto& generator) -> CopyableIdTable<0> {
  CopyableIdTable<0> result(ad_utility::testing::makeAllocator());
  for (const auto& blockStatic : generator) {
    auto block = blockStatic.clone().toDynamic();
    if (result.empty()) {
      result.setNumColumns(block.numColumns());
    } else {
      AD_CORRECTNESS_CHECK(result.numColumns() == block.numColumns());
    }
    size_t numColumns = result.numColumns();
    size_t size = result.size();
    result.resize(result.size() + block.size());
    for (auto i : ql::views::iota(0U, numColumns)) {
      decltype(auto) blockCol = block.getColumn(i);
      decltype(auto) resultCol = result.getColumn(i);
      ql::ranges::copy(blockCol, resultCol.begin() + size);
    }
  }
  return result;
};

// From a generator that generates rows of an `IdTable`, create an `IdTable`.
// The number of static and dynamic columns has to be specified (see `IdTable.h`
// for details).
template <size_t NumStaticColumns>
auto idTableFromRowGenerator = [](auto&& generator, size_t numColumns) {
  CopyableIdTable<NumStaticColumns> result(
      numColumns, ad_utility::testing::makeAllocator());
  for (const auto& row : generator) {
    result.push_back(row);
  }
  return result;
};
}  // namespace

TEST(CompressedExternalIdTable, compressedExternalIdTableWriter) {
  using namespace ad_utility::memory_literals;

  auto runTestForBlockSize = [](ad_utility::MemorySize memoryToUse,
                                ad_utility::source_location l =
                                    AD_CURRENT_SOURCE_LOC()) {
    auto trace = generateLocationTrace(l);
    std::string filename = "idTableCompressedWriter.compressedWriterTest.dat";
    ad_utility::CompressedExternalIdTableWriter writer{
        filename, 3, ad_utility::testing::makeAllocator(), memoryToUse};
    std::vector<CopyableIdTable<0>> tables;
    tables.push_back(makeIdTableFromVector({{2, 4, 7}, {3, 6, 8}, {4, 3, 2}}));
    tables.push_back(
        makeIdTableFromVector({{2, 3, 7}, {3, 6, 8}, {4, 2, 123}}));
    tables.push_back(makeIdTableFromVector({{0, 4, 7}}));

    for (const auto& table : tables) {
      writer.writeIdTable(table);
    }

    auto generators = writer.getAllGenerators();
    ASSERT_EQ(generators.size(), tables.size());

    using namespace ::testing;
    std::vector<CopyableIdTable<0>> result;
    auto tr = ql::ranges::transform_view(generators, idTableFromBlockGenerator);
    ql::ranges::copy(tr, std::back_inserter(result));
    EXPECT_THAT(result, ElementsAreArray(tables));
  };
  // With 10 bytes per block, the first and second IdTable are split up into
  // multiple blocks.
  runTestForBlockSize(10_B);
  // With 48 bytes, each IdTable is stored in a single block.
  runTestForBlockSize(48_B);
}

template <size_t NumStaticColumns>
void testExternalSorterImpl(size_t numDynamicColumns, size_t numRows,
                            ad_utility::MemorySize memoryToUse,
                            bool mergeMultipleTimes,
                            source_location l = AD_CURRENT_SOURCE_LOC()) {
  auto tr = generateLocationTrace(l);
  std::string filename = "idTableCompressedSorter.testExternalSorter.dat";
  using namespace ad_utility::memory_literals;

  ad_utility::EXTERNAL_ID_TABLE_SORTER_IGNORE_MEMORY_LIMIT_FOR_TESTING = true;
  ad_utility::CompressedExternalIdTableSorter<SortByOSP, NumStaticColumns>
      writer{filename, numDynamicColumns, memoryToUse,
             ad_utility::testing::makeAllocator(), 5_kB};

  for (size_t i = 0; i < 2; ++i) {
    CopyableIdTable<NumStaticColumns> randomTable =
        createRandomlyFilledIdTable(numRows, numDynamicColumns)
            .toStatic<NumStaticColumns>();

    for (const auto& row : randomTable) {
      writer.push(row);
    }

    ql::ranges::sort(randomTable, SortByOSP{});

    if (mergeMultipleTimes) {
      writer.moveResultOnMerge() = false;
    }

    auto testMultipleTimesImpl = [&](auto k) {
      // Also test the case that the blocksize does not exactly divides the
      // number of inputs.
      auto blocksize = k == 1 ? 1 : 17;
      using namespace ::testing;
      auto generator = [&]() {
        if constexpr (k == 0) {
          // Also check that we don't accidentally get empty blocks yielded,
          // which would be unexpected.
          auto checkNonEmpty = [](auto&& idTable) -> decltype(auto) {
            EXPECT_FALSE(idTable.empty());
            return AD_FWD(idTable);
          };
          return ql::views::join(
              ad_utility::OwningView{writer.getSortedBlocks(blocksize)} |
              ql::views::transform(checkNonEmpty));
        } else {
          return writer.sortedView();
        }
      };
      if (mergeMultipleTimes || k == 0) {
        auto result = idTableFromRowGenerator<NumStaticColumns>(
            generator(), numDynamicColumns);
        ASSERT_THAT(result, ::testing::ElementsAreArray(randomTable))
            << "k = " << k;
      } else {
        EXPECT_ANY_THROW((idTableFromRowGenerator<NumStaticColumns>(
            generator(), numDynamicColumns)));
      }
      // We cannot access or change this value after the first merge.
      EXPECT_ANY_THROW(writer.moveResultOnMerge());
    };
    ad_utility::ConstexprForLoopVi(std::make_index_sequence<5>(),
                                   testMultipleTimesImpl);
    writer.clear();
  }
};

template <size_t NumStaticColumns>
void testExternalSorter(size_t numDynamicColumns, size_t numRows,
                        ad_utility::MemorySize memoryToUse,
                        source_location l = AD_CURRENT_SOURCE_LOC()) {
  testExternalSorterImpl<NumStaticColumns>(numDynamicColumns, numRows,
                                           memoryToUse, true, l);
  testExternalSorterImpl<NumStaticColumns>(numDynamicColumns, numRows,
                                           memoryToUse, false, l);
}

// Test for static (`<NUM_COLS>) and dynamic (`<0>`) tables. The second
// argument to `testExternalSorter` is the number of rows, the third argument
// is the memory limit for the sorter.
TEST(CompressedExternalIdTable, sorterRandomInputs) {
  using namespace ad_utility::memory_literals;
  testExternalSorter<NUM_COLS>(NUM_COLS, 10'000, 10_kB);
  testExternalSorter<NUM_COLS>(NUM_COLS, 1000, 1_MB);
  testExternalSorter<NUM_COLS>(NUM_COLS, 0, 1_MB);

  testExternalSorter<0>(NUM_COLS, 10'000, 10_kB);
  testExternalSorter<0>(NUM_COLS, 1000, 1_MB);
  testExternalSorter<0>(NUM_COLS, 0, 1_MB);
}

// Test that destroying the sorter while an async block-sorting task is still
// running does not cause a use-after-free (caught by ASAN). This used to be a
// bug, which was fixed by calling `waitForFuture()` in the destructor of
// `CompressedExternalIdTableSorter`.
TEST(CompressedExternalIdTable, stillSortingOnDestruction) {
  struct SlowDummySorter : SortByOSP {
    std::vector<size_t> data_ = {1, 2, 3};
    std::shared_ptr<std::atomic<bool>> sleptOnce_ =
        std::make_shared<std::atomic<bool>>(false);
    SlowDummySorter() = default;
    SlowDummySorter(const SlowDummySorter& other)
        : SortByOSP(other), data_(other.data_), sleptOnce_(other.sleptOnce_) {
      if (!sleptOnce_->exchange(true)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      [[maybe_unused]] volatile auto x = other.data_[0];
    }
  };
  ad_utility::EXTERNAL_ID_TABLE_SORTER_IGNORE_MEMORY_LIMIT_FOR_TESTING = true;
  ad_utility::CompressedExternalIdTableSorter<SlowDummySorter, 0> sorter{
      "stillSortingOnDestruction.dat", NUM_COLS, 10_kB,
      ad_utility::testing::makeAllocator()};
  // With 10 kB memory and NUM_COLS (4) columns, blocksize = 10000 / (4*8*2)
  // = 156 rows. Push enough to trigger exactly one `pushBlock`.
  auto table = createRandomlyFilledIdTable(200, NUM_COLS);
  for (const auto& row : table) {
    sorter.push(row);
  }
}

TEST(CompressedExternalIdTable, sorterMemoryLimit) {
  std::string filename = "idTableCompressedSorter.memoryLimit.dat";

  // only 100 bytes of memory, not sufficient for merging
  ad_utility::EXTERNAL_ID_TABLE_SORTER_IGNORE_MEMORY_LIMIT_FOR_TESTING = false;
  ad_utility::CompressedExternalIdTableSorter<SortByOSP, 0> writer{
      filename, NUM_COLS, 100_B, ad_utility::testing::makeAllocator()};

  CopyableIdTable<0> randomTable = createRandomlyFilledIdTable(100, NUM_COLS);

  // Pushing always works
  for (const auto& row : randomTable) {
    writer.push(row);
  }

  auto generator = [&writer]() { return writer.sortedView(); };
  AD_EXPECT_THROW_WITH_MESSAGE(
      (idTableFromRowGenerator<0>(generator(), NUM_COLS)),
      ::testing::ContainsRegex("Insufficient memory"));
}

// While a generator obtained from a `CompressedExternalIdTableWriter` has not
// been fully consumed, the writer is considered to be "iterated over", so
// calling `writeIdTable` or `clear` must throw. Once all generators have been
// fully consumed, both work again. This used to be covered (transitively, via
// `getRows()`) by the now-removed `CompressedExternalIdTable` class.
TEST(CompressedExternalIdTable, exceptionsWhenWritingWhileIterating) {
  std::string filename = "idTableCompressor.exceptionsWhenWritingTest.dat";
  using namespace ad_utility::memory_literals;

  // A small block size so that the written table is split into multiple blocks.
  ad_utility::CompressedExternalIdTableWriter writer{
      filename, 3, ad_utility::testing::makeAllocator(), 10_B};

  auto table = makeIdTableFromVector({{2, 4, 7}, {3, 6, 8}, {4, 3, 2}});
  writer.writeIdTable(table);

  // Obtaining the generators already marks the writer as being iterated over,
  // even though iteration has not started yet.
  auto generators = writer.getAllGenerators();
  ASSERT_EQ(generators.size(), 1);

  AD_EXPECT_THROW_WITH_MESSAGE(
      writer.writeIdTable(table),
      ::testing::ContainsRegex("currently being iterated"));
  AD_EXPECT_THROW_WITH_MESSAGE(
      writer.clear(), ::testing::ContainsRegex("currently being iterated"));

  // Starting, but not finishing the iteration also keeps the writer locked.
  auto& generator = generators.at(0);
  auto it = generator.begin();
  AD_EXPECT_THROW_WITH_MESSAGE(
      writer.writeIdTable(table),
      ::testing::ContainsRegex("currently being iterated"));
  AD_EXPECT_THROW_WITH_MESSAGE(
      writer.clear(), ::testing::ContainsRegex("currently being iterated"));

  // Fully consume the generator.
  for (; it != generator.end(); ++it) {
  }

  // All generators have been fully consumed, so writing and clearing work
  // again.
  ASSERT_NO_THROW(writer.writeIdTable(table));
  ASSERT_NO_THROW(writer.clear());
}

TEST(CompressedExternalIdTable, WrongNumberOfColsWhenPushing) {
  std::string filename = "idTableCompressor.wrongNumCols.dat";
  using namespace ad_utility::memory_literals;
  auto alloc = ad_utility::testing::makeAllocator();

  ad_utility::CompressedExternalIdTableSorter<SortByOSP, NUM_COLS> writer{
      filename, NUM_COLS, 10_B, alloc};
  ad_utility::CompressedExternalIdTableSorterTypeErased& erased = writer;
  IdTableStatic<0> t1{NUM_COLS, alloc};
  EXPECT_NO_THROW(erased.pushBlock(t1));
  EXPECT_NO_THROW(t1.setNumColumns(NUM_COLS + 1));
  EXPECT_ANY_THROW(erased.pushBlock(t1));
}
