// Copyright 2011, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Björn Buchhold <buchholb>

#include <gtest/gtest.h>

#include <cstdio>
#include <vector>

#include "index/Vocabulary.h"
#include "index/vocabulary/VocabularyType.h"
#include "util/json.h"

using json = nlohmann::json;
using std::string;

// _____________________________________________________________________________
TEST(VocabularyTest, getIdForWordTest) {
  std::vector<TextVocabulary> vec(2);

  ad_utility::HashSet<std::string> s{"a", "ab", "ba", "car"};
  for (auto& v : vec) {
    auto filename = "vocTest1.dat";
    v.createFromSet(s, filename);
    WordVocabIndex idx;
    ASSERT_TRUE(v.getId("ba", &idx));
    ASSERT_EQ(2u, idx.get());
    ASSERT_TRUE(v.getId("a", &idx));
    ASSERT_EQ(0u, idx.get());
    ASSERT_FALSE(v.getId("foo", &idx));
    ad_utility::deleteFile(filename);
  }

  // with case insensitive ordering
  TextVocabulary voc;
  voc.setLocale("en", "US", false);
  ad_utility::HashSet<string> s2{"a", "A", "Ba", "car"};
  auto filename = "vocTest2.dat";
  voc.createFromSet(s2, filename);
  WordVocabIndex idx;
  ASSERT_TRUE(voc.getId("Ba", &idx));
  ASSERT_EQ(2u, idx.get());
  ASSERT_TRUE(voc.getId("a", &idx));
  ASSERT_EQ(0u, idx.get());
  // getId only gets exact matches;
  ASSERT_FALSE(voc.getId("ba", &idx));
  ad_utility::deleteFile(filename);

  // A TextVocabulary does not return anything for getGeoInfo
  ASSERT_EQ(voc.getGeoInfo(WordVocabIndex::make(0)), std::nullopt);
  ASSERT_EQ(voc.getGeoInfo(WordVocabIndex::make(1)), std::nullopt);
  ASSERT_EQ(voc.getGeoInfo(WordVocabIndex::make(2)), std::nullopt);
}

// _____________________________________________________________________________
TEST(VocabularyTest, getIdRangeForFullTextPrefixTest) {
  TextVocabulary v;
  ad_utility::HashSet<string> s{"wordA0", "wordA1", "wordB2", "wordB3",
                                "wordB4"};
  auto filename = "vocTest3.dat";
  v.createFromSet(s, filename);

  uint64_t word0 = 0;
  // Match exactly one
  auto retVal = v.getIdRangeForFullTextPrefix("wordA1*");
  ASSERT_TRUE(retVal.has_value());
  ASSERT_EQ(word0 + 1, retVal.value().first().get());
  ASSERT_EQ(word0 + 1, retVal.value().last().get());

  // Match all
  retVal = v.getIdRangeForFullTextPrefix("word*");
  ASSERT_TRUE(retVal.has_value());
  ASSERT_EQ(word0, retVal.value().first().get());
  ASSERT_EQ(word0 + 4, retVal.value().last().get());

  // Match first two
  retVal = v.getIdRangeForFullTextPrefix("wordA*");
  ASSERT_TRUE(retVal.has_value());
  ASSERT_EQ(word0, retVal.value().first().get());
  ASSERT_EQ(word0 + 1, retVal.value().last().get());

  // Match last three
  retVal = v.getIdRangeForFullTextPrefix("wordB*");
  ASSERT_TRUE(retVal.has_value());
  ASSERT_EQ(word0 + 2, retVal.value().first().get());
  ASSERT_EQ(word0 + 4, retVal.value().last().get());

  retVal = v.getIdRangeForFullTextPrefix("foo*");
  ASSERT_FALSE(retVal.has_value());
  ad_utility::deleteFile(filename);
}

// _____________________________________________________________________________
TEST(VocabularyTest, createFromSetTest) {
  ad_utility::HashSet<string> s;
  s.insert("a");
  s.insert("ab");
  s.insert("ba");
  s.insert("car");
  TextVocabulary v;
  auto filename = "vocTest4.dat";
  v.createFromSet(s, filename);
  WordVocabIndex idx;
  ASSERT_TRUE(v.getId("ba", &idx));
  ASSERT_EQ(2u, idx.get());
  ASSERT_TRUE(v.getId("a", &idx));
  ASSERT_EQ(0u, idx.get());
  ASSERT_FALSE(v.getId("foo", &idx));
  ad_utility::deleteFile(filename);
}

// _____________________________________________________________________________
TEST(VocabularyTest, IncompleteLiterals) {
  TripleComponentComparator comp("en", "US", false);

  ASSERT_TRUE(comp("\"fieldofwork", "\"GOLD\"@en"));
}

// _____________________________________________________________________________
TEST(Vocabulary, PrefixFilter) {
  RdfsVocabulary vocabulary;
  vocabulary.setLocale("en", "US", true);
  ad_utility::HashSet<string> words;

  words.insert("\"exa\"");
  words.insert("\"exp\"");
  words.insert("\"ext\"");
  words.insert(R"("["Ex-vivo" renal artery revascularization]"@en)");
  auto filename = "vocTest5.dat";
  vocabulary.createFromSet(words, filename);

  // Found in internal but not in external vocabulary.
  auto ranges = vocabulary.prefixRanges("\"exp");
  RdfsVocabulary::PrefixRanges expectedRanges{
      {std::pair{VocabIndex::make(1u), VocabIndex::make(2u)}}};
  ASSERT_EQ(ranges, expectedRanges);
  ad_utility::deleteFile(filename);
}

// _____________________________________________________________________________
TEST(Vocabulary, IsGeoInfoAvailable) {
  using ad_utility::VocabularyType;
  using enum VocabularyType::Enum;

  RdfsVocabulary v1;
  v1.resetToType(VocabularyType{OnDiskCompressed});
  ASSERT_FALSE(v1.isGeoInfoAvailable());

  RdfsVocabulary v2;
  v2.resetToType(VocabularyType{InMemoryUncompressed});
  ASSERT_FALSE(v2.isGeoInfoAvailable());

  RdfsVocabulary v3;
  v3.resetToType(VocabularyType{OnDiskCompressedGeoSplit});
  ASSERT_TRUE(v3.isGeoInfoAvailable());

  TextVocabulary v4;
  ASSERT_FALSE(v4.isGeoInfoAvailable());
}
