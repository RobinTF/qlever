//  Copyright 2023, University of Freiburg,
//                  Chair of Algorithms and Data Structures.
//  Author: Nick Göckel <nick.goeckel@students.uni-freiburg.de>

#include "engine/TextIndexScanForWord.h"

// _____________________________________________________________________________
TextIndexScanForWord::TextIndexScanForWord(QueryExecutionContext* qec,
                                           Variable textRecordVar, string word)
    : Operation(qec),
      textRecordVar_(std::move(textRecordVar)),
      word_(std::move(word)),
      isPrefix_(word_.ends_with('*')) {}

// _____________________________________________________________________________
ProtoResult TextIndexScanForWord::computeResult(
    [[maybe_unused]] bool requestLaziness) {
  IdTable idTable = getExecutionContext()->getIndex().getWordPostingsForTerm(
      word_, getExecutionContext()->getAllocator());

  // This filters out the word column. When the searchword is a prefix this
  // column shows the word the prefix got extended to
  if (!isPrefix_) {
    using CI = ColumnIndex;
    idTable.setColumnSubset(std::array{CI{0}, CI{2}});
    return {std::move(idTable), resultSortedOn(), LocalVocab{}};
  }

  // Add details to the runtimeInfo. This is has no effect on the result.
  runtimeInfo().addDetail("word: ", word_);

  return {std::move(idTable), resultSortedOn(), LocalVocab{}};
}

// _____________________________________________________________________________
VariableToColumnMap TextIndexScanForWord::computeVariableToColumnMap() const {
  VariableToColumnMap vcmap;
  auto addDefinedVar = [&vcmap,
                        index = ColumnIndex{0}](const Variable& var) mutable {
    vcmap[var] = makeAlwaysDefinedColumn(index);
    ++index;
  };
  addDefinedVar(textRecordVar_);
  if (isPrefix_) {
    addDefinedVar(textRecordVar_.getMatchingWordVariable(
        std::string_view(word_).substr(0, word_.size() - 1)));
  }
  addDefinedVar(textRecordVar_.getWordScoreVariable(word_, isPrefix_));
  return vcmap;
}

// _____________________________________________________________________________
size_t TextIndexScanForWord::getResultWidth() const {
  return 2 + (isPrefix_ ? 1 : 0);
}

// _____________________________________________________________________________
size_t TextIndexScanForWord::getCostEstimate() {
  return getExecutionContext()->getIndex().getSizeOfTextBlockForWord(word_);
}

// _____________________________________________________________________________
uint64_t TextIndexScanForWord::getSizeEstimateBeforeLimit() {
  return getExecutionContext()->getIndex().getSizeOfTextBlockForWord(word_);
}

// _____________________________________________________________________________
vector<ColumnIndex> TextIndexScanForWord::resultSortedOn() const {
  return {ColumnIndex(0)};
}

// _____________________________________________________________________________
string TextIndexScanForWord::getDescriptor() const {
  return absl::StrCat("TextIndexScanForWord on ", textRecordVar_.name());
}

// _____________________________________________________________________________
string TextIndexScanForWord::getCacheKeyImpl() const {
  std::ostringstream os;
  os << "WORD INDEX SCAN: "
     << " with word: \"" << word_ << "\"";
  return std::move(os).str();
}

// _____________________________________________________________________________
std::unique_ptr<Operation> TextIndexScanForWord::cloneImpl() const {
  return std::make_unique<TextIndexScanForWord>(*this);
}
