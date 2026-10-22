//   Copyright 2025, University of Freiburg,
//   Chair of Algorithms and Data Structures.
//   Author: Robin Textor-Falconi <textorr@informatik.uni-freiburg.de>

#include <gmock/gmock.h>

#include "../SparqlExpressionTestHelpers.h"
#include "../util/GTestHelpers.h"
#include "../util/IndexTestHelpers.h"
#include "../util/ParsedQueryTestHelpers.h"
#include "engine/QueryPlanner.h"
#include "engine/sparqlExpressions/BlankNodeExpression.h"
#include "engine/sparqlExpressions/LiteralExpression.h"
#include "engine/sparqlExpressions/SampleExpression.h"

using namespace sparqlExpression;
using namespace ad_utility::triple_component;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Ne;
using ::testing::VariantWith;

namespace {
// Return `BNODE("label")`.
SparqlExpression::Ptr makeLabeledBlankNode(std::string_view label) {
  return makeBlankNodeExpression(std::make_unique<StringLiteralExpression>(
      ad_utility::triple_component::Literal::literalWithoutQuotes(label)));
}

// Return the result of the evaluation of `expression` as a vector of `Id`s.
std::vector<Id> evaluateToIds(SparqlExpression& expression,
                              EvaluationContext& context) {
  auto result = expression.evaluate(&context);
  if (std::holds_alternative<Id>(result)) {
    return {std::get<Id>(result)};
  }
  const auto& ids = std::get<VectorWithMemoryLimit<Id>>(result);
  return {ids.begin(), ids.end()};
}

// Matchers for the `Id` of a blank node and for an undefined `Id`.
auto isBlankNode() {
  return AD_PROPERTY(Id, getDatatype, Eq(Datatype::BlankNodeIndex));
}
auto isUndefined() { return Eq(Id::makeUndefined()); }

// Expect that all the `ids` are blank nodes that are pairwise distinct.
void expectDistinctBlankNodes(
    const std::vector<Id>& ids,
    ad_utility::source_location loc = AD_CURRENT_SOURCE_LOC()) {
  auto t = generateLocationTrace(loc);
  EXPECT_THAT(ids, Each(isBlankNode()));
  EXPECT_EQ(ad_utility::HashSet<Id>(ids.begin(), ids.end()).size(), ids.size());
}
}  // namespace

// _____________________________________________________________________________
TEST(BlankNodeExpression, labeledBlankNodesAreNewBlankNodes) {
  TestContext context;
  auto expression1 = makeLabeledBlankNode("Test");
  auto expression2 = makeLabeledBlankNode("Test");
  auto result1 = evaluateToIds(*expression1, context.context);
  auto result2 = evaluateToIds(*expression2, context.context);
  ASSERT_EQ(result1.size(), context.context.size());
  // Without sharing, each call creates a new blank node, also for the same
  // label.
  auto all = result1;
  ql::ranges::copy(result2, std::back_inserter(all));
  expectDistinctBlankNodes(all);

  // An invalid constant label leads to a single undefined value.
  auto expression0 = makeBlankNodeExpression(
      std::make_unique<IdExpression>(Id::makeUndefined()));
  EXPECT_THAT(expression0->evaluate(&context.context),
              VariantWith<Id>(isUndefined()));
}

// _____________________________________________________________________________
TEST(BlankNodeExpression, onlySimpleLiteralsAreValidLabels) {
  TestContext context;
  const auto& localVocabContext = context.qec->getLocalVocabContext();
  VectorWithMemoryLimit<IdOrLocalVocabEntry> vector{context.context._allocator};
  vector.emplace_back(
      LocalVocabEntry::literalWithoutQuotes("label", localVocabContext));
  vector.emplace_back(
      LocalVocabEntry::fromIriref("<http://example.com>", localVocabContext));
  vector.emplace_back(LocalVocabEntry::fromStringRepresentation(
      "\"label\"@en", localVocabContext));
  context.table.resize(vector.size());
  context.context._endIndex = vector.size();
  auto expression =
      makeBlankNodeExpression(std::make_unique<SingleUseExpression>(
          ExpressionResult{std::move(vector)}));
  EXPECT_THAT(evaluateToIds(*expression, context.context),
              ElementsAre(isBlankNode(), isUndefined(), isUndefined()));

  // Numbers are also invalid.
  auto number = makeBlankNodeExpression(
      std::make_unique<IdExpression>(Id::makeFromInt(42)));
  EXPECT_THAT(number->evaluate(&context.context),
              VariantWith<Id>(isUndefined()));
}

// _____________________________________________________________________________
TEST(BlankNodeExpression, undefinedLabels) {
  TestContext context;
  const auto& localVocabContext = context.qec->getLocalVocabContext();
  VectorWithMemoryLimit<IdOrLocalVocabEntry> vector{context.context._allocator};
  vector.emplace_back(
      LocalVocabEntry::literalWithoutQuotes("T1", localVocabContext));
  vector.emplace_back(Id::makeUndefined());
  vector.emplace_back(
      LocalVocabEntry::literalWithoutQuotes("T2", localVocabContext));
  auto expression =
      makeBlankNodeExpression(std::make_unique<SingleUseExpression>(
          ExpressionResult{std::move(vector)}));
  EXPECT_THAT(evaluateToIds(*expression, context.context),
              ElementsAre(isBlankNode(), isUndefined(), isBlankNode()));
}

// Calls that share their blank nodes within an evaluation return the same blank
// node for the same label and row.
TEST(BlankNodeExpression, shareWithinEvaluation) {
  TestContext context;
  auto a1 = makeLabeledBlankNode("a");
  auto a2 = makeLabeledBlankNode("a");
  auto b = makeLabeledBlankNode("b");
  for (auto* expression : {a1.get(), a2.get(), b.get()}) {
    shareLabeledBlankNodesWithinEvaluation(*expression);
  }
  auto resultA1 = evaluateToIds(*a1, context.context);
  auto resultA2 = evaluateToIds(*a2, context.context);
  auto resultB = evaluateToIds(*b, context.context);
  EXPECT_EQ(resultA1, resultA2);
  auto all = resultA1;
  ql::ranges::copy(resultB, std::back_inserter(all));
  expectDistinctBlankNodes(all);

  // A different evaluation context creates new blank nodes.
  TestContext otherContext;
  auto otherResult = evaluateToIds(*a1, otherContext.context);
  ql::ranges::copy(otherResult, std::back_inserter(all));
  expectDistinctBlankNodes(all);
}

// Calls that share their blank nodes via a solution token return the same blank
// node for the same label and token, also when they are evaluated with
// different `LocalVocab`s, as long as one is merged into the other.
TEST(BlankNodeExpression, shareViaSolutionToken) {
  TestContext context;
  // The `?ints` column has a distinct value in each row, so it can serve as a
  // solution token.
  Variable token{"?ints"};
  auto makeExpression = [&token](std::string_view label) {
    auto expression = makeLabeledBlankNode(label);
    shareLabeledBlankNodesViaSolutionToken(*expression, token);
    // Setting the token again doesn't add another child.
    shareLabeledBlankNodesViaSolutionToken(*expression, token);
    EXPECT_EQ(expression->children().size(), 2);
    return expression;
  };
  auto evaluateWithVocab = [&context](SparqlExpression& expression,
                                      LocalVocab& localVocab) {
    EvaluationContext evaluationContext{
        *context.qec,
        context.varToColMap,
        context.table.asStaticView<0>(),
        context.qec->getAllocator(),
        localVocab,
        std::make_shared<ad_utility::CancellationHandle<>>(),
        EvaluationContext::TimePoint::max()};
    return evaluateToIds(expression, evaluationContext);
  };

  LocalVocab vocab1;
  auto a1 = evaluateWithVocab(*makeExpression("a"), vocab1);
  auto b1 = evaluateWithVocab(*makeExpression("b"), vocab1);
  LocalVocab vocab2 = vocab1.clone();
  auto a2 = evaluateWithVocab(*makeExpression("a"), vocab2);
  EXPECT_EQ(a1, a2);
  auto all = a1;
  ql::ranges::copy(b1, std::back_inserter(all));
  expectDistinctBlankNodes(all);

  // Without merging, the blank nodes are new.
  LocalVocab vocab3;
  auto a3 = evaluateWithVocab(*makeExpression("a"), vocab3);
  ql::ranges::copy(a3, std::back_inserter(all));
  expectDistinctBlankNodes(all);
}

// _____________________________________________________________________________
TEST(BlankNodeExpression, groupByReturnsSingleBlankNodePerGroup) {
  TestContext context;
  context.context._isPartOfGroupBy = true;

  // `BNODE()` returns a single, constant blank node `Id`.
  EXPECT_THAT(makeUniqueBlankNodeExpression()->evaluate(&context.context),
              VariantWith<Id>(isBlankNode()));

  // `BNODE(<constant>)` returns a single, constant blank node.
  auto a1 = makeLabeledBlankNode("a");
  auto a2 = makeLabeledBlankNode("a");
  auto b = makeLabeledBlankNode("b");
  auto unshared = makeLabeledBlankNode("a");
  for (auto* expression : {a1.get(), a2.get(), b.get()}) {
    shareLabeledBlankNodesWithinEvaluation(*expression);
  }
  auto resultA1 = evaluateToIds(*a1, context.context);
  ASSERT_EQ(resultA1.size(), 1);
  EXPECT_EQ(resultA1, evaluateToIds(*a2, context.context));
  auto all = resultA1;
  ql::ranges::copy(evaluateToIds(*b, context.context), std::back_inserter(all));
  ql::ranges::copy(evaluateToIds(*unshared, context.context),
                   std::back_inserter(all));
  // The next group gets new blank nodes.
  context.context._beginIndex = 1;
  ql::ranges::copy(evaluateToIds(*a1, context.context),
                   std::back_inserter(all));
  expectDistinctBlankNodes(all);
}

// _____________________________________________________________________________
TEST(BlankNodeExpression, insideAggregateReturnsOnePerRow) {
  TestContext context;
  context.context._isPartOfGroupBy = true;
  const size_t numRows = context.context.size();

  // `BNODE()` (no arguments), wrapped in an aggregate.
  {
    auto aggregate = std::make_unique<SampleExpression>(
        false, makeUniqueBlankNodeExpression());
    auto* blankNode = aggregate->children()[0].get();
    ASSERT_TRUE(blankNode->isInsideAggregate());
    auto result = evaluateToIds(*blankNode, context.context);
    EXPECT_EQ(result.size(), numRows);
    expectDistinctBlankNodes(result);
  }
  // `BNODE(<constant>)`, wrapped in two aggregates, shares the blank nodes per
  // row, but not with the same call outside of an aggregate.
  {
    auto aggregate1 =
        std::make_unique<SampleExpression>(false, makeLabeledBlankNode("a"));
    auto aggregate2 =
        std::make_unique<SampleExpression>(false, makeLabeledBlankNode("a"));
    auto outside = makeLabeledBlankNode("a");
    for (SparqlExpression* expression :
         {static_cast<SparqlExpression*>(aggregate1.get()),
          static_cast<SparqlExpression*>(aggregate2.get()), outside.get()}) {
      shareLabeledBlankNodesWithinEvaluation(*expression);
    }
    auto* blankNode1 = aggregate1->children()[0].get();
    auto* blankNode2 = aggregate2->children()[0].get();
    ASSERT_TRUE(blankNode1->isInsideAggregate());
    auto result1 = evaluateToIds(*blankNode1, context.context);
    EXPECT_EQ(result1.size(), numRows);
    EXPECT_EQ(result1, evaluateToIds(*blankNode2, context.context));
    auto all = result1;
    ql::ranges::copy(evaluateToIds(*outside, context.context),
                     std::back_inserter(all));
    expectDistinctBlankNodes(all);
  }
}

// _____________________________________________________________________________
TEST(BlankNodeExpression, uniqueCacheKey) {
  auto expression0 = makeUniqueBlankNodeExpression();
  auto expression1 = makeUniqueBlankNodeExpression();
  EXPECT_EQ(expression0->getCacheKey({}), expression1->getCacheKey({}));
  EXPECT_NE(expression1->getCacheKey({}), expression1->getCacheKey({}));

  auto expression2 = makeLabeledBlankNode("Test");
  EXPECT_THAT(
      expression2->getCacheKey({}),
      ::testing::HasSubstr(expression2->children()[0]->getCacheKey({})));
  EXPECT_NE(expression1->getCacheKey({}), expression2->getCacheKey({}));
  EXPECT_NE(expression2->getCacheKey({}), expression2->getCacheKey({}));

  // The solution token is also part of the cache key.
  VariableToColumnMap map;
  map[Variable{"?token"}] = makeAlwaysDefinedColumn(0);
  shareLabeledBlankNodesViaSolutionToken(*expression2, Variable{"?token"});
  EXPECT_THAT(
      expression2->getCacheKey(map),
      ::testing::HasSubstr(expression2->children()[1]->getCacheKey(map)));
}

// _____________________________________________________________________________
TEST(BlankNodeExpression, correctChildren) {
  auto expression0 = makeUniqueBlankNodeExpression();
  EXPECT_TRUE(expression0->children().empty());
  auto expression1 = makeLabeledBlankNode("Test");
  EXPECT_EQ(expression1->children().size(), 1);
  shareLabeledBlankNodesViaSolutionToken(*expression1, Variable{"?token"});
  EXPECT_EQ(expression1->children().size(), 2);
  EXPECT_THAT(expression1->containedVariables(),
              ElementsAre(::testing::Pointee(Variable{"?token"})));
}

// _____________________________________________________________________________
TEST(BlankNodeExpression, uniqueValuesAcrossInstances) {
  TestContext context;
  auto expression0 = makeUniqueBlankNodeExpression();
  auto expression1 = makeUniqueBlankNodeExpression();
  auto result = evaluateToIds(*expression0, context.context);
  ql::ranges::copy(evaluateToIds(*expression1, context.context),
                   std::back_inserter(result));
  EXPECT_EQ(result.size(), 2 * context.context.size());
  expectDistinctBlankNodes(result);
}

// _____________________________________________________________________________
TEST(BlankNodeExpression, isResultAlwaysDefined) {
  EXPECT_FALSE(makeBlankNodeExpression(
                   std::make_unique<IdExpression>(Id::makeFromInt(42)))
                   ->isResultAlwaysDefined({}));

  EXPECT_TRUE(makeUniqueBlankNodeExpression()->isResultAlwaysDefined({}));
}

// _____________________________________________________________________________
TEST(BlankNodeExpression, helperFunctions) {
  auto withoutArgument = makeUniqueBlankNodeExpression();
  auto withArgument = makeLabeledBlankNode("a");
  EXPECT_TRUE(isBlankNodeExpressionWithoutArgument(*withoutArgument));
  EXPECT_FALSE(isBlankNodeExpressionWithoutArgument(*withArgument));
  EXPECT_FALSE(
      isBlankNodeExpressionWithoutArgument(IdExpression{Id::makeFromInt(42)}));

  EXPECT_EQ(countLabeledBlankNodeExpressions(*withoutArgument), 0);
  EXPECT_EQ(countLabeledBlankNodeExpressions(*withArgument), 1);
  // `BNODE(BNODE("a"))` contains two calls with an argument.
  auto nested = makeBlankNodeExpression(std::move(withArgument));
  EXPECT_EQ(countLabeledBlankNodeExpressions(*nested), 2);
}

namespace {
// Return the `Bind`s of the root graph pattern of the parsed `query`.
std::vector<parsedQuery::Bind> getRootBinds(const ParsedQuery& query) {
  std::vector<parsedQuery::Bind> binds;
  for (const auto& operation : query._rootGraphPattern._graphPatterns) {
    if (auto* bind = std::get_if<parsedQuery::Bind>(&operation)) {
      binds.push_back(*bind);
    }
  }
  return binds;
}

// Return true iff the `bind` is a `BIND(BNODE() AS ?token)` with an internal
// variable.
bool isSolutionTokenBind(const parsedQuery::Bind& bind) {
  return isBlankNodeExpressionWithoutArgument(*bind._expression.getPimpl()) &&
         ql::starts_with(bind._target.name(), QLEVER_INTERNAL_VARIABLE_PREFIX);
}
}  // namespace

// Test that the parser inserts a `BIND` for the solution token exactly when
// several consecutive `BIND`s contain `BNODE` with an argument.
TEST(BlankNodeExpression, parserInsertsSolutionTokens) {
  using ad_utility::testing::parseQuery;
  auto expectNumBinds =
      [](const std::string& query, size_t numBinds, bool hasToken,
         ad_utility::source_location loc = AD_CURRENT_SOURCE_LOC()) {
        auto t = generateLocationTrace(loc);
        auto binds = getRootBinds(parseQuery(query));
        ASSERT_EQ(binds.size(), numBinds);
        EXPECT_EQ(ql::ranges::count_if(binds, &isSolutionTokenBind),
                  hasToken ? 1 : 0);
      };
  // Consecutive `BIND`s, also with a `FILTER` in between.
  expectNumBinds(
      "SELECT * { BIND(BNODE(\"a\") AS ?x) FILTER(true) "
      "BIND(BNODE(\"a\") AS ?y) }",
      3, true);
  // A single `BIND` (the calls within it share their blank nodes via the
  // `EvaluationContext`).
  expectNumBinds("SELECT * { BIND(BNODE(\"a\") = BNODE(\"a\") AS ?x) }", 1,
                 false);
  // A `BIND` and a `SELECT` expression.
  expectNumBinds("SELECT (BNODE(\"a\") AS ?y) { BIND(BNODE(\"a\") AS ?x) }", 3,
                 true);
  // Two `SELECT` expressions.
  expectNumBinds("SELECT (BNODE(\"a\") AS ?x) (BNODE(\"b\") AS ?y) {}", 3,
                 true);
  // `BIND`s that are separated by a triple are not consecutive.
  expectNumBinds(
      "SELECT * { BIND(BNODE(\"a\") AS ?x) ?s ?p ?o BIND(BNODE(\"a\") AS ?y) "
      "}",
      2, false);
  // `BIND`s without `BNODE` with an argument.
  expectNumBinds("SELECT * { BIND(BNODE() AS ?x) BIND(3 AS ?y) }", 2, false);

  // An existing `BIND(BNODE() AS ?t)` is reused as the solution token.
  auto query = parseQuery(
      "SELECT * { BIND(BNODE() AS ?t) BIND(BNODE(\"a\") AS ?x) "
      "BIND(BNODE(\"a\") AS ?y) }");
  auto binds = getRootBinds(query);
  ASSERT_EQ(binds.size(), 3);
  for (size_t i : {1, 2}) {
    EXPECT_THAT(binds.at(i)._expression.containedVariables(),
                ::testing::Contains(::testing::Pointee(Variable{"?t"})));
  }

  // Calling `shareLabeledBlankNodes` again doesn't insert another token.
  size_t numInternalVariables = 0;
  query._rootGraphPattern.shareLabeledBlankNodes([&numInternalVariables]() {
    return Variable{absl::StrCat("?internal", numInternalVariables++)};
  });
  EXPECT_EQ(getRootBinds(query).size(), 3);
  EXPECT_EQ(numInternalVariables, 0);
}

namespace {
// Run the `query` on the `kg` and return the values of the `variables` for
// each row of the result.
std::vector<std::vector<Id>> runQuery(
    const std::string& kg, const std::string& query,
    const std::vector<std::string>& variables) {
  auto qec = ad_utility::testing::getQec(kg);
  qec->clearCacheUnpinnedOnly();
  QueryPlanner queryPlanner{
      qec, std::make_shared<ad_utility::CancellationHandle<>>()};
  auto parsedQuery = ad_utility::testing::parseQuery(query);
  auto tree = queryPlanner.createExecutionTree(parsedQuery);
  auto result = tree->getResult();
  auto table = result->idTableView();
  std::vector<std::vector<Id>> rows(table.numRows());
  for (const auto& variable : variables) {
    auto column = tree->getVariableColumn(Variable{variable});
    for (size_t i = 0; i < table.numRows(); ++i) {
      rows.at(i).push_back(table(i, column));
    }
  }
  return rows;
}

// Return the `i`-th column of the `rows`.
std::vector<Id> getColumn(const std::vector<std::vector<Id>>& rows, size_t i) {
  return ad_utility::transform(rows,
                               [i](const auto& row) { return row.at(i); });
}

const std::string kg =
    "<s1> <str> \"foo\" . <s2> <str> \"bar\" . <s1> <p> <o1> . <s1> <p> <o2> .";
}  // namespace

// Test the semantics of `BNODE` with an argument in complete queries, see
// `BlankNodeExpression.cpp` for details.
TEST(BlankNodeExpression, queries) {
  auto trueId = Id::makeFromBool(true);

  // The original bug: `BNODE` with an argument returns a blank node.
  {
    auto rows = runQuery(kg,
                         "SELECT ?b ?isBlank ?isIri { "
                         "BIND(BNODE(\"abc\") AS ?b) "
                         "BIND(ISBLANK(?b) AS ?isBlank) "
                         "BIND(ISIRI(?b) AS ?isIri) }",
                         {"?b", "?isBlank", "?isIri"});
    EXPECT_THAT(rows, ElementsAre(ElementsAre(isBlankNode(), Eq(trueId),
                                              Eq(Id::makeFromBool(false)))));
  }

  // Test `bnode01` of the W3C test suite: two `SELECT` expressions return the
  // same blank node iff the labels are the same, and different blank nodes in
  // different rows.
  {
    auto rows = runQuery(kg,
                         "SELECT ?s1 ?s2 (BNODE(?s1) AS ?b1) (BNODE(?s2) AS "
                         "?b2) { ?a <str> ?s1 . ?b <str> ?s2 }",
                         {"?s1", "?s2", "?b1", "?b2"});
    ASSERT_EQ(rows.size(), 4);
    for (const auto& row : rows) {
      EXPECT_EQ(row.at(0) == row.at(1), row.at(2) == row.at(3));
    }
    expectDistinctBlankNodes(getColumn(rows, 2));
    expectDistinctBlankNodes(getColumn(rows, 3));
  }

  // The same for `BIND`s in the `WHERE` clause, with a `FILTER` in between,
  // and with nested calls whose result is not a blank node.
  {
    auto rows = runQuery(kg,
                         "SELECT ?b1 ?b2 ?equal { ?s <p> ?o "
                         "BIND(STR(BNODE(\"a\")) AS ?str) "
                         "BIND(BNODE(\"a\") AS ?b1) FILTER(?o != <o3>) "
                         "BIND(BNODE(\"a\") AS ?b2) "
                         "BIND(STR(?b1) = ?str AS ?equal) }",
                         {"?b1", "?b2", "?equal"});
    ASSERT_EQ(rows.size(), 2);
    for (const auto& row : rows) {
      EXPECT_EQ(row.at(0), row.at(1));
      EXPECT_EQ(row.at(2), trueId);
    }
    expectDistinctBlankNodes(getColumn(rows, 0));
  }

  // `BIND`s that are separated by a triple (which is a join in the SPARQL
  // algebra) return different blank nodes.
  {
    auto rows = runQuery(kg,
                         "SELECT ?b1 ?b2 { BIND(BNODE(\"a\") AS ?b1) "
                         "?s <p> ?o BIND(BNODE(\"a\") AS ?b2) }",
                         {"?b1", "?b2"});
    ASSERT_EQ(rows.size(), 2);
    EXPECT_EQ(rows.at(0).at(0), rows.at(1).at(0));
    auto blankNodes = getColumn(rows, 1);
    blankNodes.push_back(rows.at(0).at(0));
    expectDistinctBlankNodes(blankNodes);
  }

  // Calls within the same `FILTER`.
  {
    auto rows = runQuery(
        kg, "SELECT ?o { ?s <p> ?o FILTER(BNODE(\"a\") = BNODE(\"a\")) }",
        {"?o"});
    EXPECT_EQ(rows.size(), 2);
    rows = runQuery(
        kg, "SELECT ?o { ?s <p> ?o FILTER(BNODE(\"a\") = BNODE(\"b\")) }",
        {"?o"});
    EXPECT_EQ(rows.size(), 0);
  }

  // The `SELECT` expressions of a query with `GROUP BY` (outside and inside of
  // aggregates).
  {
    auto rows =
        runQuery(kg,
                 "SELECT ?s (BNODE(\"a\") AS ?b1) (BNODE(\"a\") AS ?b2) "
                 "(COUNT(DISTINCT BNODE(\"a\")) AS ?count) "
                 "{ ?s ?p ?o } GROUP BY ?s",
                 {"?b1", "?b2", "?count"});
    ASSERT_EQ(rows.size(), 2);
    for (const auto& row : rows) {
      EXPECT_EQ(row.at(0), row.at(1));
    }
    expectDistinctBlankNodes(getColumn(rows, 0));
    EXPECT_THAT(getColumn(rows, 2),
                ::testing::UnorderedElementsAre(Id::makeFromInt(3),
                                                Id::makeFromInt(1)));
  }

  // Different subqueries return different blank nodes (see
  // https://github.com/w3c/sparql-query/issues/292).
  {
    auto rows = runQuery(kg,
                         "SELECT * { "
                         "{ SELECT ?s ?b1 { ?s <str> ?o BIND(BNODE(\"a\") AS "
                         "?b1) } } "
                         "{ SELECT ?s ?b2 { ?s <str> ?o BIND(BNODE(\"a\") AS "
                         "?b2) } } }",
                         {"?b1", "?b2"});
    ASSERT_EQ(rows.size(), 2);
    auto blankNodes = getColumn(rows, 0);
    ql::ranges::copy(getColumn(rows, 1), std::back_inserter(blankNodes));
    expectDistinctBlankNodes(blankNodes);
  }

  // Only simple literals (and equivalently `xsd:string`s) are valid labels.
  {
    auto rows = runQuery(
        kg,
        "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> "
        "SELECT * { BIND(BNODE(1) AS ?b1) BIND(BNODE(<x>) AS ?b2) "
        "BIND(BNODE(\"a\"@en) AS ?b3) BIND(BNODE(\"a\"^^xsd:string) AS ?b4) "
        "BIND(BNODE(\"a\") AS ?b5) }",
        {"?b1", "?b2", "?b3", "?b4", "?b5"});
    ASSERT_EQ(rows.size(), 1);
    EXPECT_THAT(rows.at(0),
                ElementsAre(isUndefined(), isUndefined(), isUndefined(),
                            isBlankNode(), isBlankNode()));
    EXPECT_EQ(rows.at(0).at(3), rows.at(0).at(4));
  }
}
