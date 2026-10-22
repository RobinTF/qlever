//   Copyright 2025, University of Freiburg,
//   Chair of Algorithms and Data Structures.
//   Author: Robin Textor-Falconi <textorr@informatik.uni-freiburg.de>

#include "engine/sparqlExpressions/BlankNodeExpression.h"

#include <absl/strings/str_cat.h>

#include "engine/sparqlExpressions/LiteralExpression.h"
#include "engine/sparqlExpressions/SparqlExpressionGenerators.h"
#include "engine/sparqlExpressions/SparqlExpressionValueGetters.h"
#include "util/ChunkedForLoop.h"

namespace sparqlExpression {

namespace detail {

// Return the label of `BNODE(label)` if `value` is a valid label, which is the
// case iff it is a simple literal (or equivalently an `xsd:string`). Otherwise,
// `BNODE` is undefined.
template <typename T>
std::optional<std::string> getLabel(T&& value,
                                    const EvaluationContext* context) {
  auto literal = LiteralValueGetterWithoutStrFunction{}(AD_FWD(value), context);
  if (!literal.has_value() || !literal.value().isPlain()) {
    return std::nullopt;
  }
  return std::string{literal.value().getContent()};
}

// SparqlExpression representing the term "BNODE()" and "BNODE(label)". Both
// create actual blank nodes that are distinct from all the blank nodes of the
// dataset. `BNODE()` creates a new blank node on each call. `BNODE(label)` has
// to return "distinct blank nodes for different simple literals, and the same
// blank node for calls with the same simple literal within expressions for one
// solution mapping" (SPARQL 1.1, sec. 17.4.2.9). The scope of "one solution
// mapping" is not precisely defined by the standard, see
// https://github.com/w3c/sparql-query/issues/292. We consider the calls in the
// following places to be evaluated for the same solution mapping:
//
// 1. Within a single expression (e.g. `FILTER(BNODE("a") = BNODE("a"))`).
// 2. Within consecutive `BIND`s of the same group graph pattern (only
//    separated by `FILTER`s, which are applied to the complete group anyway),
//    which includes the expressions of the `SELECT` clause (they become
//    `BIND`s at the end of the `WHERE` clause), as required by the test
//    `bnode01` of the W3C test suite. In the SPARQL algebra, these are directly
//    nested `Extend` operations.
// 3. Within the `SELECT` clause of a query with `GROUP BY` (the same solution
//    is a group outside of an aggregate, and an input row inside of one).
//
// All other calls (e.g. two `BIND`s that are separated by a triple, which
// is a join in the SPARQL algebra, or calls in different subqueries) return
// different blank nodes. Case 1 and 3 are implemented by sharing the blank
// nodes via the `EvaluationContext`. For case 2, the parser inserts a
// `BIND(BNODE() AS ?token)` before the first `BIND` with `BNODE(label)`, so
// that each solution has a unique solution token, and each `BNODE(label)`
// receives that token as an additional child. The blank node for a pair of a
// token and a label is then stored in the `LocalVocab` (see
// `LocalVocab::getOrCreateLabeledBlankNode`), from where the later `BIND`s can
// find it. The parser only does these things if there actually are several
// calls of `BNODE(label)` that can share a blank node, all other calls simply
// create a new blank node like `BNODE()`. For details see
// `GraphPattern::shareLabeledBlankNodes`.
class BlankNodeExpression : public SparqlExpression {
  // For `BNODE(label)`, the first child is the label, and the optional second
  // child is the variable with the solution token (see above). For `BNODE()`,
  // there are no children.
  std::vector<Ptr> children_;
  // True iff blank nodes have to be shared with other calls of `BNODE(label)`
  // via the `EvaluationContext` (see above).
  bool shareWithinEvaluation_ = false;
  // Counter that is incremented for each cache key to ensure its uniqueness.
  mutable std::atomic_uint64_t cacheBreaker_ = 0;

 public:
  // ___________________________________________________________________________
  explicit BlankNodeExpression(Ptr label) {
    children_.push_back(std::move(label));
  }

  // Constructor for the case of no arguments.
  BlankNodeExpression() = default;
  BlankNodeExpression(const BlankNodeExpression&) = delete;
  BlankNodeExpression& operator=(const BlankNodeExpression&) = delete;

  // Return true iff this is `BNODE(label)` (and not `BNODE()`).
  bool hasLabel() const { return !children_.empty(); }

  // See the comment at the beginning of this class for the following two
  // functions.
  void shareWithinEvaluation() {
    AD_CORRECTNESS_CHECK(hasLabel());
    shareWithinEvaluation_ = true;
  }
  void setSolutionToken(const Variable& solutionToken) {
    AD_CORRECTNESS_CHECK(hasLabel());
    children_.resize(1);
    children_.push_back(std::make_unique<VariableExpression>(solutionToken));
  }

  // Evaluate function for the case where no argument is given and each row gets
  // a new unique blank node index.
  ExpressionResult evaluateWithoutArguments(EvaluationContext* context) const {
    // As part of a GROUP BY (and outside of an aggregate) we only return a
    // single (constant) blank node per group. Inside an aggregate we must still
    // produce one blank node per row, so that e.g. `COUNT(BNODE())` counts the
    // number of rows in the group.
    if (worksOnAggregatedData(context)) {
      return makeNewBlankNode(context);
    }
    VectorWithMemoryLimit<Id> result{context->_allocator};
    const size_t numElements = context->size();
    result.reserve(numElements);

    ad_utility::chunkedForLoop<1000>(
        0, numElements,
        [&result, context](size_t) {
          result.push_back(makeNewBlankNode(context));
        },
        [context]() { context->cancellationHandle_->throwIfCancelled(); });
    return result;
  }

  // __________________________________________________________________________
  ExpressionResult evaluate(EvaluationContext* context) const override {
    if (!hasLabel()) {
      return evaluateWithoutArguments(context);
    }
    VectorWithMemoryLimit<Id> solutionTokens = evaluateSolutionTokens(context);
    auto getSolutionToken = [&solutionTokens](size_t i) -> std::optional<Id> {
      if (solutionTokens.empty()) {
        return std::nullopt;
      }
      return solutionTokens.at(solutionTokens.size() == 1 ? 0 : i);
    };
    return std::visit(
        [this, context, &getSolutionToken](auto&& labels) -> ExpressionResult {
          using T = std::decay_t<decltype(labels)>;
          // An invalid constant label leads to a single undefined value. As
          // part of a GROUP BY (and outside of an aggregate) the label is
          // constant within the group, so we only produce a single (constant)
          // blank node.
          if constexpr (isConstantResult<T>) {
            auto label = getLabel(T{labels}, context);
            if (!label.has_value()) {
              return Id::makeUndefined();
            }
            if (worksOnAggregatedData(context)) {
              return getBlankNode(context, getSolutionToken(0),
                                  context->_beginIndex, true, label.value());
            }
          }
          VectorWithMemoryLimit<Id> result{context->_allocator};
          const size_t numElements = context->size();
          result.reserve(numElements);
          auto generator =
              detail::makeGenerator(AD_FWD(labels), numElements, context);
          auto it = generator.begin();
          ad_utility::chunkedForLoop<1000>(
              0, numElements,
              [this, &result, context, &getSolutionToken, &it](size_t i) {
                auto label = getLabel(std::move(*it), context);
                ++it;
                result.push_back(label.has_value()
                                     ? getBlankNode(context,
                                                    getSolutionToken(i),
                                                    context->_beginIndex + i,
                                                    false, label.value())
                                     : Id::makeUndefined());
              },
              [context]() {
                context->cancellationHandle_->throwIfCancelled();
              });
          return result;
        },
        children_.at(0)->evaluate(context));
  }

  // ___________________________________________________________________________
  std::string getCacheKey(const VariableToColumnMap& map) const override {
    std::string key = "#BlankNode#";
    for (const auto& child : children_) {
      absl::StrAppend(&key, child->getCacheKey(map), "_");
    }
    // `BNODE` is not deterministic, so its result must never be reused.
    absl::StrAppend(&key, cacheBreaker_++);
    return key;
  }

  // BNODE() generates a fresh blank node on every invocation.
  bool isDeterministic() const override { return false; }

  // The version of `BNODE()` without arguments always returns a defined value.
  bool isResultAlwaysDefined(const VariableToColumnMap&) const override {
    return !hasLabel();
  }

 private:
  ql::span<Ptr> childrenImpl() override { return children_; }

  // Return a new blank node that is distinct from all other blank nodes.
  static Id makeNewBlankNode(EvaluationContext* context) {
    return Id::makeFromBlankNodeIndex(context->_localVocab.getBlankNodeIndex(
        context->_qec.getIndex().getBlankNodeManager()));
  }

  // Return the solution token for each row of the `context`, or a single one if
  // it is constant, or none if this expression has no solution token.
  VectorWithMemoryLimit<Id> evaluateSolutionTokens(
      EvaluationContext* context) const {
    VectorWithMemoryLimit<Id> solutionTokens{context->_allocator};
    if (children_.size() < 2) {
      return solutionTokens;
    }
    std::visit(
        [context, &solutionTokens](auto&& tokens) {
          using T = std::decay_t<decltype(tokens)>;
          if constexpr (std::is_same_v<T, Id>) {
            solutionTokens.push_back(tokens);
          } else {
            for (auto&& token : detail::makeGenerator(
                     AD_FWD(tokens), context->size(), context)) {
              // The solution token is always a variable that is bound by
              // `BIND(BNODE() AS ?token)`, so it consists of `Id`s.
              if constexpr (ad_utility::isSimilar<decltype(token), Id>) {
                solutionTokens.push_back(token);
              } else {
                AD_FAIL();
              }
            }
          }
        },
        children_.at(1)->evaluate(context));
    return solutionTokens;
  }

  // Return the blank node for `label` in the solution with the given
  // `solutionToken` (if present), or else in the given `row` of the evaluation
  // (`isGroup` is true iff the row is the first row of a group of a GROUP BY,
  // see `EvaluationContext::labeledBlankNodesPerGroup_`).
  Id getBlankNode(EvaluationContext* context, std::optional<Id> solutionToken,
                  size_t row, bool isGroup, std::string_view label) const {
    if (solutionToken.has_value() && !solutionToken.value().isUndefined()) {
      return context->_localVocab.getOrCreateLabeledBlankNode(
          context->_qec.getIndex().getBlankNodeManager(), solutionToken.value(),
          label);
    }
    if (!shareWithinEvaluation_) {
      return makeNewBlankNode(context);
    }
    auto& blankNodes = isGroup ? context->labeledBlankNodesPerGroup_
                               : context->labeledBlankNodesPerRow_;
    return blankNodes.getOrCreate(
        row, label, [context]() { return makeNewBlankNode(context); });
  }
};

// Call `function` for each `BNODE(label)` in the tree of `expression`.
template <typename Function>
void forEachLabeledBlankNodeExpression(SparqlExpression& expression,
                                       const Function& function) {
  if (auto* blankNode = dynamic_cast<BlankNodeExpression*>(&expression);
      blankNode != nullptr && blankNode->hasLabel()) {
    function(*blankNode);
  }
  for (auto& child : expression.children()) {
    forEachLabeledBlankNodeExpression(*child, function);
  }
}
}  // namespace detail

SparqlExpression::Ptr makeBlankNodeExpression(SparqlExpression::Ptr child) {
  return std::make_unique<detail::BlankNodeExpression>(std::move(child));
}
SparqlExpression::Ptr makeUniqueBlankNodeExpression() {
  return std::make_unique<detail::BlankNodeExpression>();
}

// _____________________________________________________________________________
bool isBlankNodeExpressionWithoutArgument(const SparqlExpression& expression) {
  auto* blankNode =
      dynamic_cast<const detail::BlankNodeExpression*>(&expression);
  return blankNode != nullptr && !blankNode->hasLabel();
}

// _____________________________________________________________________________
size_t countLabeledBlankNodeExpressions(SparqlExpression& expression) {
  size_t count = 0;
  detail::forEachLabeledBlankNodeExpression(expression,
                                            [&count](auto&) { ++count; });
  return count;
}

// _____________________________________________________________________________
void shareLabeledBlankNodesWithinEvaluation(SparqlExpression& expression) {
  detail::forEachLabeledBlankNodeExpression(
      expression, [](auto& blankNode) { blankNode.shareWithinEvaluation(); });
}

// _____________________________________________________________________________
void shareLabeledBlankNodesViaSolutionToken(SparqlExpression& expression,
                                            const Variable& solutionToken) {
  detail::forEachLabeledBlankNodeExpression(
      expression, [&solutionToken](auto& blankNode) {
        blankNode.setSolutionToken(solutionToken);
      });
}
}  // namespace sparqlExpression
