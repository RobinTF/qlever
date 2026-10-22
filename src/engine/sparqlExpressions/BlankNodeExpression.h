//   Copyright 2025, University of Freiburg,
//   Chair of Algorithms and Data Structures.
//   Author: Robin Textor-Falconi <textorr@informatik.uni-freiburg.de>

#ifndef QLEVER_SRC_ENGINE_SPARQLEXPRESSIONS_BLANKNODEEXPRESSION_H_
#define QLEVER_SRC_ENGINE_SPARQLEXPRESSIONS_BLANKNODEEXPRESSION_H_

#include "engine/sparqlExpressions/SparqlExpression.h"

namespace sparqlExpression {

// Create a `SparqlExpression` representing the term `BNODE(?x)`.
SparqlExpression::Ptr makeBlankNodeExpression(SparqlExpression::Ptr child);

// Create a `SparqlExpression` representing the term `BNODE()`.
SparqlExpression::Ptr makeUniqueBlankNodeExpression();

// Return true iff `expression` is `BNODE()` (without an argument).
bool isBlankNodeExpressionWithoutArgument(const SparqlExpression& expression);

// The following functions determine which calls of `BNODE` with an argument
// have to return the same blank node for the same label (see
// `BlankNodeExpression.cpp` for details). All of them consider the complete
// tree of `expression`.

// Return the number of calls of `BNODE` with an argument.
size_t countLabeledBlankNodeExpressions(SparqlExpression& expression);

// Let the calls of `BNODE` with an argument share their blank nodes with all
// other such calls that are evaluated for the same row of the same
// `EvaluationContext` (within one expression or one GROUP BY).
void shareLabeledBlankNodesWithinEvaluation(SparqlExpression& expression);

// Let the calls of `BNODE` with an argument share their blank nodes with all
// other such calls that are evaluated for the same value of the variable
// `solutionToken`, which has to be bound by `BIND(BNODE() AS ?solutionToken)`.
void shareLabeledBlankNodesViaSolutionToken(SparqlExpression& expression,
                                            const Variable& solutionToken);
}  // namespace sparqlExpression

#endif  // QLEVER_SRC_ENGINE_SPARQLEXPRESSIONS_BLANKNODEEXPRESSION_H_
