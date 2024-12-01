#import "../lib/template.typ": main
#import "../lib/simpleTable.typ": simpleTable
#import "../lib/codeBlock.typ": codeBlock

= Overfitting Problem

== Notes on Overfitting and Underfitting

=== Understanding the Problem

Overfitting occurs when a model is too closely aligned with the training data, capturing noise rather than the underlying pattern. This leads to poor generalization on new data.

Conversely, underfitting happens when the model fails to capture the patterns in the training data, resulting in poor performance both on the training set and unseen data.

=== Example: Predicting House Prices

Consider a dataset where the input feature $x$ represents the size of a house, and $y$ is the house price. Let's explore three cases:

#simpleTable(
  columns: (1fr, 1fr),
  [*Case*], [*Model Description*],
  [Linear Fit], [
    A linear regression model predicts $y = w_1 x + b$. It poorly fits the data due to its simplicity, resulting in *high bias*.
  ],
  [Quadratic Fit], [
    Adding $x^2$ as a feature, the model becomes $y = w_1 x + w_2 x^2 + b$. It better captures the pattern, balancing bias and variance.
  ],
  [Fourth-Order Polynomial], [
    A high-degree polynomial model, $y = w_1 x + w_2 x^2 + w_3 x^3 + w_4 x^4 + b$, fits the training data perfectly. However, it generalizes poorly due to *high variance*.
  ],
)

=== Visualization

#figure(
  image("./images/overfitting-examples.png"),
  caption: [
    Comparison of models: linear (underfit), quadratic (optimal fit), and fourth-order polynomial (overfit).
  ]
)

---

=== Definitions and Insights

==== Underfitting
- High bias: The model has a rigid assumption (e.g., linearity) and cannot adapt to the data.
- Example formula:
  $ y = w_1 x + b $

==== Overfitting
- High variance: The model memorizes the training data, resulting in erratic predictions for new inputs.
- Example formula:
  $ y = w_1 x + w_2 x^2 + w_3 x^3 + w_4 x^4 + b $
- Even a small change in the training set could lead to drastically different models.

==== Proper Fit
- The goal is to achieve a balance where the model captures the pattern but ignores noise.
- Intuition: A good model generalizes well to unseen data.

---

=== Key Concept: Generalization
Generalization refers to the model's ability to perform well on new, unseen data. It is essential for ensuring practical utility in machine learning applications.

---

=== Metrics for Evaluation

The following metrics are used to identify overfitting or underfitting:

#simpleTable(
  columns: (1fr, 1fr),
  [*Metric*], [*Description*],
  [Training Error], [
    The error on the training set. Overfitting occurs when this is very low, but test error is high.
  ],
  [Test Error], [
    The error on unseen data. High test error compared to training error indicates overfitting.
  ],
)

---

=== Balancing Bias and Variance
- High bias: Underfitting. The model is too simplistic.
- High variance: Overfitting. The model is too complex.

==== Regularization
A technique to prevent overfitting by penalizing overly complex models. Examples include:
- $ L_1 $ regularization (Lasso)
- $ L_2 $ regularization (Ridge)

Cost function with $ L_2 $ regularization:
$ J(w) = 1/(2m) sum_(i=1)^m ( f_(w)(x^(i)) - y^(i) )^2 + lambda sum_{j=1}^n w_j^2 $

---

=== Summary

To build effective models:
- Avoid underfitting by increasing model complexity or adding features.
- Prevent overfitting using regularization or reducing model complexity.
- Always validate on unseen data to ensure generalization.
