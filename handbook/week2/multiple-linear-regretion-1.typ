#import "../lib/template.typ": main
#import "../lib/simpleTable.typ": simpleTable
#import "../lib/codeBlock.typ": codeBlock

= Linear Regression with Multiple Features

== Introduction

- *Objective*: Extend linear regression to handle multiple features (not just one).
- *Example*: In predicting housing prices, consider features like size, number of bedrooms, floors, and age.
- This approach provides more information, potentially improving prediction accuracy.

== Notation & Terminology

- Let the features be $X_1, X_2, X_3, \u{00B7}, X_n$, where each $X_j$ represents a different characteristic.
  - *Example*: For a house, $X_1 =$ size, $X_2 =$ bedrooms, $X_3 =$ floors, $X_4 =$ age.
- Define $X^{(i)}$ as the vector of features for the $i$-th training example. For instance, if $X^{(2)} = (1416, 3, 2, 40)$, it represents the feature values for the second example.
- *Dimensions*:
  - $X_j$: a single feature across all training examples.
  - $X^{(i)}$: all features for a single training example (a vector).

== Model Definition

With multiple features, the linear regression model is expressed as:

#simpleTable(
  columns: (1fr, 1fr),
  [*Attribute*], [*Formula*],
  [*Model*], [
    $ f(w, b)(X) = w_1 X_1 + w_2 X_2 + \u{00B7} + w_n X_n + b $
  ],
  [*Parameters*], [
    $ w = (w_1, w_2, \u{00B7}, w_n) $, $ b $
  ],
)

== Example: Housing Price Prediction Model

Consider a model to predict housing prices:

- Model formula:
  $ f(w, b)(X) = 0.1 X_1 + 4 X_2 + 10 X_3 - 2 X_4 + 80 $

Interpretations of parameters:
- $ w_1 = 0.1 $: For each additional square foot, the price increases by $0.1 \u{00D7} 1000 = 100$ dollars.
- $ w_2 = 4 $: Each additional bedroom adds $4000$ dollars.
- $ w_3 = 10 $: Each additional floor adds $10000$ dollars.
- $ w_4 = -2 $: Each additional year of age decreases the price by $2000$ dollars.
- $ b = 80 $: Base price (in thousands), assuming zero for all features.

== Matrix Notation

To simplify, we can represent the model using vectors:

- Define:
  - $ W = [w_1, w_2, \u{00B7}, w_n] $ (parameter vector).
  - $ X^{(i)} = [X_1^{(i)}, X_2^{(i)}, \u{00B7}, X_n^{(i)}] $ (feature vector for example $i$).
- Model becomes:
  $ f(W, b)(X^{(i)}) = W \cdot X^{(i)} + b $

This compact form allows easier manipulation and generalization when working with multiple features.

== Cost Function for Multiple Features

The cost function measures the model's accuracy by computing the error between predictions and actual values.

- For $m$ training examples, cost $J(W, b)$ is:

  $ J(W, b) = frac(1,2m) \\sum_{i=1}^{m} (f(W, b)(X^{(i)}) - y^{(i)})^2 $

This function remains convex, ensuring a global minimum, making it suitable for gradient descent optimization.

#codeBlock(
  ```python
  # Python example: Cost function implementation for multiple features
  def compute_cost(X, y, W, b):
      m = len(y)
      cost = np.sum((X @ W + b - y) * 2) / (2 * m)
      return cost
  ```
)

== Summary

- *Multiple features* enable a richer model by incorporating more information, improving predictions.
- *Matrix/vector notation* simplifies representation and calculation, making it efficient to handle large feature sets.
