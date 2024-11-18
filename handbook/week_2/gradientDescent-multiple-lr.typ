#import "../lib/template.typ": main
#import "../lib/simpleTable.typ": simpleTable
#import "../lib/codeBlock.typ": codeBlock

= Gradient Descent for Multiple Linear Regression

== Multiple Linear Regression Overview

In multiple linear regression, we generalize the simple linear regression to work with multiple features. Using vectorized notation, the model can be expressed as:

#simpleTable(
  columns: (1fr, 1fr),
  [*Attribute*], [*Formula*],
  [*Model*], [
    $ f_(w,b)(x) = w^T x + b $
  ],
  [*Parameters*], [
    $ w \in bb{R}^n, \ b \in bb{R} $
  ],
  [*Cost Function*], [
    $ J(w, b) = frac{1}{2m} sum_{i=1}^m (f_(w,b)(x^(i)) - y^(i))^2 $
  ],
  [*Objective*], [
    $ \min_{w,b} J(w, b) $
  ],
)

In this model:
- $w$ is a vector of weights for the features.
- $b$ is the bias term (a scalar).
- $x$ is a feature vector for each training example.

== Gradient Descent for Multiple Features

Gradient descent minimizes the cost function $J(w, b)$ iteratively by updating the parameters $w$ and $b$. The update rules are:

#simpleTable(
  columns: (1fr, 1fr),
  [*Parameter*], [*Update Rule*],
  [$w_j$], [
    $ w_j := w_j - alpha frac{partial J(w, b)}{partial w_j} $
  ],
  [$b$], [
    $ b := b - alpha frac{partial J(w, b)}{partial b} $
  ],
)

The partial derivatives for the parameters are computed as:

#simpleTable(
  columns: (1fr, 1fr),
  [*Derivative*], [*Formula*],
  [$frac{partial J(w, b)}{partial w_j}$], [
    $ frac{1}{m} sum_{i=1}^m lr( f_(w,b)(x^(i)) - y^(i) lr) x_j^(i) $
  ],
  [$frac{partial J(w, b)}{partial b}$], [
    $ frac{1}{m} sum_{i=1}^m lr( f_(w,b)(x^(i)) - y^(i) lr) $
  ],
)

=== Vectorized Gradient Descent

Using vectorized computation, we can compute updates efficiently:

#codeBlock(
```python
import numpy as np

def gradient_descent(X, y, w, b, alpha, num_iters):
    m = len(y)
    for i in range(num_iters):
        predictions = np.dot(X, w) + b
        errors = predictions - y
        grad_w = (1/m) * np.dot(X.T, errors)
        grad_b = (1/m) * np.sum(errors)
        w -= alpha * grad_w
        b -= alpha * grad_b
    return w, b
    ```
)

=== Example

For a dataset with features $X \in bb{R}^{m times n}$ and target values $y \in bb{R}^m$, initialize $w$ and $b$:

#codeBlock(
```python
# Example data
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([4, 6, 8])
w = np.zeros(X.shape[1])
b = 0
alpha = 0.01
num_iters = 1000

# Perform gradient descent
w, b = gradient_descent(X, y, w, b, alpha, num_iters)
print(f"Weights: {w}, Bias: {b}")
```
)

=== Notes on Convergence

1. Learning rate ($alpha$):
   - If $alpha$ is too small, convergence will be slow.
   - If $alpha$ is too large, the algorithm might diverge.
2. Vectorization significantly speeds up computation by eliminating loops.

== Alternative: Normal Equation

For multiple linear regression, an alternative to gradient descent is the *normal equation*:

$ w = (X^T X)^{-1} X^T y $

Advantages:
- No iterative process is required.
- Provides a direct solution for $w$ and $b$.

Disadvantages:
- Computationally expensive for large feature sets.
- Not applicable to most machine learning algorithms (e.g., logistic regression, neural networks).

