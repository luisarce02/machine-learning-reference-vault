#import "../lib/template.typ": main
#import "../lib/simpleTable.typ": simpleTable
#import "../lib/codeBlock.typ": codeBlock

= Logistic Regression Notes

== Cost Function for Logistic Regression

The cost function is a critical part of logistic regression as it helps to evaluate how well the chosen parameters ($w$ and $b$) fit the training data. Unlike linear regression, the quadratic cost function is not suitable for logistic regression because it leads to a non-convex function, which can result in convergence to local minima during gradient descent.

=== Why Quadratic Cost Function Fails

In linear regression, the cost function is defined as:

#simpleTable(
  columns: (1fr, 1fr),
  [*Attribute*], [*Formula*],
  [*Cost Function*], [
    $ J(w, b) = frac{1}{2m} sum_{i=1}^m lr( f_{w,b}(x^{(i)}) - y^{(i)} lr)^2 $
  ],
  [*Convexity*], [Convex (Bowl-shaped)],
)

However, for logistic regression, where the hypothesis function is:

$ f_{w,b}(x) = frac{1}{1 + e^{-w \cdot x - b}} $

Using the same quadratic cost function would result in a non-convex function with multiple local minima, making gradient descent unreliable.

#figure(
  image("./images/non-convex-cost.png"),
  caption: [
    Example of a non-convex cost function for logistic regression using quadratic cost.
  ]
)

=== Convex Cost Function for Logistic Regression

To address this, a new cost function is defined:

$ J(w, b) = frac{1}{m} sum_{i=1}^m L(f_{w,b}(x^{(i)}), y^{(i)}) $

Where $L$ is the loss function for a single training example, defined as:

- If $y = 1$:
  $ L(f_{w,b}(x), y) = -log(f_{w,b}(x)) $

- If $y = 0$:
  $ L(f_{w,b}(x), y) = -log(1 - f_{w,b}(x)) $

This formulation ensures that the cost function is convex, guaranteeing convergence to a global minimum using gradient descent.

=== Loss Function Visualization

==== Case 1: When $y = 1$
The loss function is $-log(f_{w,b}(x))$. The graph of this function shows:
- Loss approaches $0$ as $f_{w,b}(x) \to 1$.
- Loss increases sharply as $f_{w,b}(x) \to 0$.

==== Case 2: When $y = 0$
The loss function is $-log(1 - f_{w,b}(x))$. The graph of this function shows:
- Loss approaches $0$ as $f_{w,b}(x) \to 0$.
- Loss increases sharply as $f_{w,b}(x) \to 1$.


=== Combined Cost Function

The overall cost function combines the two cases:

$ J(w, b) = -frac{1}{m} sum_{i=1}^m lr[ y^{(i)} log(f_{w,b}(x^{(i)})) + (1 - y^{(i)}) log(1 - f_{w,b}(x^{(i)})) lr] $

This formulation penalizes incorrect predictions heavily, ensuring the model learns to predict probabilities closer to the actual labels.

=== Gradient Descent for Logistic Regression

To minimize $J(w, b)$, we use gradient descent. The partial derivatives of $J(w, b)$ with respect to $w$ and $b$ are derived from the above cost function:

$ frac{partial J(w, b)}{partial w} = frac{1}{m} sum_{i=1}^m lr( f_{w,b}(x^{(i)}) - y^{(i)} lr) x^{(i)} $

$ frac{partial J(w, b)}{partial b} = frac{1}{m} sum_{i=1}^m lr( f_{w,b}(x^{(i)}) - y^{(i)} lr) $

#codeBlock(
```
python
  def gradient_descent(X, y, w, b, alpha, num_iters):
      for i in range(num_iters):
          predictions = 1 / (1 + np.exp(-(X.dot(w) + b)))
          errors = predictions - y
          w -= alpha * (X.T.dot(errors) / len(y))
          b -= alpha * np.sum(errors) / len(y)
      return w, b
```
)

This iterative process ensures that $w$ and $b$ converge to values that minimize the cost function.

#figure(
  image("./images/logistic-gradient-descent.jpg"),
  caption: [
    Gradient descent converging to the global minimum of the convex cost function.
  ]
)
