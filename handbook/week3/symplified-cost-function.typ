#import "../lib/template.typ": main
#import "../lib/simpleTable.typ": simpleTable
#import "../lib/codeBlock.typ": codeBlock

= Logistic Regression Notes

== Simplified Cost Function

In this class, we simplified the logistic regression cost function for easier implementation, especially when using gradient descent. Here's a breakdown:

=== Loss Function Simplification

The simplified loss function for binary classification (where $y \in {0, 1}$) is written as:

#simpleTable(
  columns: (1fr, 1fr),
  [*Case*], [*Simplified Loss*],
  [$y = 1$], [
    $ -log(f(x)) $
  ],
  [$y = 0$], [
    $ -log(1 - f(x)) $
  ],
)

However, we can combine these cases into a single expression:

$ L = - lr( y \cdot log(f(x)) + (1 - y) \cdot log(1 - f(x)) lr) $

This formulation works because:
- When $y = 1$, the second term vanishes (as $(1 - y) = 0$), leaving $-log(f(x))$.
- When $y = 0$, the first term vanishes (as $y = 0$), leaving $-log(1 - f(x))$.

=== Cost Function Definition

Using the simplified loss function, the cost function $J$ for the entire dataset is defined as:

$ J(w, b) = frac{1}{m} sum{i=1}^{m} - lr( y^{(i)} \cdot log(f(x^{(i)})) + (1 - y^{(i)}) \cdot log(1 - f(x^{(i)})) lr) $

Where:
- $m$ is the number of training examples.
- $f(x^{(i)})$ is the logistic function applied to the $i$-th training example.

We can rewrite it by factoring out the negative sign:

$ J(w, b) = - frac{1}{m} sum{i=1}^{m} lr( y^{(i)} \cdot log(f(x^{(i)})) + (1 - y^{(i)}) \cdot log(1 - f(x^{(i)})) lr) $

=== Properties of the Cost Function

- **Convexity**: This cost function is convex, meaning it has a single global minimum. This is crucial for gradient descent, ensuring it converges to the optimal solution.
- **Statistical Justification**: The function is derived using the principle of maximum likelihood estimation, which helps identify the most likely parameters for the logistic model.

== Example: Visualizing Cost for Two Parameter Choices

The lab exercise showed two different parameter choices ($w, b$), resulting in distinct costs. Below is a Python snippet to compute and visualize these differences:

#codeBlock(
```
python
  import numpy as np
  import matplotlib.pyplot as plt

  def logistic_function(x, w, b):
      return 1 / (1 + np.exp(-(w * x + b)))

  def cost_function(y, y_pred):
      return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

  x_train = np.linspace(-10, 10, 100)
  y_train = (x_train > 0).astype(int)  # Binary target: 1 for x > 0, else 0

  params = [
      {"w": 1, "b": 0},  # Choice 1
      {"w": 2, "b": -1}  # Choice 2
  ]

  costs = []
  for param in params:
      y_pred = logistic_function(x_train, param["w"], param["b"])
      costs.append(cost_function(y_train, y_pred))

  print(f"Costs for parameter choices: {costs}")
```
)

=== Notes

- The **blue decision boundary** with optimized parameters resulted in a lower cost compared to the **magenta decision boundary**, showing better model performance.
- Implementing this cost function in code will be essential for practical gradient descent applications.

In the next session, we'll explore how gradient descent can be applied to optimize this cost function for logistic regression.
