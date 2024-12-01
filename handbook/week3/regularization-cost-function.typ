#import "../lib/template.typ": main
#import "../lib/simpleTable.typ": simpleTable
#import "../lib/codeBlock.typ": codeBlock

= Cost Function with Regularization

In this video, we discuss the cost function in linear regression and how regularization can help prevent overfitting by reducing the magnitude of parameters.

== Regularization Intuition

When training a model, if we fit a high-degree polynomial, the model may overfit the data, as it may curve excessively to match every point. Regularization aims to penalize large weights, forcing them to shrink, which simplifies the model and helps prevent overfitting.

For instance, if we have a cost function for polynomial regression:

$ J(w) = sum_{i=1}^{m} lr( f(w) - y_i lr)^2 $

To incorporate regularization, we add a term that penalizes large parameters:

$ J(w) = sum_{i=1}^{m} lr( f(w) - y_i lr)^2 + lambda sum_{j=1}^{n} w_j^2 $

Here, $lambda$ is the regularization parameter, and the term $sum_{j=1}^{n} w_j^2$ encourages smaller weights. If $lambda$ is large, the model becomes simpler, as it minimizes the weights' magnitude.

#simpleTable(
  columns: (1fr, 1fr),
  [*Attribute*], [*Formula*],
  [*Regularized Cost Function*], [
    $ J(w) = sum_{i=1}^{m} (f(w) - y_i)^2 + lambda sum_{j=1}^{n} w_j^2 $
  ],
  [*Regularization Term*], [
    $ lambda sum_{j=1}^{n} w_j^2 $
  ],
  [*Objective*], [
    $ \min_w J(w) $
  ],
)

== Regularization Effect

The effect of regularization is to adjust the model complexity. Let's visualize the behavior of the cost function with regularization for a housing price prediction example.

For small $lambda$ (near 0), the model behaves as if there's no regularization, potentially overfitting. For a very large $lambda$, the model becomes overly simplistic, as the weights are heavily penalized.

#codeBlock(
```
python
  import numpy as np
  import matplotlib.pyplot as plt

  def plot_regularized_cost(lambda_value: float):
      # Plot cost function with regularization effect
      w_range = np.linspace(-5, 5, 100)
      cost_values = (w_range**2) + lambda_value * np.sum(w_range**2)  # Simplified cost function for demonstration
      plt.plot(w_range, cost_values, label=f"λ = {lambda_value}")
      plt.xlabel("w")
      plt.ylabel("Cost Function J(w)")
      plt.title(f"Effect of Regularization (λ = {lambda_value})")
      plt.legend()
      plt.show()

  plot_regularized_cost(0)  # No regularization
  plot_regularized_cost(10) # Strong regularization
```
)

=== Cost Function Behavior

When $lambda = 0$, the model has no regularization, and we may observe overfitting.

When $lambda$ is large (e.g., $10^{10}$), the model becomes overly simplistic, with small weights, and the curve becomes smoother.

== Choosing Lambda

Choosing $lambda$ is crucial. A small value of $lambda$ results in overfitting, while a large value leads to underfitting. Cross-validation is often used to find an optimal $lambda$.

For practical implementation, $lambda$ is typically tuned based on training data size and the complexity of the model.

=== Final Notes

- Regularization helps prevent overfitting by penalizing large weights.
- The regularized cost function combines both the original cost function and a penalty term.
- $lambda$ controls the balance between fitting the data and simplifying the model.
- In practice, choose $lambda$ using cross-validation to achieve the best generalization.

