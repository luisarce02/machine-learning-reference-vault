#import "../lib/template.typ": main
#import "../lib/simpleTable.typ": simpleTable
#import "../lib/codeBlock.typ": codeBlock

= Regularized Logistic Regression

== Key Concepts

Regularized logistic regression helps prevent overfitting by adding a regularization term to the cost function. This is particularly useful when working with polynomial features or models with many parameters.

=== Cost Function

The cost function for logistic regression is modified by adding a regularization term, which penalizes large values of the parameters to avoid overfitting. The cost function with regularization is given by:

$
J(w, b) = frac{1}{m} sum_{i=1}^{m} lr( -y^{(i)} log(f(w^T x^{(i)})) - (1 - y^{(i)}) log(1 - f(w^T x^{(i)})) lr) + frac{lambda}{2m} sum_{j=1}^{n} w_j^2
$

Where:
- $m$ is the number of training examples.
- $w_j$ are the parameters of the model.
- $lambda$ is the regularization parameter.

The term $frac{lambda}{2m} sum_{j=1}^{n} w_j^2$ penalizes large values of $w$, keeping the model more general and reducing the risk of overfitting.

=== Gradient Descent Update

To minimize the cost function, we use gradient descent. The update rules for the parameters are as follows:

1. For $w_j$:
   $
   w_j := w_j - alpha lr( frac{1}{m} sum_{i=1}^{m} (f(w^T x^{(i)}) - y^{(i)}) x_j^{(i)} + frac{lambda}{m} w_j lr)
   $

2. For $b$:
   $
   b := b - alpha lr( frac{1}{m} sum_{i=1}^{m} (f(w^T x^{(i)}) - y^{(i)}) lr)
   $

Here, $alpha$ is the learning rate, and $b$ is not regularized, so no regularization term is added to the update rule for $b$.

== Example

Consider a situation where we have a polynomial feature of a high order, leading to overfitting. The regularization term helps simplify the decision boundary, as shown in the following graph:

=== Implementing Regularized Logistic Regression

In practice, you would implement the gradient descent update for the parameters as follows:

#codeBlock(
  ```
  python
  def regularized_logistic_regression(X, y, lambda_val, alpha, num_iters):
      m, n = X.shape
      w = np.zeros(n)
      b = 0

      for _ in range(num_iters):
          z = np.dot(X, w) + b
          f = 1 / (1 + np.exp(-z))  # Sigmoid function
          dw = (1 / m) * np.dot(X.T, (f - y)) + (lambda_val / m) * w
          db = (1 / m) * np.sum(f - y)

          w -= alpha * dw
          b -= alpha * db

      return w, b
  ```
)

In this function:
- `X` is the matrix of input features.
- `y` is the vector of labels.
- `lambda_val` is the regularization parameter.
- `alpha` is the learning rate.
- `num_iters` is the number of iterations for gradient descent.

This code allows us to fit a regularized logistic regression model to a dataset, minimizing the cost function with respect to the parameters $w$ and $b$.

