#import "../lib/template.typ": main
#import "../lib/simpleTable.typ": simpleTable
#import "../lib/codeBlock.typ": codeBlock

= Logistic Regression and Decision Boundaries

== Logistic Regression: Overview

Logistic regression is one of the most commonly used classification algorithms. Unlike linear regression, which predicts continuous values, logistic regression is designed to predict probabilities between 0 and 1. For example, it can be used to classify whether a tumor is malignant (1) or benign (0).

=== Sigmoid Function

The foundation of logistic regression is the sigmoid function, also known as the logistic function. Its formula is:

$ g(z) = frac{1}{1 + e^{-z}} $

Key properties of the sigmoid function:
- $ g(z) in (0, 1) $ for all real values of $z$.
- When $ z $ to $ = 0 $, $ g(z) = 0.5 $.
- As $ z $ to $-$ As $ z $ to $ -infinity $, $ g(z) $ to $ 0 $.

#figure(
  image("./images/sigmoid-curve.png"),
  caption: [
    The sigmoid function starts near 0, increases rapidly around $ z = 0 $, and approaches 1 asymptotically.
  ]
)

=== Logistic Regression Model

Logistic regression combines linear regression with the sigmoid function. The model is defined as:

$ f(x) = g(w*x + b) $

Where:
- $ w $: Weight (slope of the linear function).
- $ b $: Bias (intercept of the linear function).
- $ z = w*x + b $: Linear combination of the input features.

Interpretation:
- $ f(x) $ gives the probability that the output is 1 (positive class) for a given input $ x $.
- $ P(y=1|x) = f(x) $
- $ P(y=0|x) = 1 - f(x) $

=== Example: Tumor Classification

Imagine a dataset where:
- The input $ x $ is the tumor size.
- The output $ y $ is 1 for malignant tumors and 0 for benign ones.

The decision boundary is determined by the point where $ f(x) = 0.5 $:
- $ w*x + b = 0 $
- $ x = -frac{b}{w} $

#figure(
  image("./images/decision-boundary.png"),
  caption: [
    The decision boundary separates the classes based on the tumor size. Data points to the right of the boundary are classified as malignant.
  ]
)

=== Decision Boundary and Interpretation

The decision boundary is a threshold where the model predicts a 50% probability of belonging to either class. For example:
- If $ f(x) = 0.7 $, the model predicts a 70% chance that the tumor is malignant.
- The complement, $ 1 - f(x) $, represents the probability of the benign class (30% in this case).

=== Advantages of Logistic Regression

- **Probabilistic Output**: Unlike hard classifications, logistic regression provides probabilities, allowing better interpretation.
- **Simplicity**: Easy to implement and computationally efficient for binary classification problems.

=== Implementation in Python

The following Python code demonstrates how to compute the sigmoid function and use it for logistic regression:

#codeBlock(
```
python
  import numpy as np

  def sigmoid(z):
      return 1 / (1 + np.exp(-z))

  # Example: Compute probability
  weights = np.array([0.5])
  bias = -1.0
  tumor_size = np.array([2.0])  # Input feature
  z = np.dot(weights, tumor_size) + bias
  probability = sigmoid(z)

  print(f"Probability of malignancy: {probability:.2f}")
```
)

In this example:
- Weight $ w = 0.5 $.
- Bias $ b = -1.0 $.
- Input tumor size $$x = 2.0$$.

The computed probability indicates the likelihood that the tumor is malignant.

=== Key Takeaways

- Logistic regression maps linear combinations of inputs to probabilities using the sigmoid function.
- The decision boundary is where the model predicts equal probabilities for both classes.
- Logistic regression is a foundational algorithm for classification tasks and is interpretable and effective for many problems.

