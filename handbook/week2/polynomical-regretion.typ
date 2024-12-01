#import "../lib/template.typ": main
#import "../lib/simpleTable.typ": simpleTable
#import "../lib/codeBlock.typ": codeBlock

= Polynomial Regression

== Introduction

Polynomial regression extends linear regression by allowing the model to fit non-linear relationships in the data. Instead of fitting a straight line, we use polynomial functions to create curves that can better capture the patterns in the dataset. This is particularly useful when the data exhibits non-linear trends.

=== Why Polynomial Regression?

Suppose we have a dataset of housing prices where the feature $x$ represents the size of the house in square feet. A straight line might not capture the pattern well, as housing prices often increase non-linearly with size. Instead, we can try fitting a polynomial function like:

1. **Quadratic**: $f(x) = w_1 x + w_2 x^2 + b$
2. **Cubic**: $f(x) = w_1 x + w_2 x^2 + w_3 x^3 + b$

Each additional term allows the model to capture more complexity in the data.

== Polynomial Features

To create polynomial regression models, we generate polynomial features by raising the input feature $x$ to higher powers. For example:
- $x$
- $x^2$
- $x^3$

These features allow us to fit curves to the data.

#simpleTable(
  columns: (1fr, 1fr),
  [*Feature*], [*Range*],
  [$x$], [$1$ to $1000$],
  [$x^2$], [$1$ to $1,000,000$],
  [$x^3$], [$1$ to $1,000,000,000$],
)

*Note*: The range of values increases dramatically with higher powers, which makes feature scaling crucial for gradient descent.

=== Feature Scaling

If the range of feature values varies significantly, gradient descent may converge very slowly or fail entirely. Therefore, it is essential to scale the features to comparable ranges.

== Alternative Polynomial Functions

Besides $x^2$ and $x^3$, other transformations like the square root of $x$ can also be used:
- Example: $f(x) = w_1 x + w_2 sqrt{x} + b$
- The square root function grows slower than higher powers, making it another useful transformation for certain datasets.

== Choosing the Right Function

Choosing which polynomial or alternative functions to include in your model depends on:
1. The dataset and its characteristics.
2. Experimenting with different functions and evaluating performance using cross-validation.

In the next courses, you'll learn systematic ways to decide on the best functions and models.

== Code Example: Implementing Polynomial Regression

Using Python and Scikit-learn, we can implement polynomial regression easily. Below is an example of how to generate polynomial features and fit a regression model:

#codeBlock(
```python
  import numpy as np
  from sklearn.preprocessing import PolynomialFeatures
  from sklearn.linear_model import LinearRegression
  from sklearn.pipeline import Pipeline

  # Generate dataset
  x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
  y = np.array([1.5, 3.2, 6.0, 10.1, 15.5])

  # Create a pipeline for polynomial regression
  degree = 3
  model = Pipeline([
      ('poly_features', PolynomialFeatures(degree=degree)),
      ('linear_regression', LinearRegression())
  ])

  # Fit the model
  model.fit(x, y)

  # Predict values
  x_pred = np.linspace(1, 5, 100).reshape(-1, 1)
  y_pred = model.predict(x_pred)
  print(f"Predicted values: {y_pred}")
```
)

This code demonstrates how polynomial regression can be implemented in practice using Scikit-learn's pipeline for feature generation and model fitting.

== Summary

- Polynomial regression generalizes linear regression to fit non-linear data.
- Feature scaling is critical when using higher-order polynomial terms.
- Tools like Scikit-learn make it easy to implement polynomial regression with just a few lines of code.
- Experimentation and evaluation are key to selecting the best model and transformations for your data.
