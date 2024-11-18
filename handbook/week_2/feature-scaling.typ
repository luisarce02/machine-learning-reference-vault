#import "../lib/template.typ": main
#import "../lib/simpleTable.typ": simpleTable
#import "../lib/codeBlock.typ": codeBlock

= Feature Scaling

== Notes on Feature Scaling and Gradient Descent

=== Overview

Feature scaling is a technique to make gradient descent converge faster by transforming features to take on comparable ranges of values. This is particularly useful when features have very different scales.

=== Example: Predicting House Prices

We have a dataset with two features:
- $x_1$: Size of the house (ranges from 300 to 2000 square feet)
- $x_2$: Number of bedrooms (ranges from 0 to 5)

For an example house:
- Size: 2000 sq. ft.
- Bedrooms: 5
- Price: $500,000$

==== Parameter Selection
1. Initial guess:
    - $w_1 = 50$, $w_2 = 0.1$, $b = 50$
    - Predicted price: $50 \cdot 2000 + 0.1 \cdot 5 + 50 = 100,050,000$, which is far off.
2. Improved guess:
    - $w_1 = 0.1$, $w_2 = 50$, $b = 50$
    - Predicted price: $0.1 \cdot 2000 + 50 \cdot 5 + 50 = 500,000$, a much better estimate.

==== Observation
- Large feature ranges (e.g., $x_1$) often correspond to smaller parameter values (e.g., $w_1 = 0.1$).
- Small feature ranges (e.g., $x_2$) often correspond to larger parameter values (e.g., $w_2 = 50$).

=== Gradient Descent and Contours
When feature scales differ significantly:
1. The cost function contours become elongated ellipses.
2. Gradient descent takes inefficient "zig-zag" paths to reach the global minimum.

By scaling features to similar ranges, the contours become more circular, allowing gradient descent to converge faster.

=== Scaling Techniques

==== Min-Max Scaling
Transform each feature by dividing by its range:
- $x_1 = frac{x_1}{2000}$ (for range $[300, 2000]$)
- $x_2 = frac{x_2}{5}$ (for range $[0, 5]$)

Result:
- Scaled $x_1$ ranges from $0.15$ to $1$
- Scaled $x_2$ ranges from $0$ to $1$

==== Standardization (Z-score Scaling)
Transform each feature to have a mean of 0 and a standard deviation of 1:
\[
z_i = \frac{x_i - \mu_i}{\sigma_i}
\]
Where:
- $\mu_i$: Mean of the feature
- $sigma_i$: Standard deviation of the feature

==== Implementation
#codeBlock(
```python
  import numpy as np

  def min_max_scaling(feature, feature_range):
      return feature / feature_range

  def z_score_scaling(feature, mean, std_dev):
      return (feature - mean) / std_dev

  # Example for x1 and x2
  x1 = np.array([300, 2000])
  x2 = np.array([0, 5])

  x1_scaled = min_max_scaling(x1, 2000)
  x2_scaled = min_max_scaling(x2, 5)

  print(f"Scaled x1: {x1_scaled}")
  print(f"Scaled x2: {x2_scaled}")
)

#figure(
  image("./images/contours-after-scaling.png"),
  caption: [
    Contours after scaling: nearly circular, allowing faster convergence in gradient descent.
  ]
```
)

=== Key Takeaway
Feature scaling significantly improves the efficiency of gradient descent, especially when feature ranges differ greatly. Use either min-max scaling or standardization to normalize features before applying machine learning algorithms.
