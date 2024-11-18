#import "../lib/template.typ": main
#import "../lib/simpleTable.typ": simpleTable
#import "../lib/codeBlock.typ": codeBlock

= Multiple Functions in Linear Regression

== Vectorization in Machine Learning

=== Introduction

Vectorization is a key concept in efficient algorithm implementation. It reduces code length and leverages modern hardware like GPUs (Graphics Processing Units) for significant speedups. Here are the main ideas discussed:

- *Vectorization Benefits*:
  - Shortens code for easier readability and maintenance.
  - Accelerates execution by utilizing parallel hardware capabilities.

=== Example: Prediction Without Vectorization

Consider a linear regression model where:
- $ w $ is a vector of parameters.
- $ x $ is a vector of features.
- $ b $ is the bias term.

For $ n = 3 $, the prediction can be written mathematically as:
$ f(x) = \\sum_(j=1)^n w_j \cdot x_j + b $

==== Non-Vectorized Implementation

In Python, a straightforward implementation uses a loop:
#codeBlock(

```python
  f = 0
  for j in range(n):  = Loop from 0 to n-1
      f += w[j] ** x[j]
  f += b
  ```
)

This approach, while functional, becomes inefficient for large $ n $ (e.g., $ n = 100,000 $).

==== Issues:
- Repeated iterations increase computational cost.
- Code readability decreases for larger feature vectors.

=== Efficient Implementation with Vectorization

Using vectorized operations, the prediction formula becomes:
$ f(x) = w \cdot x + b $

==== Vectorized Python Implementation

With NumPy, we can replace the loop with a single line of code:
#codeBlock(

```python
  f = np.dot(w, x) + b  = Vectorized implementation
  ```
)

==== Explanation:
- `np.dot(w, x)` computes the dot product between $ w $ and $ x $.
- Adding $ b $ completes the prediction.

==== Advantages:
1. *Code Compactness*: Reduced to one line.
2. *Execution Speed*: Utilizes hardware parallelism, especially for large $ n $.

=== Behind the Scenes of Vectorization

Vectorized operations leverage optimized libraries (e.g., NumPy) to:
1. Utilize CPU or GPU parallelism for mathematical operations.
2. Avoid redundant computations compared to sequential loops.

==== Practical Comparison
When $ n $ is large, vectorized code is not just shorter but significantly faster. Experimentation shows dramatic performance improvements in execution time.

=== Summary of Key Points

#simpleTable(
  columns: (1fr, 1fr),
  [*Aspect*], [*Details*],
  [*Mathematical Formula*], [
    $ f(x) = sum(j=1)^n w_j \cdot x_j + b $
  ],
  [*Non-Vectorized*], [
    Python loop with `range(n)` iterating through $ w $ and $ x $.
  ],
  [*Vectorized*], [
    `np.dot(w, x) + b` utilizing NumPy for dot product computation.
  ],
  [*Benefits*], [
    Compact code, faster execution, and easier maintenance.
  ],
)

=== Observations and Reflections

- Vectorization feels like a "magic trick" due to its dramatic speedup.
- Understanding how hardware executes vectorized code adds to its appreciation.
- Practical takeaways include consistently seeking opportunities to write vectorized code for efficiency and scalability.
