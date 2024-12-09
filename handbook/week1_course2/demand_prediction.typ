#import "../lib/template.typ": main
#import "../lib/simpleTable.typ": simpleTable
#import "../lib/codeBlock.typ": codeBlock

= Demand Prediction with Neural Networks

== Introduction

Demand prediction is a key application of machine learning that helps businesses optimize inventory, marketing, and sales strategies. This handbook explores how neural networks model demand prediction using concepts from logistic regression and artificial neurons.

---

== Logistic Regression for Demand Prediction

Logistic regression models the probability of an event occurring, such as whether a product becomes a top seller. The model is represented by:

#simpleTable(
  columns: (1fr, 1fr),
  [*Concept*], [*Formula*],
  [*Model Output*], [
    $ a = frac{1}{1 + e^{-(w * x + b)}} $
  ],
  [*Parameters*], [
    $ w $: weight, $ b $: bias
  ],
  [*Activation*], [
    The predicted probability that the product is a top seller
  ],
)

Where:
- $x$ = product feature (e.g., price)
- $a$ = activation (output probability)

---

== Artificial Neurons as Building Blocks

An artificial neuron functions like a simple computer. It receives input features, processes them through an activation function, and outputs a prediction.

**Example:** Predicting whether a T-shirt becomes a top seller:
- Input: Price ($x$)
- Output: Probability of being a top seller ($a$)

---

== Neural Network Structure

Neural networks combine multiple artificial neurons to make complex predictions. Consider a T-shirt demand prediction model with the following features:

#simpleTable(
  columns: (1fr, 1fr),
  [*Feature*], [*Description*],
  [*Price*], [Cost of the T-shirt],
  [*Shipping Cost*], [Cost of delivery],
  [*Marketing Spend*], [Advertising expenses],
  [*Material Quality*], [Perceived fabric quality],
)

**Hidden Layer Neurons:**
1. **Affordability Neuron:** Inputs - Price, Shipping Cost
2. **Awareness Neuron:** Input - Marketing Spend
3. **Quality Perception Neuron:** Inputs - Price, Material Quality

The outputs of these neurons feed into a **Final Neuron**, predicting the overall likelihood of the T-shirt being a top seller.

---

=== Mathematical Representation

For each hidden neuron, the output $a_i$ is calculated as:

$ a_i = frac{1}{1 + e^{-(w_i x_i + b_i)}} $

The final prediction combines these activations:

$ \hat{y} = frac{1}{1 + e^{-(w_f \cdot a + b_f)}} $

Where:
- $a$ = vector of activations from hidden neurons
- $w_f$, $b_f$ = weights and bias of the final neuron


== Summary

- **Logistic Regression:** Models probability using a sigmoid function.
- **Artificial Neurons:** Simple units processing features to produce predictions.
- **Neural Networks:** Combine neurons in layers for complex tasks.
