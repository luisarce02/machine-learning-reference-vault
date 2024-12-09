#import "../lib/template.typ": main
#import "../lib/simpleTable.typ": simpleTable
#import "../lib/codeBlock.typ": codeBlock

= Forward Propagation in Neural Networks

== Introduction

Forward propagation is the process of passing input data through a neural network to make predictions or inferences. It involves sequential calculations from the input layer through hidden layers to the output layer.

== Example: Handwritten Digit Recognition

Consider a binary classification problem distinguishing handwritten digits "0" and "1." We use a neural network with:

- **Input Layer:** An 8x8 grid (64 features).
- **Hidden Layers:** Two layers with 25 and 15 neurons, respectively.
- **Output Layer:** A single neuron indicating the probability of the digit being "1."

=== Step 1: From Input to First Hidden Layer

The first hidden layer calculates activations \(a^{(1)}\) using:

$ a^{(1)} = sigma(W^{(1)} x + b^{(1)}) $

Where:

- \(x\): Input features (64 pixels).
- \(W^{(1)}\): Weights connecting input to the first hidden layer.
- \(b^{(1)}\): Bias terms.
- \(sigma\): Activation function (e.g., ReLU or sigmoid).

#simpleTable(
  columns: (1fr, 1fr),
  [*Layer*], [*Calculation*],
  [Input], $x = a^{(0)}$,
  [Hidden Layer 1], $a^{(1)} = sigma(W^{(1)} x + b^{(1)})$,
)

=== Step 2: From First to Second Hidden Layer

Activations for the second hidden layer are calculated as:

$ a^{(2)} = sigma(W^{(2)} a^{(1)} + b^{(2)}) $

Where:

- \(W^{(2)}\): Weights connecting the first and second hidden layers.
- \(b^{(2)}\): Bias terms.

=== Step 3: Output Layer

The final prediction is computed using:

$ a^{(3)} = sigma(W^{(3)} a^{(2)} + b^{(3)}) $

Where \(a^{(3)}\) represents the predicted probability.

#simpleTable(
  columns: (1fr, 1fr),
  [*Layer*], [*Calculation*],
  [Hidden Layer 2], $a^{(2)} = sigma(W^{(2)} a^{(1)} + b^{(2)})$,
  [Output Layer], $a^{(3)} = sigma(W^{(3)} a^{(2)} + b^{(3)})$,
)

=== Summary

- **Forward Propagation:** Sequentially computes activations from input to output.
- **Use Case:** Inferencing using pre-trained models.
