#import "../lib/template.typ": main
#import "../lib/simpleTable.typ": simpleTable
#import "../lib/codeBlock.typ": codeBlock

= Deep Learning Handbook

== Neural Network Layers

Neural networks consist of interconnected layers of neurons. Each layer processes inputs and produces activations that are passed to the next layer.

=== Structure of a Neural Network Layer

A neural network layer takes inputs, processes them, and produces outputs through a set of neurons. Each neuron performs a weighted sum of the inputs, adds a bias, and applies an activation function.

#simpleTable(
  columns: (1fr, 1fr),
  [*Element*], [*Definition*],
  [*Input Vector*], [
    $ x = [x_1, x_2, ..., x_n]^T $
  ],
  [*Parameters*], [
    Weights $w_i$ and bias $b$
  ],
  [*Neuron Output*], [
    $ z = w^T x + b $
  ],
  [*Activation Function*], [
    $ a = g(z) = frac{1}{1 + e^{-z}} $
  ],
)

=== Example Calculation

Consider a hidden layer with three neurons and four input features:

1. First neuron:
   $ z_1 = w_1^T x + b_1 $, $ a_1 = g(z_1) $

2. Second neuron:
   $ z_2 = w_2^T x + b_2 $, $ a_2 = g(z_2) $

3. Third neuron:
   $ z_3 = w_3^T x + b_3 $, $ a_3 = g(z_3) $

The activations are passed as inputs to the next layer.

=== Notation and Layer Indexing

To distinguish layers, use superscripts:

- Layer 1 parameters: $w^{[1]}, b^{[1]}$
- Activations: $a^{[1]} = [a_1, a_2, a_3]^T$

The output of layer 1 becomes the input for layer 2.

=== Summary

- Each layer performs weighted sums and applies activation functions.
- Outputs of one layer become inputs to the next.
- Notation uses superscripts to indicate layer-specific parameters and activations.