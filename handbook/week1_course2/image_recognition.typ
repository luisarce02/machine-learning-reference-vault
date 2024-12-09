#import "../lib/template.typ": main
#import "../lib/simpleTable.typ": simpleTable
#import "../lib/codeBlock.typ": codeBlock

= Deep Learning Handbook: Image Recognition

== Neural Networks for Image Recognition

In computer vision, neural networks are used for tasks like facial recognition. A typical input is a grayscale image of size 1000x1000 pixels, represented as a 1000x1000 matrix of pixel intensity values ranging from 0 to 255. These values can be flattened into a vector of one million elements.

#simpleTable(
  columns: (1fr, 1fr),
  [*Concept*], [*Description*],
  [*Input Representation*], [
    X ={1000 x 1000 matrix of pixel intensities}
  ],
  [*Flattened Vector*], [
    X ={1,000,000-element vector}
  ],
)

=== Neural Network Architecture

A neural network processes the image through multiple layers:

1. **Input Layer:** The flattened image vector.
2. **Hidden Layers:** Extract features through learned filters.
3. **Output Layer:** Estimates probabilities of specific identities.


=== Feature Detection in Hidden Layers

Neurons in hidden layers learn different features:

1. **First Hidden Layer:** Detects edges and simple patterns.
2. **Second Hidden Layer:** Combines edges into facial parts.
3. **Third Hidden Layer:** Detects complete facial structures.

#simpleTable(
  columns: (1fr, 1fr),
  [*Layer*], [*Detected Features*],
  [*First Layer*], [Edges and lines],
  [*Second Layer*], [Facial parts like eyes and nose],
  [*Third Layer*], [Complete facial structures],
)


=== Transfer Learning Concept

The same network architecture can adapt to different tasks. If trained on car images, it will learn features like car edges, parts, and full car shapes.
