#import "../lib/template.typ": main
#import "../lib/simpleTable.typ": simpleTable
#import "../lib/codeBlock.typ": codeBlock

= Feature Scaling

== Feature Engineering and its Importance

When designing a machine learning model, the choice of features can significantly impact the model's performance. Selecting or designing the right features is often a critical step in improving the accuracy of predictions. Below are the key takeaways from the example discussed about predicting house prices:

=== Example: Predicting House Prices

Let's assume we have two initial features for a house:

#simpleTable(
  columns: (1fr, 2fr),
  [*Feature*], [*Description*],
  [$x_1$], [The width of the lot (frontage of the property)],
  [$x_2$], [The depth of the lot (rectangular assumption)],
)

A simple model can be expressed as:

$ f_(w,b)(x) = w_1 x_1 + w_2 x_2 + b $

While this model may work, there is an alternative approach that leverages domain knowledge.

=== Creating a New Feature: Lot Area

Recognizing that the *area* of the lot (width × depth) may better predict the house price, we define a new feature:

$ x_3 = x_1 times x_2 $

The updated model now incorporates this additional feature:

$ f_(w,b)(x) = w_1 x_1 + w_2 x_2 + w_3 x_3 + b $

Where:

- $w_1$, $w_2$, and $w_3$ are weights the model can optimize.
- The new feature, $x_3$, allows the model to consider whether frontage, depth, or lot area is most predictive.

This process of creating new features is called *feature engineering*. By transforming or combining existing features, we can enable the model to make more accurate predictions.

=== Benefits of Feature Engineering

1. *Leverages domain knowledge*:
   By incorporating intuition about the problem (e.g., lot area is a stronger predictor), we can design more effective features.

2. *Improves model performance*:
   New features can capture relationships that may not be apparent with the original set of features.

3. *Enables flexibility in modeling*:
   Allows fitting non-linear relationships, not just straight lines, by introducing new combinations or transformations of features.

=== Engineering Features for Non-Linear Data

Feature engineering is not limited to linear models. By creating new features or applying non-linear transformations, we can allow models to fit curves or other complex relationships in the data.

*Note for the next lesson*: The subsequent video will explore how to incorporate non-linear functions into models effectively.
