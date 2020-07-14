# Design Proposal for Dimensionality Reduction

## Motivation

In order to overcome the curse of dimensionality, many data scientists must apply dimensionality reduction to their real-world problem set which can contain too many features. This feature would make that process simple, with the user only entering their dataset and some parameters in order to obtain a dimensionally-reduced dataset.

## Goals

Implement a concise and easy-to-use feature where a user can reduce dimensions by just inputting a dataset, choosing a number of output features, and specifying a dimensionality reduction method.

## Non-Goals

Creating new dimensionality reduction methods or inferring which dimensionality reduction methods work best for the user-provided dataset.

## UI or API

The interface is based off the pattern in [#109](https://github.com/brianray/data-describe/pull/109). Current data frame compatibility includes pandas and modin.

## Design

Functions have been created around 4 different methods of dimensionality reduction:
* PCA
* Incremental PCA
* t-SNE
* Truncated SVD

The user inputs their data frame, the number of components they want the data reduced to, and the method of dimensionality reduction they want applied into the main function. Users are highly encouraged to use Incremental PCA for very large data frames. The output is a tuple consisting of the dimensionally-reduced data frame and the reductor which was applied and fitted to it.

## Alternatives Considered

Linear Discriminant Analysis is another dimensionality reduction method similar to PCA, but does not optimize based on explained variance. Autoencoder methods use neural networks to reduce dimensionality, but therefore require a large amount of data and offer limited explainability. Future enhancements can include providing explainability metrics if possible (i.e. explained variance) and including an option to plot the output data as a visual upon completion of the dimensionality reduction.
