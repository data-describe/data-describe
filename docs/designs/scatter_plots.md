# Design Proposal for Scatter Plots

## Motivation

Scatter plots are commonly used in EDA. While scatter plot matrices are a commonly used method of displaying scatter plots, it can also result in unwieldy, massive plots that are difficult to read. This design document aims to identify and formalize alternative or preferred approaches for visualizing x-y data.

## Goals

Identify the preferred approach for performing EDA using scatter plots.

## Non-Goals

Designing the interactive UI or choosing framework implementation.

## UI or API

The interface will be based off the pattern in [#109](https://github.com/brianray/data-describe/pull/109).

## Design

### Pre-processing

One method of making scatter plots applicable to large datasets is to make use of [hexagonal bins](https://matplotlib.org/2.1.1/gallery/statistics/hexbin_demo.html).


### Visualization Modes

The scatter plot feature in data describe should implement several modes:

#### Scatter plot matrix
A scatter plot matrix is a very common way for viewing scatter plots of all numeric variables. Some libraries (e.g. [Seaborn pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html)) also add in univariate distributions on the unit diagonal. This can be hard to read when there are many features and is also repetitive since the plots are mirrored. However, this should be left in as an option in data describe due to its popularity.

#### Scagnostic plots
Scatter plot diagnostics ("scagnostics") can provide a mechanism for reducing quantity of plots to those that are "interesting" by some measure. This can make it easier for users to focus on reviewing plots that stand out. Examples include "outlying", "skewed", "sparse" etc. ([Reference](https://www.cs.uic.edu/~wilkinson/Publications/sorting.pdf))

#### Interactive dashboard
A dashboard-like user interface can be used to toggle/swap through scatter plots of each pair of columns. This interface can also borrow elements from the other modes, such as providing highlight/indicators of "interesting" plots from scagnostics. While a dashboard UI is ideal for exploratory analysis, it is difficult to envision how this would translate to static reporting.

#### Singular plots
Singular plots, automatically generated for each pair of numeric columns, may be desired for the purpose of static reports.


## Alternatives Considered

Some other packages link the interactive scatter plotting with other features such as correlation matrix, such that hovering over a tile in the correlation will show the corresponding scatter plot. While this type of functionality could be considered in a future UI for data describe, this design document only focuses on scatter plots as a standalone feature.
