# Design Proposal for Distributions

## Motivation

Visualizing feature distributions is one of the primary methods of understanding a new dataset.

## Goals

Define the accepted approach for visualizing and/or diagnosing distributions.

## Non-Goals

N/A

## UI or API

Users should have the ability to:
- Create a single plot with specified feature(s)
- Generate all applicable plots (of a particular type) for any applicable data types

In the future, users should be able to:
- Display plots which meet diagnostic criteria (e.g. detected skew, outliers, etc.)
- Interactively select (e.g. using a dropdown) which features to use for plotting

## Design

There are many different ways to visualize distributions, depending on the goal of analysis. This design document attempts to identify the appropriate plot(s) that should be used in each case.

### Plots
- [Histogram](https://seaborn.pydata.org/generated/seaborn.distplot.html): Histograms are probably the most common way to look at a univariate distribution. The selection of bin size may be non-obvious.
- Bar and Line Charts: Bar and line charts are also commonly used to show counts of one feature compared against other features.
- [Kernel Density](https://seaborn.pydata.org/generated/seaborn.kdeplot.html): Kernel density estimate plots can show more fidelity than histograms, but outliers are smoothed out and the kernel size must also be selected.
- [Boxplot](https://seaborn.pydata.org/generated/seaborn.boxplot.html):  Box plots provide specific visual markers about quantiles and outliers.
- [Violin plot](https://seaborn.pydata.org/generated/seaborn.violinplot.html): Violin plots (can) serve the same information as the prior plots. However, the rotated direction and higher information density may be confusing for users who are not familiar with violin plots.
- Other: There are numerous other plots which may be under future consideration, including but not limited to:
    - https://github.com/myrthings/catscatter
    - https://seaborn.pydata.org/generated/seaborn.stripplot.html#seaborn.stripplot
    - https://plotly.com/python/parallel-categories-diagram/

### Recommended Plot by Analysis
| Input                                                    |      Plot                         |
|----------------------------------------------------------|----------------------------------:|
| Single numeric feature                                   |  Histogram & Violin               |
| Single categorical feature                               |    Bar                            |
| Numeric feature compared against categorical feature     | Overlapping KDE & Violin          |
| Categorical feature compared against categorical feature | Stacked or Side-by-side Bar       |
| Numeric feature compared against numeric feature         | WARN: Use scatterplot instead     |
| Feature compared against binary or continuous target     | Overlay Line Estimate<sup>1</sup> |

<sup>1</sup>*Sometimes it is useful to see how the target changes over the range of values in the primary feature. This may be easier to see using a line overlay that shows the expected value i.e. % for binary targets and average for continuous.*

## Alternatives Considered

[Pie Charts](https://plotly.com/python/pie-charts/): [Pie charts can be misleading](https://en.wikipedia.org/wiki/Misleading_graph#Pie_chart) and should not be used in data-describe.

[Scatterplots](https://seaborn.pydata.org/generated/seaborn.scatterplot.html#seaborn.scatterplot): While scatterplots are also a (bivariate) distribution plot, it has been defined as a distinct feature in data-describe, as it has distinct functionality to perform diagnostics ("scagnostics").