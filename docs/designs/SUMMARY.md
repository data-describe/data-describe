# Design Proposal for Data Summary

## Motivation

Understanding summary statistics and other commonly representative measures around the fields in a user-provided data frame.

## Goals

Implement the preferred approach for quickly and efficiently calculating all of the decided measures, including functionality for large datasets.

## Non-Goals

Displaying the measures on a visual plot or comparing each field to the others.

## UI or API

The interface is based off the pattern in [#109](https://github.com/brianray/data-describe/pull/109). Current data frame compatibility includes pandas and modin.

## Design

[Pandas DataFrame.agg](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html) method is used in order to calcute the following values for each field:
* Data Type
* Mean
* Standard Deviation
* Median
* Minimum
* Maximum
* Number of zeros
* Number of nulls
* How often most frequent value can be found (as a percentage)

Some are built-in functions, while others are user-defined. Each aggregation is appended to the next, and the final data frame is returned, in the shape of (9, X) where X is the number of fields present in the input data frame.


## Alternatives Considered

Pandas profiling was considered, but the output of this feature is meant to be relatively simple and just a brief overview of each field, as opposed to a full-scale report. Measures such as cardinality and mode were considered, but determined to be redundant information. Future iterations can include (but are not limited to) skew, quartiles, interquartile range, dependence (if target variable provided), and number of statistical outliers.