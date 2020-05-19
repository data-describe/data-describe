> Note to proposers: Please keep this document as brief as possible, preferably not more than two pages.

# Design Proposal for Working with Big Data (Large Datasets)

## Motivation

This package can stand out from other tools if it can handle large datasets, including those that may not fit into memory. We should interface with commonly used frameworks for parallelizing or distributing compute.

## Goals

This package should have the ability to interface with distributed computing frameworks.

## Non-Goals

This package does not seek to implement or orchestrate the compute, but will utilize the framework brought by the end user, if it is supported.

## UI or API

It is expected that most Python end users are familiar with the Pandas interface. This should be considered the standard interface to encourage user adoption.

## Design

There are two primary interface points where distributed computing may apply to exploratory analysis and visualization:

1. When summary statistics, aggregations, or other transformations are required on the entire dataset

2. When the data must be down-sampled in order to render as a visualization

This package will expect the "data frame" object, provided by the user, to implement a Pandas-like interface to execute these computational tasks.

### Pandas

Pandas will be the a hard dependency for this package.

### Modin

[Modin](https://github.com/modin-project/modin) claims to implement the Pandas DataFrame API on top of Dask, Ray, and "bring-your-own" backends. Modin should be an optional dependency for this package.

#### Backends

*Dask* is a commonly used parallel computing framework.

*Ray* can be used with [the Apache Arrow format](https://arrow.apache.org/blog/2017/10/15/fast-python-serialization-with-ray-and-arrow/).

#### Coverage

Modin is still a fairly new project. It claims to have 80%+ coverage of the Pandas DataFrame API.


## Alternatives Considered

### Koalas

[Koalas](https://github.com/databricks/koalas) claims to implement the Pandas DataFrame API on top of Apache Spark. While Koalas integration may be considered as a future enhancement, we should get more overall coverage by utilizing Modin at this time.

#### Coverage

As of last year, [coverage of the Pandas DataFrame API was reported to be 60%+.](https://www.slideshare.net/databricks/koalas-pandas-on-apache-spark)

### cuDF

[cuDF](https://github.com/rapidsai/cudf) claims to implement the Pandas DataFrame API on top of GPUs using the Apache Arrow format. While integration may be considered as a future enhancement, GPU support for data describe is not considered as a high priority.