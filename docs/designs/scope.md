# Scope Document

The goal of this document is to articulate the scope, motivations, and aspirations for the data describe package.

## Motivation

> "[...exploratory data analysis (EDA) is an approach to analyzing data sets to summarize their main characteristics](https://en.wikipedia.org/wiki/Exploratory_data_analysis)"

data describe aims to accelerate the exploratory data analysis (EDA) process by enabling users to focus on analyzing data

The current Python ecosystem has many open source packages that are commonly used in EDA, with varied components for visualization (such as [matplotlib](https://matplotlib.org/), [Plotly](https://plotly.com/), [bokeh](https://docs.bokeh.org/en/latest/index.html)), tabular data manipulation and summarization ([Pandas](https://pandas.pydata.org/)), or analysis-specific toolkits ([pyLDAvis](https://github.com/bmabey/pyLDAvis), [geopandas](https://residentmario.github.io/geoplot/index.html)). However, few or none of them aim to provide a seamless experience for analyzing data from end-to-end. In practice, it is not uncommon to develop a significant amount of repetitive, boilerplate code to assemble these components into a cohesive analysis. data describe seeks to fill this gap by providing an end-to-end solution for exploratory data analysis

## Guiding Principles

### data describe is **accessible**

data describe should be easy and straightforward to use. A good user experience (UX) should be an integral part of designing the API/UI of data describe. Following the Pareto principle, 80% of common user tasks should be achievable with minimal user configuration by the core 20% functionality

### data describe is **opportunistic**

There are a multitude of ways to analyze data; data describe does not claim to support all of them. data describe instead seeks to prioritize implementing features (analyses) that are widely used

### data describe is **opinionated**

Not all visualizations are created equal, and data describe seeks to avoid those that may be misleading or sub-optimal. For example, while pie charts may be ubiquitous, they do not strongly align with the vision of aiding in exploratory analysis

### data describe is **exploratory**

Exploratory data analysis is one of the first steps in unraveling and understanding complex relationships in data. It is often messy and iterative. data describe does not seek to be a business intelligence tool in which recurring reports are provided

### data describe is **analytical**

While visualization is an important method of understanding data, the creation of the visualization itself is not the end goal. For example, the creation of "infographics", in which the purpose is to present and educate, is not goal of data describe

## UI or API

In addition to specific data analyses features, data describe provides interfaces and convenient features for the purpose of *minimizing entry barriers to performing data analysis*. This may include, but is not limited to:

- Integrations with data loading / processing frameworks, to allow analysis of extremely large datasets
- Integrations with visualization frameworks, to enable different methods of visualizing data which may aid in understanding (e.g. interactive vs static plots)
- Modeling algorithms, where the insights from the model is the focus rather than the pure predictive power


## Supported Data Types: Approach

As an initial launch goal, data describe does not explicitly define industry or domain specific support of data types, but rather focuses on the form of the data and the way in which it is analyzed.

- EDA for classical ML problems (i.e. classification, regression) is most common in the space and least likely to stand out against other tools. Coverage in this area should be focused on providing a polished, "one-stop shop" that can handle the most common analyses
- Other types of data that can be used as a predictive feature in classical ML should be strongly considered. For example, locality (e.g. country, state, city, zip) is very common to include as a predictive feature, while other geospatial-specific analyses (e.g. flight maps) are much less commonly-applicable and may or may not be easily represented in tabular format
- In scoping out data sources, do not conflate the source of data with the analysis approach. For example, signal data and alarm data can framed as generic time series analysis or even classic ML/tabular. Computer vision data can be framed as a correlation heatmap on the raw pixels
- Keep an eye out for opportunities to cover more niche data types where there is a need and is not already covered by other (open source) tools
- Be judicious about how/where we provide "automated data preparation" / transformation for specific data types as this package is not intended to eliminate the need for data preparation. Users are still expected to manipulate their data as needed to prepare it for analysis

## Roadmap

Some features that are on the roadmap for future releases include (in no particular order):

- Additional integrations for big data processing, such as with Spark
- Features/utilities for geospatial mapping
- A "pandas-profiling"-like, single function for rendering an automated, complete report of input data
- Better export utilities for using data-describe visualizations in publications
- Enhanced interactivity with data-describe, possibly via a Jupyter Lab extension

