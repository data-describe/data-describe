..

Introduction
============================================
data-describe is a Python toolkit for Exploratory Data Analysis (EDA).
It aims to accelerate data exploration and analysis by providing automated
and opinionated analysis widgets.

Main Features
-------------
The main features of data-describe are organized as the "core". These features are 
expected to be commonly used with most EDA applications on tabular data:

    - cluster: Clustering analysis and visualization on a 2D plot
    - correlation_matrix: Association measures for both numeric and categorical features
    - data_heatmap: Data variation and missingness heatmap
    - data_summary: Selected summary (descriptive) statistics
    - distribution: Histograms, violin plots, bar charts
    - scatter_plots: Scatterplots
    - importance: Feature ranking
    - plot_time_series: Visualizing time series and other analysis

Example Usage
~~~~~~~~
    The core features are exported and can be used directly::

        import data_describe as dd
        dd.data_summary(df)

Extended Features
-----------------
Additional features of data-describe include sensitive data detection (e.g. PII), text
analysis, dimensionality reduction, and more. For more information on using these,
check out the :doc:`Examples <examples/index>` or :doc:`API Reference <data_describe/index>` sections.

.. _`data-describe`: https://github.com/data-describe/data-describe/