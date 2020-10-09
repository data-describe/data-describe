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

    - :doc:`clustering <data_describe/core/clustering/index>`: Clustering analysis and visualization on a 2D plot
    - :doc:`correlation <data_describe/core/correlation/index>`: Association measures for both numeric and categorical features
    - :doc:`data heatmap <data_describe/core/heatmap/index>`: Data variation and missingness heatmap
    - :doc:`data summary <data_describe/core/summary/index>`: Selected summary (descriptive) statistics
    - :doc:`distribution <data_describe/core/distributions/index>`: Histograms, violin plots, bar charts
    - :doc:`scatter plots <data_describe/core/scatter/index>`: Scatterplots
    - :doc:`feature importance <data_describe/core/importance/index>`: Feature ranking
    - :doc:`time series <data_describe/core/time/index>`: Visualizing time series and other analysis

Example Usage
~~~~~~~~
    The core features (functions) are exported and can be used directly::

        import data_describe as dd
        dd.data_summary(df)

    Non-core features need to be imported explicitly. For example, for text preprocessing::

        from data_describe.text.text_preprocessing import preprocess_texts
        preprocess_texts(df.TEXT_COLUMN)

Extended Features
-----------------
Additional features of data-describe include sensitive data detection (e.g. PII), text
analysis, dimensionality reduction, and more. For more information on using these,
check out the :doc:`Examples <examples/index>` or :doc:`API Reference <data_describe/index>` sections.

.. _`data-describe`: https://github.com/data-describe/data-describe/