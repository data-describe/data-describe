..

Package overview
============================================
The current Python ecosystem has many open source packages that are commonly used in EDA, however, few or none of them aim to provide a seamless experience for analyzing data from end-to-end. `data-describe`_ seeks to fill this gap by providing a holistic solution for exploratory data analysis.

Core Design Principles
======================

data describe is *accessible*
-------------------------------

data describe should be easy and straightforward to use. A good user experience (UX) should be an integral part of designing the API/UI of data describe. Following the Pareto principle, 80% of common user tasks should be achievable with minimal user configuration by the core 20% functionality.

data describe is *opportunistic*
----------------------------------

There are a multitude of ways to analyze data; data describe does not claim to support all of them. data describe instead seeks to prioritize implementing features (analyses) that are widely used.

data describe is *opinionated*
--------------------------------

Not all visualizations are created equal, and data describe seeks to avoid those that may be misleading or sub-optimal. For example, while pie charts may be ubiquitous, data describe does not consider them to be suitable for exploratory analysis.

data describe is *exploratory*
--------------------------------

Exploratory data analysis is one of the first steps in unraveling and understanding complex relationships in data. It is often messy and iterative. data describe does not seek to be a business intelligence tool in which recurring reports are provided.

data describe is *analytical*
-------------------------------

While visualization is an important method of understanding data, the creation of the visualization itself is not the end goal. For example, the creation of "infographics", in which the purpose is to present and educate, is not goal of data describe.

.. _`data-describe`: https://github.com/data-describe/data-describe/