# Design Proposal for GUI/UI/UX

## Motivation

Motivation is having some GUI for interacting with Data Describe components. 

## Goals

The goal is to allow users:

 * Visiblity to what is available in Data Describe
 * Easy Access to output form Data Describe
 * A manner to display dashboard like functionality

## Non-Goals

There is no plan to create a Desktop native app.

## Design

The design will be broken into 2 parts:

1. Data Science Working View
2. Report View

#### 1. Data Science Working View

For this we will write custom jupyterlab extension. It should simply allow two following things:

- capture of graphs and output into a seperate jupyter lab window
- a menu for code shortcuts and to make it easy to navigate what is available. Also provide additional helpers similar to the contextual helper today.

An example of this may be found in [Dask Lab Extension](https://www.npmjs.com/package/dask-labextension)

#### 2. Report View

This is a more fullblown EDA report dashboard. For this we will create a seperate dashboard that collects all meaningful report elements and allows them to be displayed in a seperate app. **This part 2 is defered at this time** and Part 1 will be the focus. Most likely we will want to consider the Alternatives listed at that time. Likely candidates are Dash.


## Alternatives Considered

We looked at the following, and for reasons listed we did not use at this time. This doc is this very helpful comparision https://panel.holoviz.org/Comparisons.html

 * Panel https://panel.holoviz.org/ plotly library agnostic and is based on Bokeh widgets.
 * Dash is more about standalone, more scalable but require dashboards to be rewritten.
 * ipywidgets native to jupyter and highly customizable.
 * Viola an alternative to Bokeh server can deploy these other apps.
 * streamlit is alternative to the jupyter aware methods above.
 
 Other notable libraries
 
 * https://deck.gl/#/showcases/overview does some really nice data science modeling using WebGL
 * Facets (Which is used by TFDV) provides very useful ML visualizations https://pair-code.github.io/facets/ https://github.com/PAIR-code/facets  The visualizations are implemented as [Polymer web components](https://www.polymer-project.org/), backed by [Typescript](https://www.typescriptlang.org/) code and can be easily embedded into Jupyter notebooks or webpages.
 


 
