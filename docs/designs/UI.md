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

The design will be a custom jupyterlab extension.

An example of this may be found in [Dask Lab Extension](https://www.npmjs.com/package/dask-labextension)

## Alternatives Considered

We looked at the following, and for reasons listed we did not use at this time.

 * Panel https://panel.holoviz.org/ plotly library agnostic and is based on Bokeh widgets.
 * Dash is more about standalone, more scalable but require dashboards to be rewritten.
 * ipywidgets native to jupyter and highly customizable.
 * Viola an alternative to Bokeh server can deploy these other apps.
 * streamlit is alternative to the jupyter aware methods above.
 
 Other notable libraries
 
 * https://deck.gl/#/showcases/overview does some really nice data science modeling using WebGL
 * Facets (Which is used by TFDV) provides very useful ML visualizations https://pair-code.github.io/facets/ https://github.com/PAIR-code/facets  The visualizations are implemented as [Polymer web components](https://www.polymer-project.org/), backed by [Typescript](https://www.typescriptlang.org/) code and can be easily embedded into Jupyter notebooks or webpages.
 
 This is more for full blown dashboard. Does have very nice methods for integration of more advanced graphs and widgets https://deck.gl/#/showcases/overview They do this very helpful comparision https://panel.holoviz.org/Comparisons.html

 
