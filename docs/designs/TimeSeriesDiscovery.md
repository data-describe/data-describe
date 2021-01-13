# Design Proposal for Sensor Discovery and Insight Generation

## Motivation
Provide support for anomaly detection
## Goals
* Quickly identify and plot anomalies in time series data.
* Create interactive visualizations, such as selecting time window.
* Provide helpful example notebooks.

## Non-Goals
* Automatic visualizations (at the click of a button), such as GUI's. Proposed [here](https://github.com/data-describe/data-describe/blob/master/docs/designs/UI.md)
* Creating notebooks for specific use cases i.e. sensor data, accelerometers, manufacturing.
## UI or API
* 2-D plot of time series data with confidence band. 
* Use markers/colors to highlight anomalies.
* Interactive plots for selecting time window.

## Design
There are multiple design aspects to be considered when creating this functionality.
1. Supervised: Partitioning the data into train and test sets to calculate the confidence bands and error rates.
2. Unsupervised: Training a model on the entire data, i.e. HDBSCAN and isolation forests
3. Statistical methods, i.e. standard deviation from rolling mean and SH-ESD.
```python
import data_describe as dd
dd.anomaly_detection(df) #Autodetect datatype for classification, regression, timeseries

# Returns
# Time series plot with anomalies marked, prediction intervals.
# Dataframe with all the marked records that are anomalies.
```
[Medium Article](https://towardsdatascience.com/anomaly-detection-with-time-series-forecasting-c34c6d04b24a): Contains relevant time series plots and analysis.
## Alternatives Considered
[R Package](https://github.com/twitter/AnomalyDetection): Open source R package developed by twitter for anomaly detection

[FB Prophet](https://towardsdatascience.com/anomaly-detection-time-series-4c661f6f165f): Facebook time series model. Prophet can be used as an estimator in the anomaly detection widget, but would require additional dependencies.
