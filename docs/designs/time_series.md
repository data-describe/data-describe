# Design Proposal for Time Series Analysis

## Motivation

Data scientists doing exploratory data analysis should be able look at a univariate, time series, plot of a response or dependent variable with respect to a date column from the same dataset. They should be able to see a decomposition of the variable into trend, seasonality, cyclicity and any residual effect after their removal. They should also be able to do smoothing and other transforms of the variable.

Another common use case for time series is determining training and evaluation periods. Cross validation should only be done on training examples from the past, or as near in the past as possible, to maintain similarity to testing examples. Data Describe should help the user pick appropriate training, validation and testing periods when there is a time dimension to the model.

## Short-Term Goals

Automatically detect date columns. Use date column as an index.

Overview of most time-series methods available in Python:
[Time Series in Python Part 1](https://towardsdatascience.com/time-series-in-python-exponential-smoothing-and-arima-processes-2c67f2a52788)
[Time Series in Python Part 2](https://towardsdatascience.com/time-series-in-python-part-2-dealing-with-seasonal-data-397a65b74051)

All of the following can be made available either through [pandas](https://pandas.pydata.org/) or the [statsmodels](https://www.statsmodels.org/stable/index.html) package:

* Decomposition of a time series
    * Trend
	* Seasonality
	* Cyclicity
	* Residuals

* Transforms
	* De-trending: removing the underlying trend in the series
	* Differencing: substracting periodical values
	* Logging: linearizing a series with an exponential trend by taking the log. Not meant for price index variables.
	* Other time-based aggregations

* Smoothing methods
	* Plotting Rolling statistics
	* Simple exponential smoothing
	* Holt-Winters

## Mid-Term Goals

Since the following are also available in statsmodels, they can be implemented after the above are complete.

* Statistical tests for stationairity
	* Dickey-Fuller Test
	* KPSS

* Autocorrelation/Partial Autocorrelation
	* ARIMA
	* Future predictions 

## Long-Term or Non-Goals

Most of the following techniques require a sophisticated understanding of time-series to begin with. We may be able to plot and do simple predictions and give some general direction, but the user should look into the specific packages and do their own research.

[Time Series in Python Part 3](https://towardsdatascience.com/time-series-in-python-part-3-forecasting-taxi-trips-with-lstms-277afd4f811)

Bayesian forecasting using LSTM and RNN.

## UI or API

We should incorporate date columns into the existing code that auto detects column types. We would need to add plots that at the very least plot the response variable against the date variable and output the raw univariate graph as well as decompositions into trend, seasonality, cyclicity and residual. This can be a simple function call as described below to output to a notebook.

By default, pandas objects are time zone unaware. When autodetecting datetime columns, Data Describe should warn the user if a time zone is not set for the column and either set it to the local time zone explicitly or use UTC time. Outputs of datetime columns should always have the time zone set.

## Design

Add datetime vars to
```python
guess_dtypes(df)

{'loan_default': 'Boolean',
	...
 'earliest_credit_line': 'DateTime', # <----
 'payer_code': 'Category',
  ...}
```

Add calls to plot the response or other variable against a time dependent variable. Graph the raw time series variable along with any trend, seasonality or cyclic properties as well as a rolling average. (See part 1 link above for examples.) All of these should have user selectable options.

```python
plot_time_series(y_var='loan_default', date_time_var='earliest_credit_line', smoothing='Holt-Winters', rolling_avg='7_days')
```

## Alternatives Considered

[Lubridate](https://www.r-graph-gallery.com/time-series.html) R-package, not Python.
[Facebook Prophet](https://facebook.github.io/prophet/) This is a good package to include at a later date.
[Pytorch](https://pytorch.org/) For the Bayesian Neural Network example above. Sophisticated example for advanced users, not for EDA.