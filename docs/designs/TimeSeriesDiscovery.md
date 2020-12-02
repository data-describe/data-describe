# Design Proposal for Sensor Discovery and Insight Generation

## Motivation

A better way to explore time series data especially with use cases like coorelations of analog/digital signals from sensors.

## Challenges

 * will a general UI work with these specific problems
 * can this work within the constraints of the user
 * will the jupyter UI alone be good enough

## Examples

 * With a click of the button I want to automatically:
 * Build a summary report identifying how many rows and columns we have, which columns are continuous/discrete etc.
 * Build an outlier report identifying outliers per sensor.
 * Generate static or interactive time series charts for each continuous sensor
 * Generate scatter plots for pairs of sensors and identify interesting ones.
 * Given a failure/response column run a random forest to identify important variables. 
 * We should have notebooks demonstrating the type of analysis data-describe can do based on type of data (accelerometer data, sensor data, alarm data, work order data etc) - We should prioritize this based on existing/potential cusomers in our sales pipeline.

## Potential Data Sets

Manufacturing:
* Production Line Performance: https://www.kaggle.com/c/bosch-production-line-performance
* CNC Tool Wear: https://www.kaggle.com/shasun/tool-wear-detection-in-cnc-mill
* Steel Defect Detection: https://www.kaggle.com/c/severstal-steel-defect-detection

Transportation:
* Diesel Engine Faults Data: https://data.mendeley.com/datasets/k22zxz29kr/1
* Autonomous vehicles: https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles

Predictive Maintenance on Nasa Turbofan Data: https://towardsdatascience.com/predictive-maintenance-of-turbofan-engines-ec54a083127
Azure predictive maintenance: https://github.com/Azure/PySpark-Predictive-Maintenance

Bearing Fault Accelerometer Data: https://csegroups.case.edu/bearingdatacenter/pages/download-data-file

Earthquake prediction: https://towardsdatascience.com/earthquake-prediction-faffd7160f98

 
 

## Alternatives Considered

