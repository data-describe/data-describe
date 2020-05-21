# Design Proposal for Sensitive Data Detection/Anonymizer

## Motivation

Create additional capability to identify, anonymize, and visualize sensitive data such as PII, PHI, and Financial data, which would distinguish Data Describe from other competitors.

## Goals

Specific new functionalities or other changes.
- Create an "abstract data schema" for select [infotypes](https://cloud.google.com/dlp/docs/infotypes-reference)
- Identify PII/PHI/Financial data 
- Anonymize the sensitive data synthetically so it retains distributions
- Must work with [issue #6](https://github.com/brianray/data-describe/issues/6). We need be able to handle large datasets using a distributed framework. 


## Non-Goals
We do not intend to create synthetic data from scratch and will be utlizing open-sourced packages.

## UI or API

Users declare as input arguments, how their data should be handled. The data returns in the same format entered. For example, if the input is dataframe or a list of text, the output should be the same type. 

## Design

New functionalities are as follows:

 
1. Identification of sensitive data
    - Provide both automatic detection or specific searches upon request
2. Custom identification
    - For sensitive data types not supported, users should have an interface to create their own class or type of sensitive data.
3. Anonymization of data 
    - Once sensitive data is identified, users should have the ability to anonymize their data while maintaining distributions.
4. Handle large datasets
    - A Dataframe, as the input, can be processed in parallel using *modin*, which is compatible with existing pandas code.

### Presidio
*[Presidio](https://github.com/microsoft/presidio)* is a context aware and customizable PII anonymization service for text and images. 

### Presidio
*[Faker](https://faker.readthedocs.io/en/master/)* is a python package that generates fake data. 

### Modin
*[Modin](https://github.com/modin-project/modin)* is a a distributed dataframe library that scales up your pandas workflow

## Alternatives Considered

Other packages looked at are [Trumania](https://github.com/RealImpactAnalytics/trumania), [scrubadub](https://scrubadub.readthedocs.io/en/stable/index.html#), and [PIIdetect](https://github.com/edwardcooper/piidetect). These were not chosen because they don't have much popularity and are not as mature as Presidio and Faker.