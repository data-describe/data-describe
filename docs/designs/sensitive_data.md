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

Users declare as input arguments, how their data should be handled. The focus is on identification, redaction, and anonymization of sensitive data. 

### Example
```python
# import custom functions
from sensitive_data import identify_pii, anonymize_data, redact_data


# identify_pii: search for columns containing infotypes
# returns dictionary {infotype:<column name>}
sensitive_columns = identify_pii(data=df, automatic=True, get_columns=True)
sensitive_columns = identify_pii(data=df, infotype=['email', 'address'], automatic=False)

# redact_data: returns dataframe with redacted data 
redacted_df = redact_data(data=df, automatic=True)
redacted_df = redact_data(data=df, infotype=['email', 'credit card'], automatic=False)

# anonymize_data: returns dataframe with anonymized data
anonymize_df = anonymize_data(data=df, automatic=True)
anonymize_df = anonymize_data(data=df, infotype=['email', 'name'], automatic=False)

```


## Design

New functionalities are as follows:

 
1. Identification of sensitive data
    - Provide both automatic detection or specific searches upon request
2. Custom identification
    - For sensitive data types not supported, users should have an interface to create their own class or type of sensitive data.
3. Anonymization of data 
    - Once sensitive data is identified, users should have the ability to anonymize their data while maintaining distributions.
4. Handle large datasets
    - A Dataframe, as the input, can be processed in parallel using *modin*, which is _compatible with existing pandas code.

### Presidio
*[Presidio](https://github.com/microsoft/presidio)* is a context aware and customizable PII anonymization service for text and images. 

### Faker
*[Faker](https://faker.readthedocs.io/en/master/)* is a python package that generates fake data. 

### Modin
*[Modin](https://github.com/modin-project/modin)* is a distributed dataframe library that scales up your pandas workflow.

## Alternatives Considered

Other packages looked at were [Trumania](https://github.com/RealImpactAnalytics/trumania), [scrubadub](https://scrubadub.readthedocs.io/en/stable/index.html#), and [PIIdetect](https://github.com/edwardcooper/piidetect). These were not chosen because they don't have much popularity and are not as mature as Presidio and Faker.