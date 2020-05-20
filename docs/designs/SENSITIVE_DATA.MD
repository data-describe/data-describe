# Design Proposal for Sensitive Data Detection/Anonymizer

## Motivation

Create additional capability to identify, anoyminze, and visualize sensitive data such as PII, PHI, and Financial data, which would distiguish Data Describe from other competitors.

## Goals

Specific new functionalities or other changes.
- Create an "abstract data schema" for select [infotypes](https://cloud.google.com/dlp/docs/infotypes-reference)
- Identify PII/PHI/Financial data 
- Anomynize the sensitive data synthetically so it retains distributions
- Must work with [issue #6](https://github.com/brianray/data-describe/issues/6). We need be able to handle large datasets using a distributed framework. 


## Non-Goals
We do not intend to create synthetic data from scratch and will be utlizing open-sourced packages.

## UI or API

Presidio and faker have support for Python 3.7.

## Design

To identify these sensitive data, we can utilize [Presidio](https://github.com/microsoft/presidio). Presidio also supports some level of customizable anonymization. 
Another package is [Faker](https://faker.readthedocs.io/en/master/), which is able to generate fake data.

## Alternatives Considered

Other packages looked at are [Trumania](https://github.com/RealImpactAnalytics/trumania), [scrubadub](https://scrubadub.readthedocs.io/en/stable/index.html#), and [PIIdetect](https://github.com/edwardcooper/piidetect). These were not chosen because they don't have much popularity and are not as mature as Presidio and Faker.