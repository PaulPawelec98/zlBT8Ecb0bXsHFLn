# Apziva Project 3 - HR Talent Search

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Overview

This project is designed to facilitate HR talent search processes. It leverages data-driven methodologies to enhance the recruitment workflow, aiming to match candidates effectively with keywords using Natural Language Processing (NLP) and Large Language Models (LLMs).

## Dataset

he data comes from our sourcing efforts. We removed any field that could directly reveal personal details and gave a unique identifier for each candidate.

    id : unique identifier for candidate (numeric)
    job_title : job title for candidate (text)
    location : geographical location for candidate (text)
    connections: number of connections candidate has, 500+ means over 500 (text)


## Project Structure

The repository follows a structured layout to organize code, data, and documentation effectively:

```
    ├── README.md           <- Project overview and instructions.
    ├── apzivaproject3      <- All the Scripts Used to Generate the Results
    │   ├── classes         <- Any custom classes I wrote goes here
    │   ├── dataset         <- Scripts related to cleaning the inital data
    │   ├── functions       <- Scripts containing standalone functions
    │   ├── modeling        <- All model training and predictions
    │   ├── setup           <- Scripts for additional cleaning and feature creation
    ├── docs                <- Project documentation.
    ├── notebooks           <- Jupyter notebooks for exploration and analysis.
    ├── references          <- Reference materials and related resources.
    ├── reports             <- Generated reports and visualizations.
    ├── environment.yml     <- Conda environment specifications.
    ├── requirements.txt    <- Python package dependencies.
    ├── setup.cfg           <- Configuration for package distribution.
    ├── pyproject.toml      <- Build system requirements.
    ├── main.py             <- Main execution script.
```

## Acknowledgments

Apziva for providing the dataset and project framework.​ The open-source community for their invaluable tools and libraries.​