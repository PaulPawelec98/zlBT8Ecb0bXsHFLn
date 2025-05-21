# Apziva Project 3: HR Talent Ranking

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Overview

This project is designed to facilitate HR talent search by leveraging data science techniques. It aims to streamline the process of identifying and evaluating potential candidates through data-driven methodologies.

## Dataset

The repository includes datasets pertinent to HR analytics.

The data comes from sourcing efforts. Any field that could directly reveal personal details has been removed and each candidate is given a unique identifier.

    id : unique identifier for candidate (numeric)
    job_title : job title for candidate (text)
    location : geographical location for candidate (text)
    connections: number of connections candidate has, 500+ means over 500 (text)

The project follows a structured organization to maintain clarity and scalability:

```
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

## Installation

To set up the project environment, follow these steps:

Clone the repository:

```bash
git clone https://github.com/PaulPawelec98/zlBT8Ecb0bXsHFLn.git
cd zlBT8Ecb0bXsHFLn
```

Create and activate a virtual environment:

Using Conda:

```bash
conda env create -f environment.yml
conda activate apzivaproject3
```

## Acknowledgments

Apziva for providing the dataset and project framework.​ The open-source community for their invaluable tools and libraries.​


