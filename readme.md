# GPT-Data-Extender

## Overview

GPT-Data-Extender is a Python library designed to aid data scientists and analysts in enriching their datasets with the help of AI-generated content. This tool leverages state-of-the-art language models to automate a range of Natural Language Processing (NLP) tasks, saving time and increasing efficiency.

## Use Case

In today's world, data comes in various shapes and forms. However, incomplete or sparse datasets can often limit the analytical capabilities of researchers and professionals alike. The primary use case for GPT-Data-Extender is to address these limitations by supplementing datasets with AI-generated content. For example:

- Imagine working with customer reviews but lacking information on sentiment or relevant topics. GPT-Data-Extender can automatically generate these additional columns.
- If you have a dataset of product names but need detailed descriptions, this tool can facilitate that as well.

## Main Features

### Column Extension

Easily generate new columns in your dataset based on custom AI prompts. For instance, if you have a column of company names, you can use GPT-Data-Extender to generate another column that contains brief descriptions or histories of these companies.

### NLP Tasks Automation

GPT-Data-Extender supports a host of NLP tasks like:

- Topic Detection
- Sentiment Classification
- Summarization
- And much more!

## How it works?

### DataExtender Class

The `DataExtender` class is the workhorse for extending the DataFrame. The class requires that the DataFrame contains at least one column of text data. Upon initialization, you can use various extension methods to add new columns with processed information.

- **Sentiment Analysis**: Adds a column that contains sentiment scores or labels like 'positive', 'neutral', 'negative'.
- **Translation**: Adds a column with translated text. You can specify the source and target language.
- **Topic Recognition**: Adds a column with recognized topics or categories related to the text.
  
### ExtendTemplate Class

The `ExtendTemplate` class allows users to customize the prompts for extension methods. By defining the details of a prompt, you can then use the `build` method to construct a query that the `DataExtender` methods can use to process the DataFrame.


## Repo Structure

- `gpt_extender/`: This folder contains all the source code.
  - Files for DataExtender class, ExtendTemplate class, etc.
- `example.py`: A script that demonstrates the use-cases and how to interact with the functionalities.
- `requirements.txt`: The file contains all the dependencies needed to run the code.

## Installation

To install the required packages, you can run:

```bash
pip install -r requirements.txt
```

## Examples

For a full example, check out `example.py`.

## Project Status

This project is currently in its early stages, and contributions or suggestions for improvement are highly welcomed.