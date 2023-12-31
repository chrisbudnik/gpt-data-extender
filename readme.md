# :books: GPT-Data-Extender

## :mag_right: Overview

GPT-Data-Extender is a Python library designed to aid data scientists and analysts in enriching their datasets with the help of AI-generated content. This tool leverages state-of-the-art language models to automate a range of Natural Language Processing (NLP) tasks, saving time and increasing efficiency.

## :bulb: Use Case

In today's world, data comes in various shapes and forms. However, incomplete or sparse datasets can often limit the analytical capabilities of researchers and professionals alike. The primary use case for GPT-Data-Extender is to address these limitations by supplementing datasets with AI-generated content. For example:

- Imagine working with customer reviews but lacking information on sentiment or relevant topics. GPT-Data-Extender can automatically generate these additional columns.
- If you have a dataset of product names but need detailed descriptions, this tool can facilitate that as well.

## :star: Main Features

### Column Extension

Easily generate new columns in your dataset based on custom AI prompts. For instance, if you have a column of company names, you can use GPT-Data-Extender to generate another column that contains brief descriptions or histories of these companies.

### NLP Tasks Automation

GPT-Data-Extender supports a host of NLP tasks like:

- Generate ai suggestions for data extension
- Apply embeddings and perform similarity search
- Topic Detection
- Sentiment Classification
- Summarization
- And much more!

### Adding synthetic data
The add_synthetic_data method first samples existing rows from the specified text column, which then guide the OpenAI model in generating new, contextually relevant entries. These synthetic records are added back to the original DataFrame, enriching it effectively.


## :hammer_and_wrench: How it works?

### DataExtender Class

The `DataExtender` class is the workhorse for extending the DataFrame. The class requires that the DataFrame contains at least one column of text data. Upon initialization, you can use various extension methods to add new columns with processed information.

- **Sentiment Analysis**: Adds a column that contains sentiment scores or labels like 'positive', 'neutral', 'negative'.
- **Translation**: Adds a column with translated text. You can specify the source and target language.
- **Topic Recognition**: Adds a column with recognized topics or categories related to the text.
  
### ExtendTemplate Class

The `ExtendTemplate` class allows users to customize the prompts for extension methods. By defining the details of a prompt, you can then use the `build` method to construct a query that the `DataExtender` methods can use to process the DataFrame.

### Architecture
For more detailed and design focused overview please check technical docs: [technical-readme](gpt_extender/readme.md)

## :film_projector: Functionality Walkthrough

### Importing Libraries and Initialize DataExtender
Begin by importing the necessary libraries and initializing the `DataExtender` with your DataFrame.

```python
import pandas as pd
from gpt_extender import DataExtender, ExtendTemplate

df = pd.read_csv("reviews.csv")
extender = DataExtender(df)
```

### Adding Synthetic Data
If you need to extend your dataset with synthetic data, use the `add_synthetic_data` method. The following code snippet will add 30 new rows to the "reviews" column.

```python
extender.add_synthetic_data(column_name="reviews", output_size=30)
```

### Performing Topic Recognition
Utilize the `add_topic` method to categorize the text in your DataFrame. The method adds a new column that labels each text entry based on specified topics. In this example, we add a new column named "category" that categorizes the reviews into "product", "service", or "other"

```python
extender.add_topic(column_name="reviews", new_column_name="category", outputs=["product", "service", "other"])
```

### Implementing Custom Extensions
For more specialized tasks, you can define a custom template using the `ExtendTemplate` class and then extend your data using the `chat_extend` method. 

```python
template = ExtendTemplate(column_name="reviews",
                          new_column_name="sentiment",
                          context="You are a specialist in text review sentiment recognition.",
                          task="Based on the provided review, evaluate the sentiment.",
                          output="Format the response as one word. True if positive, False if negative or neutral.")

extender.chat_extend(template=template)
```

### Adding Embeddings and Similarity Search

Add text embeddings to your DataFrame and then perform a similarity search based on a query string.

```python
# Add embeddings to the 'reviews' column
extender.add_embeddings(column_name="reviews")

# Perform a similarity search using cosine similarity
search_results: pd.DataFrame = extender.search_similarity(embedding_column="reviews_embedding", 
                                                          query="There was a problem with service.", 
                                                          n_best_results=5)
search_results.head()
```

### Token Usage Summary

Check the summary of token usage for the API calls made by the DataExtender class.

```python
# Get a summary of token usage
extender.usage_summary()
```

### AI-Based Data Extension Suggestions

Leverage the AI model to suggest potential ways you could extend your DataFrame.

```python
# Suggest potential extensions for the 'reviews' column
extender.ai_analyze(column_name="reviews")
```

## :file_folder: Repo Structure

- `gpt_extender/`: This folder contains all the source code.
- `example.py`: A script that demonstrates the use-cases and how to interact with the functionalities.
- `requirements.txt`: The file contains all the dependencies needed to run the code.
- `walkthrough.py`: A script that includes walkthrough code.

## :floppy_disk: Installation

To install the required packages, you can run:

```bash
pip install -r requirements.txt
```

## :computer: Examples

For a full example, check out `example.py` or `walkthrough.py`.

## Project Status

This project is currently in its early stages, and contributions or suggestions for improvement are highly welcomed.


