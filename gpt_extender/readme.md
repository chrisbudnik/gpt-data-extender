# :gear: GPT-Extender Architecture Overview

## :mag: Overview

The `gpt_extender` module is structured to facilitate easy data augmentation and feature extraction from textual data. It comprises several layers: helper methods for direct API interactions, backbone methods for core functionalities, and a specialized class for managing prompt engineering.


## :wrench: Helper Methods

### `_chat_completion` & `_gpt_completion`

These methods serve as the primary interface with the OpenAI API. They handle the execution of the language model queries and receive the output. Additionally, these methods are responsible for monitoring token usage via the internal method `_update_token_usage`, ensuring that you keep track of your API usage.

## :hammer_and_wrench: Backbone Methods

### `gpt_extend` & `chat_extend`

These methods are the core of the module, responsible for applying machine learning understanding to textual data within the DataFrame. They iterate over each row and, based on the specific task, apply either the GPT or chat model to extend the DataFrame horizontally by adding new columns. These methods often use `ExtendTemplate` instances to customize the AI prompts for specific tasks.


## :bulb: Prompt Engineering

### `ExtendTemplate` Class

The `ExtendTemplate` class serves as a utility for shaping the AI prompts used in the backbone methods. It holds crucial information such as the names of the columns to be processed, the context for the AI task, and the required format for the output. This class makes it easier to create customized, reusable prompts for different DataFrame extension tasks.

## :rocket: High-Level Methods

### `add_topic`, `add_sentiment`, etc.

These are specialized methods that make use of pre-defined `ExtendTemplate` objects. They are essentially wrappers around the `chat_extend` method, tailored for common tasks like topic assignment or sentiment analysis.

### Example: `add_topic`

```python
def add_topic(self, column_name: str, new_column_name: str, outputs: list[str]):
    template = ExtendTemplate(column_name=column_name,
                              new_column_name=new_column_name,
                              context="You are a careful reader and researcher.",
                              task="Based on the text, assign a topic.",
                              output=f"Choose from: {outputs}",
                              temperature=0)
    return self.chat_extend(template=template)
```

By understanding this modular architecture, you can more effectively extend or adapt the functionalities of `gpt_extender` for your unique use-cases.


## :clipboard: `DataExtender`` Class: Method List

| Method Name      | Short Description                                                                                          |
|------------------|-----------------------------------------------------------------------------------------------------------|
| `chat_extend`    | Uses a chat model to extend the DataFrame horizontally by creating new columns based on user-defined prompts. |
| `gpt_extend`     | Leverages a GPT model to create new columns in the DataFrame.                                              |
| `synthetic_extend` | Utilizes GPT to extend the DataFrame vertically, adding new records based on existing data.                |
| `add_embeddings` | Adds a column containing embeddings of the text, which can be used for machine learning models.            |
| `add_translation` | Adds a column with translated text, allowing you to specify the source and target language.                 |
| `add_summary`    | Adds a column that contains summarized versions of the text data.                                          |
| `add_topic`      | Adds a new column that categorizes the text based on pre-defined topics.                                   |
| `add_synthetic_data` | Samples existing rows and adds new, synthetic records to extend the DataFrame.                            |
| `search_similarity` | Searches for similar records in the DataFrame based on a query string.                                     |
| `usage_summary`  | Generates a report summarizing the token usage for all API calls made by the class.                         |
| `ai_analyze`     | Uses a chat model to suggest potential extensions based on the existing data and structure.                  |

This table provides a quick reference for the various methods available in the `DataExtender` class, each serving specific data extension or analysis needs.