import os
import pandas as pd
from gpt_extender import DataExtender, ExtendTemplate


# Creating a list of 5 reviews
reviews_list = [
    "The product is excellent.",
    "Would not recommend, poor quality.",
    "Service was okay, could be better.",
    "Great experience!",
    "It's alright, not the best but not the worst."
]

# Creating the DataFrame
df = pd.DataFrame({'reviews': reviews_list})

# Displaying the DataFrame
print(df)

# Define data extension template
template = ExtendTemplate(column_name="reviews",
                          new_column_name="sentiment",
                          context="You are a specialist in text review sentiment recognition.",
                          task="Based on provided review evaluate the sentiment.",
                          output="Format response as one word. True if positive, False if negative or neutral."
                          )

extender = DataExtender(df)

# extend data with template
extender.chat_extend(template=template)

# translate reviews into polish
extender.add_translation(column_name="reviews", language="polish")

# topic recognition
extender.add_topic(column_name="reviews", new_column_name="category", outputs=["product", "service", "other"])

print(extender.df, "\n")

print(extender.usage_summary())
