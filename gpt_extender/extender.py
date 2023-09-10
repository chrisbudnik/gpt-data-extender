import pandas as pd
import os
import openai
from openai.openai_object import OpenAIObject
from typing import Union, Literal

from .templates import ExtendTemplate


class DataExtender:
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    def __init__(self, 
                 df: pd.DataFrame, 
                 gpt_model: str = "text-davinci-002",
                 chat_model: str = "gpt-3.5-turbo",
                 embeddings_model: str = "text-embedding-ada-002"
                 ) -> None:
        
        self.df = df
        self.gpt_model = gpt_model
        self.chat_model = chat_model
        self.embeddings_model = embeddings_model

    @staticmethod
    def _process_chat_response(res: OpenAIObject) -> str:
        response_dict = res.to_dict_recursive()
        return response_dict.get("choices", {})[0].get("message", {}).get("content", None)
    
    def chat_extend(self, template: ExtendTemplate) -> pd.DataFrame:
        generated_data = []

        for index, row in self.df.iterrows():
            data = row[template.column_name] 
            
            response = openai.ChatCompletion.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": template.context},
                {"role": "user", "content": template.prompt(data)},
            ],
            **template.extra_args
            )

            prompt_result = self._process_chat_response(response)
            generated_data.append(prompt_result)

        self.df[template.new_column_name] = generated_data
        return self.df

    def gpt_extend(self, template: ExtendTemplate) -> pd.DataFrame:
        generated_data = []

        for index, row in self.df.iterrows():
            data = row[template.column_name]

            response = openai.Completion.create(
                engine=self.gpt_model,
                prompt=data,
                **template.extra_args  
            )

            generated_text = response.choices[0].text.strip()
            generated_data.append(generated_text)

        self.df[template.new_column_name] = generated_data
        return self.df
    
    def add_embeddings(self, column_name: str):
        pass

    def add_sentiment(self, column_name: str, new_column_name: str, outputs: list[str]):

        template = ExtendTemplate(column_name=column_name,
                                  new_column_name=new_column_name,
                                  context="You are a specialist in text sentiment recognition.",
                                  task="Based on provided review evaluate the sentiment.",
                                  output=f"Format response as one word. Use the following labels: {outputs}",
                                  temperature=0 # Low temperature decreases creativity thus increasing predictibility 
                                  )
        return self.chat_extend(template=template)

    def add_translation(self, column_name: str, language: str):
        new_column_name = column_name + "_" + language
        template = ExtendTemplate(column_name=column_name,
                                  new_column_name=new_column_name,
                                  context="You are a highly qualified translator.",
                                  task=f"Based on provided text, translate is it into {language} as closely as possible.",
                                  output=""
                                  )
        return self.chat_extend(template=template)

    def add_summary(self, column_name: str, new_column_name: str):
        template = ExtendTemplate(column_name=column_name,
                                  new_column_name=new_column_name,
                                  context="You are careful reader and researcher who does not make things up.",
                                  task=f"Based on provided text, create short and descriptive summary.",
                                  output="Format your output in no more then few sentences that contain most important information only."
                                  )
        return self.chat_extend(template=template)

    def add_topic(self, column_name: str, new_column_name: str, outputs: list[str]):
        template = ExtendTemplate(column_name=column_name,
                                  new_column_name=new_column_name,
                                  context="You are careful reader and researcher who does not make things up.",
                                  task=f"Based on provided text, assign a topic that is aligned most closely with the text.",
                                  output=f"Format your output in one word. Choose from the following options: {outputs}",
                                  temperature=0 # Low temperature decreases creativity thus increasing predictibility 
                                  )
        return self.chat_extend(template=template)


