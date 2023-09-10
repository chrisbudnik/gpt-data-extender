import pandas as pd
import os
import openai
from openai.openai_object import OpenAIObject
from typing import Union, Literal

from .templates import ExtendTemplate


class DataExtender:
    openai_key = os.environ.get("OPENAI_API_KEY")

    def __init__(self, 
                 df: pd.DataFrame, 
                 model: str = "text-davinci-002",
                 chat_model: str = "gpt-3.5-turbo"
                 ) -> None:
        
        self.df = df
        self.model = model
        self.chat_model = chat_model

    def gpt_extend(self, template: ExtendTemplate) -> pd.DataFrame:
        generated_data = []

        for index, row in self.df.iterrows():
            data = row[template.column_name]
            prompt = template.prompt().replace("something", data)  

            response = openai.Completion.create(
                engine=self.model,
                prompt=prompt,
                **template.extra_args  
            )

            generated_text = response.choices[0].text.strip()
            generated_data.append(generated_text)

        self.df[template.new_column_name] = generated_data
        return self.df

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
            generated_data.append(eval(prompt_result))

        self.df[template.new_column_name] = generated_data
        return self.df

    @staticmethod
    def _process_chat_response(res: OpenAIObject) -> str:
        response_dict = res.to_dict_recursive()
        return response_dict.get("choices", {})[0].get("message", {}).get("content", None)
