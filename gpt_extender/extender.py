import pandas as pd
import os
import openai
from openai.openai_object import OpenAIObject
from typing import Union, Literal

from .templates import ExtendTemplate


class DataExtender:

    def __init__(self, 
                 df: pd.DataFrame, 
                 gpt_model: str = "text-davinci-003",
                 chat_model: str = "gpt-3.5-turbo",
                 embeddings_model: str = "text-embedding-ada-002"
                 ) -> None:
        
        self.df = df
        self.gpt_model = gpt_model
        self.chat_model = chat_model
        self.embeddings_model = embeddings_model

        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _chat_completion(self, context: str, prompt: str, **kwargs):
        response = openai.ChatCompletion.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": prompt},
            ],
            **kwargs
            )
        self._update_token_usage(response)
        return self._process_chat_response(response)

    def _gpt_completion(self, prompt: str, **kwargs):
        response = openai.Completion.create(
            engine=self.gpt_model,
            prompt=prompt,
            **kwargs 
        )
        self._update_token_usage(response)
        return response.choices[0].text.strip()

    @staticmethod
    def _process_chat_response(res: OpenAIObject) -> str:
        response_dict = res.to_dict_recursive()
        return response_dict.get("choices", {})[0].get("message", {}).get("content", None)

    def _update_token_usage(self, res: OpenAIObject) -> None:
        self.prompt_tokens += res.get("usage", {}).get("prompt_tokens", 0)
        self.completion_tokens += res.get("usage", {}).get("completion_tokens", 0)

    def sample_to_text(self, column_name: str, sample_size: int = 5) -> str:
        sample = self.df[column_name].sample(sample_size)
        return "\n".join(sample)
    
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
            self._update_token_usage(response)
            prompt_result = self._process_chat_response(response)
            generated_data.append(prompt_result)

        self.df[template.new_column_name] = generated_data
        return self.df

    def gpt_extend(self, template: ExtendTemplate) -> pd.DataFrame:
        generated_data = []

        for index, row in self.df.iterrows():
            data = row[template.column_name]

            prompt = template.context + template.prompt(data)
            response = openai.Completion.create(
                engine=self.gpt_model,
                prompt=prompt,
                **template.extra_args  
            )
            self._update_token_usage(response)
            generated_text = response.choices[0].text.strip()
            generated_data.append(generated_text)

        self.df[template.new_column_name] = generated_data
        return self.df
    
    def synthetic_extend(self, 
                         template: ExtendTemplate, 
                         output_size: int = 5, 
                         sample_size: int = 5, 
                         inplace: bool = True,
                         flag_synthetic_data: bool = False
                         ) -> pd.DataFrame:

        sample_records = self.sample_to_text(template.column_name, sample_size)
        
        if len(self.df.columns) > 1:
            raise NotImplementedError("synthetic_extend method currently does not support multi-dimensional data.")

        response = openai.Completion.create(
                engine=self.gpt_model,
                prompt=template.prompt_synthetic(text=sample_records, 
                                                 output_size=output_size),
                **template.extra_args  
            )
        self._update_token_usage(response)
        new_records = response.choices[0].text.strip()
        new_records_df = pd.DataFrame({template.column_name: new_records.split("\n")})
        
        # create a flag: is_synthetic 
        if flag_synthetic_data:
            self.df["is_synthetic"] = False
            new_records_df["is_synthetic"] = True

        # apply extension to instance DataFrame
        extended_df = pd.concat([self.df, new_records_df], ignore_index=True)
        if inplace:
            self.df = extended_df
        return extended_df
    
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
    
    def add_synthetic_data(self, column_name: str, **kwargs):
        template = ExtendTemplate(column_name=column_name,
                                  new_column_name="",
                                  context="You are an expert data specialist capable of understanding and learing patterns from data.",
                                  task=f"Based on provided sample of text inputs, learn the patterns and context. Then generate synthetic ones that may well be part of sample.",
                                  output=f"Format your output so it as closely resembles formatting and style of initial inputs. New records separate with new line.",
                                  )
        return self.synthetic_extend(template=template, **kwargs)
    
    def usage_summary(self, cost_1k_tokens=0.002) -> dict[str: int]:
        usage_dict = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "estimated_cost": (self.prompt_tokens + self.completion_tokens)/1000*cost_1k_tokens
        }
        return usage_dict
    
    def ai_analyze(self, column_name: str, **kwargs) -> None:
        sample = self.sample_to_text(column_name, **kwargs)
        template = ExtendTemplate(column_name=column_name,
                                  new_column_name="",
                                  context="You are a specialist in data analysis. Given text sample you learn its patterns and context.",
                                  task="""Propose new columns or measures to help better understand the provided text, 
                                  try diverse feature enginnering like: boolen/categorical columns.""",
                                  output="Format response a list of ideas, include explenation on how new features may be beneficial.",
                                  temperature=0.7 # increased creativity
                                  )
        
        response = openai.ChatCompletion.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": template.context},
                {"role": "user", "content": template.prompt(sample)},
            ],
            **template.extra_args
        )
        self._update_token_usage(response)
        prompt_result = self._process_chat_response(response)
        print(prompt_result)

    def _validate_column_name(self, name) -> None:
        if name not in self.df.columns:
            raise NameError(f"Column name: {name} is not part of the instance DataFrame.")



