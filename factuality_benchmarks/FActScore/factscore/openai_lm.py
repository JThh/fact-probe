from factscore.lm import LM
import openai
import sys
import time
import os
import numpy as np
import logging

from openai import OpenAI

CLIENT = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


class OpenAIModel(LM):

    def __init__(self, model_name, cache_file=None, key_path="api.key"):
        self.model_name = 'gpt-4o-mini'  # Altered to 'gpt-4o-mini'
        self.key_path = key_path
        self.temp = 0.7
        self.save_interval = 100
        super().__init__(cache_file)

    def load_model(self):
        self.model = self.model_name

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()

        # Return a tuple of string (generated text) and metadata (any format)
        response = call_GPT4o(prompt, temp=self.temp, max_len=max_sequence_length)
        output = response.choices[0].message.content
        return output, response

def call_GPT4o(prompt, model_name="gpt-4o-mini-2024-07-18", max_len=1024, temp=0.5, verbose=False):
    # Call GPT-4o-mini API until result is provided and then return it
    response = None
    messages = [
        {"role": "user", "content": prompt},
    ]
    response = CLIENT.chat.completions.create(model=model_name,
                                        messages=messages,
                                        max_tokens=max_len,
                                        temperature=temp)
    return response


