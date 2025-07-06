from abc import ABC, abstractmethod
from openai import OpenAI
import logging
import torch
import os
import re

import time 
import requests
import json

import src.utils as utils

# Consolidated exclusion patterns using regex alternation and common keywords
exclusion_patterns = re.compile(
    r'\b(?:I\s+(?:cannot|was unable to|do not|have no)\s+(?:find|provide|locate|access)|'
    r'(?:No|Insufficient|Cannot|Unable)\s+(?:information|data|details|records|bio|profile|content)|'
    r'(?:I\s+am\s+not\s+sure|Note\s+(?:that|,))|'
    r'(?:No\s+(?:additional|further|relevant)\s+information))\b',
    flags=re.IGNORECASE
)


CLIENT = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# OR_CLIENT = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.environ["OPENROUTER_API_KEY"],
# )
OR_CLIENT = None


class BaseModel(ABC):
    def __init__(self, model_name, args):
        self.model_name = model_name
        self.args = args

    @abstractmethod
    def generate_given_prompt(self, prompt):
        pass

    @abstractmethod
    def generate_n_given_prompt(self, prompt):
        pass

class Llama405BModel(BaseModel):
    def __init__(self, model_name, args):
        super().__init__(model_name, args)
        # self.url = "https://api.segmind.com/v1/llama-v3p1-405b-instruct"
        self.url = "https://api.hyperbolic.xyz/v1/chat/completions"
        # api_key = os.getenv("SEG_API_KEY")
        api_key = os.getenv("HYPER_API_KEY")
        if not api_key:
            raise Exception("SEG_API_KEY not set.")
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def _call_with_retries(self, func, retries=30, backoff=1.0):
        for _ in range(retries):
            try:
                resp = func()
                if resp.status_code == 429:
                    raise Exception("Rate limit")
                return resp
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    print("Backing off")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    logging.exception(e)
                    raise
        raise Exception("Max retries exceeded")

    def generate_given_prompt(self, prompt):
        data = {"messages": [{"role": "user", "content": prompt}], 
                "model": "meta-llama/Meta-Llama-3.1-405B-Instruct",
                "max_tokens": 1000,
                "temperature":self.args.temperature}
        try:
            response = self._call_with_retries(lambda: requests.post(self.url, headers=self.headers, json=data))
            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}", "prompt": prompt}
            json_resp = response.json()
            print(json_resp)
            choices = json_resp.get("choices", [])
            if not choices:
                return {"error": "No choices", "prompt": prompt}
            content = choices[0].get("message", {}).get("content")
            if not content:
                return {"error": "No content", "prompt": prompt}
            return {"generation": content, "prompt": prompt}
        except Exception as e:
            logging.exception(e)
            return {"error": str(e), "prompt": prompt}

    def generate_n_given_prompt(self, prompt):
        gens = []
        for _ in range(self.args.num_generations_per_prompt):
            res = self.generate_given_prompt(prompt)
            if "generation" in res:
                gens.append(res["generation"])
        return {"generation": gens, "prompt": prompt}
    

# class Llama405BModel(BaseModel):
#     def __init__(self, model_name, args):
    #     super().__init__(model_name, args)
    #     global OR_CLIENT

    #     if model_name == 'llama-3.1-405b-instruct':
    #         self.model_name = 'llama-3.1-405b-instruct'

    #     try:
    #         api_key = os.environ["OPENROUTER_API_KEY"]
    #     except KeyError:
    #         logging.error("Environment variable OPENROUTER_API_KEY not found. Please set it.")
    #         raise

    #     OR_CLIENT = OpenAI(
    #         base_url="https://openrouter.ai/api/v1",
    #         api_key=api_key,
    #     )

    # def _call_with_retries(self, create_func, max_retries=30, initial_backoff=5.0):
    #     """
    #     Helper method to call the API with retries on rate limit errors.
    #     `create_func` is a callable that executes the API request.
    #     """
    #     backoff = initial_backoff
    #     for attempt in range(max_retries):
    #         response = create_func()
    #         # Check if the response has an error indicating rate limit
    #         if response and hasattr(response, "error") and response.error['code'] == 429:
    #             print(response)
    #             logging.warning(f"Rate limit encountered on attempt {attempt + 1}. "
    #                             f"Sleeping for {backoff} seconds before retrying.")
    #             time.sleep(backoff)
    #             backoff *= 2  # Exponential backoff
    #         else:
    #             return response
    #     return response  # Return the last response after retries

    # def generate_given_prompt(self, prompt, temp=0.0):
    #     global OR_CLIENT
    #     try:
    #         # Define a lambda to delay actual API call until inside _call_with_retries
    #         api_call = lambda: OR_CLIENT.chat.completions.create(
    #             extra_headers={
    #                 "HTTP-Referer": "<YOUR_SITE_URL>",
    #                 "X-Title": "<YOUR_SITE_NAME>",
    #             },
    #             model="meta-llama/llama-3.1-405b-instruct:free",
    #             messages=[{"role": "user", "content": prompt}],
    #             max_tokens=1000,
    #             temperature=temp,
    #         )
    #         completion = self._call_with_retries(api_call)
            
    #         print(completion)

    #         if not completion:
    #             logging.error("No response received from OpenRouter API after retries.")
    #             return {'error': 'No response from API', 'prompt': prompt}

    #         if not hasattr(completion, 'choices') or not completion.choices:
    #             logging.error("API response contained no choices.")
    #             return {'error': 'No choices in response', 'prompt': prompt}

    #         return {'generation': completion.choices[0].message.content, 'prompt': prompt}

    #     except Exception as e:
    #         logging.exception(f"Exception during generate_given_prompt: {e}")
    #         return {'error': str(e), 'prompt': prompt}
        
    # def generate_n_given_prompt(self, prompt):
    #     """
    #     Generate multiple responses by issuing single requests sequentially.
    #     """
    #     generations = []
    #     errors = []

    #     # Loop over the desired number of generations
    #     for _ in range(self.args.num_generations_per_prompt):
    #         result = self.generate_given_prompt(prompt, temp=self.args.temperature)
    #         if 'generation' in result:
    #             generations.append(result['generation'])
    #         else:
    #             # Collect errors separately if needed
    #             errors.append(result.get('error', 'Unknown error'))

    #     # Optionally handle errors: log them or combine them in the response
    #     if errors:
    #         logging.error(f"Errors encountered during generation: {errors}")

    #     return {'generation': generations, 'prompt': prompt}

    # def generate_n_given_prompt(self, prompt):
    #     global OR_CLIENT
    #     try:
    #         api_call = lambda: OR_CLIENT.chat.completions.create(
    #             extra_headers={
    #                 "HTTP-Referer": "<YOUR_SITE_URL>",
    #                 "X-Title": "<YOUR_SITE_NAME>",
    #             },
    #             model="meta-llama/llama-3.1-405b-instruct:free",
    #             messages=[{"role": "user", "content": prompt}],
    #             max_tokens=1000,
    #             temperature=self.args.temperature,
    #             n=self.args.num_generations_per_prompt
    #         )
    #         completion = self._call_with_retries(api_call)
            
    #         print(completion)

    #         if not completion:
    #             logging.error("No response received from OpenRouter API after retries.")
    #             return {'error': 'No response from API', 'prompt': prompt}

    #         if completion.error:
    #             logging.error(f"API returned an error: {completion.error}")
    #             return {'error': completion.error, 'prompt': prompt}

    #         if not hasattr(completion, 'choices') or not completion.choices:
    #             logging.error("API response contained no choices.")
    #             return {'error': 'No choices in response', 'prompt': prompt}

    #         generations = []
    #         for index, choice in enumerate(completion.choices):
    #             if not choice or not hasattr(choice, 'message'):
    #                 logging.warning(f"Choice {index} missing message field.")
    #                 continue
    #             message = choice.message
    #             if not message or not getattr(message, 'content', None):
    #                 logging.warning(f"Choice {index} contains no content.")
    #                 continue
    #             generations.append(message.content)

    #         if not generations:
    #             logging.error("No valid generations found in response choices.")
    #             return {'error': 'No valid generations', 'prompt': prompt}

    #         return {'generation': generations, 'prompt': prompt}

    #     except Exception as e:
    #         logging.exception(f"Exception during generate_n_given_prompt: {e}")
    #         return {'error': str(e), 'prompt': prompt}



class OpenAIModel(BaseModel):

    def __init__(self, model_name, args):
        super().__init__(model_name, args)
        if args:
            self.cache_file_path = f"openai_model_cache_{args.model}.json"
        else:
            self.cache_file_path = "openai_model_cache.json"
        if model_name == 'gpt-3.5-turbo':
            self.model_name = 'gpt-3.5-turbo-0125'
        elif model_name == 'gpt-4o-mini':
            self.model_name = 'gpt-4o-mini'
        elif model_name == 'gpt-4o':
            self.model_name = 'gpt-4o'
        else:
            self.model_name = model_name

        # Load cache from disk if it exists
        self.cache = {}
        if os.path.exists(self.cache_file_path):
            try:
                with open(self.cache_file_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                logging.warning(f"Could not load cache file {self.cache_file_path}: {e}")
                self.cache = {}

        print("self.cache", list(self.cache.values())[:2])

    def _save_cache(self):
        """Write the current in-memory cache to disk."""
        try:
            with open(self.cache_file_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"Could not save cache file {self.cache_file_path}: {e}")

    def generate_given_prompt(self, prompt):
        """
        Generate a single response given a prompt.
        Checks local cache before calling the API.
        """
        # Convert prompt to a string key (if prompt is a list/dict, etc.)
        key = f"single|{str(prompt)}"

        if key in self.cache:
            # Return cached result
            return self.cache[key]

        # If prompt is a string, convert to the chat-like format for OpenAI
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt}]

        if 'gpt' in self.model_name:
            # Make the actual call
            response = CLIENT.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=2048,
                temperature=0.0,
            )
            result = {
                'generation': response.choices[0].message.content,
                'prompt': prompt
            }
        else:
            raise NotImplementedError("Only GPT-based models implemented here.")

        # Store result in cache and save
        self.cache[key] = result
        self._save_cache()
        return result

    def generate_n_given_prompt(self, prompt):
        """
        Generate multiple responses (n) given a prompt.
        Checks local cache before calling the API.
        """
        key = f"multi|{str(prompt)}|num_gen={self.args.num_generations_per_prompt}|temp={self.args.temperature}"
        if key in self.cache:
            return self.cache[key]

        if 'gpt' in self.model_name:
            # Make the actual call
            response = CLIENT.chat.completions.create(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=2048,
                temperature=self.args.temperature,
                n=self.args.num_generations_per_prompt
            )
            result = {
                'generation': [choice.message.content for choice in response.choices],
                'prompt': prompt
            }
        else:
            raise NotImplementedError("Only GPT-based models implemented here.")

        # Store result in cache and save
        self.cache[key] = result
        self._save_cache()
        return result

class Llama3Model(BaseModel):
    def __init__(self, model_name, args):
        super().__init__(model_name, args)
        if 'llama-3-70b-instruct' == model_name:
            self.model, self.tokenizer = utils.load_llama3_70b_model_and_tokenizer()
        elif 'llama-3.1-70b-instruct' == model_name:
            self.model, self.tokenizer = utils.load_llama31_70b_model_and_tokenizer()
        elif 'llama-3.1-8b-instruct' == model_name:
            self.model, self.tokenizer = utils.load_llama31_8b_model_and_tokenizer()
        elif 'llama-3.1-405b-instruct' == model_name:
            self.model, self.tokenizer = utils.load_llama31_405b_model_and_tokenizer()
        else:
            raise ValueError("Invalid model name:", model_name)

    def post_process_bio(self, generated_text: str, prompt: str) -> str:
        """
        Post-process the generated bio to ensure completeness and remove extraneous content.
        
        Args:
            generated_text (str): The raw text generated by the model.
            prompt (str): The prompt that was sent to the model.
        
        Returns:
            str: The cleaned and validated bio.
        """
        # Remove the prompt from the generated text
        bio = generated_text.replace(prompt, "").strip()
        # Ensure the bio ends with a period. If not, truncate to the last complete sentence.
        if not bio.endswith('.'):
            last_period = bio.rfind('.')
            if last_period != -1:
                bio = bio[:last_period + 1]
        
        # Remove any text after the first bio if multiple bios are generated
        bio = re.split(r'\nTell me a paragraph bio of', bio, flags=re.IGNORECASE)[0].strip()
        
        # Remove explanations about inability to find more information using the consolidated pattern
        bio = exclusion_patterns.split(bio)[0].strip()
        
        return bio

    def generate_given_prompt(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        
        torch.cuda.empty_cache()

        terminators = [
            self.tokenizer.eos_token_id,
        ]

        outputs = self.model.generate(
            input_ids, max_new_tokens=512, eos_token_id=terminators, 
            do_sample=True, temperature=0.0001, pad_token_id=self.tokenizer.eos_token_id,
        )
        generation = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
        generation = self.post_process_bio(generation, prompt)

        return {'generation': generation, 'prompt': prompt}

    def generate_n_given_prompt(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
        ]

        outputs = self.model.generate(
            input_ids, max_new_tokens=512, eos_token_id=terminators, 
            do_sample=True, temperature=self.args.temperature, 
            num_return_sequences=self.args.num_generations_per_prompt, 
            pad_token_id=self.tokenizer.eos_token_id
        )
        generations = [self.tokenizer.decode(decoded[input_ids.shape[-1]:], skip_special_tokens=True).strip() for decoded in outputs]
        generations = [self.post_process_bio(gen, prompt) for gen in generations]
        return {'generation': generations, 'prompt': prompt}
    

class Gemma2Model(BaseModel):
    def __init__(self, model_name, args):
        super().__init__(model_name, args)
        if 'gemma-2-9b-it' == model_name:
            self.model, self.tokenizer = utils.load_gemma2_9b_model_and_tokenizer()
        else:
            raise ValueError("Invalid model name:", model_name)
            
    def post_process_bio(self, generated_text: str, prompt: str) -> str:
        """
        Post-process the generated bio to ensure completeness and remove extraneous content.
        
        Args:
            generated_text (str): The raw text generated by the model.
            prompt (str): The prompt that was sent to the model.
        
        Returns:
            str: The cleaned and validated bio.
        """
        # Remove the prompt from the generated text
        bio = generated_text.replace(prompt, "").strip()
        # Ensure the bio ends with a period. If not, truncate to the last complete sentence.
        if not bio.endswith('.'):
            last_period = bio.rfind('.')
            if last_period != -1:
                bio = bio[:last_period + 1]
        
        # Remove any text after the first bio if multiple bios are generated
        bio = re.split(r'\nTell me a bio of', bio, flags=re.IGNORECASE)[0].strip()
        
        # Remove explanations about inability to find more information using the consolidated pattern
        bio = exclusion_patterns.split(bio)[0].strip()
        
        return bio

    def generate_given_prompt(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        
        torch.cuda.empty_cache()

        terminators = [
            self.tokenizer.eos_token_id,
        ]

        outputs = self.model.generate(
            input_ids, max_new_tokens=1000, eos_token_id=terminators, 
            do_sample=True, temperature=0.0001, pad_token_id=self.tokenizer.eos_token_id
        )
        generation = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
        # generation = self.post_process_bio(generation, prompt)

        return {'generation': generation, 'prompt': prompt}

    def generate_n_given_prompt(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
        ]

        outputs = self.model.generate(
            input_ids, max_new_tokens=1000, eos_token_id=terminators, 
            do_sample=True, temperature=self.args.temperature, 
            num_return_sequences=self.args.num_generations_per_prompt, 
            pad_token_id=self.tokenizer.eos_token_id
        )
        generations = [self.tokenizer.decode(decoded[input_ids.shape[-1]:], skip_special_tokens=True).strip() for decoded in outputs]
        # generations = [self.post_process_bio(gen, prompt) for gen in generations]
        return {'generation': generations, 'prompt': prompt}


def get_model(model_name, args):
    if 'gpt' in model_name:
        return OpenAIModel(model_name, args)
    # elif '405b' in model_name:
        # return Llama405BModel(model_name, args)
    elif 'llama-3' in model_name:
        return Llama3Model(model_name, args)
    elif 'gemma-2' in model_name:
        return Gemma2Model(model_name, args)
    else:
        raise NotImplementedError
    
class A:
    pass

if __name__ == '__main__':
    model_name = 'gemma-2-9b-it'
    args = A()
    setattr(args, "num_generations_per_prompt", 6)
    setattr(args, "temperature", 0.7)
    model = get_model(model_name, args)
    prompt = 'What is the meaning of life?'
    results = model.generate_given_prompt(prompt)
    print(results)
    more_results = model.generate_n_given_prompt(prompt)
    print(more_results)
