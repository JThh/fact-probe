"""Utility functions."""
import os
import re
import functools
import logging
import pickle
import hashlib

import wandb
from openai import OpenAI
from tenacity import (retry, wait_random_exponential)


import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

from eval_utils import (
    bootstrap, compatible_bootstrap, auroc, accuracy_at_quantile,
    area_under_thresholded_accuracy)

from data import MAJOR

CLIENT = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
api = wandb.Api()
api.entity = os.environ['WANDB_ENT']
MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available


def post_process_answer(model_output: str) -> str:
    """
    Extracts the first sentence or short phrase from the model's output.
    
    Args:
        model_output (str): The raw output from the model.
    
    Returns:
        str: The cleaned and concise answer.
    """
    if not model_output:
        return ""
    
    # Remove templating characters like {{ and }}
    model_output = re.sub(r'[\{\}]+', '', model_output).strip()
    
    # Split the output at sentence-ending punctuation followed by a space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', model_output)
    
    if sentences:
        # Take the first sentence
        first_sentence = sentences[0].strip()
        
        # Further split by line breaks and take the first non-empty part
        first_sentence = re.split(r'\n+', first_sentence)[0].strip()
        
        return first_sentence
    else:
        # If no sentence-ending punctuation is found, return the stripped output
        return model_output.strip()

def load_llama_model(model_name: str, device: str):
    """
    Load the LLaMA model and tokenizer.
    
    Args:
        model_name (str): Hugging Face model identifier.
        device (str): Device to load the model on ('cuda' or 'cpu').
    
    Returns:
        tokenizer: The tokenizer associated with the model.
        model: The loaded LLaMA model.
    """
    try:
        path = snapshot_download(
            repo_id=model_name,
            allow_patterns=['*.json', '*.model', '*.safetensors'],
            ignore_patterns=['pytorch_model.bin.index.json']
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
        model.eval()
        print(f"LLaMA model '{model_name}' loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        exit(1)

# Load the model and tokenizer
tokenizer, model = load_llama_model(MODEL_NAME, DEVICE)


def wandb_restore(wandb_run, filename):
    files_dir = 'tmp_wandb/'
    os.system(f'rm -rf {files_dir}')
    os.system(f'mkdir -p {files_dir}')

    run = api.run(wandb_run)
    run.file(filename).download(
        root=files_dir, replace=True, exist_ok=False)
    with open(f'{files_dir}/{filename}', 'rb') as f:
        out = pickle.load(f)
    return out, run.config


def setup_logger():
    """Setup logger to always print time and level."""
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)  # logging.DEBUG


def log_w_indent(text, indent, symbol='>>'):
    """Log and add indent."""
    ind2col = {i: f"\x1b[{a}" for i, a in enumerate([
        '1m', '31m', '33m', '34m', '35m'])}
    reset = "\x1b[0m"

    if indent > 0:
        logging.info(
            ind2col[indent] + (indent * 2) * symbol + ' ' + text + reset)
    else:
        logging.info(ind2col[indent] + text + reset)


def get_sentences(response):
    """Extract sentences from response."""
    # Some manual exceptions for sentence extraction.
    facts = response.replace('Ph.D.', 'PhD').replace('\n', ' ').split('. ')
    facts = [f.strip() + '.' if i < len(facts) - 1 else f.strip() for i, f in enumerate(facts)]
    for i in facts:
        print(i)


@retry(wait=wait_random_exponential(min=1, max=10))
def oai_predict(prompt):
    """Predict with GPT-4 model."""

    if isinstance(prompt, str):
        messages = [
            {"role": "user", "content": prompt},
        ]
    else:
        messages = prompt

    output = CLIENT.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        max_tokens=200,
    )
    response = output.choices[0].message.content
    return response


def llama_predict(prompt: str, max_new_tokens: int = 50, temperature: float = 1.0) -> str:
    """
    Generate a prediction using the LLaMA-3-70B model.
    
    Args:
        prompt (str): The input prompt for the model.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 200.
        temperature (float, optional): Sampling temperature. Higher values mean more randomness. Defaults to 1.0.
    
    Returns:
        str: The generated response from the model.
    """
    try:
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Generate the output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,  # Enable sampling for diversity
            )
        
        # Decode the generated tokens, skipping the prompt
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        # Post-process the response to extract only the first sentence or short phrase
        concise_answer = post_process_answer(response)
        
        return concise_answer
    except Exception as e:
        print(f"Error during prediction: {e}")
        return ""


def llama_predict_w_log(prompt, indent):
    """Predict and log inputs and outputs."""
    log_w_indent(f'Input: {prompt}', indent)
    response = llama_predict(prompt)
    # response = oai_predict(prompt)
    log_w_indent(f'Output: {response}', indent, symbol='xx')
    return response

def predict_w_log(prompt, indent):
    """Predict and log inputs and outputs."""
    log_w_indent(f'Input: {prompt}', indent)
    response = oai_predict(prompt)
    # response = oai_predict(prompt)
    log_w_indent(f'Output: {response}', indent, symbol='xx')
    return response


def get_metrics(all_labels, all_uncertainties):
    all_labels = np.array(all_labels)
    all_uncertainties = np.array(all_uncertainties)

    rng = np.random.default_rng(41)
    # Base accuracy of GPT-4 propositions correctly.
    out = dict(uncertainties=all_uncertainties)

    for wrongness in ['major_only', 'minor_and_major']:
        out[wrongness] = dict()

        if wrongness == 'major_only':
            # Only MAJOR becomes False.
            labels = [True if lab != MAJOR else False for lab in all_labels]
        elif wrongness == 'minor_and_major':
            # Only True is True. Both MINOR and MAJOR map to false.
            labels = [True if lab == 'True' else False for lab in all_labels]
        else:
            raise ValueError
        labels = np.array(labels)
        # assert set(labels) == {True, False}, set(labels)
        out[wrongness]['per_question'] = dict(labels=labels)

        out[wrongness]['performance'] = dict(
            mean=np.mean(labels),
            bootstrap=bootstrap(np.mean, rng)(labels))

        # Next, evaluate mean and bootstrap estimates of other metrics.

        eval_metrics = dict(zip(
            ['auroc', 'area_under_thresholded_accuracy', 'mean_uncertainty'],
            list(zip(
                [auroc, area_under_thresholded_accuracy, np.mean],
                [compatible_bootstrap, compatible_bootstrap, bootstrap]
            )),
        ))

        answer_fractions = [0.8, 0.9, 0.95]
        for answer_fraction in answer_fractions:
            key = f'accuracy_at_{answer_fraction}_answer_fraction'
            eval_metrics[key] = [
                functools.partial(accuracy_at_quantile, quantile=answer_fraction),
                compatible_bootstrap]

        fargs = {
            'auroc': [1 - labels, all_uncertainties],
            'area_under_thresholded_accuracy': [labels, all_uncertainties],
            'mean_uncertainty': [all_uncertainties]}
        for answer_fraction in answer_fractions:
            fargs[f'accuracy_at_{answer_fraction}_answer_fraction'] = [labels, all_uncertainties]

        out[wrongness]['uncertainty'] = {}
        for fname, (function, bs_function) in eval_metrics.items():
            metric_i = function(*fargs[fname])
            logging.info("%s: %f", fname, metric_i)
            out[wrongness]['uncertainty'][fname] = {
                'mean': metric_i,
                'bootstrap': bs_function(function, rng)(*fargs[fname])
            }
    return out


def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)


def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy


def extract_questions(gen_questions):

    compatibility = (
        os.environ['HALLU_RESTORE_ID'] in ['hallu_long/5yfel47n', 'hallu_long/rok13nf2'] and
        'gen_qs' in os.environ['HALLU_RESTORE_STAGES'])

    questions = []
    for i, q in enumerate(gen_questions.split('\n')):
        if q.startswith(f'{i + 1}. '):
            questions.append(q[3:])
        else:
            if not compatibility:
                questions.append(q)
            else:
                questions.append(q[3:])

    return questions


def get_yes_no(response):
    binary_response = response.lower()[:10]
    if 'yes' in binary_response:
        uncertainty = 0
    elif 'no' in binary_response:
        uncertainty = 1
    else:
        uncertainty = 1
        logging.warning('MANUAL NO!')
    return uncertainty
