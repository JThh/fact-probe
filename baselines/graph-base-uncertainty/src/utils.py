import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, roc_auc_score
import networkx as nx
import os
import argparse
import re

import time

from huggingface_hub import snapshot_download 

LLAMA3_70B_PATH = ""

def load_huggingface_model_and_tokenizer(model_id, cache_dir):
    print_cuda_memory()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map='auto', 
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2", 
        cache_dir=cache_dir
    )
    # NOTE: snapshot_download might not be applicable if network bandwidth is limited
    # path = snapshot_download(
    #     repo_id=model_id,
    #     allow_patterns=['*.json', '*.model', '*.safetensors'],
    #     ignore_patterns=['pytorch_model.bin.index.json'],
    # )
    # config = AutoConfig.from_pretrained(model_id)
    # with accelerate.init_empty_weights():
    #     model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16,
    #                                              attn_implementation="flash_attention_2", cache_dir=cache_dir)
    # model.tie_weights()

    # model = accelerate.load_checkpoint_and_dispatch(
    #     model, path, device_map="auto",
    #     dtype='bfloat16', skip_keys='past_key_values')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print_cuda_memory()
    return model, tokenizer

def load_llama3_70b_model_and_tokenizer():
    cache_path = os.environ.get("HF_DATASETS_CACHE", '')
    return load_huggingface_model_and_tokenizer("meta-llama/Meta-Llama-3-70B-Instruct", cache_dir=cache_path)

def load_llama31_70b_model_and_tokenizer():
    cache_path = os.environ.get("HF_DATASETS_CACHE", '')
    return load_huggingface_model_and_tokenizer("meta-llama/Llama-3.1-70B-Instruct", cache_dir=cache_path)

def load_llama31_8b_model_and_tokenizer():
    cache_path = os.environ.get("HF_DATASETS_CACHE", '')
    return load_huggingface_model_and_tokenizer("meta-llama/Llama-3.1-8B-Instruct", cache_dir=cache_path)

def load_llama31_405b_model_and_tokenizer():
    cache_path = os.environ.get("HF_DATASETS_CACHE", '')
    return load_huggingface_model_and_tokenizer("meta-llama/Llama-3.1-405B-Instruct", cache_dir=cache_path)

def load_gemma2_9b_model_and_tokenizer():
    cache_path = os.environ.get("HF_DATASETS_CACHE", '')
    return load_huggingface_model_and_tokenizer("google/gemma-2-9b-it", cache_dir=cache_path)

def substitute_prompt_factscore(example, dataset):
    entity = example['entity']
    example['prompt'] = f'Tell me a paragraph bio of {entity}.\n'
    example['question'] = entity
    return example

def substitute_prompt_pop_qa(example):
    example['prompt'] = f'Provide me with a paragraph detailing some facts related to {example["s_wiki_title"]}.'
    example['wiki_title'] = example['s_wiki_title']
    example['question'] = example['wiki_title']
    return example

def print_cuda_memory():
    if torch.cuda.is_available():
        print("CUDA is available. Listing memory usage:")
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device=i)
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            cached_memory = torch.cuda.memory_reserved(i)
            print(f"Device {i}:")
            print(f"  Total Memory: {total_memory / (1024 ** 3):.2f} GB")
            print(f"  Allocated Memory: {allocated_memory / (1024 ** 3):.2f} GB")
            print(f"  Cached Memory: {cached_memory / (1024 ** 3):.2f} GB")
    else:
        print("CUDA is not available.")

VC_OPTIONS = {'nochance': 0, 'littlechance': 0.2, 'lessthaneven': 0.4, 'fairlypossible': 0.6, 'verygoodchance': 0.8, 'almostcertain': 1.0}

def parse_confidence(input_string, with_options=False):
    """
    Parses the input string to find a percentage or a float between 0.0 and 1.0 within the text.
    If a percentage is found, it is converted to a float.
    If a float between 0.0 and 1.0 is found, it is also converted to a float.
    In other cases, returns -1.

    :param input_string: str, the string to be parsed.
    :return: float, the parsed number or -1 if no valid number is found.
    """
    if with_options:
        split_list = input_string.split(':')
        if len(split_list) > 1:
            input_string = split_list[1]
        only_alpha = re.sub(r'[^a-zA-Z]', '', input_string).lower()
        if only_alpha in VC_OPTIONS:
            return VC_OPTIONS[only_alpha]
    else:
        # Search for a percentage in the text
        percentage_match = re.search(r'(\d+(\.\d+)?)%', input_string)
        if percentage_match:
            return float(percentage_match.group(1)) / 100

        # Search for a float between 0.0 and 1.0 in the text
        float_match = re.search(r'\b0(\.\d+)?\b|\b1(\.0+)?\b', input_string)
        if float_match:
            return float(float_match.group(0))

    # If neither is found, return -1
    return -1

def get_verbalized_confidence(question, generation, args, model_instance, raw_result=None, problem_type='qa', with_options=False):
    if raw_result:
        results = raw_result
    else:
        if problem_type == 'qa':
            prompt = f'You are provided with a question and a possible answer. Provide the probability that the possible answer is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\nProbability: <the probability that your guess is correct as a percentage, without any extra commentary whatsoever; just the probability!>\n\nThe question is: {question}\nThe possible answer is: {generation}'
        elif problem_type == 'fact':
            prompt = f'You are provided with some possible information about a person. Provide the probability that the information is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\nProbability: <the probability that your guess is correct as a percentage, without any extra commentary whatsoever; just the probability!>\n\nThe person is: {question}\nThe possible information is: {generation}'
        
        if with_options:
            if problem_type == 'qa':
                prompt = f'You are provided with a question and a possible answer. Describe how likely it is that the possible answer is correct as one of the following expressions:\nNo chance (0%)\nLittle chance  (20%)\nLess than even (40%)\nFairly possible (60%)\nVery good chance (80%)\nAlmost certain (100%)\n\nGive ONLY your confidence phrase, no other words or explanation. For example:\n\nConfidence: <description of confidence, without any extra commentary whatsoever; just a short phrase!>\n\nThe question is: {question}\nThe possible answer is: {generation}'
            elif problem_type == 'fact':
                prompt = f'You are provided with some possible information about a person. Describe how likely it is that the possible answer is correct as one of the following expressions:\nNo chance (0%)\nLittle chance  (20%)\nLess than even (40%)\nFairly possible (60%)\nVery good chance (80%)\nAlmost certain (100%)\n\nGive ONLY your confidence phrase, no other words or explanation. For example:\n\nConfidence: <description of confidence, without any extra commentary whatsoever; just a short phrase!>\n\nThe person is: {question}\nThe possible information is: {generation}'
                
        results = model_instance.generate_given_prompt(prompt)

    if 'generation' in results:
        generation = results['generation']
    else:
        choice = results["choices"][0]
        generation = choice['message']['content'].strip()
    confidence = parse_confidence(generation, with_options=with_options)

    return confidence, results


def get_ptrue_confidence(entity, claim, args, models=None):
    prompt = f"The following claim is about {entity}. Is the claim true or false? (Answer with only one word True/False)\nClaim: {claim}"
    # TODO: Fix this call - generate_n_given_prompt should be called on a model instance
    # results = generate_n_given_prompt(prompt, args, alpacafarm_models=models)
    results = {"choices": [{"message": {"content": "True"}}]}  # Placeholder for now
    if 'generation' in results:
        generated_texts = results['generation']
    else:
        generated_texts = [choice['message']['content'] for choice in results["choices"]]
    filtered_texts = []
    for text in generated_texts:
        only_alpha = re.sub(r'[^a-zA-Z]', '', text).lower()
        if only_alpha in ['true', 'false']:
            filtered_texts.append(only_alpha)
    if len(filtered_texts) == 0:
        print(f'No valid answer found for this question')
        return -1
    return len([text for text in filtered_texts if text == 'true']) / len(filtered_texts)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def calibration_measure(path, confidence_dict, accuracy_dict,
                     ax=None,
                     n_bins=10,
                     strategy="quantile",
                     label=""):
    """Plot the calibration, i.e., accuracy vs confidence, of the predictions.
    Code adapted from https://github.com/scikit-learn/scikit-learn/blob/98cf537f5c538fdbc9d27b851cf03ce7611b8a48/sklearn/calibration.py#L910
    Args:
        result_dict: the result dict for a single setup and global random seed
        ax: the ax object to make the plot
        data_split: the data split to plot, choices are ['val', 'test', 'gen']
        n_bins: int, default=10
            Number of bins to discretize the [0, 1] interval. A bigger number
            requires more data. Bins with no samples (i.e. without
            corresponding values in `y_prob`) will not be returned, thus the
            returned arrays may have less than `n_bins` values.
        strategy: {'uniform', 'quantile'}, default='uniform'
            Strategy used to define the widths of the bins.
            uniform
                The bins have identical widths.
            quantile
                The bins have the same number of samples and depend on `y_prob`.
    """
    if not ax:
        fig, ax = plt.subplots()
    colors = plt.cm.get_cmap('viridis', len(confidence_dict))

    assert confidence_dict.keys() == accuracy_dict.keys()
    for idx, name in enumerate(confidence_dict.keys()):
        acc, conf = accuracy_dict[name], confidence_dict[name] 

        valid_indices = np.where((acc >= 0) & (acc <= 1) & (conf >= 0) & (conf <= 1))[0]
        pred_acc, pred_confs = acc[valid_indices], conf[valid_indices]

        acc = np.mean(pred_acc)
        if pred_confs.min() < 0 or pred_confs.max() > 1:
            raise ValueError("pred_confs has values outside [0, 1].")

        if strategy == "quantile":  # Determine bin edges by distribution of data
            quantiles = np.linspace(0, 1, n_bins + 1)
            bins = np.percentile(pred_confs, quantiles * 100)
        elif strategy == "uniform":
            bins = np.linspace(0.0, 1.0, n_bins + 1)
        else:
            raise ValueError("Invalid entry to 'strategy' input. Strategy "
                            "must be either 'quantile' or 'uniform'.")
        binids = np.searchsorted(bins[1:-1], pred_confs)
        bin_sum_accs = np.bincount(binids, weights=pred_acc, minlength=len(bins))
        bin_sum_confs = np.bincount(binids,
                                    weights=pred_confs,
                                    minlength=len(bins))
        bin_num = np.bincount(binids, minlength=len(bins))
        nonzero = bin_num != 0
        bin_mean_accs = bin_sum_accs[nonzero] / bin_num[nonzero]
        bin_mean_confs = bin_sum_confs[nonzero] / bin_num[nonzero]
        bin_num = bin_num[nonzero]

        ECE = np.sum(
            np.abs(bin_mean_accs - bin_mean_confs) * bin_num) / np.sum(bin_num)
    
        AUROC_top_prob = roc_auc_score(pred_acc, pred_confs)

        brier_score = brier_score_loss(pred_acc, pred_confs)
        if not ax:
            fig, ax = plt.subplots()
        ax.scatter(bin_mean_confs, bin_mean_accs, s=bin_num * 5,
                   color=colors(idx), alpha=0.5,
                   label=f'{name} - ECE: {ECE:.3f}, Acc: {acc:.3f}, AUROC: {AUROC_top_prob:.3f}, Brier Score: {brier_score:.3f}, Valid Size: {len(valid_indices)}')

    ax_min, ax_max = 0.0 - 0.04, 1.0 + 0.04
    ax.set_xlim([ax_min, ax_max])
    ax.set_ylim([ax_min, ax_max])
    ax.set_xticks([0., 0.2, 0.4, 0.6, 0.8, 1.])
    ax.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1.])
    ax.set_xticklabels([0., 0.2, 0.4, 0.6, 0.8, 1.], fontsize=16)
    ax.set_yticklabels([0., 0.2, 0.4, 0.6, 0.8, 1.], fontsize=16)
    ax.plot(np.linspace(ax_min, ax_max, num=20),
            np.linspace(ax_min, ax_max, num=20),
            '--',
            color='grey')  # diagonal
    ax.set_xlabel('Confidence', fontsize=18)
    ax.set_ylabel('Accuracy', fontsize=18)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=1, fontsize=11)
    plt.savefig(path, bbox_inches='tight')

    return None

CENTRALITY_TYPE_LIST = ['eigenvector_centrality', 'betweenness_centrality', 'closeness_centrality', 'pagerank', 'closeness_centrality_with_node_confidence']

def calculate_bg_centrality(lsts, length, vc_lst=None):
    centrality_dict = {}
    for centrality_name in CENTRALITY_TYPE_LIST:
        centrality_dict[centrality_name] = np.ones(length) * -1    
    
    filtered_lists = np.array(lsts)
    gen_num, flitered_breakdown_len = np.shape(filtered_lists)[0], np.shape(filtered_lists)[1]
    adjacency_matrix = np.zeros((gen_num + flitered_breakdown_len, gen_num + flitered_breakdown_len))
    adjacency_matrix[:gen_num, gen_num:] = filtered_lists
    adjacency_matrix[gen_num:, :gen_num] = filtered_lists.T
    G = nx.from_numpy_array(adjacency_matrix)
    if vc_lst is not None:
        combined_c = bg_closeness_centrality_with_node_confidence(G, gen_num, vc_lst)

    centrality = np.ones(length) * -1 
    for function_name in centrality_dict:
        try:
            if function_name in ['eigenvector_centrality', 'pagerank']:
                centrality = getattr(nx, function_name)(G, max_iter=5000)
            elif function_name == 'closeness_centrality_with_node_confidence' and vc_lst is not None:
                centrality = combined_c
            else:
                centrality = getattr(nx, function_name)(G)
        except:
            pass

        assert length == flitered_breakdown_len
        centrality = [centrality[i] for i in sorted(G.nodes())] # List of scores in the order of nodes
        centrality_dict[function_name] = centrality[gen_num:]
                
    return centrality_dict

def bg_closeness_centrality_with_node_confidence(G, gen_num, verb_conf):
    n = len(G)
    combined_centrality = np.ones(len(G)) * -1
    
    for i in range(gen_num, n):
        sum_shortest_path_to_gens, sum_vc, sum_shortest_path_to_gens_vc_unweighted, sum_shortest_path_to_gens_vc_product = 0, 0, 0, 0
        reachable = 0
        
        for gen_id in range(n):
            if i == gen_id:
                continue
            try:
                shortest_path = nx.shortest_path(G, source=i, target=gen_id)
                shortest_path = [node for node in shortest_path if node >= gen_num]
                verb_conf_sum = np.sum([1 - verb_conf[node - gen_num] for node in shortest_path])
                                
                path_length = nx.shortest_path_length(G, source=i, target=gen_id)
                
                sum_vc += verb_conf_sum
                sum_shortest_path_to_gens += path_length + verb_conf_sum
                
                sum_shortest_path_to_gens_vc_unweighted += path_length + np.sum([1 - verb_conf[node - gen_num] for node in shortest_path[1:]])
                reachable += 1
            except nx.NetworkXNoPath:
                continue
        
        scaling_factor = reachable / (n - 1)
        combined_centrality[i] = scaling_factor * reachable / sum_shortest_path_to_gens if reachable > 0 else 0

    return combined_centrality

def collect_all_values(df, key_required):
    corr = []
    for index, row in df.iterrows():
        for i, item in enumerate(row['pointwise_dict']):            
            corr.append(item[key_required])

    return np.array(corr)

def remove_non_alphanumeric(input_string):
    return re.sub(r'[^a-zA-Z0-9]', '', input_string).lower()
