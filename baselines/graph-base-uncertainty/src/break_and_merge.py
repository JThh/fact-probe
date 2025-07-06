import json
import os
import re
import logging
import pickle as pkl
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm

import src.utils as utils
from src.factscore_utils import FactScorer, General_Wiki_Eval
from src.models import get_model

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

###############################################################################
SYMBOL = 'Foo'
NOT_SYMBOL = 'Not Foo'

_PROMPT_PLACEHOLDER = '[PROMPT]'
_RESPONSE_PLACEHOLDER = '[RESPONSE]'
_STATEMENT_PLACEHOLDER = '[ATOMIC FACT]'

_RELEVANCE_FORMAT = f"""\
In a given RESPONSE, two subjects are considered "{SYMBOL}" if the RESPONSE \
contains information that explains how the two subjects are related.


Instructions:
1. The following STATEMENT has been extracted from the broader context of the \
given RESPONSE to the given QUESTION.
2. First, state the broad subject of the STATEMENT and the broad subject of \
the QUESTION.
3. Next, determine whether the subject of the STATEMENT and the subject of the \
QUESTION should be considered {SYMBOL}, based on the given definition of \
"{SYMBOL}."
4. Before showing your answer, think step-by-step and show your specific \
reasoning.
5. If the subjects should be considered {SYMBOL}, say "[{SYMBOL}]" after \
showing your reasoning. Otherwise show "[{NOT_SYMBOL}]" after showing your \
reasoning.
6. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". \
Some examples have been provided for you to learn how to do this task.

QUESTION:
{_PROMPT_PLACEHOLDER}

RESPONSE:
{_RESPONSE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""

_REVISE_FORMAT = f"""\
Vague references include but are not limited to:
- Pronouns (e.g., "his", "they", "her")
- Unknown entities (e.g., "this event", "the research", "the invention")
- Non-full names (e.g., "Jeff..." or "Bezos..." when referring to Jeff Bezos)


Instructions:
1. The following STATEMENT has been extracted from the broader context of the \
given RESPONSE.
2. Modify the STATEMENT by replacing vague references with the proper entities \
from the RESPONSE that they are referring to.
3. You MUST NOT change any of the factual claims made by the original STATEMENT.
4. You MUST NOT add any additional factual claims to the original STATEMENT.
5. Before giving your revised statement, think step-by-step and show your \
reasoning. As part of your reasoning, be sure to identify the subjects in the \
STATEMENT and determine whether they are vague references. If they are vague \
references, identify the proper entity that they are referring to and be sure \
to revise this subject in the revised statement.
6. After showing your reasoning, provide the revised statement and wrap it in \
a markdown code block.
7. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". \
Some examples have been provided for you to learn how to do this task.

STATEMENT:
{_STATEMENT_PLACEHOLDER}

RESPONSE:
{_RESPONSE_PLACEHOLDER}
"""

def strip_string(s):
    return s.strip()

def extract_first_square_brackets(text):
    import re
    match = re.search(r'\[([^\[\]]+)\]', text)
    result = match.group(1) if match else None
    logging.debug(f"Extracted from square brackets: {result}")
    return result

def extract_first_code_block(text, ignore_language=False):
    if not isinstance(text, str):
        logging.error(f"Expected string in extract_first_code_block, got {type(text)}")
        return None
    import re
    # If ignoring language, we allow: ``` + possible language + \n ... ```
    pattern = r'```(?:[\w\+\-\.]+\n)?([\s\S]+?)```' if ignore_language else r'```[\w\+\-\.]+\n([\s\S]+?)```'
    match = re.search(pattern, text)
    result = match.group(1).strip() if match else None
    logging.debug(f"Extracted code block: {result}")
    return result

def is_response_abstained(response, abstain_detection_type=None):
    """
    Basic function that checks if response is effectively 'abstaining', e.g.
    returning "I cannot answer" or "Insufficient data" based on `abstain_detection_type`.
    """
    if abstain_detection_type is None or abstain_detection_type.lower() == 'none':
        return False
    # In practice, use some advanced detection (like perplexity).
    # Here, we do a naive check
    low_resp = response.lower()
    if 'i cannot' in low_resp or 'i can\'t answer' in low_resp or 'not enough info' in low_resp:
        return True
    return False

def convert_to_claim_list(breakdown_dicts):
    """
    Converts a list of dictionaries (each containing a 'claim' key) 
    into a list of claim strings.
    """
    return [d['claim'] for d in breakdown_dicts]

class BreakdownProcessor:
    """
    Handles the decomposition of text generations into smaller atomic facts,
    caching and reusing existing breakdowns where possible.
    """
    def __init__(self, args, llm_model):
        self.args = args
        self.llm_model = llm_model
        self.breakdown_prompt = self._get_breakdown_prompt()

    def _get_breakdown_prompt(self):
        """
        Returns the base prompt used for requesting an LLM to deconstruct
        a paragraph into minimal self-contained facts.
        """
        return (
            "Please deconstruct the following paragraph into the smallest possible "
            "standalone self-contained facts without semantic repetition, and return "
            "the output as a jsonl, where each line is {{claim:[CLAIM], gpt-confidence:[CONF]}}.\n"
            "The confidence score [CONF] should represent your confidence in the claim, "
            "where a 1 is obvious facts and results like 'The earth is round' and '1+1=2'. "
            "A 0 is for claims that are very obscure or difficult for anyone to know, "
            "like the birthdays of non-notable people. The input is:\n'{text_generation}'"
        )

    def break_down_single(self, data, gen_id, cached_results):
        """
        Break down multiple generations from 'data' into atomic facts, either
        reusing cached breakdowns or generating new ones if not cached.
        """
        breakdown_dicts_list = []
        bd_raw = []

        # Merge the "most likely generation" + optional multiple generations
        generation_list = [data['most_likely_generation']] + data['more_generations'][:self.args.num_samples_for_claims]

        for bd_id, generation in enumerate(generation_list):
            if generation:
                # Build the breakdown prompt
                breakdown_prompt = self.breakdown_prompt.format(text_generation=generation)

                # Check if we can reuse cache
                if gen_id < len(cached_results['breakdown']) and bd_id < len(cached_results['breakdown'][gen_id]):
                    logging.info(
                        f"[Cache] Reusing breakdown for gen_id={gen_id}, generation index={bd_id}"
                    )
                    breakdown_raw_result = cached_results['breakdown'][gen_id][bd_id]

                    # Check if the prompt has changed
                    if generation not in breakdown_prompt:
                        mismatch_file = f'{self.args.folder_name}/breakdown_mismatch.txt'
                        with open(mismatch_file, 'a') as f:
                            f.write(
                                f"Prompt mismatch for gen_id={gen_id}, bd_id={bd_id}, "
                                f"cached prompt={breakdown_raw_result['prompt']} \n\n"
                                f" new breakdown prompt={breakdown_prompt}\n\n"
                            )
                else:
                    logging.info(
                        f"[Gen] Generating new breakdown for gen_id={gen_id}, generation index={bd_id}"
                    )
                    raw_result = self.llm_model.generate_given_prompt(breakdown_prompt)
                    breakdown_raw_result = {'return': raw_result, 'prompt': breakdown_prompt}

                # Parse the breakdown into structured dicts
                breakdown_dicts = self.get_subclaims(breakdown_raw_result['return'], 
                                                     response=generation, 
                                                     topic=data['entity'])
                breakdown_dicts = self._clean_breakdown_dicts(breakdown_dicts)

                breakdown_dicts_list.append(breakdown_dicts)
                bd_raw.append(breakdown_raw_result)
            else:
                # If generation is empty
                logging.warning(f"[Warning] Empty generation at index {bd_id}")
                breakdown_dicts_list.append([])
                bd_raw.append({'return': '', 'prompt': ''})

        # Update the cache with newly generated breakdown results
        self._update_cached_breakdown_results(gen_id, bd_raw, cached_results)
        return breakdown_dicts_list

    def _clean_breakdown_dicts(self, breakdown_dicts):
        """
        Cleans each breakdown dictionary: if 'claim' is a list, keep only the first element.
        """
        cleaned = []
        for d in breakdown_dicts:
            if isinstance(d['claim'], list):
                d['claim'] = d['claim'][0]
            cleaned.append(d)
        return cleaned

    def _update_cached_breakdown_results(self, gen_id, bd_raw, cached_results):
        """
        Updates the cache with the newly computed breakdowns if they are more complete
        than what was previously stored.
        """
        breakdown_cache = cached_results['breakdown']
        if gen_id >= len(breakdown_cache):
            breakdown_cache.append(bd_raw)
        elif len(bd_raw) > len(breakdown_cache[gen_id]):
            breakdown_cache[gen_id] = bd_raw

    def get_subclaims(self, completion, response=None, topic=None):
        """
        Convert the LLM completion into a list of subclaims (dicts). 
        Then revise each subclaim to handle pronouns or vague references.
        """
        output = self._extract_output_from_completion(completion)

        # Attempt direct JSON parse
        try:
            parsed = json.loads(output)
            subclaims = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            # Attempt bracket-wrapped array
            trimmed_output = output.strip().rstrip(',')
            wrapped_output = f"[{trimmed_output}]"
            try:
                subclaims = json.loads(wrapped_output)
            except json.JSONDecodeError:
                logging.warning(
                    "Encountered JSON decode error; falling back to line-by-line parse."
                )
                subclaims = self._parse_json_lines(output)

        # If not a list, wrap
        if not isinstance(subclaims, list):
            subclaims = [subclaims]

        # 1) For each subclaim, revise the claim text to remove vague references
        #    e.g. "He joined Google" => "Jeff Dean joined Google"
        final_claims = []
        for claim_dict in subclaims:
            if isinstance(claim_dict, dict) and 'claim' in claim_dict:
                original_claim = claim_dict['claim']
                if response is None or topic is None:
                    # Optionally pass the entire context from outside if you have it
                    # Or set `response = ''` if you have no bigger context
                    response = topic = ''
                revised_claim = self.revise_fact(response, original_claim)
                is_rel = self.check_relevance(f"Tell me a paragraph bio of {topic}.\n", response, revised_claim)
                if not is_rel:
                    logging.info("Not relevant. Skipping...")
                    continue
                final_claims.append({"claim": revised_claim, "gpt-confidence": claim_dict['gpt-confidence']}) 
            else:
                logging.warning(
                    "Subclaim not in expected format, skipping revision: %s", claim_dict
                )

        return final_claims
    
    def revise_fact(self, response, atomic_fact):
        """
        Uses the refinement prompt to remove vague references from `atomic_fact`.
        """
        full_prompt = _REVISE_FORMAT.replace(_STATEMENT_PLACEHOLDER, atomic_fact)
        full_prompt = full_prompt.replace(_RESPONSE_PLACEHOLDER, response)
        full_prompt = strip_string(full_prompt)

        try:
            model_response = self.llm_model.generate_given_prompt(full_prompt)["generation"]
            revised_fact = extract_first_code_block(model_response, ignore_language=True)
            if revised_fact:
                return revised_fact.strip()
            else:
                return atomic_fact  # fallback
        except Exception as e:
            logging.error(f"Error refining fact: {e}")
            return atomic_fact

    def check_relevance(self, prompt, response, atomic_fact):
        """
        Use the relevance prompt to determine if the atomic_fact is relevant 
        to the question (prompt) in the context of response.
        """
        full_prompt = _RELEVANCE_FORMAT.replace(_STATEMENT_PLACEHOLDER, atomic_fact)
        full_prompt = full_prompt.replace(_PROMPT_PLACEHOLDER, prompt)
        full_prompt = full_prompt.replace(_RESPONSE_PLACEHOLDER, response)
        full_prompt = strip_string(full_prompt)

        try:
            model_response = self.llm_model.generate_given_prompt(full_prompt)["generation"]
            answer = extract_first_square_brackets(model_response)
            if answer:
                return answer.strip().lower() == SYMBOL.lower()
            else:
                return True  # fallback
        except Exception as e:
            logging.error(f"Error checking relevance: {e}")
            return True  # fallback

    def _extract_output_from_completion(self, completion):
        """
        Extract the actual text from an LLM completion result. Remove code fences and 
        convert certain keys to valid JSON by forcibly rewriting them with quotes.
        """
        if 'generation' in completion:
            output = completion['generation']
        else:
            output = completion['choices'][0]["message"]["content"]

        # Remove any code fences
        output = (
            output
            .replace("```jsonl\n", "")
            .replace("```json\n", "")
            .replace("```", "")
        )

        # Replace keys if missing quotes
        output = (
            output
            .replace('claim:', '"claim":')
            .replace('gpt-confidence:', '"gpt-confidence":')
        )

        # Attempt to isolate just the JSON part from first '{' to last '}'
        start = output.find('{')
        end = output.rfind('}')
        if start != -1 and end != -1:
            output = output[start:end + 1]

        return output

    def _parse_json_lines(self, output):
        """
        Fallback line-by-line JSON parsing if direct parse fails.
        """
        results = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning(f"Skipping line due to JSON error: {line}")
        return results

class MatchProcessor:
    """
    Merges new claim lists into an 'Original Claim List' by verifying if each 
    new claim is entailed or not, then accumulating any non-entailed claims.
    """
    def __init__(self, args, llm_model):
        self.args = args
        self.llm_model = llm_model
        self.match_prompt = self._get_match_prompt()

    def _get_match_prompt(self):
        """
        Returns the instruction prompt for merging claim lists, verifying 
        if new claims are already covered or not, and enumerating extra ones.
        """
        return (
            "Given two lists titled \"Original Claim List\" and \"New Claim List\", "
            "your task is to integrate information from the \"New Claim List\" "
            "into the \"Original Claim List\". Please follow these detailed steps to "
            "ensure accuracy and clarity:\n\n"
            "Task 1. **Verification Process:**  Go through each statement in "
            "the \"New Claim List\" one by one, and determine if it is fully "
            "entailed or mentioned by any statement in the \"Original Claim List.\"\n\n"
            "Task 2. **Compilation of Non-Entailed Claims:** Generate a list of statements "
            "from the \"New Claim List\" that are not already covered or implied by the "
            "\"Original Claim List.\" For each new or unique claim that is not found in "
            "the original list, format your output by starting each line with '-'.\n\n"
            "**Original Claim List:**\n{{original_claim_list}}\n\n"
            "**New Claim List:**\n{{new_claim_list}}\n\n"
            "Begin with the Verification Process to assess each claim's uniqueness "
            "and relevance, then produce the list of claims that provide new insights."
        )

    def match_single(self, gen_id, breakdown_dicts_list, cached_results):
        """
        For each generation's breakdown list, merges them into a single list 
        of claims, using cached match results if available, or re-calling the LLM if not.
        """
        breakdown_dicts_list = breakdown_dicts_list.copy()  # local copy
        raw_match_results = []
        total_dicts_list = breakdown_dicts_list[0]

        for j, dicts_list in enumerate(breakdown_dicts_list[1:], start=1):
            if not dicts_list:
                logging.warning(
                    f"Empty breakdown for gen_id={gen_id}, breakdown index={j}"
                )
                continue

            # Turn the subclaims into string forms
            lst_tb_merged = convert_to_claim_list(dicts_list)
            point_list = convert_to_claim_list(total_dicts_list)
            
            print(lst_tb_merged, point_list)

            match_prompt = self.match_prompt.format(
                new_claim_list=self.invert_fact_list(lst_tb_merged),
                original_claim_list=self.invert_fact_list(point_list)
            )

            # Attempt to fetch from cached results
            if gen_id < len(cached_results['match']) and j - 1 < len(cached_results['match'][gen_id]):
                logging.info(f"[Cache] Found match for gen_id={gen_id}, idx={j}")
                match_raw_result = cached_results['match'][gen_id][j - 1]

                # Check for mismatch in prompt
                if match_raw_result['prompt'] != match_prompt:
                    mismatch_path = f'{self.args.folder_name}/match_mismatch.txt'
                    with open(mismatch_path, 'a') as f:
                        f.write(
                            f"Prompt mismatch for gen_id={gen_id}, breakdown={j}\n\n"
                            f"Cached prompt:\n{match_raw_result['prompt']}\n\n"
                            f"New prompt:\n{match_prompt}\n\n"
                        )
            else:
                logging.info(
                    f"[Gen] No cached match; regenerating for claim: {lst_tb_merged[0]}"
                )
                new_result = self.llm_model.generate_given_prompt(match_prompt)
                match_raw_result = {'return': new_result, 'prompt': match_prompt}

            # Extract textual portion
            if 'generation' in match_raw_result['return']:
                match_result = match_raw_result['return']['generation']
            else:
                match_result = match_raw_result['return']['choices'][0]["message"]["content"]

            # Gather additional points
            additional_points = self.robust_match_gpt_confidence(match_result, dicts_list)
            total_dicts_list += additional_points
            raw_match_results.append(match_raw_result)

        self._update_cached_match_results(gen_id, raw_match_results, cached_results)
        return total_dicts_list

    @staticmethod
    def invert_fact_list(fact_list):
        """
        Converts a list of facts into an enumerated string, one fact per line.
        """
        return '\n'.join(f"{i+1}. {fact}" for i, fact in enumerate(fact_list))

    def robust_match_gpt_confidence(self, text, dicts_list):
        """
        Searches for lines in 'text' that start with '-', indicating new or distinct claims.
        Each such line is associated with the best matching claim's gpt-confidence.
        """
        def calculate_overlap(c1, c2):
            words1 = set(c1.lower().split())
            words2 = set(c2.lower().split())
            return len(words1 & words2)

        lines = text.split('\n')
        compiled_list = []
        start_collecting = False

        for line in reversed(lines):
            line_stripped = line.strip()
            if line_stripped.startswith('-'):
                start_collecting = True
                item = line_stripped[1:].strip()
                item = item.split(":", 1)[1].strip() if ":" in item else item

                highest_overlap, highest_conf = 0, -1
                for d in dicts_list:
                    overlap = calculate_overlap(d['claim'], item)
                    if overlap > highest_overlap:
                        highest_overlap = overlap
                        highest_conf = d['gpt-confidence']
                compiled_list.insert(0, {
                    'claim': item, 
                    'gpt-confidence': highest_conf
                })
            elif start_collecting:
                # we've reached a region that might belong to higher-level text
                break

        return compiled_list

    def _update_cached_match_results(self, gen_id, raw_match_results, cached_results):
        """
        Updates the 'match' cache with new or more complete results for this generation.
        """
        match_cache = cached_results['match']
        if gen_id >= len(match_cache):
            match_cache.append(raw_match_results)
        elif len(raw_match_results) > len(match_cache[gen_id]):
            match_cache[gen_id] = raw_match_results

class AutoEvalWrapper:
    """
    A utility to wrap the fact-scoring or wiki evaluation process, caching results
    and reusing them if available.
    """
    def __init__(self, folder_name, args, gpt_annotate):
        self.folder_name = folder_name
        self.args = args
        self.args.folder_name = folder_name
        self.gpt_annotate = gpt_annotate

        # Initialize FactScorer / Wiki Evaluator
        self.fact_scorer = FactScorer(
            data_dir=os.environ["HF_DATASETS_CACHE"],
            cache_dir=os.environ["HF_DATASETS_CACHE"],
            args=args
        )
        self.general_wiki_eval = General_Wiki_Eval(
            error_file=f'{self.folder_name}/annotate_mismatch.txt',
            args=args
        )

        # Load cache from local file if present
        self.fs_cache = []
        self.fs_raw_path = f'{self.folder_name}/annotate_raw_all.json'
        if Path(self.fs_raw_path).exists():
            with open(self.fs_raw_path, 'r') as f:
                self.fs_cache = json.load(f)
            logging.info(f"Loaded from cache: {self.fs_raw_path}")

    def annotate_with_fact_scores(self, gen_id, data, all_claims_lst):
        """
        Applies either fact_scorer or wiki_evaluator to 'all_claims_lst' for the generation
        if not already cached, otherwise reuses existing cache.
        """
        mismatch_file = f'{self.folder_name}/annotate_mismatch.txt'
        if not os.path.exists(mismatch_file):
            with open(mismatch_file, 'w') as f:
                f.write('Checking for prompt mismatches\n\n')

        # If partially in cache
        if gen_id < len(self.fs_cache) and len(self.fs_cache[gen_id]) == 0:
            logging.info(f"[Cache] Found uncached facts for gen_id={gen_id}, entity={data['entity']}")

        cached = False
        not_cached_idx = []
        annotates = None
        fs_raw = None

        # If fully in cache
        if (gen_id < len(self.fs_cache) and 
            len(self.fs_cache[gen_id]) == 2):

            annotates, fs_raw = self.fs_cache[gen_id]
            logging.info(f"[Cache] Fetched factscore annotations for gen_id={gen_id}")

            extract_input = lambda x:re.search(r"(?s)Input:\s*(.*?)\nIs the input", x).group(1).strip()[:-1]

            # Check for mismatches in the prompts
            if 'factscore' in self.args.dataset:
                indices = []
                prompts = [extract_input(val["prompt"]) for val in fs_raw]
                for idx, claim in enumerate(all_claims_lst):
                    local_cached = False
                    for i, prompt in enumerate(prompts):
                        if prompt == claim:
                            indices.append(i)
                            local_cached = True
                    if not local_cached:
                        not_cached_idx.append(idx)
                if len(indices) < len(all_claims_lst):
                    logging.warning(f"{len(not_cached_idx)} claims not found in cached factscores")
                    cached = False
                else:
                    cached = True
                annotates = np.array(annotates)[indices].tolist()
                fs_raw = np.array(fs_raw)[indices].tolist()
                # for claim, fs_val in zip(all_claims_lst, fs_raw):
                #     if claim not in fs_val['prompt']:
                #         with open(mismatch_file, 'a') as f:
                #             f.write(
                #                 f"Prompt mismatch for gen_id={gen_id}, claim={claim}\n\n"
                #                 f"{fs_val['prompt']}\n\n"
                #             )
        if not cached and len(not_cached_idx) > 0:
            # Not or partially cached => run new annotation
            if 'factscore' in self.args.dataset:
                new_annotates, new_fs_raw = self.fact_scorer.fact_check_with_gpt(
                    topics=[data['entity']],
                    atomic_facts=[[all_claims_lst[idx] for idx in not_cached_idx]],
                    dataset=self.args.dataset
                )
            elif self.args.dataset in ['pop_qa_filtered', 'nq']:
                new_annotates, new_fs_raw = self.general_wiki_eval.general_wiki_eval(
                    topic=data['wiki_title'],
                    atomic_facts=all_claims_lst,
                    batch_size=10
                )
            else:
                raise ValueError(f"Unknown dataset: {self.args.dataset}")   

            final_annotates, final_fs_raw = [], []
            
            not_cached_idx_set = set(not_cached_idx)

            for idx, _ in enumerate(all_claims_lst):
                if idx in not_cached_idx_set:
                    final_annotates.append(new_annotates.pop(0))
                    final_fs_raw.append(new_fs_raw.pop(0))
                elif annotates is not None:
                    final_annotates.append(annotates.pop(0))
                    final_fs_raw.append(fs_raw.pop(0))

            assert len(final_annotates) == len(final_fs_raw) == len(all_claims_lst)

            # Insert or append new results into cache
            if gen_id <= len(self.fs_cache) - 1:
                self.fs_cache[gen_id] = (final_annotates, final_fs_raw)
            else:
                self.fs_cache.append((final_annotates, final_fs_raw))

            # Save updated cache
            with open(self.fs_raw_path, 'w') as f:
                json.dump(self.fs_cache, f, indent=2)

            annotates = final_annotates

        return annotates

class CacheManager:
    """
    Manages caching for breakdown results (match_raw_return.json) and 
    inline confidence results (vc_raw.json).
    """
    def __init__(self, folder_name, args):
        self.raw_results_path = f'{folder_name}/match_raw_return.json'
        self.collected_results_path = f'{folder_name}/{args.dataset}_{args.model}_bipartite.json'
        self.cached_results = {'breakdown': [], 'match': []}
        self.vc_raw = []
        self.vc_raw_file = f'{folder_name}/vc_raw.json'
        self._load_cache()

    def _load_cache(self):
        """
        Loads caches for breakdown and match from local JSON files if present.
        """
        if Path(self.raw_results_path).exists():
            with open(self.raw_results_path, 'r') as f:
                self.cached_results = json.load(f)
            logging.info(f"Loaded from cache: {self.raw_results_path}")
        if Path(self.vc_raw_file).exists():
            with open(self.vc_raw_file, 'r') as f:
                self.vc_raw = json.load(f)
            logging.info(f"Loaded vc_raw from cache: {self.vc_raw_file}")

    def save_cache(self):
        """
        Saves updated breakdown/match results and vc_raw to local JSON files.
        """
        with open(self.raw_results_path, 'w') as f:
            json.dump(self.cached_results, f, indent=2)
        with open(self.vc_raw_file, 'w') as f:
            json.dump(self.vc_raw, f, indent=2)

    def save_collected_results(self, breakdown_match):
        """
        Persists the final results (breakdown + match + correctness) 
        to a single JSON file for further analysis.
        """
        with open(self.collected_results_path, 'w') as outfile:
            json.dump(breakdown_match, outfile, indent=4)

class Break_And_Merge:
    """
    Main pipeline orchestrator for:
      1) Breaking down text generations into subclaims
      2) Merging them (match) across multiple generations
      3) (Optional) Annotating claims with fact-scorer or wiki-based correctness
    """
    def __init__(self, args, generations, llm_model, folder_name='', gpt_annotate=False, annotate_all=False):
        self.args = args
        self.generations = generations
        self.folder_name = folder_name
        self.llm_model = llm_model
        self.openai_model = get_model("gpt-4o-mini", args)

        self.gpt_annotate = gpt_annotate
        self.annotate_all = annotate_all

        self.breakdown_processor = BreakdownProcessor(args, llm_model=self.openai_model)
        self.match_processor = MatchProcessor(args, llm_model=self.openai_model)
        self.fact_scorer_wrapper = AutoEvalWrapper(folder_name, args, gpt_annotate)
        self.cache_manager = CacheManager(folder_name, args)

        # Pre-create mismatch files if needed
        mismatch_check = ['breakdown', 'match', 'annotate']
        for chk in mismatch_check:
            mismatch_path = f'{self.folder_name}/{chk}_mismatch.txt'
            if not os.path.exists(mismatch_path):
                with open(mismatch_path, 'w') as f:
                    f.write('Checking for prompt mismatches\n\n')

    def break_down_match(self):
        """
        The main routine:
          - For each generation, break it down into atomic claims
          - Then 'match' across multiple generations
          - Optionally annotate with a FactScorer
        """
        breakdown_match = self.generations.copy()

        # Process each generation
        for gen_id, data in tqdm(enumerate(self.generations), desc='Processing Breakdown and Match'):
            # 1) Break down the generation(s)
            breakdown_dicts_list = self.breakdown_processor.break_down_single(
                data, gen_id, self.cache_manager.cached_results
            )
            ml_breakdown = convert_to_claim_list(breakdown_dicts_list[0])
            logging.info(f"Breakdown for gen_id={gen_id} => {len(breakdown_dicts_list)} sub-lists")

            # 2) Merge claims across multiple sub-lists
            all_claims_dicts = self.match_processor.match_single(
                gen_id, breakdown_dicts_list, self.cache_manager.cached_results
            )
            all_claims_lst = convert_to_claim_list(all_claims_dicts)
            logging.info(f"gen_id={gen_id}, entity={data['entity']}, total claims => {len(all_claims_lst)}")

            # 3) Fact-scorer annotation if enabled
            if self.gpt_annotate:
                annotates = self.fact_scorer_wrapper.annotate_with_fact_scores(
                    gen_id, data, all_claims_lst
                )

            # Save updated caches after each generation
            self.cache_manager.save_cache()

            # Store partial results
            breakdown_match[gen_id]['most_likely_breakdown'] = ml_breakdown
            breakdown_match[gen_id]['most_likely_breakdown_len'] = len(ml_breakdown)
            breakdown_match[gen_id]['breakdown'] = all_claims_lst
            breakdown_match[gen_id]['breakdown_len'] = len(all_claims_lst)

            # Next: gather claim details for final processing
            dict_list, vc_raw_list = [], []
            for claim_id, claim_item in enumerate(all_claims_dicts):
                logging.debug(f"Processing claim_id={claim_id} => {claim_item['claim']}")
                # Get confidence
                vc, vc_raw_result = self._get_verbalized_confidence(
                    gen_id, claim_id, data, claim_item['claim']
                )
                vc_raw_list.append(vc_raw_result)

                # Possibly get a GPT-based correctness annotation
                gpt_annotation = ''
                if self.gpt_annotate:
                    gpt_annotation = self._get_gpt_annotation(claim_id, all_claims_lst, annotates)

                # Construct the final dictionary for each claim
                inline_dict = {
                    'claim': claim_item['claim'],
                    'correctness': '',
                    'inline_verbalized_confidence': 0.0,
                    'gpt-score': (
                        'S' 
                        if isinstance(gpt_annotation, dict) and
                           gpt_annotation.get('objectivity', '').strip().lower() in ['subjective','s']
                        else gpt_annotation
                    ),
                    'gpt_annotation_result': gpt_annotation,
                    'verbalized_confidence_with_options': vc,
                }
                dict_list.append(inline_dict)

            # Update the vc_raw cache with the newly processed claims
            self._update_vc_raw_cache(gen_id, vc_raw_list)

            # Attach final pointwise dict
            breakdown_match[gen_id]['pointwise_dict'] = dict_list

        # Save final results to disk
        self.cache_manager.save_collected_results(breakdown_match)
        return breakdown_match

    def _get_verbalized_confidence(self, gen_id, claim_id, data, new_breakdown):
        """
        Retrieve or compute a confidence measure for a single claim, 
        storing partial results in vc_raw cache.
        """
        vc_raw_result = None
        if gen_id < len(self.cache_manager.vc_raw) and claim_id < len(self.cache_manager.vc_raw[gen_id]):
            logging.info(f"Reusing cached vc results for entity={data['entity']}")
            vc_raw_result = self.cache_manager.vc_raw[gen_id][claim_id]

        vc, vc_raw_result = utils.get_verbalized_confidence(
            question=data['entity'],
            generation=new_breakdown,
            args=self.args,
            model_instance=self.llm_model,
            raw_result=vc_raw_result,
            problem_type='fact',
            with_options=True,
        )
        return vc, vc_raw_result

    def _update_vc_raw_cache(self, gen_id, vc_raw_list):
        """
        Store updated verbalized confidence data in the vc_raw cache.
        """
        vc_cache = self.cache_manager.vc_raw
        if gen_id >= len(vc_cache):
            logging.info("Appending new vc data to cache")
            vc_cache.append(vc_raw_list)
        elif len(vc_raw_list) > len(vc_cache[gen_id]):
            logging.info("Updating existing vc cache data")
            vc_cache[gen_id] = vc_raw_list
        with open(self.cache_manager.vc_raw_file, 'w') as f:
            json.dump(vc_cache, f, indent=2)

    def _get_gpt_annotation(self, idx, all_claims_lst, annotates):
        """
        Attempts to match a GPT annotation for the claim at index=idx. 
        If a mismatch occurs, returns '' (empty).
        """
        assert len(annotates) >= len(all_claims_lst), (
            "Mismatch: annotates fewer than claims."
        )
        annotation = annotates[idx] if idx < len(all_claims_lst) else ''
        if isinstance(annotation, dict):
            # Validate that content matches the claim
            if utils.remove_non_alphanumeric(annotation['content']) == utils.remove_non_alphanumeric(all_claims_lst[idx]):
                return annotation.get('gpt-score', '')
            else:
                return ''
        return annotation

    def plot_auroc(self, results):
        """
        Plots an AUROC curve for each of the relevant keys in 'results', 
        saves both the figure and stats in JSON.
        """
        first_n = len(results)
        sample_num = self.args.sc_samples
        key_map = {
            f'sc_score_{sample_num}samples': 'SC',
            'verbalized_confidence_with_options': 'PH-VC',
            f'breakdown_closeness_centrality_{sample_num}samples': r'$C_C$',
            f'sc_+_vc': 'SC+VC',
        }

        dict_collected = self.collect_data_for_plotting(results)
        result_dict = {}
        plt.figure(0).clf()

        data_size = len(dict_collected['corr'])
        result_dict['data_size'] = data_size

        for key, value in dict_collected.items():
            if key != 'corr':
                auroc = metrics.roc_auc_score(dict_collected['corr'], value)
                auprc = metrics.average_precision_score(dict_collected['corr'], value)
                auprc_n = metrics.average_precision_score(1 - dict_collected['corr'], -value)
                method_results = {
                    'auroc': auroc,
                    'auprc': auprc,
                    'auprc_n': auprc_n
                }
                if key in key_map:
                    fpr, tpr, _ = metrics.roc_curve(dict_collected['corr'], value)
                    label_str = (
                        f"{key_map[key]}, auc={round(auroc,3)}, "
                        f"auprc={round(auprc,3)}, auprc_n={round(auprc_n,3)}"
                    )
                    plt.plot(fpr, tpr, label=label_str)
                result_dict[key] = method_results

        df_stats = pd.DataFrame(dict_collected)
        plt.legend(loc='best')
        fig_path = f'{self.folder_name}/plot_roc_curve_{self.args.model}_{self.args.num_samples_for_claims}matches_{sample_num + 1}samples.png'
        json_path = f'{self.folder_name}/plot_roc_curve_{self.args.model}_{self.args.num_samples_for_claims}matches_{sample_num + 1}samples.json'
        
        plt.savefig(fig_path)
        with open(json_path, 'w') as f:
            json.dump(result_dict, f, indent=2)

    def collect_data_for_plotting(self, results):
        """
        Assembles a dictionary of arrays for each relevant key 
        (excluding 'claim', 'correctness', 'gpt-score', etc.). 
        'corr' becomes a 0/1 vector from 'gpt-score' (Y => 1, N => 0).
        """
        df = pd.DataFrame(results)
        sample_num = self.args.sc_samples

        corr = utils.collect_all_values(df, 'gpt-score')
        # Filter only Y or N for correlation
        idx = np.where((corr == 'Y') | (corr == 'N'))[0].astype(int)
        corr_bin = (corr == 'Y').astype(float)
        dict_collection = {'corr': corr_bin}

        # Identify the relevant keys from the first row's 'pointwise_dict'
        # e.g. ignoring keys that are obviously not numeric
        ignore_keys = ["claim", "correctness", "gpt-score", "gpt_annotation_result"]
        collect_keys = [
            k for k in df['pointwise_dict'][0][0].keys() 
            if k not in ignore_keys
        ]

        for key in collect_keys:
            vals = utils.collect_all_values(df, key).astype(np.float32)
            # Filter out invalid values
            # from the original script => skip -1, inf, or NaN
            valid_mask = (
                (vals != -1) & 
                (~np.isnan(vals)) & 
                (vals != np.inf)
            )
            # Intersect with indices that have Y/N
            idx = np.intersect1d(idx, np.where(valid_mask)[0].astype(int))
            dict_collection[key] = vals

        # Subselect only the valid rows in dict_collection
        for k in dict_collection:
            dict_collection[k] = dict_collection[k][idx]

        return dict_collection
