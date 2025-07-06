"""Different prompts (and their logic) used to check for semantic equivalence."""
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import utils
from utils import (
    extract_questions,
    log_w_indent,
    cluster_assignment_entropy,
    llama_predict_w_log,
    md5hash,
    predict_w_log,
)

# pylint: disable=unnecessary-lambda-assignment, invalid-name, line-too-long
# pylint: disable=missing-function-docstring, missing-class-docstring

GEN_QS = 'gen_qs'
CHECK_PROP = 'check_prop'
ANSWER_QS = 'answer_qs'
EQUIVALENCE = 'equivalence'

predict_func = lambda category: llama_predict_w_log if category in [ANSWER_QS, CHECK_PROP] else predict_w_log


class SpoofData:
    def __getitem__(self, item):
        return f'<{item}>'


class BaseModel:
    def __init__(
        self,
        *,
        n_questions,
        n_regenerate,
        n_stochastic_questions,
        restored,
        restore_stages,
        accept_restore_failure,
        entailment_type
    ):
        super().__init__()
        self.n_questions = n_questions
        self.n_regenerate = n_regenerate
        self.n_stochastic_questions = n_stochastic_questions
        self.entailment_type = entailment_type

        self.restored = restored
        self.restore_stages = restore_stages
        self.accept_restore_failure = accept_restore_failure
        # Dict of dict of list.
        self.export_predictions = defaultdict(lambda: defaultdict(list))

    def predict_w_log(self, prompt, indent, qidx, stage, reuse=False):
        prompt_hash = md5hash(prompt)
        if stage in self.restore_stages and (restored := self.restored.get(qidx, {})):
            if self.accept_restore_failure and not len(restored.get(prompt_hash, [])):
                logging.warning('Spoofing aborted!')
                prediction = predict_func(stage)(prompt, indent)
            else:
                # Re-use predictions from previous runs.
                log_w_indent(f'Spoofed Input: {prompt}', indent)
                prediction = restored[prompt_hash].pop(0)
                log_w_indent(f'Spoofed Output: {prediction}', indent, symbol='xx')
        elif reuse and (previous := self.export_predictions.get(qidx, {}).get(prompt_hash, False)):
            # Typically avoid reusing, but beneficial for deterministic answers.
            log_w_indent(f'Reused Input: {prompt}', indent)
            prediction = previous[0]
            log_w_indent(f'Reused Output: {prediction}', indent, symbol='xx')
        else:
            prediction = predict_func(stage)(prompt, indent)

        self.export_predictions[qidx][prompt_hash].append(prediction)
        return prediction

    def gen_facts(self, data):
        del data
        return (
            "Please list the specific factual propositions included in the answer above. "
            "Be complete and do not leave any factual claims out. Provide each claim as a "
            "separate sentence in a separate bullet point."
        )

    def get_all_prompts_for_log(self):
        # Spoof data to log prompting format.
        data = SpoofData()
        prompts = {
            'gen_facts': self.gen_facts(data),
            'base_gen_questions': self.base_gen_questions(data),
            'base_answer_question': self.base_answer_question(data),
            'base_equivalence': self.base_equivalence(data),
        }
        return prompts

    def base_gen_questions(self, data):
        del data
        raise NotImplementedError

    def base_answer_question(self, data):
        del data
        raise NotImplementedError

    def base_equivalence(self, data):
        del data
        raise NotImplementedError


class QAEquivalent(BaseModel):
    """Questions from context with ground truth answer. Short answers with context. LLM Entailment without context."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def base_gen_questions(self, data):
        instruction = (
            f"Generate a list of {self.n_questions} questions that might have generated the sentence in the context "
            "of the preceding original text, as well as their answers. Please do not use specific facts that appear "
            "in the follow-up sentence when formulating the question.\n"
            "Make the questions and answers diverse. Avoid yes-no questions.\n"
            'The answers should not be a full sentence and as short as possible, e.g., only a name, place, or thing. '
            'Use the format "1. {{question}} -- {{answer}}"'
        )

        if data['text_so_far'] is None:
            return f"""You see the sentence:

{data["proposition"]}

{instruction}"""
        else:
            return f"""Following this text:

{data["text_so_far"]}

You see the sentence:

{data["proposition"]}

{instruction}"""

    def base_answer_question(self, data):

        if data['text_so_far'] is None:
            return f"""We are writing an answer to the question "{data["user_question"]}". First, we observe the following question:

Question: {data["question"]}
Answer:"""

        else:
            return f"""We are writing an answer to the request "{data["user_question"]}". So far we have written:

{data["text_so_far"]}


Question: {data["question"]}

Answer:"""

    # def base_answer_question(self, data):
    #     # Enhanced instruction for clarity and conciseness
    #     instruction = (
    #         "Please provide a concise answer to the following question. "
    #         "Your response should be short, using only a few words such as a name, place, or thing. "
    #         "Avoid full sentences or elaborate explanations."
    #     )

    #     if data['text_so_far'] is None:
    #         return f"""We are composing an answer to the question "{data["user_question"]}". First, consider the following question:

    # Question: {data["question"]}

    # {instruction}"""
    #     else:
    #         return f"""We are continuing to compose an answer to the request "{data["user_question"]}". So far, we have written:

    # {data["text_so_far"]}

    # Question: {data["question"]}

    # {instruction}"""

    def base_equivalence(self, data):
        prompt = 'Are the following answers equivalent?'
        for i in range(1, self.n_regenerate + 2):
            prompt += f'\nPossible Answer {i}: ' + '{}'
        prompt += '\nRespond only with "yes" or "no".'

        return prompt.format(data['expected_answers'], *data['regen_answers'])

    def check_truth(self, *, rp, wait, data):
        uq = data['didx']

        gen_questions_prompt = self.base_gen_questions(data)
        expected_answers, questions = [], []
        for _ in range(self.n_stochastic_questions):
            success = False
            while not success:
                try:
                    gen_questions = self.predict_w_log(gen_questions_prompt, 2, uq, GEN_QS).split('\n')
                    expected_answers.extend([q.split(' -- ')[1] for q in gen_questions if q])
                    questions.extend([q[3:].split(' -- ')[0] for q in gen_questions if q])
                    success = True
                except Exception as e:
                    logging.warning('Retrying `gen_questions`, failed with error: %s', e)

        log_w_indent(f'Extracted questions: {questions}', 2)
        log_w_indent(f'Extracted expected answers: {expected_answers}', 2)
        wait()

        uncertainties = []
        for qidx, (expected_answer, question) in enumerate(zip(expected_answers, questions)):
            log_w_indent(f'Regenerate answers for question {qidx} "{question}":', 2)

            regen_answers = []
            # Answer each question multiple times
            fdata = {**data, 'question': question}
            for _ in range(self.n_regenerate):
                answer_prompt = self.base_answer_question(fdata)
                answer = self.predict_w_log(answer_prompt, 3, uq, ANSWER_QS)
                regen_answers.append(answer)

            # Check if answers are equivalent
            if self.__class__.__name__ in ['QADebertaEntailment', 'QALLMEntailment']:
                answers = [expected_answer, *regen_answers]
                clusters, uncertainty = self.get_semantic_uncertainty(answers, fdata)

                # Account for GPT refusal to answer questions.
                stop_words = ['not available', 'not provided', 'unknown', 'unclear']
                unknown_count = sum(
                    any(stop_word in answer.lower() for stop_word in stop_words)
                    for answer in answers
                )
                if unknown_count >= len(answers) // 2:
                    logging.warning('Not answerable, setting uncertainty to maximum.')
                    uncertainty = -np.log(1 / len(answers))
                    clusters = f"{clusters} not answerable!"

                log_w_indent(f'Semantic Clustering Input: {answers}', 3)
                log_w_indent(
                    f'Semantic Clustering Output: {clusters}, uncertainty: {uncertainty}', 3
                )
                equiv_response = clusters
            else:
                equiv_prompt = self.base_equivalence({
                    'expected_answers': expected_answer,
                    'regen_answers': regen_answers
                })
                equiv_response = self.predict_w_log(equiv_prompt, 3, uq, EQUIVALENCE)
                uncertainty = utils.get_yes_no(equiv_response)

            uncertainties.append(uncertainty)

            rp[f'question-{qidx}'] = {
                'question': question,
                'answers': regen_answers,
                'expected_answer': expected_answer,
                'equiv_response': equiv_response,
                'uncertainty': uncertainty,
            }
            wait()

        return np.mean(uncertainties)


class QADebertaEntailment(QAEquivalent):
    """Questions from context with ground truth answer. Short answers with context. Deberta Entailment."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = 'microsoft/deberta-v2-xlarge-mnli'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

    def get_all_prompts_for_log(self):
        # Spoof data to log prompting format.
        data = SpoofData()
        prompts = {
            'gen_facts': self.gen_facts(data),
            'base_gen_questions': self.base_gen_questions(data),
            'base_answer_question': self.base_answer_question(data),
        }
        return prompts

    def get_semantic_uncertainty(self, answers, fdata):
        semantic_ids = self.get_semantic_ids(answers, fdata)
        uncertainty = cluster_assignment_entropy(semantic_ids)
        return semantic_ids, uncertainty

    def are_equivalent(self, text1, text2, data=None):
        del data

        def check_implication(t1, t2):
            inputs = self.tokenizer(t1, t2, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            largest_index = torch.argmax(probabilities).cpu().item()
            # Deberta-MNLI returns `neutral` and `entailment` at indices 1 and 2.
            return largest_index

        implication_1 = check_implication(text1, text2)
        implication_2 = check_implication(text2, text1)
        assert implication_1 in [0, 1, 2] and implication_2 in [0, 1, 2]
        implications = [implication_1, implication_2]

        if self.entailment_type == 'lax':
            # Check if none are contradictions and not both neutral.
            semantically_equivalent = (0 not in implications) and (implications != [1, 1])
        elif self.entailment_type == 'strict':
            semantically_equivalent = (implications[0] == 2) and (implications[1] == 2)
        else:
            raise ValueError("Invalid entailment_type. Choose 'lax' or 'strict'.")

        return semantically_equivalent

    def get_semantic_ids(self, strings_list, data):
        """Group list of predictions into semantic meaning."""
        semantic_set_ids = [-1] * len(strings_list)
        next_id = 0
        for i, string1 in enumerate(strings_list):
            if semantic_set_ids[i] == -1:
                semantic_set_ids[i] = next_id
                for j in range(i + 1, len(strings_list)):
                    if self.are_equivalent(string1, strings_list[j], data):
                        semantic_set_ids[j] = next_id
                next_id += 1
        assert -1 not in semantic_set_ids
        return semantic_set_ids


class QALLMEntailment(QADebertaEntailment):
    """QALLM Entailment Model."""

    def get_all_prompts_for_log(self):
        # Spoof data to log prompting format.
        data = SpoofData()
        prompts = {
            'gen_facts': self.gen_facts(data),
            'base_gen_questions': self.base_gen_questions(data),
            'base_answer_question': self.base_answer_question(data),
            'base_equivalence': self.base_equivalence(data),
        }
        return prompts

    def base_equivalence(self, data):
        prompt = f'We are writing an answer to the question "{data["user_question"]}".'

        if data['text_so_far'] is None:
            prompt += f' First, we are trying to answer the subquestion "{data["question"]}".\n'
        else:
            prompt += f""" So far we have written:

{data["text_so_far"]}

Next, we are trying to answer the subquestion "{data["question"]}".
Does at least one of the following two possible answers entail the other?

Possible Answer 1: {data["text1"]}
Possible Answer 2: {data["text2"]}

Respond with yes or no."""

        return prompt

    def are_equivalent(self, text1, text2, data):
        if text1 == text2:
            log_w_indent(f'Skip entailment check: {text1} == {text2}.', 3)
            return True

        equivalence_prompt = self.base_equivalence({'text1': text1, 'text2': text2, **data})
        equivalence = self.predict_w_log(equivalence_prompt, 3, data['didx'], EQUIVALENCE, reuse=True)
        uncertainty = utils.get_yes_no(equivalence)

        # If yes in equivalence --> uncertainty == 0 --> return True.
        return {0: True, 1: False}.get(uncertainty, False)


class SelfCheckBaseline(BaseModel):
    """Baseline model that simply asks if a proposition is true."""

    def get_all_prompts_for_log(self):
        # Spoof data to log prompting format.
        data = SpoofData()
        prompts = {'base_check_prop': self.base_check_prop(data)}
        return prompts

    def base_check_prop(self, data):
        """More instructive prompt for less tuned models."""
        instruction = "Decide whether the following statement is likely true or not. Respond with 'yes' or 'no'."

        example = """Example:
        Statement: The Earth orbits the Sun.
        Answer: yes

        """

        if data['text_so_far'] is None:
            return f"""{example}{instruction}

        Statement: {data["proposition"]}
        Answer:"""
        else:
            return f"""{example}Following this text:

        {data["text_so_far"]}

        {instruction}

        Statement: {data["proposition"]}
        Answer:"""

    def check_truth(self, *, rp, wait, data):
        uq = data['didx']

        check_prop_prompt = self.base_check_prop(data)
        prop_truth_answer = self.predict_w_log(check_prop_prompt, 2, uq, CHECK_PROP)
        wait()

        uncertainty = utils.get_yes_no(prop_truth_answer)

        rp['prop_truth_answer'] = prop_truth_answer
        rp['uncertainties'] = [uncertainty]

        return uncertainty


class PTrueOriginalBaseline(BaseModel):
    """Adaptation of p(True) baseline for paragraph-length generations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def base_gen_questions(self, data):
        instruction = (
            "Generate a question that, in the context of the preceding original text, might have generated the "
            "follow-up sentence. Please do not use specific facts that appear in the follow-up sentence when "
            "formulating the question. Provide only the text of the question with no additional text."
        )

        if data['text_so_far'] is None:
            return f"""You see the sentence:

{data["proposition"]}

{instruction}"""
        else:
            return f"""Following this text:

{data["text_so_far"]}

You see the sentence:

{data["proposition"]}

{instruction}"""

    def base_answer_question(self, data):
        instruction = "Respond with a single sentence that answers the question concisely."

        if data['text_so_far'] is None:
            return f"""We are writing an answer to the question "{data["user_question"]}". First, we observe the following question:

{data["question"]}

{instruction}"""
        else:
            return f"""We are writing an answer to the question "{data["user_question"]}". So far we have written:

{data["text_so_far"]}

The next sentence should be the answer to the following question:

{data["question"]}

{instruction}"""

    def base_equivalence(self, data):
        """More instructive prompt for less tuned models."""
        # Include an example to demonstrate the expected behavior
        example_prompt = (
            'Example:\n'
            'Question: What is the capital of France?\n'
            'Here are some brainstormed ideas:\n'
            'Paris\n'
            'London\n'
            'Possible Answer: Paris\n'
            'Is the possible answer true based on the question?\n'
            'Answer: yes\n\n'
        )
        # Start building the actual prompt
        prompt = example_prompt
        prompt += f'Question: {data["question"]}\n'
        prompt += 'Here are some brainstormed ideas:\n'
        # Include the proposition and regenerated answersp
        import pdb; pdb.set_trace()
        all_answers = [data['proposition']] + data['regen_answers']
        for answer in all_answers:
            prompt += f'{answer}\n'
        prompt += f'Possible Answer: {data["proposition"]}\n'
        prompt += 'Is the possible answer true based on the question?\n'
        prompt += 'Answer:'
        return prompt

    def check_truth(self, *, rp, wait, data):
        uq = data['didx']

        gen_questions_prompt = self.base_gen_questions(data)
        question = self.predict_w_log(gen_questions_prompt, 2, uq, GEN_QS)
        wait()

        # Answer question multiple times
        regen_answers = []
        for _ in range(self.n_regenerate):
            answer_prompt = self.base_answer_question({**data, 'question': question})
            answer = self.predict_w_log(answer_prompt, 3, uq, ANSWER_QS)
            regen_answers.append(answer)

        # Check if answers are equivalent
        equiv_prompt = self.base_equivalence({
            **data,
            'regen_answers': regen_answers,
            'question': question
        })

        # Monte Carlo sampling to approximate token probabilities.
        uncertainties = []
        for i in range(10):
            equiv_response = self.predict_w_log(equiv_prompt, 3, uq, EQUIVALENCE)
            print(f'Averaging over different samples at T=1: prediction {i}: {equiv_response}')
            uncertainties.append(utils.get_yes_no(equiv_response))
        uncertainty = np.mean(uncertainties)
        print(f'Final uncertainty: {uncertainty}')

        wait()

        rp['question-0'] = {
            'question': question,
            'answers': regen_answers,
            'equiv_response': equiv_response,
            'uncertainty': uncertainty,
        }

        return uncertainty


all_models = {
    'QAEquivalent': QAEquivalent,
    'QADebertaEntailment': QADebertaEntailment,
    'QALLMEntailment': QALLMEntailment,
    'SelfCheckBaseline': SelfCheckBaseline,
    'PTrueOriginalBaseline': PTrueOriginalBaseline,
}
