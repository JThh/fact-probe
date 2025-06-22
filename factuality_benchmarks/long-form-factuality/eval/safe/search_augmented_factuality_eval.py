# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Use a search-augmented LLM to evaluate factuality."""
import time
import collections
import dataclasses
import logging
from typing import Any, List, Optional

# pylint: disable=g-bad-import-order
from common import modeling
from common import utils
from eval.safe import classify_relevance
from eval.safe import get_atomic_facts
from eval.safe import rate_atomic_fact
# pylint: enable=g-bad-import-order

# Define constants for labeling
IRRELEVANT_LABEL = 'Irrelevant'
SUPPORTED_LABEL = rate_atomic_fact.SUPPORTED_LABEL
NOT_SUPPORTED_LABEL = rate_atomic_fact.NOT_SUPPORTED_LABEL

# Maximum number of retries for pipeline operations
_MAX_PIPELINE_RETRIES = 3

# Configure module-level logger
logger = logging.getLogger(__name__)

class CheckedStatement:
    """Class for storing checked statements."""

    def __init__(
        self,
        sentence: str,
        atomic_fact: str,
        self_contained_atomic_fact: str,
        relevance_data: dict[str, Any] | None = None,
        rate_data: rate_atomic_fact.FinalAnswer | None = None,
        annotation: str = '',
    ):
        """
        Initializes a CheckedStatement instance.

        Args:
            sentence (str): The original sentence containing the atomic fact.
            atomic_fact (str): The extracted atomic fact from the sentence.
            self_contained_atomic_fact (str): The atomic fact rephrased to be self-contained.
            relevance_data (dict[str, Any], optional): Data regarding the relevance classification.
            rate_data (rate_atomic_fact.FinalAnswer, optional): Data regarding the factuality rating.
            annotation (str, optional): The final annotation label.
        """
        self.sentence = sentence
        self.atomic_fact = atomic_fact
        self.self_contained_atomic_fact = self_contained_atomic_fact
        self.relevance_data = relevance_data
        self.rate_data = rate_data
        self.annotation = annotation

        # Compile all data into a dictionary for easy access and serialization
        self.data = {
            'sentence': self.sentence,
            'atomic_fact': self.atomic_fact,
            'self_contained_atomic_fact': self.self_contained_atomic_fact,
            'relevance_data': self.relevance_data if self.relevance_data else None,
            'rate_data': (
                dataclasses.asdict(self.rate_data) if self.rate_data else None
            ),
            'annotation': self.annotation,
        }

        logger.debug(
            f"Initialized CheckedStatement: {self.data}"
        )


def count_labels(checked_statements: List[CheckedStatement]) -> dict[str, int]:
    """
    Extract scores from the checked statements for a single response.

    Args:
        checked_statements (List[CheckedStatement]): List of checked statements.

    Returns:
        dict[str, int]: A dictionary mapping labels to their respective counts.
    """
    logger.info("Counting labels from checked statements.")
    result_dict = collections.defaultdict(int)

    # Initialize counts for known labels
    for label in [SUPPORTED_LABEL, IRRELEVANT_LABEL, NOT_SUPPORTED_LABEL]:
        result_dict[label] = 0
        logger.debug(f"Initialized count for label '{label}' to 0.")

    # Iterate through each checked statement and count annotations
    for statement in checked_statements:
        if not isinstance(statement, CheckedStatement):
            logger.warning(
                f"Encountered non-CheckedStatement object: {type(statement)}. Skipping."
            )
            continue
        if not statement.annotation:
            logger.debug("Found CheckedStatement without annotation. Skipping.")
            continue

        annotation_lower = statement.annotation.lower()
        if annotation_lower == SUPPORTED_LABEL.lower():
            result_dict[SUPPORTED_LABEL] += 1
            logger.debug(f"Incremented count for label '{SUPPORTED_LABEL}'.")
        elif annotation_lower == IRRELEVANT_LABEL.lower():
            result_dict[IRRELEVANT_LABEL] += 1
            logger.debug(f"Incremented count for label '{IRRELEVANT_LABEL}'.")
        elif annotation_lower == NOT_SUPPORTED_LABEL.lower():
            result_dict[NOT_SUPPORTED_LABEL] += 1
            logger.debug(f"Incremented count for label '{NOT_SUPPORTED_LABEL}'.")
        else:
            result_dict[statement.annotation] += 1
            logger.error(
                f"Unknown statement factuality type: {statement.annotation}. Incremented '{statement.annotation}' count."
            )

    logger.info(f"Label counts: {dict(result_dict)}")
    return dict(result_dict)


def classify_relevance_and_rate_single(
    prompt: str,
    response: str,
    sentence: str,
    atomic_fact: str,
    rater: modeling.Model,
) -> tuple[CheckedStatement, dict[str, Any], dict[str, Any]]:
    """
    Classify relevance of and rate a single atomic fact.

    Args:
        prompt (str): The original prompt given to the LLM.
        response (str): The LLM's response to the prompt.
        sentence (str): The sentence containing the atomic fact.
        atomic_fact (str): The extracted atomic fact from the sentence.
        rater (modeling.Model): The language model used for classification and rating.

    Returns:
        tuple[CheckedStatement, dict[str, Any], dict[str, Any]]:
            - The checked statement instance.
            - Revised fact dictionary after relevance classification.
            - Past steps dictionary from the rating process.
    """
    logger.info(
        f"Classifying relevance and rating atomic fact for sentence: '{sentence}'"
    )
    try:
        is_relevant, self_contained_atomic_fact, revised_fact_dict = (
            classify_relevance.main(
                prompt, response, atomic_fact=atomic_fact, model=rater
            )
        )
        logger.debug(
            f"Relevance classification result - is_relevant: {is_relevant}, "
            f"self_contained_atomic_fact: '{self_contained_atomic_fact}', "
            f"revised_fact_dict: {revised_fact_dict}"
        )
    except Exception as e:
        logger.error(
            f"Error during relevance classification for sentence '{sentence}': {e}"
        )
        raise

    if not is_relevant:
        logger.info(
            f"Atomic fact is irrelevant for sentence: '{sentence}'. Labeling as '{IRRELEVANT_LABEL}'."
        )
        checked_statement = CheckedStatement(
            sentence=sentence,
            atomic_fact=atomic_fact,
            self_contained_atomic_fact=self_contained_atomic_fact,
            relevance_data=revised_fact_dict,
            annotation=IRRELEVANT_LABEL,
        )
        return checked_statement, revised_fact_dict, {}

    logger.info(
        f"Atomic fact is relevant for sentence: '{sentence}'. Proceeding to factuality rating."
    )
    try:
        rate_data, past_steps_dict = rate_atomic_fact.check_atomic_fact(
            atomic_fact=self_contained_atomic_fact, rater=rater
        )
        logger.debug(
            f"Factuality rating result - rate_data: {rate_data}, past_steps_dict: {past_steps_dict}"
        )
    except Exception as e:
        logger.error(
            f"Error during factuality rating for atomic fact '{self_contained_atomic_fact}': {e}"
        )
        raise

    if not isinstance(rate_data, rate_atomic_fact.FinalAnswer):
        error_msg = 'No rate data found for atomic fact.'
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(
        f"Factuality rating completed with annotation '{rate_data.answer}' for atomic fact '{self_contained_atomic_fact}'."
    )
    checked_statement = CheckedStatement(
        sentence=sentence,
        atomic_fact=atomic_fact,
        self_contained_atomic_fact=self_contained_atomic_fact,
        relevance_data=revised_fact_dict,
        rate_data=rate_data,
        annotation=rate_data.answer,
    )

    return checked_statement, revised_fact_dict, past_steps_dict


def classify_relevance_and_rate(
    prompt: str,
    response: str,
    sentences_and_atomic_facts: List[dict[str, Any]],
    rater: modeling.Model,
) -> dict[str, Any]:
    """
    Classify relevance of and rate all given atomic facts.

    Args:
        prompt (str): The original prompt given to the LLM.
        response (str): The LLM's response to the prompt.
        sentences_and_atomic_facts (List[dict[str, Any]]): List of sentences and their corresponding atomic facts.
        rater (modeling.Model): The language model used for classification and rating.

    Returns:
        dict[str, Any]: Aggregated results including checked statements, revised facts, past steps, and label counts.
    """
    logger.info("Starting relevance classification and factuality rating for all atomic facts.")
    checked_statements, revised_fact_dicts, past_steps_dicts = [], [], []

    for sentence_data in sentences_and_atomic_facts:
        sentence = sentence_data.get('sentence', '')
        atomic_facts = sentence_data.get('atomic_facts', [])
        logger.debug(
            f"Processing sentence: '{sentence}' with {len(atomic_facts)} atomic facts."
        )

        for atomic_fact in atomic_facts:
            checked_statement, revised_fact_dict, past_steps_dict = None, {}, {}
            num_fails = 0

            while checked_statement is None and num_fails < _MAX_PIPELINE_RETRIES:
                try:
                    checked_statement, revised_fact_dict, past_steps_dict = (
                        classify_relevance_and_rate_single(
                            prompt=prompt,
                            response=response,
                            sentence=sentence,
                            atomic_fact=atomic_fact,
                            rater=rater,
                        )
                    )
                    logger.debug(
                        f"Successfully classified and rated atomic fact '{atomic_fact}' for sentence '{sentence}'."
                    )
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error(
                        f"Attempt {num_fails + 1} failed for atomic fact '{atomic_fact}' in sentence '{sentence}': {e}"
                    )
                    checked_statement, revised_fact_dict, past_steps_dict = None, {}, {}
                    num_fails += 1
                    time.sleep(1)  # Optional: wait before retrying

            if isinstance(checked_statement, CheckedStatement):
                checked_statements.append(checked_statement)
                revised_fact_dicts.append(revised_fact_dict)
                past_steps_dicts.append(past_steps_dict)
                logger.debug(
                    f"Added CheckedStatement for atomic fact '{atomic_fact}' in sentence '{sentence}'."
                )
            else:
                logger.warning(
                    f"Failed to process atomic fact '{atomic_fact}' in sentence '{sentence}' after {num_fails} attempts."
                )

    label_counts = count_labels(checked_statements=checked_statements)
    aggregated_result = {
        'checked_statements': [item.data for item in checked_statements],
        'revised_fact_jsonified_all': revised_fact_dicts,
        'past_steps_jsonified_all': past_steps_dicts,
        **label_counts,
    }

    logger.info(
        f"Completed relevance classification and factuality rating. Aggregated result: {aggregated_result}"
    )
    return aggregated_result


def main(prompt: str, response: str, rater: modeling.Model) -> dict[str, Any]:
    """
    Main function to evaluate factuality using search-augmented LLM.

    Args:
        prompt (str): The original prompt given to the LLM.
        response (str): The LLM's response to the prompt.
        rater (modeling.Model): The language model used for classification and rating.

    Returns:
        dict[str, Any]: Comprehensive evaluation results including prompts, responses, atomic facts, and ratings.
    """
    logger.info("Starting main factuality evaluation process.")

    try:
        atomic_facts = get_atomic_facts.main(response=response, model=rater)
        logger.debug(f"Extracted atomic facts: {atomic_facts}")
    except Exception as e:
        logger.error(f"Error extracting atomic facts: {e}")
        raise

    try:
        rating_result = classify_relevance_and_rate(
            prompt=prompt,
            response=response,
            sentences_and_atomic_facts=atomic_facts.get('all_atomic_facts', []),
            rater=rater,
        )
        logger.debug(f"Rating result: {rating_result}")
    except Exception as e:
        logger.error(f"Error during relevance classification and factuality rating: {e}")
        raise

    final_result = {
        'prompt': prompt,
        'response': response,
        **atomic_facts,
        **rating_result
    }

    logger.info(f"Completed main evaluation. Final result: {final_result}")
    return final_result
