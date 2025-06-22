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
"""All code for loading prompts and data."""

import logging
import random
import re
from typing import Iterator, Optional, List

# pylint: disable=g-bad-import-order
from common import longfact
from common import utils
# pylint: enable=g-bad-import-order

# Define constants
DEFAULT_CUSTOM_PROMPTS = [
    'Who is Quoc V. Le?',
    'Who is Xochitl Gomez?',
    'What happened in the first modern Olympics?',
    'What happened during the 5th Academy Awards?',
    'Tell me about the company Acorns.',
    'Tell me about the Carnegie Steel Company.',
    'Give me an introduction of the city of Bucaramanga.',
    'Give me an introduction of the city of Khartoum.',
    'Tell me about the Antiguan racer snake.',
    'Tell me about the South Philippine Dwarf Kingfisher.',
]
PROMPT_FIELD = 'prompt'
CORRECT_FIELD = 'correct_answers'
INCORRECT_FIELD = 'incorrect_answers'
TASK_TUPLE_LENGTH = 4

_NONE = 'none'
_PER_PROMPT_DATA_FIELD = 'per_prompt_data'

# Configure module-level logger
logger = logging.getLogger(__name__)

# Set up logging configuration (can be adjusted as needed)
def setup_logging():
    """
    Sets up the logging configuration for the module.
    """
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # Adjust as needed (INFO, WARNING, etc.)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    if not logger.handlers:
        logger.addHandler(ch)


def is_biography_prompt(prompt: str) -> bool:
    """
    Determines whether a given prompt is asking for a biography.

    Args:
        prompt (str): The prompt string to evaluate.

    Returns:
        bool: True if the prompt is a biography request, False otherwise.
    """
    # Define regex patterns for biography-related prompts
    bio_patterns = [
        r'^who\s+is\s+.+\?$',
        r'^tell\s+me\s+a\s+bio\s+of\s+.+\.$',
    ]

    # Check if the prompt matches any of the biography patterns
    for pattern in bio_patterns:
        if re.match(pattern, prompt.strip(), re.IGNORECASE):
            logger.debug(f"Prompt matched biography pattern: '{prompt}'")
            return True

    logger.debug(f"Prompt did not match any biography patterns: '{prompt}'")
    return False


class DataPackage:
    """Wraps a set of prompts."""

    def __init__(self) -> None:
        self.prompts: List[str] = []
        self.correct_answers: Optional[List[List[str]]] = []
        self.incorrect_answers: Optional[List[List[str]]] = []
        self.load_path: str = ''
        self.prompt_field_name: str = ''
        self.correct_answer_field_name: str = ''
        self.incorrect_answer_field_name: str = ''

    def verify_lengths(self) -> bool:
        """
        Verifies that the lengths of prompts, correct_answers, and incorrect_answers match.

        Returns:
            bool: True if lengths match, False otherwise.
        """
        lengths_match = (
            len(self.prompts)
            == len(self.correct_answers)
            == len(self.incorrect_answers)
        )
        if not lengths_match:
            logger.error(
                "Mismatch in lengths: "
                f"prompts({len(self.prompts)}), "
                f"correct_answers({len(self.correct_answers)}), "
                f"incorrect_answers({len(self.incorrect_answers)})"
            )
        return lengths_match

    def iterate(self) -> Iterator[tuple[str, List[str], List[str]]]:
        """
        Creates an iterator over the prompts and their corresponding answers.

        Returns:
            Iterator[tuple[str, List[str], List[str]]]: Iterator of (prompt, correct_answers, incorrect_answers).
        """
        return zip(self.prompts, self.correct_answers, self.incorrect_answers)

    def num_items(self) -> int:
        """
        Returns the number of items in the DataPackage.

        Returns:
            int: Number of prompts, or -1 if lengths do not match.
        """
        if not self.verify_lengths():
            return -1
        return len(self.prompts)

    def load_from_filepath(
        self,
        filepath: str,
        prompt_field_name: str,
        correct_answer_field_name: str,
        incorrect_answer_field_name: str,
    ) -> bool:
        """
        Loads a file into the data package, filtering out biography prompts.

        Args:
            filepath (str): Path to the JSONL file.
            prompt_field_name (str): Key for the prompt in each data point.
            correct_answer_field_name (str): Key for correct answers.
            incorrect_answer_field_name (str): Key for incorrect answers.

        Returns:
            bool: True if loading is successful and lengths match, False otherwise.
        """
        logger.info(f"Loading data from filepath: '{filepath}'")
        data = utils.read_from_jsonlines(filepath)

        for idx, data_point in enumerate(data, start=1):
            prompt = data_point.get(prompt_field_name, '').strip()
            if not prompt:
                logger.warning(f"Data point {idx} missing prompt. Skipping.")
                continue

            if is_biography_prompt(prompt):
                logger.info(f"Filtered out biography prompt: '{prompt}'")
                continue  # Skip biography prompts

            self.prompts.append(prompt)
            logger.debug(f"Loaded prompt: '{prompt}'")

            # Load correct answers
            if not correct_answer_field_name or correct_answer_field_name.lower() == _NONE:
                self.correct_answers.append([])
                logger.debug("No correct answers provided.")
            elif correct_answer_field_name in data_point:
                correct_answers = [ans.strip() for ans in data_point[correct_answer_field_name].split('; ') if ans.strip()]
                self.correct_answers.append(correct_answers)
                logger.debug(f"Loaded correct answers: {correct_answers}")
            else:
                error_msg = f"Invalid field '{correct_answer_field_name}' in data point {idx}."
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Load incorrect answers
            if not incorrect_answer_field_name or incorrect_answer_field_name.lower() == _NONE:
                self.incorrect_answers.append([])
                logger.debug("No incorrect answers provided.")
            elif incorrect_answer_field_name in data_point:
                incorrect_answers = [ans.strip() for ans in data_point[incorrect_answer_field_name].split('; ') if ans.strip()]
                self.incorrect_answers.append(incorrect_answers)
                logger.debug(f"Loaded incorrect answers: {incorrect_answers}")
            else:
                error_msg = f"Invalid field '{incorrect_answer_field_name}' in data point {idx}."
                logger.error(error_msg)
                raise ValueError(error_msg)

        self.load_path = filepath
        self.prompt_field_name = prompt_field_name
        self.correct_answer_field_name = correct_answer_field_name
        self.incorrect_answer_field_name = incorrect_answer_field_name
        logger.info("Finished loading data from filepath.")

        return self.verify_lengths()

    def load_from_results_json(self, filepath: str) -> bool:
        """
        Loads prompts from a saved results JSON from a run, filtering out biography prompts.

        Args:
            filepath (str): Path to the results JSON file.

        Returns:
            bool: True if loading is successful and lengths match, False otherwise.
        """
        logger.info(f"Loading data from results JSON: '{filepath}'")
        try:
            results = utils.read_json(filepath)
            logger.debug(f"Loaded JSON data: {results}")
        except IOError as e:
            logger.error(f"IOError while reading JSON file '{filepath}': {e}")
            logger.info("Reverting to default custom prompts.")
            return self.force_load_data(DEFAULT_CUSTOM_PROMPTS)

        # Check format of the loaded JSON file is correct
        if (
            _PER_PROMPT_DATA_FIELD in results and
            isinstance(results[_PER_PROMPT_DATA_FIELD], list) and
            results[_PER_PROMPT_DATA_FIELD] and
            isinstance(results[_PER_PROMPT_DATA_FIELD][0], dict) and
            PROMPT_FIELD in results[_PER_PROMPT_DATA_FIELD][0]
        ):
            prompts = [r[PROMPT_FIELD].strip() for r in results[_PER_PROMPT_DATA_FIELD] if r.get(PROMPT_FIELD, '').strip()]
            correct_answers = [
                r[CORRECT_FIELD].split('; ') if CORRECT_FIELD in r else []
                for r in results[_PER_PROMPT_DATA_FIELD]
            ]
            incorrect_answers = [
                r[INCORRECT_FIELD].split('; ') if INCORRECT_FIELD in r else []
                for r in results[_PER_PROMPT_DATA_FIELD]
            ]

            # Filter out biography prompts
            filtered_prompts = []
            filtered_correct_answers = []
            filtered_incorrect_answers = []
            for prompt, correct, incorrect in zip(prompts, correct_answers, incorrect_answers):
                if is_biography_prompt(prompt):
                    logger.info(f"Filtered out biography prompt from results JSON: '{prompt}'")
                    continue
                filtered_prompts.append(prompt)
                filtered_correct_answers.append(correct)
                filtered_incorrect_answers.append(incorrect)

            return self.force_load_data(filtered_prompts, filtered_correct_answers, filtered_incorrect_answers)
        else:
            error_message = (
                f'Invalid JSON format at {filepath}. Reverting to the default custom prompts.'
            )
            logger.error(error_message)
            return self.force_load_data(DEFAULT_CUSTOM_PROMPTS)

    def force_load_data(
        self,
        prompts: List[str],
        correct_answers: Optional[List[List[str]]] = None,
        incorrect_answers: Optional[List[List[str]]] = None,
    ) -> bool:
        """
        Loads a list of prompts and answers into the data package, filtering out biography prompts.

        Args:
            prompts (List[str]): List of prompt strings.
            correct_answers (Optional[List[List[str]]], optional): List of correct answers.
            incorrect_answers (Optional[List[List[str]]], optional): List of incorrect answers.

        Returns:
            bool: True if loading is successful and lengths match, False otherwise.
        """
        logger.info("Forcing load of data.")
        assert prompts, 'ERROR: Must provide at least one prompt.'
        
        # Filter out biography prompts
        filtered_prompts = []
        filtered_correct_answers = []
        filtered_incorrect_answers = []
        for idx, prompt in enumerate(prompts, start=1):
            prompt = prompt.strip()
            if is_biography_prompt(prompt):
                logger.info(f"Filtered out biography prompt: '{prompt}'")
                continue
            filtered_prompts.append(prompt)
            if correct_answers:
                filtered_correct_answers.append(correct_answers[idx - 1])
            else:
                filtered_correct_answers.append([])
            if incorrect_answers:
                filtered_incorrect_answers.append(incorrect_answers[idx - 1])
            else:
                filtered_incorrect_answers.append([])
            logger.debug(f"Loaded prompt: '{prompt}' with correct answers: {filtered_correct_answers[-1]} and incorrect answers: {filtered_incorrect_answers[-1]}")

        self.prompts = filtered_prompts
        self.correct_answers = (
            filtered_correct_answers if correct_answers else [[] for _ in filtered_prompts]
        )
        self.incorrect_answers = (
            filtered_incorrect_answers if incorrect_answers else [[] for _ in filtered_prompts]
        )
        logger.info("Finished forcing load of data.")

        return self.verify_lengths()

    def shuffle_data(self, random_seed: int) -> bool:
        """
        Shuffles the order of data, preserving correspondence.

        Args:
            random_seed (int): Seed for random shuffling.

        Returns:
            bool: True if shuffling is successful and lengths match, False otherwise.
        """
        logger.info(f"Shuffling data with random seed: {random_seed}")
        assert self.verify_lengths(), "Data lengths do not match before shuffling."

        zipped = list(self.iterate())
        random.seed(random_seed)
        random.shuffle(zipped)
        self.prompts, self.correct_answers, self.incorrect_answers = zip(*zipped)
        self.prompts = list(self.prompts)
        self.correct_answers = list(self.correct_answers)
        self.incorrect_answers = list(self.incorrect_answers)
        logger.info("Data shuffling completed.")

        return self.verify_lengths()

    def cap_num_examples(self, max_num_examples: int) -> bool:
        """
        Caps the number of examples in the data package to max_num_examples.

        Args:
            max_num_examples (int): Maximum number of examples to retain.

        Returns:
            bool: True if capping is successful and lengths match, False otherwise.
        """
        logger.info(f"Capping number of examples to: {max_num_examples}")
        if max_num_examples <= 0 or max_num_examples >= self.num_items():
            logger.info('No capping applied.')
            return True

        self.prompts = self.prompts[:max_num_examples]
        self.correct_answers = self.correct_answers[:max_num_examples]
        self.incorrect_answers = self.incorrect_answers[:max_num_examples]
        logger.info(f"Number of examples capped to: {len(self.prompts)}")

        return self.verify_lengths()

    def load_and_prepare(
        self,
        filepath: str,
        shuffle_data: bool,
        random_seed: int,
        max_num_examples: int,
        task: Optional[tuple[str, str, str, str] | str] = None,
    ) -> None:
        """
        Loads data for usage, filtering out biography prompts.

        Args:
            filepath (str): Path to the data file.
            shuffle_data (bool): Whether to shuffle the data.
            random_seed (int): Seed for shuffling.
            max_num_examples (int): Maximum number of examples to retain.
            task (Optional[tuple[str, str, str, str] | str], optional): Task specification.

        Raises:
            ValueError: If the task is invalid or missing.
        """
        logger.info(f"Loading and preparing data from filepath: '{filepath}' with task: {task}")
        if not task:
            error_msg = 'Task not provided.'
            logger.error(error_msg)
            utils.maybe_print_error(error_msg)
            utils.stop_all_execution(True)

        if isinstance(task, str):
            if task.endswith('.json'):
                success = self.load_from_results_json(task)
                logger.debug(f"Loaded from results JSON: {success}")
            elif task.endswith('/'):
                loaded_data = longfact.load_datasets_from_folder(task)
                success = self.force_load_data(loaded_data)
                logger.debug(f"Loaded from folder '{task}': {success}")
            elif task == 'longfact_concepts':
                loaded_data = longfact.load_longfact_concepts()
                success = self.force_load_data(loaded_data)
                logger.debug(f"Loaded longfact_concepts: {success}")
            elif task == 'longfact_objects':
                loaded_data = longfact.load_longfact_objects()
                success = self.force_load_data(loaded_data)
                logger.debug(f"Loaded longfact_objects: {success}")
            elif task == 'custom':
                success = self.force_load_data(prompts=DEFAULT_CUSTOM_PROMPTS)
                logger.debug(f"Loaded custom prompts: {success}")
            else:
                error_msg = 'Invalid task.'
                logger.error(error_msg)
                utils.maybe_print_error(error_msg)
                utils.stop_all_execution(True)
        elif isinstance(task, tuple) and len(task) == TASK_TUPLE_LENGTH:
            task_name, prompt_name, correct_answer_name, incorrect_answer_name = task
            dataset_path = f'{filepath}{task_name}.jsonl'
            success = self.load_from_filepath(
                dataset_path, prompt_name, correct_answer_name, incorrect_answer_name
            )
            logger.debug(f"Loaded from filepath '{dataset_path}': {success}")
        else:
            error_msg = 'Invalid task.'
            logger.error(error_msg)
            utils.maybe_print_error(error_msg)
            utils.stop_all_execution(True)

        if self.num_items() <= 0:
            error_msg = f'Did not load any data with task {task}.'
            logger.error(error_msg)
            utils.maybe_print_error(error_msg)
            utils.stop_all_execution(True)

        if shuffle_data:
            success = self.shuffle_data(random_seed)
            logger.debug(f"Data shuffled: {success}")

        success = self.cap_num_examples(max_num_examples=max_num_examples)
        logger.debug(f"Data capped: {success}")

        logger.info("Data loading and preparation completed.")


if __name__ == "__main__":
    # Example usage (this block can be removed or modified as needed)
    setup_logging()

    logger.info("Initializing DataPackage instance.")
    data_package = DataPackage()

    # Define example parameters
    example_filepath = 'path/to/data.jsonl'
    example_prompt_field = 'prompt'
    example_correct_field = 'correct_answers'
    example_incorrect_field = 'incorrect_answers'
    example_shuffle = True
    example_seed = 42
    example_max_examples = 10
    example_task = ('task_name', 'prompt_field', 'correct_field', 'incorrect_field')

    # Load and prepare data
    try:
        data_package.load_and_prepare(
            filepath=example_filepath,
            shuffle_data=example_shuffle,
            random_seed=example_seed,
            max_num_examples=example_max_examples,
            task=example_task  # Replace with appropriate task or string
        )
        logger.info(f"Number of loaded prompts: {len(data_package.prompts)}")
    except Exception as e:
        logger.error(f"Failed to load and prepare data: {e}")
