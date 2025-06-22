import argparse
import string
import json
import numpy as np
import os
import logging

from tqdm import tqdm
from factscore.abstain_detection import is_response_abstained
from factscore.atomic_facts import AtomicFactGenerator
from factscore.clm import CLM
from factscore.npm import NPM
from factscore.openai_lm import OpenAIModel
from factscore.retrieval import DocDB, Retrieval

# Constants for relevance classification
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


Example 1:
QUESTION:
Who is Quoc Le?

RESPONSE:
After completing his Ph.D., Quoc Le joined Google Brain, where he has been \
working on a variety of deep learning projects. Quoc is well-respected by many \
of his peers, such as Geoffrey Hinton, who is an adjunct professor at the \
University of Montreal and teaches courses on deep learning.

STATEMENT:
Geoffrey Hinton is at the University of Montreal.

SOLUTION:
The subject of the QUESTION is Quoc Le. The subject of the STATEMENT is \
Geoffrey Hinton. The phrase "Quoc is well-respected by many of his peers, such \
as Geoffrey Hinton" from the RESPONSE shows that the relationship between Quoc \
Le and Geoffrey Hinton is that they are peers. For this reason, the subjects \
Quoc Le and Geoffrey Hinton are [{SYMBOL}].


Example 2:
QUESTION:
Who is Quoc Le?

RESPONSE:
After completing his Ph.D., Quoc Le joined Google Brain, where he has been \
working on a variety of deep learning projects. Geoffrey Hinton is an adjunct \
professor at the University of Montreal, where he teaches courses on deep \
learning.

STATEMENT:
Geoffrey Hinton is at the University of Montreal.

SOLUTION:
The subject of the QUESTION is Quoc Le. The subject of the STATEMENT is \
Geoffrey Hinton. While both subjects seem to be related to deep learning, \
the RESPONSE does not contain any phrases that explain what the relationship \
between Quoc Le and Geoffrey Hinton is. Thus, the subjects Quoc Le and \
Geoffrey Hinton are [{NOT_SYMBOL}].


Your Task:
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
4. You MUST NOT add any additional factual claims to the original STATEMENT. \
For example, given the response "Titanic is a movie starring Leonardo \
DiCaprio," the statement "Titanic is a movie" should not be changed.
5. Before giving your revised statement, think step-by-step and show your \
reasoning. As part of your reasoning, be sure to identify the subjects in the \
STATEMENT and determine whether they are vague references. If they are vague \
references, identify the proper entity that they are referring to and be sure \
to revise this subject in the revised statement.
6. After showing your reasoning, provide the revised statement and wrap it in \
a markdown code block.
7. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". \
Some examples have been provided for you to learn how to do this task.


Example 1:
STATEMENT:
Acorns is a company.

RESPONSE:
Acorns is a financial technology company founded in 2012 by Walter Cruttenden, \
Jeff Cruttenden, and Mark Dru that provides micro-investing services. The \
company is headquartered in Irvine, California.

REVISED STATEMENT:
The subject in the statement "Acorns is a company" is "Acorns". "Acorns" is \
not a pronoun and does not reference an unknown entity. Furthermore, "Acorns" \
is not further specified in the RESPONSE, so we can assume that it is a full \
name. Therefore "Acorns" is not a vague reference. Thus, the revised statement \
is:
```
Acorns is a company.
```


Example 2:
STATEMENT:
He teaches courses on deep learning.

RESPONSE:
After completing his Ph.D., Quoc Le joined Google Brain, where he has been \
working on a variety of deep learning projects. Le is also an adjunct \
professor at the University of Montreal, where he teaches courses on deep \
learning.

REVISED STATEMENT:
The subject in the statement "He teaches course on deep learning" is "he". \
From the RESPONSE, we can see that this statement comes from the sentence "Le \
is also an adjunct professor at the University of Montreal, where he teaches \
courses on deep learning.", meaning that "he" refers to "Le". From the \
RESPONSE, we can also see that "Le" refers to "Quoc Le". Therefore "Le" is a \
non-full name that should be replaced by "Quoc Le." Thus, the revised response \
is:
```
Quoc Le teaches courses on deep learning.
```


Example 3:
STATEMENT:
The television series is called "You're the Worst."

RESPONSE:
Xochitl Gomez began her acting career in theater productions, and she made her \
television debut in 2016 with a guest appearance on the Disney Channel series \
"Raven's Home." She has also appeared in the television series "You're the \
Worst" and "Gentefied."

REVISED STATEMENT:
The subject of the statement "The television series is called "You're the \
Worst."" is "the television series". This is a reference to an unknown entity, \
since it is unclear what television series is "the television series". From \
the RESPONSE, we can see that the STATEMENT is referring to the television \
series that Xochitl Gomez appeared in. Thus, "the television series" is a \
vague reference that should be replaced by "the television series that Xochitl \
Gomez appeared in". Thus, the revised response is:
```
The television series that Xochitl Gomez appeared in is called "You're the \
Worst."
```


Example 4:
STATEMENT:
Dean joined Google.

RESPONSE:
Jeff Dean is a Google Senior Fellow and the head of Google AI, leading \
research and development in artificial intelligence. Dean joined Google in \
1999 and has been essential to its continued development in the field.

REVISED STATEMENT:
The subject of the statement "Dean joined Google" is "Dean". From the \
response, we can see that "Dean" is the last name of "Jeff Dean". Therefore \
"Dean" is a non-full name, making it a vague reference. It should be replaced \
by "Jeff Dean", which is the full name. Thus, the revised response is:
```
Jeff Dean joined Google.
```


Your Task:
STATEMENT:
{_STATEMENT_PLACEHOLDER}

RESPONSE:
{_RESPONSE_PLACEHOLDER}
"""

# Utility functions for extracting information from model responses
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
    pattern = r'```(?:[\w\+\-\.]+\n)?([\s\S]+?)```' if ignore_language else r'```[\w\+\-\.]+\n([\s\S]+?)```'
    match = re.search(pattern, text)
    result = match.group(1).strip() if match else None
    logging.debug(f"Extracted code block: {result}")
    return result

def strip_string(s):
    result = s.strip()
    logging.debug(f"Stripped string: {result}")
    return result

class FactScorer(object):

    def __init__(self,
                 model_name="retrieval+ChatGPT",
                 data_dir=None,
                 model_dir=None,
                 cache_dir=None,
                 openai_key="api.key",
                 cost_estimate="consider_cache",
                 abstain_detection_type=None,
                 batch_size=256):
        if data_dir is None:
            data_dir = os.environ.get("FACTSCORE_CACHE_DIR", "./cache/factscore")
        if cache_dir is None:
            cache_dir = os.environ.get("FACTSCORE_CACHE_DIR", "./cache/factscore")
            
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        assert model_name in ["retrieval+llama", "retrieval+llama+npm", "retrieval+ChatGPT", "npm", "retrieval+ChatGPT+npm"]
        self.model_name = model_name

        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size  # batch size for retrieval
        self.openai_key = openai_key
        self.abstain_detection_type = abstain_detection_type

        self.data_dir = data_dir
        self.cache_dir = cache_dir

        self.af_generator = None
        self.cost_estimate = cost_estimate

        if "llama" in model_name:
            logging.debug("Initializing CLM model...")
            self.lm = CLM("inst-llama-7B",
                          model_dir=os.path.join(model_dir, "inst-llama-7B"),
                          cache_file=os.path.join(cache_dir, "inst-llama-7B.pkl"))
        elif "ChatGPT" in model_name:
            logging.debug("Initializing ChatGPT model...")
            self.lm = OpenAIModel("ChatGPT",
                                  cache_file=os.path.join(cache_dir, "ChatGPT.pkl"),
                                  key_path=openai_key)
        else:
            self.lm = None

        # Initialize the model for relevance classification and fact refinement
        # For simplicity, we'll use the same OpenAIModel as self.lm
        logging.debug("Initializing relevance and refinement model...")
        self.relevance_lm = OpenAIModel("ChatGPT",
                                        cache_file=os.path.join(cache_dir, "RelevanceChatGPT.pkl"),
                                        key_path=openai_key)

    def save_cache(self):
        logging.debug("Saving caches...")
        if self.lm:
            self.lm.save_cache()
        if self.relevance_lm:
            self.relevance_lm.save_cache()
        if "npm" in self.model_name:
            for k, v in self.npm.items():
                v.save_cache()
        for k, v in self.retrieval.items():
            v.save_cache()

    def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
        logging.debug(f"Registering knowledge source: {name}")
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

        self.db[name] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)
        if "npm" in self.model_name:
            cache_path = os.path.join(self.cache_dir, f"bm25-{name}.json")
            embed_cache_path = os.path.join(self.cache_dir, f"bm25-{name}.pkl")
            self.npm[name] = NPM(Retrieval(self.db[name], cache_path, embed_cache_path, "bm25"),
                                 "npm-single",
                                 cache_file=os.path.join(self.cache_dir, f"npm-{name}.pkl"))

    def print_cost_estimates(self, total_words, task, model):
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        # Number of tokens are roughly 4/3 of the number of words
        total_tokens = total_words * 4.0 / 3

        # https://openai.com/pricing
        # Assuming a rate for ChatGPT (gpt-4o-mini)
        rate = 0.00075  # $0.00075 per 1K tokens

        total_cost = total_tokens * rate / 1000

        # Print the total words, tokens, and cost along with rate
        logging.critical("Estimated OpenAI API cost for %s ($%.3f per 1000 tokens): $%.2f for %d words and %d tokens",
                         task, rate, total_cost, total_words, total_tokens)

    def get_score(self,
                  topics,
                  generations,
                  gamma=10,
                  atomic_facts=None,
                  knowledge_source=None,
                  verbose=False):
        if knowledge_source is None:
            # Use the default knowledge source
            knowledge_source = "enwiki-20230401"

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)

        if type(topics) == type(generations) == str:
            topics = [topics]
            generations = [generations]
        else:
            assert type(topics) == type(generations) == list, "`topics` and `generations` should be lists."
            assert len(topics) == len(generations), "`topics` and `generations` should have the same length"

        if atomic_facts is not None:
            assert len(topics) == len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else:
            if self.af_generator is None:
                logging.debug("Initializing AtomicFactGenerator...")
                self.af_generator = AtomicFactGenerator(key_path=self.openai_key,
                                                        demon_dir=os.path.join(self.data_dir, "factscore", "demos"),
                                                        gpt3_cache_file=os.path.join(self.cache_dir, "GPT4o.pkl"))

            # Estimate the total cost of atomic fact generation
            total_words = 0
            for gen in generations:
                words = self.af_generator.run(gen, cost_estimate=self.cost_estimate)
                logging.debug(f"Estimated words for atomic fact generation: {words}")
                total_words += words

            self.print_cost_estimates(total_words, task="atomic fact generation", model="davinci-003")

            if verbose:
                iterator = tqdm(zip(topics, generations), total=len(topics))
            else:
                iterator = zip(topics, generations)

            atomic_facts = []
            for topic, gen in iterator:
                logging.debug(f"Processing topic: {topic}")
                # Optionally, first detect if the response is abstained
                response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                logging.debug(f"Response abstained: {response_abstained}")
                if response_abstained:
                    atomic_facts.append(None)
                    continue
                # Continue only when the response is not abstained
                curr_afs, _ = self.af_generator.run(gen)
                curr_afs = [fact for _, facts in curr_afs for fact in facts]
                logging.debug(f"Extracted atomic facts: {curr_afs}")
                if len(curr_afs) == 0:
                    atomic_facts.append(None)
                else:
                    atomic_facts.append(curr_afs)
                if len(atomic_facts) % 10 == 0:
                    self.af_generator.save_cache()

            assert len(atomic_facts) == len(topics)
            self.af_generator.save_cache()

        respond_ratio = np.mean([facts is not None for facts in atomic_facts])
        logging.info(f"Respond ratio: {respond_ratio}")

        if "ChatGPT" in self.model_name:
            # Estimate the total cost of response generation
            total_words = 0
            for topic, generation, facts in zip(topics, generations, atomic_facts):
                if facts is not None:
                    words = self._get_score(topic, generation, facts, knowledge_source, cost_estimate=self.cost_estimate)
                    logging.debug(f"Estimated words for factscore evaluation: {words}")
                    total_words += words

            self.print_cost_estimates(total_words, task="factscore evaluation", model="gpt-3.5-turbo")

        if verbose:
            iterator = tqdm(zip(topics, generations, atomic_facts), total=len(topics))
        else:
            iterator = zip(topics, generations, atomic_facts)

        scores = []
        init_scores = []
        decisions = []
        for topic, generation, facts in iterator:
            logging.debug(f"Evaluating topic: {topic}")
            if facts is None:
                decisions.append(None)
                logging.debug("No atomic facts to process.")
            else:
                decision = self._get_score(topic, generation, facts, knowledge_source)
                if decision is not None and len(decision) > 0:
                    score = np.mean([d["is_supported"] for d in decision])
                else:
                    score = 0.0  # If no relevant facts, score is 0
                logging.debug(f"Initial score: {score}")

                if gamma:
                    init_scores.append(score)
                    penalty = 1.0 if len(facts) > gamma else np.exp(1 - gamma / len(facts))
                    score = penalty * score
                    logging.debug(f"Adjusted score with penalty: {score}")

                decisions.append(decision)
                scores.append(score)
                if len(scores) % 10 == 0:
                    self.save_cache()

        self.save_cache()

        out = {"score": np.mean(scores),
               "respond_ratio": respond_ratio,
               "decisions": decisions,
               "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None and len(d) > 0])}

        if gamma:
            out["init_score"] = np.mean(init_scores)

        logging.info(f"Final FActScore: {out['score']}")
        return out

    def _get_score(self, topic, generation, atomic_facts, knowledge_source, cost_estimate=None):
        decisions = []
        total_words = 0
        for atom in atomic_facts:
            logging.info(f"Processing atomic fact: {atom}")
            atom = atom.strip()
            # Step 1: Refine the atomic fact
            revised_atom = self.revise_fact(generation, atom, cost_estimate=cost_estimate)
            logging.info(f"Revised atomic fact: {revised_atom}")
            if cost_estimate:
                total_words += revised_atom
                continue

            # Step 2: Check relevance
            is_relevant = self.check_relevance(topic, generation, revised_atom, cost_estimate=cost_estimate)
            logging.info(f"Is relevant: {is_relevant}")
            if cost_estimate:
                total_words += is_relevant
                continue

            if not is_relevant:
                # Skip irrelevant facts
                logging.info("Atomic fact is irrelevant, skipping...")
                continue

            # Step 3: Proceed with supportedness classification
            is_subjective = False
            is_supported = False
            if self.lm:
                passages = self.retrieval[knowledge_source].get_passages(topic, revised_atom, k=5)
                logging.debug(f"Retrieved passages: {passages}")
                definition = "Answer the question about {} based on the given context.\n\n".format(topic)
                context = ""
                for psg_idx, psg in enumerate(reversed(passages)):
                    context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
                definition += context.strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                # prompt = "{}\n\nInput: {} True or False?\nOutput:".format(definition.strip(), revised_atom.strip())
                prompt = "\n\n{}\n\nInput: {}.\nIs the input Subjective or Objective? If objective, True or False? Format your answer as one word as 'ANSWER: <Subjectve/True/False>'\nOutput:".format(definition.strip(), atom.strip())
                logging.debug(f"Prompt for supportedness classification: {prompt}")

                if cost_estimate:
                    if cost_estimate == "consider_cache" and (prompt.strip() + "_0") not in self.lm.cache_dict:
                        total_words += len(prompt.split())
                    elif cost_estimate == "ignore_cache":
                        total_words += len(prompt.split())
                    continue

                output = self.lm.generate(prompt)
                logging.debug(f"Model output: {output}")

                if type(output[1]) == np.ndarray:
                    # When logits are available
                    logits = np.array(output[1])
                    assert logits.shape[0] in [32000, 32001]
                    true_score = logits[5852]
                    false_score = logits[7700]
                    is_supported = true_score > false_score
                    logging.debug(f"Logits - True score: {true_score}, False score: {false_score}")
                else:
                    # When logits are unavailable
                    # generated_answer = output[0].lower()
                    generated_answer = output[0].split('ANSWER:')[1].strip().lower()
                    
                    logging.debug(f"Generated answer: {generated_answer}")
                    if "subjective" in generated_answer:
                        is_subjective = True
                    elif "true" in generated_answer or "false" in generated_answer:
                        if "true" in generated_answer and "false" not in generated_answer:
                            is_supported = True
                        elif "false" in generated_answer and "true" not in generated_answer:
                            is_supported = False
                        else:
                            is_supported = generated_answer.index("true") > generated_answer.index("false")
                    else:
                        is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split()
                                            for keyword in ["not", "cannot", "unknown", "information"]])
                        logging.info(f"Is supported after keyword check: {is_supported}")

            else:
                is_supported = True

            if is_supported and "npm" in self.model_name:
                npprob = self.npm[knowledge_source].get_probabilty(topic, revised_atom)
                logging.debug(f"NPM probability: {npprob}")
                is_supported = npprob > 0.3

            if not is_subjective:
                decisions.append({"atom": revised_atom, "is_supported": is_supported})
                logging.info(f"Decision for atomic fact (is subjective: {is_subjective}): {decisions[-1]}")
            else:
                logging.info(f"Decision for atomic fact (is subjective: {is_subjective}): skipping")

        if cost_estimate:
            return total_words
        else:
            return decisions

    def revise_fact(self, response, atomic_fact, cost_estimate=None):
        # Use the model to revise the atomic fact
        full_prompt = _REVISE_FORMAT.replace(_STATEMENT_PLACEHOLDER, atomic_fact)
        full_prompt = full_prompt.replace(_RESPONSE_PLACEHOLDER, response)
        full_prompt = strip_string(full_prompt)
        logging.debug(f"Prompt for revising fact: {full_prompt}")

        if cost_estimate:
            total_words = len(full_prompt.split())
            logging.debug(f"Estimated words for revising fact: {total_words}")
            return total_words

        try:
            model_response = self.relevance_lm.generate(full_prompt)
            logging.debug(f"Model response for revising fact: {model_response}")
            if isinstance(model_response, tuple):
                model_response = model_response[0]
            if not isinstance(model_response, str):
                logging.error(f"Model response is not a string: {type(model_response)}")
                model_response = None
        except Exception as e:
            logging.error(f"Error generating revised fact: {e}")
            model_response = None

        if model_response is None:
            logging.warning("Failed to obtain model response, using original atomic fact.")
            return atomic_fact

        revised_fact = extract_first_code_block(model_response, ignore_language=True)
        if revised_fact:
            return revised_fact.strip()
        else:
            # If the revision fails, use the original atomic fact
            logging.warning("Failed to revise atomic fact, using original.")
            return atomic_fact

    def check_relevance(self, prompt, response, atomic_fact, cost_estimate=None):
        # Use the model to check if the atomic fact is relevant
        full_prompt = _RELEVANCE_FORMAT.replace(_STATEMENT_PLACEHOLDER, atomic_fact)
        full_prompt = full_prompt.replace(_PROMPT_PLACEHOLDER, prompt)
        full_prompt = full_prompt.replace(_RESPONSE_PLACEHOLDER, response)
        full_prompt = strip_string(full_prompt)
        logging.debug(f"Prompt for checking relevance: {full_prompt}")

        if cost_estimate:
            total_words = len(full_prompt.split())
            logging.debug(f"Estimated words for checking relevance: {total_words}")
            return total_words

        model_response = self.relevance_lm.generate(full_prompt)[0]
        logging.debug(f"Model response for checking relevance: {model_response}")
        answer = extract_first_square_brackets(model_response)
        if answer:
            is_relevant = answer.strip().lower() == SYMBOL.lower()
            logging.debug(f"Extracted answer: {answer}, Is relevant: {is_relevant}")
            return is_relevant
        else:
            # If we cannot determine relevance, assume it's relevant
            logging.warning("Failed to determine relevance, assuming relevant.")
            return True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        type=str,
                        default="data/labeled/InstructGPT.jsonl")
    parser.add_argument('--model_name',
                        type=str,
                        default="retrieval+ChatGPT")
    parser.add_argument('--gamma',
                        type=int,
                        default=10,
                        help="hyperparameter for length penalty")

    parser.add_argument('--openai_key',
                        type=str,
                        default="api.key")
    parser.add_argument('--data_dir',
                        type=str,
                        default="/scratch/ms23jh/.cache/factscore/")
    parser.add_argument('--model_dir',
                        type=str,
                        default="/scratch/ms23jh/.cache/factscore/")
    parser.add_argument('--cache_dir',
                        type=str,
                        default="/scratch/ms23jh/.cache/factscore/")
    parser.add_argument('--knowledge_source',
                        type=str,
                        default=None)

    parser.add_argument('--cost_estimate',
                        type=str,
                        default="consider_cache",
                        choices=["consider_cache", "ignore_cache"])
    parser.add_argument('--abstain_detection_type',
                        type=str,
                        default=None,
                        choices=["perplexity_ai", "generic", "none"])
    parser.add_argument('--use_atomic_facts',
                        action="store_true")
    parser.add_argument('--verbose',
                        action="store_true",
                        help="for printing out the progress bar")
    parser.add_argument('--print_rate_limit_error',
                        action="store_true",
                        help="for printing out rate limit error when using OpenAI keys")
    parser.add_argument('--n_samples',
                        type=int,
                        default=None)
    parser.add_argument('--log_level',
                        type=str,
                        default='WARNING',
                        help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=getattr(logging, args.log_level.upper(), logging.WARNING))

    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)

    fs = FactScorer(model_name=args.model_name,
                    data_dir=args.data_dir,
                    model_dir=args.model_dir,
                    cache_dir=args.cache_dir,
                    openai_key=args.openai_key,
                    cost_estimate=args.cost_estimate,
                    abstain_detection_type=args.abstain_detection_type)

    tot = 0
    topics, generations, atomic_facts = [], [], []
    with open(args.input_path) as f:
        for line in f:
            dp = json.loads(line)
            tot += 1
            if args.use_atomic_facts:
                assert "annotations" in dp, "You can specify `--use_atomic_facts` only when atomic facts are available in the input data already."
                if dp["annotations"] is None:
                    continue
                topics.append(dp["topic"])
                generations.append(dp["output"])
                atomic_facts.append([atom["text"] for sent in dp["annotations"] for atom in sent["model-atomic-facts"]])
            else:
                topics.append(dp["topic"])
                generations.append(dp["output"])
            if args.n_samples is not None and tot == args.n_samples:
                break
    out = fs.get_score(topics=topics,
                       generations=generations,
                       gamma=args.gamma,
                       atomic_facts=atomic_facts if args.use_atomic_facts else None,
                       knowledge_source=args.knowledge_source,
                       verbose=args.verbose)
    logging.critical("FActScore = %.1f%%", 100 * out["score"])
    if "init_score" in out:
        logging.critical("FActScore w/o length penalty = %.1f%%", 100 * out["init_score"])
    logging.critical("Respond ratio = %.1f%%", 100 * out["respond_ratio"])
    logging.critical("# Atomic facts per valid response = %.1f", out["num_facts_per_response"])

    # Save out as a json file
    output_file = args.input_path.replace(".jsonl", f"_factscore_output.json")
    logging.debug(f"Saving output to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(out, f)

