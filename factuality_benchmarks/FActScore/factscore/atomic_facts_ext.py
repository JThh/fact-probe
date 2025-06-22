import json
import numpy as np
import re
import functools
import string
import spacy
import sys
import nltk
import openai
from rank_bm25 import BM25Okapi
import os
import time
from nltk.tokenize import sent_tokenize

from factscore.openai_lm import OpenAIModel

nltk.download("punkt")


class AtomicFactGenerator(object):
    def __init__(self, key_path, demon_dir, gpt3_cache_file=None):
        self.nlp = spacy.load("en_core_web_sm")
        self.is_bio = True
        self.demon_path = os.path.join(demon_dir, "demons.json" if self.is_bio else "demons_complex.json")

        self.openai_lm = OpenAIModel("InstructGPT", cache_file=gpt3_cache_file, key_path=key_path)

        # get the demos
        with open(self.demon_path, 'r') as f:
            self.demons = json.load(f)

        tokenized_corpus = [doc.split(" ") for doc in self.demons.keys()]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save_cache(self):
        self.openai_lm.save_cache()

    def run(self, generation, cost_estimate=None):
        """Convert the generation into a set of atomic facts with non-overlapping token spans."""
        assert isinstance(generation, str), "generation must be a string"
        paragraphs = [para.strip() for para in generation.split("\n") if len(para.strip()) > 0]
        return self.get_atomic_facts_from_paragraph(paragraphs, cost_estimate=cost_estimate)

    def get_atomic_facts_from_paragraph(self, paragraphs, cost_estimate=None):
        sentences = []
        para_breaks = []
        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx > 0:
                para_breaks.append(len(sentences))

            initials = detect_initials(paragraph)

            curr_sentences = sent_tokenize(paragraph)
            curr_sentences_2 = sent_tokenize(paragraph)

            curr_sentences = fix_sentence_splitter(curr_sentences, initials)
            curr_sentences_2 = fix_sentence_splitter(curr_sentences_2, initials)

            # checking this, just to ensure the credibility of the sentence splitter fixing algorithm
            assert curr_sentences == curr_sentences_2, (paragraph, curr_sentences, curr_sentences_2)

            sentences += curr_sentences

        atoms_or_estimate = self.get_init_atomic_facts_from_sentence(
            [sent for i, sent in enumerate(sentences) if not (not self.is_bio and (
                (i == 0 and (sent.startswith("Sure") or sent.startswith("Here are"))) or
                (i == len(sentences) - 1 and (sent.startswith("Please") or sent.startswith("I hope") or sent.startswith("Here are")))))],
            cost_estimate=cost_estimate,
            original_sentences=sentences  # Pass the original sentences for token span mapping
        )

        if cost_estimate:
            return atoms_or_estimate
        else:
            atoms = atoms_or_estimate

        atomic_facts_pairs = []
        for i, sent in enumerate(sentences):
            if not self.is_bio and (
                    (i == 0 and (sent.startswith("Sure") or sent.startswith("Here are"))) or
                    (i == len(sentences) - 1 and (
                            sent.startswith("Please") or sent.startswith("I hope") or sent.startswith("Here are")))):
                atomic_facts_pairs.append((sent, []))
            elif self.is_bio and sent.startswith("This sentence does not contain any facts"):
                atomic_facts_pairs.append((sent, []))
            elif sent.startswith("Sure") or sent.startswith("Please") or (i == 0 and sent.startswith("Here are")):
                atomic_facts_pairs.append((sent, []))
            else:
                atomic_facts_pairs.append((sent, atoms[sent]))

        atomic_facts_pairs, para_breaks = postprocess_atomic_facts(atomic_facts_pairs, list(para_breaks), self.nlp)

        return atomic_facts_pairs, para_breaks

    def get_init_atomic_facts_from_sentence(self, sentences, cost_estimate=None, original_sentences=None):
        """Get the initial atomic facts from the sentences along with non-overlapping token spans."""
        is_bio = self.is_bio
        demons = self.demons

        k = 1 if is_bio else 0
        n = 7 if is_bio else 8

        prompts = []
        prompt_to_sent = {}
        atoms = {}
        for idx, sentence in enumerate(sentences):
            if sentence in atoms:
                continue
            top_matchings = best_demos(sentence, self.bm25, list(demons.keys()), k)
            prompt = ""

            # Add demonstration examples
            for i in range(n):
                prompt += f"Please break down the following sentence into independent facts. For each fact, list the words from the original sentence that contribute to it. Ensure that the words for each fact do not overlap with words from other facts.\n\n"
                prompt += f"Sentence: {list(demons.keys())[i]}\n"
                for j, fact in enumerate(demons[list(demons.keys())[i]], 1):
                    prompt += f"Fact {j}: {fact}\n"
                    # Since demons don't have contributing words, we'll skip that in the examples
                prompt += "\n"
                
            # Prepare the full prompt with multiple few-shot samples
            prompt = "Please break down the following sentence into independent facts. For each fact, list the words from the original sentence that contribute to it (words can be discontinuous). Ensure that the words for each fact are self-contained and do not overlap with words from other facts.\n\n"

            # Few-Shot Example 1
            prompt += "Sentence: John was born on 10 July 1985 and became a famous singer.\n"
            prompt += "Fact 1: John was born on 10 July 1985.\n"
            prompt += "Contributing words: \"John was born on 10 July 1985\"\n\n"
            prompt += "Fact 2: John became a famous singer.\n"
            prompt += "Contributing words: \"became a famous singer\"\n\n"

            # Few-Shot Example 2
            prompt += "Sentence: Le is a Vietnamese-American computer scientist and entrepreneur.\n"
            prompt += "Fact 1: Le is a Vietnamese-American.\n"
            prompt += "Contributing words: \"Le is a Vietnamese American\"\n\n"
            prompt += "Fact 2: Le is a computer scientist.\n"
            prompt += "Contributing words: \"computer scientist\"\n\n"
            prompt += "Fact 3: Le is an entrepreneur.\n"
            prompt += "Contributing words: \"entrepreneur\"\n\n"

            # Few-Shot Example 3
            prompt += "Sentence: He is best known for co-founding the Google subsidiary DeepMind.\n"
            prompt += "Fact 1: He is best known for co-founding DeepMind.\n"
            prompt += "Contributing words: \"He is best known for co-founding DeepMind\"\n\n"
            prompt += "Fact 2: DeepMind is a Google subsidiary.\n"
            prompt += "Contributing words: \"Google subsidiary\"\n\n"

            # Few-Shot Example 4
            prompt += "Sentence: Le is also a researcher and inventor in the field of artificial intelligence and machine learning.\n"
            prompt += "Fact 1: Le is a researcher.\n"
            prompt += "Contributing words: \"Le is a researcher\"\n\n"
            prompt += "Fact 2: Le is an inventor.\n"
            prompt += "Contributing words: \"inventor\"\n\n"
            prompt += "Fact 3: Le is in the field of artificial intelligence.\n"
            prompt += "Contributing words: \"in the field of artificial intelligence\"\n\n"
            prompt += "Fact 4: Le is in the field of machine learning.\n"
            prompt += "Contributing words: \"machine learning\"\n\n"

            # Few-Shot Example 5
            prompt += "Sentence: He was born in 1964 in Vietnam and immigrated to the United States in the 1970s.\n"
            prompt += "Fact 1: He was born in 1964.\n"
            prompt += "Contributing words: \"He was born in 1964\"\n\n"
            prompt += "Fact 2: He was born in Vietnam.\n"
            prompt += "Contributing words: \"in Vietnam\"\n\n"
            prompt += "Fact 3: He immigrated to the United States.\n"
            prompt += "Contributing words: \"immigrated to the United States\"\n\n"
            prompt += "Fact 4: He immigrated to the United States in the 1970s.\n"
            prompt += "Contributing words: \"in the 1970s\"\n\n"

            # Now add the target sentence
            prompt += f"Sentence: {sentence}\n"

            prompts.append(prompt)
            prompt_to_sent[prompt] = sentence

        if cost_estimate:
            total_words_estimate = 0
            for prompt in prompts:
                if cost_estimate == "consider_cache" and (prompt.strip() + "_0") in self.openai_lm.cache_dict:
                    continue
                total_words_estimate += len(prompt.split())
            return total_words_estimate
        else:
            for prompt in prompts:
                output, _ = self.openai_lm.generate(prompt)
                atoms[prompt_to_sent[prompt]] = parse_model_output(output, prompt_to_sent[prompt], self.nlp)
            for key, value in demons.items():
                if key not in atoms:
                    atoms[key] = value
            return atoms


def best_demos(query, bm25, demons_sents, k):
    tokenized_query = query.split(" ")
    top_matchings = bm25.get_top_n(tokenized_query, demons_sents, k)
    return top_matchings


def parse_model_output(text, sentence, nlp):
    """Parse the model output to extract facts and contributing words, ensuring non-overlapping token spans."""
    facts = []
    lines = text.strip().split('\n')
    current_fact = None
    current_words = None
    for line in lines:
        if line.startswith('Fact '):
            if current_fact is not None:
                facts.append((current_fact, current_words))
            current_fact = line.split(':', 1)[1].strip()
            current_words = None
        elif line.startswith('Contributing words:'):
            current_words = line.split(':', 1)[1].strip().strip('"')
        else:
            # Possible continuation of a fact
            if current_fact is not None:
                current_fact += ' ' + line.strip()
    if current_fact is not None:
        facts.append((current_fact, current_words))

    # Map words back to token spans, ensuring non-overlapping tokens
    token_spans = []
    doc = nlp(sentence)
    tokens = [token.text for token in doc]
    assigned_indices = set()
    for fact, words in facts:
        if words is None:
            token_spans.append((fact, []))
            continue
        contributing_tokens = [token for word in words.strip('"').split() for token in word.split('-')]
        indices = []
        contributing_tokens_norm = [w.strip(string.punctuation).lower() for w in contributing_tokens]
        for i, token in enumerate(tokens):
            # Normalize token and contributing word for matching
            token_norm = token.strip(string.punctuation).lower()
            if token_norm in contributing_tokens_norm and i not in assigned_indices:
                indices.append(i)
                assigned_indices.add(i)
                contributing_tokens_norm.remove(token_norm) # allow for multiple occurrences
        token_spans.append((fact, indices))
    return token_spans


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MONTHS = [m.lower() for m in MONTHS]

def is_num(text):
    try:
        text = int(text)
        return True
    except Exception:
        return False

def is_date(text):
    text = normalize_answer(text)
    for token in text.split(" "):
        if (not is_num(token)) and token not in MONTHS:
            return False
    return True

def extract_numeric_values(text):
    pattern = r'\b\d+\b'  # regular expression pattern for integers
    numeric_values = re.findall(pattern, text)  # find all numeric values in the text
    return set([value for value in numeric_values])  # convert the values to float and return as a list

def detect_entities(text, nlp):
    doc = nlp(text)
    entities = set()

    def _add_to_entities(text):
        if "-" in text:
            for _text in text.split("-"):
                entities.add(_text.strip())
        else:
            entities.add(text)


    for ent in doc.ents:
        # spacy often has errors with other types of entities
        if ent.label_ in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:
            if is_date(ent.text) or is_num(ent.text):
                _add_to_entities(ent.text)
            else:
                for token in ent.text.split():
                    if is_date(token) or is_num(token):
                        _add_to_entities(token)
        
    for new_ent in extract_numeric_values(text):
        if not np.any([new_ent in ent for ent in entities]):
            entities.add(new_ent)

    return entities

def postprocess_atomic_facts(_atomic_facts, para_breaks, nlp):
    verbs = ["born.", " appointed.", " characterized.", " described.", " known.", " member.", " advocate.", "served.", "elected."]
    permitted_verbs = ["founding member."]

    atomic_facts = []
    new_atomic_facts = []
    new_para_breaks = []

    for i, (sent, facts) in enumerate(_atomic_facts):
        sent = sent.strip()
        if len(sent.split()) == 1 and i not in para_breaks and i > 0:
            assert i not in para_breaks
            atomic_facts[-1][0] += " " + sent
            atomic_facts[-1][1] += facts
        else:
            if i in para_breaks:
                new_para_breaks.append(len(atomic_facts))
            atomic_facts.append([sent, facts])

    for i, (sent, facts) in enumerate(atomic_facts):
        entities = detect_entities(sent, nlp)
        covered_entities = set()
        new_facts = []
        for j, (fact, words) in enumerate(facts):
            if any([fact.endswith(verb) for verb in verbs]) and not any([fact.endswith(verb) for verb in permitted_verbs]):
                if any([fact[:-1] in other_fact[0] for k, other_fact in enumerate(facts) if k != j]):
                    continue
            sent_entities = detect_entities(fact, nlp)
            covered_entities |= set([e for e in sent_entities if e in entities])
            new_entities = sent_entities - entities
            if len(new_entities) > 0:
                do_pass = False
                for new_ent in new_entities:
                    pre_ent = None
                    for ent in entities:
                        if ent.startswith(new_ent):
                            pre_ent = ent
                            break
                    if pre_ent is None:
                        do_pass = True
                        break
                    fact = fact.replace(new_ent, pre_ent)
                    covered_entities.add(pre_ent)
                if do_pass:
                    continue
            if fact in [nf[0] for nf in new_facts]:
                continue
            new_facts.append((fact, words))
        try:
            assert entities == covered_entities
        except Exception:
            new_facts = facts  # there is a bug in spacy entity linker, so just go with the previous facts

        new_atomic_facts.append((sent, new_facts))

    return new_atomic_facts, new_para_breaks

def is_integer(s):
    try:
        s = int(s)
        return True
    except Exception:
        return False

def detect_initials(text):
    pattern = r"[A-Z]\. ?[A-Z]\."
    match = re.findall(pattern, text)
    return [m for m in match]

def fix_sentence_splitter(curr_sentences, initials):
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split(".") if len(t.strip()) > 0]
            for i, (sent1, sent2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                    # merge sentence i and i+1
                    curr_sentences = curr_sentences[:i] + [curr_sentences[i] + " " + curr_sentences[i + 1]] + curr_sentences[i + 2:]
                    break
    sentences = []
    combine_with_previous = None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split()) <= 1 and sent_idx == 0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split()) <= 1:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combined_with_previous = False
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)
    return sentences

def main():
    # Open the output file in write mode
    with open('output.txt', 'w', encoding='utf-8') as outfile:
        generator = AtomicFactGenerator("api.key", os.environ.get("FACTSCORE_CACHE_DIR", "./cache/factscore/demos"), gpt3_cache_file=None)
        atomic_facts, para_breaks = generator.run("""
Le is a Vietnamese-American computer scientist and entrepreneur. He is best known for co-founding the Google subsidiary DeepMind. Le is also a researcher and inventor in the field of artificial intelligence and machine learning. He was born in 1964 in Vietnam and immigrated to the United States in the 1970s. Le received his undergraduate degree from the University of Washington and his graduate degree from Carnegie Mellon University. He is a member of the National Academy of Engineering and has received numerous awards and honors for his work, including the MacArthur Fellowship. Le's work has focused on developing algorithms for machine learning and natural language processing. He is also known for his role in developing the AlphaGo artificial intelligence system, which defeated a human world champion in the game of Go. Le has also made significant contributions to the field of computer vision, including the development of the DeepDream algorithm. He is currently a director at DeepMind and has been recognized as one of the most influential people in the world by TIME magazine. Le has also received the ACM A.M. Turing Award, which is considered the "Nobel Prize of Computing". What does Quoc V. Le's work focus on? Quoc V. Le's work focuses on developing algorithms for machine learning and natural language processing. He is also known for his role in developing the AlphaGo artificial intelligence system, which defeated a human world champion in the game of Go. Additionally, he has made significant contributions to the field of computer vision, including the development of the DeepDream algorithm. Is Quoc V. Le a researcher or an entrepreneur? Quoc V. Le is both a researcher and an entrepreneur. He is a researcher in the field of artificial intelligence and machine learning, and has made significant contributions to the field through his work on developing algorithms and systems such as AlphaGo and DeepDream. He is also an entrepreneur, having co-founded the Google subsidiary DeepMind. When was Quoc V. Le born? Quoc V. Le was born in 1964 in Vietnam. Where did Quoc V. Le immigrate to? Quoc V. Le immigrated to the United States in the 1970s. What is Quoc V. Le's current role? Quoc V. Le is currently a director at DeepMind. What award has Quoc V. Le received? Quoc V. Le has received the ACM A.M. Turing Award, which is considered the "Nobel Prize of Computing".
                                              """)

        for sentence, facts in atomic_facts:
            # Prepare the sentence output
            sentence_output = f"Sentence: {sentence}"
            print(sentence_output)
            outfile.write(sentence_output + '\n')
            
            for fact, indices in facts:
                # Retrieve contributing tokens
                contributing_tokens = [generator.nlp(sentence)[i].text for i in indices]
                
                # Prepare the fact output
                fact_output = f"  Fact: {fact}"
                print(fact_output)
                outfile.write(fact_output + '\n')
                
                # Prepare the contributing words output
                contributing_words_output = f"  Contributing Words: {contributing_tokens}"
                print(contributing_words_output)
                outfile.write(contributing_words_output + '\n')
                
                # Prepare the token indices output
                token_indices_output = f"  Token Indices: {indices}"
                print(token_indices_output)
                outfile.write(token_indices_output + '\n')
            
            # Print and write a newline for separation
            print()
            outfile.write('\n')
        
        # Prepare the paragraph breaks output
        para_breaks_output = f"Paragraph Breaks: {para_breaks}"
        print(para_breaks_output)
        outfile.write(para_breaks_output + '\n')

if __name__ == "__main__":
    main()