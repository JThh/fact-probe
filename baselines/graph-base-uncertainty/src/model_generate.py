import json
from pathlib import Path

class Generation():
    def __init__(self, args, dataset, llm_model, folder_name=''):
        self.args = args
        self.llm_model = llm_model
        self.dataset = dataset
        self.foldername = folder_name 
        self.raw_results_path = f'{folder_name}/{self.args.dataset}_raw_return.json'
        self.collected_results_path = f'{folder_name}/{self.args.dataset}_{self.args.model}_generations.json'
        self.all_results = {'most_likely': [], 'more': []}
        if Path(self.raw_results_path).exists():
            with open(self.raw_results_path, 'r') as f:
                self.all_results = json.load(f)
            print('Loaded from cache', self.raw_results_path)
  
    def _generate_single_most_likely(self, prompt, gen_id):
        if gen_id < len(self.all_results['most_likely']):
            most_likely_results = self.all_results['most_likely'][gen_id]
        else:
            most_likely_results = self.llm_model.generate_given_prompt(prompt)
            self.all_results['most_likely'] = self.all_results['most_likely'] + [most_likely_results]

        print(most_likely_results)
        if 'generation' in most_likely_results:
            most_likely_generation = most_likely_results['generation']
        else:
            # Legacy code for previous versions of cached results
            most_likely_generation = most_likely_results["choices"][0]['message']['content'].strip()
        
        return most_likely_generation
    
    def _generate_single_more(self, prompt, gen_id):
        generated_texts = []
        average_neg_log_likelihoods = []
        neg_log_likelihoods = []
        if gen_id < len(self.all_results['more']):
            more_results = self.all_results['more'][gen_id]
        else:
            more_results = {'generation': [], 'prompt': prompt}
            try_id = 0
            while len(more_results['generation']) < self.args.num_generations_per_prompt and try_id < 5:
                more_generations = self.llm_model.generate_n_given_prompt(prompt)['generation']
                filtered_generations = [g for g in more_generations if not self.ignore_generation(g)]
                more_results['generation'] += filtered_generations[:self.args.num_generations_per_prompt - len(more_results['generation'])]
                try_id += 1
                print('Try:', try_id)

            # Padding with empty strings, saved for response records, but not for further processing since it is not a real answer
            while len(more_results['generation']) < self.args.num_generations_per_prompt:
                more_results['generation'].append('I apologize, I do not know.')

            self.all_results['more'] = self.all_results['more'] + [more_results]
        
        for j in range(self.args.num_generations_per_prompt):
            if 'generation' in more_results and len(more_results['generation']) > j:
                text_generation = more_results['generation'][j]
            elif len(more_results['choices']) > j:
                text_generation = more_results["choices"][j]['message']['content'].strip()

            generated_texts.append(text_generation)

        return generated_texts

    def generate(self):
        sequences = []
        for gen_id, data in enumerate(self.dataset):
            most_likely_generation = self._generate_single_most_likely(data['prompt'], gen_id)
            more_generations = self._generate_single_more(data['prompt'], gen_id)
            with open(self.raw_results_path, 'w') as f:
                json.dump(self.all_results, f, indent=2)

            if self.ignore_generation(most_likely_generation) or self.ignore_generation(more_generations[-1]):
                print("ignoring")
                continue

            sequence_dict = {
                'prompt': data['prompt'],
                'entity': data['question'],
                'wiki_title': data['wiki_title'] if 'wiki_title' in data else data['question'],
                'most_likely_generation': most_likely_generation,
                'more_generations': more_generations,
            }
            sequences.append(sequence_dict)

        with open(self.collected_results_path, 'w') as outfile:
            json.dump(sequences, outfile, indent=4)
   
        return sequences
    
    def ignore_generation(self, text):
        if 'i apologize' in text.lower():
            return True
        if "i don't know" in text.lower():
            return True
        if "i'm not sure" in text.lower():
            return True
        if "i'm sorry" in text.lower():
            return True
        if "i'm not familiar" in text.lower():
            return True
        if 'unfortunately' in text.lower():
            return True
        if "i couldn't" in text.lower():
            return True
        if "i'm not aware" in text.lower():
            return True
        return False

        
    