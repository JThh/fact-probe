import numpy as np
import src.utils as utils
import json
from pathlib import Path
from tqdm import tqdm 

class EdgeConstruction():
    def __init__(self, args, source_data, llm_model, folder_name='') -> None:
        self.args = args
        self.source_data = source_data
        self.llm_model=llm_model
        self.folder_name = folder_name
        self.raw_results_path = f'{folder_name}/sc_match_raw_return.json'
        self.collected_results_path = f'{folder_name}/{self.args.dataset}_{self.args.model}_bipartite_sc_{self.args.sc_samples + 1}samples_{self.args.num_samples_for_claims}matches.json'
        self.cached_results = {'match': []}

        # Load cached results if they exist
        if Path(self.raw_results_path).exists():
            with open(self.raw_results_path, 'r') as f:
                self.cached_results = json.load(f)
            print('Loaded from cache', self.raw_results_path)

    def evaluate_single_match(self, instance_data, generation_id):
        claims = instance_data['breakdown'].copy()
        all_match_lists = []
        raw_match_results = []

        # Combine the most likely generation with a limited number of generated texts
        all_generations = [instance_data['most_likely_generation']] + instance_data['more_generations'][:self.args.sc_samples]

        for gen_index, generation in enumerate(all_generations):
            current_match_list = []
            current_raw_match_results = []

            for claim_index, claim in enumerate(claims):
                # Check if the result is already cached
                if (generation_id < len(self.cached_results['match']) and
                    gen_index < len(self.cached_results['match'][generation_id]) and
                    claim_index < len(self.cached_results['match'][generation_id][gen_index])):
                    match_raw_result = self.cached_results['match'][generation_id][gen_index][claim_index]
                else:
                    match_raw_result = self.perform_faithfulness_check(
                        generation=generation,
                        claim=claim,
                    )

                # Parse the response
                if type(match_raw_result['return']) is not str and 'choices' in match_raw_result['return']:
                    choice = match_raw_result['return']["choices"][0]
                    response = choice['message']['content'].strip()
                else:
                    response = match_raw_result['return'].strip()

                # Clean and validate the response
                response = ''.join([char for char in response if char.isalpha()]).lower()
                if 'notsupported' in response or 'no' in response:
                    response = 'no'
                elif 'supported' in response or 'yes' in response:
                    response = 'yes'
                else:
                    print(f'Invalid response: {response}, gen_index: {gen_index}, claim_index: {claim_index}')
                current_match_list.append(float(response == 'yes'))
                current_raw_match_results.append(match_raw_result)

            all_match_lists.append(current_match_list)
            raw_match_results.append(current_raw_match_results)

        # Update the cached results
        if generation_id >= len(self.cached_results['match']):
            self.cached_results['match'].append(raw_match_results)
        elif len(raw_match_results) > len(self.cached_results['match'][generation_id]):
            self.cached_results['match'][generation_id] = raw_match_results

        return all_match_lists

    
    def evaluate_all_matches(self):
        all_breakdown_matches = self.source_data.copy()

        for generation_id, instance_data in tqdm(enumerate(self.source_data), "Constructing Edges"):
            match_lists = self.evaluate_single_match(instance_data, generation_id)

            # Save the cached results
            with open(self.raw_results_path, 'w') as f:
                json.dump(self.cached_results, f, indent=2)

            match_lists = np.array(match_lists)
            assert np.shape(match_lists) == (self.args.sc_samples + 1, len(instance_data['breakdown']))
            average_scores = np.mean(match_lists, axis=0)

            all_breakdown_matches[generation_id][f'sc_match_{self.args.sc_samples}samples'] = match_lists.tolist()
            verbalized_confidence_list = np.zeros(len(instance_data['breakdown']))
            verbalized_confidence_with_options_list = np.zeros(len(instance_data['breakdown']))

            for claim_index, pointwise_dict in enumerate(instance_data['pointwise_dict']):
                verbalized_confidence_with_options_list[claim_index] = pointwise_dict['verbalized_confidence_with_options']
                verbalized_confidence_list[claim_index] = pointwise_dict['inline_verbalized_confidence']

            # Calculate centrality scores
            centrality_scores_vcwo = utils.calculate_bg_centrality(match_lists, len(instance_data['breakdown']), vc_lst=verbalized_confidence_with_options_list)
            centrality_scores_vc = utils.calculate_bg_centrality(match_lists, len(instance_data['breakdown']), vc_lst=verbalized_confidence_list)

            for claim_index, pointwise_dict in enumerate(instance_data['pointwise_dict']):
                assert pointwise_dict['claim'] == instance_data["breakdown"][claim_index]
                pointwise_dict[f'sc_score_{self.args.sc_samples}samples'] = average_scores[claim_index]

                for centrality_name, centrality_score in centrality_scores_vcwo.items():
                    pointwise_dict[f'breakdown_{centrality_name}_{self.args.sc_samples}samples'] = centrality_score[claim_index]

                pointwise_dict[f'sc_plus_vc'] = average_scores[claim_index] + pointwise_dict['verbalized_confidence_with_options']
                pointwise_dict[f'sc_based_vc'] = average_scores[claim_index] + pointwise_dict['verbalized_confidence_with_options'] * 0.1
                
                # A baseline uses neglectable extra computation than sc, using inline verbalized confidence to break ties in sc 
                pointwise_dict[f'sc_based_ilvc'] = average_scores[claim_index] + float(pointwise_dict['inline_verbalized_confidence']) * 0.1 
                
        # Save the final results
        with open(self.collected_results_path, 'w') as outfile:
            json.dump(all_breakdown_matches, outfile, indent=4)

        return all_breakdown_matches

    def perform_faithfulness_check(self, generation, claim):
        prompt = f"""Context: {generation}
            Claim: {claim}
            Is the claim supported by the context above?
            Answer Yes or No:
            """

        model_response = self.llm_model.generate_given_prompt([{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': prompt}])
        model_response = model_response['generation']

        return {'return': model_response, 'prompt': prompt}
   
    