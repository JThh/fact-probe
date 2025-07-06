import numpy as np
import pandas as pd
import os
import subprocess
from typing import Any, Dict, List

class UncertaintyAwareDecoding():
    DEFAULT_KEYS = ['closeness', 'sc', 'sc_ml', 'sc+vc', 'sc+ilvc']
    DEFAULT_PERCENTILES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    DEFAULT_SUBFOLDER_NAME = 'merged_output'

    def __init__(
        self,
        args: Any,
        generations: List[Dict],
        llm_model: Any,
        output_keys: List[str] = ['closeness', 'sc', 'sc+ilvc'],
        threshold: Dict = None,
        folder_name: str = '',
        threshold_path: str = None,
        percentile_lst: List[int] = None,
        eval_baselines: bool = False
    ):
        self.args = args
        self.generations = generations
        self.folder_name = folder_name
        self.llm_model = llm_model
        self.keys = self.DEFAULT_KEYS
        self.output_keys = output_keys
        self.save_path = f'{self.folder_name}/{self.DEFAULT_SUBFOLDER_NAME}'
        self.percentile_lst = percentile_lst or self.DEFAULT_PERCENTILES
        self.threshold = self.calibrate_threshold_by_file(percentile_lst=self.percentile_lst)
        
        
    def _filter_claims(self, row: Dict, threshold_num: int) -> Dict[str, List]:
        """Filter claims based on different scoring methods and thresholds."""
        filtered_claims = {key: [] for key in self.keys}
        
        for i, item in enumerate(row['pointwise_dict']):
            sc_score = item[f'sc_score_{self.args.sc_samples}samples']
            closeness = item[f'breakdown_closeness_centrality_{self.args.sc_samples}samples']
            
            # Filter for most likely claims
            if i < row['most_likely_breakdown_len']:
                if sc_score >= self.threshold['sc'][threshold_num]['threshold']:
                    filtered_claims['sc_ml'].append(item)
            
            # Filter based on different metrics
            if closeness >= self.threshold['closeness'][threshold_num]['threshold']:
                filtered_claims['closeness'].append(item)
            
            if sc_score >= self.threshold['sc'][threshold_num]['threshold']:
                filtered_claims['sc'].append(item)
            
            vc_score = sc_score + 0.1 * item['verbalized_confidence_with_options']
            if vc_score >= self.threshold['sc+vc'][threshold_num]['threshold']:
                filtered_claims['sc+vc'].append(item)
            
            if item[f'sc_based_ilvc'] >= self.threshold['sc+ilvc'][threshold_num]['threshold']:
                filtered_claims['sc+ilvc'].append(item)
        
        return filtered_claims

    def _save_results(self, df: pd.DataFrame, threshold_num: int, key: str):
        """Save filtered results to JSON files."""
        os.makedirs(f'{self.save_path}/{key}', exist_ok=True)
        
        df_eval = df[['prompt', f'merged_output_{key}_percentile{self.threshold["closeness"][threshold_num]["percentile"]}', 'entity']]
        df_eval = df_eval.rename(columns={
            'prompt': 'input',
            f'merged_output_{key}_percentile{self.threshold["closeness"][threshold_num]["percentile"]}': 'output',
            'entity': 'topic'
        })

        threshold_value = round(self.threshold[key.split('_')[0]][threshold_num]['threshold'], 3)
        path = f'{self.save_path}/{key}/merged_output_threshold{threshold_value}_percentile{self.threshold["closeness"][threshold_num]["percentile"]}.json'
        
        with open(path, 'w') as outfile:
            for _, result_row in df_eval.iterrows():
                outfile.write(result_row.to_json() + '\n')
        
        if self.args.fs_eval:
            self.factscore_eval(path, key, threshold_value)

    def merge_uad(self):
        """Main method to merge and process uncertainty-aware decodings."""
        df = pd.DataFrame(self.generations)
        
        # Process each threshold
        for threshold_num in range(len(self.threshold['closeness'])):
            for id_, row in df.iterrows():
                filtered_claims = self._filter_claims(row, threshold_num)
                prompt = row['prompt']
                
                # Generate merged outputs for each key
                for key in self.output_keys:
                    percentile = self.threshold["closeness"][threshold_num]["percentile"]
                    merged_output = self.merge_subclaims(filtered_claims[key], original_prompt=prompt)
                    df.at[id_, f'merged_output_{key}_percentile{percentile}'] = merged_output
            
            # Save results for each output key
            for key in self.output_keys:
                self._save_results(df, threshold_num, key)
        
        # Save complete results
        df.to_json(f'{self.folder_name}/merged_output_check.json', orient='records', lines=True, indent=4)
        
        # Process baseline evaluations
        self._process_baselines(df)

    def _process_baselines(self, df: pd.DataFrame):
        """Process and save baseline evaluations."""
        eval_keys = ['most_likely_generation']
        
        for eval_key in eval_keys:
            df_ml = df[['prompt', eval_key, 'entity']].rename(columns={
                'prompt': 'input',
                eval_key: 'output',
                'entity': 'topic'
            })
            
            path = f'{self.save_path}/merged_output_{eval_key}.json'
            with open(path, 'w') as outfile:
                for _, result_row in df_ml.iterrows():
                    outfile.write(result_row.to_json() + '\n')
            
            # if self.args.fs_eval:
            #     self.factscore_eval(path, eval_key, None)

    def factscore_eval(self, path, key, threshold):
        command = [
            'sbatch', 'submit_factscorer_eval.sh', f'{path}', f'.cache/factscore/{self.args.model}_{key}_{threshold}'
        ]
        print(f"Processing file: {path}")
        subprocess.run(command)
           
    def calibrate_threshold_by_file(self, percentile_lst):
        threshold_lst = {key: [] for key in self.keys}
        for percentile in percentile_lst:
            threshold = self.get_threshold_given_percentile(percentail=percentile, train_size=20)
            print(f'Threshold for {percentile} percentile is {threshold}')
            for key in self.keys:
                threshold_lst[key].append({'threshold': threshold[key.split('_')[0]], 'percentile': percentile})
        return threshold_lst

    def _collect_scores(self, train_size: int = 10) -> Dict[str, List[float]]:
        """Collect all scores from the training data once."""
        scores = {
            'sc': [],
            'closeness': [],
            'sc+vc': [],
            'sc+ilvc': []
        }
        
        df = pd.DataFrame(self.generations)
        for id_, row in df.iterrows():
            if id_ >= train_size:
                break
                
            for item in row['pointwise_dict']:
                sc_score = item[f'sc_score_{self.args.sc_samples}samples']
                scores['sc'].append(sc_score)
                scores['closeness'].append(item[f'breakdown_closeness_centrality_{self.args.sc_samples}samples'])
                scores['sc+vc'].append(sc_score + 0.1 * item['verbalized_confidence_with_options'])
                scores['sc+ilvc'].append(item[f'sc_based_ilvc'])
        
        return scores

    def calibrate_threshold_by_file(self, percentile_lst: List[int]):
        """Calibrate thresholds based on percentiles."""
        # Collect scores once
        scores = self._collect_scores(train_size=20)
        threshold_lst = {key: [] for key in self.keys}
        
        # Calculate thresholds for each percentile using the collected scores
        for percentile in percentile_lst:
            thresholds = {
                'sc': np.percentile(scores['sc'], percentile),
                'closeness': np.percentile(scores['closeness'], percentile),
                'sc+vc': np.percentile(scores['sc+vc'], percentile),
                'sc+ilvc': np.percentile(scores['sc+ilvc'], percentile)
            }
            print(f'Threshold for {percentile} percentile is {thresholds}')
            
            for key in self.keys:
                threshold_lst[key].append({
                    'threshold': thresholds[key.split('_')[0]],
                    'percentile': percentile
                })
        
        return threshold_lst
    
    @staticmethod
    def default_merge_prompt(subclaims, prompt):
        claim_string = "\n".join(
            [str(i+1) + ": " + subclaim["claim"] for i, subclaim in enumerate(subclaims)]
        )
        return f"Task: You are provided with a list of facts about {prompt}. Your goal is to synthesize these facts into a coherent paragraph. Use all the provided facts where possible, ensuring that no fact is misrepresented or overlooked. If there are redundant facts, choose the most comprehensive one for inclusion. The length of the paragraph should naturally reflect the number of provided facts—shorter for fewer facts and longer for more. Avoid unnecessary filler and focus on presenting the information clearly and concisely. \n\nThe facts:\n{claim_string}"

    def merge_subclaims(
        self, subclaims, original_prompt
    ):
        """
        Takes in a list of sub-claims like [{'subclaim': 'Percy Liang is a computer scientist.', 'score': 5.0}, ...] and produces a merged output.
        """
        prompt = self.default_merge_prompt(subclaims, original_prompt)

        if subclaims:
            output = self.llm_model.generate_given_prompt(prompt)['generation']
        else:
            output = "I'm sorry, I don't know enough to answer this question."
            
        return output
