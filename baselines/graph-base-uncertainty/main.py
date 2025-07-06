import argparse
import os
import datasets
import torch
import src.utils as utils
from src.model_generate import Generation
from src.edge_construction import EdgeConstruction
from src.break_and_merge import Break_And_Merge
from src.uad import UncertaintyAwareDecoding
from src.models import get_model
import random

# os.environ["HF_DATASETS_CACHE"] = '/'
OUTPUT_DIR = 'experiments'
DATA_DIR = 'data'

parser = argparse.ArgumentParser()
parser.add_argument('--num_generations_per_prompt', type=int, default=10)  # as consistent with long_hallu
parser.add_argument('--model', type=str, default='llama-3.1-8b-instruct')
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--data_size', type=int, default=30)
parser.add_argument('--dataset', type=str, default='factscore')
parser.add_argument('--breakdown', type=utils.str2bool, default=True)
parser.add_argument('--num_samples_for_claims', type=int, default=4)
parser.add_argument('--gpt_annotate', type=utils.str2bool, default=True)
parser.add_argument('--sc_samples', type=int, default=4)
parser.add_argument('--uad', type=utils.str2bool, default=False)
parser.add_argument('--fs_eval', type=utils.str2bool, default=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--suff', type=str, default='')

args = parser.parse_args()

seed_value = args.seed
random.seed(seed_value)

dataset = datasets.load_from_disk(f'{DATA_DIR}/{args.dataset}')

indices = list(range(len(dataset)))
# random.Random(seed_value).shuffle(indices)
train_dataset = dataset[indices[:args.data_size]]
questions = datasets.Dataset.from_dict(train_dataset)
print(questions["entity"])

if 'factscore' in args.dataset:
    questions = questions.map(lambda x: utils.substitute_prompt_factscore(x, dataset=args.dataset))
elif 'nq' == args.dataset:
    questions = questions.map(lambda x: utils.substitute_prompt_nq(x))
elif 'pop_qa' in args.dataset:
    questions = questions.map(lambda x: utils.substitute_prompt_pop_qa(x))
else:
    args.breakdown = False

folder_name = f'{OUTPUT_DIR}/{args.dataset}/{args.model}{args.suff}' 
print('Folder Path:', folder_name)
os.makedirs(folder_name, exist_ok=True)

# llm_model = None 
llm_model = get_model(args.model, args)
openai_model = get_model("gpt-4o-mini", args)

def main():
    with torch.no_grad():
        g = Generation(args, dataset=questions, folder_name=folder_name, llm_model=llm_model)
        generation_result = g.generate()
        sequences = None
        if args.breakdown:
            bm = Break_And_Merge(args=args, generations=generation_result, llm_model=llm_model, folder_name=folder_name, gpt_annotate=args.gpt_annotate)
            sequences = bm.break_down_match()
        if args.sc_samples > 0:
            edge_const = EdgeConstruction(args=args, source_data=sequences, folder_name=folder_name, llm_model=openai_model)
            sequences = edge_const.evaluate_all_matches()
            if args.gpt_annotate:
                bm.plot_auroc(sequences)
        if args.uad:
            uad = UncertaintyAwareDecoding(args=args, generations=sequences, llm_model=llm_model, folder_name=folder_name, threshold_path=edge_const.collected_results_path, percentile_lst=[40], output_keys=['closeness', 'sc'])
            sequences = uad.merge_uad()
    
    print('Done!')

if __name__ == '__main__':
    main()
