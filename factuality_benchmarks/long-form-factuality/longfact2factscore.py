import pickle
import json

# load 25+2 longfact generations
with open('./results/2024-09-19-18-57-18-SAFE.json', 'r') as f: # 25 
    scores = json.load(f)

with open('./results/2024-09-19-06-01-49-SAFE.json', 'r') as f: # 2
    sc2 = json.load(f)
    
tosave = []
for sc in [scores, sc2]:
    for score in sc['per_prompt_data']:
        for stat in score['side2_posthoc_eval_data']['checked_statements']:
            # import pdb; pdb.set_trace()
            if stat['annotation'] != 'Irrelevant': # irrelevant facts excluded
                tosave.append([stat['self_contained_atomic_fact'], 
                # tosave.append([stat['atomic_fact'], 
                               True if stat['annotation'] == 'Supported' else False])

print('Total records:', len(tosave))

with open('../metrics/longfact_27gen_notrefined.pkl', 'wb') as g:
    pickle.dump(tosave, g)
