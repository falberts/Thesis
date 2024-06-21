import json
import os
import pandas as pd

def main():
    dir_json = './data_construction/summaries_json/'

    dir_disc = './data_construction/discussions_final/'
    
    dir_generated = './data_generated/'

    dir_eval = './eval_json/'
    
    folders = ["zeroshot_original", "zeroshot_improved", "article_top", "article_all", "article_all_explanation"]

    ids = os.listdir(dir_generated+"zeroshot_original/")
    ids = [id[:-4] for id in ids]

#     generated_summ = {}
#     example_summ = {}

#     for id in ids:
#         generated_summ[id] = {}
    
#         jsonf = open(dir_json+id+'.json', 'r')
#         json_data = json.load(jsonf)
#         for n in json_data:
#             json_data[n].pop('rank')

#         for folder in folders:
#             json_data[folder] = {}
#             file_path = os.path.join(dir_generated, folder, f"{id}.txt")
#             with open(file_path, 'r') as f:
#                 summ = f.read()
#                 json_data[folder]['summ'] = summ
#                 json_data[folder]['quality'] = {'factual consistency': ['?', "EXPLAIN RATING HERE"],
#                                                 'coverage': ['?', "EXPLAIN RATING HERE"],
#                                                 'coherence': ['?', "EXPLAIN RATING HERE"]}
#         # f = open(dir_eval+id+'.json', 'w')
#         # f.write(json.dumps(json_data, indent=4))
#         f = open(dir_disc+id+'.txt', 'r')
#         prompt = f"[START TEXT]{f.read()}[END TEXT]"
        
#         json_s = json.dumps(json_data, indent=4)

#         prompt += json_s
#         prompt += """
# \n\nPlease return the scores for the metrics factual consistency, coverage and coherence for the summaries of settings zeroshot_original, zeroshot_improved, article_top, article_all and article_all_explanation.
        
# For factual consistency, a score of 1 is assigned when all information is in line with the original text. A score of 2 is assigned, for example when the information is correct but contains little to no detail. A score of 3 is assigned when the summary contains information that is not in line with the original text.
        
# For coverage, a score of 1 is assigned when the summary covers all limitations mentioned in the original text. A score of 2 is assigned when all limitations are mentioned, but not explained. A score of 3 is assigned when the summary fails to mention all limitations.
        
# For coherence, a score of 1 is assigned when the summary is coherent and does not contain hallucinations. A score of 2 is assigned when the summary does not follow the instructions in the prompt perfectly, but is still coherent and does not include hallucinations. A score of 3 is assigned when the summary contains hallucinations or is incomprehensible, for example when abbreviations are used but never explained.
        
# Return your ratings in the same JSON format as used in the examples, so:


# {
# {<summary setting>: {
#         "summ": "<you can leave this string empty>",
#         "quality": {
#             "factual consistency": [
#                 <score>,
#                 "<explanation>"
#             ],
#             "coverage": [
#                 <score>,
#                 "<explanation>"
#             ],
#             "coherence": [
#                 <score>,
#                 "<explanation>"
#             ]
#         }
#     },
#     ...
# }

# Make sure use JSON formatting in your response, and use an indentation of 4 spaces.
# """
#         f.close()
#         f2 = open(dir_eval+'prompt/'+id+'.txt', 'w')
#         f2.write(prompt)
#         f2.close()

        # f3 = open(dir_eval+id+'.json', 'w')
        # json.dump({}, f3)

    
    metric_scores = {}
    for id in ids:

        jsonf = open(dir_json+id+'.json', 'r')
        json_data = json.load(jsonf)

        jsonf2 = open(dir_eval+id+'.json', 'r')
        json_data2 = json.load(jsonf2)

        json_data.update(json_data2)

        for summ_type in json_data:
            if summ_type not in metric_scores:
                metric_scores[summ_type] = {'factual consistency': {1: 0, 2: 0, 3: 0},
                                            'coverage': {1: 0, 2: 0, 3: 0},
                                            'coherence': {1: 0, 2: 0, 3: 0}}

            for metric in json_data[summ_type]['quality']:
                x = json_data[summ_type]['quality'][metric][0]
                metric = metric.lower()
                metric_scores[summ_type][metric][x] += 1

    df_metrics = pd.DataFrame(metric_scores).stack().apply(pd.Series).stack().unstack(level=-2)
    df_metrics.index.names = ['metric', 'score']
    print("\nMETRICS-DESCRIPTION\n" + "-" * 40)
    df_metrics = df_metrics.transpose()
    print(df_metrics)
    print(df_metrics.to_latex())


if __name__=="__main__":
    main()