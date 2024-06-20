import os
import json
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer

def calculate_bertscore(ids, dir_json, dir_generated, folders):
        generated_summ = {}
        example_summ = {}
        metric_summ = {
            "Factual consistency": {
                1: {},
                2: {},
                3: {}
            },
            "Coverage": {
                1: {},
                2: {},
                3: {}
            },
            "Coherence": {
                1: {},
                2: {},
                3: {}
            }
        }

        metric_count = {
            "Factual consistency": {
                1: 0,
                2: 0,
                3: 0
            },
            "Coverage": {
                1: 0,
                2: 0,
                3: 0
            },
            "Coherence": {
                1: 0,
                2: 0,
                3: 0
            }
        }

        for id in ids:
            generated_summ[id] = {}
            for folder in folders:
                file_path = os.path.join(dir_generated, folder, f"{id}.txt")
                with open(file_path, 'r') as f:
                    summ = f.read()
                    generated_summ[id][folder] = summ

            example_summ[id] = {}
            jsonf = open(dir_json+id+'.json', 'r')
            json_data = json.load(jsonf)
            for n in json_data:
                rank = json_data[n]['rank']
                summ = json_data[n]['summ']
                example_summ[id][rank] = summ

                for x in json_data[n]['quality']:
                    score = json_data[n]['quality'][x][0]
                    metric_count[x][score] += 1
                    metric_summ[x][score][id] = summ

        # initialization BERT
        scorer = BERTScorer(model_type='bert-base-uncased', lang='en', rescale_with_baseline=True)

        total_scores = {}
        count_scores = {}
        total_scores_f1 = {}
        total_scores_by_metric = {}
        count_scores_by_metric = {}

        for summ_type in generated_summ[next(iter(generated_summ))]:  # use first id to find summ types
            total_scores[summ_type] = {
                1: {'precision': 0, 'recall': 0, 'f1': 0},
                2: {'precision': 0, 'recall': 0, 'f1': 0},
                3: {'precision': 0, 'recall': 0, 'f1': 0}
            }
            count_scores[summ_type] = {
                1: 0,
                2: 0,
                3: 0
            }

            total_scores_f1[summ_type] = {1: 0, 2: 0, 3: 0}
            total_scores_by_metric[summ_type] = {
                "Factual consistency": {1: 0, 2: 0, 3: 0},
                "Coverage": {1: 0, 2: 0, 3: 0},
                "Coherence": {1: 0, 2: 0, 3: 0}
                }
            count_scores_by_metric[summ_type] = {
                "Factual consistency": {1: 0, 2: 0, 3: 0},
                "Coverage": {1: 0, 2: 0, 3: 0},
                "Coherence": {1: 0, 2: 0, 3: 0}
                }
        
        for metric in metric_summ:
            for x in metric_summ[metric]:
                for id in metric_summ[metric][x]:
                    reference = metric_summ[metric][x][id]
                    for summ_type in generated_summ[id]:
                        candidate = generated_summ[id][summ_type]

                        # BERTScore
                        _, _, F1 = scorer.score([candidate], [reference])
                        f1 = F1.mean().item()

                        total_scores_by_metric[summ_type][metric][x] += f1
                        count_scores_by_metric[summ_type][metric][x] += 1

        ###################
        # BY METRIC
        ###################

        bertscores_avg_by_metric = {}
        for summ_type in total_scores_by_metric:
            bertscores_avg_by_metric[summ_type] = {
                "Factual consistency": {1: 0, 2: 0, 3: 0},
                "Coverage": {1: 0, 2: 0, 3: 0},
                "Coherence": {1: 0, 2: 0, 3: 0}
            }
            for metric in total_scores_by_metric[summ_type]:
                for x in total_scores_by_metric[summ_type][metric]:
                    if count_scores_by_metric[summ_type][metric][x] > 0:
                        bertscores_avg_by_metric[summ_type][metric][x] = round(total_scores_by_metric[summ_type][metric][x] / count_scores_by_metric[summ_type][metric][x], 3)

        ###############
        # BY RANK
        ###############

        for id in ids:
            for summ_type in generated_summ[id]:
                for n in range(1, 4):
                    reference = example_summ[id][n]
                    candidate = generated_summ[id][summ_type]

                    # BERTScore
                    P, R, F1 = scorer.score([candidate], [reference])
                    precision = P.mean().item()
                    recall = R.mean().item()
                    f1 = F1.mean().item()

                    total_scores[summ_type][n]['precision'] += precision
                    total_scores[summ_type][n]['recall'] += recall
                    total_scores[summ_type][n]['f1'] += f1
                    total_scores_f1[summ_type][n] += f1

                    count_scores[summ_type][n] += 1

        ######################
        # calculate avg scores
        ######################

        bertscores_avg = {}
        bertscores_avg_f1 = {}
        for summ_type in total_scores:
            bertscores_avg[summ_type] = {
                1: {'precision': 0, 'recall': 0, 'f1': 0},
                2: {'precision': 0, 'recall': 0, 'f1': 0},
                3: {'precision': 0, 'recall': 0, 'f1': 0}
            }
            bertscores_avg_f1[summ_type] = {1: 0, 2: 0, 3: 0}

            for n in total_scores[summ_type]:
                if count_scores[summ_type][n] > 0:
                    bertscores_avg[summ_type][n]['precision'] = round(total_scores[summ_type][n]['precision'] / count_scores[summ_type][n], 3)
                    bertscores_avg[summ_type][n]['recall'] = round(total_scores[summ_type][n]['recall'] / count_scores[summ_type][n], 3)
                    bertscores_avg[summ_type][n]['f1'] = round(total_scores[summ_type][n]['f1'] / count_scores[summ_type][n], 3)
                    bertscores_avg_f1[summ_type][n] = round(total_scores_f1[summ_type][n] / count_scores[summ_type][n], 3)


        data = {
        (rank, metric): [bertscores_avg[summ_type][rank][metric] for summ_type in bertscores_avg]
        for rank in range(1, 4)
        for metric in ['precision', 'recall', 'f1']
        }

        data_f1 = {
            rank: [bertscores_avg_f1[summ_type][rank] for summ_type in bertscores_avg_f1]
            for rank in range(1, 4)
        }

        data_by_metric = {
        (metric, rank): [bertscores_avg_by_metric[summ_type][metric][rank] for summ_type in bertscores_avg_by_metric]
        for metric in ['Factual consistency', 'Coverage', 'Coherence']
        for rank in range(1, 4)
        }

        # create pandas dataframes and print LaTeX code to generate tables
        df_metrics = pd.DataFrame(metric_count).transpose()
        df_metrics.columns.names = ['score']
        df_metrics.index.names = ['metrics']

        print("\METRICS-DESCIPTIVE\n"+"-"*40)
        print(df_metrics)
        print(df_metrics.to_latex())

        df_by_metric = pd.DataFrame(data_by_metric, index=bertscores_avg_by_metric.keys()).transpose()
        df_by_metric.index.names = ['metric', 'rank']
        df_by_metric.columns.names = ['summ_type']
        df_by_metric = df_by_metric.transpose()

        print("\nBERTSCORES BY METRIC\n"+"-"*40)
        print(df_by_metric)
        print(df_by_metric.to_latex(float_format="%.3f"))

        df_f1 = pd.DataFrame(data_f1, index=bertscores_avg_f1.keys()).transpose()
        df_f1.index.names = ['rank']
        df_f1.columns.names = ['summ_type']
        df_f1 = df_f1.transpose()

        print("\BERTSCORES BY RANK\n"+"-"*40)
        print(df_f1)
        print(df_f1.to_latex(float_format="%.3f"))

        df = pd.DataFrame(data, index=bertscores_avg.keys()).transpose()
        df.index.names = ['rank', 'metric']
        df.columns.names = ['summ_type']
        df = df.transpose()

        print("\nBERTSCORES BY RANK - DETAILED\n"+"-"*40)
        print(df)
        print(df.to_latex(float_format="%.3f"))



def main():
    dir_json = './data_construction/summaries_json/'
    dir_generated = './data_generated/'
    folders = ["zeroshot_original", "zeroshot_improved", "article_top", "article_all", "article_all_explanation"]

    ids = os.listdir(dir_generated+"zeroshot_original/")
    ids = [id[:-4] for id in ids]
    ids_no_lim = ['PMC10003412', 'PMC10180647','PMC10278010', 'PMC10628605']
    ids_lim = [i for i in ids if i not in ids_no_lim]

    print("#"*40)
    print("#"*40)
    print("GENERAL:")
    print("#"*40)
    print("#"*40)
    calculate_bertscore(ids, dir_json, dir_generated, folders)
    print("#"*40)
    print("#"*40)
    print("NO LIMITATIONS:")
    print("#"*40)
    print("#"*40)
    calculate_bertscore(ids_no_lim, dir_json, dir_generated, folders)
    print("#"*40)
    print("#"*40)
    print("LIMITATIONS PROVIDED:")
    print("#"*40)
    print("#"*40)
    calculate_bertscore(ids_lim, dir_json, dir_generated, folders)


if __name__=="__main__":
    main()