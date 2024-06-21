import matplotlib
import matplotlib.pyplot as plt
import numpy as np


font = {'family' : 'sans',
        'weight' : 'normal',
        'size'   : 12}
matplotlib.rc('font', **font)

################
# BY METRIC
################

metrics = ["Factual consistency", "Coverage", "Coherence"]
ranks = ["1", "2", "3"]
summ_types = ["zeroshot_original", "zeroshot_improved", "article_top", "article_all", "article_all_explanation"]

data = {
    "zeroshot_original": [
        [0.470, 0.600, 0.497],
        [0.473, 0.457, 0.474],
        [0.469, 0.478, 0.462]
    ],
    "zeroshot_improved": [
        [0.520, 0.701, 0.446],
        [0.519, 0.538, 0.472],
        [0.516, 0.480, 0.563]
    ],
    "article_top": [
        [0.514, 0.569, 0.508],
        [0.524, 0.516, 0.491],
        [0.512, 0.492, 0.527]
    ],
    "article_all": [
        [0.519, 0.580, 0.435],
        [0.531, 0.496, 0.457],
        [0.510, 0.489, 0.545]
    ],
    "article_all_explanation": [
        [0.475, 0.631, 0.414],
        [0.479, 0.473, 0.432],
        [0.465, 0.469, 0.450]
    ]
}

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Plot data
bar_width = 0.15
x = np.arange(len(ranks))

for i, metric in enumerate(metrics):
    ax = axs[i]
    for j, summ_type in enumerate(summ_types):
        scores = [data[summ_type][i][k] for k in range(len(ranks))]
        ax.bar(x + j * bar_width, scores, bar_width, label=summ_type)
    ax.set_xlabel('Rank')
    ax.set_title(metric)
    ax.set_xticks(x + bar_width * (len(summ_types) - 1) / 2)
    ax.set_xticklabels(ranks)

axs[0].set_ylabel('Scores')
axs[0].legend(loc='upper right', bbox_to_anchor=(1.2, 1))

plt.tight_layout()
# plt.show()
plt.savefig('by_metric_plot.png')

################
# BY RANK
################

# Data from the new table
ranks = ["1", "2", "3"]
summ_types = ["zeroshot_original", "zeroshot_improved", "article_top", "article_all", "article_all_explanation"]

new_data = {
    "zeroshot_original": [0.495, 0.491, 0.478],
    "zeroshot_improved": [0.520, 0.558, 0.499],
    "article_top": [0.510, 0.521, 0.504],
    "article_all": [0.499, 0.522, 0.490],
    "article_all_explanation": [0.465, 0.499, 0.443]
}

# Create bar plot
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.15
x = np.arange(len(ranks))

for j, summ_type in enumerate(summ_types):
    scores = new_data[summ_type]
    ax.bar(x + j * bar_width, scores, bar_width, label=summ_type)

ax.set_xlabel('Rank')
ax.set_ylabel('Scores')
ax.set_title('Scores by Summary Type and Rank')
ax.set_xticks(x + bar_width * (len(summ_types) - 1) / 2)
ax.set_xticklabels(ranks)
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

plt.tight_layout()
# plt.show()
plt.savefig('by_rank_plot.png')

##############################
# SCORES EVALUATION
##############################

categories = ["Factual consistency", "Coverage", "Coherence"]
ranks = ["1", "2", "3"]
models = ["ChatGPT", "Gemini", "MSCopilot", "Zero-shot original", "Zero-shot improved", "Article top", "Article all", "Article all explanation"]

data_new = {
    "ChatGPT": {
        "Factual consistency": [8, 1, 3],
        "Coverage": [4, 2, 6],
        "Coherence": [8, 2, 2]
    },
    "Gemini": {
        "Factual consistency": [11, 1, 0],
        "Coverage": [9, 1, 2],
        "Coherence": [8, 3, 1]
    },
    "MSCopilot": {
        "Factual consistency": [11, 0, 1],
        "Coverage": [10, 2, 0],
        "Coherence": [8, 3, 1]
    },
    "Zero-shot original": {
        "Factual consistency": [4, 2, 6],
        "Coverage": [4, 3, 5],
        "Coherence": [11, 0, 1]
    },
    "Zero-shot improved": {
        "Factual consistency": [9, 2, 1],
        "Coverage": [6, 4, 2],
        "Coherence": [10, 2, 0]
    },
    "Article top": {
        "Factual consistency": [11, 0, 1],
        "Coverage": [9, 2, 1],
        "Coherence": [11, 1, 0]
    },
    "Article all": {
        "Factual consistency": [9, 2, 1],
        "Coverage": [9, 1, 2],
        "Coherence": [10, 2, 0]
    },
    "Article all explanation": {
        "Factual consistency": [8, 2, 2],
        "Coverage": [10, 1, 1],
        "Coherence": [10, 0, 2]
    }
}

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

bar_width = 0.1
x = np.arange(len(ranks))

for i, category in enumerate(categories):
    ax = axs[i]
    for j, model in enumerate(models):
        scores = data_new[model][category]
        ax.bar(x + j * bar_width, scores, bar_width, label=model)
    ax.set_xlabel('Rank')
    ax.set_title(category)
    ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
    ax.set_xticklabels(ranks)

axs[0].set_ylabel('Scores')
axs[0].legend(loc='lower right', bbox_to_anchor=(1.05, 0.5))

plt.tight_layout()
# plt.show()
plt.savefig('scores_count.png')
