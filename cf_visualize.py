import os

import matplotlib.pyplot as plt
import numpy as np

from utils import load_pickle

dataset = 'wikiqa'
dataset_strong = 'wikiqa_question_cf_strong'
dataset_weak = 'wikiqa_question_cf_weak'

# row: test samples; col: train samples
if_datainf = load_pickle(os.path.join(f'output/{dataset}', 'if_proposed.pkl')).values.flatten()
if_datainf_cf_strong = load_pickle(os.path.join(f'output/{dataset_strong}', 'if_proposed.pkl')).values.flatten()
if_datainf_cf_weak = load_pickle(os.path.join(f'output/{dataset_weak}', 'if_proposed.pkl')).values.flatten()

# if_datainf = np.abs(if_datainf)
# if_datainf_cf_strong = np.abs(if_datainf_cf_strong)
# if_datainf_cf_weak = np.abs(if_datainf_cf_weak)

min_val = min(np.min(if_datainf), np.min(if_datainf_cf_strong), np.min(if_datainf_cf_weak))
max_val = min(np.max(if_datainf), np.max(if_datainf_cf_strong), np.max(if_datainf_cf_weak))

fig, axs = plt.subplots(1, 2, dpi=300, sharey=True, figsize=(12.8, 4.8))
axs[0].scatter(if_datainf_cf_strong, if_datainf, s=2)
axs[0].plot(np.linspace(min_val, max_val, 10), np.linspace(min_val, max_val, 10), c='black')
axs[0].set_title(f'avg diff: {np.mean(if_datainf - if_datainf_cf_strong):.4f}')
axs[0].set_ylabel('DataInf')
axs[0].set_xlabel('question CF (strong)')

axs[1].scatter(if_datainf_cf_weak, if_datainf, s=2)
axs[1].plot(np.linspace(min_val, max_val, 10), np.linspace(min_val, max_val, 10), c='black')
axs[1].set_title(f'avg diff: {np.mean(if_datainf - if_datainf_cf_weak):.4f}')
axs[1].set_ylabel('DataInf')
axs[1].set_xlabel('question CF (weak)')

plt.tight_layout()
plt.savefig(f'figs/cf/datainf_question.png')
