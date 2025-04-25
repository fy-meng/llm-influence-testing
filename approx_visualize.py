import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from utils import load_pickle

dataset = 'wikiqa'

output_dir = f'output/{dataset}'

# each entry is a train sample
if_accurate = load_pickle(os.path.join(output_dir, 'if_accurate.pkl'))
# row: test samples; col: train samples
if_datainf = load_pickle(os.path.join(output_dir, 'if_proposed.pkl'))
if_ekfac = np.load(os.path.join(output_dir, 'wikiqa_influence_matrix.npy')).T

plt.figure(dpi=300)
plt.scatter(if_accurate, np.mean(if_datainf, axis=0), s=5)
r = pearsonr(if_accurate, np.mean(if_datainf, axis=0)).statistic
plt.xlabel('ground-truth IF')
plt.ylabel('DataInf')
plt.title(f'corr = {r:.4f}')
plt.savefig(f'figs/{dataset}/gt_vs_datainf.png')

plt.figure(dpi=300)
plt.scatter(if_accurate, np.mean(if_ekfac, axis=0), s=5)
r = pearsonr(if_accurate, np.mean(if_ekfac, axis=0)).statistic
plt.xlabel('ground-truth IF')
plt.ylabel('EK-FAC')
plt.title(f'corr = {r:.4f}')
plt.savefig(f'figs/{dataset}/gt_vs_ekfac.png')

plt.figure(dpi=300)
c = np.arange(100)
c = np.repeat(c, 100)
plt.scatter(if_datainf.values.flatten(), if_ekfac.flatten(), s=5, c=c, cmap='tab10')
r = pearsonr(if_datainf.values.flatten(), if_ekfac.flatten()).statistic
plt.xlabel('DataInf')
plt.ylabel('EK-FAC')
plt.title(f'corr = {r:.4f}')
plt.savefig(f'figs/{dataset}/datainf_vs_ekfac.png')
