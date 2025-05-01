import os.path
import warnings

import numpy as np
import torch

from src.influence import IFEngineGeneration, IFEngine
from src.lora_model import LORAEngineGeneration
from utils import load_pickle, save_pickle

warnings.filterwarnings("ignore")

# Please change the following objects to  "YOUR-LLAMA-PATH" and "YOUR-DATAINF-PATH"
project_path = "/mnt/share/fymeng/llm-influence-testing"
base_path = f"{project_path}/models/llama-wikiqa-lora"
dataset_name = 'wikiqa_answer_cf_strong'
output_dir = f'{project_path}/output/{dataset_name}'
lora_engine = LORAEngineGeneration(base_path=base_path,
                                   project_path=project_path,
                                   dataset_name=dataset_name)

# limit train and test samples size
# np.random.seed(42)
#
# train_size = 100
# train_idx = np.random.choice(np.arange(len(lora_engine.train_dataset)), train_size, replace=False)
# lora_engine.train_dataset = lora_engine.train_dataset.select(train_idx)
#
# test_size = 100
# test_idx = np.random.choice(np.arange(len(lora_engine.validation_dataset)), test_size, replace=False)
# lora_engine.validation_dataset = lora_engine.validation_dataset.select(test_idx)

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

print('loading dataset...')
tokenized_datasets, collate_fn = lora_engine.create_tokenized_datasets()

# preprocess gradients
if not os.path.isfile(os.path.join(output_dir, 'tr_grad_dict.pkl')):
    print('preprocessing gradients...')
    tr_grad_dict, val_grad_dict = lora_engine.compute_gradient(tokenized_datasets, collate_fn)
    save_pickle(os.path.join(output_dir, 'tr_grad_dict.pkl'), tr_grad_dict)
    save_pickle(os.path.join(output_dir, 'val_grad_dict.pkl'), val_grad_dict)
else:
    print('loading gradients')
    tr_grad_dict = load_pickle(os.path.join(output_dir, 'tr_grad_dict.pkl'))
    val_grad_dict = load_pickle(os.path.join(output_dir, 'val_grad_dict.pkl'))

if_engine_datainf = IFEngineGeneration()
if_engine_accurate = IFEngine()

if_engine_datainf.preprocess_gradients(tr_grad_dict, val_grad_dict)
if_engine_accurate.preprocess_gradients(tr_grad_dict, val_grad_dict, 0)

# computing HVP
if not os.path.isfile(os.path.join(output_dir, 'hvp_proposed.pkl')):
    print('computing DataInf HVP...')
    if_engine_datainf.compute_hvp_proposed()
    save_pickle(os.path.join(output_dir, 'hvp_proposed.pkl'), if_engine_datainf.hvp_dict['proposed'])
else:
    print('loading DataInf HVP')
    if_engine_datainf.hvp_dict['proposed'] = load_pickle(os.path.join(output_dir, 'hvp_proposed.pkl'))

if not os.path.isfile(os.path.join(output_dir, 'hvp_accurate.pkl')):
    print('computing ground-truth HVP...')
    if_engine_accurate.compute_hvp_accurate()
    save_pickle(os.path.join(output_dir, 'hvp_accurate.pkl'), if_engine_accurate.hvp_dict['accurate'])
else:
    print('loading ground-truth HVP')
    if_engine_accurate.hvp_dict['accurate'] = load_pickle(os.path.join(output_dir, 'hvp_accurate.pkl'))

# clear NaNs
for item in if_engine_datainf.hvp_dict['proposed'].values():
    for k, v in item.items():
        item[k] = torch.nan_to_num(v)

for k, v in if_engine_accurate.hvp_dict['accurate'].items():
    if_engine_accurate.hvp_dict['accurate'][k] = torch.nan_to_num(v)

# computing IF
if not os.path.isfile(os.path.join(output_dir, 'if_proposed.pkl')):
    print('computing DataInf IF...')
    if_engine_datainf.compute_IF()
    save_pickle(os.path.join(output_dir, 'if_proposed.pkl'), if_engine_datainf.IF_dict['proposed'])
else:
    print('loading DataInf IF')
    if_engine_datainf.IF_dict['proposed'] = load_pickle(os.path.join(output_dir, 'if_proposed.pkl'))

if not os.path.isfile(os.path.join(output_dir, 'if_accurate.pkl')):
    print('computing ground-truth IF...')
    if_engine_accurate.compute_IF()
    save_pickle(os.path.join(output_dir, 'if_accurate.pkl'), if_engine_accurate.IF_dict['accurate'])
else:
    print('loading ground-truth IF')
    if_engine_accurate.IF_dict['accurate'] = load_pickle(os.path.join(output_dir, 'if_accurate.pkl'))
