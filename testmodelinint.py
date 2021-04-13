import os
import json
import requests
import time
import random

import pdb

from typing import Dict, Iterable, List, Tuple
from allennlp.data import (
        DataLoader,
        DatasetReader,
        Instance,
        Vocabulary,
        TextFieldTensors,
)
from allennlp.models import Model
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer


from allennlp_models import pretrained
from allennlp_models.rc.models import BidirectionalAttentionFlow
from allennlp_models.rc.dataset_readers import SquadReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.training.util import evaluate

from allennlp.data.batch import Batch

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.matrix_attention import LinearMatrixAttention

from train import *
from bidaf_distill import BidirectionalAttentionFlowDistill, SquadReaderDistill

import pandas as pd
import torch

def init_weights(m):
    # initializer = torch.nn.init.ones_ # for testing
    # LinearAttention -> reset implemented
    if isinstance(m, LinearMatrixAttention):
        m.reset_parameters()
    # Embedding -> weight
    if isinstance(m, Embedding):
        initializer = torch.nn.init.xavier_uniform_
        if m.weight.requires_grad:
            initializer(m.weight)
    # Conv1d -> weight, bias
    if isinstance(m, torch.nn.Conv1d):
        k = m.groups/(m.in_channels*m.kernel_size[0])
        initializer = torch.nn.init.uniform_
        initializer(m.weight, -1*k**0.5, k**0.5)
        initializer(m.bias, -1*k**0.5, k**0.5)
    # Linear -> weight, bias
    if isinstance(m, torch.nn.Linear):
        k = 1/m.in_features
        initializer = torch.nn.init.uniform_
        initializer(m.weight, -1*k**0.5, k**0.5)
        initializer(m.bias, -1*k**0.5, k**0.5)
    # LSTM -> ...
    if isinstance(m, torch.nn.LSTM):
        k = 1/m.hidden_size
        initializer = torch.nn.init.uniform_
        for p in m.parameters():
            if p.requires_grad:
                initializer(p, -1*k**0.5, k**0.5)

data_dir = "data/"
data_path = os.path.join(data_dir, "train-spacy-logits-small.csv")
# 
# pdb.set_trace()
# 
token_indexers = {'token_characters': TokenCharactersIndexer(), 'tokens': SingleIdTokenIndexer()}
dataset_reader = SquadReaderDistill.squad1(token_indexers=token_indexers)
# 
# print("reading data")
# tic = time.time()
instances = dataset_reader.read(data_path)
# 
# print(time.time() - tic)
# 
# pdb.set_trace()
instance_list = [i for i in instances]
# 
# for i in instance_list:
#     # teacher_start = int(torch.argmax(i["span_start_teacher_logits"].tensor))
#     # teacher_end = int(torch.argmax(i["span_end_teacher_logits"].tensor))
#     # teacher_answer = i["passage"][teacher_start:teacher_end+1]
#     teacher_start = None
#     teacher_end = None
#     teacher_answer = None
# 
#     start = i["span_start"].sequence_index
#     end = i["span_end"].sequence_index
#     actual_answer = i["passage"][start:end+1]
# 
#     print("question:", i["question"])
#     print("teacher answer:", teacher_answer,"actual answer:", actual_answer)
#     print("teacher span:", teacher_start, teacher_end, "actual span:", start, end)
#     print("\n")
# 
# pdb.set_trace()

# load model and try evaluating one batch

mdl = BidirectionalAttentionFlowDistill.from_pretrained(temperature=10, reduction="sum")

pdb.set_trace()

for i in instance_list[:3]:
    start = i["span_start"].sequence_index
    end = i["span_end"].sequence_index
    actual_answer = i["passage"][start:end+1]
    answer = " ".join([str(token) for token in actual_answer])
    print(answer)
    print("")

pdb.set_trace()

dataset = Batch(instance_list[:3])
dataset.index_instances(mdl.vocab)
mdl_input = dataset.as_tensor_dict()

output_1 = mdl(**mdl_input)
for ans in output_1['best_span_str']:
    print(ans)
    print("")

pdb.set_trace()

# for n, p in mdl.named_parameters():
#     key = "._module"
#     try:
#         var = n[:(n.index(key) + len(key))]
#         print(var)
#         exec("print(mdl." + var+")")
#     except:
#         pass

mdl.apply(init_weights)

for n, p in mdl.named_parameters():
    if not p.requires_grad:
        print(n, p.requires_grad, p)

pdb.set_trace()

output_2 = mdl(**mdl_input)
for ans in output_2['best_span_str']:
    print(ans)
    print("")

pdb.set_trace()

