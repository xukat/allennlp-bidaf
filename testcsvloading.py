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

from train import *
from bidaf_distill import BidirectionalAttentionFlowDistill, SquadReaderDistill

import pandas as pd
import torch

data_dir = "data/"
data_path = os.path.join(data_dir, "traindata-small.json")
# data_path = os.path.join(data_dir, "train-spacy-logits-small.csv")

# data = json.load(open(data_path))

### open csv, read, and save to instances
# data = pd.read_csv(data_path, keep_default_na=False)

pdb.set_trace()

token_indexers = {'token_characters': TokenCharactersIndexer(), 'tokens': SingleIdTokenIndexer()}
dataset_reader = SquadReader.squad1(token_indexers=token_indexers)
# dataset_reader = SquadReaderDistill.squad1(token_indexers=token_indexers)

print("reading data")
tic = time.time()
instances = dataset_reader.read(data_path)

# for i in instances:
#     print(i)

print(time.time() - tic)

pdb.set_trace()
instance_list = [i for i in instances]

for i in instance_list:
    # teacher_start = int(torch.argmax(i["span_start_teacher_logits"].tensor))
    # teacher_end = int(torch.argmax(i["span_end_teacher_logits"].tensor))
    # teacher_answer = i["passage"][teacher_start:teacher_end+1]
    teacher_start = None
    teacher_end = None
    teacher_answer = None

    start = i["span_start"].sequence_index
    end = i["span_end"].sequence_index
    actual_answer = i["passage"][start:end+1]

    print("question:", i["question"])
    print("teacher answer:", teacher_answer,"actual answer:", actual_answer)
    print("teacher span:", teacher_start, teacher_end, "actual span:", start, end)
    print("\n")

pdb.set_trace()

# load model and try evaluating one batch
bidaf_mdl = BidirectionalAttentionFlowDistill.from_pretrained()

dataset = Batch(instance_list)
dataset.index_instances(bidaf_mdl.vocab)
mdl_input = dataset.as_tensor_dict()

# output = bidaf_mdl(mdl_input['question'], mdl_input['passage'])
output = bidaf_mdl(**mdl_input)

pdb.set_trace()
