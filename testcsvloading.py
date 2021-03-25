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

data_dir = "data/"
data_path = os.path.join(data_dir, "train-logits-small.csv")

# data = json.load(open(data_path))

# token_indexers = {'token_characters': TokenCharactersIndexer(), 'tokens': SingleIdTokenIndexer()}
# dataset_reader = SquadReader.squad1(token_indexers=token_indexers)
# dataset_reader = SquadReaderDistill.squad1(token_indexers=token_indexers)
# instances = dataset_reader.read(data_path)

### open csv, read, and save to instances
data = pd.read_csv(data_path)

pdb.set_trace()

token_indexers = {'token_characters': TokenCharactersIndexer(), 'tokens': SingleIdTokenIndexer()}
# dataset_reader = SquadReader.squad1(token_indexers=token_indexers)
dataset_reader = SquadReaderDistill.squad1(token_indexers=token_indexers)

print("reading data")
tic = time.time()
instances = dataset_reader.read(data_path)

# for i in instances:
#     print(i)

print(time.time() - tic)

pdb.set_trace()
instance_list = [i for i in instances]

pdb.set_trace()

# load model and try evaluating one batch
bidaf_mdl = BidirectionalAttentionFlowDistill.from_pretrained()

dataset = Batch(instance_list)
dataset.index_instances(bidaf_mdl.vocab)
mdl_input = dataset.as_tensor_dict()

# output = bidaf_mdl(mdl_input['question'], mdl_input['passage'])
output = bidaf_mdl(**mdl_input)

pdb.set_trace()
