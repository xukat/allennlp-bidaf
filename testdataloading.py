
import os
import json
import requests
import time

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

data_dir = "data/"
data_path = os.path.join(data_dir, "test.json")

# # download data and load
# train_data_path, dev_data_path = download_data(data_dir, squad_ver)
# train_data, dev_data = load_data(train_data_path, dev_data_path, squad_ver)
# train_loader, dev_loader = build_data_loaders(train_data, dev_data, batch_size)

bidaf_pred = pretrained.load_predictor("rc-bidaf")

token_indexers = {'token_characters': TokenCharactersIndexer(), 'tokens': SingleIdTokenIndexer()}

dataset_reader = SquadReader.squad1(token_indexers=token_indexers)

instances = dataset_reader.read(data_path)
# for i in instances:
#     print(i)

instance_list = [i for i in instances]

pdb.set_trace()

bidaf_mdl = bidaf_pred._model

dataset = Batch(instance_list)
dataset.index_instances(bidaf_mdl.vocab)
mdl_input = dataset.as_tensor_dict()

# output = bidaf_mdl(mdl_input['question'], mdl_input['passage'])
output = bidaf_mdl(**mdl_input)

pdb.set_trace()
