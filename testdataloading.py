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

data_dir = "data/"
data_path = os.path.join(data_dir, "test_with_teacher_logits.json")

data = json.load(open(data_path))

# pdb.set_trace()
# # download data and load
# train_data_path, dev_data_path = download_data(data_dir, squad_ver)
# train_data, dev_data = load_data(train_data_path, dev_data_path, squad_ver)
# train_loader, dev_loader = build_data_loaders(train_data, dev_data, batch_size)

token_indexers = {'token_characters': TokenCharactersIndexer(), 'tokens': SingleIdTokenIndexer()}
# dataset_reader = SquadReader.squad1(token_indexers=token_indexers)
dataset_reader = SquadReaderDistill.squad1(token_indexers=token_indexers)
instances = dataset_reader.read(data_path)
# for i in instances:
#     print(i)

instance_list = [i for i in instances]

pdb.set_trace()

# i = 0
# newdata = data.copy()
# for article in newdata['data']:
#     for paragraph in article['paragraphs']:
#         for qa in paragraph['qas']:
#             assert qa['id'] == instance_list[i]['metadata'].metadata['id']
#             qa['teacher_logits'] = {'start': [random.randint(0,10) for _ in range(len(instance_list[i]['passage']))],
#                                     'end': [random.randint(0,10) for _ in range(len(instance_list[i]['passage']))]
#                                    }
#             i += 1
# 
# pdb.set_trace()
# 
# with open('data/test_with_teacher_logits.json', 'w') as outfile:
#     json.dump(newdata, outfile)
# 
# pdb.set_trace()

# bidaf_pred = pretrained.load_predictor("rc-bidaf")
# bidaf_mdl = bidaf_pred._model
bidaf_mdl = BidirectionalAttentionFlowDistill.from_pretrained()

dataset = Batch(instance_list)
dataset.index_instances(bidaf_mdl.vocab)
mdl_input = dataset.as_tensor_dict()

# output = bidaf_mdl(mdl_input['question'], mdl_input['passage'])
output = bidaf_mdl(**mdl_input)

pdb.set_trace()
