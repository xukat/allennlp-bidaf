import os
import json
import requests
import time

import pdb

from typing import Dict, Iterable, List, Tuple, Optional
import copy

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

from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.training.util import evaluate

from allennlp_models import pretrained
from allennlp_models.rc.models import BidirectionalAttentionFlow
from allennlp_models.rc.dataset_readers import SquadReader

from allennlp.nn.regularizers import RegularizerApplicator, L2Regularizer

from bidaf_distill import BidirectionalAttentionFlowDistill
from train import download_data, load_data, build_data_loaders

bidaf_pred = pretrained.load_predictor("rc-bidaf")
model = bidaf_pred._model

new_model = BidirectionalAttentionFlowDistill(model.vocab,
                                      copy.deepcopy(model._text_field_embedder),
                                      2,
                                      copy.deepcopy(model._phrase_layer),
                                      copy.deepcopy(model._matrix_attention),
                                      copy.deepcopy(model._modeling_layer),
                                      copy.deepcopy(model._span_end_encoder),
                                      mask_lstms = copy.deepcopy(model._mask_lstms),
                                      regularizer = copy.deepcopy(model._regularizer)
                                      )

new_model._highway_layer = copy.deepcopy(model._highway_layer)

pdb.set_trace()

# define parameters
data_dir = "data/"
squad_ver=1.1

batch_size = 32
learning_rate = 0.001

# download data and load
_, dev_data_path = download_data(data_dir, squad_ver)
_, dev_data = load_data(dev_data_path, dev_data_path, squad_ver)
_, dev_loader = build_data_loaders(dev_data, dev_data, batch_size)

print("Indexing")
dev_loader.index_with(model.vocab)

# evaluate trained model
print("Evaluating model")
tic = time.time()

results = evaluate(model, dev_loader)

print("Time elapsed:", time.time() - tic)
print(results)

pdb.set_trace()
