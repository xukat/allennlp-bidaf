import os
import json
import requests
import time
import random
import argparse
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

from allennlp.common.tqdm import Tqdm

from train import *
from bidaf_distill import BidirectionalAttentionFlowDistill, SquadReaderDistill

import pandas as pd
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--squad_ver", type=float, default=1.1)
parser.add_argument("--data_dir", default="data/")
parser.add_argument("--data_file", default="dev-v1.1.json")
parser.add_argument("--save_dir", default="tmp/")
parser.add_argument("--epoch", default="best")
parser.add_argument("--cuda_device", type=int, default=-1)

parser.add_argument("--use_pretrained", action="store_true")

parser.add_argument("--predictions_output_file", default="predictions.csv")

args = parser.parse_args()

# define parameters
data_dir = args.data_dir
data_path = os.path.join(data_dir, args.data_file)

save_dir = args.save_dir
epoch = args.epoch
if epoch == "best":
    model_path = os.path.join(save_dir, "best.th")
else:
    model_path = os.path.join(save_dir, "model_state_epoch_{}.th".format(epoch))

cuda_device = args.cuda_device

use_pretrained = args.use_pretrained

predictions_file = os.path.join(save_dir, args.predictions_output_file)

# build dataset reader
token_indexers = {'token_characters': TokenCharactersIndexer(), 'tokens': SingleIdTokenIndexer()}
dataset_reader = SquadReader.squad1(token_indexers=token_indexers)

# read data
print("reading data")
tic = time.time()
instances = dataset_reader.read(data_path)
print(time.time() - tic)

# get list of instances

# load pretrained model
print("Loading model")
bidaf_pred = pretrained.load_predictor("rc-bidaf")
model = bidaf_pred._model

# load trained model
if not use_pretrained:
    if cuda_device < 0:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_path))

# move to gpu if using
if cuda_device >= 0:
    model.cuda(cuda_device)

# iterate through instances and save logits
print("Generating logits and saving")
tic = time.time()
rows = []
generator_tqdm = Tqdm.tqdm(instances)
instance_number = 0
# instance_list = [i for i in instances]
for i in generator_tqdm:
    try:
        # data = Batch([i])
        # data.index_instances(model.vocab)
        # mdl_input = data.as_tensor_dict()

        output = model.forward_on_instance(i)

        row = {}
        row['qas_id'] = i['metadata'].metadata['id']
        row['context_tokens'] = i['passage'].tokens
        row['question_tokens'] = i['question'].tokens
        row['span_start'] = i['span_start'].sequence_index
        row['span_end'] = i['span_end'].sequence_index
        row['start_logits'] = output['span_start_logits']
        row['end_logits'] = output['span_end_logits']
        row['start_probs'] = output['span_start_probs']
        row['end_probs'] = output['span_end_probs']

        rows.append(row)
    except:
        print("error occured, qas_id:", i['metadata'].metadata['id'], "instance number:", instance_number)
    instance_number += 1
print(time.time() - tic)

df = pd.DataFrame(rows)
df.to_csv(predictions_file,  mode='w', index = False)

pdb.set_trace()
