import os
import json
import requests
import time
import argparse
import copy
import torch

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

from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer

from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.training.util import evaluate

from allennlp_models import pretrained
from allennlp_models.rc.models import BidirectionalAttentionFlow
from allennlp_models.rc.dataset_readers import SquadReader

from bidaf_distill import BidirectionalAttentionFlowDistill

from train import download_data, load_data, build_data_loaders

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--squad_ver", type=float, default=1.1)
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--save_dir", default="tmp/")
    parser.add_argument("--epoch", default="best")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cuda_device", type=int, default=-1)

    parser.add_argument("--use_pretrained", action="store_true")

    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--predictions_output_file", default="predictions.json")

    args = parser.parse_args()

    # define parameters
    data_dir = args.data_dir
    squad_ver = args.squad_ver

    save_dir = args.save_dir
    epoch = args.epoch
    if epoch == "best":
        model_path = os.path.join(save_dir, "best.th")
    else:
        model_path = os.path.join(save_dir, "model_state_epoch_{}.th".format(epoch))

    batch_size = args.batch_size
    cuda_device = args.cuda_device

    use_pretrained = args.use_pretrained

    if args.save_predictions:
        predictions_file = os.path.join(save_dir, args.predictions_output_file)
    else:
        predictions_file = None

    # download data and load
    _, dev_data_path = download_data(data_dir, squad_ver)
    _, dev_data = load_data(dev_data_path, dev_data_path, squad_ver)
    _, dev_loader = build_data_loaders(dev_data, dev_data, batch_size)

    # load pretrained model
    print("Loading model")
    bidaf_pred = pretrained.load_predictor("rc-bidaf")
    model = bidaf_pred._model

    # index with vocab
    print("Indexing")
    tic = time.time()

    dev_loader.index_with(model.vocab)

    print("Time elapsed:", time.time()-tic)

    # load trained model
    if not use_pretrained:
        if cuda_device < 0:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(model_path))

    # move to gpu if using
    if cuda_device >= 0:
        model.cuda(cuda_device)

    # evaluate trained model
    print("Evaluating")
    tic = time.time()
    results = evaluate(model, dev_loader, cuda_device, output_file=None, predictions_output_file=predictions_file)
    print("Time elapsed:", time.time()-tic)

    # pdb.set_trace()
