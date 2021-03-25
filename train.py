import os
import json
import requests
import time
import argparse
import copy

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

from bidaf_distill import BidirectionalAttentionFlowDistill, SquadReaderDistill

def download_data(data_dir, squad_ver):
    train_data_filename = "train-v"+str(squad_ver)+".json"
    dev_data_filename = "dev-"+str(squad_ver)+".json"

    train_data_path = os.path.join(data_dir, train_data_filename)
    dev_data_path = os.path.join(data_dir, dev_data_filename)

    train_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v"+str(squad_ver)+".json"
    dev_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v"+str(squad_ver)+".json"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(train_data_path):
        print("Downloading train data...")
        item = requests.get(train_data_url)
        data = item.json()
        with open(train_data_path, 'w') as f:
                json.dump(data, f)
    else:
        print("Train data already downloaded.")

    if not os.path.exists(dev_data_path):
        print("Downloading dev data...")
        item = requests.get(dev_data_url)
        data = item.json()
        with open(dev_data_path, 'w') as f:
                json.dump(data, f)
    else:
        print("Dev data already downloaded.")

    return train_data_path, dev_data_path

def load_data(train_data_path, dev_data_path, squad_ver, distill=False):
    token_indexers = {'token_characters': TokenCharactersIndexer(), 'tokens': SingleIdTokenIndexer()}
    # tokenizer = SpacyTokenizer(start_tokens=start_tokens, end_tokens=end_tokens)
    tokenizer = SpacyTokenizer()

    if distill:
        Reader = SquadReaderDistill
    else:
        Reader = SquadReader

    if squad_ver==1.1:
        dataset_reader = Reader.squad1(tokenizer=tokenizer, token_indexers=token_indexers)
    elif squad_ver==2.0:
        dataset_reader = Reader.squad2(tokenizer=tokenizer, token_indexers=token_indexers)
    else:
        raise

    print("Reading data...")
    tic = time.time()

    train_data = list(dataset_reader.read(train_data_path))
    dev_data = list(dataset_reader.read(dev_data_path))

    print("Time elapsed:", time.time()-tic)

    return train_data, dev_data

def build_data_loaders(train_data: List[Instance],
                       dev_data: List[Instance],
                       batch_size
                      ) -> Tuple[DataLoader, DataLoader]:

    train_loader = SimpleDataLoader(train_data, batch_size, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, batch_size, shuffle=False)

    return train_loader, dev_loader

def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    num_epochs: int,
    learning_rate: float = 0.001,
    cuda_device = None
) -> Trainer:

    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters, lr=learning_rate)  # type: ignore
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        cuda_device=cuda_device
    )
    print("Will train for", num_epochs, "epochs")
    return trainer

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--squad_ver", type=float, default=1.1)
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--save_dir", default="tmp/")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--cuda_device", type=int, default=-1)

    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--distill_weight", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=1)

    parser.add_argument("--distill_data_file", default="train-logits.csv")

    args = parser.parse_args()

    # pdb.set_trace()

    # define parameters
    data_dir = args.data_dir
    squad_ver = args.squad_ver
    save_dir = args.save_dir

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    cuda_device = args.cuda_device
    distill = args.distill
    distill_weight = args.distill_weight
    temperature = args.temperature

    distill_data_path = os.path.join(data_dir, args.distill_data_file)

    # download data and load
    if distill:
        _, dev_data_path = download_data(data_dir, squad_ver)
        train_data_path = distill_data_path
    else:
        train_data_path, dev_data_path = download_data(data_dir, squad_ver)
    train_data, dev_data = load_data(train_data_path, dev_data_path, squad_ver, distill)
    train_loader, dev_loader = build_data_loaders(train_data, dev_data, batch_size)

    # load pretrained model
    print("Loading model")
    if args.distill:
        model = BidirectionalAttentionFlowDistill.from_pretrained(distill_weight, temperature)
    else:
        bidaf_pred = pretrained.load_predictor("rc-bidaf")
        model = bidaf_pred._model

    # move to gpu if using
    if cuda_device >= 0:
        model.cuda(cuda_device)

    # index with vocab
    print("Indexing")
    tic = time.time()

    train_loader.index_with(model.vocab)
    dev_loader.index_with(model.vocab)

    print("Time elapsed:", time.time()-tic)

    # build trainer
    print("Building trainer")
    trainer = build_trainer(model, save_dir, train_loader, dev_loader, num_epochs, learning_rate, cuda_device)

    # train
    print("Starting training")
    tic = time.time()

    trainer.train()

    print("Finished training")
    print("Time elapsed:", time.time()-tic)

    # evaluate trained model
    print("Evaluating")
    tic = time.time()
    results = evaluate(model, dev_loader, cuda_device, output_file=None, predictions_output_file=None)
    print("Time elapsed:", time.time()-tic)

    pdb.set_trace()
