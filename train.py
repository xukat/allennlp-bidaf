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

from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.training.util import evaluate

from allennlp_models import pretrained
from allennlp_models.rc.models import BidirectionalAttentionFlow
from allennlp_models.rc.dataset_readers import SquadReader


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

def load_data(train_data_path, dev_data_path, squad_ver):
    token_indexers = {'token_characters': TokenCharactersIndexer(), 'tokens': SingleIdTokenIndexer()}
    if squad_ver==1.1:
        dataset_reader = SquadReader.squad1(token_indexers=token_indexers)
    elif squad_ver==2.0:
        dataset_reader = SquadReader.squad2(token_indexers=token_indexers)
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
    # define parameters
    data_dir = "data/"
    squad_ver=1.1

    save_dir = "tmp/"
    num_epochs = 1
    batch_size = 32
    learning_rate = 0.001
    cuda_device = None

    # download data and load
    train_data_path, dev_data_path = download_data(data_dir, squad_ver)
    train_data, dev_data = load_data(train_data_path, dev_data_path, squad_ver)
    train_loader, dev_loader = build_data_loaders(train_data, dev_data, batch_size)

    # load pretrained model
    print("Loading model")
    bidaf_pred = pretrained.load_predictor("rc-bidaf")
    model = bidaf_pred._model

    # index with vocab
    print("Indexing")
    tic = time.time()

    train_loader.index_with(model.vocab)
    dev_loader.index_with(model.vocab)

    print("Time elapsed:", time.time()-tic)

    # build trainer
    print("Building trainer")
    # trainer = build_trainer(model, save_dir, train_loader, dev_loader, num_epochs)
    trainer = build_trainer(model, save_dir, train_loader, dev_loader, num_epochs, learning_rate, cuda_device)

    # train
    print("Starting training")
    tic = time.time()

    trainer.train()

    print("Finished training")
    print("Time elapsed:", time.time()-tic)

    # evaluate trained model
    results = evaluate(model, dev_loader)

    # Pretrained model: start_acc: 0.30, end_acc: 0.31, span_acc: 0.20, em: 0.27, f1: 0.41, loss: 7.04 ||: : 1322it [05:46,  3.82it/s]
    # Trained one epoch: start_acc: 0.53, end_acc: 0.57, span_acc: 0.44, em: 0.53, f1: 0.65, loss: 3.39 ||: : 1322it [05:57,  3.70it/s]

    pdb.set_trace()
