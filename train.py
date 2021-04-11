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

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.matrix_attention import LinearMatrixAttention
import torch

def download_data(data_dir, squad_ver):
    """
    Downloads specified version of SQuAD dataset to specified directory

    Parameters
        data_dir : string
            folder to save dataset to
        squad_ver : float
            squad version to download (1.1 or 2.0)

    Returns
        train_data_path : string
            path to downloaded train data
        dev_data_path : string
            path to downloaded dev data
    """
    train_data_filename = "train-v"+str(squad_ver)+".json"
    dev_data_filename = "dev-v"+str(squad_ver)+".json"

    train_data_path = os.path.join(data_dir, train_data_filename)
    dev_data_path = os.path.join(data_dir, dev_data_filename)

    train_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v"+str(squad_ver)+".json"
    dev_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v"+str(squad_ver)+".json"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # download train data if file does not already exist
    if not os.path.exists(train_data_path):
        print("Downloading train data...")
        item = requests.get(train_data_url)
        data = item.json()
        with open(train_data_path, 'w') as f:
                json.dump(data, f)
    else:
        print("Train data already downloaded.")

    # download dev data if file does not already exist
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
    """
    Loads data from files
    Adapted from https://guide.allennlp.org/training-and-prediction

    Parameters
        train_data_path : string
            Path to train data of specified squad version
        dev_data_path : string
            Path to dev data of specified squad version
        squad_ver : float
            Squad version to use (1.1 or 2.0)
        distill : bool (default: False)
            Whether or not we are training with knowledge distillation.
            If True, assumes file at train_data_path is the csv file containing BERT logits.
            If False, assumes file at train_data_path is the original json file.

    Returns
        train_data : list of Instances
        dev_data : list of Instances
    """
    # make appropriate dataset reader classes
    token_indexers = {'token_characters': TokenCharactersIndexer(), 'tokens': SingleIdTokenIndexer()}
    tokenizer = SpacyTokenizer()

    if distill:
        TrainReader = SquadReaderDistill
    else:
        TrainReader = SquadReader

    DevReader = SquadReader

    if squad_ver==1.1:
        train_reader = TrainReader.squad1(tokenizer=tokenizer, token_indexers=token_indexers)
        dev_reader = DevReader.squad1(tokenizer=tokenizer, token_indexers=token_indexers)
    elif squad_ver==2.0:
        train_reader = TrainReader.squad2(tokenizer=tokenizer, token_indexers=token_indexers)
        dev_reader = DevReader.squad2(tokenizer=tokenizer, token_indexers=token_indexers)
    else:
        raise

    # load data from files and preprocess (save as Instances)
    print("Reading data")
    tic = time.time()

    if train_data_path is not None:
        train_data = list(train_reader.read(train_data_path))
    else:
        train_data = None

    if dev_data_path is not None:
        dev_data = list(dev_reader.read(dev_data_path))
    else:
        dev_data = None
    print("Time elapsed:", time.time()-tic)

    return train_data, dev_data

def build_data_loaders(train_data: List[Instance],
                       dev_data: List[Instance],
                       batch_size: int
                      ) -> Tuple[DataLoader, DataLoader]:
    """
    Creates data loaders which loads data in batches of size batch_size for training and validation
    Adapted from https://guide.allennlp.org/training-and-prediction
    """
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
    """
    Builds instance of Trainer class with specified training hyperparameters
    Adapted from https://guide.allennlp.org/training-and-prediction

    Parameters
        model : Model
            The model to train
        serialization_dir : str
            Directory to save checkpoints and results
        train_loader : DataLoader
            Previously built dataset loader for training data
        dev_loader : DataLoader
            Previously built loader for dev data
        num_epochs : int
            Number of epochs to train for
        learning_rate : float (default: 0.001)
        cuda_device : int (default: None)
            >=0 if using GPU

    Returns
        trainer : Trainer
    """
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

def init_weights(m):
    """
    Initializes bidfaf with random weights according to defaults of pytorch and allennlp
    (except for glove embeddings)
    """
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
    parser.add_argument("--distill_weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--reduction", default="mean")

    parser.add_argument("--distill_data_file", default="train-spacy-logits.csv")

    parser.add_argument("--from_scratch", action="store_true"
                       )
    args = parser.parse_args()

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
    reduction = args.reduction

    distill_data_path = os.path.join(data_dir, args.distill_data_file)

    from_stratch = args.from_scratch

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
    if distill:
        model = BidirectionalAttentionFlowDistill.from_pretrained(distill_weight, temperature, reduction)
    else:
        bidaf_pred = pretrained.load_predictor("rc-bidaf")
        model = bidaf_pred._model

    # reset weights if training from scratch
    if from_scratch:
        model.apply(init_weights)

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
    trainer.train()

    # pdb.set_trace()
