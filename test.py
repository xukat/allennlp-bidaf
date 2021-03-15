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


# pretrained_mdls = pretrained.get_pretrained_models()
# print(pretrained_mdls)


data_dir = "data/"
squad_ver=2.0

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

bidaf_pred = pretrained.load_predictor("rc-bidaf")

token_indexers = {'token_characters': TokenCharactersIndexer(), 'tokens': SingleIdTokenIndexer()}
if squad_ver==1.1:
    dataset_reader = SquadReader.squad1(token_indexers=token_indexers)
elif squad_ver==2.0:
    dataset_reader = SquadReader.squad2(token_indexers=token_indexers)
else:
    raise

instances = dataset_reader.read(dev_data_path)

tmp = True
for i in instances:
    if tmp:
        tmp = False
        instance = i
    else:
        break
print(instance)

# pdb.set_trace()
#for instance in instances[:10]:
#        print(instance)

bidaf_mdl = bidaf_pred._model
# pdb.set_trace()
# scratch_bidaf_mdl = BidirectionalAttentionFlow()


def run_training_loop(model, dataset_reader, train_data_path, dev_data_path):

    print("Reading data...")
    tic = time.time()
    train_data = list(dataset_reader.read(train_data_path))
    dev_data = list(dataset_reader.read(dev_data_path))
    print("Time elapsed:", time.time()-tic)

    train_loader, dev_loader = build_data_loaders(train_data, dev_data)
    train_loader.index_with(model.vocab)
    dev_loader.index_with(model.vocab)

    # You obviously won't want to create a temporary file for your training
    # results, but for execution in binder for this guide, we need to do this
    serialization_dir = "tmp/"
    trainer = build_trainer(model, serialization_dir, train_loader, dev_loader, num_epochs=1)
    print("Starting training")
    tic - time.time()
    trainer.train()
    print("Finished training")
    print("Time elapsed:", time.time()-tic)

    return model, dataset_reader


def build_data_loaders(train_data: List[Instance], dev_data: List[Instance], batch_size=8) -> Tuple[DataLoader, DataLoader]:

    train_loader = SimpleDataLoader(train_data, batch_size, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, batch_size, shuffle=False)

    return train_loader, dev_loader


def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    num_epochs: int
) -> Trainer:

    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters)  # type: ignore
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
    )

    return trainer

# pdb.set_trace()
run_training_loop(bidaf_mdl, dataset_reader, dev_data_path, dev_data_path)
pdb.set_trace()
