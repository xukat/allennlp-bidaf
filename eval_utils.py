""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import time
import pdb

from allennlp_models import pretrained
from allennlp_models.rc.models import BidirectionalAttentionFlow

import torch

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, model_path, cuda_device=-1):
    predictor = pretrained.load_predictor("rc-bidaf")
    if cuda_device < 0:
        predictor._model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        predictor._model.load_state_dict(torch.load(model_path))
        predictor._model.cuda(cuda_device)

    predictor._model.eval()

    f1 = exact_match = total = 0
    idx = 0
    tic = time.time()
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                ground_truths = list(map(lambda x: x['text'], qa['answers']))

                with torch.no_grad():
                    prediction = predictor.predict(qa['question'], paragraph['context'])

                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction['best_span_str'], ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction['best_span_str'], ground_truths)

        print("Article {}/{}, exact_match: {}, f1: {}, time: {}".format(idx, len(dataset), 100.0*exact_match/total, 100.0*f1/total, time.time()-tic))
        idx += 1

    time_elapsed = time.time() - tic
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1, 'time_elapsed': time_elapsed}


if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('model_path', help='Model weights')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    print(json.dumps(evaluate(dataset, args.model_path)))
