from typing import Dict, List, Tuple, Optional, Any
from allennlp.data import Vocabulary
from allennlp.data.fields import TensorField
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention import MatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

from allennlp_models import pretrained
from allennlp_models.rc.models import BidirectionalAttentionFlow
from allennlp_models.rc.dataset_readers import SquadReader

from allennlp.common.file_utils import cached_path

from overrides import overrides
import torch
import json
import logging
import copy
import pandas as pd
import numpy as np
import csv

import pdb

logger = logging.getLogger(__name__)

SQUAD2_NO_ANSWER_TOKEN = "@@<NO_ANSWER>@@"

def get_distill_loss(span_start_logits, span_end_logits,
                     span_start_teacher_logits, span_end_teacher_logits,
                     passage_mask, temperature=1, reduction="mean"):
    """
    Computes distill loss using teacher logits as the target
    Assumes teacher and student logits are both unnormalized and have size batch_size x context_len

    Parameters
        span_start_logits
            Student logits representing score for start of answer
        span_end_logits
            Student logits representing score for end of answer
        span_start_teacher_logits
            Teacher logits representing score for start of answer
        span_end_teacher_logits
            Teacher logits representing score for end of answer
        passage_mask
            To handle passages of different lengths within a batch,
            elements not corresponding to tokens in the passage are masked out
            so that the softmax is only taken over the tokens in the passage.
        temperature (default: 1)
            Rescaling logits for knowledge distillation. If temperature is 1, no rescaling happens.
        reduction ("mean" or "sum", default: "mean")
            Whether to take the sum or mean of loss contributions from all examples in the batch

    Returns
        Scalar loss value
    """
    # truncate length of logits in case they don't match. for testing.
    logit_len = span_start_logits.shape[1]
    span_start_teacher_logits = span_start_teacher_logits[:, 0:logit_len]
    span_end_teacher_logits = span_end_teacher_logits[:, 0:logit_len]

    # assert span_start_teacher_logits.shape[1] == span_start_logits.shape[1]
    # assert span_end_teacher_logits.shape[1] == span_end_logits.shape[1]

    # log softmax over logits, masking to handle passages of different lengths
    masked_start_logits = util.masked_log_softmax(span_start_logits/temperature, passage_mask)
    masked_end_logits = util.masked_log_softmax(span_end_logits/temperature, passage_mask)

    # softmax over targets, masking to handle passages of different lengths
    masked_start_teacher_logits = util.masked_softmax(span_start_teacher_logits/temperature, passage_mask)
    masked_end_teacher_logits = util.masked_softmax(span_end_teacher_logits/temperature, passage_mask)

    # log likelihood: dot product of softmax(logits) and logsoftmax(targets) for each (logit, target) pair in the batch
    span_start_loss = masked_start_teacher_logits.unsqueeze(1) @ masked_start_logits.unsqueeze(2)
    span_end_loss = masked_end_teacher_logits.unsqueeze(1) @ masked_end_logits.unsqueeze(2)

    # loss is negative log likelihood, sum contributions from span start and span end, and rescale by T^2
    distill_loss = -1 * temperature**2 * (span_start_loss + span_end_loss)

    # mean or sum over batch - default: mean
    if reduction == "mean":
        return torch.mean(distill_loss)
    elif reduction == "sum":
        return torch.sum(distill_loss)
    else:
        raise

class BidirectionalAttentionFlowDistill(BidirectionalAttentionFlow):
    """
    Inherit from `BidirecitonalAttentionFlow` defined at
    https://github.com/allenai/allennlp-models/blob/main/allennlp_models/rc/models/bidaf.py
    """
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        num_highway_layers: int,
        phrase_layer: Seq2SeqEncoder,
        matrix_attention: MatrixAttention,
        modeling_layer: Seq2SeqEncoder,
        span_end_encoder: Seq2SeqEncoder,
        dropout: float = 0.2,
        mask_lstms: bool = True,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        distill_weight: float = 1,
        temperature: float = 1
    ) -> None:
        """
        No changes from parent class except the addition of `distill_weight` and `temperature` attributes
        which are hyperparameters for knowledge distillation.

        Use class method `from_pretrained` to make a model with pretrained weights.
        """

        super().__init__(vocab, text_field_embedder, num_highway_layers,
                         phrase_layer, matrix_attention, modeling_layer,
                         span_end_encoder, dropout, mask_lstms, initializer, regularizer)

        self.distill_weight = distill_weight
        self.temperature = temperature

    def forward(
        self,
        question: Dict[str, torch.LongTensor],
        passage: Dict[str, torch.LongTensor],
        span_start: torch.IntTensor = None,
        span_end: torch.IntTensor = None,
        span_start_teacher_logits = None,
        span_end_teacher_logits = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs `forward` of parent class and then adds distill loss to the original loss if teacher logits are availiable.
        No changes from parent class if teacher logits are not availiable.
        """
        # run forward of parent class to get base loss
        output_dict = super().forward(question, passage, span_start, span_end, metadata)

        # add distill loss to base loss if teacher logits are availiable
        if span_start_teacher_logits is not None:
            passage_mask = util.get_text_field_mask(passage)

            span_start_logits = output_dict["span_start_logits"]
            span_end_logits = output_dict["span_end_logits"]

            distill_loss = get_distill_loss(span_start_logits, span_end_logits,
                                                                  span_start_teacher_logits, span_end_teacher_logits,
                                                                  passage_mask, self.temperature)
            output_dict["distill_loss"] = distill_loss
            output_dict["loss"] = (1 - self.distill_weight) * output_dict["loss"] + (self.distill_weight) * distill_loss

        return output_dict

    @classmethod
    def from_pretrained(cls, distill_weight=1, temperature=1):
        """
        Use this to initialize model with weights from the pretrained BiDAF model.
        Details on the pretrained model available here https://github.com/allenai/allennlp-models/blob/main/allennlp_models/modelcards/rc-bidaf.json
        """
        bidaf_pred = pretrained.load_predictor("rc-bidaf")
        model = bidaf_pred._model

        distill_model = BidirectionalAttentionFlowDistill(model.vocab,
                                              copy.deepcopy(model._text_field_embedder),
                                              2,
                                              copy.deepcopy(model._phrase_layer),
                                              copy.deepcopy(model._matrix_attention),
                                              copy.deepcopy(model._modeling_layer),
                                              copy.deepcopy(model._span_end_encoder),
                                              mask_lstms = copy.deepcopy(model._mask_lstms),
                                              regularizer = copy.deepcopy(model._regularizer),
                                              distill_weight = distill_weight,
                                              temperature = temperature
                                              )

        distill_model._highway_layer = copy.deepcopy(model._highway_layer)

        return distill_model

class SquadReaderDistill(SquadReader):
    """
    Inherit from `SquadReader` defined at
    https://github.com/allenai/allennlp-models/blob/main/allennlp_models/rc/dataset_readers/squad.py

    _read method copied from parent class and modified to read squad data from
    csv containing teacher logits instead of the original json file

    Note: can only handle squad 1.1
    """
    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)

        # load data from csv file
        dataset = pd.read_csv(file_path, dtype=str, keep_default_na=False)

        logger.info("Reading the dataset")

        # for error catching
        flag = True
        count_total = 0
        count_mismatch = 0
        random_logits = True

        # loop through data and convert each data point to an Instance
        # generates an iterable of Instances 
        for i, datapoint in dataset.iterrows():
            count_total += 1
            try:
                # context
                paragraph = datapoint.at["context_text"]
                tokenized_paragraph = self._tokenizer.tokenize(paragraph)

                # query
                question_text = datapoint.at["question_text"].strip().replace("\n", "")

                # because we are using squad 1.1
                is_impossible = False

                # answer
                answer_texts = [datapoint.at["answer_text"]]
                span_starts = [int(datapoint.at["start_position_character"])]
                span_ends = [start + len(answer) for start, answer in zip(span_starts, answer_texts)]

                # id
                additional_metadata = {"id": datapoint.at["qas_id"]}

                # create Instance (without teacher logits)
                instance = self.text_to_instance(
                    question_text,
                    paragraph,
                    is_impossible=is_impossible,
                    char_spans=zip(span_starts, span_ends),
                    answer_texts=answer_texts,
                    passage_tokens=tokenized_paragraph,
                    additional_metadata=additional_metadata,
                )

                # teacher logits
                span_start_teacher_logits = np.fromstring(datapoint.at["start_logits"].replace("\n", "").strip("[]"), sep=" ")
                span_end_teacher_logits = np.fromstring(datapoint.at["end_logits"].replace("\n", "").strip("[]"), sep=" ")

                # for testing
                if random_logits:
                    span_start_teacher_logits = np.random.random(span_start_teacher_logits.shape)
                    span_end_teacher_logits = np.random.random(span_end_teacher_logits.shape)

                # add teacher logits to Instance
                instance.add_field("span_start_teacher_logits", TensorField(torch.tensor(span_start_teacher_logits, dtype=torch.float32)))
                instance.add_field("span_end_teacher_logits", TensorField(torch.tensor(span_end_teacher_logits, dtype=torch.float32)))

                # check to make sure length of teacher logits is correct
                assert len(span_start_teacher_logits) == len(tokenized_paragraph)
                assert len(span_end_teacher_logits) == len(tokenized_paragraph)

            except AssertionError:
                # if length of eacher logits is incorrect, save information about the problematic data point
                count_mismatch += 1
                if flag:
                    with open("mismatch_errors.csv", "w") as fp:
                        flag = False
                        writer = csv.writer(fp)
                        writer.writerow(["id", "text", "len tokenized_paragraph", "len span_start_teacher_logits", "len span_end_teacher_logits"])
                        writer.writerow([additional_metadata["id"], paragraph, len(tokenized_paragraph), len(span_start_teacher_logits), len(span_end_teacher_logits)])
                else:
                    with open("mismatch_errors.csv", "a") as fp:
                        writer = csv.writer(fp)
                        writer.writerow([additional_metadata["id"], paragraph, len(tokenized_paragraph), len(span_start_teacher_logits), len(span_end_teacher_logits)])

                # exclude problematic data point from training
                instance = None

            except:
                # something else whent wrong, exclude data point from training
                print("ERROR! skipped datapoint:", i, datapoint.at["qas_id"], answer_texts, span_starts)
                instance = None

            # if nothing went wrong, yield the instance that was generated
            if instance is not None:
                yield instance

        if not flag:
            print("Number of logit length mismatches (data points skipped): ", count_mismatch, "/", count_total)
