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

import pdb

logger = logging.getLogger(__name__)

SQUAD2_NO_ANSWER_TOKEN = "@@<NO_ANSWER>@@"

def get_distill_loss(span_start_logits, span_end_logits,
                     span_start_teacher_logits, span_end_teacher_logits,
                     passage_mask, temperature=1, reduction="mean"):
    """
    Computes distill loss based on teacher logits
    Assumes teacher and student logits are both unnormalized and have size batch_size x context_len
    """
    # print("it works!")
    # print(span_start_teacher_logits.shape) # batch x context_len
    # print(span_end_teacher_logits.shape)
    # print(span_start_logits.shape)
    # print(span_end_logits.shape)

    # for now let's truncate logits to match their sizes
    # TODO remove this
    logit_len = span_start_logits.shape[1]
    span_start_teacher_logits = span_start_teacher_logits[:, 0:logit_len]
    span_end_teacher_logits = span_end_teacher_logits[:, 0:logit_len]

    masked_start_logits = util.masked_log_softmax(span_start_logits/temperature, passage_mask)
    masked_end_logits = util.masked_log_softmax(span_end_logits/temperature, passage_mask)

    masked_start_teacher_logits = util.masked_softmax(span_start_teacher_logits/temperature, passage_mask)
    masked_end_teacher_logits = util.masked_softmax(span_end_teacher_logits/temperature, passage_mask)

    span_start_loss = masked_start_teacher_logits.unsqueeze(1) @ masked_start_logits.unsqueeze(2)
    span_end_loss = masked_end_teacher_logits.unsqueeze(1) @ masked_end_logits.unsqueeze(2)

    distill_loss = -1 * temperature**2 * (span_start_loss + span_end_loss)
    # print("start loss:", torch.mean(span_start_loss), "end loss:", torch.mean(span_end_loss), "total loss:", torch.mean(distill_loss))

    if reduction == "mean":
        return torch.mean(distill_loss)
    elif reduction == "sum":
        return torch.sum(distill_loss)
    else:
        raise

class BidirectionalAttentionFlowDistill(BidirectionalAttentionFlow):
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

        super().__init__(vocab, text_field_embedder, num_highway_layers,
                         phrase_layer, matrix_attention, modeling_layer,
                         span_end_encoder, dropout, mask_lstms, initializer, regularizer)

        self.distill_weight = distill_weight
        self.temperature = temperature

    def forward(  # type: ignore
        self,
        question: Dict[str, torch.LongTensor],
        passage: Dict[str, torch.LongTensor],
        span_start: torch.IntTensor = None,
        span_end: torch.IntTensor = None,
        span_start_teacher_logits = None,
        span_end_teacher_logits = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        output_dict = super().forward(question, passage, span_start, span_end, metadata)

        if span_start_teacher_logits is not None:
            passage_mask = util.get_text_field_mask(passage)

            span_start_logits = output_dict["span_start_logits"]
            span_end_logits = output_dict["span_end_logits"]

            distill_loss = self.distill_weight * get_distill_loss(span_start_logits, span_end_logits,
                                                                  span_start_teacher_logits, span_end_teacher_logits,
                                                                  passage_mask, self.temperature)
            output_dict["distill_loss"] = distill_loss
            output_dict["loss"] += distill_loss

        return output_dict

    @classmethod
    def from_pretrained(cls, distill_weight=1, temperature=1):
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
    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        dataset = pd.read_csv(file_path)

        logger.info("Reading the dataset")

        for i, datapoint in dataset.iterrows():
            paragraph = datapoint.at["context_text"]
            tokenized_paragraph = self._tokenizer.tokenize(paragraph)
            question_text = datapoint.at["question_text"].strip().replace("\n", "")

            is_impossible = False

            answer_texts = [datapoint.at["answer_text"]]
            span_starts = [int(datapoint.at["start_position_character"])]
            span_ends = [start + len(answer) for start, answer in zip(span_starts, answer_texts)]

            additional_metadata = {"id": datapoint.at["qas_id"]}

            instance = self.text_to_instance(
                question_text,
                paragraph,
                is_impossible=is_impossible,
                char_spans=zip(span_starts, span_ends),
                answer_texts=answer_texts,
                passage_tokens=tokenized_paragraph,
                additional_metadata=additional_metadata,
            )

            span_start_teacher_logits = np.fromstring(datapoint.at["start_logits"].replace("\n", "").strip("[]"), sep=" ")
            span_end_teacher_logits = np.fromstring(datapoint.at["end_logits"].replace("\n", "").strip("[]"), sep=" ")

            instance.add_field("span_start_teacher_logits", TensorField(torch.tensor(span_start_teacher_logits, dtype=torch.float32)))
            instance.add_field("span_end_teacher_logits", TensorField(torch.tensor(span_end_teacher_logits, dtype=torch.float32)))

            if instance is not None:
                yield instance

### for reading from json ###s
#     def _read(self, file_path: str):
#         # if `file_path` is a URL, redirect to the cache
#         file_path = cached_path(file_path)
# 
#         logger.info("Reading file at %s", file_path)
#         with open(file_path) as dataset_file:
#             dataset_json = json.load(dataset_file)
#             dataset = dataset_json["data"]
#         logger.info("Reading the dataset")
#         for article in dataset:
#             for paragraph_json in article["paragraphs"]:
#                 paragraph = paragraph_json["context"]
#                 tokenized_paragraph = self._tokenizer.tokenize(paragraph)
# 
#                 for question_answer in self.shard_iterable(paragraph_json["qas"]):
#                     question_text = question_answer["question"].strip().replace("\n", "")
#                     is_impossible = question_answer.get("is_impossible", False)
#                     if is_impossible:
#                         answer_texts: List[str] = []
#                         span_starts: List[int] = []
#                         span_ends: List[int] = []
#                     else:
#                         answer_texts = [answer["text"] for answer in question_answer["answers"]]
#                         span_starts = [
#                             answer["answer_start"] for answer in question_answer["answers"]
#                         ]
#                         span_ends = [
#                             start + len(answer) for start, answer in zip(span_starts, answer_texts)
#                         ]
#                     additional_metadata = {"id": question_answer.get("id", None)}
#                     instance = self.text_to_instance(
#                         question_text,
#                         paragraph,
#                         is_impossible=is_impossible,
#                         char_spans=zip(span_starts, span_ends),
#                         answer_texts=answer_texts,
#                         passage_tokens=tokenized_paragraph,
#                         additional_metadata=additional_metadata,
#                     )
#                     #### Begin added code for teacher logits
#                     # TODO update after confirming how teacher logits are saved
#                     span_start_teacher_logits = question_answer["teacher_logits"]["start"]
#                     span_end_teacher_logits = question_answer["teacher_logits"]["end"]
# 
#                     instance.add_field("span_start_teacher_logits", TensorField(torch.tensor(span_start_teacher_logits, dtype=torch.float32)))
#                     instance.add_field("span_end_teacher_logits", TensorField(torch.tensor(span_end_teacher_logits, dtype=torch.float32)))
#                     #### End added code
#                     if instance is not None:
#                         yield instance
