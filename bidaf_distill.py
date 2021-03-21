from typing import Dict, List, Tuple, Optional, Any
from allennlp_models.rc.models import BidirectionalAttentionFlow

import torch
import pdb

def get_distill_loss(span_start_logits, span_end_logits, span_start_teacher_logits, span_end_teacher_logits):
    """
    Computes distill loss based on teacher logits
    """
    distill_loss = None

    return distill_loss

class BidirectionalAttentionFlowDistill(BidirectionalAttentionFlow):
    #def __init__(
    #    self,
    #    vocab: Vocabulary,
    #    text_field_embedder: TextFieldEmbedder,
    #    num_highway_layers: int,
    #    phrase_layer: Seq2SeqEncoder,
    #    matrix_attention: MatrixAttention,
    #    modeling_layer: Seq2SeqEncoder,
    #    span_end_encoder: Seq2SeqEncoder,
    #    dropout: float = 0.2,
    #    mask_lstms: bool = True,
    #    initializer: InitializerApplicator = InitializerApplicator(),
    #    regularizer: Optional[RegularizerApplicator] = None,
    #) -> None:

    #    super().__init__()

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

            span_start_logits = output_dict["span_start_logits"]
            span_end_logits = output_dict["span_end_logits"]

            distill_loss = get_distill_loss(span_start_logits, span_end_logits, span_start_teacher_logits, span_end_teacher_logits)

            output_dict["loss"] += distill_loss

        return output_dict
