import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple, Union
import logging
import sys

sys.path.append('/workspace/SAC-XLM-short/src/Struct_XLM')

from model import StructTransformer
from actor import Actor
from critic import Critic

logger = logging.getLogger(__name__)


class BertForSequenceClassification_custom(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, args, config_class, model_class):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config

        # 与 XLM-R 自定义版本一致：使用 StructTransformer 包装 backbone
        self.encoder = StructTransformer(args, config_class, model_class)

        # 与 XLM-R 自定义版本一致：保留 A2C 模块
        self.actor = Actor(
            experiment=args,
            d_model=args.edim,
            dropout=args.act_dropout,
            no_cuda=args.no_cuda
        )

        self.critic = Critic(
            experiment=args,
            d_model=args.edim,
            dropout=args.act_dropout,
            no_cuda=args.no_cuda
        )

        # 模仿 HuggingFace 官方 BertForSequenceClassification
        classifier_dropout = (
            config.classifier_dropout
            if getattr(config, "classifier_dropout", None) is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def get_encoder(self, text, mask):
        return self.encoder.encoder(
            text,
            attention_mask=mask,
            struct_action_probs=None
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        action: Optional[torch.LongTensor] = None,
        get_embedding=False,
    ) -> Union[Tuple, SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if get_embedding:
            return self.encoder.encoder(
                input_ids,
                attention_mask=attention_mask,
                struct_action_probs=None
            )

        outputs = self.encoder.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            struct_action_probs=action,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # HuggingFace 官方 BERT 分类器使用 pooled_output = outputs[1]
        # 但你的 StructTransformer 不一定稳定返回 pooler_output，
        # 所以这里手动取 [CLS]，等价于 BERT 分类任务常用做法。
        pooled_output = sequence_output[:, 0, :]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)

            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels),
                    labels.view(-1)
                )

            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )