import os

import torch
from torch.nn import CrossEntropyLoss
from transformers import RobertaPreTrainedModel
import torch.nn as nn
from typing import Optional, Tuple, Union

from transformers.modeling_outputs import TokenClassifierOutput
import sys
sys.path.append("/workspace/SAC-XLM/src/Struct_XLM/")
from model import StructTransformer
from actor import Actor
from critic import Critic

import logging
logger = logging.getLogger(__name__)

class RobertaForTokenClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, args, config_class, model_class):
        super().__init__(config, args, config_class, model_class)
        self.num_labels = config.num_labels
        self.num_labels = config.num_labels
        self.args = args  # 保存args以便后续使用

        # 主模型结构
        self.encoder = StructTransformer(args, config_class, model_class)
        
        # 添加Actor和Critic作为子模块
        self.actor = Actor(experiment=args, d_model=args.edim, dropout=args.act_dropout, no_cuda=args.no_cuda)
        self.critic = Critic(experiment=args, d_model=args.edim, dropout=args.act_dropout, no_cuda=args.no_cuda)

       
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
        #self.encoder_init(args)

    def encoder_init(self, config):
        output_dir = os.path.join(config.critic_output_dir, "checkpoint-best")
        logger.info("Training/evaluation parameters %s", config)
        self.encoder.load_state_dict(torch.load(os.path.join(output_dir, "target_model_critic.bin")))

    def get_encoder(self, text, mask):
        return self.encoder.encoder(text, attention_mask=mask, struct_action_probs=None)

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
        #pad_token_label_id: int = -100,  # 新增：填充标签的ID（通常为-100）
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # self.encoder.eval()
        if get_embedding:
            return self.encoder.encoder(input_ids, attention_mask=attention_mask, struct_action_probs=None)
        outputs = self.encoder.encoder(
            input_ids,
            attention_mask=attention_mask,
            struct_action_probs=action,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if output_hidden_states:
            sequence_output = outputs['hidden_states'][15]
        else:
            sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        #sample_loss = None  # 新增：存储每个样本的损失（[batch_size]）
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            ## 1. 计算每个token的损失（不压缩，保留原始形状）
            #loss_fct = CrossEntropyLoss(reduction='none')  # 关键：reduction='none'
            # 输入形状：logits [batch_size*seq_len, num_labels]，labels [batch_size*seq_len]
            #token_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # [batch_size*seq_len]

            # 2. 重塑为 [batch_size, seq_len]，方便按样本处理
            #batch_size, seq_len = labels.shape
            #token_loss = token_loss.view(batch_size, seq_len)  # [batch_size, seq_len]
            
            # 3. 创建掩码：过滤填充标签（如-100）
            #valid_mask = (labels != pad_token_label_id).float()  # [batch_size, seq_len]，1表示有效token，0表示填充
            
            # 4. 计算每个样本的有效token数（避免除零）
            #valid_token_counts = valid_mask.sum(dim=1)  # [batch_size]
            #valid_token_counts = torch.clamp(valid_token_counts, min=1)  # 至少1个有效token
            
            # 5. 对每个样本的有效token损失取平均，得到样本级损失
            #sample_loss = (token_loss * valid_mask).sum(dim=1) / valid_token_counts  # [batch_size]
            #sample_loss = token_loss * valid_mask  # [batch_size,seq_len]
            # 6. 批次级标量损失（兼容原有逻辑，取样本损失的平均）
            #loss = sample_loss.mean()
            # 批次级总损失（对所有有效token取平均，保持原有loss计算）
            #loss = sample_loss.sum() / valid_mask.sum()  # 总损失（标量，避免除零）

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
            # 返回值：(总损失, 样本损失, logits, ...)
            #return ((loss, sample_loss) + output) if loss is not None else output

        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            )

