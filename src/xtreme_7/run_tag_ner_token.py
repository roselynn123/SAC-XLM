# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors,
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning models for NER and POS tagging."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import numpy
import torch
# from seqeval.metrics import precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import sys
sys.path.append("/workspace/SAC-XLM-short/src/Struct_XLM/")
from BertEncoder import BertModel
from RobertaEncoder import RobertaModel
from actor import Actor
from critic import Critic
from RobertaForTokenClassification import RobertaForTokenClassification
import torch.nn.functional as F
#from sklearn.metrics import f1_score

from seqeval.metrics.sequence_labeling import precision_score, recall_score, f1_score
from utils_tag import convert_examples_to_features
from utils_tag import get_labels
from utils_tag import read_examples_from_file

from transformers import (
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    BertConfig,
    BertTokenizer,
    BertForTokenClassification,
    XLMConfig,
    XLMTokenizer,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMRobertaForTokenClassification
)
from torch.optim import AdamW

from xlm import XLMForTokenClassification

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer, BertModel),
    "xlm": (XLMConfig, XLMForTokenClassification, XLMTokenizer, RobertaModel),
    "xlmr": (XLMRobertaConfig, RobertaForTokenClassification, XLMRobertaTokenizer, RobertaModel),
}

emb_config = {
    'mbert': 768,
    'xlmr': 1024,
    'infoxlm': 1024
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def Sampling_RL(args, predicted, mask, head_mask, epsilon, Random=True):
    actor_attention = torch.zeros(args.max_len, args.max_len)
    actions = []
    length = torch.sum(mask).item()
    constituent_num = 0  # (1, seq_len, 2)
    length_true = torch.sum(head_mask).item()
    for pos in range(length):
        if Random:
            if random.random() > epsilon:
                action = (0 if random.random() < float(predicted[pos][0].item()) else 1)
            else:
                action = (1 if random.random() < float(predicted[pos][0].item()) else 0)
        else:
            action = np.argmax(predicted[pos].cpu().detach().numpy()).item()
        if action == 1 and head_mask[pos] == 1:
            constituent_num += 1
        actions.append(action)
    pos = 1
    end_one = False
    while pos < length:
        # actor_attention[pos][pos] = 1
        if actions[pos] == 0:
            begin_sub = pos
            for i in range(pos + 1, length):
                if head_mask[i] == -1 and i != pos + 1:
                    continue
                elif head_mask[i] == -1 and i == begin_sub + 1:
                    begin_sub += 1
                actor_attention[pos][i] = 1
                actor_attention[i][pos] = 1
                if actions[i] == 1:
                    end_one = True
                    break
        elif actions[pos] == 1:
            if end_one:
                end_one = False
                if pos + 1 < length and actions[pos + 1] == -1:
                    for i in range(pos + 1, length):
                        actor_attention[pos][i] = 1
                        actor_attention[i][pos] = 1
                        if i + 1 < length and actions[i + 1] != -1:
                            break
                continue
            else:
                actor_attention[pos][pos] = 1
        else:
            if pos + 1 < length and head_mask[pos + 1] != -1:
                pos += 1
                continue
            for i in range(pos + 1, length):
                actor_attention[pos][i] = 1
                actor_attention[i][pos] = 1
                if i + 1 < length and head_mask[i + 1] != -1:
                    break
        pos += 1
    if sum(actions) > length or sum(actions) == 0:
        Rinput = actor_attention.masked_fill(actor_attention == 0, 1)
    else:
        Rinput = actor_attention.masked_fill(actor_attention == 0, -1e9)

    Rinput = Rinput.unsqueeze(0).unsqueeze(0).cuda()  # (1, seq_len, seq_len)
    return actions, Rinput, constituent_num/length_true if constituent_num!=0 else 1

def sampling_random(args, mask, p_action):#没有使用attention_scores
    lenth = torch.sum(mask).item()
    actions = np.copy(p_action.cpu()).tolist()
    # actions += [0] * (args.max_len - lenth)
    actor_attention = torch.zeros(args.max_len, args.max_len).cuda()
    for i in range(lenth):
        actor_attention[0][i] = 1
    pos = 1
    end_one = False
    while pos < lenth:
        #actor_attention[pos][pos] = 1
        if actions[pos] == 0:
            begin_sub=pos
            for i in range(pos + 1, lenth):
                if actions[i] == -1 and i!= pos+1:
                    continue
                elif actions[i] == -1 and i== begin_sub+1:
                    begin_sub +=1
                actor_attention[pos][i] = 1
                actor_attention[i][pos] = 1
                if actions[i] == 1:
                    end_one = True
                    break
        elif actions[pos] == 1:
            if end_one:
                end_one = False
                if pos+1<lenth and actions[pos+1] == -1:
                    for i in range(pos + 1, lenth):
                        actor_attention[pos][i] = 1
                        actor_attention[i][pos] = 1
                        if i+1<lenth and actions[i+1] != -1:
                            break
                continue
            else:
                actor_attention[pos][pos] = 1
        else:
            if pos + 1 < lenth and actions[pos + 1] != -1:
                pos += 1
                continue
            for i in range(pos + 1, lenth):
                actor_attention[pos][i] = 1
                actor_attention[i][pos] = 1
                if i+1 < lenth and actions[i+1] != -1:
                    break
        pos += 1
    # print(actor_attention)
    Rinput = actor_attention.masked_fill(actor_attention == 0, 0)

    Rinput = Rinput.unsqueeze(0).unsqueeze(0).cuda()
    return Rinput

#使用batch的随机采样
def Sampling_RL_batch(args, p_batch, mask_batch, head_mask_batch, epsilon=0.1, Random=True):
    """
    批量版的 Sampling_RL 函数。
    输入：
        - p_batch: (B, L)，策略输出
        - mask_batch: (B, L)
        - head_mask_batch: (B, L)
    输出：
        - actions_batch: (B, L)，策略采样动作
        - Rinput_batch: (B, 1, L, L)，结构连接矩阵
        - constituent_num_batch: (B,)，每句话中的片段数
    """
    
    device = p_batch.device
    if p_batch.dim() == 3 and p_batch.size(-1) == 2:
        # 输入是 [B, L, 2]，转为类别 {0,1}
        p_batch = torch.argmax(p_batch, dim=-1)  # [B, L]
    elif p_batch.dim() != 2:
        raise ValueError("Expected input p_batch to be [B, L] or [B, L, 2]")

    B, L = p_batch.shape
    actions_batch = torch.zeros_like(p_batch, dtype=torch.long)  # 初始化动作序列
    Rinput_batch = torch.zeros(B, L, L, device=device)            # 初始化结构矩阵
    constituent_num_batch = torch.zeros(B, dtype=torch.long)      # 存每个样本的结构数量

    for b in range(B):
        p = p_batch[b]
        mask = mask_batch[b]
        head_mask = head_mask_batch[b]

        length = int(torch.sum(mask).item())
        pred = p[:length]
        head = head_mask[:length]

        # 转换为动作值（假设：0 => non-boundary (-1)，1 => boundary start/end (1)）
        actions = []
        for val in pred.detach().cpu().tolist():
            if val == 0:
                actions.append(-1)
            elif val == 1:
                actions.append(1)
            else:
                actions.append(0)

        #actions = pred.detach().cpu().tolist()
        # ε-greedy 处理
        if Random:
            for i in range(length):
                if random.random() < epsilon:
                    actions[i] = random.choice([-1, 0, 1])

        # 记录动作序列
        #for i in range(length):
        #    actions_batch[b][i] = actions[i]
        actions_batch[b, :length] = torch.tensor(actions, device=device)

        # 构建连接矩阵
        actor_attention = torch.zeros(L, L, device=device)
        #for i in range(length):
        #    actor_attention[0][i] = 1

        pos = 1
        end_one = False
        constituent_num = 0

        while pos < length:
            if actions[pos] == 0:
                constituent_num += 1
                begin_sub = pos
                for i in range(pos + 1, length):
                    if actions[i] == -1 and i != pos + 1:
                        continue
                    elif actions[i] == -1 and i == begin_sub + 1:
                        begin_sub += 1
                    actor_attention[pos][i] = 1
                    actor_attention[i][pos] = 1
                    if actions[i] == 1:
                        end_one = True
                        break
            elif actions[pos] == 1:
                if end_one:
                    end_one = False
                    if pos + 1 < length and actions[pos + 1] == -1:
                        for i in range(pos + 1, length):
                            actor_attention[pos][i] = 1
                            actor_attention[i][pos] = 1
                            if i + 1 < length and actions[i + 1] != -1:
                                break
                        continue
                else:
                    actor_attention[pos][pos] = 1
                    constituent_num += 1
            else:
                if pos + 1 < length and actions[pos + 1] != -1:
                    pos += 1
                    continue
                for i in range(pos + 1, length):
                    actor_attention[pos][i] = 1
                    actor_attention[i][pos] = 1
                    if i + 1 < length and actions[i + 1] != -1:
                        break
            pos += 1

        #Rinput_batch[b] = actor_attention
        Rinput_batch[b, :, :] = actor_attention
        constituent_num_batch[b] = constituent_num

    Rinput_batch = Rinput_batch.unsqueeze(1)  # (B, 1, L, L)
    return actions_batch, Rinput_batch, constituent_num_batch


#batch版本随机采样
def sampling_random_batch(args, mask_batch, p_action_batch):
    """
    批量版本的 sampling_random，输入:
        - mask_batch: (B, L)
        - p_action_batch: (B, L)
    输出:
        - actor_attention_batch: (B, 1, L, L)
    """
    batch_size, max_len = mask_batch.shape
    device = mask_batch.device

    actor_attention_batch = torch.zeros(batch_size, max_len, max_len, device=device)

    for b in range(batch_size):
        mask = mask_batch[b]
        actions = p_action_batch[b].cpu().tolist()  # 转 list（避免 index 错误）
        lenth = int(torch.sum(mask).item())

        actor_attention = torch.zeros(max_len, max_len, device=device)
        for i in range(lenth):
            actor_attention[0][i] = 1

        pos = 1
        end_one = False
        while pos < lenth:
            if actions[pos] == 0:
                begin_sub = pos
                for i in range(pos + 1, lenth):
                    if actions[i] == -1 and i != pos + 1:
                        continue
                    elif actions[i] == -1 and i == begin_sub + 1:
                        begin_sub += 1
                    actor_attention[pos][i] = 1
                    actor_attention[i][pos] = 1
                    if actions[i] == 1:
                        end_one = True
                        break
            elif actions[pos] == 1:
                if end_one:
                    end_one = False
                    if pos + 1 < lenth and actions[pos + 1] == -1:
                        for i in range(pos + 1, lenth):
                            actor_attention[pos][i] = 1
                            actor_attention[i][pos] = 1
                            if i + 1 < lenth and actions[i + 1] != -1:
                                break
                        continue
                else:
                    actor_attention[pos][pos] = 1
            else:
                if pos + 1 < lenth and actions[pos + 1] != -1:
                    pos += 1
                    continue
                for i in range(pos + 1, lenth):
                    actor_attention[pos][i] = 1
                    actor_attention[i][pos] = 1
                    if i + 1 < lenth and actions[i + 1] != -1:
                        break
            pos += 1

        actor_attention_batch[b] = actor_attention

    actor_attention_batch = actor_attention_batch.masked_fill(actor_attention_batch == 0, 0)
    actor_attention_batch = actor_attention_batch.unsqueeze(1)  # (B, 1, L, L)
    return actor_attention_batch

# 保存训练前的模型参数
def get_model_parameters(model):
    return {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

#用来检查模型参数是否改变了的函数
def check_model_updated_and_log(before_params, after_params, log_path, model_name="model"):
    with open(log_path, "a") as f:  # 以追加模式打开文件
        f.write(f"\n==== {model_name} 参数更新检查 ====\n")
        for name in before_params:
            if not torch.equal(before_params[name], after_params[name]):
                diff = (after_params[name] - before_params[name]).abs().sum().item()
                f.write(f"{name} updated | Δsum: {diff:.6f}\n")
            else:
                f.write(f"{name} NOT updated\n")

def calculate_f1_reward_per_batch(y_true, logits, pad_token_label_id, label_map):
    """
    计算一个批次的 F1 分数作为奖励，并正确地将数据格式化为 seqeval 所需的形式。

    Args:
        y_true (torch.Tensor): 真实标签，形状为 [batch_size, seq_len]。
        logits (torch.Tensor): 模型输出的 logits，形状为 [batch_size, seq_len, num_labels]。
        pad_token_label_id (int): 填充 token 的标签 ID。
        label_map (dict): 标签 ID 到标签名称的映射，例如 {0: 'O', 1: 'B-PER', ...}。

    Returns:
        float: F1 分数作为奖励值。
    """
    # 1. 从 logits 中获取预测标签
    predicted_labels = torch.argmax(logits, dim=-1)

    # 2. 将真实标签和预测标签转换为列表的列表，并排除填充 token
    true_labels_list = []
    pred_labels_list = []

    for i in range(y_true.size(0)): # 遍历批次中的每个样本
        # 用于存储当前样本的真实和预测标签
        sample_true = []
        sample_pred = []

        for j in range(y_true.size(1)): # 遍历序列中的每个 token
            # 排除填充 token
            if y_true[i, j] != pad_token_label_id:
                # 使用 label_map 将 ID 转换为字符串标签，这对于 seqeval 至关重要
                sample_true.append(label_map[y_true[i, j].item()])
                sample_pred.append(label_map[predicted_labels[i, j].item()])

        # 如果样本中有有效标签，则添加到列表中
        if sample_true:
            true_labels_list.append(sample_true)
            pred_labels_list.append(sample_pred)
        
    # 如果整个批次都没有有效标签，返回0
    if not true_labels_list:
        return 0.0

    # 3. 计算 F1 分数
    # seqeval 的 f1_score 函数的输入就是列表的列表
    f1 = f1_score(true_labels_list, pred_labels_list)
    
    return f1

def calculate_f1_reward_per_sample(y_true, logits, pad_token_label_id, label_map):
    predicted_labels = torch.argmax(logits, dim=-1)  # 形状 [batch_size, seq_len]
    # 奖励张量，初始值设为 -1.0
    rewards = torch.full((y_true.size(0), y_true.size(1)), -1.0, dtype=torch.float, device=y_true.device)
    # 获取每个样本的F1分数
    f1_scores_per_sample = []
    for i in range(y_true.size(0)):
        sample_true = []
        sample_pred = []
        for j in range(y_true.size(1)):
            if y_true[i, j] != pad_token_label_id:
                sample_true.append(label_map[y_true[i, j].item()])
                sample_pred.append(label_map[predicted_labels[i, j].item()])  # 直接使用argmax后的标签
        if not sample_true:
            f1_scores_per_sample.append(0.0)
        else:
            f1_scores_per_sample.append(f1_score([sample_true], [sample_pred]))
    

    # 遍历每个样本的每个token来计算奖励
    for i in range(y_true.size(0)):
        for j in range(y_true.size(1)):
            if y_true[i, j] != pad_token_label_id:
                is_correct = (predicted_labels[i, j] == y_true[i, j])
                
                # 核心改变：奖励与该样本的F1分数相关
                if is_correct:
                    rewards[i, j] = f1_scores_per_sample[i] # 正确的token获得该样本的F1分数作为奖励
                else:
                    rewards[i, j] = -1.0 # 错误的token得到固定惩罚
            else:
                rewards[i, j] = 0.0 # 填充token的奖励为0

    return torch.tensor(rewards, device=y_true.device)  # 返回形状 [batch_size]

def calculate_token_reward(y_true, logits, pad_token_label_id):
    """
    计算逐个 Token 的奖励，返回一个与输入形状相同的张量。

    Args:
        y_true (torch.Tensor): 真实标签，形状为 [batch_size, seq_len]。
        logits (torch.Tensor): 模型输出的 logits，形状为 [batch_size, seq_len, num_labels]。
        pad_token_label_id (int): 填充 token 的标签 ID。

    Returns:
        torch.Tensor: 逐个 Token 的奖励，形状为 [batch_size, seq_len]，值在 [0.0, 1.0] 之间。
    """
    # 1. 从 logits 中获取预测标签
    predicted_labels = torch.argmax(logits, dim=-1)

    # 2. 比较预测标签和真实标签
    # 奖励张量，初始值设为惩罚值
    rewards = torch.full_like(y_true, -1.0, dtype=torch.float) 
    
    # 找到填充 Token 的位置，将其奖励设为 0.0 (不参与计算)
    rewards[y_true == pad_token_label_id] = 0.0
    
    # 3. 逐个 Token 赋值奖励
    # 找到预测正确且不是填充 Token 的位置
    correct_and_valid = (predicted_labels == y_true) & (y_true != pad_token_label_id)
    
    # 将这些位置的奖励设置为 1.0
    rewards[correct_and_valid] = 1.0
    '''
    # 在这里添加打印语句
    # 打印rewards张量，为了更好地查看，可以先将其移至CPU并转换为numpy数组
    print("--- Token-level Rewards ---")
    print("Shape:", rewards.shape)
    print("Rewards Tensor (first few rows):", rewards.cpu().numpy())
    
    # 你还可以打印其他相关信息进行对比
    print("--- Corresponding True Labels (first few rows) ---")
    print("True Labels:", y_true.cpu().numpy())
    print("--- Corresponding Predicted Labels (first few rows) ---")
    print("Predicted Labels:", predicted_labels.cpu().numpy())
    print("-----------------------------------")'''

    return rewards

def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id, lang2id=None):
    """Train the model."""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    # model.to('cuda:2')
    # 在函数开头创建一个标签 ID 到标签名称的映射
    label_map = {i: label for i, label in enumerate(labels)}

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    def optimizer_schedule(model):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=t_total * args.warmup_steps,
                                                    num_training_steps=t_total)
        return optimizer, scheduler
    #分别出actor_optimizer与critic_optimizer
    args.learning_rate = args.learning_rate*args.tau#强化学习时计算出学习率
    main_optimizer, main_schedule = optimizer_schedule(model)#主模型的
    actor_target_optimizer, actor_target_schedule = optimizer_schedule(model.actor.policy_network)#for policynetwork，使用actor的主函数获得model(也就是embedding的内容actor，以及影射好的对子们context_emb)作为输入，输入到这个optimizer_schedule函数中获得优化器以及学习率调节器
    critic_target_optimizer, critic_target_schedule = optimizer_schedule(model.critic.policy_network)#用来控制critic网络的

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    rl_start_epoch = 2
    best_score = 0.0
    best_checkpoint = None
    patience = 0
    gpu_epoch_times = []  # 记录每个 epoch 的 GPU 时间
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Add here for reproductibility (even between python 2 and 3)

    meta_train_error = 0.0
    step = 0

    #for _ in train_iterator:
    for epoch_idx, _ in enumerate(train_iterator):
        # 开始记录当前epoch的GPU时间
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch if t is not None)
            text = batch[0]
            mask = batch[1]
            segment = batch[2]
            label = batch[3]#[bachisize,seq_len]
            
            head_mask = mask.masked_fill(label == -100, 0).long()

            context = model(input_ids=text, attention_mask=mask, get_embedding=True)['last_hidden_state']#(16,128,1024)=>(batch_size,seq_len,hidden_dim)
            #predicted = model.actor.get_target_output(context, mask)#（16，128,2）
            predicted,context_emb = model.actor.policy_network(context, mask)

            sentence_actor_attention = None
            # On the query data
            #loss = model(input_ids=text, attention_mask=mask, token_type_ids=segment, labels=label,
                         #action=sentence_actor_attention, output_hidden_states=args.output_hidden_states)[0].mean()
            # loss_qry_all += loss_qry
            # loss += (1 * constituent_num_total + 0.1 / constituent_num_total - 0.6) * 0.1 * args.grained
            
            actions, sentence_actor_attention, _ = Sampling_RL_batch(args, predicted, mask, head_mask, args.epsilon,Random=False)         
            outputs = model(input_ids=text, attention_mask=mask, token_type_ids=segment, labels=label,
                         action=sentence_actor_attention, output_hidden_states=args.output_hidden_states)
            loss, logits = outputs[:2]  
            #精度更高的f1分数reward
            f1_rewards = calculate_f1_reward_per_sample(label, logits, pad_token_label_id, label_map)

            # 计算逐个token 分数作为奖励
            #token_rewards = calculate_token_reward(label, logits, pad_token_label_id)
            #计算出来的f1分数
            #f1_rewards = calculate_f1_reward_per_sample(label, logits, pad_token_label_id, label_map)
            #final_rewards = token_rewards.clone()
            final_rewards=f1_rewards
            # 2. 创建一个布尔掩码（mask），用于标识 token_delta 中值为 1.0 的位置,只在预测正确的 token 上添加 F1 奖励
            #correct_mask = (token_rewards == 1.0)
            # 4. 只在 correct_token_mask 为 True 的位置上，将 F1 奖励值添加到 final_rewards
            #f1_rewards_expanded = (f1_rewards).unsqueeze(1).expand_as(token_rewards)  # [batch_size, seq_len]
            #final_rewards[correct_mask] += f1_rewards_expanded[correct_mask]
            # 严格忽略填充token（设为0）
            final_rewards[label == pad_token_label_id] = 0.0
            #critic估值
            state_values=model.critic.policy_network(context_emb)#torch.Size( [16,128])
        
            # 显式指定dtype与state_values一致
            norm_delta_tensor = torch.tensor(
                final_rewards, 
                device=args.device, 
                dtype=state_values.dtype  # 确保与state_values数据类型一致
            )
            # 4. 计算advantage
            advantages = norm_delta_tensor - state_values.detach()  # [batch_size,seq_len]
            #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            #计算策略的熵，防止actor过于自信从而过早地停止了学习与探索
            log_probs = predicted.log_softmax(dim=-1)
            probs = predicted.softmax(dim=-1)
            # 计算策略的熵
            entropy = -(log_probs * probs).sum(dim=-1).mean()
            actor_loss = -(log_probs * advantages.unsqueeze(-1)).mean() + 0.5 * entropy
            # 6. 计算critic loss
            critic_loss = F.mse_loss(state_values, norm_delta_tensor)

            '''
            totalloss = loss + actor_loss + critic_loss
            totalloss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            meta_train_error += totalloss.item()
            actor_target_optimizer.step()
            actor_target_schedule.step()
            critic_target_optimizer.step()
            critic_target_schedule.step()
            main_optimizer.step()
            main_schedule.step()'''
             # 判断当前是否处于纯监督训练阶段
            if epoch_idx < rl_start_epoch:
                # 第一阶段：纯监督微调
                totalloss = loss
                print(f"Stage 1: Pure Supervised Learning | Total Loss: {totalloss.item():.4f}")

            else:
                # 第二阶段：混合微调
                totalloss = loss + 10 * actor_loss + 10 * critic_loss
                # 打印信息，以便监控训练阶段
                print(f"Stage 2: Mixed Learning | Total Loss: {totalloss.item():.4f}")
            # ------------------- 核心修改部分结束 -------------------
            
            totalloss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            meta_train_error += totalloss.item()

            # 它们有独立的优化器，则需要分别调用
            if epoch_idx < rl_start_epoch:
                main_optimizer.step()
                main_schedule.step()
            else:
                actor_target_optimizer.step()
                actor_target_schedule.step()
                critic_target_optimizer.step()
                critic_target_schedule.step()
                main_optimizer.step()
                main_schedule.step()

            model.zero_grad()

            global_step += 1

            if global_step % 100 == 0:
                logger.info(f"Step {global_step}:")
                logger.info(f"advantages {advantages}:")
                logger.info(f"  output1 Loss: {loss.item():.4f}")
                logger.info(f"  Actor Loss: {actor_loss.item():.4f}")
                logger.info(f"  Critic Loss: {critic_loss.item():.4f}")
                logger.info(f"  norm_delta_tensor: {norm_delta_tensor}")

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                if args.local_rank == -1 and args.evaluate_during_training:
                    # Only evaluate on single GPU otherwise metrics may not average well
                    results, _ = evaluate(args, model,tokenizer, labels, pad_token_label_id, mode="dev",
                                          lang=args.train_langs, lang2id=lang2id)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                tb_writer.add_scalar("lr", main_schedule.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                if args.save_only_best_checkpoint:
                    result, _ = evaluate(args, model,tokenizer, labels, pad_token_label_id, mode="dev",
                                         prefix=global_step, lang=args.train_langs, lang2id=lang2id)
                    if result["f1"] > best_score:
                        logger.info("result['f1']={} > best_score={}".format(result["f1"], best_score))
                        best_score = result["f1"]
                        # Save the best model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-best" + (
                            "" if args.train_langs == 'en' else f"_{args.train_langs}"))
                        best_checkpoint = output_dir
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving the best model checkpoint to %s", output_dir)
                        logger.info("Reset patience to 0")
                        patience = 0
                    else:
                        patience += 1
                        logger.info("Hit patience={}".format(patience))
                        if args.eval_patience > 0 and patience > args.eval_patience:
                            logger.info("early stop! patience={}".format(patience))
                            epoch_iterator.close()
                            train_iterator.close()
                            if args.local_rank in [-1, 0]:
                                tb_writer.close()
                            return global_step, tr_loss / global_step
                else:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step) + (
                        "" if args.train_langs == 'en' else f"_{args.train_langs}"))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            if patience > 10:
                break
        
        end_event.record()
        torch.cuda.synchronize()
        epoch_time_ms = start_event.elapsed_time(end_event)
        epoch_time_sec = epoch_time_ms / 1000.0
        gpu_epoch_times.append(epoch_time_sec)
        logger.info(f"Epoch {epoch_idx} GPU time: {epoch_time_sec:.2f} seconds")
        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
            # Save the best model checkpoint
        output_dir = os.path.join(args.output_dir,
                                  "checkpoint-final" + ("" if args.train_langs == 'en' else f"_{args.train_langs}"))
        # best_checkpoint = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving the final model checkpoint to %s", output_dir)

        if patience > 10:
            break
        #训练结束
        
    
    #写入GPU时间日志
    gpu_log_file = os.path.join(args.output_dir, "NERtest2_gpu_time_log.txt")
    with open(gpu_log_file, "a") as f:
        for i, t in enumerate(gpu_epoch_times):
            f.write(f"Epoch {i}: {t:.2f} seconds\n")
        f.write(f"\nTotal GPU training time: {sum(gpu_epoch_times):.2f} seconds\n")
        f.write(f"Average per epoch: {sum(gpu_epoch_times)/len(gpu_epoch_times):.2f} seconds\n")

    logger.info(f"GPU 时间日志已写入：{gpu_log_file}")

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model,  tokenizer, labels, pad_token_label_id, mode, prefix="", lang="en", lang2id=None,
             print_result=True):
    global preds_list
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode, lang=lang,
                                           lang2id=lang2id)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #  model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s in %s *****" % (prefix, lang))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
   
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            text = batch[0]
            mask = batch[1]
            segment = batch[2]
            label = batch[3]
            head_mask = mask.masked_fill(label == -1, 0).long()

            context = model(input_ids=text, attention_mask=mask, get_embedding=True)['last_hidden_state']
            predicted = model.actor.get_target_output(context, mask)
            # loss_qry_all = 0.0
            sentence_actor_attention = None
            for j in range(text.size()[0]):  # bachsize
                p = predicted[j]
                m = mask[j]
                h = head_mask[j]
                actions, Rinput, constituent_num = Sampling_RL(args, p, m, h, args.epsilon, Random=False)
                if j == 0:
                    sentence_actor_attention = Rinput
                else:
                    sentence_actor_attention = torch.cat([sentence_actor_attention, Rinput])
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                segment = batch[2] if args.model_type in ["bert", "xlnet"] else None

            outputs = model(input_ids=text, attention_mask=mask, token_type_ids=segment, labels=label,
                            action=sentence_actor_attention,output_hidden_states=args.output_hidden_states)
            tmp_eval_loss, logits = outputs[:2]
            #tmp_eval_loss=outputs['loss']
            #logits=outputs['logits']

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel evaluating
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = label.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label.detach().cpu().numpy(), axis=0)

    if nb_eval_steps == 0:
        results = {k: 0 for k in ["loss", "precision", "recall", "f1"]}
        preds_list = [[] for _ in range(out_label_ids.shape[0])]
    else:
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list)
        }

    if print_result:
        logger.info("***** Evaluation result %s in %s *****" % (prefix, lang))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


def build_support_features(base_features, target_features=None, support_size=2, cl_model=None):
    if target_features == None:
        target_features = base_features
    TOP_K = support_size
    # pdist = nn.PairwiseDistance(p=2)
    target_reprs = np.stack([numpy.array(item.representation) for item in target_features])
    base_reprs = np.stack([numpy.array(item.representation) for item in base_features])  # sample_num x feature_dim

    # compute pairwise cosine distance
    dis = np.matmul(target_reprs, base_reprs.T)  # target_num x base_num

    base_norm = np.linalg.norm(base_reprs, axis=1)  # base_num
    base_norm = np.stack([base_norm] * len(target_features), axis=0)  # target_num x base_num

    dis = dis / base_norm  # target_num x base_num
    relevance = np.argsort(dis, axis=1)
    if cl_model is not None:
        support_size = 64
    for i, item in tqdm(enumerate(target_features), desc='[get meta data]'):
        chosen_ids = relevance[i][-1 * (support_size + 1): -1]
        # logger.info('  Support set info: {}: {}'.format(i, ', '.join([str(id) for id in chosen_ids])))
        # support = [base_features[id] for id in chosen_ids]
        # support_set.append(support)
        if cl_model is not None:
            chosen_dict = {}
            target_features[i].meta_set_num = []
            target_representation = torch.tensor([target_features[i].input_rep], dtype=torch.float).repeat(support_size,
                                                                                                           1, 1).to(
                cl_model.device)
            target_attention = torch.tensor([target_features[i].attention], dtype=torch.float).repeat(support_size,
                                                                                                      1).to(
                cl_model.device)
            target_latent_representation = torch.matmul(target_representation, cl_model.probe.proj)

            chosen_representation = torch.tensor([base_features[id].input_rep for id in chosen_ids],
                                                 dtype=torch.float).to(cl_model.device)
            chosen_attention = torch.tensor([base_features[id].attention for id in chosen_ids], dtype=torch.float).to(
                cl_model.device)
            chosen_latent_representation = torch.matmul(chosen_representation, cl_model.probe.proj)
            score = cl_model.get_wmd(target_attention, chosen_attention, target_latent_representation,
                                     chosen_latent_representation).tolist()
            for i_, id in enumerate(chosen_ids):
                if id not in chosen_dict:
                    chosen_dict[id] = score[i_]
            chosen_dict = sorted(chosen_dict.items(), key=lambda x: x[1])
            for pos in chosen_dict[:TOP_K]:
                pos = int(pos[0])
                try:
                    target_features[i].rankings_spt.append(pos)
                except:
                    target_features[i].rankings_spt = [pos]
            target_features[i].rankings_spt = np.array(target_features[i].rankings_spt)
        else:
            target_features[i].rankings_spt = np.array(chosen_ids)
    return target_features


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode, lang, lang2id=None, few_shot=-1):
    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(mode, lang,
                                                                                   list(filter(None,
                                                                                               args.model_name_or_path.split(
                                                                                                   "/"))).pop(),
                                                                                   str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        langs = lang.split(',')
        logger.info("all languages = {}".format(lang))
        features = []
        for lg in langs:
            data_file = os.path.join(args.data_dir, lg, "{}".format(mode if lg == 'en' else 'test'))
            logger.info("Creating features from dataset file at {} in language {}".format(data_file, lg))
            examples = read_examples_from_file(data_file, lg, lang2id)
            features_lg = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer, args,
                                                       is_training=(mode == 'train'),
                                                       cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                       cls_token=tokenizer.cls_token,
                                                       cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                       sep_token=tokenizer.sep_token,
                                                       sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]),
                                                       pad_on_left=bool(args.model_type in ["xlnet"]),
                                                       pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[
                                                           0],
                                                       pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                       pad_token_label_id=pad_token_label_id,
                                                       lang=lg
                                                       )
            features.extend(features_lg)
        if args.local_rank in [-1, 0]:
            logger.info(
                "Saving features into cached file {}, len(features)={}".format(cached_features_file, len(features)))
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    if few_shot > 0 and mode == 'train':
        logger.info("Original no. of examples = {}".format(len(features)))
        features = features[: few_shot]
        logger.info('Using few-shot learning on {} examples'.format(len(features)))

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    if args.model_type == 'xlm' and features[0].langs is not None:
        all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
        logger.info('all_langs[0] = {}'.format(all_langs[0]))
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_langs)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def load_and_cache_examples_meta(args, tokenizer, labels, pad_token_label_id, mode, lang,
                                 lang2id=None, few_shot=-1, sbert=True, ner_model=None, cl_model=None):
    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(mode, 'en_task',
                                                                                   list(filter(None,
                                                                                               args.model_name_or_path.split(
                                                                                                   "/"))).pop(),
                                                                                   str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        spt_features = torch.load(cached_features_file)
    else:
        logger.info("all languages = {}".format(lang))
        spt_features = []
        data_file = os.path.join(args.data_dir, 'en', "train")
        logger.info("Creating features from dataset file at {} in language {}".format(data_file, 'en'))
        examples = read_examples_from_file(data_file, 'en', lang2id)
        spt_features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer, args,
                                                    is_training=(mode == 'train'),
                                                    cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]),
                                                    pad_on_left=bool(args.model_type in ["xlnet"]),
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[
                                                        0],
                                                    pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                    pad_token_label_id=pad_token_label_id,
                                                    lang='en',
                                                    sbert=True)

    langs = lang.split(',')
    logger.info("all languages = {}".format(lang))
    features = []
    for lg in langs:
        data_file = os.path.join(args.data_dir, lg, "{}".format(mode if lg == 'en' else 'dev'))
        logger.info("Creating features from dataset file at {} in language {}".format(data_file, lg))
        examples = read_examples_from_file(data_file, lg, lang2id)
        features_lg = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer, args,
                                                   is_training=(mode == 'train'),
                                                   cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                   cls_token=tokenizer.cls_token,
                                                   cls_token_segment_id=2 if args.model_type in [
                                                       "xlnet"] else 0,
                                                   sep_token=tokenizer.sep_token,
                                                   sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]),
                                                   pad_on_left=bool(args.model_type in ["xlnet"]),
                                                   pad_token=
                                                   tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[
                                                       0],
                                                   pad_token_segment_id=4 if args.model_type in [
                                                       "xlnet"] else 0,
                                                   pad_token_label_id=pad_token_label_id,
                                                   lang=lg,
                                                   spt_feature=spt_features,
                                                   sbert=sbert,
                                                   ner_model=ner_model,
                                                   cl_model=cl_model
                                                   )
        features.extend(features_lg)
        if args.local_rank in [-1, 0]:
            logger.info(
                "Saving features into cached file {}, len(features)={}".format(cached_features_file, len(features)))
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    if few_shot > 0 and mode == 'train':
        logger.info("Original no. of examples = {}".format(len(features)))
        features = features[: few_shot]
        logger.info('Using few-shot learning on {} examples'.format(len(features)))

    qry_features = build_support_features(spt_features, target_features=features, support_size=2, cl_model=None)
    l_input_ids_s = []
    l_attention_masks_s = []
    l_token_type_ids_s = []
    l_label_ids_s = []

    l_input_ids_q = []
    l_attention_masks_q = []
    l_token_type_ids_q = []
    l_label_ids_q = []

    random.shuffle(qry_features)
    # 出现频率过大的样本，限制在一定的概率出现
    dic_num = {}
    for index in range(len(qry_features)):
        spt = qry_features[index].rankings_spt[0]
        if spt not in dic_num.keys():
            dic_num[spt] = 1
        else:
            dic_num[spt] += 1
    dic_num = sorted(dic_num.items(), key=lambda d: d[1], reverse=True)
    print(dic_num)
    print(len(dic_num))

    s = 0
    print("META Batching")
    for _ in tqdm(range(args.batch_sz)):
        l_input_ids_spt = []
        l_attention_masks_spt = []
        l_token_type_ids_spt = []
        l_all_label_ids_spt = []

        l_input_ids_qry = []
        l_attention_masks_qry = []
        l_token_type_ids_qry = []
        l_all_label_ids_qry = []

        # Pick q_qry from qry examples randomly
        if s + args.q_qry < len(qry_features):
            qry_indices = range(s, s + args.q_qry)
            s = s + args.q_qry
        elif s < len(qry_features):
            t = s + args.q_qry - len(qry_features)
            qry_indices = list(range(s, len(qry_features))) + list(range(0, t))
            s = t
        else:
            s = 0
            qry_indices = range(s, len(qry_features))

        all_feature_index = torch.tensor(qry_indices, dtype=torch.long)
        # l_feature_index.append(all_feature_index)

        # random.sample(len(qry_examples), k=data_args.q_qry)

        for index in qry_indices:
            l_input_ids_qry.append(qry_features[index].input_ids)
            l_attention_masks_qry.append(qry_features[index].input_mask)
            l_token_type_ids_qry.append(qry_features[index].segment_ids)
            l_all_label_ids_qry.append(qry_features[index].label_ids)

            spt_indices = qry_features[index].rankings_spt[:args.meta_example_num]

            for spt_index in spt_indices:
                l_input_ids_spt.append(spt_features[spt_index].input_ids)
                l_attention_masks_spt.append(spt_features[spt_index].input_mask)
                l_token_type_ids_spt.append(spt_features[spt_index].segment_ids)
                l_all_label_ids_spt.append(spt_features[spt_index].label_ids)

        all_input_ids = torch.tensor(l_input_ids_spt, dtype=torch.long)
        l_input_ids_s.append(all_input_ids)
        l_attention_masks_s.append(torch.tensor(l_attention_masks_spt, dtype=torch.long))
        l_token_type_ids_s.append(torch.tensor(l_token_type_ids_spt, dtype=torch.long))
        l_label_ids_s.append(torch.tensor(l_all_label_ids_spt, dtype=torch.long))

        all_input_ids = torch.tensor(l_input_ids_qry, dtype=torch.long)
        l_input_ids_q.append(all_input_ids)
        l_attention_masks_q.append(torch.tensor(l_attention_masks_qry, dtype=torch.long))
        l_token_type_ids_q.append(torch.tensor(l_token_type_ids_qry, dtype=torch.long))
        l_label_ids_q.append(torch.tensor(l_all_label_ids_qry, dtype=torch.long))

    # Convert to Tensors and build dataset
    if args.model_type == 'xlm' and features[0].langs is not None:
        all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
        logger.info('all_langs[0] = {}'.format(all_langs[0]))
        dataset = TensorDataset(torch.stack(l_input_ids_s), torch.stack(l_attention_masks_s),
                                torch.stack(l_token_type_ids_s), torch.stack(l_label_ids_s),
                                torch.stack(l_input_ids_q), torch.stack(l_attention_masks_q),
                                torch.stack(l_token_type_ids_q), torch.stack(l_label_ids_q),
                                all_langs)
    else:
        dataset = TensorDataset(torch.stack(l_input_ids_s), torch.stack(l_attention_masks_s),
                                torch.stack(l_token_type_ids_s), torch.stack(l_label_ids_s),
                                torch.stack(l_input_ids_q), torch.stack(l_attention_masks_q),
                                torch.stack(l_token_type_ids_q), torch.stack(l_label_ids_q))
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='data/NER', type=str,
                        help="The input data dir. Should contain the training files for the NER/POS task.")
    parser.add_argument("--model_type", default='xlmr', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='/data/home10b/wlj/cached_models/xlm-roberta-large',
                        type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + "bert-base-multilingual-cased, xlm-roberta-large")
    parser.add_argument("--output_dir", default='output/NER', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--critic_output_dir", default="/data/home10b/wlj/code/Struct-XLM/src/Struct_XLM/output_new/checkpoint-best", type=str)
    parser.add_argument("--action", default=True)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--act_dropout", default=0.0, type=float)
    parser.add_argument("--margin", default=0.5, type=float)
    parser.add_argument("--s", default=1, type=float)
    parser.add_argument("--sentence_type", default='mean', type=str, help="mean, pool")
    parser.add_argument("--epsilon", type=float, default=0.05)

    ## Other parameters
    parser.add_argument("--labels", default="data/NER/label.txt", type=str,
                        help="Path to a file containing all labels. If not specified, NER/POS labels are used.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev.txt set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--do_predict_dev", action="store_true",
                        help="Whether to run predictions on the dev.txt set.")
    parser.add_argument("--output_hidden_states", default=True)
    parser.add_argument("--init_checkpoint",
                        default='', type=str,
                        help="initial checkpoint for train/predict")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--few_shot", default=-1, type=int,
                        help="num of few-shot exampes")

    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=8e-6, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0.05, type=float,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_only_best_checkpoint", action="store_true",
                        help="Save only the best checkpoint during training")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", default=True,
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--predict_langs", type=str,
                        default="en, af, ar, bg, bn, de, el, es, et, eu, fa, fi, fr, he, hi, hu, id, "
                                "it, ja, jv, ka, kk, ko, ml, mr, ms, my, nl, pt, ru, sw, ta, te, th, tl, tr, ur, vi, yo, zh",
                        help="prediction languages")
    parser.add_argument("--train_langs", default="en", type=str,
                        help="The languages in the training sets.")
    parser.add_argument("--log_file", type=str, default='NERlog2.txt', help="log file")
    parser.add_argument("--eval_patience", type=int, default=-1,
                        help="wait N times of decreasing dev.txt score before early stop during training")
    args = parser.parse_args()

    args.edim = emb_config[args.model_type]
    args.ml_model_path = args.model_name_or_path
    args.max_len = args.max_seq_length
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which sychronizes nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logging.info("Input args: %r" % args)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare NER/POS task
    labels = get_labels(args.labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id
    # so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class, base_model = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    if args.init_checkpoint:
        logger.info("loading from init_checkpoint={}".format(args.init_checkpoint))
        model = model_class.from_pretrained(args.init_checkpoint,
                                            args=args,
                                            config_class=config_class,
                                            model_class=base_model,
                                            config=config,
                                            cache_dir=args.init_checkpoint)
    else:
        logger.info("loading from cached model = {}".format(args.model_name_or_path))
        model = model_class.from_pretrained(args.model_name_or_path,
                                            args=args,
                                            config_class=config_class,
                                            model_class=base_model,
                                            from_tf=bool(".ckpt" in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    lang2id = config.lang2id if args.model_type == "xlm" else None
    logger.info("Using lang2id = {}".format(lang2id))

    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank == 0:
        torch.distributed.barrier()
    model.to(args.device)
    #output_dir = args.critic_output_dir
    output_dir = os.path.join(args.critic_output_dir, "checkpoint-best")
    #定义要使用的模型路径
    actor_path = os.path.join(output_dir, "target_model_actor.bin")
    critic_path = os.path.join(output_dir, "target_model_critic.bin")
    reward_path = os.path.join(output_dir, "target_model_reward.bin")

    #actorModel = Actor(expirement=args, d_model=args.edim, dropout=args.act_dropout, no_cuda=args.no_cuda)
    #actorModel.policy_network.load_state_dict(torch.load(os.path.join(output_dir, "target_model_actor.bin")))
    #model.encoder.load_state_dict(torch.load(os.path.join(output_dir, "target_model_reward.bin")))
    #actorModel.to(args.device)
    #陆续加载模型
    #加载actor模
    if os.path.exists(actor_path):
        logger.info(f"Loading Actor model from {actor_path}")
        model.actor.policy_network.load_state_dict(torch.load(actor_path,map_location=args.device))
    else:
        logger.error(f"Actor model file not found: {actor_path}")
    #model.actor.to(args.device)

    #加载Critic模型
    if os.path.exists(critic_path):
        logger.info(f"Loading Critic model from {critic_path}")
        model.critic.load_state_dict(torch.load(critic_path, map_location=args.device))
    else:
        logger.error(f"Critic model file not found: {critic_path}")

    #model.critic.to(args.device)

    # **加载 Reward 计算模块**
    if os.path.exists(reward_path):
        logger.info(f"Loading Reward model from {reward_path}")
        model.encoder.load_state_dict(torch.load(reward_path, map_location=args.device))
    else:
        logger.error(f"Reward model file not found: {reward_path}")

    log_path = os.path.join(args.output_dir, "POS_log0702.txt")#检查模型是否更新的日志文件
    actor_before = get_model_parameters(model.actor)#获取模型训练之前的参数
    critic_before = get_model_parameters(model.critic)
    main_before = get_model_parameters(model.encoder)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train",
                                                lang=args.train_langs, lang2id=lang2id, few_shot=args.few_shot)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id,
                                     lang2id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    actor_after = get_model_parameters(model.actor)#获取模型训练之后的参数
    critic_after = get_model_parameters(model.critic)
    main_after = get_model_parameters(model.encoder)
    
    #对比模型训练前后的变化，然后及那个前后参数的变化 写入文件
    check_model_updated_and_log(actor_before, actor_after, log_path, model_name="ActorModel")
    check_model_updated_and_log(critic_before, critic_after, log_path, model_name="CriticModel")
    check_model_updated_and_log(main_before, main_after, log_path, model_name="MainModel")

    # Saving best-practices: if you use default names for the model,
    # you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # Save model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Initialization for evaluation
    results = {}
    """if os.path.exists(os.path.join(args.output_dir,
                                   'checkpoint-final' + ("" if args.train_langs == 'en' else f"_{args.train_langs}"))):
        best_checkpoint = os.path.join(args.output_dir, 'checkpoint-final' + (
            "" if args.train_langs == 'en' else f"_{args.train_langs}"))
    el"""
    if os.path.exists(os.path.join(args.output_dir,
                                     'checkpoint-best' + (
                                     "" if args.train_langs == 'en' else f"_{args.train_langs}"))):
        best_checkpoint = os.path.join(args.output_dir,
                                       'checkpoint-best' + (
                                           "" if args.train_langs == 'en' else f"_{args.train_langs}"))
    else:
        best_checkpoint = args.output_dir
    best_f1 = 0

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint,
                                                args=args,
                                                config_class=config_class,
                                                model_class=base_model)
            model.to(args.device)
            result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev",
                                 prefix=global_step,
                                 lang=args.train_langs, lang2id=lang2id)
            if result["f1"] > best_f1:
                best_checkpoint = checkpoint
                best_f1 = result["f1"]
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))
            writer.write("best checkpoint = {}, best f1 = {}\n".format(best_checkpoint, best_f1))

    # Prediction
    all_f1 = ""
    result_file = open("src/xtreme_7/output/ner_results.txt", "a+", encoding='utf-8')
    if args.do_predict and args.local_rank in [-1, 0]:
        logger.info("Loading the best checkpoint from {}\n".format(best_checkpoint))
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(best_checkpoint,
                                            args=args,
                                            config_class=config_class,
                                            model_class=base_model)
        model.to(args.device)

        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        AVG_F1 = 0
        I = 0
        with open(output_test_results_file, "a") as result_writer:
            for lang in args.predict_langs.split(', '):
                if not os.path.exists(os.path.join(args.data_dir, lang, 'test')):
                    logger.info("Language {} does not exist".format(lang))
                    continue
                result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id,
                                               mode="test",
                                               lang=lang, lang2id=lang2id)

                # Save results
                result_writer.write("=====================\nlanguage={}\n".format(lang))
                for key in sorted(result.keys()):
                    result_writer.write("{} = {}\n".format(key, str(result[key])))
                all_f1 += "%.1f " % (result['f1'] * 100)
                AVG_F1 += result['f1']
                I += 1
            all_f1 += "%.1f " % (AVG_F1 / I * 100)
            # Save predictions
            # output_test_predictions_file = os.path.join(args.output_dir, "test_{}_predictions.txt".format(lang))
            # infile = os.path.join(args.data_dir, lang, "test")
            # idxfile = infile + '.idx'
            # save_predictions(args, predictions, output_test_predictions_file, infile, idxfile)
            print(all_f1, file=result_file)
        if os.path.exists(os.path.join(args.output_dir,
                                       'checkpoint-final' + (
                                       "" if args.train_langs == 'en' else f"_{args.train_langs}"))):
            best_checkpoint = os.path.join(args.output_dir, 'checkpoint-final' + (
                "" if args.train_langs == 'en' else f"_{args.train_langs}"))
            model = model_class.from_pretrained(best_checkpoint,
                                                args=args,
                                                config_class=config_class,
                                                model_class=base_model)
            model.to(args.device)
            all_f1 = ""
            output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
            AVG_F1 = 0
            I = 0
            with open(output_test_results_file, "a") as result_writer:
                for lang in args.predict_langs.split(', '):
                    if not os.path.exists(os.path.join(args.data_dir, lang, 'test')):
                        logger.info("Language {} does not exist".format(lang))
                        continue
                    result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id,
                                                   mode="test",
                                                   lang=lang, lang2id=lang2id)

                    # Save results
                    result_writer.write("=====================\nlanguage={}\n".format(lang))
                    for key in sorted(result.keys()):
                        result_writer.write("{} = {}\n".format(key, str(result[key])))
                    all_f1 += "%.1f " % (result['f1'] * 100)
                    AVG_F1 += result['f1']
                    I += 1
                all_f1 += "%.1f " % (AVG_F1 / I * 100)
                # Save predictions
                # output_test_predictions_file = os.path.join(args.output_dir, "test_{}_predictions.txt".format(lang))
                # infile = os.path.join(args.data_dir, lang, "test")
                # idxfile = infile + '.idx'
                # save_predictions(args, predictions, output_test_predictions_file, infile, idxfile)
                print(all_f1, file=result_file)

def save_predictions(args, predictions, output_file, text_file, idx_file, output_word_prediction=False):
    # Save predictions
    with open(text_file, "r") as text_reader, open(idx_file, "r") as idx_reader:
        text = text_reader.readlines()
        index = idx_reader.readlines()
        assert len(text) == len(index)

    # Sanity check on the predictions
    with open(output_file, "w") as writer:
        example_id = 0
        prev_id = int(index[0])
        for line, idx in zip(text, index):
            if line == "" or line == "\n":
                example_id += 1
            else:
                cur_id = int(idx)
                output_line = '\n' if cur_id != prev_id else ''
                if output_word_prediction:
                    output_line += line.split()[0] + '\t'
                output_line += predictions[example_id].pop(0) + '\n'
                writer.write(output_line)
                prev_id = cur_id


if __name__ == "__main__":
    main()

    # --do_train --do_eval
