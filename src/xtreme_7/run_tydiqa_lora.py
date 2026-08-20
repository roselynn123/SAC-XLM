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
"""Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""

import argparse
import glob
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random
import timeit
import time

from evaluate_mlqa import mlqa_evaluate
from peft import LoraConfig, get_peft_model,PeftModel
from peft.utils import get_peft_model_state_dict
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    #AdamW,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    get_linear_schedule_with_warmup,
    XLMRobertaTokenizer
)
from torch.optim import AdamW
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)

from processors.squad import (
    SquadResult,
    SquadV1Processor,
    SquadV2Processor,
    squad_convert_examples_to_features
)
from RobertaForQA import RobertaForQuestionAnswering

import sys
sys.path.append('/workspace/SAC-XLM/src/Struct_XLM/')
from BertEncoder import BertModel
from RobertaEncoder import RobertaModel
from actor import Actor
from critic import Critic
from xlm_roberta import XLMRobertaConfig

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer, BertModel),
    "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer, RobertaModel),
    "xlmr": (XLMRobertaConfig, RobertaForQuestionAnswering, XLMRobertaTokenizer, RobertaModel)
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


def to_list(tensor):
    return tensor.detach().cpu().tolist()

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


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    #actorModel.eval()
    model.to(args.device).train()

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
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(t_total * args.warmup_steps), num_training_steps=t_total
        )
        return optimizer, scheduler
    main_optimizer, main_schedule = optimizer_schedule(model)#主模型的

    # Check if saved optimizer or scheduler states exist
    '''
    if not args.init_model and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

        # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.init_model, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.init_model, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.init_model, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.init_model, "scheduler.pt")))
    '''
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
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
   
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    gpu_epoch_times = []  # 记录每个 epoch 的 GPU 时间
    patience = 0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    best_score = 0
    #for _ in train_iterator:
    for epoch_idx, _ in enumerate(train_iterator):
        # 开始记录当前epoch的GPU时间
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": None if args.model_type in ["xlm", "xlmr", "distilbert"] else batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }
            head_mask = batch[1].masked_fill(batch[2] == 0, 0).long()

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
            if args.model_type == "xlm":
                inputs["langs"] = batch[7]

           
            context = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], get_embedding=True)[
                'last_hidden_state']
            #predicted = actorModel.get_target_output(context, inputs["attention_mask"])
            predicted,context_emb = model.actor.policy_network(context, inputs["attention_mask"])
            # loss_qry_all = 0.0
            sentence_actor_attention = None
            constituent_num_total = 0
            actions,sentence_actor_attention,_ =Sampling_RL_batch(args,predicted,inputs["attention_mask"],head_mask,args.epsilon, Random=False)
            
            constituent_num_total /= args.train_batch_size
            inputs.update({"action": sentence_actor_attention})
            

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                main_optimizer.step()
                main_schedule.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % 100 == 0:
                    print(f"Step {global_step}:")
                    print(f"  output1 Loss: {loss.item():.4f}")
                    
                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model,tokenizer)
                        logger.info(f"Step {global_step} - Evaluation results: {results}")
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", main_schedule.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    print(f"Step {step}, Loss: {loss.item()}")
                    if args.save_only_best_checkpoint:
                        results = evaluate(args, model, tokenizer)
                        logger.info(f"Step {global_step} - Checkpoint evaluation: {results}")
                        if results['f1'] > best_score:
                            for key, value in results.items():
                                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                            logger.info(" result['acc']={} > best_score={}".format(results['f1'], best_score))
                            output_dir = os.path.join(args.output_dir, "checkpoint-best")
                            best_checkpoint = output_dir
                            best_score = results['f1']
                
                            # Save model checkpoint
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, "module") else model
                            model_to_save.save_pretrained(output_dir)
                            

                            tokenizer.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)

                            torch.save(main_optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(main_schedule.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                           
                            logger.info("Reset patience to 0")
                            patience = 0
                        else:
                            patience += 1
                            logger.info("Hit patience={}".format(patience))
                            if args.eval_patience > 0 and patience > args.eval_patience:
                                logger.info("early stop! patience={}".format(patience))
                                # 记录当前epoch已用时间（即使未完成整个epoch）
                                end_event.record()
                                torch.cuda.synchronize()
                                partial_epoch_time_ms = start_event.elapsed_time(end_event)
                                partial_epoch_time_sec = partial_epoch_time_ms / 1000.0
                                # 记录部分epoch时间（标记为不完整）
                                gpu_epoch_times.append(f"Epoch {epoch_idx+1} (partial): {partial_epoch_time_sec:.2f}s")
                                # 强制写入时间日志
                                write_time_log(args.output_dir, gpu_epoch_times)
                                #在早停训练最终阶段保存最后的模型
                                # 1. 检查模型是否为 LoRA 模型
                                if hasattr(model_to_save, 'merge_and_unload'):
                                    # 2. 调用 merge_and_unload() 方法来合并 LoRA 权重
                                    # 这会返回一个新的、完整的、可用于推理的模型
                                    logger.info("Merging LoRA weights and saving the full model...")
                                    full_model_to_save = model_to_save.merge_and_unload()
                                    full_model_to_save.save_pretrained(output_dir)
                                    logger.info("Successfully saved the merged model to %s", output_dir)
                                else:
                                    # 如果模型不是 LoRA 模型，则按原样保存
                                    logger.info("Saving the model without merging LoRA weights (not a PEFT model).")
                                    model_to_save.save_pretrained(output_dir)
                                # 清理并返回
                                epoch_iterator.close()
                                train_iterator.close()
                                if args.local_rank in [-1, 0]:
                                    tb_writer.close()
                                return global_step, tr_loss / global_step
                    else:
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(main_optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(main_schedule.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            
        end_event.record()
        torch.cuda.synchronize()
        epoch_time_ms = start_event.elapsed_time(end_event)
        epoch_time_sec = epoch_time_ms / 1000.0
        #gpu_epoch_times.append(epoch_time_sec)
        gpu_epoch_times.append(f"Epoch {epoch_idx+1}: {epoch_time_sec:.2f}s")
        logger.info(f"Epoch {epoch_idx} GPU time: {epoch_time_sec:.2f} seconds")

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break   
    '''
    
    #-----------性能最大化------------------
    output_dir = os.path.join(args.output_dir,
                                "checkpoint-final")
    # best_checkpoint = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    #model_to_save.save_pretrained(output_dir)
    # 1. 检查模型是否为 LoRA 模型
    if hasattr(model_to_save, 'merge_and_unload'):
        # 2. 调用 merge_and_unload() 方法来合并 LoRA 权重
        # 这会返回一个新的、完整的、可用于推理的模型
        logger.info("Merging LoRA weights and saving the full model...")
        full_model_to_save = model_to_save.merge_and_unload()
        full_model_to_save.save_pretrained(output_dir)
        logger.info("Successfully saved the merged model to %s", output_dir)
    else:
        # 如果模型不是 LoRA 模型，则按原样保存
        logger.info("Saving the model without merging LoRA weights (not a PEFT model).")
        model_to_save.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)
    '''
    #---------------性能最大化--------------------------------
    write_time_log(args.output_dir, gpu_epoch_times)


    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

# 新增：独立的时间日志写入函数，确保早停和正常结束时都能调用
def write_time_log(output_dir, epoch_times):
    gpu_log_file = os.path.join(output_dir, "TyDiQA_gpu_time_log.txt")
    with open(gpu_log_file, "a+") as f:  # 使用"w"模式避免重复追加
        for time_record in epoch_times:
            f.write(f"{time_record}\n")
        # 计算总时间（仅统计完整epoch）
        total_seconds = 0.0
        for record in epoch_times:
            sec = float(record.split(": ")[1].replace("s", ""))
            total_seconds += sec
        f.write(f"\nTotal GPU training time (complete epochs): {total_seconds:.2f} seconds\n")
        # 计算平均时间（仅完整epoch）
        complete_epochs = [r for r in epoch_times if "(partial)" not in r]
        if complete_epochs:
            avg_seconds = total_seconds / len(complete_epochs)
            f.write(f"Average per complete epoch: {avg_seconds:.2f} seconds\n")
    logger.info(f"GPU时间日志已写入：{gpu_log_file}")

def evaluate(args, model, tokenizer, prefix="", language='en', lang2id=None, dataset_name='squad'):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True,
                                                          language=language, lang2id=lang2id)
    
    model.eval()
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": None if args.model_type in ["xlm", "distilbert", "xlmr"] else batch[2],
            }
            example_indices = batch[3]
            head_mask = batch[1].masked_fill(batch[2] == 0, 0).long()

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
            if args.model_type == "xlm":
                inputs["langs"] = batch[6]
            context = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], get_embedding=True)[
                'last_hidden_state']

            #predicted = model.actor.get_target_output(context, inputs["attention_mask"])
            predicted,context_emb = model.actor.policy_network(context, inputs["attention_mask"])
            # loss_qry_all = 0.0
            sentence_actor_attention = None
            if args.action:
                
                constituent_num_total = 0
                for j in range(inputs["input_ids"].size()[0]):  # bachsize
                    p = predicted[j]
                    m = inputs["attention_mask"][j]
                    h = head_mask[j]
                    actions, Rinput, constituent_num = Sampling_RL(args, p, m, h, args.epsilon, Random=False)
                    constituent_num_total += constituent_num
                    if j == 0:
                        sentence_actor_attention = Rinput
                    else:
                        sentence_actor_attention = torch.cat([sentence_actor_attention, Rinput])
                #actions,sentence_actor_attention,_ =Sampling_RL_batch(args,predicted,inputs["attention_mask"],head_mask,args.epsilon, Random=False)
            
            # constituent_num_total /= args.train_batch_size
            inputs.update({"action": sentence_actor_attention})
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}_{}.json".format(language, prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}_{}.json".format(language, prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )
    if dataset_name == 'squad':
        # Compute the F1 and exact scores.
        results = squad_evaluate(examples, predictions)
    else:
        results = mlqa_evaluate(examples, predictions, language)
    return results



def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False,
                            language='en', lang2id=None):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}".format(
            os.path.basename(args.predict_file) if evaluate else os.path.basename(args.train_file),
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(language)
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        if not output_examples:
            features, dataset = features_and_dataset["features"], features_and_dataset["dataset"]
        else:
            features, dataset, examples = features_and_dataset["features"], features_and_dataset["dataset"], features_and_dataset["examples"]
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate, language=language)
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file, language=language)
            else:
                examples = processor.get_train_examples(args.data_dir, filename=args.train_file, language=language)


        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
            lang2id=lang2id
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            if not evaluate:
                torch.save({"features": features, "dataset": dataset}, cached_features_file)
            else:
                torch.save({"examples": examples, "features": features, "dataset": dataset}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default='xlmr',
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default='/data/home10b/wlj/cached_models/xlm-roberta-large',
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--output_dir",
        default='output/tydiqa/',
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument("--init_model", default="")

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default='data/MRC/',
        type=str,
        help="The input data dir. Should contain the .json files for the task."
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default='tydiqa/tydiqa.en.train.json',
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default='tydiqa/tydiqa.en.dev.json',
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument("--test_file", default="tydiqa", type=str)
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
             "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Ncumber of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0.0, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
             "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
             "A number of warnings are expected for a normal SQuAD evaluation.",
    )

    parser.add_argument("--logging_steps", type=int, default=25, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=25, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

    parser.add_argument("--train_lang", type=str, default="en", help="The language of the training data")
    parser.add_argument("--eval_lang", type=str, default="en", help="The language of the test data")

    parser.add_argument("--save_only_best_checkpoint", action="store_true",)
    parser.add_argument("--critic_output_dir", default="/data/home10b/wlj/code/Struct-XLM/src/Struct_XLM/output_new/checkpoint-best", type=str)
    parser.add_argument("--action", default=True)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--act_dropout", default=0.0, type=float)
    parser.add_argument("--margin", default=0.5, type=float)
    parser.add_argument("--s", default=1, type=float)
    parser.add_argument("--sentence_type", default='mean', type=str, help="mean, pool")
    parser.add_argument("--epsilon", type=float, default=0.05)#原本的--epsilon=0.05
    parser.add_argument("--eval_patience", type=int, default=5,
                        help="wait N times of decreasing dev.txt score before early stop during training")
    parser.add_argument("--log_file", type=str, default='TyDiQAlog.txt', help="log file")
    args = parser.parse_args()

    args.edim = emb_config[args.model_type]
    args.ml_model_path = args.model_name_or_path
    args.max_len = args.max_seq_length

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class, base_model = MODEL_CLASSES[args.model_type]
    
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # Set using of language embedding to True
    if args.model_type == "xlm":
        config.use_lang_emb = True
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.init_model:
        model = model_class.from_pretrained(
            args.init_model,
            args=args,
            config_class=config_class,
            model_class=base_model,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    else:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            args=args,
            config_class=config_class,
            model_class=base_model,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    # **加载 Reward 计算模块**,在使用lora之前加载model的模型
    output_dir = os.path.join(args.critic_output_dir, "checkpoint-best")
    reward_path = os.path.join(output_dir, "target_model_reward.bin")
    actor_path = os.path.join(output_dir, "target_model_actor.bin")
    if os.path.exists(reward_path):
        logger.info(f"Loading Reward model from {reward_path}")
        model.encoder.load_state_dict(torch.load(reward_path, map_location=args.device),strict=True)
        logger.info("Successfully loaded reward model with strict=True")
    else:
        logger.error(f"Reward model file not found: {reward_path}")
    #加载actor模
    if os.path.exists(actor_path):
        logger.info(f"Loading Actor model from {actor_path}")
        model.actor.policy_network.load_state_dict(torch.load(actor_path,map_location=args.device),strict=True)
        logger.info("Successfully loaded reward model with strict=True")
    else:
        logger.error(f"Actor model file not found: {actor_path}")

    lora_config =LoraConfig(
        r=4,#越大性能越好，但是速度也会随之减慢
        lora_alpha=8,#缩放因子，会是r的4倍
        #target_modules=["query","value","dense","qa_outputs","actor.policy_network.linear_key","actor.policy_network.linear_query"],
        target_modules=["query","value","key","dense",
        #"actor.policy_network.linear_key",
        #"actor.policy_network.linear_query",
        #"actor.policy_network.linear_value",
        #"actor_linear",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="QUESTION_ANS"
    )
    lang2id = config.lang2id if args.model_type == "xlm" else None
    logger.info("lang2id = {}".format(lang2id))
    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
    #定义要使用的模型路径
    critic_path = os.path.join(output_dir, "target_model_critic.bin")
    #加载Critic模型
    if os.path.exists(critic_path):
        logger.info(f"Loading Critic model from {critic_path}")
        model.critic.load_state_dict(torch.load(critic_path, map_location=args.device))
    else:
        logger.error(f"Critic model file not found: {critic_path}")

    model = get_peft_model(model, lora_config)
    model.to(args.device)
    
    # 冻结 embeddings
    #for param in model.base_model.model.encoder.encoder.embeddings.parameters():
    #    param.requires_grad = False
    
    # 冻结前6层 encoder
    '''
    # 冻结前4层 encoder（索引0-3）
    for i in range(4):
        for param in model.base_model.model.encoder.encoder.encoder.layer[i].parameters():
            param.requires_grad = False

    # 冻结最后2层 encoder（总层数24层，最后两层索引为22、23）
    total_layers = 24  # 明确总层数
    for i in range(total_layers - 2, total_layers):  # 24-2=22，即范围22-23
        for param in model.base_model.model.encoder.encoder.encoder.layer[i].parameters():
            param.requires_grad = False
    '''
    for name, param in model.named_parameters():
    # 如果参数名包含 lora、actor、critic 或 qa_outputs，则设为可训练
        if ('lora' in name.lower() or
            #'actor' in name.lower() or 
            #'critic' in name.lower() or 
            'qa_outputs' in name.lower()):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    model.print_trainable_parameters()
    logger.info("Training/evaluation parameters %s", args)
    # 定义总参数量、可训练参数量及非可训练参数量变量
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    logger.info(f"Total params: {total_params}")
    logger.info(f"Trainable params: {trainable_params}")
    logger.info(f"Non-trainable params: {non_trainable_params}")
    '''
    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    #for param in model.actor.target_policy.parameters():
    for param in model.actor.policy_network.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量'''

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False,
                                                language=args.train_lang, lang2id=lang2id)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        #model_to_save.save_pretrained(args.output_dir)
        # 检查模型是否为 LoRA 模型，如果是，则合并权重并保存
        if hasattr(model_to_save, 'merge_and_unload'):
            logger.info("Merging LoRA weights and saving the full model at the end of training...")
            # 调用 merge_and_unload() 方法来合并 LoRA 权重
            full_model_to_save = model_to_save.merge_and_unload()
            full_model_to_save.save_pretrained(args.output_dir)
            logger.info("Successfully saved the merged model to %s", args.output_dir)
        else:
            # 如果模型不是 LoRA 模型，则按原样保存
            logger.info("Saving the model without merging LoRA weights (not a PEFT model)...")
            model_to_save.save_pretrained(args.output_dir)

        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir, force_download=True,
                                            args=args,
                                            config_class=config_class,
                                            model_class=base_model, )
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            base_model = model_class.from_pretrained(args.model_name_or_path,args=args,
                                                config_class=config_class,
                                                model_class=base_model,)
            #lora_adapter_path =  os.path.join(checkpoint, "adapter_model.bin")
            lora_adapter_path =  checkpoint
            from peft import PeftModel
            # 将 LoRA 适配器附加到基础模型上
            model = PeftModel.from_pretrained(base_model, lora_adapter_path)
            
            # 3. 加载 tokenizer
            # Tokenizer 在 LoRA 微调中通常不需要改变
            tokenizer = tokenizer_class.from_pretrained(lora_adapter_path, do_lower_case=args.do_lower_case)
            
            # 将模型发送到 GPU
            model.to(args.device)
            
            # 将模型设置为评估模式
            model.eval()
            # Evaluate
            result = evaluate(args, model,tokenizer, prefix=global_step, language=args.eval_lang,
                              lang2id=lang2id)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    if args.do_predict and args.local_rank in [-1, 0]:
        if os.path.exists(os.path.join(args.output_dir, 'checkpoint-final')):#原本是checkpoint-best
            best_checkpoint = os.path.join(args.output_dir, 'checkpoint-final')
        else:
            best_checkpoint = args.output_dir
        
        mid_model = model_class.from_pretrained(args.output_dir,#原本是args.output_dir
                                            force_download=True,
                                            args=args,
                                            config_class=config_class,
                                            model_class=base_model, )
        '''
        mid_model = model_class.from_pretrained(args.model_name_or_path,args=args,
                                                config_class=config_class,
                                                model_class=base_model,)'''
        #lora_adapter_path =  os.path.join(best_checkpoint, "adapter_model.bin")
        #lora_adapter_path =  best_checkpoint
        
        #from peft import PeftModel
        # 将 LoRA 适配器附加到基础模型上
        #model = PeftModel.from_pretrained(mid_model, best_checkpoint)
            
        # 3. 加载 tokenizer
        # Tokenizer 在 LoRA 微调中通常不需要改变
        #tokenizer = tokenizer_class.from_pretrained(best_checkpoint, do_lower_case=args.do_lower_case)
        model=mid_model   
        # 将模型发送到 GPU
        model.to(args.device)
            
        # 将模型设置为评估模式
        model.eval()

        # Evaluate
        if "mlqa" in args.test_file:
            result_str = ""
            result_file = open("src/xtreme_7/output/mlqa_results.txt", "a+", encoding='utf-8')
            f1_all = 0
            em_all = 0
            for lang in ["ar", "de", "en", "es", "hi", "vi", "zh"]:
                args.predict_file = u"data/mlqa/MLQA_V1/test/test-context-{}-question-{}.json".format(lang, lang)
                result = evaluate(args, model, tokenizer, prefix=u"mlqa {}".format(lang), language=lang,
                                  lang2id=lang2id, dataset_name="mlqa")
                result = dict((k, v) for k, v in result.items())
                result_str += "{%.1f}" % result['exact'] + "/{%.1f}    " % result['f1']
                f1_all += result['f1']
                em_all += result['exact']
            result_str += "{%.1f}" % (em_all/7) + "/{%.1f}    " % (f1_all/7)
            print(result_str, file=result_file)

        if "xquad" in args.test_file:
            result_str = ""
            result_file = open("src/xtreme_7/output/xquad_results.txt", "a+", encoding='utf-8')
            f1_all = 0
            em_all = 0
            for lang in ["ar", "de", "el", "en", "es", "hi", "ru", "th", "tr", "vi", "zh"]:
                args.predict_file = os.path.join(args.test_file, u"data/xquad/xquad.{}.json".format(lang, lang))
                result = evaluate(args, model, tokenizer, prefix=u"mlqa {}".format(lang), language=lang,
                                  lang2id=lang2id)
                result = dict((k, v) for k, v in result.items())
                result_str += "{%.1f}" % result['exact'] + "/{%.1f}    " % result['f1']
                f1_all += result['f1']
                em_all += result['exact']
            result_str += "{%.1f}" % (em_all/11) + "/{%.1f}    " % (f1_all/11)
            print(result_str, file=result_file)

        if "tydiqa" in args.test_file:
            result_str = ""
            result_file = open("src/xtreme_7/output/tydiqa_results.txt", "a+", encoding='utf-8')
            f1_all = 0
            em_all = 0
            for lang in ["english", "arabic", "bengali", "finnish", "indonesian", "korean", "russian", "swahili", "telugu"]:
                args.predict_file = u"tydiqa/tydiqa-goldp-v1.1-dev/tydiqa-goldp-dev-{}.json".format(lang, lang)

                result = evaluate(args, model, tokenizer, prefix=u"mlqa {}".format(lang), language=lang,
                                  lang2id=lang2id)
                result = dict((k, v) for k, v in result.items())
                print(result)
                result_str += "%.1f" % result['f1'] + "/%.1f    " % result['exact']
                f1_all += result['f1']
                em_all += result['exact']
            result_str += "%.1f" % (f1_all/9) + "/%.1f    " % (em_all/9)
            print(result_str, file=result_file)


            #第二轮预测，由于保存方式的改变，第二次预测与第一次预测是完全一样的内容
            '''
            best_checkpoint = args.output_dir
            model = model_class.from_pretrained(best_checkpoint, force_download=True,
                                                args=args,
                                                config_class=config_class,
                                                model_class=base_model, )
            
            model.to(args.device)
            print("完全没有改变时的结果\n")
            result_str = ""
            f1_all = 0
            em_all = 0
            for lang in ["english", "arabic", "bengali", "finnish", "indonesian", "korean", "russian", "swahili",
                         "telugu"]:
                args.predict_file = u"tydiqa/tydiqa-goldp-v1.1-dev/tydiqa-goldp-dev-{}.json".format(lang, lang)
                result = evaluate(args, model, tokenizer, prefix=u"mlqa {}".format(lang), language=lang,
                                  lang2id=lang2id)
                result = dict((k, v) for k, v in result.items())
                print(result)
                result_str += "%.1f" % result['f1'] + "/%.1f    " % result['exact']
                f1_all += result['f1']
                em_all += result['exact']
            result_str += "%.1f" % (f1_all / 9) + "/%.1f    " % (em_all / 9)
            print(result_str, file=result_file)'''

    return results


if __name__ == "__main__":
    main()
