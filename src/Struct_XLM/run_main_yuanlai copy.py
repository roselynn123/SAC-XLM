import argparse
import os
import pickle
import random
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import logging
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange, tqdm
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import time
from actor import Actor
from data_utils import read_parallel_txt, DataProcessor
from model import Reward #reward
from critic import Critic #critic

from transformers import BertConfig, XLMRobertaConfig

from BertEncoder import BertModel
from RobertaEncoder import RobertaModel
from torch.nn.utils import parameters_to_vector
import time
import torch.nn.functional as F
MODEL_CLASSES = {
    "mbert": (BertConfig, BertModel),
    "xlmr": (XLMRobertaConfig, RobertaModel),
    "infoxlm": (XLMRobertaConfig, RobertaModel),
}

logger = logging.getLogger(__name__)

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


def config():
    parser = argparse.ArgumentParser()
    # data config
    parser.add_argument("--data_dir", default='/data/home10b/wlj/code/Struct-XLM/data/UD_data/', type=str)
    parser.add_argument("--output_dir", default="/data/home10b/wlj/code/Struct-XLM/src/Struct_XLM/output_new/")
    parser.add_argument("--max_len", default=256, type=int, help="the max number of token in a sentence")
    parser.add_argument("--margin", default=0.5, type=float)
    parser.add_argument("--s", default=1, type=float)
    parser.add_argument("--sentence_type", default='pool_mean', type=str, help="pool_mean, mean, pool")
    parser.add_argument("--act_dropout", default=0.2, type=float)
    parser.add_argument("--samplecnt", default=5, type=int)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--grained", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--save_step", default=50)
    parser.add_argument("--warmup_steps", default=200, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=4.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev.txt set.")
    parser.add_argument("--warm_start", action="store_true",
                        help="Whether to run eval on the dev.txt set.")
    parser.add_argument("--LMpretrain", action="store_true",
                        help="Whether to run eval on the dev.txt set.")
    parser.add_argument("--RLpretrain", action="store_true",
                        help="Whether to run eval on the dev.txt set.")
    parser.add_argument("--warm_by_action", default=True)
    parser.add_argument("--do_lower_case", default=False,
                        help="Set this flag if you are using an uncased model.")

    # pretrained model config
    parser.add_argument('--ml_type', type=str, default="xlmr",
                        help='pretrained language model type:mbert, xlmr, infoxlm')
    parser.add_argument('--ml_model_path', type=str, default="/data/home10b/wlj/cached_models/xlm-roberta-large",
                        help='pretrained language model file: bert-base-multilingual-cased, xlm-roberta-large, infoxlm-large')

    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--log_file", default="./log0728.txt")

    args = parser.parse_args()
    return args

#用于序列标注任务，应该是确定边界与否，如果是词组词尾则为1，如果是词组中间则为0并且继续往下探索
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
            action = np.argmax(predicted[pos]).item()
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
        Rinput = actor_attention.masked_fill(actor_attention == 0, 0)

    Rinput = Rinput.unsqueeze(0).unsqueeze(0).cuda()  # (1, seq_len, seq_len)
    return actions, Rinput, constituent_num/length_true if constituent_num!=0 else 1
    #第一次理解：actions=》一个列表，表示每个位置所选择的动作（0或者是1）；Rinput=》是一个张量，句子中词组的序列关系；constituent_num/length_true=》所选择的构成部分（标记为1的位置）的数量与有效位置（head_mask为1的位置）的总数之间的比率
    #第二次理解：actions=》一个句子中的词组起始位置信息，但是这里是一维的一行其实信息，比如0010001001；actor_attention=>记录了一整个句子中的词组起始信息二维数组，是一个矩阵；Rinput=》最后经过一些简单处理的词组起始信息数组；最后的比例是句子中的词组数量与有效长度的比值

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



#转换rewards为严格的标准形状[batch_size,seq_len,2]
def preprocess_rewards(reward_list, device):
    max_len = max(len(r) for r in reward_list)
    reward_tensor = torch.zeros(len(reward_list), max_len, 2, device=device)
    for i, r in enumerate(reward_list):  # r: [seq_len, 2] 或 [seq_len]
        if isinstance(r, torch.Tensor):
            reward_tensor[i, :len(r)] = r.to(device)
        else:
            reward_tensor[i, :len(r)] = torch.tensor(r, device=device)
    return reward_tensor  # [batch_size, seq_len, 2]
#def train(args, data, train_dataloader, dev_dataloader, test_dataloader, criticModel, actorModel, LM_trainable=True,
          #RL_trainable=True):
    
def train(args, data, train_dataloader, dev_dataloader, test_dataloader, rewardModel, actorModel,criticModel, LM_trainable=True,
          RL_trainable=True,use_dynamic_epoch=False,early_stop_patience=12):#rewardModel=>translation_ranking_task,actorModel=>policynetwork,criticModel=>Criticnetwork

    """Train the model."""
    #分配主进程与其他进程，并以此来确定训练的步数
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    #根据模型的参数构建优化器和调度器
    # Prepare optimizer and schedule (linear warmup and decay)
    def optimizer_schedule(model):
        no_decay = ["bias", "LayerNorm.weight"]#不对配置项bias和层归一化layernorm.weight进行权重衰减，这两项不参与衰变
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay},#第一组：当n不在no_decay中时进行衰减
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}#不在no_decay的就进行衰减
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)#使用AdamW优化器
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)#一种线性学习率调节器
        return optimizer, scheduler

    if RL_trainable:
        args.learning_rate = args.learning_rate*args.tau#强化学习时计算出学习率
        #actor_target_optimizer, actor_target_schedule = optimizer_schedule(actorModel.target_policy)=》原有代码
        actor_target_optimizer, actor_target_schedule = optimizer_schedule(actorModel.policy_network)#for policynetwork，使用actor的主函数获得model(也就是embedding的内容actor，以及影射好的对子们context_emb)作为输入，输入到这个optimizer_schedule函数中获得优化器以及学习率调节器
        critic_target_optimizer, critic_target_schedule = optimizer_schedule(criticModel.policy_network)#用来控制critic网络的
        #在这里actort与critic共享优化器与学习率
        # actor_target_optimizer = torch.optim.Adam(actorModel.target_policy.parameters(), lr=args.learning_rate)
    
    #是不是强化学习都有这个部分，另外获取计算reward的critic
    reward_target_optimizer, reward_target_schedule = optimizer_schedule(rewardModel.target_pred)#target_pred就是structtransformer.mean_pool_embedding,这个函数会return一个embeds

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:#多GPU时安排的分布式训练
        rewardModel.target_pred=torch.nn.DataParallel(rewardModel.target_pred)#给计算reward的分配一个
        #actorModel.target_policy = torch.nn.DataParallel(actorModel.target_policy)
        actorModel.policy_network= torch.nn.DataParallel(actorModel.policy_network)#给actor分配一个
        criticModel.policy_network=torch.nn.DataParallel(criticModel.policy_network)#给真正的critic分别配一个

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(data.train_data))#记录训练数据集的样本数量
    logger.info("  Num Epochs = %d", args.num_train_epochs)#记录训练的epoch数量，即整个数据集将被遍历的次数
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)#记录每个GPU上的即时批次大小
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))#记录分布式训练的信息
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)#记录梯度累计的步骤数，也就是走了多少步之后才进行一次反向传播
    logger.info("  Total optimization steps = %d", t_total)#记录总的优化步骤

    best_loss = 999
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_actor_loss=999
    gpu_epoch_times = []  # 记录每个 epoch 的 GPU 时间
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    CRITIC_UPDATE_INTERVAL = 5  # 每3次Actor更新更新5次Critic
    critic_update_counter = 0
    #动态epoch控制
    early_stop_counter = 0
    #early_stop_flag = False

    #在训练集里训练时
    for epoch in train_iterator:
        # 开始记录当前epoch的GPU时间
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Iteration", disable=args.local_rank not in [-1, 0])#获取到循环器，因此在循环器里开始遍历训练集数据
        for step, batch in enumerate(epoch_iterator):#进入这个训练数据集中开始遍历
            global_step += 1
            rewardModel.zero_grad()  # 复制target模型的数data参数
            actorModel.zero_grad()
            criticModel.zero_grad()#用于清除梯度，通常会在每次反向传播前清除前一次计算的梯度
            totloss = 0.
            batch = tuple(t.to(args.device) for t in batch if t is not None)#用tuple()将处理后的元素重新组成一个元组

            text1 = batch[0]
            action1 = batch[4]
            mask1 = batch[2]
            head_mask1 = mask1.masked_fill(action1 == -1, 0)#这个函数将action1里等于-1的位置找出来，如何在mask1里action1=-1的对应位置都替换成0，如何把这个替换好的mask1给head_mask1;比如mask1=torch.tensor([1,2,3]),action=torch.tensor=([0,-1,1]),那么用了这个函数之后会得到head_mask1=torch.tensor([1,0,3])
            text2 = batch[1]
            mask2 = batch[3]
            action2 = batch[5]
            head_mask2 = mask2.masked_fill(action2 == -1, 0)
            sentence_id = batch[6]
            true_batch = min(args.batch_size * args.n_gpu, text1.size()[0])#来确定每个batch里的batch_size,设定batch大小一定小于二者其中一个

           
            sentence_actor_attention1 = None
            sentence_actor_attention2 = None
            sentence_actor_attention_perturbed1=None
            sentence_actor_attention_perturbed2=None
            losslist1 = []
            losslist2 = []
            lossperturbedlist1=[]
            lossperturbedlist2=[]
            aveloss_list1 = [0.0 for _ in range(true_batch)]
            aveloss_list2 = [0.0 for _ in range(true_batch)]
            actionlist1 = []
            actionlist2 = []
            total_loss = 0.

            if RL_trainable:#如果强化学习
                rewardModel.train(False)
                actorModel.train()
                criticModel.train()
                #处理输入的两个句子
                context1, context2 = \
                    rewardModel(text1, text2, mask1, mask2, None, None, sentence_id,
                                print_sentence=True)#借助model.py获得两个句子对应的encodeing之后的内容
                context1 = context1['last_hidden_state'].masked_fill(head_mask1.unsqueeze(-1) == 0, -1e9)#1，使用unsqueeze（-1）来给head_mask1的最后一层扩展一层；2，然后去找扩充好的head_mask1里等于-1的位置，拿这些位置去找context1最后一层里对应的位置，把context1最后一层里这些对应的位置换成-1e9
                context2 = context2['last_hidden_state'].masked_fill(head_mask2.unsqueeze(-1) == 0, -1e9)#约等于在最后一层换成掩码
                #此时我认为context才是处理了之后的state information
                predicted1 = actorModel.get_target_output(context1, mask1)#使用戴着掩码的context与掩码mask1来再次生成新的embedding，并以log的形式输出
                predicted2 = actorModel.get_target_output(context2, mask2)#同理，使用心得context与mask再次embedding#(batch_size,seq_len,2)
                #predicted1,2是经过linear层之后的部分action information，在这里actor和critic公用
                

                #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            
                #进入state information中进行进一步处理，actor负责生成action information

                for j in range(args.samplecnt):#config里的samplecnt为5，这里是获取action information
                    action_list1 = []
                    action_list2 = []
                    loss_list1 = []
                    loss_list2 = []
                    loss_perturbed_list1=[]
                    loss_perturbed_list2=[]
        
                    
                    # 批量采样
                    actions1, sentence_actor_attention1, _ = Sampling_RL_batch(args, predicted1, mask1, head_mask1, args.epsilon,Random=True)
                    actions2, sentence_actor_attention2, _ = Sampling_RL_batch(args, predicted2, mask2, head_mask2, args.epsilon,Random=True)
                    
                   
                    #load_start = time.perf_counter()
                    loss = rewardModel(text1, text2, mask1, mask2, sentence_actor_attention1,
                                       sentence_actor_attention2, sentence_id)[0]#计算损失loss(两个的相似度，reward），并在每个批次上进行反向传播，这是全局的loss
                    
                    #非强化学习时的loss
                    with torch.no_grad():
                        sentence_actor_attention_perturbed1 = sampling_random_batch(args, mask1, action1)
                        sentence_actor_attention_perturbed2 = sampling_random_batch(args, mask2, action2)
                        loss_perturbed = rewardModel(text1, text2, mask1, mask2, sentence_actor_attention_perturbed1,
                                    sentence_actor_attention_perturbed2, sentence_id)[0]
                    
                    
                    #load_time = time.perf_counter() - load_start
                                        
                    #print(f"|- 计算构建rewards的两个loss: {load_time*1000:.1f}ms")
                
                    #以上循环是用于生成action information（也就是记录了一个句子中词组起始信息的二维矩阵），并且获得一个全局的损失

                    for k in range(true_batch):#再次进入true_batch中根据GPU的数量来处理loss
                        # print(loss[1])
                        if args.n_gpu == 1:#主进程中
                            if len(loss_list1) != true_batch:#只有主进程有资格来初始化这个loss_list，从0开始累计添加到loss_list中
                                loss_list1.append(loss[1][int(k // args.n_gpu)].item())
                                loss_list2.append(loss[2][int(k // args.n_gpu)].item())
                                loss_perturbed_list1.append(loss_perturbed[1][int(k // args.n_gpu)].item())
                                loss_perturbed_list2.append(loss_perturbed[2][int(k // args.n_gpu)].item())

                            else:#直接在已经初始化好的loss_list上找到对应位置一点一点累加loss
                                loss_list1[k] += loss[1][int(k // args.n_gpu)].item()
                                loss_list2[k] += loss[2][int(k // args.n_gpu)].item()
                                loss_perturbed_list1[k] += loss_perturbed[1][int(k // args.n_gpu)].item()
                                loss_perturbed_list2[k] += loss_perturbed[2][int(k // args.n_gpu)].item()

                        else:#其他进程也只能在原有的loss_list上对应的位置累加loss
                            loss_list1[k] += loss[1][int(k // args.n_gpu)][int(k % args.n_gpu)].item()
                            loss_list2[k] += loss[2][int(k // args.n_gpu)][int(k % args.n_gpu)].item()
                            loss_perturbed_list1[k] += loss_perturbed[1][int(k // args.n_gpu)][int(k % args.n_gpu)].item()
                            loss_perturbed_list2[k] += loss_perturbed[2][int(k // args.n_gpu)][int(k % args.n_gpu)].item()
                        aveloss_list1[k] += loss_list1[k]#累加起来得到每个batch上的loss，这里的loss用于之后计算reward用的
                        aveloss_list2[k] += loss_list2[k]
                    loss[0].sum().backward()#将loss[0]的总和进行反向传播
                    losslist1.append(loss_list1)
                    losslist2.append(loss_list2)
                    lossperturbedlist1.append(loss_perturbed_list1)
                    lossperturbedlist2.append(loss_perturbed_list2)

                    #actionlist1.append(action_list1)#在这里生成最后的action information
                    #actionlist2.append(action_list2)
                    actionlist1.append(actions1)#在这里生成最后的action information
                    actionlist2.append(actions2)

                if LM_trainable:#在强化学习中进行机器学习的话
                    rewardModel.train()
                    actorModel.train()
                    criticModel.train()
                    reward_target_optimizer.zero_grad()#每次训练之前先清空梯度
                    
                    loss = rewardModel(text1, text2, mask1, mask2, sentence_actor_attention1,
                                       sentence_actor_attention2, sentence_id)[0][0]#这里的loss是loss1，也就是L12
                    
                    if not RL_trainable:#如果在机器学习里步进行强化学习就将这个L12累加起来
                        total_loss += loss
                    loss.sum().backward()#把L12的总和向后传播
                    reward_target_optimizer.step()#更新优化器
                    reward_target_schedule.step()#更新学习率调度器
                
                #load_time=0
                #load_start = time.perf_counter()
                baseline1 = 0.0  # 初始基线值
                baseline_decay1 = 0.8  # 基线更新系数，控制新观测值的权重
                baseline2 = 0.0  # 初始基线值
                baseline_decay2 = 0.8  # 基线更新系数，控制新观测值的权重
                # 计算奖励并更新模型
                for j in range(args.samplecnt):
                    rr_all1 = []
                    rr_all2 = []
                    # 先计算当前批次的原始delta值
                    batch_delta1 = []
                    batch_delta2 = []
                    for m in range(true_batch):
                        batch_delta1.append(lossperturbedlist1[j][m] - losslist1[j][m])
                        batch_delta2.append(lossperturbedlist2[j][m] - losslist2[j][m])
                    
                    # 计算当前批次的delta均值作为临时基线
                    current_batch_mean1 = np.mean(batch_delta1)
                    current_batch_mean2 = np.mean(batch_delta2)
                    current_batch_std1 = np.std(batch_delta1)+ 1e-6  # 防止除零
                    current_batch_std2 = np.std(batch_delta2)+ 1e-6  # 防止除零
                    # 更新移动平均基线（指数平滑）
                    baseline1 = baseline_decay1 * baseline1 + (1 - baseline_decay1) * current_batch_mean1
                    baseline2 = baseline_decay2 * baseline2 + (1 - baseline_decay2) * current_batch_mean2

                    for m in range(true_batch):
                        avg1 = aveloss_list1[m] / args.samplecnt
                        total_loss -= avg1
                        avg2 = aveloss_list2[m] / args.samplecnt
                        total_loss -= avg2
                        norm_delta1 = (batch_delta1[m] - current_batch_mean1) / current_batch_std1
                        rr_temp=[]
                        for pos in range(len(actionlist1[j][m])):
                            rr = [0, 0]
                            #rr[actionlist1[j][m][pos]] = ((losslist1[j][m] - avg1) * args.alpha )#用于计算每个详细位置上的reward
                            #delta=lossperturbedlist1[j][m]-losslist1[j][m]
                            # 关键变化：减去基线
                            #adjusted_delta = delta - baseline1
                            #rr[actionlist1[j][m][pos]] = adjusted_delta*1000* args.alpha
                            rr[actionlist1[j][m][pos]] = norm_delta1 * args.alpha
                            rr_temp.append(rr)
                        rr_all1.append(rr_temp)
                        
                        norm_delta2 = (batch_delta2[m] - current_batch_mean2) / current_batch_std2
                        rr_temp=[]
                        for pos in range(len(actionlist2[j][m])):
                            rr = [0, 0]
                            #rr[actionlist2[j][m][pos]] = ((losslist2[j][m] - avg2) * args.alpha)
                            #delta=lossperturbedlist2[j][m]-losslist2[j][m]
                            # 关键变化：减去基线
                            #adjusted_delta = delta - baseline2
                            #rr[actionlist2[j][m][pos]] = adjusted_delta *1000* args.alpha
                            rr[actionlist2[j][m][pos]] = norm_delta2 * args.alpha
                            rr_temp.append(rr)
                        rr_all2.append(rr_temp)
                    
                    #print("Reward mean:", np.mean(rr_all1), "Reward std:", np.std(rr_all1))

                    #load_time = time.perf_counter() - load_start
                                        
                    #print(f"|- 计算构建rewards的两个loss: {load_time*1000:.1f}ms")
                    
                    # 决定当前步骤是否更新Critic
                    should_update_critic = (critic_update_counter % CRITIC_UPDATE_INTERVAL == 0)
                    critic_update_counter += 1
                    
                    # 梯度清零
                    actor_target_optimizer.zero_grad()
                    if should_update_critic:
                        critic_target_optimizer.zero_grad()

                    # 批量计算梯度
                    actor_grad1, critic_grad1, _, actor_loss1, critic_loss1 = actorModel.get_batch_gradient(
                        context1, mask1, reward_batch=rr_all1, criticmodel=criticModel,update_critic=should_update_critic)
                    
                    actor_grad2, critic_grad2, _, actor_loss2, critic_loss2 = actorModel.get_batch_gradient(
                        context2, mask2, reward_batch=rr_all2, criticmodel=criticModel,update_critic=should_update_critic)
                    
                    # 平均梯度
                    actor_loss = (actor_loss1 + actor_loss2) / 2
                    critic_loss = (critic_loss1 + critic_loss2) / 2 if should_update_critic else None
                    
                    # 将列表拼接为一个 1D 向量
                    def flatten_grad_list(grad_list):
                        return torch.cat([g.view(-1) for g in grad_list if g is not None])

                    # 应用Actor梯度(始终执行)
                    for param, g1, g2 in zip(actorModel.policy_network.parameters(), actor_grad1, actor_grad2):
                        if param.grad is None:
                            param.grad = (g1 + g2) / 2
                        else:
                            param.grad += (g1 + g2) / 2
                    # 平均两个批次的梯度
                    avg_actor_grad = [(g1 + g2) / 2 for g1, g2 in zip(actor_grad1, actor_grad2)]
                    actor_grad_vector = flatten_grad_list(avg_actor_grad)
                    #计算CV
                    actor_grad_var = torch.var(actor_grad_vector)  # 计算梯度方差
                    actor_grad_mean = torch.mean(actor_grad_vector.abs())  # 计算梯度绝对值的均值
                    actor_grad_cv = actor_grad_var / (actor_grad_mean + 1e-8) # 计算梯度 CV（变异系数）
                    # 追加内容到文件末尾
                    if LM_trainable:#训练checkpoint-best的CV
                        with open("src/xtreme_7/output/checkpoint-best-actor-cv-mbert-0515.txt", "a") as file:
                            file.write('avg_actor_grad:'+str(avg_actor_grad)+'actor_grad_mean:'+str(actor_grad_mean)+'actor_grad_variance:'+str(actor_grad_var)+'actor_grad_cv:'+str(actor_grad_cv)+'\n') 
                        with open("src/xtreme_7/output/checkpoint-best-actor-loss-mbert-0515.txt", "a") as file:
                            file.write('actor_loss:'+str(actor_loss)+'\n')
                    else:#否则就是训练actor_only时的cv
                        with open("src/xtreme_7/output/checkpoint-only-actor-actor-loss-mbert-0515.txt", "a") as file:
                            file.write('actor_loss:'+str(actor_loss)+'\n')
                        with open("src/xtreme_7/output/checkpoint-only-actor-actor-cv-mbert-0515.txt", "a") as file:
                            file.write('avg_actor_grad:'+str(avg_actor_grad)+'actor_grad_mean:'+str(actor_grad_mean)+'actor_grad_variance:'+str(actor_grad_var)+'actor_grad_cv:'+str(actor_grad_cv)+'\n') 
                    actor_target_optimizer.step()
                    actor_target_schedule.step(actor_loss)

                    # Critic条件更新
                    if should_update_critic:
                        for param, g1, g2 in zip(criticModel.parameters(), critic_grad1, critic_grad2):
                            if param.grad is None:
                                param.grad = (g1 + g2) / 2
                            else:
                                param.grad += (g1 + g2) / 2
                        # 平均两个批次的梯度
                        avg_critic_grad = [(g1 + g2) / 2 for g1, g2 in zip(critic_grad1, critic_grad2)]
                        critic_grad_vector = flatten_grad_list(avg_critic_grad)
                        #计算cv
                        critic_grad_var = torch.var(critic_grad_vector)  # 计算梯度方差
                        critic_grad_mean = torch.mean(critic_grad_vector.abs())  # 计算梯度绝对值的均值
                        critic_grad_cv = critic_grad_var / (critic_grad_mean+ 1e-8)  # 计算梯度 CV（变异系数）
                        # 追加内容到文件末尾
                        if LM_trainable:#训练checkpoint-best的CV
                            with open("src/xtreme_7/output/checkpoint-best-critic-cv-mbert-0515.txt", "a") as file:
                                file.write('avg_critic_grad:'+str(avg_critic_grad)+'critic_grad_mean:'+str(critic_grad_mean)+'critic_grad_variance:'+str(critic_grad_var)+'critic_grad_cv:'+str(critic_grad_cv)+'\n') 
                            # 追加内容到文件末尾
                            with open("src/xtreme_7/output/checkpoint-best-critic-loss-mbert-0515.txt", "a") as file:
                                file.write('critic_loss:'+str(critic_loss)+'\n')
                        else:#否则就是训练actor_only时的cv
                            with open("src/xtreme_7/output/checkpoint-only-actor-critic-cv-mbert-0515.txt", "a") as file:
                                file.write('avg_critic_grad:'+str(avg_critic_grad)+'critic_grad_mean:'+str(critic_grad_mean)+'critic_grad_variance:'+str(critic_grad_var)+'critic_grad_cv:'+str(critic_grad_cv)+'\n') 
                            # 追加内容到文件末尾
                            with open("src/xtreme_7/output/checkpoint-only-actor-critic-loss-mbert-0515.txt", "a") as file:
                                file.write('critic_loss:'+str(critic_loss)+'\n')
                        critic_target_optimizer.step()
                        if should_update_critic and critic_loss is not None:
                            critic_target_schedule.step(critic_loss)
                    #将获得的梯度1D化
                    #actor_grad_vector = parameters_to_vector(actor_grad)  # 把所有梯度拼接成 1D 向量
                    #critic_grad_vector = parameters_to_vector(critic_grad)  # 把所有梯度拼接成 1D 向量
                    
                    # 调整学习率
                    #actor_target_schedule.step()
                    #critic_target_schedule.step()
                    # 仅当有有效critic_loss时才更新scheduler
                    
                   
            else:#非强化学习的情况下
                
                for i in range(true_batch):
                    act1 = action1[i]
                    act2 = action2[i]
                    m1 = mask1[i]
                    m2 = mask2[i]
                    if args.warm_by_action:
                        if i == 0:#第一次进入循环时，构建一个有内容的二维矩阵
                            sentence_actor_attention1 = sampling_random(args, m1, act1)
                            sentence_actor_attention2 = sampling_random(args, m2, act2)
                        else:#之后就直接拼接这些二维矩阵
                            sentence_actor_attention1 = torch.cat(
                                [sentence_actor_attention1, sampling_random(args, m1, act1)])
                            sentence_actor_attention2 = torch.cat(
                                [sentence_actor_attention2, sampling_random(args, m2, act2)])

                rewardModel.train()
                reward_target_optimizer.zero_grad()
                loss = rewardModel(text1, text2, mask1, mask2, sentence_actor_attention1,
                                   sentence_actor_attention2, sentence_id)[0][0]#这里得到的时综合的loss
                if not RL_trainable:#如果不强化学习直接累加loss
                    total_loss += loss
                loss.sum().backward()
                reward_target_optimizer.step()  # 更新active
                reward_target_schedule.step()

            if RL_trainable:
                rewardModel.train(False)
                actorModel.train()
                criticModel.train()#添加的真critic
                if LM_trainable:#监督学习下
                    rewardModel.train()
                    actorModel.train()
                    criticModel.train()#添加的真critic
            else:
                # print("Again RL False and LSTM True")
                rewardModel.train()
                actorModel.train(False)
                criticModel.train(False)

            # 获取当前时间
            #current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            #logger.info(f"[Step {global_step}] Evaluation triggered at: {current_time}")
            #trainning的期间做评估，每隔save_step就做一次评估，每次评估都会打印出来相关的训练步数以及各种loss，如果dev_loss小于之前保存的dev_loss就保存模型
            if global_step % args.save_step == 0:#根据config里的信息可知，默认save-step是50步
                # 获取当前时间
                current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                logger.info(f"[Step {global_step}] Evaluation triggered at: {current_time}")
                loss_dev_no_actor = eval_model(args, rewardModel, dev_dataloader)#评估模型，没有actor,需要用到相似性
                
                #情况一：又有热身活动又有强化学习
                if args.warm_by_action and RL_trainable:
                    
                    #loss_dev = eval_model_RL(args, valueModel, actorModel, dev_dataloader, RL_trained=RL_trainable)
                    loss_dev = eval_model_RL(args, rewardModel, actorModel, dev_dataloader, RL_trained=RL_trainable)#在这里使用强化学习的方法来测评获得第二种loss，dev_loss
                    print("batch ", global_step, "total loss ", total_loss, "| dev: ", loss_dev, "| dev_no_actor: ", loss_dev_no_actor)

                    #收集step和loss
                    with open("src/xtreme_7/output/step_with_loss-mbert-0515.txt","a")as file:
                        file.write('step:'+str(global_step)+'|'+'total loss:'+str(total_loss)+'|'+'dev:'+str(loss_dev)+'|'+'dev_no_actor:'+str(loss_dev_no_actor)+'\n')

                    if loss_dev < best_loss:#评价结果达到要求进行模型保存，在这里用loss_dev作为评判标准
                        best_loss = loss_dev
                        early_stop_counter = 0  # 重置耐心计数
                        # Save the best model checkpoint
                        output_dir = os.path.join(args.output_dir,
                                                  u"checkpoint-best{}{}".format("_no_policy" if not RL_trainable else "", "_actor_only" if RL_trainable and not LM_trainable else ""))

                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = rewardModel.target_pred.module if hasattr(rewardModel.target_pred,
                                                                                    "module") else rewardModel.target_pred
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_reward.bin"))


                        model_to_save = actorModel.policy_network.module if hasattr(actorModel.policy_network,
                                                                                   "module") else actorModel.policy_network
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_actor.bin"))

                        model_to_save = criticModel.module if hasattr(criticModel.policy_network,
                                                                        "module") else criticModel
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_critic.bin"))
                        logger.info('Saved model to %s' % output_dir)
                        logger.info(f"Model improved and saved at step {global_step}")
                    else:
                        early_stop_counter += 1
                        logger.info(f"No improvement count: {early_stop_counter}/{early_stop_patience}")
                    # 动态 Epoch 结束判断
                    if use_dynamic_epoch and early_stop_counter >= early_stop_patience:
                        logger.info(f"No improvement for {early_stop_patience} evaluations, early stop epoch {epoch+1}")
                        #early_stop_flag = True
                        early_stop_counter = 0  # 重置耐心计数
                        break  # 跳出本 epoch

                #情况二：没有热身运动,没有强化学习
                else:#此时评判标准有所变化，不是使用评估结果来做保存模型的条件，而是直接使用一开始不使用强化学习的评判结果
                    print("batch ", global_step, "total loss ", total_loss, "| dev_no_actor: ",
                          loss_dev_no_actor)
                    #收集step和loss
                    with open("src/xtreme_7/output/step_with_loss-mbert-0515.txt","a")as file:
                        file.write('step:'+str(global_step)+'|'+'total loss:'+str(total_loss)+'|'+'dev_no_actor:'+str(loss_dev_no_actor)+'\n')

                    if loss_dev_no_actor < best_loss:#这里的评判标准是loss_dev_no_actor
                        best_loss = loss_dev_no_actor
                        early_stop_counter = 0
                        # Save the best model checkpoint
                        output_dir = os.path.join(args.output_dir,
                                        u"checkpoint-best{}{}".format("_no_policy" if not RL_trainable else "", "_actor_only" if RL_trainable and not LM_trainable else ""))

                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        #保存translation ranking task
                        model_to_save = rewardModel.target_pred.module if hasattr(rewardModel.target_pred,
                                                                                "module") else rewardModel.target_pred
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_reward.bin"))

                        
                        #保存actor
                        model_to_save = actorModel.policy_network.module if hasattr(actorModel.policy_network,
                                                                                "module") else actorModel.policy_network
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_actor.bin"))

                        #保存value，也就是真正的critic
                        model_to_save = criticModel.module if hasattr(criticModel.policy_network,
                                                                                 "module") else criticModel.policy_network
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_critic.bin"))


                        logger.info('Saved model to %s' % output_dir)
                        logger.info(f"Model improved and saved at step {global_step}")
                    
                    '''
                    else:
                        early_stop_counter += 1
                        logger.info(f"No improvement count: {early_stop_counter}/{early_stop_patience}")'''

                tr_loss += total_loss#在这里累加total_loss
    #训练数据集的训练迭代结束
        # 结束epoch时判断是否提前终止
        #if use_dynamic_epoch and early_stop_counter >= early_stop_patience:
        #if early_stop_flag:
        #    logger.info(f"Early stopped at epoch {epoch+1}")
        #    break
        # 结束记录当前epoch的GPU时间
        end_event.record()
        torch.cuda.synchronize()
        epoch_time_ms = start_event.elapsed_time(end_event)
        epoch_time_sec = epoch_time_ms / 1000.0
        gpu_epoch_times.append(epoch_time_sec)
        print(f"Epoch {epoch} GPU time: {epoch_time_sec:.2f} seconds")
    

    #在训练集里训练之后使用测试集和验证集获取另外两种loss
    loss_test_no_actor = eval_model(args, rewardModel, test_dataloader)
    loss_dev_no_actor = eval_model(args, rewardModel, dev_dataloader)
   
    #如果热身加强化学习，则用带有注意力分数的方法来获取另外两种loss
    if args.warm_by_action and RL_trainable:
        
        loss_test = eval_model_RL(args, rewardModel, actorModel, test_dataloader, RL_trained=RL_trainable)
        loss_dev = eval_model_RL(args, rewardModel, actorModel, dev_dataloader, RL_trained=RL_trainable)
      
        print("batch", global_step, "total loss ", total_loss, "----test: ", loss_test, "| dev: ", loss_dev,
              "----test_no_actor: ", loss_test_no_actor, "| dev_no_actor: ", loss_dev_no_actor)
        
        #收集step和loss
        with open("src/xtreme_7/output/step_with_loss-mbert-0515.txt","a")as file:
            file.write('step:'+str(global_step)+'|'+'total loss:'+str(total_loss)+'|'+'test_loss:'+str(loss_test)+'|'+'dev:'+str(loss_dev)+'|'+'test_no_actor:'+str(loss_test_no_actor)+'|'+'dev_no_actor:'+str(loss_dev_no_actor)+'\n')

        if loss_dev < best_loss:
            # Save the best model checkpoint
            output_dir = os.path.join(args.output_dir, u"checkpoint-best{}{}".format("_no_policy" if not RL_trainable else "", "_actor_only" if RL_trainable and not LM_trainable else ""))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Take care of distributed/parallel training
            #保存translation ranking task
            model_to_save = rewardModel.target_pred.module if hasattr(rewardModel.target_pred,
                                                                                  "module") else rewardModel.target_pred
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_reward.bin"))

                       
            #保存actor
            model_to_save = actorModel.policy_network.module if hasattr(actorModel.policy_network,
                                                                                   "module") else actorModel.policy_network
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_actor.bin"))

            #保存value，也就是真正的critic
            model_to_save = criticModel.module if hasattr(criticModel.policy_network,
                                                                                   "module") else criticModel.policy_network
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_critic.bin"))
            
            logger.info('Saved model to %s' % output_dir)
    #又不强化学习又不热身
    elif loss_dev_no_actor < best_loss:
        print("batch ", global_step, "total loss ", total_loss,
              "----test_no_actor: ", loss_test_no_actor, "| dev_no_actor: ", loss_dev_no_actor)
        #收集step和loss
        with open("src/xtreme_7/output/step_with_loss-mbert-0515.txt","a")as file:
            file.write('[step:'+str(global_step)+'|'+'total loss:'+str(total_loss)+'|'+'test_no_actor:'+str(loss_test_no_actor)+'|'+'dev_no_actor:'+str(loss_dev_no_actor)+'\n')

        # Save the best model checkpoint
        output_dir = os.path.join(args.output_dir, u"checkpoint-best{}{}".format("_no_policy" if not RL_trainable else "", "_actor_only" if RL_trainable and not LM_trainable else ""))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Take care of distributed/parallel training
        model_to_save = rewardModel.target_pred.module if hasattr(rewardModel.target_pred,
                                                                  "module") else rewardModel.target_pred
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_reward.bin"))

        model_to_save = actorModel.policy_network.module if hasattr(actorModel.policy_network,
                                                                   "module") else actorModel.policy_network
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_actor.bin"))

        #保存value，也就是真正的critic
        model_to_save = criticModel.module if hasattr(criticModel.policy_network,
                                                                    "module") else criticModel.policy_network
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_critic.bin"))

        # torch.save(actorModel.state_dict(), os.path.join(output_dir, "training_model_actor.bin"))
        # torch.save(args, os.path.join(output_dir, "training_model_actor.bin"))
        logger.info('Saved model to %s' % output_dir)
    
    #没有热身的强化学习时直接保存模型
    elif RL_trainable:
        # Save the best model checkpoint
        output_dir = os.path.join(args.output_dir, u"checkpoint-final")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Take care of distributed/parallel training
        #保存translation ranking task
        model_to_save = rewardModel.target_pred.module if hasattr(rewardModel.target_pred,
                                                                                  "module") else rewardModel.target_pred
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_reward.bin"))

                       
        #保存actor
        model_to_save = actorModel.policy_network.module if hasattr(actorModel.policy_network,
                                                                                   "module") else actorModel.policy_network
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_actor.bin"))

        #保存value，也就是真正的critic
        model_to_save = criticModel.module if hasattr(criticModel.policy_network,
                                                                                   "module") else criticModel.policy_network
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_critic.bin"))

        # torch.save(actorModel.state_dict(), os.path.join(output_dir, "training_model_actor.bin"))
        # torch.save(args, os.path.join(output_dir, "training_model_actor.bin"))
        logger.info('Saved model to %s' % output_dir)

    # 训练全部结束，写入GPU时间日志
    gpu_log_file = os.path.join(args.output_dir, "mbert-0515gpu_time_log.txt")
    with open(gpu_log_file, "a") as f:
        for i, t in enumerate(gpu_epoch_times):
            f.write(f"Epoch {i}: {t:.2f} seconds\n")
        f.write(f"\nTotal GPU training time: {sum(gpu_epoch_times):.2f} seconds\n")
        f.write(f"Average per epoch: {sum(gpu_epoch_times)/len(gpu_epoch_times):.2f} seconds\n")

    print(f"GPU 时间日志已写入：{gpu_log_file}")
    
    return global_step, tr_loss / global_step


def eval_model(args, model, val_iter):#model=>选择要用的model，val_iter=》要用到的数据集，一般时测试数据集
    total_epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            batch = tuple(t.to(args.device) for t in batch if t is not None)
            text1 = batch[0]
            mask1 = batch[2]
            text2 = batch[1]
            mask2 = batch[3]
            sentence_id = batch[6]
            loss = model(text1, text2, mask1, mask2, None, None, sentence_id)[0][0]#直接获取model函数里的loss
            total_epoch_loss += loss.mean().item()#累加获取loss的平均值

    return total_epoch_loss / len(val_iter)


def eval_model_RL(args, criticModel, actorModel, val_iter, RL_trained=False):#根据词组边界确定，句子中词组序列等信息来计算loss
    total_epoch_loss = 0
    criticModel.eval()
    actorModel.eval()
    sentence_actor_attention1 = None
    sentence_actor_attention2 = None

    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            batch = tuple(t.to(args.device) for t in batch if t is not None)
            text1 = batch[0]
            action1 = batch[4]
            mask1 = batch[2]
            head_mask1 = mask1.masked_fill(action1 == -1, 0)
            text2 = batch[1]
            mask2 = batch[3]
            action2 = batch[5]
            head_mask2 = mask2.masked_fill(action2 == -1, 0)
            sentence_id = batch[6]
            true_batch = min(args.batch_size * args.n_gpu, text1.size()[0])

            context1, context2 = \
                criticModel(text1, text2, mask1, mask2, None, None, sentence_id,
                            print_sentence=True)
            context1 = context1['last_hidden_state'].masked_fill(head_mask1.unsqueeze(-1) == 0, -1e9)
            context2 = context2['last_hidden_state'].masked_fill(head_mask2.unsqueeze(-1) == 0, -1e9)

            predicted1 = actorModel.get_target_output(context1, mask1)
            predicted2 = actorModel.get_target_output(context2, mask2)

            if RL_trained:#如果强化学习
                for i in range(true_batch):
                    p1 = predicted1[i].cpu()
                    p2 = predicted2[i].cpu()
                    m1 = mask1[i]
                    m2 = mask2[i]
                    h1 = head_mask1[i]
                    h2 = head_mask2[i]
                    actions1, Rinput1, constituent_num1 = Sampling_RL(args, p1, m1, h1, args.epsilon, Random=False)

                    actions2, Rinput2, constituent_num2 = Sampling_RL(args, p2, m2, h2, args.epsilon, Random=False)
                    if i == 0:
                        sentence_actor_attention1 = Rinput1
                        sentence_actor_attention2 = Rinput2
                    else:
                        sentence_actor_attention1 = torch.cat([sentence_actor_attention1, Rinput1])
                        sentence_actor_attention2 = torch.cat([sentence_actor_attention2, Rinput2])
            else:#如果不强化学习
                for i in range(true_batch):
                    act1 = action1[i]
                    act2 = action2[i]
                    m1 = mask1[i]
                    m2 = mask2[i]
                    if i == 0:
                        sentence_actor_attention1 = sampling_random(args, m1, act1)
                        sentence_actor_attention2 = sampling_random(args, m2, act2)
                    else:
                        sentence_actor_attention1 = torch.cat([sentence_actor_attention1, sampling_random(args, m1, act1)])
                        sentence_actor_attention2 = torch.cat([sentence_actor_attention2, sampling_random(args, m2, act2)])
            loss = criticModel(text1, text2, mask1, mask2, sentence_actor_attention1, sentence_actor_attention2,
                               sentence_id)[0][0]
            total_epoch_loss += loss.mean().item()

    return total_epoch_loss / len(val_iter)



    total_epoch_loss = 0
    criticModel.eval()
    actorModel.eval()
    sentence_actor_attention1 = None
    sentence_actor_attention2 = None

    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            batch = tuple(t.to(args.device) for t in batch if t is not None)
            text1 = batch[0]
            action1 = batch[4]
            mask1 = batch[2]
            head_mask1 = mask1.masked_fill(action1 == -1, 0)
            text2 = batch[1]
            mask2 = batch[3]
            action2 = batch[5]
            head_mask2 = mask2.masked_fill(action2 == -1, 0)
            sentence_id = batch[6]
            true_batch = min(args.batch_size * args.n_gpu, text1.size()[0])

            context1, context2 = \
                criticModel(text1, text2, mask1, mask2, None, None, sentence_id,
                            print_sentence=True)
            context1 = context1['last_hidden_state'].masked_fill(head_mask1.unsqueeze(-1) == 0, -1e9)
            context2 = context2['last_hidden_state'].masked_fill(head_mask2.unsqueeze(-1) == 0, -1e9)

            predicted1 = actorModel.get_target_output(context1, mask1)
            predicted2 = actorModel.get_target_output(context2, mask2)

            if RL_trained:#如果强化学习
                # 初始化存储变量
                for i in range(true_batch):
                    # 批量采样动作和注意力矩阵
                    actions1, sentence_actor_attention1, constituent_nums1 = Batch_Sampling_RL(
                            args, predicted1, mask1, head_mask1, args.epsilon, Random=False)
                    actions2, sentence_actor_attention2, constituent_nums2 = Batch_Sampling_RL(
                            args, predicted2, mask2, head_mask2, args.epsilon, Random=False)
                    
            else:#如果不强化学习  
                for i in range(true_batch):
                    act1 = action1[i]
                    act2 = action2[i]
                    m1 = mask1[i]
                    m2 = mask2[i]
                    if i == 0:
                        sentence_actor_attention1 = sampling_random(args, m1, act1)
                        sentence_actor_attention2 = sampling_random(args, m2, act2)
                    else:
                        sentence_actor_attention1 = torch.cat([sentence_actor_attention1, sampling_random(args, m1, act1)])
                        sentence_actor_attention2 = torch.cat([sentence_actor_attention2, sampling_random(args, m2, act2)])
                
            loss = criticModel(text1, text2, mask1, mask2, sentence_actor_attention1, sentence_actor_attention2,
                               sentence_id)[0][0]
            total_epoch_loss += loss.mean().item()

    return total_epoch_loss / len(val_iter)

def eval_model_syn(args, criticModel, actorModel, val_iter):
    total_epoch_loss = 0
    criticModel.eval()
    actorModel.eval()
    sam_num_en = 0
    pred_num_en = 0
    true_num_en = 0
    sam_num_tgt = 0
    pred_num_tgt = 0
    true_num_tgt = 0

    const_num_en = []
    const_num_tgt = []

    const_num_en_g = []
    const_num_tgt_g = []

    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            batch = tuple(t.to(args.device) for t in batch if t is not None)
            text1 = batch[0]
            action1 = batch[4]
            mask1 = batch[2]
            head_mask1 = mask1.masked_fill(action1 == -1, 0)
            text2 = batch[1]
            mask2 = batch[3]
            action2 = batch[5]
            head_mask2 = mask2.masked_fill(action2 == -1, 0)
            sentence_id = batch[6]
            true_batch = min(args.batch_size * args.n_gpu, text1.size()[0])

            context1, context2 = \
                criticModel(text1, text2, mask1, mask2, None, None, sentence_id,
                            print_sentence=True)
            context1 = context1['last_hidden_state'].masked_fill(head_mask1.unsqueeze(-1) == 0, -1e9)
            context2 = context2['last_hidden_state'].masked_fill(head_mask2.unsqueeze(-1) == 0, -1e9)

            predicted1 = actorModel.get_target_output(context1, mask1)
            predicted2 = actorModel.get_target_output(context2, mask2)

            for i in range(true_batch):
                p1 = predicted1[i].cpu()
                p2 = predicted2[i].cpu()
                m1 = mask1[i]
                m2 = mask2[i]
                h1 = head_mask1[i]
                h2 = head_mask2[i]
                actions1, Rinput1, constituent_num1 = Sampling_RL(args, p1, m1, h1, args.epsilon, Random=False)

                actions2, Rinput2, constituent_num2 = Sampling_RL(args, p2, m2, h2, args.epsilon, Random=False)

                act1 = action1[i]
                act2 = action2[i]
                count_en = 0
                count_tgt = 0
                count_en_g = 0
                count_tgt_g =0
                for h_1, a1, a2 in zip(h1, actions1, act1):
                    if h_1 == 0:
                        continue
                    if a1 == 1:
                        count_en+=1
                        pred_num_en += 1
                    if a2 == 1:
                        count_en_g += 1
                        true_num_en += 1
                    if a1 == a2 and a2 == 1:
                        sam_num_en += 1
                for h_1, a1, a2 in zip(h2, actions2, act2):
                    if h_1 == 0:
                        continue
                    if a1 == 1:
                        count_tgt+=1
                        pred_num_tgt += 1
                    if a2 == 1:
                        count_tgt_g += 1
                        true_num_tgt += 1
                    if a1 == a2 and a2 == 1:
                        sam_num_tgt += 1
                const_num_en.append(count_en)
                const_num_en_g.append(count_en_g)
                const_num_tgt.append(count_tgt)
                const_num_tgt_g.append(count_tgt_g)

    P_en = sam_num_en/true_num_en
    R_en = sam_num_en/pred_num_en
    P_tgt = sam_num_tgt/true_num_tgt
    R_tgt = sam_num_tgt/pred_num_tgt
    F_en = (2*P_en*R_en)/(P_en+R_en)
    F_tgt = (2*P_tgt*R_tgt)/(P_tgt+R_tgt)

    print(sum(const_num_en)/len(const_num_en))
    print(sum(const_num_en_g)/len(const_num_en_g))
    print(sum(const_num_tgt)/len(const_num_tgt))
    print(sum(const_num_tgt_g)/len(const_num_tgt_g))
    print(P_en, R_en, P_tgt, R_tgt)

    return F_en*100, F_tgt*100


def main():
    args = config()
    args.edim = emb_config[args.ml_type]#选择基础模型

    #数据准备阶段
    print("*" * 25 + " DATA PREPARATION " + "*" * 25)
    data_cache_file = os.path.join(args.data_dir, u"train_{}.pkl".format(args.ml_type))
    #部署数据集
    if not os.path.exists(data_cache_file):#当缓存文件不存在，直接将这些数据集写入缓存文件中
        train_dp = read_parallel_txt(args.data_dir + "train.csv")
        dev_dp = read_parallel_txt(args.data_dir + "dev.csv")
        test_dp = read_parallel_txt(args.data_dir + "test.csv")
        dp = DataProcessor(dp_train=train_dp,
                           dp_dev=dev_dp,
                           dp_test=test_dp,
                           experiment=args)
        data = dp.process()#返回处理好的数据，一般是字典
        output_hal = open(data_cache_file, 'wb')
        str = pickle.dumps(data, protocol=4)#将数据对象data序列化为二进制格式
        output_hal.write(str)#写入缓存文件
        output_hal.close()
    else:
        with open(data_cache_file, 'rb') as file:#如果存在缓存文件就直接打开文件
            data = pickle.loads(file.read())

    print("*" * 25 + " DATA PREPARATION " + "*" * 25)

    #分布训练设备配置
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:#不用分布训练或则是只能用CPU时
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:#分布式训练模式
        # Initializes the distributed backend which sychronizes nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 0
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

    print("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)
    #模型初始化
    config_class, model_class = MODEL_CLASSES[args.ml_type]
    rewardModel = Reward(experiment=args, config_class=config_class, model_class=model_class)#计算reward用的
    actorModel = Actor(experiment=args, d_model=args.edim, dropout=args.act_dropout, no_cuda=args.no_cuda)#主体Actor
    criticModel=Critic(experiment=args, d_model=args.edim, dropout=args.act_dropout, no_cuda=args.no_cuda)#Critic
    # Make sure only the first process in distributed training loads model/vocab

    #进程0为主进程，确保所有进程同步
    if args.local_rank == 0:
        torch.distributed.barrier()
    logger.info("Training/evaluation parameters %s", args)
    train_dataset = data.train_data
    dev_dataset = data.dev_data
    test_dataset = data.test_data
    #配置数据加载器
    args.train_batch_size = args.batch_size * max(1, args.n_gpu)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=RandomSampler(train_dataset),
                                  batch_size=args.train_batch_size,
                                  drop_last=True)

    args.dev_batch_size = args.batch_size * max(1, args.n_gpu)
    dev_dataloader = DataLoader(dev_dataset,
                                sampler=RandomSampler(dev_dataset),
                                batch_size=args.dev_batch_size,
                                drop_last=True)

    args.test_batch_size = args.batch_size * max(1, args.n_gpu)
    test_dataloader = DataLoader(test_dataset,
                                 sampler=RandomSampler(test_dataset),
                                 batch_size=args.test_batch_size,
                                 drop_last=True,
                                 shuffle=False)
 
    #模型移动至指定1设备
    rewardModel.to(args.device)
    actorModel.to(args.device)
    criticModel.to(args.device)

    #如果训练的话
    if args.do_train:
        if not args.warm_start:#如果不warm_start直接进行5轮训练
            args.num_train_epochs = 5
        else:
            if not args.LMpretrain:#如果启动warm_start但是不机器学习的话,训练两轮获取reward
                args.num_train_epochs = 2#训练两轮
                #使用train()函数进行训练
                global_step, tr_loss = train(args, data, train_dataloader, dev_dataloader, test_dataloader, rewardModel,
                                             actorModel,criticModel, LM_trainable=True, RL_trainable=False,use_dynamic_epoch=False)
                logger.info("LM pretrained global_step = %s, average loss = %s", global_step, tr_loss)
           
           #将tartget_pred指定到设备args.device
            rewardModel.target_pred.to(args.device)
            output_dir = os.path.join(args.output_dir, "checkpoint-best_no_policy")
            print("Load pretrained LM from ", output_dir)
            rewardModel.target_pred.load_state_dict(
                    torch.load(os.path.join(output_dir, "target_model_reward.bin"), map_location=torch.device('cpu')))#保存没有机器学习下的的模型
            #获取训练好的reward model

            if not args.RLpretrain:#如果启动warm_start并且不强化学习的话，训练两轮获取actor
                args.num_train_epochs = 2#训练两轮
                #使用train()函数来进行训练
                global_step, tr_loss = train(args, data, train_dataloader, dev_dataloader, test_dataloader, rewardModel,
                                             actorModel,criticModel, LM_trainable=False, RL_trainable=True,use_dynamic_epoch=False)
                logger.info("RL pretrained global_step = %s, average loss = %s", global_step, tr_loss)
    
            output_dir = os.path.join(args.output_dir, "checkpoint-best_actor_only")
            print("Load action from ", output_dir)
            actorModel.policy_network.load_state_dict(torch.load(os.path.join(output_dir, "target_model_actor.bin")))#从指定的路径即(target_model_actor.bin)加载策略模型（actorModel.target_policy)的预训练参数到actorModel.target_policy
            actorModel.policy_network.to(args.device)
            #获取训练好的actor

            #最后，将训练轮数设置为3轮
            args.num_train_epochs = 3
        #调用train（）函数同时训练机器学习与强化学习，用之前训练好的model来一起训练三轮
        global_step, tr_loss = train(args, data, train_dataloader, dev_dataloader, test_dataloader, rewardModel,
                                     actorModel, criticModel,LM_trainable=True, RL_trainable=True,use_dynamic_epoch=False)
        logger.info("total global_step = %s, average loss = %s", global_step, tr_loss)

    #如果测评的话
    if args.do_eval:
        if not args.do_train:
            output_dir = os.path.join(args.output_dir, u"checkpoint-best")

            rewardModel.load_state_dict(torch.load(os.path.join(output_dir, "target_model_reward.bin")))
            actorModel.policy_network.load_state_dict(torch.load(os.path.join(output_dir, "target_model_actor.bin")))
            
            loss_test_no_actor = eval_model(args, rewardModel, test_dataloader)
            loss_test = eval_model_RL(args, rewardModel, actorModel, test_dataloader, RL_trained=True)

            print("----test: ", loss_test,
                  "----test_no_actor: ", loss_test_no_actor)

def main_lm():
    args = config()
    args.edim = emb_config[args.ml_type]

    print("*" * 25 + " DATA PREPARATION " + "*" * 25)
    data_cache_file = os.path.join(args.data_dir, u"train_{}.pkl".format(args.ml_type))
    if not os.path.exists(data_cache_file):
        train_dp = read_parallel_txt(args.data_dir + "train.csv")
        dev_dp = read_parallel_txt(args.data_dir + "dev.csv")   
        test_dp = read_parallel_txt(args.data_dir + "test.csv")
        dp = DataProcessor(dp_train=train_dp,
                           dp_dev=dev_dp,
                           dp_test=test_dp,
                           experiment=args)
        data = dp.process()
        output_hal = open(data_cache_file, 'wb')
        str = pickle.dumps(data, protocol=4)
        output_hal.write(str)
        output_hal.close()
    else:
        with open(data_cache_file, 'rb') as file:
            data = pickle.loads(file.read())

    print("*" * 25 + " DATA PREPARATION " + "*" * 25)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which sychronizes nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 0
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

    print("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    config_class, model_class = MODEL_CLASSES[args.ml_type]
    criticModel = Critic(experiment=args, config_class=config_class, model_class=model_class)
    actorModel = Actor(experiment=args, d_model=args.edim, dropout=args.act_dropout, no_cuda=args.no_cuda)
    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank == 0:
        torch.distributed.barrier()
    logger.info("Training/evaluation parameters %s", args)
    train_dataset = data.train_data
    dev_dataset = data.dev_data
    test_dataset = data.test_data

    args.train_batch_size = args.batch_size * max(1, args.n_gpu)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=RandomSampler(train_dataset),
                                  batch_size=args.train_batch_size,
                                  drop_last=True)

    args.dev_batch_size = args.batch_size * max(1, args.n_gpu)
    dev_dataloader = DataLoader(dev_dataset,
                                sampler=RandomSampler(dev_dataset),
                                batch_size=args.dev_batch_size,
                                drop_last=True)

    args.test_batch_size = args.batch_size * max(1, args.n_gpu)
    test_dataloader = DataLoader(test_dataset,
                                 sampler=RandomSampler(test_dataset),
                                 batch_size=args.test_batch_size,
                                 drop_last=True)

    criticModel.to(args.device)
    actorModel.to(args.device)
    if args.do_train:
        if not args.LMpretrain:
            args.num_train_epochs = 5
            global_step, tr_loss = train(args, data, train_dataloader, dev_dataloader, test_dataloader, criticModel,
                                         actorModel, LM_trainable=True, RL_trainable=False)
            logger.info("LM pretrained global_step = %s, average loss = %s", global_step, tr_loss)
        criticModel.target_pred.to(args.device)
        output_dir = os.path.join(args.output_dir, "checkpoint-best_no_policy")
        print("Load pretrained LM from ", output_dir)
        criticModel.target_pred.load_state_dict(
                torch.load(os.path.join(output_dir, "target_model_critic.bin"), map_location=torch.device('cpu')))

    if args.do_eval:
        output_dir = os.path.join(args.output_dir, "checkpoint-best_no_policy")
        criticModel.target_pred.load_state_dict(torch.load(os.path.join(output_dir, "target_model_critic.bin")))
        actorModel.target_policy.load_state_dict(torch.load(os.path.join(output_dir, "target_model_actor.bin")))
        print("Load pretrained LM from ", output_dir)
        loss_test_no_actor = eval_model(args, criticModel, test_dataloader)
        print("----test_no_actor: ", loss_test_no_actor)


def main_syn_test():
    args = config()
    args.edim = emb_config[args.ml_type]

    print("*" * 25 + " DATA PREPARATION " + "*" * 25)
    data_cache_file = os.path.join(args.data_dir, u"test_{}.pkl".format(args.ml_type))
    if not os.path.exists(data_cache_file):
        train_dp = read_parallel_txt(args.data_dir + "train.csv")
        dev_dp = read_parallel_txt(args.data_dir + "dev.csv")
        test_dp = read_parallel_txt(args.data_dir + "test.csv")
        dp = DataProcessor(dp_train=train_dp,
                           dp_dev=dev_dp,
                           dp_test=test_dp,
                           experiment=args)
        data = dp.process()
        output_hal = open(data_cache_file, 'wb')
        str = pickle.dumps(data, protocol=4)
        output_hal.write(str)
        output_hal.close()
    else:
        with open(data_cache_file, 'rb') as file:
            data = pickle.loads(file.read())

    print("*" * 25 + " DATA PREPARATION " + "*" * 25)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which sychronizes nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 0
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

    print("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    config_class, model_class = MODEL_CLASSES[args.ml_type]
    criticModel = Critic(experiment=args, config_class=config_class, model_class=model_class)
    actorModel = Actor(experiment=args, d_model=args.edim, dropout=args.act_dropout, no_cuda=args.no_cuda)
    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank == 0:
        torch.distributed.barrier()
    logger.info("Training/evaluation parameters %s", args)
    train_dataset = data.train_data
    dev_dataset = data.dev_data
    test_dataset = data.test_data

    args.test_batch_size = args.batch_size * max(1, args.n_gpu)
    test_dataloader = DataLoader(test_dataset,
                                 sampler=RandomSampler(test_dataset),
                                 batch_size=args.test_batch_size,
                                 drop_last=True)

    criticModel.to(args.device)
    actorModel.to(args.device)

    output_dir = os.path.join(args.output_dir, "checkpoint-best")
    criticModel.target_pred.load_state_dict(torch.load(os.path.join(output_dir, "target_model_critic.bin")))
    actorModel.target_policy.load_state_dict(torch.load(os.path.join(output_dir, "target_model_actor.bin")))

    print("Load pretrained LM from ", output_dir)
    F1_en, F1_tgt = eval_model_syn(args, criticModel, actorModel, test_dataloader)
    print("----F1_en: ", F1_en, "----F1_tgt: ",F1_tgt)



if __name__ == '__main__':
    main()
    # main_lm()
    #main_syn_test()
    
