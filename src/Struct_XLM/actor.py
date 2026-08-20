from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/workspace/SAC-XLM/Struct-XLM-main/src/Struct_XLM/')


#归一化类
class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps#防止除0错误

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)#计算均值
        std = x.std(-1, keepdim=True)#计算标准差
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2#归一化数据

#self-attention层的输出处理模块
class SelfOutput(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps=1e-05, hidden_dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)#一个线性层
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)#一个归一化层
        self.dropout = nn.Dropout(hidden_dropout_prob)#一个dropout

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)#残差链接
        return hidden_states
    

class PolicyNetwork(nn.Module):
    def __init__(self, d_model, dropout=0.8):
        super(PolicyNetwork, self).__init__()
        self.d_model = d_model
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        # self.linear_output = nn.Linear(d_model, d_model)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = SelfOutput(d_model)

        self.actor_linear = nn.Linear(d_model, 2)  # 0 represent "in the constituent", 1 represent the tail of
        # constituent, 2 represent apart of puncuation
        #输出是一个二维数组，0代表是词组内部，1代表词组结尾
       

    def forward(self, context, mask):
        #torch.autograd.set_detect_anomaly(True)


        
        # (batch, seq_len, d_model)=》context的形状
        context = self.norm(context)

        key = self.linear_key(context)  # (batch, seq_len, d_model)
        query = self.linear_query(context)  # (batch, seq_len, d_model)
        value = self.linear_value(context)  # (batch, seq_len, d_model)
        #这个部分用来得出句子中各个词组的起始位置
        mask_new = []
        for i in range(mask.size()[0]):#循环遍历掩码张量的第0维，也即是0维的每一行
            length = torch.sum(mask[i]).item()#这里是得到有效长度
            mask_ = mask[i].unsqueeze(0).repeat(length,1).cpu()#首先使用unsqueeze（）函数来添加一个维度，（可能本来是一维度，现在变成了二维度）；如何沿着第0维度重复length（有效长度）次，从而生成一个形状为（length,seq_len)的张量
            zero_tensor = torch.zeros((mask_.size()[1]-length, mask_.size()[1]))#创建零张量，形状为（msak_size-length,mask_size)msak_size-length,mask_size)
            mask_ = torch.cat((mask_, zero_tensor), dim=0)#将mask_张量与zero_tensor张量在第0维拼接
            mask_ = torch.tril(mask_, diagonal=0)#生成下三角矩阵
            mask_new.append(mask_)
        mask = torch.stack(mask_new, dim=0).cuda()#将结果堆叠到GPU上

        #计算注意力分数并应用掩码
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_model#QK(T)/根号d(k)
        # print(scores.size(), ",", mask.size())
        scores = scores.masked_fill(mask == 0, -1e9)#将scores里等于0的部分用-1e9取代
        scores = F.softmax(scores, dim=-1)#softmax(QK(T)/根号d(k))
        # attention_probs = self.dropout(scores)

        #计算上下文向量
        context_layer = torch.matmul(scores, value)#将scores与V进行矩形相乘
        context_emb = self.out(context_layer, context)#将二者投影起来，找到conext对应的conext_layer


        #生成策略
        actor = self.actor_linear(context_emb)#将上下文embedding映射到2个类别上
        actor = F.softmax(actor, dim=-1)#使其变成概率分布，策略输出
        # (batch, seq_len, d_model)


        return actor, context_emb


class PolicyNetwork1(nn.Module):
    def __init__(self, d_model, dropout=0.8, no_cuda=False):
        super(PolicyNetwork1, self).__init__()
        self.d_model = d_model
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        # self.linear_output = nn.Linear(d_model, d_model)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.no_cuda = no_cuda

    def forward(self, context, eos_mask):
        batch_size, seq_len = context.size()[:2]

        context = self.norm(context)

        if self.no_cuda:
            a = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32), 1))
            b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32), 0))
            c = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32), -1))
            tri_matrix = torch.from_numpy(np.triu(np.ones([seq_len, seq_len], dtype=np.float32), 0))
        else:
            a = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32), 1)).cuda()
            b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32), 0)).cuda()
            c = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32), -1)).cuda()
            tri_matrix = torch.from_numpy(np.triu(np.ones([seq_len, seq_len], dtype=np.float32), 0)).cuda()

        # mask = eos_mask & (a+c) | b
        mask = eos_mask & (a + c)

        key = self.linear_key(context)
        query = self.linear_query(context)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_model

        scores = scores.masked_fill(mask == 0, -1e9)
        neibor_attn = F.softmax(scores, dim=-1)
        neibor_attn = torch.sqrt(neibor_attn * neibor_attn.transpose(-2, -1) + 1e-9)

        t = torch.log(torch.tensor(neibor_attn + 1e-9)).masked_fill(a == 0, 0).matmul(tri_matrix)
        g_attn = tri_matrix.matmul(t).exp().masked_fill((tri_matrix.int() - b) == 0, 0)
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(b == 0, 1e-9)

        return g_attn, neibor_attn


class Actor(nn.Module):
    def __init__(self,experiment, d_model, dropout=0.8,no_cuda=False):
        super(Actor, self).__init__()
        self.experiment = experiment
        self.d_model = d_model
        self.policy_network = PolicyNetwork(d_model=d_model, dropout=dropout)

    
    def get_batch_gradientfortest(self, context,mask, reward_batch,criticmodel, gamma=0.99):
        # 1. 计算 Actor 网络输出的动作概率
        out, context_emb = self.policy_network(context, mask)#context.shape=(batch_size,256,1024)
        #pooled_context=context.sum(dim=1)#(batch_size,1024)
        state_value=criticmodel.policy_network(context_emb)#(batch_size,max_len)

        logout = torch.log(out)

        #因为要使用A2C算法，在A2C中，使用Critic-network的状态价值和advantage来指导更新，而不是直接依赖奖励为0的索引处理，因此将这个部分删除掉
        #整理reward，找到rewrad出现0的索引，将其对应索引的位置设置为-1，如果当前奖励是最后一个奖励，则将后续位置也设置为-1
        #因为等于0会干扰计算，所有将出现0的部分的值处理成-1
        for i, reward_sent in enumerate(reward_batch):
            for j, reward in enumerate(reward_sent):
                index = reward.index(0)
                logout[i][j][index] = -1
                if j == len(reward_sent)-1:
                    for k in range(j+1, logout.size()[1]):
                        logout[i][j][0] = -1
                        logout[i][j][1] = -1                  
      
        # 3. 初始化梯度输出
        grad_output = torch.zeros_like(logout)
        targets_output=torch.zeros_like(state_value)
        #计算GAE
        def compute_gae(rewards, values, gamma=0.99, lambda_=0.95):
            """
            计算 GAE（Generalized Advantage Estimation）

            参数：
                rewards: 形状 [T]，时间步的奖励
                values: 形状 [T+1]，Critic 计算的状态值（包括终止状态）
                gamma: 折扣因子
                lambda_: GAE 衰减因子

            返回：
                advantages: 形状 [T]，GAE 计算的 Advantage 值
                targets: 形状 [T]，TD 目标值（用于 Critic 训练）
            """
            T = len(rewards)
            advantages = torch.zeros(T,2)
            targets=torch.zeros(T)
            last_advantage = 0
            
            for t in reversed(range(T)):  # 从后往前计算 GAE
                #如果t是最后一步，那么让v[t+1]=0
                next_value = 0 if t == T - 1 else values[t + 1]
                for r in range(2):
                    if rewards[t][r] !=0:
                        delta = rewards[t][r] + gamma * next_value - values[t]  # TD 误差 δ_t
                        advantages[t][r] = last_advantage = delta + gamma * lambda_ * last_advantage  # 递推 GAE
                        #targets.append(advantages[t][r])
                        targets[t]=advantages[t][r]
                    else:
                        advantages[t][r]=0
                     
            targets=targets+values[:-1]#计算TD error
        
            return advantages, targets
      
        
        #print("reward_batch",reward_batch)#每个rewards_batch里有5个句子，每个句子的长度都长短不一，
        for i, reward_sent in enumerate(reward_batch):
            #rewards = torch.tensor([reward[index] for reward in reward_sent])  # 获取奖励序列
            
            valid_length=len(reward_sent)
           
            # 裁剪 state_value[i] 以匹配 reward_sent 的长度
            #values = state_value[i,:valid_length+1].detach().cpu().numpy().astype(np.float32) 
            values = state_value[i, :valid_length+1].detach().cpu().float()


            advantages, targets = compute_gae(reward_sent, values)  # 计算 GAE

            for j,adv in enumerate(advantages):
                grad_output[i][j]=adv
               
            for m, tar in enumerate(targets):
                targets_output[i][m] = tar# 更新 TD 目标'''   


        #5.计算actor_loss
        #actor_loss0 = - (logout * grad_output).sum()
        #print(f"actor_loss0",actor_loss0)

        #以下操作是归一化actor_loss
        # 构造有效 mask（仅在 advantage ≠ 0 的位置计算 loss）
        valid_mask = (grad_output != 0).float()

        # 避免除 0
        valid_count = valid_mask.sum()
        if valid_count == 0:
            actor_loss = torch.tensor(0.0, device=logout.device, requires_grad=True)
        else:
            # masked element-wise product
            log_adv_product = logout * grad_output * valid_mask
            # 求 actor loss：仅基于非零 advantage 的位置
            actor_loss = - log_adv_product.sum() / valid_count
            
    
        #6.获取critic loss
        critic_loss = F.mse_loss(state_value, targets_output)
        
        #critic_loss=F.mse_loss(state_value,targets_output)
        
        #grad = torch.autograd.grad(logout, self.policy_network.parameters(), grad_outputs=grad_output)
        actor_grad = torch.autograd.grad(actor_loss, self.policy_network.parameters(),retain_graph=True)  # 计算 Actor 梯度
        critic_grad = torch.autograd.grad(critic_loss, criticmodel.parameters(),retain_graph=True)  # 计算 Critic 梯度
   

        return actor_grad, critic_grad, out, actor_loss, critic_loss
        


    def get_batch_gradient(self, context, mask, reward_batch, criticmodel,update_critic=True, gamma=0.99, lambda_=0.95):
        """
        context: [batch_size, seq_len, hidden_dim]
        reward_batch: List[ [ [r0, r1], [r0, r1], ... ] ] -> 每个样本的每个 token 一个 [2] 向量
        """
        batch_size, seq_len = context.size(0), context.size(1)

        # 1. 策略网络前向传播
        out, context_emb = self.policy_network(context, mask)  # out: [batch, seq_len, 2], context_emb: [batch, hidden_dim]

        # 2. Critic 前向传播 - 获取每个token的state value
        #state_value = criticmodel(context) # [batch_size, seq_len]
        # 2. Critic前向传播（动态控制计算图）
        if criticmodel is None:
            # w/o Critic: 不使用 value baseline
            state_value = torch.zeros(batch_size, seq_len, device=context.device)
        else:
            with torch.set_grad_enabled(update_critic):
                state_value = criticmodel(context_emb)  # [batch_size, seq_len]
                if not update_critic:
                    state_value = state_value.detach()  # 非更新步骤时截断梯度

        # 3. 预处理reward_batch  
        # 将reward_batch转换为tensor并提取有效reward（根据你的描述，每个位置只有r0或r1有效）
        rewards = torch.zeros(batch_size, seq_len, device=context.device)
        action_mask = torch.zeros(batch_size, seq_len, dtype=torch.long, device=context.device)#记录有效action的位置
        for i in range(batch_size):
            for j in range(min(len(reward_batch[i]), seq_len)):  # 处理可能存在的长度不匹配
                r0, r1 = reward_batch[i][j]
                # 确定有效reward和对应动作
                if r0 != 0:
                    rewards[i,j] = r0
                    action_mask[i,j] = 0  # 选择动作0
                else:
                    rewards[i,j] = r1
                    action_mask[i,j] = 1  # 选择动作1

        # 4. 计算 TD target & advantage
        with torch.no_grad():
            td_target = rewards # [batch_size, seq_len]
            advantage = td_target - state_value.detach()    # [batch_size, seq_len]

        # 5. 计算actor_loss (策略梯度损失)
        log_probs = torch.log_softmax(out, dim=-1)  # [batch, seq_len, 2]
        selected_log_probs = log_probs.gather(2, action_mask.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len]
        
        # 只对有效token计算loss（mask处理）
        valid_mask = mask.float() * (rewards != 0).float()  # 结合padding mask和reward有效标记
        actor_loss = - (selected_log_probs * advantage.detach() * valid_mask).sum() / (valid_mask.sum() + 1e-10)

        
        # 6. 计算critic_loss (价值函数损失)
        #critic_loss = F.mse_loss(state_value * valid_mask, td_target * valid_mask, reduction='sum') / (valid_mask.sum() + 1e-10)
        #有条件得计算critic_loss
        critic_loss = None
        if (criticmodel is not None) and update_critic:
            critic_loss = F.mse_loss(state_value * valid_mask, td_target * valid_mask, reduction='sum') / (valid_mask.sum() + 1e-10)

        # 7. 计算梯度
        actor_grad = torch.autograd.grad(actor_loss, self.policy_network.parameters(), retain_graph=True)

        #有条件地计算critic的梯度
        critic_grad = None
        if (criticmodel is not None) and update_critic:
            #critic_grad = torch.autograd.grad(critic_loss, criticmodel.parameters(), retain_graph=True)
            critic_grad = torch.autograd.grad(critic_loss, criticmodel.parameters(), retain_graph=False)# 不再需要保留计算图

        return actor_grad, critic_grad, out, actor_loss.item(), critic_loss.item() if critic_loss is not None else 0.0


    def compute_actor_loss(self, log_probs, advantages):
        # 计算 Actor 的损失：-log(π(a|s)) * A(s, a)
        actor_loss = -torch.mean(log_probs * advantages)
        return actor_loss


    def get_action(self, context, mask):
        # 获取动作的概率分布，通常用于采样动作
        policy_output, _ = self.policy_network(context, mask)
        action_probs = F.softmax(policy_output, dim=-1)
        action = torch.multinomial(action_probs, num_samples=1)  # 采样动作
        log_probs = torch.log(action_probs.gather(-1, action))
        return action, log_probs

    
    def assign_network_gradients(self, grad):#将计算好的梯度分配到对应的参数去
        i = 0
        tau = 0.5
        for name, x in self.policy_network.named_parameters():
            x.grad = deepcopy(grad[i].data * (tau) + x.data * (1 - tau))
            i += 1

    def get_target_logOutput(self, h, x):
        out, _ = self.policy_network(h, x)
        logOut = torch.log(out)
        return logOut

    # 调用 policy 网络输出策略分布
    def get_target_output(self, context, mask):
        out, _ = self.policy_network(context, mask)
        return out
    
  