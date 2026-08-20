from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#归一化
class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

#self-attention层的输出处理模块
class SelfOutput(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps=1e-05, hidden_dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    

class CriticNetwork(nn.Module):
    """简化版Critic网络：共享Actor的context_emb特征，仅用单层线性层输出价值"""
    def __init__(self, input_dim=1024, dropout=0.3):  # 降低dropout防止价值估计波动
        super(CriticNetwork, self).__init__()
        #如果价值估计不正确则多加一层layernorm
        #self.ln1=nn.LayerNorm(input_dim)
        # 直接映射Actor输出的context_emb到价值（无额外降维层，减少参数）
        self.value_head = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout)  # 轻量正则化
        
        # 初始化权重（保持数值稳定性）
        nn.init.xavier_uniform_(self.value_head.weight)

    def forward(self, actor_context_emb):
        """
        输入：复用Actor的context_emb（[batch_size, seq_len, input_dim]）
        输出：每个token的状态价值 [batch_size, seq_len]
        """
        #x = self.ln1(actor_context_emb)
        #x = self.dropout(x)
        x = self.dropout(actor_context_emb)  # 仅保留必要的dropout
        return self.value_head(x).squeeze(-1)  # 单层映射输出价值

class Critic(nn.Module):
    """Critic封装类，与Actor逻辑对齐"""
    def __init__(self, experiment, d_model, dropout=0.3, no_cuda=False):
        super(Critic, self).__init__()
        self.experiment = experiment
        self.d_model = d_model
        self.policy_network = CriticNetwork(input_dim=d_model, dropout=dropout)
        self.no_cuda = no_cuda

    def forward(self, actor_context_emb):
        """直接使用Actor的context_emb计算状态价值"""
        return self.policy_network(actor_context_emb)

    def assign_network_gradients(self, grad):
        """梯度分配（与Actor保持一致的tau更新策略）"""
        i = 0
        tau = 0.5
        for name, param in self.policy_network.named_parameters():
            if grad[i].shape == param.shape:
                param.grad = deepcopy(grad[i].data * tau + param.data * (1 - tau))
            else:
                raise ValueError(f"梯度形状不匹配：{grad[i].shape} vs {param.shape}")
            i += 1

    # 以下方法保留以确保兼容性（实际训练中可能无需调用）
    def get_target_logOutput(self, x):
        out = self.policy_network(x)
        return torch.log(out + 1e-9)  # 避免log(0)

    def get_target_output(self, x):
        return self.policy_network(x)