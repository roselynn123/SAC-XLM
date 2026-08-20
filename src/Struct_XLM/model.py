from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

#实现了一种加权的AM-Softmax损失（Angular Margin Softmax loss)
class AdMSoftmaxLoss(nn.Module):

    def __init__(self, s=1, m=0.4):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m

    def forward(self, x1, x2, sentence_id):
        """
        x1,x2 shape (N, in_features)
        """
        #计算句子1，2的相似度，1=》2，2=》1都要计算出来
        N = x1.size()[0]
        x12 = F.normalize(torch.matmul(x1, x2.transpose(0, 1)), dim=1)   # (N,N)维度是（N,N)
        x21 = F.normalize(torch.matmul(x2, x1.transpose(0, 1)), dim=1)  # (N,N)
        labels = sentence_id.cpu().numpy().tolist()

        loss1 = []
        loss2 = []
        loss = 0.0
        more_label = False
        label = []
        #标签处理，处理标签并创建label列表
        for i in range(N):
            label_1 = []
            for j,l in enumerate(labels):
                if labels[i] == l:
                    label_1.append(j)
            label.append(label_1)
            if len(label_1) > 1:
                more_label = True


        if not more_label:#多标签处理
            numerator = self.s * (torch.diagonal(x12) - self.m)   # (1,N)计算损失的分子，s是缩放因子，m是边距因子，计算对角线上元素的值，这些对角线元素表示样本与自己（相同标签)的相似度
            excl = torch.cat([torch.cat((x12[i, :y[0]], x12[i, y[0] + 1:])).unsqueeze(0) for i, y in enumerate(label)], dim=0)  #(N,N-1)，这个部分是计算对角线以外的元素，excel是排除了主对角线的相似度，表示样本与其他标签的相似度
            denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)  #计算损失的分母，计算了目标类别的指数相似度和非目标类别的指数相似度之和
            L12 = numerator - torch.log(denominator)

            numerator = self.s * (torch.diagonal(x21) - self.m)
            excl = torch.cat([torch.cat((x21[i, :y[0]], x21[i, y[0] + 1:])).unsqueeze(0) for i, y in enumerate(label)], dim=0)
            denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
            L21 = numerator - torch.log(denominator)
        else:#单标签处理
            numerator = self.s * (torch.cat([torch.sum(torch.exp(x12[i,y] - self.m), dim=0).view(-1,) for i,y in enumerate(label)], dim=0))#在单标签情况下，计算每个样本与正确标签的相似度（减去m)并积累这些相似度
            excl = self.s * (torch.cat([torch.sum(torch.exp(x12[i,list(set([k for k in range(N)]).difference(set(y)))]), dim=0).view(-1,) for i,y in enumerate(label)], dim=0))#计算样本与非目标标签的相似度之和
            denominator = numerator + excl
            L12 = torch.log(numerator) - torch.log(denominator)

            numerator = self.s * (
                torch.cat([torch.sum(torch.exp(x21[i, y] - self.m), dim=0).view(-1,) for i, y in enumerate(label)], dim=0))
            excl = self.s * (torch.cat(
                [torch.sum(torch.exp(x21[i, list(set([k for k in range(N)]).difference(set(y)))]), dim=0).view(-1,) for i, y in
                 enumerate(label)], dim=0))
            denominator = numerator + excl
            L21 = torch.log(numerator) - torch.log(denominator)

        loss1 = L12
        loss2 = L21
        loss = -torch.mean(L12)-torch.mean(L21)
        return loss, loss1, loss2

#用于处理两个句子的输入，并根据配置不同的句子embedding提取方式
class StructTransformer(nn.Module):

    def __init__(self, experiment, config_class, model_class):
        super(StructTransformer, self).__init__()
        self.expe = experiment

        self.config = config_class.from_pretrained(self.expe.ml_model_path)#config_class是由一个叫做MODEL_CLASSES的选择函数根据输入命令中选择的基础模型类型来决定的
        self.model_dim = self.config.hidden_size
        self.encoder = model_class.from_pretrained(self.expe.ml_model_path)#model_class也是由MODEL_CLASSES的选择函数根据输入命令中选择的基础模型类型来决定的，model_class和config_class对应的是输入的预训练模型和配置文件
        self.admSoftmaxLoss = AdMSoftmaxLoss(s=self.expe.s, m=self.expe.margin)
        self.sentence_type = self.expe.sentence_type#sentence_type指的是句子embedding的提取方式，有mean，pool_mean,以及pool

        #classifier_dropout = (
        #    self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        #)
        #self.dropout = nn.Dropout(classifier_dropout)
    #均值池化函数
    def mean_pool_embedding(self, embeds, masks):
        """
        Args:
          embeds: list of torch.FloatTensor, (B, L, D)
          masks: torch.FloatTensor, (B, L)
        Return:
          sent_emb: list of torch.FloatTensor, (B, D)
      """
        embeds = (embeds * masks.unsqueeze(2).float()).sum(dim=1) / masks.sum(dim=1).view(-1, 1).float()#计算句子的均池化表示，忽略填充部分
        return embeds

    def forward(self, input1, input2, mask1, mask2, action1, action2, sentence_id, print_sentence=False):
        output1 = self.encoder(input1.long(), attention_mask=mask1, struct_action_probs=action1)#根据配置来获取encoding之后的内容
        output2 = self.encoder(input2.long(), attention_mask=mask2, struct_action_probs=action2)
        if print_sentence:
            return output1, output2
        if self.sentence_type == 'mean':
            input1 = output1['last_hidden_state'].mean(dim=1)
            input2 = output2['last_hidden_state'].mean(dim=1)
        elif self.sentence_type == 'pool_mean':
            input1 = self.mean_pool_embedding(output1['last_hidden_state'], mask1)
            input2 = self.mean_pool_embedding(output2['last_hidden_state'], mask2)
        elif self.sentence_type == 'pool':
            input1 = output1['pooler_output']
            input2 = output2['pooler_output']

        #sequence1_output = self.dropout(input1)
        #sequence2_output = self.dropout(input2)
        loss = self.admSoftmaxLoss(input1, input2, sentence_id)#计算句子对的损失

        return loss, (input1, input2)


#这个部分不是真正的Critic，这里是为StructTransformer完成embedding之后的内容计算reward的
class Reward(nn.Module):
    def __init__(self, experiment, config_class, model_class):
        super(Reward, self).__init__()
        self.expe = experiment
        self.target_pred = StructTransformer(experiment, config_class, model_class)

    def forward(self, input1, input2, mask1, mask2, action1, action2, sentence_id, print_sentence=False):
        out = self.target_pred(input1, input2, mask1, mask2, action1, action2, sentence_id, print_sentence=print_sentence)
        return out

    '''
    def compute_token_rewards(self, input1, input2, mask1, mask2, action1, action2, sentence_id, flip_prob=0.2):
        """
        计算每个 token 的 reward，方式为翻转结构边产生扰动，并观察 loss 的变化。
        
        Args:
            action1: (B, 1, T, T) 的结构连接矩阵
            flip_prob: 每个边被翻转的概率（建议 0.1 - 0.3）

        Returns:
            token_rewards: (B, T) 的每个 token 的 reward 值
        """
        B, _, T, _ = action1.shape
        device = action1.device

        # 原始结构的损失
        with torch.no_grad():
            loss_orig, _ = self.target_pred(input1, input2, mask1, mask2, action1, action2, sentence_id)
            loss_orig1 = loss_orig[1]  # tensor of shape (B,)
            loss_orig2 = loss_orig[2]

        
        token_rewards1 = torch.zeros(B, T, device=device)
        token_rewards2 = torch.zeros(B, T, device=device)

        for i in range(T):
            # 创建一个扰动版本
            perturbed_action1 = action1.clone()

            # 对第 i 个 token 的结构连接行进行扰动
            for b in range(B):
                row = perturbed_action1[b, 0, i]  # shape: (T,)
                flip_mask = (torch.rand(T, device=device) < flip_prob).float()
                # XOR 翻转 0/1
                row = (row + flip_mask) % 2
                perturbed_action1[b, 0, i] = row

            # 使用扰动结构再次 forward
            with torch.no_grad():
                loss_perturbed, _ = self.target_pred(input1, input2, mask1, mask2, perturbed_action1, action2, sentence_id)
                loss_perturbed1 = loss_perturbed[1]  # shape: (B,)
                loss_perturbed2 = loss_perturbed[2]

            # reward = 原始 loss - 扰动 loss（损失升高说明该 token 更重要）
            token_rewards1[:, i] = loss_orig1  - loss_perturbed1 
            token_rewards2[:, i] =  loss_orig2 -  loss_perturbed2



        return token_rewards1,token_rewards2
        '''
    def compute_token_rewards(self, input1, input2, mask1, mask2, action1, action2, sentence_id, flip_prob=0.2):
        B, _, T, _ = action1.shape
        device = action1.device

        # === 1. 原始结构的损失 (一次性计算) ===
        with torch.no_grad():
            loss_orig, _ = self.target_pred(input1, input2, mask1, mask2, action1, action2, sentence_id)
            loss_orig1 = loss_orig[1]  # (B,)
            loss_orig2 = loss_orig[2]

         # 初始化结果
        token_rewards1 = torch.zeros(B, T, device=device)
        token_rewards2 = torch.zeros(B, T, device=device)

        # 构造扰动结构：只改变第 i 行
        for i in range(T):
            # 当前 token 对应的结构扰动：只翻转第 i 行的某些位置
            flip_mask = (torch.rand(B, T, device=device) < flip_prob).float()  # shape: (B, T)

            # 克隆原始结构
            perturbed_action1 = action1.clone()  # shape: (B, 1, T, T)

            # 对第 i 行执行 XOR 翻转（注意：仅第 i 行被修改）
            perturbed_action1[:, 0, i, :] = (perturbed_action1[:, 0, i, :] + flip_mask) % 2

            # 结构输入变了，forward 一次
            with torch.no_grad():
                loss_perturbed, _ = self.target_pred(input1, input2, mask1, mask2, perturbed_action1, action2, sentence_id)
                loss_perturbed1 = loss_perturbed[1]  # shape: (B,)
                loss_perturbed2 = loss_perturbed[2]

            # reward = 原始 - 扰动后（越变差 reward 越大）
            token_rewards1[:, i] = loss_orig1 - loss_perturbed1
            token_rewards2[:, i] = loss_orig2 - loss_perturbed2

        return token_rewards1, token_rewards2
