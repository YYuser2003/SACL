import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch_geometric.utils import softmax
from torch_scatter import scatter_add


# ============================================================================
# 工具函数
# ============================================================================

def pad(tensor, length, cuda_flag):
    """将张量填充到指定长度"""
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if cuda_flag:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if cuda_flag:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def feature_transfer(bank_s_, bank_p_, bank_sp_, seq_lengths, cuda_flag=False):
    """特征转换：将特征重新组织为序列格式"""
    input_conversation_length = torch.tensor(seq_lengths)
    start_zero = input_conversation_length.data.new(1).zero_()
    
    if cuda_flag:
        input_conversation_length = input_conversation_length.cuda()
        start_zero = start_zero.cuda()

    max_len = max(seq_lengths)
    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)
    
    # (l,b,h) 格式转换
    bank_s = torch.stack(
        [pad(bank_s_.narrow(0, s, l), max_len, cuda_flag) 
         for s, l in zip(start.data.tolist(), input_conversation_length.data.tolist())], 0
    ).transpose(0, 1) if bank_s_ is not None else None
    
    bank_p = torch.stack(
        [pad(bank_p_.narrow(0, s, l), max_len, cuda_flag) 
         for s, l in zip(start.data.tolist(), input_conversation_length.data.tolist())], 0
    ).transpose(0, 1) if bank_p_ is not None else None
    
    bank_sp = torch.stack(
        [pad(bank_sp_.narrow(0, s, l), max_len, cuda_flag) 
         for s, l in zip(start.data.tolist(), input_conversation_length.data.tolist())], 0
    ).transpose(0, 1) if bank_sp_ is not None else None
    
    return bank_s, bank_p, bank_sp


# ============================================================================
# 推理模块
# ============================================================================

class ReasonModule(nn.Module):
    """推理模块：实现多步推理机制"""
    
    def __init__(self, in_channels=200, processing_steps=0, num_layers=1):
        super(ReasonModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        
        if processing_steps > 0:
            self.lstm = nn.LSTM(self.out_channels, self.in_channels, num_layers)
            self.lstm.reset_parameters()
        
        print(self)

    def forward(self, x, bank_kg, batch, q_star):
        """前向传播"""
        if self.processing_steps <= 0: 
            return q_star

        batch_size = batch.max().item() + 1
        # 初始化隐藏状态
        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        
        for i in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)
            
            # 计算注意力权重
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            a = softmax(e, batch, num_nodes=batch_size)
            
            # 聚合特征
            r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)

        return q_star

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


# ============================================================================
# 认知网络
# ============================================================================

class CognitionNetwork(nn.Module):
    """认知网络：多轮推理模块"""
    
    def __init__(self, n_features=200, n_classes=7, dropout=0.2, cuda_flag=False, reason_steps=None):
        super(CognitionNetwork, self).__init__()
        self.cuda_flag = cuda_flag
        self.reason_flag = False  # 是否为推理模式
        
        # 推理模块配置
        if self.reason_flag:
            self.steps = reason_steps if reason_steps is not None else [0, 0]
            self.fc = nn.Linear(n_features, n_features * 2)
            self.reason_modules = nn.ModuleList([
                ReasonModule(in_channels=n_features, processing_steps=self.steps[0], num_layers=1),
                ReasonModule(in_channels=n_features, processing_steps=self.steps[1], num_layers=1)
            ])

        # 分类器配置
        self.n_features = n_features
        if not self.reason_flag:
            self.smax_fc = nn.Linear(n_features * 2, n_classes)
        else:
            self.smax_fc = nn.Linear(n_features * 4, n_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, U_s, U_p, seq_lengths):
        """前向传播"""
        batch_size = U_s.size(1)
        
        # 准备批次数据
        batch_index, context_s_, context_p_ = [], [], []
        for j in range(batch_size):
            if self.reason_flag: 
                batch_index.extend([j] * seq_lengths[j])
            context_s_.append(U_s[:seq_lengths[j], j, :])
            context_p_.append(U_p[:seq_lengths[j], j, :])

        # 合并所有序列
        if self.reason_flag: 
            batch_index = torch.tensor(batch_index)
        bank_s_ = torch.cat(context_s_, dim=0)
        bank_p_ = torch.cat(context_p_, dim=0)
        
        # 移动到GPU
        if self.cuda_flag:
            if self.reason_flag: 
                batch_index = batch_index.cuda()
            bank_s_ = bank_s_.cuda()
            bank_p_ = bank_p_.cuda()

        # 特征转换
        bank_s, bank_p, _ = feature_transfer(bank_s_, bank_p_, None, seq_lengths, self.cuda_flag)
        feature_s, feature_p = bank_s, bank_p

        # 推理处理
        if self.reason_flag:
            feature_s = self._apply_reasoning(bank_s, bank_s_, batch_index, 0)  # 情境推理
            feature_p = self._apply_reasoning(bank_p, bank_p_, batch_index, 1)  # 说话人推理

        # 特征融合和分类
        hidden = torch.cat([feature_s, feature_p], dim=-1)
        return self._classify(hidden, seq_lengths)

    def _apply_reasoning(self, bank, bank_, batch_index, module_idx):
        """应用推理模块"""
        features = []
        for t in range(bank.size(0)):
            q_star = self.fc(bank[t])
            q_reasoned = self.reason_modules[module_idx](bank_, None, batch_index, q_star)
            features.append(q_reasoned.unsqueeze(0))
        return torch.cat(features, dim=0)

    def _classify(self, hidden, seq_lengths):
        """分类输出"""
        hidden0 = self.smax_fc(self.dropout(F.relu(hidden)))
        log_prob = F.log_softmax(hidden0, 2)
        
        # 展平输出
        log_prob_flat = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        hidden_flat = torch.cat([hidden[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        
        return log_prob_flat, hidden_flat


# ============================================================================
# 主模型：DialogueCRN
# ============================================================================

class DialogueCRN(nn.Module):
    """对话上下文推理网络"""
    
    def __init__(self, base_model='LSTM', base_layer=2, input_size=None, hidden_size=None, 
                 n_speakers=2, n_classes=7, dropout=0.2, cuda_flag=False, reason_steps=None):
        super(DialogueCRN, self).__init__()
        
        # 基础配置
        self.base_model = base_model
        self.n_speakers = n_speakers
        self.base_layer = base_layer
        self.hidden_size = hidden_size
        
        # 基础编码器
        self._setup_base_encoder(input_size, hidden_size, dropout, n_classes)
        
        # 认知网络（如果不是线性基础模型）
        if self.base_model != 'Linear':
            self.cognition_net = CognitionNetwork(
                n_features=2 * hidden_size, n_classes=n_classes, dropout=dropout, 
                cuda_flag=cuda_flag, reason_steps=reason_steps
            )
        print("="*10)
        print(self)

    def _setup_base_encoder(self, input_size, hidden_size, dropout, n_classes):
        """设置基础编码器"""
        if self.base_model == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_size + 768 * 0 + 128 * 0, 
                hidden_size=hidden_size, 
                num_layers=self.base_layer, 
                bidirectional=True, 
                dropout=dropout
            )
            self.rnn_parties = nn.LSTM(
                input_size=input_size + 768 * 0 + 128 * 0, 
                hidden_size=hidden_size, 
                num_layers=self.base_layer, 
                bidirectional=True, 
                dropout=dropout
            )
            
        elif self.base_model == 'GRU':
            self.rnn = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, 
                num_layers=self.base_layer, bidirectional=True, dropout=dropout
            )
            self.rnn_parties = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, 
                num_layers=self.base_layer, bidirectional=True, dropout=dropout
            )
            
        elif self.base_model == 'Linear':
            self.base_linear = nn.Linear(input_size, hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.smax_fc = nn.Linear(hidden_size, n_classes)
            
        else:
            raise ValueError('Base model must be one of LSTM/GRU/Linear')

    def init_hidden(self, num_directs, num_layers, batch_size, d_model):
        """初始化隐藏状态"""
        return Variable(torch.zeros(num_directs * num_layers, batch_size, d_model))

    def forward(self, r1, qmask, seq_lengths):
        """前向传播"""
        U = r1
        
        if self.base_model == 'Linear':
            return self._forward_linear(U, seq_lengths)
        else:
            U_s, U_p = self._encode_sequences(U, qmask)
            return self.cognition_net(U_s, U_p, seq_lengths)

    def _forward_linear(self, U, seq_lengths):
        """线性基础模型的前向传播"""
        U = self.base_linear(U)
        U = self.dropout(F.relu(U))
        hidden = self.smax_fc(U)
        log_prob = F.log_softmax(hidden, 2)
        
        # 展平输出
        logits = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        logits2 = torch.cat([U[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        
        return logits, logits2

    def _encode_sequences(self, U, qmask):
        """编码输入序列"""
        if self.base_model == 'LSTM':
            return self._encode_lstm(U, qmask)
        elif self.base_model == 'GRU':
            return self._encode_gru(U, qmask)

    def _encode_lstm(self, U, qmask):
        """LSTM编码器"""
        # 转换维度：(b,l,h), (b,l,p)
        U_, qmask_ = U.transpose(0, 1), qmask.transpose(0, 1)
        U_p_ = torch.zeros(U_.size()[0], U_.size()[1], self.hidden_size * 2).type(U.type())
        
        # 为每个说话人创建单独的序列
        U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]
        pb_flag = torch.zeros(self.n_speakers, U_.size(0))
        pl_min = 2  # 最小序列长度
        
        # 按说话人分离序列
        for b in range(U_.size(0)):
            for p in range(len(U_parties_)):
                index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                if index_i.size(0) >= pl_min:
                    pb_flag[p][b] = 1
                    U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]

        # 编码每个说话人的序列
        for p in range(len(U_parties_)):
            index_b = torch.nonzero(pb_flag[p]).squeeze(-1)
            temp_ = U_parties_[p][index_b]
            h_temp = torch.zeros_like(U_p_).type(U_p_.type())
            h_temp[index_b] = self.rnn_parties(temp_.transpose(0, 1))[0].transpose(0, 1)

            # 重组结果
            for b in range(U_p_.size(0)):
                index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                if index_i.size(0) >= pl_min:
                    U_p_[b][index_i] = h_temp[b][:index_i.size(0)]

        U_p = U_p_.transpose(0, 1)
        U_s, _ = self.rnn(U)  # 情境编码
        
        return U_s, U_p

    def _encode_gru(self, U, qmask):
        """GRU编码器"""
        U_, qmask_ = U.transpose(0, 1), qmask.transpose(0, 1)
        U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 200).type(U.type())
        U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]
        
        # 按说话人分离序列
        for b in range(U_.size(0)):
            for p in range(len(U_parties_)):
                index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                if index_i.size(0) > 0:
                    U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
        
        # 编码每个说话人
        E_parties_ = [self.rnn_parties(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) 
                     for p in range(len(U_parties_))]

        # 重组结果
        for b in range(U_p_.size(0)):
            for p in range(len(U_parties_)):
                index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                if index_i.size(0) > 0: 
                    U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
        
        U_p = U_p_.transpose(0, 1)
        U_s, _ = self.rnn(U)  # 情境编码
        
        return U_s, U_p
