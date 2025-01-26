# ---encoding:utf-8---

import os, sys

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import config

config = config.get_train_config()


def get_attn_pad_mask(seq):
    # print("seq.size = ", seq.size())
    batch_size, seq_len = seq.size()
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    pad_attn_mask_expand = pad_attn_mask.expand(batch_size, seq_len,
                                                seq_len)  # [batch_size, seq_len, seq_len]
    # print("pad_attn_mask_expand.shape = ",pad_attn_mask_expand.shape)
    return pad_attn_mask_expand


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size,
                                      d0_model)  # token embedding (look-up table)
        self.pos_embed = nn.Embedding(max_len, d0_model)  # position embedding
        self.norm = nn.LayerNorm(d0_model)

    def forward(self, x):
        seq_len = x.size(1)  # x: [batch_size, seq_len] 37
        pos = torch.arange(seq_len, device=device, dtype=torch.long)  # [seq_len]

        # pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        pos = torch.arange(seq_len, device=device, dtype=torch.long)
        pos = pos.unsqueeze(0).repeat(x.size(0), 1)  # [seq_len] -> [batch_size, seq_len] [64,37]

        embedding = self.pos_embed(pos)  # [64,37,1024]
        embedding = embedding + self.tok_embed(x)  # [64,37,1024]+[64,37,1024]-> [64,37,1024]
        embedding = self.norm(embedding)
        return embedding


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            d_k)  # scores : [batch_size, n_head, seq_len, seq_len]
        scores.masked_fill_(attn_mask,
                            -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_head, seq_len, seq_len]
        context = torch.matmul(attn, V)  # [batch_size, n_head, seq_len, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_head)
        self.W_K = nn.Linear(d_model, d_k * n_head)
        self.W_V = nn.Linear(d_model, d_v * n_head)

        self.linear = nn.Linear(n_head * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)
        # self.dropout = nn.Dropout(config.dropout)  # 添加dropout层

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)  # Q,K,V [64,37,1280]
        # residual, batch_size = Q, Q.shape(0) # Q,K,V [64,33,1156]
        # print("Q.shape",Q.shape)
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1,
                                                                      2)  # q_s: [batch_size, n_head, seq_len, d_k] [64,8,37,32]
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1,
                                                                      2)  # k_s: [batch_size, n_head, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1,
                                                                      2)  # v_s: [batch_size, n_head, seq_len, d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        context, attention_map = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_head * d_v)  # context: [batch_size, seq_len, n_head * d_v]
        output = self.linear(context)
        output = self.norm(output + residual)  # [64,37,1280]
        # self.dropout = nn.Dropout(config.dropout)  # 添加dropout层

        return output, attention_map


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.attention_map = None
        self.dropout = nn.Dropout(config.dropout)  # 添加dropout层

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attention_map = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                        enc_self_attn_mask)
        self.attention_map = attention_map
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        enc_outputs = self.dropout(enc_outputs)  # 添加dropout操作  [64.37,1280]
        return enc_outputs


class ResNetConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True):
        super(ResNetConv1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation,
                               groups, bias)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, dilation,
                               groups, bias)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.maxpool = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.maxpool(out)

        out += identity
        out = self.relu(out)

        return out


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [Batch, Length, Channels]
        avg_out = torch.mean(x, dim=2, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=2, keepdim=True)  # 最大池化
        combined = torch.cat([avg_out, max_out], dim=2)  # 拼接池化结果
        combined = combined.permute(0, 2, 1)  # 调整维度以匹配Conv1d输入
        attention = self.conv(combined)  # 空间注意力权重
        attention = self.sigmoid(attention).permute(0, 2, 1)  # 激活并调整维度
        return x * attention  # 应用注意力权重


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 第一个卷积层（升维）
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层（保持维度）
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # 空间注意力模块
        self.spatial_attention = SpatialAttention1D(kernel_size=7)
        # 跳跃连接调整维度
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x  # 原始输入
        # 主路径处理
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 应用空间注意力（需调整维度顺序）
        out = out.permute(0, 2, 1)  # [B, L, C]
        out = self.spatial_attention(out)
        out = out.permute(0, 2, 1)  # [B, C, L]
        # 处理跳跃连接
        residual = self.shortcut(residual)
        # 残差连接与激活
        out += residual
        out = self.relu(out)
        return out


class ProteinResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义三个残差块，逐步升维
        self.block1 = ResidualBlock(in_channels=256, out_channels=512)
        self.block2 = ResidualBlock(in_channels=512, out_channels=768)
        self.block3 = ResidualBlock(in_channels=768, out_channels=1024)

    def forward(self, x):
        # 调整输入维度顺序以适应PyTorch卷积层
        x = x.permute(0, 2, 1)  # [Batch, Channels, Length]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # 恢复输出维度顺序
        x = x.permute(0, 2, 1)  # [Batch, Length, Channels]
        return x


class WeightedTriChannelCrossAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(WeightedTriChannelCrossAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # 定义查询、键、值的线性变换
        self.query_transforms = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(3)])
        self.key_transforms = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(3)])
        self.value_transforms = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(3)])

        # 定义输出变换，将加权后的值合并回原始维度
        self.output_transform = nn.Linear(3072, 1024)

    def forward(self, x1, x2, x3):
        # x1, x2, x3 形状均为 [batch, seq_len, dim]
        batch_size, seq_len, dim = x1.size()

        # 计算查询、键、值
        q1, k1, v1 = self.query_transforms[0](x1), self.key_transforms[0](x1), \
            self.value_transforms[0](x1)
        q2, k2, v2 = self.query_transforms[1](x2), self.key_transforms[1](x2), \
            self.value_transforms[1](x2)
        q3, k3, v3 = self.query_transforms[2](x3), self.key_transforms[2](x3), \
            self.value_transforms[2](x3)

        # 计算交叉注意力得分
        scores1 = torch.bmm(q1.permute(0, 2, 1), k2) + torch.bmm(q1.permute(0, 2, 1), k3)
        scores2 = torch.bmm(q2.permute(0, 2, 1), k1) + torch.bmm(q2.permute(0, 2, 1), k3)
        scores3 = torch.bmm(q3.permute(0, 2, 1), k1) + torch.bmm(q3.permute(0, 2, 1), k2)

        # 缩放得分以稳定训练
        scale = dim ** -0.5
        scores1 = scores1 * scale
        scores2 = scores2 * scale
        scores3 = scores3 * scale

        # 计算注意力权重
        attn_weights1 = F.softmax(scores1, dim=-1)
        attn_weights2 = F.softmax(scores2, dim=-1)
        attn_weights3 = F.softmax(scores3, dim=-1)

        # 应用注意力权重加权值
        weighted_v1 = torch.bmm(attn_weights1, v1)
        weighted_v2 = torch.bmm(attn_weights2, v2)
        weighted_v3 = torch.bmm(attn_weights3, v3)

        # 将加权后的值合并
        combined_weighted_values = torch.cat([weighted_v1, weighted_v2, weighted_v3],
                                             dim=-1)  # [batch, seq_len, 3*dim]

        # 通过线性层恢复原始维度
        output = self.output_transform(combined_weighted_values)  # [batch, seq_len, dim]

        # 应用dropout
        output = self.dropout(output)

        return output


from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="cuda",
            num_heads=4,
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    ),
    context_length=35,
    num_blocks=7,
    embedding_dim=1024,
    slstm_at=[0],

)


class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()

        global max_len, n_layers, n_head, d0_model, d_model, d_ff, d_k, d_v, vocab_size, device
        max_len = config.max_len
        n_layers = config.num_layer
        n_head = config.num_head
        d0_model = config.dim_embedding
        # d_model = config.dim_embedding
        d_model = config.dim_embedding + 256
        d_ff = config.dim_feedforward
        d_k = config.dim_k
        d_v = config.dim_v
        vocab_size = config.vocab_size
        if config.task == 'test':
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if config.cuda else "cpu")

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, config.num_head, config.dim_feedforward,
                                       dropout=0.1)
            for _ in range(config.num_layer)
        ])
        self.embedding = Embedding(config)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

        self.conv1dEmbed = nn.Sequential(
            # 这个函数用来把预训练embedding的维度从33变成37
            nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=4),
            nn.ReLU(),  # 86
        )

        self.conv1dStr = nn.Sequential(
            nn.Conv1d(124, 256, kernel_size=5, stride=1, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.resconv1d = nn.Sequential(
            ResNetConv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1,
                         padding=2),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.xLSTM = xLSTMBlockStack(cfg)
        self.resconv3d = ProteinResNet()
        self.attention = WeightedTriChannelCrossAttention()
        self.fc_task = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

        self.classifier = nn.Linear(2, 2)

    def forward(self, input_ids, x_embedding, str_embedding):
        str_embedding = str_embedding.to(
            torch.float32)

        # You need to permute the dimensions before passing it to the conv layer
        str_embedding = str_embedding.permute(0, 2, 1)
        output_data = self.conv1dStr(str_embedding)
        # Now permute it back to [batch_size, seq_len, feature_size]
        str_embedding = output_data.permute(0, 2, 1)
        print(f'str_embedding.shape={str_embedding.shape}')

        # Embedding 通过1dCNN维度调整
        # encoding中需要补0，所以还是需要再cnn调整成35 [64, 35, 1024]
        x_embedding = x_embedding.permute(0, 2, 1)  # 调整维度顺序输入cnn,[64,1024,31]
        x_embedding = self.conv1dEmbed(x_embedding)  # [64,1024,35]
        x_embedding = x_embedding.permute(0, 2, 1)  # [64, 35, 1024]
        print("x_embedding_conv.shape=", x_embedding.shape)

        # 这里的self.embedding是单纯的独热编码
        self_embedding = self.embedding(
            input_ids)  # [bach_size, seq_len, d_model = dim_embedding]
        print("self_embedding.shape", self_embedding.shape)  # [64,35,1024]

        input1 = self_embedding + x_embedding  # [64,35,1024]
        input2 = x_embedding  # [64,35,1024]
        input3 = str_embedding  # [64,35,256]

        # branch1-transformer-xlstm
        enc_self_attn_mask = get_attn_pad_mask(input_ids)  # [64,35,35]

        for layer in self.layers:
            output_t = layer(input1, enc_self_attn_mask)  # [64,35，1024]
        output1 = self.xLSTM(output_t)

        # branch2-CNN
        output_c = self.resconv1d(input2.permute(0, 2, 1))
        output2 = output_c.permute(0, 2, 1)  # [64,35,1024]

        # branch3-ResNet
        output3 = self.resconv3d(input3)

        # Fusion
        output = self.attention(output1, output2, output3)

        # only use [CLS]
        representation = output[:, 0, :]
        reduction_feature = self.fc_task(representation)
        reduction_feature = reduction_feature.view(reduction_feature.size(0), -1)
        logits_clsf = self.classifier(reduction_feature)

        return logits_clsf, representation
