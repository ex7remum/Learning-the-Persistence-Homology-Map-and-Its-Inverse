import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn.functional import relu
import torch.nn.functional as F
from torch import Tensor
import math


# Transformer classes
class Attention(nn.Module):
    # Single-head attention
    def __init__(self, embed_dim, num_heads, dropout = 0.1):
        super().__init__()
        attention_dim = embed_dim // num_heads

        self.WQ = nn.Linear(embed_dim, attention_dim, bias=False)
        self.WK = nn.Linear(embed_dim, attention_dim, bias=False)
        self.WV = nn.Linear(embed_dim, attention_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch_size, length, embed_dim)
        # mask: (batch_size, length, length)

        Q = self.WQ(query)
        K = self.WK(key)
        V = self.WV(value)
        # Q, K, V: (batch_size, length, attention_dim)

        norm_factor = math.sqrt(Q.shape[-1])
        dot_products = torch.bmm(Q, K.transpose(1, 2)) / norm_factor
        # dot_products: (batch_size, length, length)

        if mask is not None:
            dot_products = dot_products.masked_fill(mask, -math.inf)

        attention_score = nn.functional.softmax(dot_products, dim=-1)
        attention = torch.bmm(self.dropout(attention_score), V)
        # attention_score: (batch_size, length, length)
        # attention: (batch_size, length, attention_dim)

        return attention, attention_score

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0
        self.attention_heads = nn.ModuleList([Attention(embed_dim, num_heads, dropout)
                                              for _ in range(num_heads)])

        self.linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch_size, length, embed_dim)
        # mask: (batch_size, length, length)
        attentions, attention_scores = [], []

        for head in self.attention_heads:
            attention, attention_score = head(query, key, value, mask)
            attentions += [attention]
            attention_scores += [attention_score]

        attentions = torch.cat(attentions, dim=-1)
        attention_scores = torch.stack(attention_scores, dim=-1)
        # attentions: (batch_size, length, embed_dim)
        # attention_scores: (batch_size, length, length, num_heads)

        outputs = self.linear(attentions)
        outputs = self.dropout(outputs)
        # outputs: (batch_size, length, embed_dim)

        return outputs, attention_scores

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, fc_dim, num_heads, dropout = 0.0, activation = 'relu', use_skip = True):
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.skip = use_skip
        if activation == 'relu':
            self.feedforward = nn.Sequential(
                nn.Linear(embed_dim, fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_dim, embed_dim),
                nn.Dropout(dropout)
            )
        elif activation == 'gelu':
            self.feedforward = nn.Sequential(
                nn.Linear(embed_dim, fc_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fc_dim, embed_dim),
                nn.Dropout(dropout)
            )
        else:
            raise NotImplementedError
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, inputs, mask = None):
        attention, attention_score = self.self_attention(query=inputs, key=inputs,
                                                         value=inputs, mask=mask)
        outputs = inputs + attention
        outputs = self.norm1(outputs)
        outputs = outputs + self.feedforward(outputs)
        outputs = self.norm2(outputs)
        if self.skip:
            return outputs + inputs, attention_score
        else:
            return outputs, attention_score

def create_padding_mask(mask: Tensor):
    # tokens: (batch_size, length)
    length = mask.shape[-1]
    padding_mask = (mask == 0)
    padding_mask = padding_mask.unsqueeze(1).repeat(1, length, 1)
    # padding_mask: (batch_size, length, length)

    return padding_mask

class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, width: int, nb_layers: int, skip=1, bias=True,
                 dim_in_2: int=None, modulation: str = '+'):
        """
        Args:
            dim_in: input dimension
            dim_out: output dimension
            width: hidden width
            nb_layers: number of layers
            skip: jump from residual connections
            bias: indicates presence of bias
            modulation (str): "+", "*" or "film". Used only if  dim_in_2 is not None (2 inputs to MLP)
        """
        super(MLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.width = width
        self.nb_layers = nb_layers
        self.modulation = modulation
        self.hidden = nn.ModuleList()
        self.lin1 = nn.Linear(self.dim_in, width, bias)
        if dim_in_2 is not None:
            self.lin2 = nn.Linear(dim_in_2, width)
            if modulation == 'film':
                self.lin3 = nn.Linear(dim_in_2, width)
        self.skip = skip
        self.residual_start = dim_in == width
        self.residual_end = dim_out == width
        for i in range(nb_layers-2):
            self.hidden.append(nn.Linear(width, width, bias))
        self.lin_final = nn.Linear(width, dim_out, bias)

    def forward(self, x: Tensor, y: Tensor=None):
        """
        MLP is overloaded to be able to take two arguments.
        This is used in the first layer of the decoder to merge the set and the latent vector
        Args:
            x: a tensor with last dimension equals to dim_in
        """
        out = self.lin1(x)
        if y is not None:
            out2 = self.lin2(y)
            if self.modulation == '+':
                out = out + out2
            elif self.modulation == '*':
                out = out * out2
            elif self.modulation == 'film':
                out3 = self.lin3(y)
                out = out * torch.sigmoid(out2) + out3
            else:
                raise ValueError(f"Unknown modulation parameter: {self.modulation}")
        out = F.relu(out) + (x if self.residual_start else 0)
        for layer in self.hidden:
            out = out + layer(F.relu(out))
        out = self.lin_final(F.relu(out)) + (out if self.residual_end else 0)
        return out