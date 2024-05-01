import torch
import torch.nn as nn
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted
import torch.nn.functional as F

from mamba_ssm import Mamba


class Model3(nn.Module):
    def __init__(self, configs):
        super(Model3, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=configs.d_model,  # Model dimension d_model
                        d_state=configs.d_state,  # SSM state expansion factor
                        d_conv=2,  # Local convolution width
                        expand=1,  # Block expansion factor)
                    ),
                    Mamba(
                        d_model=configs.d_model,  # Model dimension d_model
                        d_state=configs.d_state,  # SSM state expansion factor
                        d_conv=2,  # Local convolution width
                        expand=1,  # Block expansion factor)
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        self.fc1 = nn.Linear(configs.pred_len * configs.dec_in, configs.pred_len, bias=True)
        self.activation = F.gelu
        self.fc2 = nn.Linear(configs.pred_len * 2, configs.d_model, bias=True)

        self.fc3 = nn.Linear(configs.pred_len * 4, configs.pred_len, bias=True)
        self.fc4 = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_t, y_t):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # print("dec",dec_out.shape)
        x = dec_out[:, -self.pred_len:, -1]
        outputs_expanded = x.unsqueeze(-1)
        merged_tensor = torch.cat((x_t, outputs_expanded), dim=-1)
        x = merged_tensor.reshape([merged_tensor.shape[0], -1])
        # print("merged",merged_tensor.shape)
        x = self.fc1(x)
        x = self.activation(x)

        y = y_t.reshape(x.shape[0], -1)
        y = self.fc3(y)
        y = self.activation(y)
        x = torch.cat((x, y), dim=-1)
        x = self.fc2(x)
        x = self.activation(x)
        dec_out = self.fc4(x)
        # print("out",dec_out.shape)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_t, y_t, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, x_t, y_t)

        return dec_out  # [B, L, D]