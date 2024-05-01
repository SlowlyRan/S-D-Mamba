import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np



class Transformer(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Transformer, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        if configs.channel_independence:
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
        else:
            self.enc_in = configs.enc_in
            self.dec_in = configs.dec_in
            self.c_out = configs.c_out

        # Embedding
        self.enc_embedding = DataEmbedding(self.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.dec_embedding = DataEmbedding(self.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True))

        self.fc1 = nn.Linear(configs.pred_len * configs.dec_in, configs.pred_len, bias=True)
        self.activation = F.gelu
        self.fc2 = nn.Linear(configs.pred_len * 2, configs.d_model, bias=True)

        self.fc3 = nn.Linear(configs.pred_len * 4, configs.pred_len, bias=True)
        self.fc4 = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_t, y_t):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # enc_out = self.mamba(enc_out)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
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

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_t, y_t, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, x_t, y_t)
        return dec_out  # [B, L, D]
