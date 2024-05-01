import torch
import torch.nn as nn



class ModelMLP2(nn.Module):
    def __init__(self, configs):
        super(ModelMLP2, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.fc01 = nn.Linear(configs.seq_len * configs.dec_in, configs.d_model, bias=True)
        self.fc02 = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        self.fc1 = nn.Linear(configs.pred_len * (configs.dec_in - 1), configs.pred_len, bias=True)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(configs.pred_len * 3, configs.d_model, bias=True)

        self.fc3 = nn.Linear(configs.pred_len * 4, configs.pred_len, bias=True)
        self.fc4 = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_t, y_t):
        z = x_enc.reshape([x_enc.shape[0], -1])
        z = self.fc01(z)
        z = self.fc02(z)
        x = x_t.reshape([x_t.shape[0], -1])
        x = self.fc1(x)
        x = self.activation(x)
        y = y_t.reshape(x.shape[0], -1)
        y = self.fc3(y)
        y = self.activation(y)
        x = torch.cat((x, y, z), dim=-1)
        x = self.fc2(x)
        x = self.activation(x)
        dec_out = self.fc4(x)
        # print("out",dec_out.shape)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_t, y_t, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, x_t, y_t)

        return dec_out  # [B, L, D]