import torch
import torch.nn as nn



class ModelLSTM(nn.Module):
    def __init__(self, configs):
        super(ModelLSTM, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.lstm = nn.LSTM(input_size=configs.dec_in,
                            hidden_size=configs.d_model,
                            batch_first=True)
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        self.fc1 = nn.Linear(configs.pred_len * configs.dec_in, configs.pred_len, bias=True)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(configs.pred_len * 2, configs.d_model, bias=True)

        self.fc3 = nn.Linear(configs.pred_len * 4, configs.pred_len, bias=True)
        self.fc4 = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_t, y_t):
        x, _ = self.lstm(x_enc)
        # B N E -> B N S -> B S N
        enc_out = x[:, -1, :]
        dec_out = self.projector(enc_out)  # filter the covariates
        x = dec_out[:, -self.pred_len:]
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