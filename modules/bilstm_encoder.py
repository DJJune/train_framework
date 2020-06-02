import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        # self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        # self.bn = nn.BatchNorm1d(self.hidden_size * 2)
        self.layer_norm = nn.LayerNorm((self.hidden_size*2, ))
        
    def forward(self, sequence, lengths):
        # print(sequence.size())
        self.rnn.flatten_parameters()
        # max_len = sequence.size(1)

        packed_sequence = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)

        packed_h, final_h = self.rnn(packed_sequence)
        out_sequence, _ = pad_packed_sequence(packed_h, batch_first=True)

        final_h = torch.cat((final_h[0],final_h[1]),1)
        return out_sequence, final_h
