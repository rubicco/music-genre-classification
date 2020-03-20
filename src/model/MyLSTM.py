import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MyLSTM(nn.Module):
    def __init__(self,
                 emb_vectors,
                 lstm_layers,
                 hidden_dim,
                 target_size,
                 dropout_prob,
                 device,
                 seq_len=250):
        super().__init__()
        # variable definitions
        self.hidden_dim = hidden_dim
        self.n_layers = lstm_layers
        self.device = device
        # add zero tensor at index 0 for PADDING tensor.
        # with this operation we incremented all input indexes in the dataloader.
        emb_vectors = np.insert(emb_vectors, 0, [np.zeros(300)], 0)
        self.embedding_dim = emb_vectors.shape[1]
        self.word_embeddings = nn.Embedding.from_pretrained(torch.Tensor(emb_vectors))

        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim,
                            batch_first=True, num_layers=lstm_layers,
                            dropout=dropout_prob)

        self.dropout = nn.Dropout(0.1)
        self.hidden2tag = nn.Linear(hidden_dim * seq_len, target_size)
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, sentence, hidden):
        embeds = self.word_embeddings(sentence)
        x, (hid, out) = self.lstm(embeds, hidden)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.hidden2tag(x)
        #x = self.sigmoid(x)
        x = self.log_softmax(x)
        return x, (hid, out)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden

    def switch_train(self):
        self.train()