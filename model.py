import torch
import torch.nn as nn

class PubMedLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=3,
                 num_layers=2, drop_prob=0.5, bidirectional=True, use_attention=True):
        super(PubMedLSTMClassifier, self).__init__()

        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=drop_prob,
            batch_first=True
        )

        self.dropout = nn.Dropout(drop_prob)

        if use_attention:
            self.attn = nn.Linear(hidden_dim * self.num_directions, 1)

        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)  # [B, T, H*2]

        if self.use_attention:
            # Attention mechanism
            attn_weights = torch.tanh(self.attn(lstm_out))      # [B, T, 1]
            attn_weights = torch.softmax(attn_weights, dim=1)   # [B, T, 1]
            context = torch.sum(attn_weights * lstm_out, dim=1) # [B, H*2]
        else:
            # Use the last output from the sequence
            context = lstm_out[:, -1, :]

        output = self.dropout(context)
        logits = self.fc(output)
        return self.softmax(logits)
