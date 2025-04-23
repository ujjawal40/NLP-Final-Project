import torch
import torch.nn as nn
import torch.nn.functional as F


class PubMedLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=3,
                 num_layers=2, drop_prob=0.5, bidirectional=True, use_attention=True):
        super(PubMedLSTMClassifier, self).__init__()

        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=drop_prob if num_layers > 1 else 0,  # No dropout for single layer
            batch_first=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * self.num_directions, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )

        # Output layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def forward(self, x):
        # Embedding layer
        embeds = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        # LSTM layer
        lstm_out, _ = self.lstm(embeds)  # [batch_size, seq_len, hidden_dim * num_directions]

        # Attention mechanism
        if self.use_attention:
            # Calculate attention weights
            attn_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
            attn_weights = F.softmax(attn_weights, dim=1)

            # Apply attention weights
            context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch_size, hidden_dim * num_directions]
        else:
            # Use last hidden state
            context = lstm_out[:, -1, :]

        # Dropout and output
        output = self.dropout(context)
        logits = self.fc(output)
        return F.log_softmax(logits, dim=1)
