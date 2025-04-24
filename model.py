import torch
import torch.nn as nn
import torch.nn.functional as F


class PubMedLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=3,
                 num_layers=2, drop_prob=0.5, bidirectional=True, use_attention=True):
        super().__init__()

        # Architecture configuration
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
            dropout=drop_prob if num_layers > 1 else 0,
            batch_first=True
        )

        # Attention mechanism
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * self.num_directions, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1))

            # Final classification layer
            self.dropout = nn.Dropout(drop_prob)
            self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def forward(self, input_ids):
        # Embedding layer
        x = self.embedding(input_ids)

        # LSTM outputs
        lstm_out, _ = self.lstm(x)

        # Attention processing
        if self.use_attention:
            # Calculate attention weights
            attn_weights = self.attention(lstm_out)
            attn_weights = F.softmax(attn_weights, dim=1)

            # Apply attention
            context = torch.sum(attn_weights * lstm_out, dim=1)
        else:
            # Use last hidden state
            context = lstm_out[:, -1, :]

        # Final prediction
        output = self.dropout(context)
        logits = self.fc(output)
        return F.log_softmax(logits, dim=1)