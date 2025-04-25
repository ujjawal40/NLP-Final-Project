import torch
import torch.nn as nn
import torch.nn.functional as F


class PubMedLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, output_dim=3,
                 num_layers=2, drop_prob=0.4, bidirectional=True, use_attention=True):
        super().__init__()

        # Architecture configuration
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        # Embedding layer with better initialization
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)

        # Enhanced LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=drop_prob if num_layers > 1 else 0,
            batch_first=True,
            proj_size=hidden_dim // 2 if bidirectional else 0  # Better memory usage
        )

        # Improved attention mechanism
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * self.num_directions, hidden_dim),
                nn.GELU(),  # Better than Tanh for attention
                nn.Linear(hidden_dim, 1),
                nn.Dropout(drop_prob / 2)  # Regularization
            )
            nn.init.xavier_uniform_(self.attention[0].weight)
            nn.init.xavier_uniform_(self.attention[2].weight)

        # Classification layers with skip connection
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

        # Initialize classifier properly
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, input_ids, attention_mask=None):
        # Embedding layer
        x = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)

        # LSTM layer
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim * num_directions)

        # Attention mechanism
        if self.use_attention:
            batch_size, seq_len, hidden_dim = lstm_out.shape

            # Calculate attention weights
            attn_logits = self.attention(
                lstm_out.view(-1, hidden_dim)  # (batch_size*seq_len, hidden_dim)
            ).view(batch_size, seq_len, 1)  # (batch_size, seq_len, 1)

            # Apply attention mask if provided
            if attention_mask is not None:
                attn_logits = attn_logits.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)

            attn_weights = F.softmax(attn_logits, dim=1)
            context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch_size, hidden_dim)
        else:
            # Use last hidden state (with masking support)
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1)
                context = lstm_out[torch.arange(batch_size), lengths - 1]
            else:
                context = lstm_out[:, -1]

        # Final prediction
        output = self.dropout(context)
        logits = self.fc(output)

        return F.log_softmax(logits, dim=1)
    def get_attention_weights(self, input_ids):
        """Extract attention weights for visualization"""
        with torch.no_grad():
            x = self.embedding(input_ids)
            lstm_out, _ = self.lstm(x)
            attn_weights = F.softmax(self.attention(lstm_out), dim=1)
            return attn_weights.squeeze().cpu().numpy()