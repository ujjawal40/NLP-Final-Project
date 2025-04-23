class HybridPubMedModel(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, num_classes, lstm_layers=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():  # Freeze BERT initially
            bert_output = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)
        return self.classifier(lstm_out[:, -1, :])
