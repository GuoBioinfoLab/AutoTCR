"""Training-compatible BERT encoder and hidden_size -> 128 -> 32 -> 2 head."""
import torch
from torch import nn
from torch.nn import functional as F


class MultiClass(nn.Module):
    """Preserve state_dict keys and pooling from classify_model_1217.py.

    Average pooling intentionally includes padding and [CLS]/[SEP]. The first
    state is encoder layer 1 (hidden_states[1]), not the embedding state.
    """
    def __init__(self, bert_encode_model, model_config, dropout,
                 num_classes=2, pooling_type="first-last-avg"):
        super().__init__()
        if pooling_type not in {"cls", "last-avg", "first-last-avg"}:
            raise ValueError("Supported pooling: cls, last-avg, first-last-avg. Legacy pooler is not reproducible.")
        self.bert = bert_encode_model
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(model_config.hidden_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.pooling = pooling_type

    def forward(self, batch_token, batch_segment, batch_attention_mask):
        states = self.bert(batch_token, attention_mask=batch_attention_mask,
                           token_type_ids=batch_segment, output_hidden_states=True, return_dict=True)
        if self.pooling == "cls":
            pooled = states.last_hidden_state[:, 0, :]
        elif self.pooling == "last-avg":
            last = states.last_hidden_state.transpose(1, 2)
            pooled = F.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
        else:
            first = states.hidden_states[1].transpose(1, 2)
            last = states.hidden_states[-1].transpose(1, 2)
            first_avg = F.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)
            last_avg = F.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)
            pooled = F.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)
        x = self.dropout(torch.relu(self.fc1(pooled)))
        x = self.dropout(torch.relu(self.fc2(x)))
        return self.fc3(x)
