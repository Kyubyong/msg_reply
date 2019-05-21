import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertForSequenceClassification
from hparams import hp

class Net(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                  num_labels=n_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        '''
        x: (N, T). int64

        Returns
        logits: (N, n_classes)
        y_hat: (N, n_candidates)
        y_hat_prob: (N, n_candidates)

        '''
        if self.training:
            self.bert.train()
            logits = self.bert(x)
        else:
            self.bert.eval()
            with torch.no_grad():
                logits = self.bert(x)

        activated = self.softmax(logits)
        y_hat_prob, y_hat = activated.sort(-1, descending=True)
        y_hat_prob = y_hat_prob[:, :hp.n_candidates]
        y_hat = y_hat[:, :hp.n_candidates]

        return logits, y_hat, y_hat_prob

