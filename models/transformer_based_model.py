
from torch import nn
from transformers import BertModel, AutoTokenizer

class TransformerForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased',patch_size=13):
        super(TransformerForSequenceClassification, self).__init__()
        self.patch_size = patch_size
        self.embedding = nn.Linear(patch_size, 768)

        pretrained_bert = BertModel.from_pretrained(pretrained_model_name)
        self.encoder = pretrained_bert.encoder
        self.pooler = pretrained_bert.pooler
        self.freeze()

        self.projector = nn.Linear(768, 128)
        self.classifier = nn.Linear(128, 1)

    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.pooler.parameters():
            param.requires_grad = False
        print("==> Parameters in transformer encoder are freezed!")

    def forward(self, seq_data, attention_mask=None):
        seq_data_emb = self.patch_embedding(seq_data)
        seq_data_emb = self.encoder(seq_data_emb, attention_mask=attention_mask).last_hidden_state
        seq_data_emb = self.pooler(seq_data_emb)
        seq_data_emb = self.projector(seq_data_emb)
        output = self.classifier(seq_data_emb)
        return output


    def patch_embedding(self, seq_data):
        N, seq_len = seq_data.size()

        padding_length = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size

        if padding_length > 0:
            seq_data = nn.functional.pad(seq_data, (0, padding_length))

        new_seq_len = seq_data.size(1)

        seq_data_patch = seq_data.view(N, new_seq_len // self.patch_size, self.patch_size)

        seq_data_patch = seq_data_patch.to(self.embedding.weight.dtype)

        seq_data_emb = self.embedding(seq_data_patch)
        return seq_data_emb

