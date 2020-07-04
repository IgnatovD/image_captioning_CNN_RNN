import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, hid_dim=512, n_layers=2, cnn_feature_dim=2048):
        super(Encoder, self).__init__()

        self.n_layers = n_layers
        self.cnn2h0 = nn.Linear(cnn_feature_dim, hid_dim)
        self.cnn2c0 = nn.Linear(cnn_feature_dim, hid_dim)

    def forward(self, image_vectors):
        # cnn2h0: [bs, hid_dim]
        # unsqueeze(0).repeat: [n_layers, bs, hid_dim]
        # The last batch of the dataloader has a different size. The remnants did not form a complete batch.
        initial_hid = self.cnn2h0(image_vectors).unsqueeze(0).repeat(self.n_layers, 1, 1)
        initial_cell = self.cnn2c0(image_vectors).unsqueeze(0).repeat(self.n_layers, 1, 1)
        return initial_hid, initial_cell


class CaptionNet(nn.Module):
    def __init__(self, emb_dim=256, hid_dim=512, n_layers=2, cnn_feature_dim=2048, dropout=0.3, vocab_size=30000):
        super(CaptionNet, self).__init__()

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers)

        self.fc1 = nn.Linear(hid_dim, hid_dim * 2)
        self.fc2 = nn.Linear(hid_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, captions_ix, hidden, cell):
        # [bs] >>> [1, bs]
        captions_ix = captions_ix.unsqueeze(0)
        # [1, bs, emb_dim]
        captions_emb = self.embedding(captions_ix)

        captions_emb = self.dropout(captions_emb)

        # outputs: [1, bs, hid_dim]
        # hidden & cell: [n_layers, bs, hid_dim]
        outputs, (hidden, cell) = self.lstm(captions_emb, (hidden, cell))

        # [bs, vocab_size]
        logits = self.fc2(F.relu( self.fc1(outputs.squeeze()) ))

        return logits, hidden, cell

class ImgCap(nn.Module):
    def __init__(self, encoder, decoder, device, max_len):
        super(ImgCap, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_len = max_len
        self.vocab_size = self.decoder.vocab_size

    def forward(self, image_vectors, captions, teacher_forcing_ratio=0.5):

        batch_size = image_vectors.shape[0]
        seq_len = captions.shape[0]

        # [seq_len, bs, vocab_size]
        outputs = torch.zeros(seq_len, batch_size, self.vocab_size).to(self.device)
        # hidden and cell: [n_layers, bs, hid_dim]
        hidden, cell = self.encoder(image_vectors)
        # first tokens of sentences: [bs]
        input = captions[0]

        for t in range(1, seq_len):
            # output: [bs, vocab_size]
            # hidden and cell: [n_layers, bs, hid_dim]
            output, hidden, cell = self.decoder(input, hidden, cell)
            # outputs[t]: [bs, vocab_size]
            outputs[t] = output
            # random.random() - random number from 0 to 1
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (captions[t] if teacher_force else top1)
        return outputs

    def generate_one_example(self, image, inception, tokenizer):

        image = torch.tensor(image.transpose([2, 0, 1]), dtype=torch.float32)

        vectors_8x8, vectors_neck, logits = inception(image[None])

        outputs = []

        image_vectors = vectors_neck.to(self.device)

        hidden, cell = self.encoder(image_vectors)

        input = torch.tensor([tokenizer.token_to_id('[CLS]')]).to(self.device)

        for t in range(1, self.max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            top1 = output.max(0)[1]
            outputs.append(top1)
            input = (top1.unsqueeze(0))

        EOS_IDX = tokenizer.token_to_id('[SEP]')

        for t in outputs:
            if t.item() != EOS_IDX:
                print(tokenizer.id_to_token(t.item()), end=' ')
            else:
                break