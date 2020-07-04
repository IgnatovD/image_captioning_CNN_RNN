import torch
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

def get_blue(logits, captions, tokenizer):
    predict = torch.argmax(logits, -1)

    sentences = []
    targets = []

    for i in range(predict.shape[-1]):
        sentence_ids = predict[:, i].tolist()
        sentence = tokenizer.decode(sentence_ids)
        target_ids = captions[:, i].tolist()
        target = tokenizer.decode(target_ids)
        sentences.append([sentence])
        targets.append(target)

    return corpus_bleu(sentences, targets, smoothing_function=SmoothingFunction().method1)


def train(model, dataloader, optimizer, criterion, clip, device, tokenizer, captions):
    model.train()
    epoch_loss = 0.
    blue = 0.

    for batch in dataloader:
        # [bs, image_vec]
        image_vectors = batch['input_ids'].to(device)
        # [bs, len_seq]
        captions = batch['outputs'].to(device)
        # [len_seq, bs]
        captions = torch.transpose(captions, 1, 0)

        optimizer.zero_grad()

        # [seq_len, bs, vocab_size]
        logits = model(image_vectors, captions)

        blue += get_blue(logits, captions, tokenizer)

        # [len_seq - 1 * bs, vocab_size]
        logits = logits[1:].contiguous().view(-1, logits.shape[-1])
        # [len_seq-1, * bs]
        captions = captions[1:].contiguous().view(-1)

        loss = criterion(logits, captions)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    train_loss = round((epoch_loss / len(dataloader)), 3)

    blue_mean = round((blue / len(dataloader)), 3) * 100

    return train_loss, blue_mean


def evaluate(model, dataloader, criterion, device, tokenizer, captions):
    model.eval()
    epoch_loss = 0
    blue = 0.

    with torch.no_grad():
        for batch in dataloader:
            image_vectors = batch['input_ids'].to(device)
            captions = batch['outputs'].to(device)
            captions = torch.transpose(captions, 1, 0)

            logits = model(image_vectors, captions)

            blue += get_blue(logits, captions, tokenizer)

            logits = logits[1:].contiguous().view(-1, logits.shape[-1])

            captions = captions[1:].contiguous().view(-1)

            loss = criterion(logits, captions)

            epoch_loss += loss.item()

        valid_loss = round((epoch_loss / len(dataloader)), 3)

        blue_mean = round((blue / len(dataloader)), 3) * 100

    return valid_loss, blue_mean