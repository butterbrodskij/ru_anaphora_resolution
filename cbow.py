import random
import torch
import numpy as np
import re
import pymorphy2
import os
from torch.nn import Embedding, Linear
import pickle
morph = pymorphy2.MorphAnalyzer()

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_traindata():
    res = ''

    for file_name in sorted(os.listdir(r"train/texts")):
        file_name = r"train/texts/" + file_name

        with open(file_name, 'r', encoding='utf-16') as f:
            s = f.read()
            res += s

    return res

def lemmatize(data):
    separators = ['!', '?', '(', ')', ';', ':']
    dot = '.'

    for sep in separators:
        data = data.replace(sep, dot)

    spaces = ['\n', '\t']
    space = ' '

    for sp in spaces:
        data = data.replace(sp, space)

    data = data.lower()

    data = data.split('.')

    dataset = []

    for sen in data:
        sentence = re.sub('[^A-Za-zА-ЯЁа-яё0-9]+', ' ', sen)
        sentence = sentence.split()

        sentence_lemm = [morph.parse(i)[0].normal_form for i in sentence]

        if len(sentence) > 0:
            dataset.append(sentence_lemm)

    return dataset

def get_token2idx(token_seq):
    token2idx = {}
    idx2token = {}

    token2idx['<BEG>'] = 0
    token2idx['<END>'] = 1
    token2idx['<UNK>'] = -1
    idx2token[0] = '<BEG>'
    idx2token[1] = '<END>'
    idx2token[-1] = '<UNK>'
    i = 2

    for sentence in token_seq:
        for token in sentence:
            if not(token in token2idx):
                token2idx[token] = i
                idx2token[i] = token
                i += 1

    return token2idx, idx2token

def sentence_marking(dataset, tkn2idx, context_size):
    marked_data = []
    marked_data_idx = []

    for sen in dataset:
        for i in range(context_size):
            sen = ['<BEG>'] + sen + ['<END>']

        for i in range(context_size, len(sen) - context_size):
            context = []
            context_idx = []

            for j in range(context_size, 0, -1):
                context.append(sen[i - j])
                context_idx.append(tkn2idx.get(sen[i - j], -1))

            for j in range(1, context_size + 1):
                context.append(sen[i + j])
                context_idx.append(tkn2idx.get(sen[i + j], -1))

            target = sen[i]
            marked_data.append((context, target))
            marked_data_idx.append((torch.tensor(context_idx), tkn2idx.get(target, -1)))

    return marked_data, marked_data_idx

class CBOW(torch.nn.Module):
    def __init__(
        self,
        context_size: int,
        embedding_dim: int,
        vocab_size: int
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.linear = Linear(2 * context_size * embedding_dim, vocab_size)
        self.activate = torch.nn.LogSoftmax(dim=1)

    def forward(self, tokens: torch.LongTensor):
        embed = self.embedding(tokens).reshape(1, -1)
        vec = self.linear(embed)

        return self.activate(vec)[0]

def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    train_data,
    epoch: int
) -> None:
    model.train()

    epoch_loss = []

    for i, (tokens, res) in enumerate(
        train_data
    ):
        tokens = tokens.to(device)

        optimizer.zero_grad()
        output = model(tokens).to(device)
        outputs = torch.zeros_like(output).to(device)
        outputs[res] = 1
        loss = criterion(output, outputs)
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

    avg_loss = np.mean(epoch_loss)
    print(f"Train loss: {avg_loss}\n")

def train(
    n_epochs: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    train_data
) -> None:
    for epoch in range(n_epochs):

        print(f"Epoch [{epoch+1} / {n_epochs}]\n")

        train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train_data=train_data,
            epoch=epoch
        )

if __name__ == '__main__':
    set_global_seed(42)

    traindata = get_traindata()
    dataset = lemmatize(traindata)
    tkn2idx, idx2tkn = get_token2idx(dataset)
    vocab_size = len(tkn2idx)
    embed_dim = 100
    context_size = 2
    print(f'different word count: {vocab_size}')
    marked_data, marked_data_idx = sentence_marking(dataset, tkn2idx, context_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    model = CBOW(context_size, embed_dim, vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    criterion = torch.nn.CrossEntropyLoss()

    train(
        n_epochs=20,
        model=model.to(device),
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_data=marked_data_idx
    )

    with open('cbow.pickle', 'wb') as f:
        pickle.dump(model, f)

    with open('tkn2idx.pickle', 'wb') as f:
        pickle.dump(tkn2idx, f)