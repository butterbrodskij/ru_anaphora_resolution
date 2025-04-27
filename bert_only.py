import transformers
import csv
import torch
import numpy as np
from torch.nn import Linear
import codecs
import random
from tqdm import tqdm
from help_processes import (
    find_anaphors_clean,
    parsing_process
)
import os
import pickle
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

def set_default_settings() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(42)
    return device

class BERTAnaphora(torch.nn.Module):
    def __init__(
        self,
        bert,
        embedding_dim
    ):
        super().__init__()
        self.bert = bert
        self.hidden = Linear(embedding_dim, 300, bias=True)
        self.head = Linear(300, 2, bias=True)

    def forward(self, tokens):
        embed = self.bert(**tokens, output_hidden_states=True).last_hidden_state[:,0,:]
        vec = self.hidden(embed)
        return self.head(vec)

class TransformersCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        tokenizer_kwargs
    ):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(
        self,
        batch,
    ):
        tokens, labels = zip(*batch)
        tokens = self.tokenizer(tokens, **self.tokenizer_kwargs)

        return tokens, torch.tensor(labels).float()

class TransformersDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_seq,
        label_seq
    ):
        self.token_seq = token_seq
        self.label_seq = label_seq

    def __len__(self):
        return len(self.token_seq)

    def __getitem__(self, idx):
        tokens = self.token_seq[idx]
        labels = self.label_seq[idx]

        return tokens, labels
    
def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int
) -> None:
    model.train()

    epoch_loss = []

    for i, (tokens, labels) in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="loop over train batches",
    ):
        tokens, labels = tokens.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(tokens)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

    avg_loss = np.mean(epoch_loss)
    print(f"Train loss: {avg_loss}\n")

def train(
    n_epochs: int,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device
) -> None:
    for epoch in range(n_epochs):

        print(f"Epoch [{epoch+1} / {n_epochs}]\n")

        train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch
        )

def humanize(doc):
    s = ""
    for token in doc:
        s += token.text + " "
    return s

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def support_sentence(propn:str, candidate:str)->str:
    return f'Отсылается ли местоимение {propn} к сущности {candidate}?'

def create_dataset(folder:str):
    dataset = []
    targets = []
    it = 1

    for file_name in sorted(os.listdir(f"{folder}/texts")):
        print(it, "out of", len(os.listdir(f"{folder}/texts")), "files processed")
        it += 1
        text_file = f"{folder}/texts/" + file_name
        mark_file = f"{folder}/marks/" + file_name
        text = open(text_file, encoding='utf-16').read()
        mark = open(mark_file, encoding='utf-16').read().split('\n')
        mark_map = {}

        for i in mark:
            s = i.split()

            if s[1] != "-":
                mark_map[int(s[0])] = s[1:]

        doc = parsing_process(text)
        anaph = find_anaphors_clean(doc)

        for word, cands, sen in anaph:

            if word.start in mark_map:
                target_lems = mark_map[word.start]
                
                for c in cands:
                    dataset.append(humanize(sen) + support_sentence(word.text, c.lemma))
                    if c.lemma in target_lems:
                        targets.append(1)
                    else:
                        targets.append(0)

    return dataset, targets

if __name__ == "__main__":
    device = set_default_settings()
    model_name = 'DeepPavlov/rubert-base-cased'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    dataset, targets = create_dataset("train")

    train_dataset = TransformersDataset(
        token_seq=dataset,
        label_seq=targets,
    )

    print(train_dataset[0])

    tokenizer_kwargs = {
        "padding":                True,
        "truncation":             True,
        "max_length":             512,
        "return_tensors":         "pt",
    }

    collator = TransformersCollator(
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collator,
    )

    tokens, labels = next(iter(train_dataloader))

    tokens = tokens.to(device)
    labels = labels.to(device)
    print(tokens)
    print(labels)

    bert = transformers.AutoModel.from_pretrained(model_name)
    embedding_dim = 768
    model = BERTAnaphora(bert, embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.0]).to(device))

    train(
        n_epochs=12,
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )

    with open('bert_anaphora.pickle', 'wb') as f:
        pickle.dump(model, f)

    print("Test files processing")
    print("-------------------------")
    it = 1
    true_labels = 0
    false_labels = 0

    with torch.no_grad():
        model.eval()
        for file_name in sorted(os.listdir("test/texts")):
            print(it, "out of", len(os.listdir("test/texts")), "files processed")
            it += 1

            text_file = "test/texts/" + file_name
            mark_file = "test/marks/" + file_name

            with open(text_file, encoding='utf-16') as f:
                text = f.read()

            with open(mark_file, encoding='utf-16') as f:
                mark = f.read().split('\n')
            
            mark_map = {}

            for i in mark:
                s = i.split()

                if s[1] != "-":
                    mark_map[int(s[0])] = s[1:]

            doc = parsing_process(text)
            anaphs = find_anaphors_clean(doc)
            predictions = {}

            for word, candidates, sen in anaphs:
                for c in candidates:
                    tokens = tokenizer(humanize(sen) + support_sentence(word.text, c.lemma), **tokenizer_kwargs).to(device)
                    result = model(tokens)
                    if torch.argmax(result).item() == 1:
                        if word.start in predictions:
                            predictions[word.start].append(c.lemma)
                        else:
                            predictions[word.start] = [c.lemma]

                if word.start in predictions:
                    predictions[word.start] = set(predictions[word.start])

            for i in mark_map:
                if not i in predictions:
                    false_labels += 1
                    print(mark_map[i])
                    continue
                print(predictions[i], mark_map[i])
                if predictions[i].issubset(set(mark_map[i])):
                    true_labels += 1
                else:
                    false_labels += 1

        print("Done")
        print("-------------------------")
        print("got", true_labels + false_labels, "test marks")
        print("model accuracy:", true_labels / (true_labels + false_labels))
