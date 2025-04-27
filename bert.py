import transformers
import torch

from help_processes import (
    find_anaphors,
    parsing_process
)

model_name = "DeepPavlov/rubert-base-cased"

model = transformers.AutoModel.from_pretrained(model_name)
embedding_matrix = model.embeddings.word_embeddings.weight
print(embedding_matrix.shape)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

text = "Компьютерные лингвисты из разных стран посетили консультацию. Они делали подробные конспекты."
s = parsing_process(text)
word = s.tokens[8]
sents = [i.text if i != word else '[MASK]' for i in s.tokens]

tokens = tokenizer(sents, is_split_into_words=True, return_tensors="pt")
print(tokens)
mask_id = tokens.input_ids == 103
mask_pos = mask_id.nonzero()[0][1]

res = model(**tokens, output_hidden_states=True).last_hidden_state[:,mask_pos,:]
cos = torch.nn.CosineSimilarity()

token = tokenizer('лингвисты', return_tensors="pt")
embed = torch.mean(embedding_matrix[token.input_ids[0, 1:-1]], dim=0, keepdim=True)
print(cos(embed, res).item())
token = tokenizer('стран', return_tensors="pt")
embed = torch.mean(embedding_matrix[token.input_ids[0, 1:-1]], dim=0, keepdim=True)
print(cos(embed, res).item())
token = tokenizer('консультацию', return_tensors="pt")
embed = torch.mean(embedding_matrix[token.input_ids[0, 1:-1]], dim=0, keepdim=True)
print(cos(embed, res).item())