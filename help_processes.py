from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    PER,
    NamesExtractor,
    Doc
)
import os
import numpy as np
from cbow import CBOW
import torch
import transformers

model_name = "DeepPavlov/rubert-base-cased"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = transformers.AutoModel.from_pretrained(model_name).to(device)
embedding_matrix = model.embeddings.word_embeddings.weight.to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
cos = torch.nn.CosineSimilarity()

def hand_teach_model():
    with open("ForTraining/tmp.txt", 'w', encoding='utf-16') as res_file:
        for file_name in sorted(os.listdir("ForTraining/Texts")):
            res_file.write(f'{file_name}\n')
            res_file.write("-------\n")
            file_name = "ForTraining/Texts/" + file_name

            with open(file_name, 'r', encoding='utf-16') as f:
                s = f.read()
                doc = parsing_process(s)
                for j in find_anaphors(doc):
                    res_file.write(f'{j[0].start}\n')

                    for vec, w in zip(j[1], j[2]):
                        res_file.write(f'{w.lemma}: {vec}\n')

                    res_file.write('\n')

                res_file.write("-------\n")

def hand_test_model():
    with open("ForTest/tmp.txt", 'w', encoding='utf-16') as res_file:
        for file_name in sorted(os.listdir("ForTest/Texts")):
            res_file.write(f'{file_name}\n')
            res_file.write("-------\n")
            file_name = "ForTest/Texts/" + file_name

            with open(file_name, encoding='utf-16') as f:
                s = f.read()
                doc = parsing_process(s)
                for j in find_anaphors(doc):
                    res_file.write(f'{j[0].start}\n')

                    for vec, w in zip(j[1], j[2]):
                        res_file.write(f'{w.lemma}: {vec}\n')

                    res_file.write('\n')

                res_file.write("-------\n")

def delete_not_in_scope():
    for file_name in sorted(os.listdir("ForTraining/morph")):
        if not (file_name in os.listdir("ForTraining/Texts")):
            os.remove("ForTraining/morph/" + file_name)

    for file_name in sorted(os.listdir("ForTest/morph")):
        if not (file_name in os.listdir("ForTest/Texts")):
            os.remove("ForTest/morph/" + file_name)

def get_word_dist(can, word, l):
    return l.index(word) - l.index(can)

def get_sent_dist(s1, s2):
    sen1 = int(s1.split('_')[0])
    sen2 = int(s2.split('_')[0])
    return sen2 - sen1

def get_tree(id, id_to_head_id):
    tree = []

    while id_to_head_id.get(id, 0) != 0 and not(id in tree):
        tree.append(id)
        id = id_to_head_id[id]

    return tree

def get_clause_dist(can, word, sen):
    id_to_head_id = {}

    for i in sen:
        id_to_head_id[i.id] = i.head_id

    can_tree = get_tree(can.id, id_to_head_id)
    word_tree = get_tree(word.id, id_to_head_id)

    if word.id in can_tree:
        return can_tree.index(word.id)
    
    if can.id in word_tree:
        return word_tree.index(can.id)
    
    tree_set_intersect = list(set(can_tree) & set(word_tree))

    if len(tree_set_intersect) > 0:
        return np.min([can_tree.index(i) + word_tree.index(i) for i in tree_set_intersect])

    return get_sent_dist(can.id, word.id) + len(word_tree) + len(can_tree)

def semantic_dist_bert(can, word, sen):
    with torch.no_grad():
        tokens = tokenizer(sen, is_split_into_words=True, return_tensors="pt").to(device)
        mask_id = tokens.input_ids == 103
        mask_pos = mask_id.nonzero()[0][1]

        res = model(**tokens, output_hidden_states=True).last_hidden_state[:,mask_pos,:]

        token = tokenizer(can, return_tensors="pt")
        embed = torch.mean(embedding_matrix[token.input_ids[0, 1:-1]], dim=0, keepdim=True)
        
        return cos(embed, res).item()
    
def semantic_dist_bert_context(can, word, sen, sen_unk):
    with torch.no_grad():
        tokens = tokenizer(sen, is_split_into_words=True, return_tensors="pt").to(device)
        tokens_unk = tokenizer(sen_unk, is_split_into_words=True, return_tensors="pt").to(device)
        mask_id = tokens.input_ids == 103
        unk_id = tokens_unk.input_ids == 100
        mask_pos = mask_id.nonzero()[0][1]
        unk_pos = unk_id.nonzero()[0][1]

        last_hidden_state = model(**tokens, output_hidden_states=True).last_hidden_state

        res = last_hidden_state[:,mask_pos,:]

        token = tokenizer(can, return_tensors="pt")
        can_len = len(token.input_ids[0]) - 2
        embed = 0

        for i in range(can_len):
            embed += last_hidden_state[:,unk_pos,:]
            unk_pos += 1
        
        embed /= can_len
        
        return cos(embed, res).item()

def semantic_dist_cbow(can, word, sen):
    import pickle
    with open('cbow.pickle', 'rb') as f:
        model = pickle.load(f)

    with open('tkn2idx.pickle', 'rb') as f:
        tkn2idx = pickle.load(f)

    context_size = 2

    prevprev = '<BEG>'
    prev = '<BEG>'

    for i in range(len(sen)):
        if sen[i] == word:
            break
        
        prevprev = prev
        prev = sen[i].text

    i += 1

    if i == len(sen):
        nextnext = '<END>'
        next = '<END>'
    elif i == len(sen) - 1:
        nextnext = '<END>'
        next = sen[i]
    else:
        nextnext = sen[i + 1]
        next = sen[i]

    sent = [prevprev, prev, next, nextnext]

    vec = torch.tensor([tkn2idx[w] for w in sent]).to('cuda')
    can_idx = tkn2idx[can]
    res = model(vec)

    return res[can_idx]
    

def make_vector(can, word, cur, prev, dist_im):
    vec = np.zeros(15)

    vec[0] = len(can.text) #длина слова-кандидата
    vec[1] = word.start - can.start #расстояние между кандидатом и местоимением в символах
    vec[2] = get_word_dist(can, word, prev + cur) #расстояние между кандидатом и местоимением в токенах
    vec[3] = dist_im #расстояние между кандидатом и местоимением в кандидатах между ними
    vec[4] = get_sent_dist(can.id, word.id) #расстояние между кандидатом и местоимением в предложениях
    vec[5] = 0 if can.pos == "NOUN" else 1 #признак типа кандидата, 0 - для существительного, 1 - для собственного

    if (not ("Number" in can.feats)) or (can.feats["Number"] == word.feats["Number"]):
        vec[6] = 1 #согласование по числу

    if not ("Gender" in can.feats):
        vec[7] = 1 #согласование по роду
    elif word.feats["Number"] == "Plur":
        vec[7] = vec[6]
    elif can.feats["Gender"] == word.feats["Gender"]:
        vec[7] = 1
    elif word.feats["Case"] != "Nom" and word.feats["Gender"] == 'Masc' and can.feats["Gender"] == 'Neut':
        vec[7] = 1

    vec[8] = 1 if ("Case" in can.feats) and (can.feats["Case"] == "Nom") else 0 #признак именительного падежа
    
    vec[9] = 1 if syntactic_correct(cur, word, can) else 0 #синтаксическое согласование
    vec[10] = 1 if can.head_id == word.head_id else 0 #признак нахождения в одной клаузе
    vec[11] = get_clause_dist(can, word, prev + cur) if vec[10] == 0 else 0 #расстояние в клаузах
    vec[12] = [i.lemma for i in cur + prev].count(can.lemma) #частота встречаемости кандидата
    cur_prev = prev + cur
    sents = [i.text if i != word else '[MASK]' for i in cur_prev]
    vec[13] = semantic_dist_bert(can.text, word, sents) #семантическое согласование со статическим эмбеддингом
    sents_with_unk = [i.text if i != word and i != can else '[MASK]' if i == word else '[UNK]' for i in cur_prev]
    vec[14] = semantic_dist_bert_context(can.text, word, sents, sents_with_unk) #семантическое согласование с контекстным эмбеддингом

    return vec

def make_vectors(l, word, cur, prev):
    result = []

    for i in range(len(l)):
        result.append(make_vector(l[i], word, cur, prev, len(l) - i - 1))

    return result

def syntactic_correct(sen, anaph, antec):
    if not(antec in sen):
        return True

    if ("Case" in anaph.feats) and (anaph.feats["Case"] == "Nom") and (anaph.head_id == antec.head_id):
        return False
    
    if ("Case" in antec.feats) and (anaph.feats["Case"] != "Nom") and (anaph.head_id == antec.head_id):
        return False

    return True

def syntactic_ins_correct(sen, anaph, antec):
    return True

def find_candidates(l, word, sen):
    result = []

    for i in l:
        if i.pos == "NOUN" or i.pos == "PROPN":
             result.append(i)

    return result

def find_anaphors(doc):
    result = []
    prevprev = []
    prev = []

    for sen in doc.sents:
        cur = []

        for word in sen.tokens:
            cur.append(word)

            if word.pos == "PRON":
                if "Person" in  word.feats and word.feats["Person"] == '3':
                    candidates = find_candidates(prevprev + prev + cur, word, sen)
                    candidates_vec = make_vectors(candidates, word, sen.tokens, prevprev + prev)
                    result.append([word, candidates_vec, candidates])

        prevprev = prev
        prev = cur

    return result

def parsing_process(s):
    segmenter = Segmenter()
    morph_vocab = MorphVocab()

    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)
    ner_tagger = NewsNERTagger(emb)

    names_extractor = NamesExtractor(morph_vocab)
    doc = Doc(s)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    doc.parse_syntax(syntax_parser)
    return doc

if __name__ == "__main__":
    #delete_not_in_scope()
    #hand_teach_model()
    #hand_test_model()
    pass