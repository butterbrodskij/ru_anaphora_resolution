from help_processes import parsing_process
from collections import deque
import os

def is_3rd_person_pronoun(token):
    return (
        token.pos == "PRON"
        and token.feats.get("Person") == "3"
        and token.lemma not in ["это", "то"]
    )

def is_possible_antecedent(token, pronoun):
    if token.pos not in ["NOUN", "PROPN"]:
        return False
    return (token.start < pronoun.start and
        (token.feats.get("Gender") == pronoun.feats.get("Gender") or token.feats.get("Number") == 'Plur' and pronoun.feats.get("Number") == 'Plur')
        and token.feats.get("Number") == pronoun.feats.get("Number")
    )

def get_token_by_id(doc, token_id):
    for token in doc.tokens:
        if token.id == token_id:
            return token
    return None

def hobbs_algorithm(doc, pronoun):
    current_token = pronoun
    visited_nodes = set()

    while current_token.head_id[current_token.head_id.find('_') + 1:] != "0":
        current_token = get_token_by_id(doc, current_token.head_id)
        if current_token is None or current_token.id in visited_nodes:
            break

        if is_possible_antecedent(current_token, pronoun):
            return current_token.lemma

        visited_nodes.add(current_token.id)

    queue = deque()
    visited = set()
    
    root_tokens = [token for token in doc.tokens if token.rel == "root"]
    for token in root_tokens:
        queue.append(token)
        visited.add(token.id)
    
    while queue:
        current_token = queue.popleft()
        
        if is_possible_antecedent(current_token, pronoun):
            return current_token.lemma
        
        for child in doc.tokens:
            if child.head_id == current_token.id and child.id not in visited:
                queue.append(child)
                visited.add(child.id)
    
    return None

def bfs_search_antecedent(doc, pronoun):
    queue = deque()
    visited = set()
    
    root_tokens = [token for token in doc.tokens if token.rel == "root"]
    for token in root_tokens:
        queue.append(token)
        visited.add(token.id)
    
    while queue:
        current_token = queue.popleft()
        
        if is_possible_antecedent(current_token, pronoun):
            return current_token.lemma
        
        for child in doc.tokens:
            if child.head_id == current_token.id and child.id not in visited:
                queue.append(child)
                visited.add(child.id)
    
    return None

def resolve_anaphora(text):
    """Улучшенное разрешение анафоры с межпредложенным поиском."""
    doc = parsing_process(text)
    sentences = list(doc.sents)

    m = {}
    
    for i in range(len(sentences)):
        sent = sentences[i]
        for token in sent.tokens:
            if is_3rd_person_pronoun(token):
                # 1. Поиск в текущем предложении (Хоббс)
                referent = hobbs_algorithm(sent, token)
                
                # 2. Если не найдено, ищем в предыдущем предложении (BFS)
                if not referent and i > 0:
                    referent = bfs_search_antecedent(sentences[i-1], token)
                
                m[token.start] = referent

    return m

if __name__ == "__main__":
    texts = [
        "Иван увидел Петра, и он засмеялся.",
        "Мария дала книгу Анне, потому что она была добрая.",
        "Собака гналась за котом, но он убежал.",
        "Компьютерные лингвисты из разных стран посетили консультацию. Они делали подробные конспекты.",
        "По словам главного редактора издания Александра Писарева, это большой плюс: все остальные издания, и уже существующие, и свежезапустившиеся, рассчитывали на рекламные деньги как в конце года, так и в начале следующего. Они этих денег не получат, как и Infox, но Infox на них и не рассчитывал",
    ]

    for text in texts:
        print(f"\nПредложение: '{text}'")
        ref = resolve_anaphora(text)
        print(ref)

    print("Test files processing")
    print("-------------------------")
    it = 1
    true_labels = 0
    false_labels = 0

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

        predictions = resolve_anaphora(text)

        for i in mark_map:
            print(predictions[i], mark_map[i])
            if predictions[i] in mark_map[i]:
                true_labels += 1
            else:
                false_labels += 1

    print("Done")
    print("-------------------------")
    print("got", true_labels + false_labels, "test marks")
    print("model accuracy:", true_labels / (true_labels + false_labels))

# резюме: чек статью, метод норм, но ограничен на очень хорошую разметку текста, итоговая точность составила 0.62