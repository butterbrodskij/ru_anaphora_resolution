from help_processes import (
    find_anaphors,
    parsing_process
)

import numpy as np
from generate_model import make_model
import os
import pickle
import sys

def find_best_cand(y, words):
    return words[np.argmax(y)].lemma

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        if not (model_name in ["tree", "svm", "rf", "gb"]):
            model_name = "gb"
    else:
        model_name = "gb"

    if not (model_name + ".pickle" in os.listdir()):
        with open(model_name + '.pickle', 'wb') as f:
            pickle.dump(make_model(model_name), f)

    with open(model_name + '.pickle', 'rb') as f:
        model = pickle.load(f)

    text = "Компьютерные лингвисты из разных стран посетили консультацию. Они делали подробные конспекты."
    text = "Дом стоял тёмный и молчаливый, огня в нем не было."
    text = "В зоопарке люди кормили животных фруктами, они выглядели счастливыми."
    text = "В зоопарке люди кормили животных фруктами, они выглядели сочными."
    text = "В зоопарке дети кормили животных фруктами, они выглядели вкусными."

    predictions = {}
    doc = parsing_process(text)
    anaphs = find_anaphors(doc)

    for word, x, candidates in anaphs:
        predictions[word.start] = find_best_cand(model.predict(x), candidates)
        print(x)

        print(f'anaphora: {word.text} [{word.start}]')

        for pred, score in zip(candidates, model.predict(x)):
            print(f'{pred.lemma}: {score}')

        print()

    for sen in doc.sents:
        for i in sen.tokens:
            if i.start in predictions:
                print(i.text, "[" + predictions[i.start] + "]", end=' ')
            elif i.rel == 'punct':
                print(f'\b{i.text}', end=' ')
            else:
                print(i.text, end=' ')

        print()