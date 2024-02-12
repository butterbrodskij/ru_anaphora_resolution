from help_processes import (
    find_anaphors,
    parsing_process
)
from main import find_best_cand
from generate_model import make_model
import os
import pickle
import sys

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
                mark_map[int(s[0])] = s[1]

        doc = parsing_process(text)
        anaphs = find_anaphors(doc)
        predictions = {}

        for word, x, candidates in anaphs:
            predictions[word.start] = find_best_cand(model.predict(x), candidates)

        for i in mark_map:
            if mark_map[i] == predictions[i]:
                true_labels += 1
            else:
                false_labels += 1

    print("Done")
    print("-------------------------")
    print("got", true_labels + false_labels, "test marks")
    print("model accuracy:", true_labels / (true_labels + false_labels))
