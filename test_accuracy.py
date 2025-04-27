from help_processes import (
    find_anaphors,
    parsing_process
)
from main import find_best_cand
from generate_model import make_model
import os
import pickle
import sys
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

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
    y_pred = np.array([])
    y_true = np.array([])

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
        anaphs = find_anaphors(doc)
        predictions = {}

        for word, x, candidates, _ in anaphs:
            y = model.predict(x)
            y_pred = np.append(y_pred, y)
            for c in candidates:
                if c.lemma in mark_map[word.start]:
                    y_true = np.append(y_true, 1)
                else:
                    y_true = np.append(y_true, 0)
            predictions[word.start] = find_best_cand(y, candidates)

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
    print("mse:", mean_squared_error(y_true, y_pred))
    print("r2:", r2_score(y_true, y_pred))
