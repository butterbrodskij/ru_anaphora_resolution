from help_processes import (
    find_anaphors,
    parsing_process
)
import os
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

def make_Xy(dir):
    print("-------------------------")
    vectors = []
    targets = np.array([])
    it = 1

    for file_name in sorted(os.listdir(dir + "/texts")):
        print(it, "out of", len(os.listdir(dir + "/texts")), "files processed")
        it += 1
        text_file = dir + "/texts/" + file_name
        mark_file = dir + "/marks/" + file_name
        text = open(text_file, encoding='utf-16').read()
        mark = open(mark_file, encoding='utf-16').read().split('\n')
        mark_map = {}

        for i in mark:
            s = i.split()

            if s[1] != "-":
                mark_map[int(s[0])] = s[1:]

        doc = parsing_process(text)
        anaph = find_anaphors(doc)

        for i in anaph:
            word = i[0]
            vecs = i[1]
            lems = [j.lemma for j in i[2]]

            if word.start in mark_map:
                target_lems = mark_map[word.start]
                
                for lem in lems:
                    if lem in target_lems:
                        targets = np.append(targets, 1)
                    else:
                        targets = np.append(targets, 0)

                vectors += vecs

    vectors = np.array(vectors)
    print("Done")
    print("-------------------------")
    return vectors, targets

def make_model(model):
    print("Training files processing")
    X, y = make_Xy("train")
    print("got", len(X), "training vectors")

    if model == "tree":
        regressor = DecisionTreeRegressor(random_state=42)
    elif model == "svm":
        regressor = svm.SVR()
    elif model == "rf":
        regressor = RandomForestRegressor(random_state=42)
    else:
        regressor = GradientBoostingRegressor(random_state=42)

    print()
    print("teaching regressor")
    regressor.fit(X, y)
    print("Done")
    print()

    return regressor

