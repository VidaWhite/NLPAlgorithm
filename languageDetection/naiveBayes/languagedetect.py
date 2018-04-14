#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Author: VidaWhite
# Source: 某网络课程稍作修改
# Date: 2018/4/13
# Description: 语言检测

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

class LanguageDetector():

    def __init__(self, classifier=MultinomialNB()):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=1000, preprocessor=self._remove_noise)

    def _remove_noise(self, document):
        noise_pattern = re.compile("|".join(["http\S+", "\@\w+", "\#\w+"]))
        clean_text = re.sub(noise_pattern, "", document)
        return clean_text

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)


if __name__ == '__main__':
    in_f = open('data.csv')
    lines = in_f.readlines()
    in_f.close()
    dataset = [(line.strip().split(',')[0], line.strip().split(',')[1]) for line in lines]
    x, y = zip(*dataset)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    language_detector = LanguageDetector()
    language_detector.fit(x_train, y_train)
    print(language_detector.predict('This is an English sentence'))
    print(language_detector.score(x_test, y_test))
