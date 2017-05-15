import glob
import random
import re
from collections import Counter

from src.naive_bayes import NaiveBayesClassifier


def split_data(data_set, train_data_portion):
    train_set = []
    test_set = []
    for record in data_set:
        if random.random() < train_data_portion:
            train_set.append(record)
        else:
            test_set.append(record)

    return train_set, test_set


path = r"/home/haosun/PycharmProjects/DSfS-naive-bayes/spam/*/*"

data = []

for fn in glob.glob(path):
    is_spam = "ham" not in fn

    with open(fn, 'r') as file:
        for line in file:
            if line.startswith("Subject:"):
                subject = re.sub(r"^Subject: ", "", line).strip()
                data.append((subject, is_spam))

random.seed(0)
train_data, test_data = split_data(data, 0.75)

classifier = NaiveBayesClassifier()
classifier.train(train_data)

classified = [(subject, is_spam, classifier.classify(subject))
              for subject, is_spam in test_data]

counts = Counter((is_spam, spam_probability > 0.5)
                 for _, is_spam, spam_probability in classified)
