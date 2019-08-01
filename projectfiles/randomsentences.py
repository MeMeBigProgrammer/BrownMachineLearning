import re
import numpy as np
import random
words = []

with open ("wordlists/mueller.txt", "r") as myfile:
    data=myfile.readlines()
    for word in data:
        extender = []
        for x in re.split(r'(?<=[^\w\'.\"])', word):
            if x != " " and x != "":
                extender.append(x.strip().lower())
        words.extend(extender)

# the graph was A PAIN, I HOPE IT BRINGS YOU MORE JOY THAN IT BROUGHT ME
graph = {}
for word_i in range(1, len(words)):
    graph[words[word_i - 1]] = {}

for word_i in range(1, len(words)):
    graph[words[word_i - 1]][words[word_i]] = 0

for word_i in range(1, len(words)):
    graph[words[word_i - 1]][words[word_i]] += 1

sentence = []
cur_node = words[random.randint(0, len(words))]
for i in range(20):
    sum_of_probs = sum(graph[cur_node].values())
    probs = {}
    for key, value in graph[cur_node].items():
        probs[key] = value/sum_of_probs

    node = np.random.choice(list(probs.keys()),p=list(probs.values()))
    sentence.append(node)
    cur_node = node

print(sentence)