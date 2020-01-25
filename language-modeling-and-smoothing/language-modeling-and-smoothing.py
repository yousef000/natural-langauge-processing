import re
import math

# To handle out-of-vocabulary (OOV) words, convert tokens
# that occur less than three times into a special
# UNK token during training. If you did this correctly,
# your language model’s vocabulary (including the UNK
# token and STOP, but excluding START) should have 26602
# unique tokens (types).


def get_freq(freqs, test_data):
    for sentence in test_data:
        freqs["<STOP>"] += 1
        for word in sentence.split():
            if word in freqs:
                freqs[word] += 1
            else:
                freqs[word] = 1

    remove = [key for key in freqs if freqs[key] < 3 and key != "UNK"]
    for key in remove:
        del freqs[key]
        freqs["UNK"] += 1
    print(len(freqs))
    return freqs


def get_bi_count(test_data):
    # compute the count of bigram C(x,y)
    bigram_count = {}
    for sentence in test_data:
        words = sentence.split()
        for i in range(len(words)):
            if i == len(words)-1:
                break
            elif (words[i], words[i+1]) in bigram_count:
                bigram_count[(words[i], words[i+1])] += 1
            else:
                bigram_count[(words[i], words[i+1])] = 1
    print(bigram_count)
    return bigram_count


def get_bi_model(bigram_count):
    bigram_model = {}
    probability = 0.0
    for key in bigram_count:

        # To compute bigram probability of a word y (key[1]) given
        # previous word x (key[0]), divide the count of bigram
        # C(x,y) and sum of all bigrams that share the same first word x
        # which means freqs of x

        # whether word is OOV or not
        if key[0] in freqs:
            probability = bigram_count[key]/float(freqs[key[0]])
        else:
            probability = bigram_count[key]/float(freqs["UNK"])

        if key[0] in bigram_model:
            bigram_model[key[0]].append({
                key[1]: probability
            })
        else:
            bigram_model[key[0]] = [{
                key[1]: probability
            }]
    return bigram_model


if __name__ == '__main__':

    with open('data/1b_benchmark.train.tokens') as my_file:
        test_data = my_file.readlines()

    freqs = {
        "UNK": 0,
        "<STOP>": 0
    }
    get_freq(freqs, test_data)
    print('freqs\n', freqs)
    print('len freqs\n', len(freqs))

    # bigram_count = get_bi_count(test_data)
    # print('bigram count\n', bigram_count)

    # bigram_model = get_bi_model(bigram_count)

    # print("bigram model: \n", bigram_model)

    # total = 0

    # for key in bigram_model['Having']:
    #     for k in key:
    #         total += key[k]

    # print('total', total)