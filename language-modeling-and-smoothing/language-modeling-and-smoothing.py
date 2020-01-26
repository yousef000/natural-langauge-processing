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

def get_uni_model(test_data, freqs):
    unigram_model = {}
    total_freq = 0
    for i in freqs:
        total_freq += freqs[i]
    for sentence in test_data:
        words = sentence.split()
        for i in range(len(words)):
            if words[i] in freqs:
                probability = freqs[words[i]]/total_freq
            else:
                probability = freqs['UNK']/total_freq

            if words[i] in unigram_model:
                unigram_model[words[i]] += probability
            else:
                unigram_model[words[i]] = probability
    return unigram_model


def get_bi_count(test_data, freqs):
    ## compute the count of bigram C(x,y)
    bigram_count = {}
    for sentence in test_data:
        words = sentence.split()
        for i in range(len(words)):
            ## if word is unknown then change token to UNK
            if words[i] not in freqs:
                words[i] = "UNK"
            if i != len(words)-1 and words[i+1] not in freqs:
                words[i+1] = "UNK"
            
            ## if last word
            if i == len(words)-1:
                if (words[i], "<STOP>") in bigram_count:
                    bigram_count[(words[i], "<STOP>")] += 1
                else:
                    bigram_count[(words[i], "<STOP>")] = 1     
            elif (words[i], words[i+1]) in bigram_count:
                bigram_count[(words[i], words[i+1])] += 1
            else:
                bigram_count[(words[i], words[i+1])] = 1
    return bigram_count


def get_bi_model(bigram_count, freqs):
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

def get_tri_count(test_data, freqs):
    ## compute the count of trigram C(x,y,z)
    trigram_count = {}
    for sentence in test_data:
        words = sentence.split()
        for i in range(len(words)):
            ## if word is unknown then change token to UNK
            if words[i] not in freqs:
                words[i] = "UNK"
            if i != len(words)-1 and words[i+1] not in freqs:
                words[i+1] = "UNK"
            if i != len(words)-2 and i != len(words)-1 and words[i+2] not in freqs:
                words[i+2] = "UNK"
            
            ## if last word
            if i == len(words)-1:
                break
            if i == len(words)-2:
                if (words[i], words[i+1], "<STOP>") in trigram_count:
                    trigram_count[(words[i], words[i+1], "<STOP>")] += 1
                else:
                    trigram_count[(words[i], words[i+1], "<STOP>")] = 1 
            elif (words[i], words[i+1], words[i+2]) in trigram_count:
                trigram_count[(words[i], words[i+1], words[i+2])] += 1
            else:
                trigram_count[(words[i], words[i+1], words[i+2])] = 1
    return trigram_count

def get_tri_model(trigram_count, bigram_count):
    trigram_model = {}
    probability = 0.0
    for key in trigram_count:

        # To compute trigram probability of a word z (key[2]) given
        # previous word x (key[0]) and y (key[1]), divide the count of trigram
        # C(x,y,z) and sum of all trigrams that share the same first word x
        # and y which means bigram_count of x and y

        # don't need to check for OOV since it was taken care in bi_count and tri_count
        probability = trigram_count[key]/float(bigram_count[(key[0], key[1])])
        
        if (key[0], key[1]) in trigram_model:
            trigram_model[(key[0], key[1])].append({
                key[2]: probability
            })
        else:
            trigram_model[(key[0], key[1])] = [{
                key[2]: probability
            }]
    return trigram_model
    
if __name__ == '__main__':

    with open('data/1b_benchmark.train.tokens') as my_file:
        test_data = my_file.readlines()

    freqs = {
        "UNK": 0,
        "<STOP>": 0
    }
    get_freq(freqs, test_data)
    print('len freqs\n', len(freqs))
    
    unigram_model = get_uni_model(test_data, freqs)

    bigram_count = get_bi_count(test_data, freqs)
    # print('bigram count\n', bigram_count)
    bigram_model = get_bi_model(bigram_count, freqs)
    # print("bigram model: \n", bigram_model)

    trigram_count = get_tri_count(test_data, freqs)
    trigram_model = get_tri_model(trigram_count, bigram_count)
    
    unigram_pp = 1
    for i in unigram_model:
        unigram_pp *= 1/unigram_model[i]
    unigram_pp = pow(unigram_pp, 1/float(len(unigram_model)))
    print(unigram_pp)

    trigram_pp = 1
    for i in trigram_model:
        for j in trigram_model[i]:
            for k in j:
                trigram_pp *= 1/j[k]
    trigram_pp = pow(trigram_pp, 1/float(len(trigram_model)))
    print(trigram_pp)
    # print('total', total)