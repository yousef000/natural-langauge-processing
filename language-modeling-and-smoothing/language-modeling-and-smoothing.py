import re
import math

# To handle out-of-vocabulary (OOV) words, convert tokens
# that occur less than three times into a special
# UNK token during training. If you did this correctly,
# your language model’s vocabulary (including the UNK
# token and STOP, but excluding START) should have 26602
# unique tokens (types).


def get_freq(freqs, data):
    for sentence in data:
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

def get_uni_model(data, freqs):
    unigram_model = {}
    total_freq = sum(freqs.values())
    
    for i in freqs:
        unigram_model[i] = freqs[i]/total_freq
    return unigram_model


def get_bi_count(data, freqs):
    ## compute the count of bigram C(x,y)
    bigram_count = {}
    for sentence in data:
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
            bigram_model[key[0]].update({
                key[1]: probability
            })
        else:
            bigram_model[key[0]] = {
                key[1]: probability
            }
    return bigram_model

def get_tri_count(data, freqs):
    ## compute the count of trigram C(x,y,z)
    trigram_count = {}
    for sentence in data:
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
            trigram_model[(key[0], key[1])].update({
                key[2]: probability
            })
        else:
            trigram_model[(key[0], key[1])] = {
                key[2]: probability
            }
    return trigram_model

# 1/M ∑∑log2(p(xij))
    # Where M is the same. The outer sum is the same as before, 
    # summing over all sentences in the file (assuming m sentences). 
    # The second sum sums over all tokens in the current sentence 
    # (this can be implemented as a nested for loop) (assuming k tokens 
    # in a sentence, this will obviously be different in each sentence). 
    # p(xij) is the probability of the current token in the current 
    # sentence, which is given by the n-gram MLE for the n you are using

def get_uni_pp(freqs, data):
    total_freq = sum(freqs.values())
    l = 0
    for i in data:
        for j in i.split():
            if j in freqs:
                l += math.log(unigram_model[j])
            else:
                l += math.log(unigram_model["UNK"])
    l *= 1/total_freq
    unigram_pp = pow(2, -l)
    return unigram_pp

def get_bi_pp(freqs, data):
    l = 0
    for sentence in data:
        words = sentence.split()
        for i in range(len(words)):
            ## if word is unknown then change token to UNK
            if words[i] not in freqs:
                words[i] = "UNK"
            if i != len(words)-1 and words[i+1] not in freqs:
                words[i+1] = "UNK"

            if i != len(words)-1:
                l += math.log(bigram_model[words[i]][words[i+1]])
            else:
                l += math.log(bigram_model[words[i]]["<STOP>"])
            
    l *= 1/sum(freqs.values())
    bigram_pp = pow(2, -l )
    return bigram_pp

def get_tri_pp(freqs, data):
    l = 0
    for sentence in data:
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
                l += math.log(trigram_model[(words[i], words[i+1])]["<STOP>"])
            else:
                l += math.log(trigram_model[(words[i], words[i+1])][words[i+2]])
            
    l *= 1/sum(freqs.values())
    trigram_pp = pow(2, -l )
    return trigram_pp

def get_smoothed_pp(data, uni_model, bi_model, tri_model, lambdas):
    total = 0
    log_probs = 0
    for sentence in data:
        words = sentence.split()
        for i in range(len(words) - 2):
            # unigram probability = probability of word in unigram model if in model otherwise 0
            uni_prob = uni_model.get(words[i], 0)

            bi_word = words[i+1]
            bi_context = words[i]
            if bi_word in bi_model[bi_context]:
                bi_prob = bi_model[bi_context][bi_word]
            else:
                bi_prob = 0
            
            # tri_word is the random variable
            tri_word = words[i+2]
            # tri_context is the context (previous n=2 words)
            tri_context = (words[i], words[i+1])

            if tri_word in tri_model[tri_context]:
                tri_prob = tri_model[tri_context][tri_word]
            else:
                tri_prob = 0

            probability = (lambdas[0] * uni_prob) + (lambdas[1] * bi_prob) + (lambdas[2] * tri_prob)


            log_probs = math.log(probability, 2) if probability != 0 else 0

            total += 1
            
    pp = math.pow(2, -(log_probs / total))
    return pp
    


if __name__ == '__main__':

    with open('data/1b_benchmark.train.tokens') as my_file:
        training_data = my_file.readlines()

    freqs = {
        "UNK": 0,
        "<STOP>": 0
    }
    get_freq(freqs, training_data)
    print('len freqs\n', len(freqs))
    
    unigram_model = get_uni_model(training_data, freqs)

    bigram_count = get_bi_count(training_data, freqs)
    # print('bigram count\n', bigram_count)
    bigram_model = get_bi_model(bigram_count, freqs)
    # print("bigram model: \n", bigram_model)

    trigram_count = get_tri_count(training_data, freqs)
    trigram_model = get_tri_model(trigram_count, bigram_count)
    
    # unigram_pp = get_uni_pp(freqs, training_data)
    # print("unigram_pp", unigram_pp)
    # bigram_pp = get_bi_pp(freqs, training_data)
    # print("bigram_pp", bigram_pp)
    # trigram_pp = get_tri_pp(freqs, training_data)
    # print("trigram_pp", trigram_pp)

    lambdas = [0.1, 0.3, 0.6]
    smooth_pp = get_smoothed_pp(training_data, unigram_model, bigram_model, trigram_model, lambdas)
    print("smooth_pp", smooth_pp)

    total = 0
    for i in unigram_model:
        total += unigram_model[i]
    print('total', total)