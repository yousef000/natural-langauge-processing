import math

START = "<START>"
STOP = "<STOP>"

gram_counts = {}
context_counts = {}
probabilities = {}
corpus_len = 0


def init_uni_bi_model(data):
    global gram_counts
    global context_counts
    global probabilities
    global corpus_len
    vocab = 0
    sentences = 0
    for sentence in data:
        words = sentence.split()
        words.append(STOP)
        words.insert(0, START)
        sentences += 1
        for i in range(1, len(words)):
            bigram = (words[i-1], words[i])
            context = words[i-1]

            gram_counts[bigram] = gram_counts.get(bigram, 0) + 1

            context_counts[context] = context_counts.get(context, 0) + 1

            gram_counts[("", words[i])] = gram_counts.get(context, 0) + 1

            context_counts[""] = context_counts.get("", 0) + 1

            vocab += 1

            # rest of loop code only for trigram
            if i < 2 or i > len(words) - 2:
                continue

            trigram = (words[i-2], words[i-1], words[i])
            context = (words[i-2], words[i-1])

            gram_counts[trigram] = gram_counts.get(trigram, 0) + 1

            context_counts[context] = context_counts.get(context, 0) + 1

    for gram, count in gram_counts.items():
        if len(gram) == 3:
            context = (gram[0], gram[1])
        else:
            context = gram[0]

        probability = gram_counts[gram]/context_counts[context]

        probabilities[gram] = probability

        if context == "":
            print(gram, len(probabilities[gram]))

    print("total", context_counts[""])


# def init_tri_model(data):
#     global gram_counts
#     global context_counts
#     global probabilities
#     global corpus_len
#     for sentence in data:
#         words = sentence.split()
#         words.append(STOP)
#         words.insert(0, START)
#         for i in range(2, len(words)-2):

#     for gram, count in gram_counts.items():
#         context = gram[0]
#         probability = gram_counts[gram]/context_counts[context]

#         probabilities[gram] = probability


if __name__ == '__main__':

    with open('data/1b_benchmark.train.tokens') as my_file:
        training_data = my_file.readlines()

    with open('data/1b_benchmark.dev.tokens') as my_file:
        dev_data = my_file.readlines()

    with open('data/test') as my_file:
        test_data = my_file.readlines()

    init_uni_bi_model(training_data)

    # print(probabilities)

    uni_prob_sum = 0
    bi_prob_sum = 0
    tri_prob_sum = 0

    for key, value in probabilities.items():
        if len(key) == 2:

            if key[0] == "":
                uni_prob_sum += value
            else:
                bi_prob_sum += value
        if len(key) == 3:
            tri_prob_sum += value

    print("uni : ", uni_prob_sum)
    print("bi : ", bi_prob_sum)
    print("tri : ", tri_prob_sum)
    # freqs = {
    #     "UNK": 0,
    #     "<STOP>": 0
    # }
    # get_freq(freqs, training_data)
    # print('len freqs ', len(freqs))
    # total_tokens_in_training = sum(freqs.values())
    # print("Total tokens: ", total_tokens_in_training)

    # unigram_model = get_uni_model(training_data, freqs)

    # bigram_count = get_bi_count(training_data, freqs)
    # # print('bigram count\n', bigram_count)
    # bigram_model = get_bi_model(bigram_count, freqs)
    # # print("bigram model: \n", bigram_model)

    # trigram_count = get_tri_count(training_data, freqs)
    # trigram_model = get_tri_model(trigram_count, bigram_count)

    # lambdas = [[0.1, 0.3, 0.6], [0, 0, 1], [
    #     0.3, 0.3, 0.4], [0.6, 0.3, 0.1], [1, 0, 0]]

    # best_lambda_set = []
    # lowest_perp = 1000000

    # for lam in lambdas:
    #     smooth_pp = get_smoothed_pp(
    #         test_data, unigram_model, bigram_model, trigram_model, lam)
    #     if smooth_pp < lowest_perp:
    #         lowest_perp = smooth_pp
    #         best_lambda_set = lam
    #     print('Lambda Set: ', lam, ', Perplexity: ', smooth_pp)

    # total = 0

    # unigram_pp = get_uni_pp(freqs, test_data)
    # print("unigram_pp", unigram_pp)
    # bigram_pp = get_bi_pp(freqs, test_data)
    # print("bigram_pp", bigram_pp)
    # trigram_pp = get_tri_pp(freqs, test_data)
    # print("trigram_pp", trigram_pp)

    # print('unigram total', sum(unigram_model.values()))
