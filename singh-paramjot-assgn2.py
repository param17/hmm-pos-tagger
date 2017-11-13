#! /usr/bin/env/python
# Paramjot Singh

def count_tag(train, test):
    vocab_set = []
    word_set = []
    tag_set = []
    tag_freq = {}
    start_tag_freq = {}
    end_tag_freq = {}
    word_freq = {'UNK': 0}
    tag_bigram = {}
    word_tag_bigram = {}
    trans_prob_table = {}
    obsrv_prob_table = {}
    init_prob_table = {}
    corpus_size = len(train) - 1
    # Counting for Transition State Probability matrix
    for i in range(corpus_size):
        if train[i][2] not in tag_set:
            # tag count
            tag_set.append(train[i][2])
            # tag freq count
            tag_freq[train[i][2]] = 1
        else:
            tag_freq[train[i][2]] += 1

        if train[i][1] not in word_freq:
            # word count
            word_set.append(train[i][1])
            # word freq count
            word_freq[train[i][1]] = 1
        else:
            word_freq[train[i][1]] += 1

        if (train[i][2], train[i + 1][2]) not in tag_bigram:
            # tag bigram freq count
            tag_bigram[(train[i][2], train[i + 1][2])] = 1
        else:
            tag_bigram[(train[i][2], train[i + 1][2])] += 1

        if (train[i][2], train[i][1]) not in word_tag_bigram:
            # word_tag_bigram freq count
            word_tag_bigram[(train[i][2], train[i][1])] = 1
        else:
            word_tag_bigram[train[i][2], train[i][1]] += 1

    vocab_set = word_set

    vocab_set.append('UNK')

    # Counting for Observation Probability matrix
    for i in range(corpus_size):
        if word_freq[train[i][1]] < 2:
            # updated UNK count for every 1 or less count word
            word_freq['UNK'] += 1
            # removed 1 count word from word freq
            word_freq.pop(train[i][1])
            # removed 1 count word from vocab_set
            vocab_set.remove(train[i][1])

            if (train[i][2], train[i][1]) in word_tag_bigram:
                if (train[i][2], 'UNK') not in word_tag_bigram:
                    # Updating the count for Tag, UNK with count of Tag, Word(Count<2)
                    word_tag_bigram[(train[i][2], 'UNK')] = word_tag_bigram[(train[i][2], train[i][1])]
                else:
                    word_tag_bigram[(train[i][2], 'UNK')] += word_tag_bigram[(train[i][2], train[i][1])]
                # removing the entries for (Tag, Word(Count<2)) from word_tag_bigram
                del word_tag_bigram[train[i][2], train[i][1]]

    total_start_tag = 0;
    total_end_tag = 0;
    # Counting for Initial Probability Matrix
    for i in range(len(train)):
        if (i == 0) or train[i - 1][2] == '.':
            total_start_tag += 1
            if train[i][2] not in start_tag_freq:
                start_tag_freq[train[i][2]] = 1
            else:
                start_tag_freq[train[i][2]] += 1

        if (i == len(train) - 1) or train[i + 1][2] == '.':
            total_end_tag += 1
            if train[i][2] not in end_tag_freq:
                end_tag_freq[train[i][2]] = 1
            else:
                end_tag_freq[train[i][2]] += 1

    tag_set_size = len(tag_set)
    # calculate the transition probability matrix
    for i in range(tag_set_size):
        for j in range(tag_set_size):
            # C(b,a)
            C_ij = tag_bigram[(tag_set[j], tag_set[i])] if (tag_set[j], tag_set[i]) in tag_bigram else 0
            # C(b)
            C_j = tag_freq[tag_set[j]] if tag_set[j] in tag_freq else 0
            # C(b,a) / C(b)
            # prob = float(C_ij + 0.0000001) / float(C_j + 0.0000001)
            prob = float(C_ij + 1)/float(C_j + len(vocab_set))
            # P(a|b) = C(b,a) / C(b)
            trans_prob_table[(tag_set[i], tag_set[j])] = prob

    tag_set_size = len(tag_set)
    vocab_set_size = len(vocab_set)
    # calculate the observation probability matrix
    for i in range(vocab_set_size):
        for j in range(tag_set_size):
            # C(b,a)
            C_ij = word_tag_bigram[(tag_set[j], vocab_set[i])] if (tag_set[j],
                                                                   vocab_set[i]) in word_tag_bigram else 0
            # C(b)
            C_j = tag_freq[tag_set[j]] if tag_set[j] in tag_freq else 0
            # C(b,a) / C(b)
            prob = C_ij / C_j
            # P(a|b) = C(b,a) / C(b)
            obsrv_prob_table[(vocab_set[i], tag_set[j])] = prob

    # calculate the initial probability matrix
    for i in range(tag_set_size):
        # C(start, a)
        C_ij = start_tag_freq[tag_set[i]] if tag_set[i] in start_tag_freq else 0
        # C(a)
        C_j = total_start_tag
        # C(start, a) / C(a)
        prob = C_ij / C_j
        # P(a|b) = C(b,a) / C(b)
        init_prob_table[(tag_set[i])] = prob

    def kneser_ney():
        d=0.4
        tag_set_size = len(tag_set)
        # calculate the transition probability matrix with kneser-ney smoothing
        for i in range(tag_set_size):
            for j in range(tag_set_size):
                # C(b,a)
                C_ji = tag_bigram[(tag_set[j], tag_set[i])] if (tag_set[j], tag_set[i]) in tag_bigram else 0
                # C(b)
                C_j = tag_freq[tag_set[j]] if tag_set[j] in tag_freq else 0
                # C(*,a)
                C_xi = tag_bigram[(_,tag_set[i])] if (_, tag_set[i]) in tag_bigram else 0
                # C(*,*) by type
                C_xi = len(tag_bigram)
                # C(b,*)
                C_xi = tag_bigram[(tag_set[i],_)] if (tag_set[i],_) in tag_bigram else 0
                # C(b,a) / C(b)
                prob = C_ji / C_j
                # P(a|b) = C(b,a) / C(b)
                trans_prob_table[(tag_set[i], tag_set[j])] = prob

    def baseline_func(input):
        print("Baseline")
        opt = []

        for t in range(len(input)):
            max = 0
            for tag in tag_set:
                if(obsrv_prob_table[input[t],tag] > max):
                    max = obsrv_prob_table[input[t],tag]
                    opt.append(tag)

        return opt


    def viterbi_func(input):

        V = [{}]

        for st in tag_set:
            V[0][st] = {"prob": init_prob_table[st] * obsrv_prob_table[input[0],st], "prev": None}

        # Run Viterbi when t > 0
        for t in range(1, len(input)):
            V.append({})
            for st in tag_set:
                max_tr_prob = max(V[t-1][prev_st]["prob"]*trans_prob_table[st,prev_st] for prev_st in tag_set)

                for prev_st in tag_set:
                    if V[t-1][prev_st]["prob"] * trans_prob_table[st,prev_st] == max_tr_prob:
                        max_prob = max_tr_prob * obsrv_prob_table[input[t],st]

                        V[t][st] = {"prob": max_prob, "prev": prev_st}
                        break

        opt = []

        # The highest probability
        max_prob = max(value["prob"] for value in V[-1].values())
        previous = None
        # Get most probable state and its backtrack
        for st, data in V[-1].items():
            if data["prob"] == max_prob:
                opt.append(st)
                previous = st
                break

        # Follow the backtrack till the first observation
        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]

        return opt


    obsrv = []
    n = 0
    for i in range(len(test)):
        if test[i][1] not in vocab_set:
            # print(test[n], " ", i)
            obsrv.append('UNK')
        else:
            obsrv.append(test[i][1])
    seq = []
    res = []
    for word in obsrv:
        if word != '.':
            seq.append(word)
        else:
            seq.append(word)
            # print(seq)
            res.append(viterbi_func(seq))
            seq = []

    output = [item for sublist in res for item in sublist]


    # output = baseline_func(obsrv)

    print(output)

    print_results_hmm(test, output)



def hmm_ricky(train, test):

    count_tag(train, test)


def print_results_hmm(test, output):
    dummy = 0
    with open('result.txt', 'w') as done:
        for i in range(len(test)):
            if (test[i][0] == '1') and (dummy == 1):
                done.write('\n')
            dummy = 1
            done.write(test[i][0] + '\t' + test[i][1] + '\t' + output[i].upper() + '\n')
    done.close()


train = []
test = []
train_data = open('berp-POS-training.txt', 'r').read().splitlines()
test_data = open('assgn2-test-set.txt', 'r').read().splitlines()
for line in train_data:
    if line:
        d = line.split()
        train.append([x.lower() for x in d])

for line in test_data:
    if line:
        d = line.split()
        test.append([x.lower() for x in d])

hmm_ricky(train, test)
