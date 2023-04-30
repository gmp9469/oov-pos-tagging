import pandas as pd
import regex as re
import collections
from collections import defaultdict
import math
import numpy as np

import argparse
import string 
oov_categories = ['unk_unk','unk_digit','unk_punc','unk_hyphen','unk_capital','unk_adj','unk_adv','unk_verb','unk_noun']
noun_suffix = ['ee','eer','er','sion','action','ion','ance','ence','ism','ity','ment','ness']
adj_suffix = ['able','ible','al','ant','ary','ful','ic','less','ious','ous','esque']
adv_suffix = ['ily','ally','ly','wards','ward','wise']
verb_suffix = ['ed','en','ing','ize','ise']

#function to identify which morphological class an unknown word belongs to
def unk_word_class(word):
    punct = set(string.punctuation)
    if any(char.isdigit() for char in word):
        return "unk_digit"
    elif any(char in punct for char in word):
        return "unk_punc"
    elif re.match("\w+-\w+",word): 
        return "unk_hyphen"
    elif any(char.isupper() for char in word):
        return "unk_capital"
    elif any(word.endswith(suf) for suf in adj_suffix):
        return "unk_adj"
    elif any(word.endswith(suf) for suf in adv_suffix):
        return "unk_adv"
    elif any(word.endswith(suf) for suf in verb_suffix):
        return "unk_verb"
    elif any(word.endswith(suf) for suf in noun_suffix):
        return "unk_noun"
    else:
        return "unk_unk"

def train(filenames):
    #filenames = ["WSJ_02-21.pos"]
    training_corpus = []
    for file in filenames:
        with open(file, 'r') as f:
            training_corpus += f.readlines()  

    #get set of all words that occur more than once in training corpus 
    #count frequency of these words
    word_count = defaultdict(int)
    for line in training_corpus:
        if line.split():
            (word,tag) = line.split()
            word_count[word] += 1

    words = [k for (k,v) in word_count.items() if v > 1]
    words.extend(oov_categories)
    words_set = set(words)

    #generate counts of (prev_tag,tag) and (tag, word) for calculation of transition and emission probabilities 
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    prev_tag = "<sentence>"
    for line in training_corpus:
        if not line.split():
            word = '<n>'
            tag = "<sentence>"
        else: 
            word,tag = line.split()
            if word not in words_set: 
                word = unk_word_class(word)
        emission_counts[(tag,word)] += 1
        transition_counts[(prev_tag,tag)] += 1
        tag_counts[(tag)] += 1
        prev_tag = tag

    tag_set = sorted(tag_counts.keys())
    num_tags = len(tag_set)
    num_words = len(words_set)

    #calculate transition probabilities 
    trans_probs = defaultdict(float)
    alpha = 0.000001
    for t1 in tag_set:
        for t2 in tag_set:
            t1_count = tag_counts.get(t1,0)
            t1_t2_count = transition_counts.get((t1,t2),0)
            trans_probs[(t1,t2)] = (t1_t2_count + alpha) / (t1_count + alpha * num_tags)

    #calculate emission probabilities 
    emission_probs = defaultdict(float)
    for t in tag_set:
        for w in words_set:
            t_count = tag_counts.get(t,0)
            t_w_count = emission_counts.get((t,w),0)
            emission_probs[(t,w)] = (t_w_count + alpha) / (t_count + alpha * num_words)
    return tag_set, trans_probs, emission_probs, words_set

def tag_sentence(sentence, tag_set, trans_probs, emission_probs, words_set):
    n = len(sentence)
    num_tags = len(tag_set)
    all_tags = list(tag_set)
    #unk_prob = 1/1000

    viterbi = np.zeros((num_tags,n))
    backpointer = np.zeros((num_tags, n), dtype = int)
    
    for i, tag in enumerate(all_tags):
        if trans_probs.get(('<sentence>',tag),0) == 0:
            viterbi[i][0] = float("-inf")
            print("negative infinite in beginning")
        else: 
            if sentence[0] not in words_set: 
                unk_w = unk_word_class(sentence[0])
                viterbi[i][0] = math.log(trans_probs.get(('<sentence>',tag))) + math.log(emission_probs.get((tag, unk_w)))
            else: 
                viterbi[i][0] = math.log(trans_probs.get(('<sentence>',tag))) + math.log(emission_probs.get((tag, sentence[0])))

    for s in range(1, n):
        for i, t in enumerate(all_tags):
            max_prob = float("-inf")
            best_path_idx = None
            for j, prev_t in enumerate(all_tags):
                if sentence[s] not in words_set: 
                    unk_w = unk_word_class(sentence[s])
                    prob = viterbi[j,s-1] + math.log(trans_probs.get((prev_t,t))) + math.log(emission_probs.get((t, unk_w)))
                else: 
                    prob = viterbi[j,s-1] + math.log(trans_probs.get((prev_t,t))) + math.log(emission_probs.get((t, sentence[s])))
                if prob > max_prob:
                    max_prob = prob
                    best_path_idx = j
            viterbi[i][s] = max_prob 
            backpointer[i][s] = best_path_idx

    #find best path using backpointer 
    best_prob_prev = float("-inf")
    best_path_idx = [None] * n 
    res = [None] * n 
    for k, pos in enumerate(all_tags):
        if viterbi[k][n-1] > best_prob_prev:
            best_prob_prev = viterbi[k][n-1]
            best_path_idx[n-1] = k
    res[n-1] = all_tags[best_path_idx[n-1]]

    for i in range(n-1,0,-1):
        pos = best_path_idx[i]
        best_path_idx[i-1] = backpointer[pos,i]
        res[i-1] = all_tags[best_path_idx[i-1]]
    return res 

def tag_file(input, output, tag_set, trans_probs, emission_probs, words_set):
    #oov_probs = oov_prob()
    inputfile = open(input, 'r')
    token = inputfile.readline()
    sentence = []
    while token != '':
        token = token.strip('\n')
        if token != '':
            sentence.append(token)
        else:
            best_path = tag_sentence(sentence, tag_set, trans_probs, emission_probs, words_set)
            for i, pos in enumerate(best_path):
                output.write(f"{sentence[i]}\t{pos}\n")
            sentence = []
            output.write("\n")
        token = inputfile.readline()
    output.close()
    return None

def main():
    parser = argparse.ArgumentParser(
    description = "Viterbi HMM POS Tagger"
    )
    parser.add_argument('--train', 
                        nargs = '+', 
                        dest = 'train_files', 
                        help = 'file name of input training corpus',
                        type = str)
    parser.add_argument('--tag', 
                        dest='tag_files', 
                        help = 'file name of test corpus')
    parser.add_argument('-out', 
                        dest='output_file', 
                        help = 'name of output file', 
                        default='submission.pos')
    args = parser.parse_args()
    print("Training Corpus Files: ", list(args.train_files))
    print("File to Tag: ", args.tag_files)
    print("Output File: ", args.output_file)

    tag_set, trans_probs, emission_probs, words_set = train(list(args.train_files))
    fout = open(args.output_file,"w")
    tagging = tag_file(args.tag_files, fout, tag_set, trans_probs, emission_probs, words_set)
    print("Tagging Complete")
if __name__ == '__main__':
    main()
