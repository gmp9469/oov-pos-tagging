import random
import math
from collections import defaultdict
from collections import Counter

f = open('trainingPOS.pos', 'r')
wordList = (f.read().strip().split('\n'))

posFreq = dict()
stateFreq = dict()
allWords = []

# LIKELIHOOD
for line in wordList:
	if line != '':
		word, pos = line.split('\t')
		
		if word not in allWords:
			allWords.append(word)
			
		if pos in posFreq:
			if word in posFreq[pos]:
				posFreq[pos][word] += 1
			else:
				posFreq[pos].update({word: 1})
		else:
			posFreq.update({pos:{word: 1}})
			
# TRANSITION
prevPos = "Start_Sentence"
for line in wordList:
	if line != '':
		word, pos = line.split('\t')
		
		if prevPos in stateFreq:
			if pos in stateFreq[prevPos]:
				stateFreq[prevPos][pos] += 1
			else:
				stateFreq[prevPos].update({pos: 1})
		else:
			stateFreq.update({prevPos:{pos: 1}})
		prevPos = pos
		
	else:
		if prevPos in stateFreq:
			if 'End_Sentence' in stateFreq[prevPos]:
				stateFreq[prevPos]['End_Sentence'] += 1
			else:
				stateFreq[prevPos].update({'End_Sentence': 1})
		else:
			stateFreq.update({prevPos:{'End_Sentence':1}})
		prevPos = 'Start_Sentence'
		
# LIKELIHOOD LAPLACE SMOOTHING
for tag in posFreq.keys():
	for word in allWords:
		if word in posFreq[tag]:
			posFreq[tag][word] = (posFreq[tag][word] + 1) / (sum(posFreq[tag].values()) + len(allWords))
		else:
			posFreq[tag][word] = 1 / (sum(posFreq[tag].values()) + len(allWords))
			
# TRANSITION LAPLACE SMOOTHING
for s in stateFreq:
	for s2 in stateFreq.keys():
		if s2 in stateFreq[s]:
			stateFreq[s][s2] = (stateFreq[s][s2] + 1) / (sum(stateFreq[s].values()) + len(stateFreq.keys()))
		else:
			stateFreq[s][s2] = 1 / (sum(stateFreq[s].values()) + len(stateFreq.keys()))
			
# HMM/VITERBI/POS TAGGING WITH LAPLACE ADDED
def likelihoodFunc(likelihood, row_mapping, sentence, s, n):
	if(sentence[n] in allWords):
		return likelihood[row_mapping[s]][sentence[n]]
	else:
		return 1 / (sum(likelihood[row_mapping[s]].values()) + len(allWords))