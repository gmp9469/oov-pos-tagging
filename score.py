#!/usr/bin/python
#
# scorer for NLP class Spring 2016
# ver.1.0
#
# score a key file against a response file
# both should consist of lines of the form:   token \t tag
# sentences are separated by empty lines
#
import sys
import os
from collections import defaultdict

def get_oov (keyFileName): 
	keyFile = open(keyFileName, 'r')
	key = keyFile.readlines()
	word_count = defaultdict(int)
	for line in key: 
		if line.split():
			(word, tag) = line.split()
			word_count[word] += 1
	words = [k for (k,v) in word_count.items() if v > 1]
	words_set = set(words)
	return words_set

def score (keyFileName, responseFileName, words_set):
	keyFile = open(keyFileName, 'r')
	key = keyFile.readlines()
	responseFile = open(responseFileName, 'r')
	response = responseFile.readlines()
	if len(key) != len(response):
		print("length mismatch between key and submitted file")
		exit()
	correct = 0
	incorrect = 0
	oov_correct = 0
	oov_incorrect = 0
	non_oov_correct = 0
	non_oov_incorrect = 0
	for i in range(len(key)):
		key[i] = key[i].rstrip(os.linesep)
		response[i] = response[i].rstrip(os.linesep)
		if key[i] == "":
			if response[i] == "":
				continue
			else:
				print ("sentence break expected at line " + str(i))
				exit()
		keyFields = key[i].split('\t')
		if len(keyFields) != 2:
			print ("format error in key at line " + str(i) + ":" + key[i])
			exit()
		keyToken = keyFields[0]
		keyPos = keyFields[1]
		responseFields = response[i].split('\t')
		if len(responseFields) != 2:
			print ("format error at line " + str(i))
			exit()
		responseToken = responseFields[0]
		responsePos = responseFields[1]
		if responseToken != keyToken:
			print ("token mismatch at line " + str(i))
			exit()

		#Calculate OOV Accuracy
		if keyToken not in words_set:
			if responsePos == keyPos:
				oov_correct = oov_correct + 1
			else:
				oov_incorrect = oov_incorrect + 1
		
		#Calculate OOV Accuracy
		if keyToken in words_set:
			if responsePos == keyPos:
				non_oov_correct = non_oov_correct + 1
			else:
				non_oov_incorrect = non_oov_incorrect + 1

		#Calculate Overall Accuracy
		if responsePos == keyPos:
			correct = correct + 1
		else:
			incorrect = incorrect + 1
	print (str(correct) + " out of " + str(correct + incorrect) + " tags correct")
	accuracy = 100.0 * correct / (correct + incorrect)
	oov_accuracy = 100.0 * oov_correct / (oov_correct + oov_incorrect)
	non_oov_accuracy = 100.0 * non_oov_correct / (non_oov_correct + non_oov_incorrect)

	print("  overall accuracy: %f" % accuracy)
	print (str(oov_correct + oov_incorrect) + " out of " + str(correct + incorrect) + " are OOV Tokens")
	print("  oov accuracy: %f" % oov_accuracy)
	print (str(non_oov_correct + non_oov_incorrect) + " out of " + str(correct + incorrect) + " are Non-OOV Tokens")
	print("  oov accuracy: %f" % non_oov_accuracy)

def main(args):
	training_file = args[1]
	key_file = args[2]
	response_file = args[3]
	words_set = get_oov(training_file)
	score(key_file,response_file, words_set)

if __name__ == '__main__': sys.exit(main(sys.argv))
