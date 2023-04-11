import nltk
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('punkt')

file = 'file.txt'
with open(file, 'r') as file:
    text = file.read()
sentences = nltk.sent_tokenize(text)
tokenized_text = []
for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    tokenized_text.append(words)
#START POS TAGGING METHODS HERE