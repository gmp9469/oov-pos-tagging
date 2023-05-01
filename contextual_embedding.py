"""
TAKEN FROM: https://huggingface.co/flair/pos-english 
credit:
@inproceedings{akbik2018coling,
  title={Contextual String Embeddings for Sequence Labeling},
  author={Akbik, Alan and Blythe, Duncan and Vollgraf, Roland},
  booktitle = {{COLING} 2018, 27th International Conference on Computational Linguistics},
  pages     = {1638--1649},
  year      = {2018}
}

"""
#MUST PIP INSTALL FLAIR

import flair
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence

def train_contextual():
    # SPECIFY PATH TO CORPUS
    train_file = './training.pos'

    # SPECIFY COLUMN FORMAT
    columns = {0: 'text', 1: 'pos'}

    corpus = ColumnCorpus('', columns, train_file=train_file)
    tag_dictionary = corpus.make_tag_dictionary(tag_type='pos')
    embedding_types = [
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ]
    embeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=tag_dictionary,
                            tag_type='pos')
    trainer = ModelTrainer(tagger, corpus)

    trainer.train('./',
                  train_with_dev=True,
                  max_epochs=5)
    
    return tagger

""" EXAMPLE USE ---------------------
from flair.data import Sentence

# create a sentence object
sentence = Sentence('This is a new sentence.')

# load the trained tagger
tagger = train_contextual()

# tag the sentence
tagger.predict(sentence)

# print the tagged sentence
print(sentence.to_tagged_string())

EVALUATION -----------------
# load the test corpus
test_corpus = ColumnCorpus(test_data_folder, columns, tag_type=tag_type)

# evaluate the model on the test corpus
result, _ = trainer.evaluate(test_corpus, mini_batch_size=32)

# print the accuracy
print(result.detailed_results)

"""
file = open('./testing.words', 'r')
words = []
sentences = []
for line in file:
    row = line.split()
    if row:
        words.append(row[0])  
    else:
        if words:
            sentences.append(words)
        words = []

tagger = train_contextual()
submission = open("submission2.pos", "w")
for words in sentences:
    sentence = Sentence(' '.join(words))
    tagger.predict(sentence)
    tags = [token.tag for token in sentence.tokens]
    for i in range(len(words)):
        line = words[i] + "\t" + tags[i] + '\n'
        submission.write(line)
    submission.write("\n")