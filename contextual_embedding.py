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

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

def train_contextual():
    # SPECIFY PATH TO CORPUS
    data_folder = 'path'#!!!!!!

    # SPECIFY COLUMN FORMAT
    columns = {0: 'text', 1: 'pos', 2: 'ner'}

    tag_type = 'pos'
    corpus = ColumnCorpus(data_folder, columns, tag_type=tag_type)
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    embedding_types = [
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ]
    embeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=tag_dictionary,
                            tag_type=tag_type)
    trainer = ModelTrainer(tagger, corpus)

    trainer.train('./',
                  train_with_dev=True,
                  max_epochs=150)
    
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