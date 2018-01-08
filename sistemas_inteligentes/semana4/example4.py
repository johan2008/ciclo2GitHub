import pandas as pd

df = pd.read_csv("shuffled_movie_data.csv")
print("example..1")
#print (df.tail()   )

import numpy as np
#import nltk
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
#from gensim import corpora, models, similarities
import gensim
import os


stop = stopwords.words('english')
porter = PorterStemmer()

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    text = [w for w in text.split() if w not in stop]
    tokenized = [porter.stem(w) for w in text]
    return text



#print (tokenizer('This :) is a <a> test! :-)</br>') )


def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

#print (  next(stream_docs(path='shuffled_movie_data.csv'))  )


from sklearn.feature_extraction.text import HashingVectorizer
vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)




def get_minibatch(doc_stream, size):
    docs, y = [], []
    for _ in range(size):
        text, label = next(doc_stream)
        docs.append(text)
        y.append(label)
    return docs, y


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()



count = 10
lenght = 100

file = open("shuffled_movie_data.csv", "r")
doclist = [ line for line in file ]
docstr = '' . join(doclist)
sentence = re.split(r'[.!?]', docstr)


#sentence = [['first', 'sentence'], ['second', 'sentence']]
#sentence = MySentences('shuffled_movie_data.csv')

def labelize( docs, labels ) :
    labelized = []
    lenght = len( docs )
    
    for i in range( lenght ) :
        label = '%s' % ( labels[i] )
        #print( _label )
        doc_tokenized = tokenizer( docs[i] )
        labelized.append( gensim.models.doc2vec.LabeledSentence( doc_tokenized, [label] ) )
        #print(" --------------------------------------")
        #print (_labelized)
    return labelized

_docs, _labels = get_minibatch( stream_docs(path='shuffled_movie_data.csv'), 50000 )

_sentences = labelize( _docs, _labels )


#print(_sentences[0])

#for _sentence in _sentences:
#	print("----------------------------------------------------------")
#	print (_sentence.words)

model = gensim.models.Word2Vec([ _sentence.words for _sentence in _sentences ]  ,min_count = 10,size = lenght)



#model = gensim.models.Word2Vec( sentences = sentence  ,min_count = 10,size = lenght)
#model = gensim.models.Word2Vec( sentences  ,min_count = count,)
print( model.most_similar( 'hello' ) )


def vectorize( doc, w2vModel ) :
    
    doc_tokenized = tokenizer( doc )
    _docVec = np.zeros( w2vModel.vector_size )
    _len = len( doc_tokenized )
    
    for _word in doc_tokenized :
        # get the embedding from the model
        if _word in w2vModel :
            _wvec = np.array( w2vModel[_word] )
            # add it to the doc vector
            _docVec = _docVec + _wvec
        
    _docVec = _docVec / _len
    
    return _docVec



print( vectorize( _docs[1], model ) )



#words = list(model.wv.vocab)
#print(words)












