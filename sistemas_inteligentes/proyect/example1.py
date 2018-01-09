
#https://ahmedbesbes.com/sentiment-analysis-on-twitter-using-word2vec-and-keras.html

import xml.etree.ElementTree as ET
import re
import pandas as pd
from random import shuffle
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
from gensim.models.doc2vec  import Doc2Vec  
from gensim.models.doc2vec import LabeledSentence,TaggedDocument
from gensim.models import FastText
from glove import Corpus,Glove

#from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

import numpy as np # high dimensional vector computing library.
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools

def remove_repeated(word):
    return ''.join(c for c, _ in itertools.groupby(word))

def remove_numbers(sentence):
    return re.sub(" \d+", " DIGITO", sentence)

def remove_single_characters(sentence):
     return re.sub(r"\b[a-zA-Z]\b", "",sentence)

def lemmatize(word):
    from nltk.stem import WordNetLemmatizer
    lemmatizer=WordNetLemmatizer()
    return lemmatizer.lemmatize(word)

def clean_tweet(tweet):
    #try:
        #tweet = unicode(tweet.decode('utf-8').lower())
        #tokenizer = TweetTokenizer()
        tweet = tweet.lower()
        tweet = remove_numbers(tweet)
        tweet = remove_single_characters(tweet)
        tokens = TweetTokenizer().tokenize(tweet)
        tokens = filter(lambda t: not t.startswith('@') 
                              and not t.startswith('http') 
                              and not t.startswith('#'), tokens)
        tokens = filter(lambda t: t.strip('#\'"?,.!<>[]-') , tokens )
        tokens = map(remove_repeated,tokens)
        tokens = map(lemmatize,tokens)
        #tokens = filter(lambda t:, tokens)
        #tokens = filter(lambda t: not t.startswith('http'), tokens)
        return tokens
    #except:
    #    return 'NC'
# print remove_repeated("holaaaaaaaaaa")
# print tokenize("@vmc1385 Muchas gracias atrasadas!! Esto del Twitter hacia que no lo miraba.... Un abrazoteee!!")

sentiment = {'N':0,'P':1,'NEU':2,'NONE':3}

#sentiment2 = {'N':0,'P':1,'NEU':2,'NONE':3}

def getTweets(file_name,type):
    #file_name = "intertass-train-tagged.xml"
    tree = ET.parse(file_name)
    root = tree.getroot()
    tweets = []
    for tweet in root.findall('tweet'):
        content = tweet.find('content').text
        #print tweet.find("tweetid").text
        tokens = clean_tweet(content)
        if type == "TRAIN" :
            polarity = tweet.find('sentiment').find('polarity').find('value').text
            tweets.append({"content" : content , "polarity" : sentiment[polarity] , "tokens" : tokens })
        else :
            #polarity = tweet.find('sentiments').find('polarity').find('value').text
            tweets.append({"content" : content , "polarity" : '' , "tokens" : tokens })
    #print content , polarity
    return pd.DataFrame(tweets)



def getTest(file_name):
    with open(file_name,'r') as file :
        answers = []
        for line in file :
            id , polarity = line.split()
            answers.append(sentiment[polarity])
        return pd.DataFrame(answers)
#n = data.size
#print n

data_train = getTweets("data/intertass-train-tagged.xml", "TRAIN")
#data_train = getTweets("data/general-test-tagged-3l.xml", "TRAIN")
data_test = getTweets("data/intertass-test.xml","TEST")

x_train = data_train.tokens
y_train = data_train.polarity

x_test = data_test.tokens
y_test = getTest("data/intertass-sentiment.qrel")


#s = '@ManuBarba Lo feo es llegar al reina sofia 989632 y que los descuentos de estudiante sean para menores de 27 anos #hola'
#print clean_tweet(s)

def getDoc2Vec(sentences,size_embeddings):
    try:
        model = Doc2Vec.load('models/model.d2v')
    except Exception as e:
        sentences = [TaggedDocument(words,[str(i)] ) for i, words in enumerate(sentences)]
        model = Doc2Vec( min_count=1, window=10, size=size_embeddings, sample=1e-4, negative=5, workers=4 )
        model.build_vocab(sentences)
        for epoch in range(100):
            shuffle(sentences)
            model.train( sentences , total_examples=model.corpus_count, epochs=model.iter )
        model.save('models/model.d2v')          
        #print e
    return model

def getWord2Vec(sentences,size_embeddings) :

    try:
        model = Word2Vec.load('models/model.w2v')
    except Exception as e:
        model = Word2Vec( min_count= 1 , size=size_embeddings ,sg = 0 , negative = 5 )
        model.build_vocab(sentences)
        for epoch in range(100):
            shuffle(sentences)
            model.train( sentences , total_examples=model.corpus_count, epochs=model.iter )
        model.save('models/model.w2v')          
        #print e
    return model


def getGlove2Vec(sentences,size_embeddings):
    try:
        model = Glove.load('models/model.glove')
    except Exception as e:
        corpus = Corpus()
        model = Glove(no_components=100, learning_rate=0.05)
        corpus.fit( sentences, window=10)
        model.fit(corpus.matrix, epochs=100, no_threads=4, verbose=False)
        model.add_dictionary(corpus.dictionary)
        model.save('models/model.glove')
        #sentences = [TaggedDocument(words,[str(i)] ) for i, words in enumerate(sentences)] 
    return model

def getFasText(sentences,size_embeddings):
    
    try:
        model = FastText.load('models/model.ft')
    except Exception as e:
        #model = FastText(min_count=1, model = 'skipgram', size = size_embeddings)
        model = FastText( min_count=1, size = size_embeddings)
        model.build_vocab(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=models.iter)
        model.save('models/model.ft')
    return model




def buildWordVector(model ,tfidf , tokens, size ):
    
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


#print data_train.head(1)
#print x_train[0]

#m = buildWordVector(x_train[0],300)
#print m.shape

def getVectorbyDoc(model , sentences, name , type ):
    if name == "GLOVE" :
        vectors = np.asarray([ model.transform_paragraph(sentence,epochs=10,ignore_missing=True) for sentence in sentences ])
    elif name == "DOC2VEC" :
        if type == "TRAIN" :
            vectors = np.asarray([ model.docvecs[v] for v in range(0,len(sentences)) ])
        elif type == "TEST" :
            vectors = np.asarray([ model.docvecs[v] for v in range(len(model.docvecs)-1,len(model.docvecs)-len(sentences)-1,-1) ])
    else :
        print 'building tf-idf matrix ...'
        vectorizer = TfidfVectorizer(analyzer=lambda x:x, min_df=10)
        matrix = vectorizer.fit_transform(sentences)
        tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
        print 'vocab size :', len(tfidf)

        vectors = np.concatenate([buildWordVector(model , tfidf ,token,size_embeddings) for token in map(lambda x: x, sentences )])
        vectors = scale(vectors)
    return vectors



size_embeddings = 50 
### obtener caracteristicas
train_sentences = np.asarray(x_train) #[x for x in x_train ] 


doc2vec = getDoc2Vec(train_sentences,size_embeddings)
word2vec = getWord2Vec(train_sentences,size_embeddings)
glove2vec = getGlove2Vec(train_sentences,size_embeddings)
fastext = getFasText(train_sentences,size_embeddings)

#print word2vec.wv.vocab
#print word2vec.most_similar("hola")
#print word2vec.most_similar('tener')
train_sentences = np.asarray(x_train) #[x for x in x_train ] 
train_vectors_glove = getVectorbyDoc(glove2vec,train_sentences,"GLOVE","TRAIN")
train_vectors_doc2vec = getVectorbyDoc(doc2vec,train_sentences,"DOC2VEC","TRAIN")
train_vectors_fastext = getVectorbyDoc(fastext,train_sentences,"FASTTEXT","TRAIN")

test_sentences = np.asarray(x_test) #[x for x in x_train ] 
test_vectors_glove = getVectorbyDoc(glove2vec,test_sentences,"GLOVE","TEST")
test_vectors_doc2vec = getVectorbyDoc(doc2vec,test_sentences,"DOC2VEC","TEST")
test_vectors_fastext = getVectorbyDoc(fastext,test_sentences,"FASTTEXT","TEST")

print len(train_vectors_doc2vec)
print len(test_vectors_doc2vec)

def reshape(start , end , y,r,g,b):
    #print start , end
    vector =  np.zeros((end-start,y, 1 , 3))
    for i in range(0,start-end) :
        for j in range(y) :
            vector[i][j][0][0] = r[i][j] 
            vector[i][j][0][1] = g[i][j] 
            vector[i][j][0][2] = b[i][j] 
    return vector


#print train_vectors_glove[0][0]
#print train_vectors_doc2vec[0][0]
#print train_vectors_fastext[0][0]

#train_2 = np.asarray([train_vectors_doc2vec,train_vectors_glove,train_vectors_fastext])
#test_2 = np.asarray([test_vectors_doc2vec,test_vectors_glove,test_vectors_fastext])

def CNN(X_train , X_test , y_train , y_test , size_vectors ):

    from keras.models import Sequential
    from keras.layers import Conv2D, Dropout, Merge, Dense, Activation ,MaxPooling2D,Flatten
    from keras.layers.embeddings import Embedding
    from keras.utils import to_categorical
    from keras import optimizers


    print X_train.shape
    print X_test.shape

    Y_train = to_categorical(y_train, 4)
    Y_test = to_categorical(y_test, 4)

    print Y_train.shape
    print Y_test.shape

    model = Sequential()

    model.add(Conv2D(512, (3, 1), activation='relu', input_shape=(size_vectors,1,3)))

    #print model.output_shape

    model.add(Conv2D(64,( 3, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,1)))
    model.add(Dropout(0.25)) # evita overfitting

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer= optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])


    model.fit(X_train, Y_train, batch_size= 16 ,  epochs=20, verbose=1 )

    score = model.evaluate(X_test, Y_test, batch_size = 16 , verbose=0)

    print score[1]

train_shape = x_train.shape
test_shape = x_test.shape
X_train = reshape( 0 , train_shape[0],size_embeddings,train_vectors_doc2vec,train_vectors_glove,train_vectors_fastext)
X_test = reshape( train_shape[0] , train_shape[0]+test_shape[0], size_embeddings , test_vectors_doc2vec,test_vectors_glove,test_vectors_fastext)

CNN(X_train,X_test,y_train,y_test,size_embeddings)