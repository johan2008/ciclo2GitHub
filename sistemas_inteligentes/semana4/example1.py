import nltk, gensim
class FileToSent(object):    
    def __init__(self, filename):
        self.filename = filename
        self.stop = set(nltk.corpus.stopwords.words('english'))

    def __iter__(self):
        for line in open(self.filename, 'r'):
        	ll = [i for i in unicode(line, 'utf-8').lower().split() if i not in self.stop]
        yield ll

sentences = FileToSent("shuffled_movie_data.csv")
model = gensim.models.Word2Vec(sentences=sentences, window=5, min_count=5, workers=4, hs=1)