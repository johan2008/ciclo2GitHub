import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import xml.etree.ElementTree as ET
from lxml import etree
from scipy.sparse import hstack
import numpy as np
import warnings


path_train = r'E:\Engineering\8th sem\nlp COMP 473\NLP projects\ABSA16_Laptops_Train_English_SB2.xml'
path_test = r'E:\Engineering\8th sem\nlp COMP 473\NLP projects\EN_LAPT_SB2_TEST.xml'

#For stanford POS Tagger
home = r'C:\Users\THe_strOX\Anaconda3\stanford-postagger-full-2017-06-09'
from nltk.tag.stanford import StanfordPOSTagger as POS_Tag
from nltk import word_tokenize
_path_to_model = home + '/models/english-bidirectional-distsim.tagger' 
_path_to_jar = home + '/stanford-postagger.jar'
stanford_tag = POS_Tag(model_filename=_path_to_model, path_to_jar=_path_to_jar)