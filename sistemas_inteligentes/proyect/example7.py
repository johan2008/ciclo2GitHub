#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 01:10:55 2017
@author: vicz
"""
print '''
     +-+-+-+-+-+-+ +-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+    
     |A|s|p|e|c|t| |b|a|s|e|d| |s|e|n|t|i|m|e|n|t| |a|n|a|l|y|s|i|s|    
     +-+-+-+-+-+-+ +-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+    
                                                               |b|y|    
                                                               +-+-+-+-+
                                                               |v|i|c|z|
                                                               +-+-+-+-+
    
'''
from textblob import TextBlob
import sys

def main():
    blob=TextBlob(sys.argv[1])
    tokens=list(blob.words)
    word=[]
    sent=[]
    c=0
    i=0
    for words,pos in blob.tags:
        if pos=='JJ' or pos=='NN' or pos=='JJR' or pos=='NNS':
            word.append(words)
    if len(word)>=2:
    	for i in range(len(word)):
    		if len(word)>=2:
    			print i
	    		firstw=word[0]
	    		secw=word[1]
	    		word.remove(firstw)
	    		word.remove(secw)
	    		findx=tokens.index(firstw)
	    		lindx=tokens.index(secw)
	    		sent.append(' '.join(tokens[findx:lindx+1]))

    print sent
    print tokens
    print "Sentence and polarity"    
    for sentence in sent:
        print sentence,TextBlob(sentence).polarity
                
if __name__=='__main__':
    main()