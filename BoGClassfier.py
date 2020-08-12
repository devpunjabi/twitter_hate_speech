
import pickle
import re
import numpy as np
from matplotlib import pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC




def preprocessTweet(TweeList):
    testList = []
    for d in range(len(TweeList)):
        testList.append([re.sub('(?<=^|(?<=[^a-zA-Z0-9-_.+A-Za-z+]))(@[A-Za-z_+A-Za-z+]+[A-Za-z0-9-_:+A-Za-z_+]+|htt.*:[//.A-Za-z0-9-_:+A-Za-z_+]+|&#[0-9]+|([^a-zA-Z0-9])|(RT))','',TweeList[d][0]),TweeList[d][1]])
    return testList

def createVectorX(tweetList):
    vect = ["".join(tweetList[x][0]) for x in range(len(tweetList))]
    return vect
   
def createVectorY(tweetList):
    y = []
    for x in range(len(tweetList)):
        y.append(tweetList[x][1])
    return y

def classifyTweet(XtrainVector, YtrainVector,XtestVector,YtestVector):
    stop_ = set(stopwords.words('english'))
    text_clf_NB = Pipeline([('vec', CountVectorizer(stop_words= stop_)), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    text_clf_NB = text_clf_NB.fit(XtrainVector, YtrainVector)
    predictedNB = text_clf_NB.predict(XtestVector)
    mean1 = np.mean(predictedNB == YtestVector)
    print("Prediction Accuracy for Bag of words with MultiNominal Naive Bayes:",mean1*100)
# text_clf_SVM = Pipeline([('vec', CountVectorizer(stop_words= stop)), ('tfidf', TfidfTransformer()), ('clf', SVC(max_iter=5, probability=True,random_state=42))])
    text_clf_SVM = Pipeline([('vec', CountVectorizer(stop_words= stop_)), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='log', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])
    text_clf_SVM = text_clf_SVM.fit(XtrainVector, YtrainVector)
    predictedSVM = text_clf_SVM.predict(XtestVector)
    mean2 = np.mean(predictedSVM == YtestVector)
    print("Prediction Accuracy for Bag of words with SGDClassifier(SVM):",mean2*100)
    return predictedNB, predictedSVM,text_clf_NB,text_clf_SVM


def createPlotData(predicted,text_clf, prob):
    x0 = []
    x1 = []
    y0 = []
    y1 = []

    for x in range(len(prob)):
        if predicted[x] == 'none':
            x0.append(prob[x][0])
            y0.append(prob[x][1])
        else:
            x1.append(prob[x][0])
            y1.append(prob[x][1])

    return x0, x1, y0, y1
   

def BOGTweet_live(tweet,text_clf_NB,text_clf_SVM):
    filterTweet = [re.sub('(?<=^|(?<=[^a-zA-Z0-9-_.+A-Za-z+]))(@[A-Za-z_+A-Za-z+]+[A-Za-z0-9-_:+A-Za-z_+]+|htt.*:[//.A-Za-z0-9-_:+A-Za-z_+]+|&#[0-9]+|([^a-zA-Z0-9])|(RT))','',tweet)]
    pre_nb = text_clf_NB.predict(filterTweet)
    pre_svm = text_clf_SVM.predict(filterTweet)
 
    return pre_svm, pre_nb
   