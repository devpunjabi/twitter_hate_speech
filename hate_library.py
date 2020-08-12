## library
import json
import codecs
from pprint import pprint 

# for storing and restoring variables
import pickle

# tweet tokenize
from nltk.tokenize import TweetTokenizer

# ml utilities
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# Load pandas
import pandas as pd
# Load numpy
import numpy as np
# plot
import matplotlib.pylab as plt
import time
from IPython import display
import warnings 
warnings.simplefilter('ignore')
