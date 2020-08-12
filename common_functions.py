import numpy as np
import matplotlib.pylab as plt
import time
from IPython import display

## separating the input data to test and train

def create_test_train(finalOffense_tweet,none_tweet,ratio_to_train=0.75):
    # Set random seed
    np.random.seed(0)
    ## creating the new data matrix
    full_data_mat = []
    tweet_none = []
    tweet_offensive = []
    for i in range(0,len(finalOffense_tweet)): 
        tweet_none.append(none_tweet[i][0])
        tweet_offensive.append(finalOffense_tweet[i][0])
        
    ## combining the arrays
    full_data_mat = np.array(tweet_offensive + tweet_none)
    ## creating the classes array to be added into the data array
    classes_array = ["offense"]*len(finalOffense_tweet)+["none"]*len(none_tweet)
    ## adding classes as a new column
    full_data_mat = np.c_[full_data_mat, classes_array]
    
    ## separating test and train by randomization
    temp = np.random.uniform(0, 1, len(full_data_mat)) <= ratio_to_train
    ## adding them as additional columns in data
    full_data_mat = np.c_[full_data_mat, temp]
    
    ## separating test and train
    # Create two new dataframes, one with the training rows, one with the test rows
    train = full_data_mat[full_data_mat[:,-1]==['True']]
    train = train[:,:-1]  ## removing the radomised column
    test = full_data_mat[full_data_mat[:,-1]==['False']]
    test = test[:,:-1]    ## removing the radomised column
    
    return(test, train)


## plot static data in live mode

def static_data_live_plot(x0,y0,x1,y1,label0='offense',label1='none'):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for i in range(len(x0)):
        try:
            ax1.scatter(x0[0:i], y0[0:i], s=10, c='red', marker="x", label=label0)
            ax1.scatter(x1[0:i],y1[0:i], s=10, c='black', marker="o", label=label1)
            handles, labels = ax1.get_legend_handles_labels()
            handle_list, label_list = [], []
            for handle, label in zip(handles, labels):
                if label not in label_list:
                    handle_list.append(handle)
                    label_list.append(label)
            plt.legend(handle_list, label_list)
            display.display(plt.gcf())
            display.clear_output(wait=True)
            time.sleep(0.1)
        except KeyboardInterrupt:
            ax1.scatter(x0, y0, s=10, c='red', marker="x", label=label0)
            ax1.scatter(x1,y1, s=10, c='black', marker="o", label=label1)
            break

            
## voting matrix function
import operator
import pandas as pd
from sklearn.metrics import accuracy_score
def voting_matrix(test_class,RFpredictions,pre_svm,pre_nb):
    THE_CLASS = list(pd.factorize(test_class)[0])
    NB_class = list(pd.factorize(pre_nb)[0])
    SVM_class = list([1]+list(pd.factorize(pre_svm[1:],order=False)[0]))
    RF_class = list(RFpredictions)
    
    Voting_class = list(map(operator.add, NB_class,SVM_class))
    Voting_class = list(map(operator.add, Voting_class,RF_class))
    for (i, item) in enumerate(Voting_class):
        if item > 1:
            Voting_class[i] = 1
        else:
            Voting_class[i] = 0
    d = {"Real_Class": THE_CLASS,"NB_class":NB_class,"SVM_class":SVM_class,"RF_class":RF_class,"Voting_Class":Voting_class}
    vot_mat = pd.DataFrame(data = d, columns =["Real_Class","NB_class","SVM_class","RF_class","Voting_Class"])
    vot_cnf_mat = pd.crosstab(np.array(vot_mat['Real_Class']), np.array(vot_mat['Voting_Class']), rownames=['Actual class'], colnames=['Predicted class'])
    ## accuracy score
    vot_acc_score = accuracy_score(np.array(vot_mat['Real_Class']), np.array(vot_mat['Voting_Class']))
    return vot_mat,vot_cnf_mat, vot_acc_score 



## reading tweet live

from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API
import io
from tweepy.streaming import StreamListener
import json
import pickle
from HAL_RF_functions import new_tweet_RF
from BoGClassfier import BOGTweet_live


#Variables that contains the user credentials to access Twitter API
access_token = "900251987161305089-HgGGuGNtSfRCAGdMqWiRkaak42RIQ4V"
access_token_secret = "QBpZd9hlSc1qVp2SguVP4KcUotjVQVEKn0b17e7D9UyMA"
consumer_key = "c8ueHMou4lOZWRMyVxzcLNENQ"
consumer_secret = "E5AbdRPJxoNCnl8TR1XCkhDg5I749yWcCB89HFuq56iCxEpIvO"

