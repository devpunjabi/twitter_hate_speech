from nltk.corpus import stopwords
# tweet tokenize
from nltk.tokenize import TweetTokenizer

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
# Load pandas
import pandas as pd
# Load numpy
import numpy as np
from sklearn.metrics import accuracy_score

## preprocesses before data is sent to create the flatenned 2D modified-HAL matrix

def modHAL_preprocess_tweets(train,test):
    ## pre-requites
    stop_words = set(stopwords.words('english'))
    tknz = TweetTokenizer()
    train_tweet_token = []
    test_tweet_token = []
    tweet_token = []
    
    ## main poreprocess
    train_class = []
    for i in train: 
        word_tokens = tknz.tokenize(i[0])
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        train_class.append(i[1])
        train_tweet_token.append(filtered_sentence)
        tweet_token = tweet_token + filtered_sentence
    tweet_token = list(set(tweet_token))
#     del finalOffense_tweet, train
    
    
    test_class = []
    for i in test: 
        word_tokens = tknz.tokenize(i[0])
        filtered_sentence = [w for w in word_tokens if not w in stop_words] 
        test_class.append(i[1])
        test_tweet_token.append(filtered_sentence)
        tweet_token = tweet_token + filtered_sentence
    tweet_token = list(set(tweet_token))
#     del none_tweet, test
    
    return tweet_token, test_tweet_token, train_tweet_token, test_class, train_class


## function for creating the modified flatenned 2D HAL matrix, with one row per tweet as feature vectors
def HAL_operation(tweet_tokens, all_tokens, window_size = 5):
    the_big_mat = []
    temp_array = [0]*len(all_tokens)
    tweet_count = 1
    ## the decay array
    decay_arr = list(range(0,window_size,1))
    ## max value possible for a given word
    default_val = ((window_size*window_size)-sum(decay_arr))
    for i in tweet_tokens:
        my_tweet_array = temp_array.copy()
        i_count = len(i)
        gen_count = 0
        for x in i:
            if((i_count-gen_count)>=window_size):
                poss_val = (((gen_count+1)*window_size)-sum(decay_arr[0:(gen_count+1)]))
                my_tweet_array[all_tokens.index(x)] = min(poss_val,default_val)
            else:
                my_tweet_array[all_tokens.index(x)] = (i_count-gen_count+1)     
            gen_count+=1
        the_big_mat.append(my_tweet_array)
        tweet_count+=1
    return(the_big_mat)


## the RF classification
def RF_on_HAL(superarray,mode="train"):
    
    
    if(mode == "train"):
        tweet_token, train_class, train_HAL_mat_2D = superarray
        # Create a list of the feature column's names
        features = tweet_token
        # create factors for classes
        y = pd.factorize(list(train_class))[0]

        ### train
        # Create a random forest Classifier. By convention, clf means 'Classifier'
        classifier_rf = RandomForestClassifier(n_jobs=6, random_state=0)

        # Train the Classifier to take the training features and learn how they relate
        # to the training y (the class)
        classifier_rf.fit(np.array(train_HAL_mat_2D), y)
        
        return(classifier_rf)
    
    elif(mode == "test"):
        classifier_rf, test_class, test_HAL_mat_2D = superarray
        ### test
        # Apply the Classifier we trained to the test data (which, remember, it has never seen before)
        predictions = classifier_rf.predict(np.array(test_HAL_mat_2D))
    
        ### View the predicted probabilities of the first 10 observations
        all_prob = classifier_rf.predict_proba(np.array(test_HAL_mat_2D))
        
        # Create confusion matrix
        cnf_mat = pd.crosstab(pd.factorize(list(test_class))[0], np.array(predictions), rownames=['Actual class'], colnames=['Predicted class'])
        ## accuracy score
        acc_score = accuracy_score(pd.factorize(list(test_class))[0], np.array(predictions))
        
        return predictions, all_prob, cnf_mat, acc_score
    else:
        print("wrong parameters as input")
    

### plot matrix for RF
def plot_outputs(test_class,predictions,all_prob):
    orig_class = np.array(pd.factorize(list(test_class))[0])
    pred_class = np.array(predictions)
    predplot_data = np.c_[all_prob, pred_class]
    class0_data = predplot_data[predplot_data[:,-1]==0]
    class1_data = predplot_data[predplot_data[:,-1]==1]
    x0 = class0_data[:,0]
    y0 = class0_data[:,1]
    x1 = class1_data[:,0]
    y1 = class1_data[:,1]
    return x0,y0,x1,y1



## live tweet prediction function
def new_tweet_RF(new_tweet,tweet_token,classifier_rf):
    stop_words = set(stopwords.words('english'))
    tknz = TweetTokenizer()
    new_tweet_arr = []
    for i in new_tweet: 
        word_tokens = tknz.tokenize(i)
        filtered_sentence = [w for w in word_tokens if not w in stop_words] 
        filtered_sentence = [w for w in filtered_sentence if w in tweet_token] 
        new_tweet_arr.append(filtered_sentence)
    HAL_new_tweet = HAL_operation(new_tweet_arr, tweet_token, window_size = 5)
    RFoutput = classifier_rf.predict(np.array(HAL_new_tweet))
    if(RFoutput == 0):
        RFoutput = "offense"
    else:
        RFoutput = "none"
    return RFoutput