#Helper file that has list SVM C parameters
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import collections
import random
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
import time

#Function that returns the average result for the ML classfiers
def calculate_average_results_ML(ml_results):
   
    # Create dictionaries to store values for each variables for each model
    model_accuracy_sum = {}
    model_f1score_sum = {}
    model_f1score_mean = {}
    model_preprocess_cputime_sum = {}
    model_learner_cputime_sum = {}
    model_classifer_cputime_sum = {}
    model_confusion_matrices = {}
    model_count = {}

    # Iterate through the data and calculate sums
    for result_set in ml_results:
        for model_data in result_set:
            model_name = model_data['Model']
            
            if model_name not in model_accuracy_sum:
                model_accuracy_sum[model_name] = 0
                model_f1score_sum[model_name] = [0, 0]  # Store F1 scores as an array [f1score_1_sum, f1score_2_sum]
                model_f1score_mean[model_name] = 0
                model_preprocess_cputime_sum[model_name] = 0
                model_learner_cputime_sum[model_name] = 0
                model_classifer_cputime_sum[model_name] = 0
                model_confusion_matrices[model_name] = np.zeros((2, 2))  # 2x2 confusion matrix
                model_count[model_name] = 0
            
            model_confusion_matrices[model_name] += model_data['confusionmatrix']
            model_accuracy_sum[model_name] += model_data['Accuracy']
            model_f1score_sum[model_name][0] += model_data['F1Score'][0]
            model_f1score_sum[model_name][1] += model_data['F1Score'][1]
            model_f1score_mean[model_name] += model_data['F1Score_mean']
            model_preprocess_cputime_sum[model_name] += model_data.get('ModelPreprocessCPUTime', 0)
            model_learner_cputime_sum[model_name] += model_data.get('ModelLearnerCPUTime', 0)
            model_classifer_cputime_sum[model_name] += model_data.get('ModelClassiferCPUTime', 0)
    
            model_count[model_name] += 1
   
    # Calculate averages
    
    model_average_accuracy = {model: accuracy_sum / model_count[model] for model, accuracy_sum in model_accuracy_sum.items()}
    
    model_average_confusionmatrices = {model: confusionmatrix_sum / model_count[model] for model, confusionmatrix_sum in model_confusion_matrices.items()}
    
    model_average_preprocesscputime = {model: preprocesscputime_sum / model_count[model] for model, preprocesscputime_sum in model_preprocess_cputime_sum.items()}
    
    model_average_learnercputime = {model: learnercputime_sum / model_count[model] for model, learnercputime_sum in model_learner_cputime_sum.items()}
    
    model_average_classifiercputime = {model: classifercputime_sum / model_count[model] for model, classifercputime_sum in model_classifer_cputime_sum.items()}
    
    model_average_f1score = {
        model: [f1score_sum[0] / model_count[model], f1score_sum[1] / model_count[model]]
        for model, f1score_sum in model_f1score_sum.items()
    }
    model_average_meanf1score = {model: meanf1score_sum / model_count[model] for model, meanf1score_sum in model_f1score_mean.items()}
    
   
    return model_average_confusionmatrices, model_average_accuracy, model_average_f1score, model_average_meanf1score, model_average_preprocesscputime, model_average_learnercputime, model_average_classifiercputime


            
#Function that returns the average result for the lexicon classfier
def calculate_average_results_lexicon(lexicon_results):
    total_results = len(lexicon_results)
    average_results = {}
    sum_results = {}

    for result in lexicon_results:
        for key, value in result.items():
            if key not in sum_results:
                sum_results[key] = value
            else:
                sum_results[key] += value

    for key, value in sum_results.items():
        average_results[key] = value / total_results

    return average_results

#CPU execution time
def cpuexecutiontime():
    current_time = time.time()
    return current_time
   

#Tokenise the Summary text
def nlpsteps(x):
    review = re.sub('[^a-zA-Z]', ' ', str(x))
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review


#Corpus data splitted into separate words
def convert(corpus_trainingdata):
    return ([i for item in corpus_trainingdata for i in item.split()])
     


# Counts of each words in the corpus
def getwordcounts(splittedWords):
    occurrences = collections.Counter(splittedWords)
    return occurrences


#Function that returns the counts for each words that falls in Severe or Non Severe category
def get_distribution(val,training_data_df):
    records = training_data_df[
        training_data_df["Summary"].str.contains(val)
    ]
    
    if len(records) > 0:
        res = training_data_df[
            training_data_df["Summary"].str.contains(val)
        ]["Severity"].value_counts(dropna=False)
        return dict(res)
    return None
    
    
# Function that returns Severe ratio
def get_r1(ns,s):
    return s/(s+ns)

# Function that returns NonSevere ratio
def get_r2(ns,s):
    return ns/(s+ns)

    

#Incase of equal frequency the classifier will be Severe
def classifier(Summary,severedictionary_list,nonseveredictionary_list):
  
    summaryList = Summary.split()
    mytest_severe = len(set(severedictionary_list).intersection(summaryList))
    mytest_nonsevere = len(set(nonseveredictionary_list).intersection(summaryList))
    
    if mytest_severe >= mytest_nonsevere:
        tag = "Severe"
    elif mytest_severe < mytest_nonsevere:
        tag = "NonSevere"
   
    return tag
   



#Function that creates dictionary on different threholds and tests with validation data and returns the confusion matrix and accuracy scores of each dictionary
def dictionary_onthresholds(severe_threshold, nonsevere_threshold, dataset, payload_train):

    severe_dictionary = {}
    nonsevere_dictionary = {}
    for keyy in payload_train:
        if payload_train[keyy]['r1'] >= severe_threshold:
            severe_dictionary[keyy] = payload_train[keyy]
        
        if payload_train[keyy]['r2'] >= nonsevere_threshold:
            nonsevere_dictionary[keyy] = payload_train[keyy]
            
    severedictionary_list = list(severe_dictionary.keys())
#     print(severedictionary_list)
    nonseveredictionary_list = list(nonsevere_dictionary.keys())
#     print(nonseveredictionary_list)
    
    dataset["Summary"]  = dataset["Summary"].apply(lambda x: nlpsteps(x))
    
    dataset["my_tag"] = dataset["Summary"].apply(lambda x: classifier(x,severedictionary_list,nonseveredictionary_list))
    
   
    TP = 0 
    for d in dataset.iterrows():
        if ((d[1]["my_tag"]== "Severe") & (d[1]["Severity"]== "Severe")):
            TP = TP+1
    FP = 0 
    for d in dataset.iterrows():
        if (d[1]["my_tag"]== "Severe" )& (d[1]["Severity"]== "NonSevere"):
            FP = FP+1
    TN = 0 
    for d in dataset.iterrows():
        if (d[1]["my_tag"]== "NonSevere") & (d[1]["Severity"]== "NonSevere"):
            TN = TN+1
    FN = 0 
    for d in dataset.iterrows():
        if (d[1]["my_tag"]== "NonSevere") & (d[1]["Severity"]== "Severe"):
            FN = FN+1

    confusion_matrix = {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN
        
    }
#     F1Score for Severe

    confusion_matrix["Precision"]= confusion_matrix["TP"]/(confusion_matrix["TP"]+confusion_matrix["FP"]) if (confusion_matrix["TP"]) !=0 else 0
    confusion_matrix["Recall"]= confusion_matrix["TP"]/(confusion_matrix["TP"]+confusion_matrix["FN"]) if (confusion_matrix["TP"]) !=0 else 0
    confusion_matrix["F1Score"] = 2*(confusion_matrix["Precision"] * confusion_matrix["Recall"])/(confusion_matrix["Precision"] + confusion_matrix["Recall"]) if (confusion_matrix["Precision"] + confusion_matrix["Recall"]) != 0 else 0
 
 #     F1Score for NonSevere
    confusion_matrix["Precision_nonsevere"]= confusion_matrix["TN"]/(confusion_matrix["TN"]+confusion_matrix["FN"]) if (confusion_matrix["TN"]) !=0 else 0
    confusion_matrix["Recall_nonsevere"]= confusion_matrix["TN"]/(confusion_matrix["TN"]+confusion_matrix["TN"]) if (confusion_matrix["TN"]) !=0 else 0
    confusion_matrix["F1Score_nonsevere"] = 2*(confusion_matrix["Precision_nonsevere"] * confusion_matrix["Recall_nonsevere"])/(confusion_matrix["Precision_nonsevere"] + confusion_matrix["Recall_nonsevere"]) if (confusion_matrix["Precision_nonsevere"] + confusion_matrix["Recall_nonsevere"]) != 0 else 0
    
 #     F1Score average for Severe and NonSevere
    confusion_matrix["F1Score_Average"]= (confusion_matrix["F1Score"]+confusion_matrix["F1Score_nonsevere"])/2 if (confusion_matrix["F1Score_nonsevere"]) !=0 else 0
                                                                                                   
    
    return confusion_matrix
   


################# outerloop breakdown##########################################################################
# Function that returns the worlists for severe and non severe 
def lexicon_preprocess(trainingdataset_length,training_data_df):

    corpus_trainingdata = []
       
    for i in range(0,trainingdataset_length):
        review = nlpsteps(str(training_data_df['Summary'][i]))
        corpus_trainingdata.append(review)
   

    #Split words from the corpus
    splittedWords = convert(corpus_trainingdata)
    
    splitted_words=getwordcounts(splittedWords)

    #Converted collection.counter into dictionary
    splitted_words_dict = dict(splitted_words)

    keys = splitted_words_dict.keys()
    
    all_data = {}
    for key in keys:
        res = get_distribution(key,training_data_df)
        if res:
            all_data[key] = res
            all_data
        
    payload_train = {}
    for key, value in all_data.items():
        ns = value.get('NonSevere', 0)
        s = value.get('Severe',0)

        r1 = get_r1(ns, s)
        r2 = get_r2(ns, s)

        payload_train[key]= {
            'r1': r1,
            'r2': r2
        }
        payload_train

    return payload_train 

# Function that returns the created dictioanries on possible combination of the severe and non severe threshold and returns best thresholds
def lexicon_learner(payload_train,validation_data):
    
    severe_threshold = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1 ,0.2, 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    nonsevere_threshold = [0.0, 0.1 ,0.2, 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    possibleThesholdCombination = list(itertools.product(severe_threshold, nonsevere_threshold))
    result_list=[]
    for i in possibleThesholdCombination:

        severe_randomthreshold = i[0]
        nonsevere_randomthreshold = i[1]

        count = dictionary_onthresholds(severe_randomthreshold,nonsevere_randomthreshold,validation_data, payload_train)
        result_dictionary = {
         "severe_threshold": severe_randomthreshold,
         "nonsevere_threshold":nonsevere_randomthreshold,
 
        }
        result_dictionary.update(count)
        result_list.append(result_dictionary)
        F1Score_df = pd.DataFrame(result_list)
    maxf1Score= F1Score_df[F1Score_df['F1Score']==F1Score_df['F1Score'].max()]
    
#     print("---------Best threshold for dictionary found testing with validation data------")
   
    
#     print(maxf1Score)
 
    severethreshold_ = maxf1Score['severe_threshold'].values[0]
    nonseverethreshold_ = maxf1Score['nonsevere_threshold'].values[0]
    
    return severethreshold_, nonseverethreshold_ 

#Function that creates a dictioanry on the best threshold and tests with testing dataset
def lexicon_classifier(severethreshold_,nonseverethreshold_,testing_data,payload_train):
    
    lexicon_classifier_results = dictionary_onthresholds(severethreshold_,nonseverethreshold_,testing_data,payload_train)
  
    return lexicon_classifier_results


############### outerloop breakdown ends #############

def get_SVM_best_C_hyperparamter(X_train,Y_train,X_validation,y_validation):
    C_hyperparameter = [0.1,0.5,1,5,10,20,50,100]
#     C_hyperparameter = [0.1]
   
    SVM_accuracy_list = []
    SVM_list= []
    
    for c in C_hyperparameter:
       
        SVM_dict = {}

        svm_model = SVC(C = c, kernel='linear', gamma='auto')
        svm_model.fit(X_train,Y_train)

        svm_pred = svm_model.predict(X_validation)
        svm_model = confusion_matrix(y_validation, svm_pred)

        SVM_accuracy_list= accuracy_score(y_validation, svm_pred)
        f1_score_svm = f1_score(y_validation, svm_pred, average=None)
        F1ScoreSVM_severe= f1_score_svm[1]
#         print(c,F1ScoreSVM_severe)
        
        SVM_dict = {"C":c,"Accuracy": SVM_accuracy_list, "SVMF1Score": F1ScoreSVM_severe }
        SVM_list.append(SVM_dict)
        SVM_list
        F1Score_df_SVM = pd.DataFrame(SVM_list)
        
    max_f1Score_svm = F1Score_df_SVM[F1Score_df_SVM['SVMF1Score']==F1Score_df_SVM['SVMF1Score'].max()]
    
    best_c_hyperparamter = max_f1Score_svm['C'].values[0]
#     print("best c", best_c_hyperparamter)
   
        
    return best_c_hyperparamter
      
    

# #---------------------------ML Classifier Starts---------------------------------


def mlclassifier_outerloop(trainingdataset_length,testingdataset_length,validationdataset_length,training_data_df,validation_data_df,testing_data_df,training_data, rs):
    
    ml_starttime_preprocess = cpuexecutiontime()
    #Tokenised the training data
    trainingdata_tokenised = []
    for i in range(0,trainingdataset_length):
        review_train = nlpsteps(str(training_data_df['Summary'][i]))
        trainingdata_tokenised.append(review_train)

    #Tokenised the testing data
    testingdata_tokenised = []
    for i in range(0,testingdataset_length):
        review_test = nlpsteps(str(testing_data_df['Summary'][i]))
        testingdata_tokenised.append(review_test)

    #Tokenised the validation data
    validationdata_tokenised = []
    for i in range(0,validationdataset_length):
        review_validation = nlpsteps(str(validation_data_df['Summary'][i]))
        validationdata_tokenised.append(review_validation)


    max_feature_list = []
    max_feature_accuracy = []
    SVM_list= []


#     features = [1000,10000,15000]
    features = [15000]
    

    for i in features:

        cv = CountVectorizer(max_features = i)
        X_train = cv.fit_transform(trainingdata_tokenised).toarray()
        Y_train = training_data.iloc[:, -2].values
        testingdata_vector = cv.transform(testingdata_tokenised)
        X_test = testingdata_vector.toarray()
        y_test = testing_data_df.iloc[:, -2].values
        validationdata_vector = cv.transform(validationdata_tokenised)
        X_validation = validationdata_vector.toarray()
        y_validation = validation_data_df.iloc[:, -2].values

        ml_endtime_preprocess = cpuexecutiontime()
        ml_preprocess_cputime = ml_endtime_preprocess - ml_starttime_preprocess

    #------------------------------------SVM------------------------------------------------------------------
        SVM_learner_starttime = cpuexecutiontime()

        C_hyperparameter = get_SVM_best_C_hyperparamter(X_train,Y_train,X_validation,y_validation)

        SVM_accuracy_list = []

        SVM_dict = {}

        svm_model = SVC(C = C_hyperparameter, kernel='linear', gamma='auto')
        svm_model.fit(X_train,Y_train)

        SVM_learner_endtime = cpuexecutiontime()
        SVM_learner_cputime = SVM_learner_endtime - SVM_learner_starttime

        SVM_classifier_starttime = cpuexecutiontime()
        svm_pred = svm_model.predict(X_test)
        svm_model = confusion_matrix(y_test, svm_pred)
        #convert to numpy
        numpy_array_CM = np.array(svm_model)
        svm_model = numpy_array_CM.tolist()

        SVM_accuracy_list= accuracy_score(y_test, svm_pred)
        f1_score_svm = f1_score(y_test, svm_pred, average=None)
        

        f1score_SVM_mean = np.mean(f1_score_svm)

        SVM_classifer_endtime = cpuexecutiontime()
        SVM_classifer_cputime = SVM_classifer_endtime - SVM_classifier_starttime



        SVM_dict = {"features":i,"c": C_hyperparameter,"Model":'SVM', "confusionmatrix":svm_model,"Accuracy": SVM_accuracy_list, "F1Score": f1_score_svm.tolist(),"F1Score_mean": f1score_SVM_mean,"ModelPreprocessCPUTime": ml_preprocess_cputime, "ModelLearnerCPUTime":SVM_learner_cputime, "ModelClassiferCPUTime":SVM_classifer_cputime,"RandomSeeds":rs}
        SVM_list.append(SVM_dict)
        SVM_list
        max_feature_list.append(SVM_dict)


     #-------------------------------------Naive Bayes-------------------------------------------------------------
        NB_learner_starttime = cpuexecutiontime()

        classifier = MultinomialNB()
        classifier.fit(X_train, Y_train)

        NB_learner_endtime = cpuexecutiontime()
        NB_learner_cputime = NB_learner_endtime - NB_learner_starttime

        NB_classifier_starttime = cpuexecutiontime()

        MultinomialNB_pred = classifier.predict(X_test)

        cm_MB = confusion_matrix(y_test, MultinomialNB_pred)
        #convert to numpy
        numpy_array_CM = np.array(cm_MB)
        cm_MB = numpy_array_CM.tolist()
        
        max_feature_accuracy = accuracy_score(y_test, MultinomialNB_pred)
        f1_score_MB = f1_score(y_test, MultinomialNB_pred, average=None)

        f1score_MB_mean = np.mean(f1_score_MB)


        NB_classifier_endtime = cpuexecutiontime()
        NB_classifer_cputime = NB_classifier_endtime - NB_classifier_starttime

        maxfeature_dict = {"features":i,"Model":'MultinomialNB', "confusionmatrix": cm_MB ,"Accuracy": max_feature_accuracy, "F1Score": f1_score_MB.tolist(), "F1Score_mean":f1score_MB_mean, "ModelPreprocessCPUTime": ml_preprocess_cputime,"ModelLearnerCPUTime":NB_learner_cputime, "ModelClassiferCPUTime":NB_classifer_cputime,"RandomSeeds":rs}
        max_feature_list.append(maxfeature_dict)


        #-------------------------------------Logistic Regression-------------------------------------------------------------
        LR_learner_starttime = cpuexecutiontime()

        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train,Y_train)

        LR_learner_endtime = cpuexecutiontime()
        LR_learner_cputime = LR_learner_endtime - LR_learner_starttime

        LR_classifer_starttime = cpuexecutiontime()
        lr_pred = lr_model.predict(X_test)


        cm_lr = confusion_matrix(y_test, lr_pred)
         #convert to numpy
        numpy_array_lr = np.array(cm_lr)
        cm_lr = numpy_array_lr.tolist()
        
        max_feature_accuracy =  accuracy_score(y_test, lr_pred)
        f1_score_lr = f1_score(y_test, lr_pred, average=None)

        f1score_LR_mean = np.mean(f1_score_lr)


        LR_classifer_endtime = cpuexecutiontime()
        LR_classifer_cputime = LR_classifer_endtime - LR_classifer_starttime

        maxfeature_dict = {"features":i, "Model":'LogisticRegression', "confusionmatrix": cm_lr ,"Accuracy": max_feature_accuracy, "F1Score": f1_score_lr.tolist(),"F1Score_mean":f1score_LR_mean,"ModelPreprocessCPUTime": ml_preprocess_cputime, "ModelLearnerCPUTime":LR_learner_cputime, "ModelClassiferCPUTime":LR_classifer_cputime,"RandomSeeds":rs}
        max_feature_list.append(maxfeature_dict)


       # -------------------------------Dummy Classification--------------------------------------
        dummy_learner_starttime = cpuexecutiontime()

        dummy_clf = DummyClassifier(strategy="most_frequent")
        dummy_clf.fit(X_train, Y_train)
        dummy_learner_endtime = cpuexecutiontime()
        dummy_learner_cputime = dummy_learner_endtime - dummy_learner_starttime

        dummy_classifer_starttime = cpuexecutiontime()

        DummyClassifier(strategy='most_frequent')
        dummy_pred = dummy_clf.predict(X_test)
        cm_dummy = confusion_matrix(y_test, dummy_pred)
         #convert to numpy
        numpy_array_dummy = np.array(cm_dummy)
        cm_dummy = numpy_array_dummy.tolist()
        
        
        dummy_accuracy = dummy_clf.score(X_test, y_test)
        f1_score_dummy = f1_score(y_test, dummy_pred, average=None)
       

        f1score_dummy_mean = np.mean(f1_score_dummy)

        dummy_classifer_endtime = cpuexecutiontime()
        dummy_classifer_cputime = dummy_classifer_endtime - dummy_classifer_starttime

        maxfeature_dict = {"features":i, "Model":'DummyClassifier', "confusionmatrix": cm_dummy ,"Accuracy": dummy_accuracy, "F1Score": f1_score_dummy.tolist(),"F1Score_mean":f1score_dummy_mean, "ModelPreprocessCPUTime": ml_preprocess_cputime, "ModelLearnerCPUTime":dummy_learner_cputime, "ModelClassiferCPUTime":dummy_classifer_cputime,"RandomSeeds":rs }

        max_feature_list.append(maxfeature_dict)

        return max_feature_list
       #-------------------------------------ML Classifer & Dummy classifier Ended here---------------------------
                           
       


