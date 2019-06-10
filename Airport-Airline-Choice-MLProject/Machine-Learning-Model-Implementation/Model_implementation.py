import pandas as pd
import patsy
import numpy as np
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
import statsmodels.tools as sm_tools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import graphviz
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve  
from sklearn.metrics import roc_auc_score

def get_metrics(y_actual, y_predicted):
    print("Confusion Matrix: \n ", format(confusion_matrix(y_actual, y_predicted.round())))
    print("Accuracy: ", format(accuracy_score(y_actual, y_predicted.round())))
    print("Precision: ", format(precision_score(y_actual, y_predicted.round())))
    print("Recall: ", format(recall_score(y_actual, y_predicted.round())))
    print("F1 score: ", format(f1_score(y_actual, y_predicted.round())))

def plot_roc_curve(fpr, tpr):  
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

raw_data = pd.read_csv("export_from_code.csv")

print(raw_data.describe())
print(raw_data.info())

raw_data_obj = raw_data.select_dtypes(['object'])

raw_data[raw_data_obj.columns] = raw_data_obj.apply(lambda x: x.str.replace(" ", ""))

# Remove Destination 999
raw_data = raw_data[raw_data.Destination != '999']

# Add column for combined airlines
raw_data["Foreign_Airline"] = raw_data["Airline"].map( lambda x: 1 if x == 'ForeignAirlines' else 0)
# print(raw_data["Foreign_Airline"])

# # # ------------------------------------------------------------- #
# # # ----------------------- Multinomial ------------------------- #
# # # ------------------------------------------------------------- #
# # mn_logit_data = raw_data.copy(deep=True)
# # # mn_logit_data = mn_logit_data[mn_logit_data["Destination"] != "999"]
# #
# # mn_logit_data["Intercept"] = 1.0
# # airport_data = mn_logit_data[["Intercept","Airline", "AccessTime", "DepartureTime", "Age", "Destination", "Province"]]
# # y_airport = mn_logit_data["Airport"].astype('category').cat.codes
# #
# # #airline_data = mn_logit_data[["Intercept","Airport", "DepartureTime", "Airfare", "Destination", "Province", "Nationality"]]
# # airline_data = mn_logit_data[["Airport", "DepartureTime", "Airfare", "Destination", "Province", "Nationality"]]
# # y_airline = mn_logit_data["Airline"].astype('category').cat.codes
# #
# # ####################
# #
# # # mod = sm.MNLogit(formula='Airline ~ Airport + DepartureTime + Airfare + Destination + Province + Nationality + Airfare * Destination', data=mn_logit_data)
# # # res = mod.fit()
# # # print("************************************Airline:*************************",res.summary())
# # ####################
# #
# # # Convert all categorical variables to a matrix of zeros and ones
# # X_airline = pd.get_dummies(airline_data, drop_first=True)
# # print(X_airline.head())
# #
# # # Split the dataset into training and testing data
# # X_train, X_test, y_train, y_test = train_test_split(X_airline,y_airline,test_size =0.3,random_state=101)
# #
# # # ------------------------------------------------------------- #
# # # ---------------------- Design Model ------------------------- #
# # # ----------------------------------- ------------------------- #
# # mn_logit_model = sm.MNLogit(y_train,X_train).fit()
# # print(mn_logit_model.params)
# #
# # # Compute predicted probabilities using validation dataset with Threshold = 0.5
# #
# # y_pred_prob_1 = mn_logit_model.predict(X_test)
# # # X_test.loc[:,'prediction']=0
# # # X_test.loc[y_pred_prob_1 > 0.5,'prediction']=1
# # #print(y_test, y_pred_prob_1)
# # # get_metrics(y_test, y_pred_prob_1)
# #
# # print("***-----------Multinomial Logistic regression Model with sklearn---------------*******")
# #
# # logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
# #
# # # Create an instance of Logistic Regression Classifier and fit the data.
# # logreg.fit(X_train, y_train)
# #
# # y_log = logreg.predict(X_test)
# # print(y_log)
# # print(logreg.coef_)
# #
# # # ------------------------------------------------------------- #
# # # ------------------------ Logistic --------------------------- #
# # # ----------------------------------- ------------------------- #
# # logit_data = raw_data.copy(deep=True)
# #
# # logit_data["Intercept"] = 1.0
# # airport_data =  logit_data[["Intercept","Foreign_Airline", "AccessTime", "DepartureTime", "Age", "Destination", "Province", "Airfare"]]
# # y_airport = logit_data["Airport"].astype('category').cat.codes
# #
# # # -------------------- Airport Model ------------------------##
# #
# # # Convert all categorical variables to a matrix of zeros and ones
# # X_airport = airport_data
# # print(X_airport.head())
# #
# # # Split the dataset into training and testing data
# # X_train_AP, X_test_AP, y_train_AP, y_test_AP = train_test_split(X_airport,y_airport,test_size =0.3,random_state=101)
# #
# # df_train = X_train_AP
# # df_train["Airport"] = y_train_AP
# # #f = 'Airport ~ Foreign_Airline + AccessTime + c(DepartureTime) + Age + c(Destination) + Airfare + c(Destination)*Airfare + c(Province)'
# # f = 'Airport ~ Foreign_Airline + Age + DepartureTime + Airfare + Province + Destination + Destination*Airfare'
# #
# # # Design Model
# # #logit_model_AP = sm.Logit(y_train_AP,X_train_AP).fit()
# # logit_model_AP = smf.logit(formula=f, data=df_train).fit()
# # #logit_model_AP = sm.Logit(y,X).fit()
# #
# # print(logit_model_AP.summary())
# #
# # # Compute predicted probabilities using validation dataset with Threshold = 0.5
# #
# # y_pred_prob_AP = logit_model_AP.predict(X_test_AP)
# # X_test_AP.loc[:,'y_prediction_05']=0
# # X_test_AP.loc[y_pred_prob_AP > 0.5,'y_prediction_05']=1
# #
# # print("***-----------Logistic regression Model with threshold:0.5---------------*******")
# # get_metrics(y_test_AP, X_test_AP["y_prediction_05"])
# #
# # # Compute AUC
# # auc = roc_auc_score(y_test_AP, y_pred_prob_AP)
# # print('AUC: %.2f' % auc)
# # # AUC: 0.81
# #
# # # Plot ROC Curve
# # fpr, tpr, thresholds = roc_curve(y_test_AP, y_pred_prob_AP)
# # plot_roc_curve(fpr, tpr)
# #
# # # Compute predicted probabilities using validation dataset with Threshold = 0.7
# # X_test_AP.loc[:,'y_prediction_07']=0
# # X_test_AP.loc[y_pred_prob_AP > 0.7,'y_prediction_07']=1
# #
# # print("***-----------Logistic regression Model with threshold:0.7---------------*******")
# # get_metrics(y_test_AP, X_test_AP["y_prediction_07"])
# #
# # # Compute predicted probabilities using validation dataset with Threshold = 0.3
# # X_test_AP.loc[:,'y_prediction_03']=0
# # X_test_AP.loc[y_pred_prob_AP > 0.3,'y_prediction_03']=1
# #
# # print("***-----------Logistic regression Model with threshold:0.3---------------*******")
# # get_metrics(y_test_AP, X_test_AP["y_prediction_03"])
# #
# # # -------------------- Airline Model ------------------------##
# #
# # airline_data = logit_data[["Intercept","Airport", "AccessTime", "DepartureTime", "Airfare", "Destination", "Province", "Nationality"]]
# # y_airline = logit_data["Foreign_Airline"].astype('category').cat.codes
# #
# # # Convert all categorical variables to a matrix of zeros and ones
# # #X_airline = pd.get_dummies(airline_data, drop_first=True)
# # #print(X_airline.head())
# # X_airline = airline_data
# #
# # # Split the dataset into training and testing data
# # X_train_AL, X_test_AL, y_train_AL, y_test_AL = train_test_split(X_airline,y_airline,test_size =0.3,random_state=101)
# #
# # df_train = X_train_AL
# # df_train["Foreign_Airline"] = y_train_AL
# # f = 'Foreign_Airline ~ Airport + DepartureTime + Airfare + Nationality + Destination + Destination*Airfare'
# #
# # # Design Model
# # #logit_model_AL = sm.Logit(y_train_AL,X_train_AL).fit()
# # logit_model_AL = smf.logit(formula=f, data=df_train).fit()
# # print(logit_model_AL.summary())
# #
# # # Compute predicted probabilities using validation dataset with Threshold = 0.5
# #
# # y_pred_prob_AL = logit_model_AL.predict(X_test_AL)
# # X_test_AL.loc[:,'y_prediction_05']=0
# # X_test_AL.loc[y_pred_prob_AL > 0.5,'y_prediction_05']=1
# #
# # print("***-----------Logistic regression Model with threshold:0.5---------------*******")
# # get_metrics(y_test_AL, X_test_AL["y_prediction_05"])
# #
# # # Compute AUC
# # auc = roc_auc_score(y_test_AL, y_pred_prob_AL)
# # print('AUC: %.2f' % auc)
# # # AUC: 0.64
# #
# # # Plot ROC Curve
# # fpr, tpr, thresholds = roc_curve(y_test_AL, y_pred_prob_AL)
# # plot_roc_curve(fpr, tpr)
# #
# # # Compute predicted probabilities using validation dataset with Threshold = 0.7
# # X_test_AL.loc[:,'y_prediction_07']=0
# # X_test_AL.loc[y_pred_prob_AL > 0.7,'y_prediction_07']=1
# #
# # print("***-----------Logistic regression Model with threshold:0.7---------------*******")
# # get_metrics(y_test_AL, X_test_AL["y_prediction_07"])
# #
# # # Compute predicted probabilities using validation dataset with Threshold = 0.3
# # X_test_AL.loc[:,'y_prediction_03']=0
# # X_test_AL.loc[y_pred_prob_AL > 0.3,'y_prediction_03']=1
# #
# # print("***-----------Logistic regression Model with threshold:0.3---------------*******")
# # get_metrics(y_test_AL, X_test_AL["y_prediction_03"])
# #
# ------------------------------------------------------------- #
# --------------------- Decision Trees ------------------------ #
# ----------------------------------- ------------------------- #
# DT_data = raw_data.copy(True)
# airport_data = DT_data[["Airline", "DepartureTime", "Age", "Destination", "Province", "Airfare"]]
# y_airport = DT_data["Airport"].astype('category').cat.codes
# airline_data = DT_data[["Airport", "DepartureTime", "Airfare", "Destination", "Province", "Nationality"]]
# y_airline = DT_data["Foreign_Airline"].astype('category').cat.codes
# # Convert all categorical variables to a matrix of zeros and ones
# airport_data = pd.get_dummies(airport_data)
# print(airport_data.head())
# airline_data = pd.get_dummies(airline_data)
# print(airline_data.head())
# # -------------------- Airport Model ------------------------#
# # decision tree model for airport data
# X_train_AP, X_test_AP, y_train_AP, y_test_AP = train_test_split(airport_data,y_airport,test_size =0.3,random_state=101)
# clf_AP = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=109, splitter='best')
# clf_AP = clf_AP.fit(X_train_AP, y_train_AP)
# y_pred_AP = clf_AP.predict(X_test_AP)
# # export estimated tree into dot graphic file
# dot_data = tree.export_graphviz(clf_AP, out_file='DtreeAirport.dot', feature_names=airport_data.columns)
# print("Decision Tree Model: Airport")
# get_metrics(y_test_AP, y_pred_AP)
# # -------------------- Airline Model ------------------------##
# # decision tree model for airline data
# X_train_AL, X_test_AL, y_train_AL, y_test_AL = train_test_split(airline_data,y_airline,test_size =0.3,random_state=101)
# clf_AL = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=109, splitter='best')
# clf_AL = clf_AL.fit(X_train_AL, y_train_AL)
# y_pred_AL = clf_AL.predict(X_test_AL)
# # export estimated tree into dot graphic file
# dot_data = tree.export_graphviz(clf_AL, out_file='DtreeAirline.dot', feature_names=airline_data.columns)
# print("Decision Tree Model: Airline")
# get_metrics(y_test_AL, y_pred_AL)
# # -------------------- Pruned Airport Model ------------------------##
# # decision tree model for airport data
# clf_AP_pruned = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4, max_features=None, max_leaf_nodes=None, min_samples_leaf=2, min_samples_split=10, min_weight_fraction_leaf=0.0, presort=False, random_state=109, splitter='best')
# clf_AP_pruned = clf_AP_pruned.fit(X_train_AP, y_train_AP)
# y_pred_AP_pruned = clf_AP_pruned.predict(X_test_AP)
# # export estimated tree into dot graphic file
# dot_data = tree.export_graphviz(clf_AP_pruned, out_file='DtreeAirport_pruned.dot', feature_names=airport_data.columns)
# print("Pruned Decision Tree Model: Airport")
# get_metrics(y_test_AP, y_pred_AP_pruned)
# # -------------------- Pruned Airline Model ------------------------##
# # decision tree model for airline data
# clf_AL_pruned = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4, max_features=None, max_leaf_nodes=None, min_samples_leaf=4, min_samples_split=20, min_weight_fraction_leaf=0.0, presort=False, random_state=109, splitter='best')
# clf_AL_pruned = clf_AL_pruned.fit(X_train_AL, y_train_AL)
# y_pred_AL_pruned = clf_AL_pruned.predict(X_test_AL)
# # export estimated tree into dot graphic file
# dot_data = tree.export_graphviz(clf_AL_pruned, out_file='DtreeAirline_pruned.dot', feature_names=airline_data.columns)
# print("Pruned Decision Tree Model: Airline")
# get_metrics(y_test_AL, y_pred_AL_pruned)
# # -------------------- GridSearch Airport Model ------------------------##
# parameters_GS_DT = {'max_depth':(3,4,5,6), 'min_samples_leaf':(1,2,4,10), 'min_samples_split':(10,20)}
# dt_model = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_features=None, max_leaf_nodes=None, min_weight_fraction_leaf=0.0, presort=False, random_state=109, splitter='best')
# # decision tree model for airport data
# clf_GS_AP = GridSearchCV(dt_model, parameters_GS_DT, cv = 5)
# clf_GS_AP = clf_GS_AP.fit(X_train_AP, y_train_AP)
# print(clf_GS_AP.best_estimator_)
# clf_GS_AP_refit = clf_GS_AP.best_estimator_
# clf_GS_AP_refit = clf_GS_AP_refit.fit(X_train_AP, y_train_AP)
# y_pred_GS_AP = clf_GS_AP_refit.predict(X_test_AP)
# # export estimated tree into dot graphic file
# dot_data = tree.export_graphviz(clf_GS_AP_refit, out_file='DtreeAirport_pruned_GS.dot', feature_names=airport_data.columns)
# print("GridSearchCV Decision Tree Model: Airport")
# get_metrics(y_test_AP, y_pred_GS_AP)
# # -------------------- GridSearch Airline Model ------------------------##
# parameters_GS_DT = {'max_depth':(3,4,5,6,10), 'min_samples_leaf':(1,2,4,8,10), 'min_samples_split':(10,20)}
# dt_model = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_features=None, max_leaf_nodes=None, min_weight_fraction_leaf=0.0, presort=False, random_state=109, splitter='best')
# # decision tree model for airline data
# clf_GS_AL = GridSearchCV(dt_model, parameters_GS_DT,cv = 5)
# clf_GS_AL = clf_GS_AL.fit(X_train_AL, y_train_AL)
# print(clf_GS_AL.best_estimator_)
# clf_GS_AL_refit = clf_GS_AL.best_estimator_
# clf_GS_AL_refit = clf_GS_AL_refit.fit(X_train_AL, y_train_AL)
# y_pred_GS_AL = clf_GS_AL_refit.predict(X_test_AL)
# # export estimated tree into dot graphic file
# dot_data = tree.export_graphviz(clf_GS_AL_refit, out_file='DtreeAirline_pruned_GS.dot', feature_names=airline_data.columns)
# print("GridSearchCV Decision Tree Model: Airline")
# get_metrics(y_test_AL, y_pred_GS_AL)
# ------------------------------------------------------------- #
# --------------------- Neural Network ------------------------ #
# ------------------------------------------------------------- #
NN_data = raw_data.copy(True)
airport_data = NN_data[["Airline", "DepartureTime", "Age", "Destination", "Province"]]
y_airport = NN_data["Airport"].astype('category').cat.codes
airline_data = NN_data[["Airport", "DepartureTime", "Airfare", "Destination", "Province", "Nationality"]]
y_airline = NN_data["Foreign_Airline"].astype('category').cat.codes
# -------------------- Airport Model ------------------------##
# Convert all categorical variables to a matrix of zeros and ones
airport_data = pd.get_dummies(airport_data)
print(airport_data.head())
## Standardizing data improves computations and makes sure all features are weighted equally for SVMs and NN
scaler = StandardScaler()
airport_data = scaler.fit_transform(airport_data)
## Split the dataset into training and testing data
X_train_AP, X_test_AP, y_train_AP, y_test_AP = train_test_split(airport_data,y_airport,test_size =0.3,random_state=101)
clf_nn_AP = MLPClassifier(hidden_layer_sizes = (10, 10, 10), activation='logistic', random_state=101)
clf_nn_AP.fit(X_train_AP, y_train_AP)
y_pred_nn_AP = clf_nn_AP.predict(X_test_AP)
print("Neural Network Classifier Model Logistic: Airport")
get_metrics(y_test_AP, y_pred_nn_AP)
clf_nn_AP = MLPClassifier(hidden_layer_sizes = (10, 10, 10), activation='identity', random_state=101)
clf_nn_AP.fit(X_train_AP, y_train_AP)
y_pred_nn_AP = clf_nn_AP.predict(X_test_AP)
print("Neural Network Classifier Model Identity: Airport")
get_metrics(y_test_AP, y_pred_nn_AP)
clf_nn_AP = MLPClassifier(hidden_layer_sizes = (20, 20, 20), activation='tanh', random_state=101)
clf_nn_AP.fit(X_train_AP, y_train_AP)
y_pred_nn_AP = clf_nn_AP.predict(X_test_AP)
print("Neural Network Classifier Model Tanh: Airport")
get_metrics(y_test_AP, y_pred_nn_AP)
# -------------------- Airline Model ------------------------##
# Convert all categorical variables to a matrix of zeros and ones
airline_data = pd.get_dummies(airline_data)
print(airline_data.head())
## Standardizing data improves computations and makes sure all features are weighted equally for SVMs and NN
scaler = StandardScaler()
airline_data = scaler.fit_transform(airline_data)
## Split the dataset into training and testing data
X_train_AL, X_test_AL, y_train_AL, y_test_AL = train_test_split(airline_data,y_airline,test_size =0.3,random_state=101)
clf_nn_AL = MLPClassifier(hidden_layer_sizes = (10, 10, 10), activation='logistic', random_state=101)
clf_nn_AL.fit(X_train_AL, y_train_AL)
y_pred_nn_AL = clf_nn_AL.predict(X_test_AL)
print("Neural Network Classifier Model: Airline")
get_metrics(y_test_AL, y_pred_nn_AL)
clf_nn_AL = MLPClassifier(hidden_layer_sizes = (10, 10, 10, 10), activation='logistic', random_state=101)
clf_nn_AL.fit(X_train_AL, y_train_AL)
y_pred_nn_AL = clf_nn_AL.predict(X_test_AL)
print("Neural Network Classifier Model Logistic: Airline")
get_metrics(y_test_AL, y_pred_nn_AL)
clf_nn_AL = MLPClassifier(hidden_layer_sizes = (10, 10, 10), activation='identity', random_state=101)
clf_nn_AL.fit(X_train_AL, y_train_AL)
y_pred_nn_AL = clf_nn_AL.predict(X_test_AL)
print("Neural Network Classifier Model Identity: Airline")
get_metrics(y_test_AL, y_pred_nn_AL)
clf_nn_AL = MLPClassifier(hidden_layer_sizes = (10, 10, 10, 10), activation='identity', random_state=101)
clf_nn_AL.fit(X_train_AL, y_train_AL)
y_pred_nn_AL = clf_nn_AL.predict(X_test_AL)
print("Neural Network Classifier Model Tanh: Airline")
get_metrics(y_test_AL, y_pred_nn_AL)
"""
# NN model for airline data
parameters_GS_NN = {'hidden_layer_sizes':((10,10,10), (20,20,20)), 'activation':('logistic','identity', 'tanh', 'relu')}
NN_model = MLPClassifier(random_state=101)
clf_GS_AP = GridSearchCV(NN_model, parameters_GS_NN,cv = 5)
clf_GS_AP = clf_GS_AP.fit(X_train_AP, y_train_AP)
print(clf_GS_AP.best_estimator_)
clf_GS_AP_refit = clf_GS_AP.best_estimator_
clf_GS_AP_refit = clf_GS_AP_refit.fit(X_train_AP, y_train_AP)
y_pred_GS_AP = clf_GS_AP_refit.predict(X_test_AP)
print("GridSearchCV NN Model: Airport")
get_metrics(y_test_AP, y_pred_GS_AP)
# NN model for airline data
parameters_GS_NN = {'hidden_layer_sizes':((10,10,10), (20,20,20)), 'activation':('logistic','identity', 'tanh', 'relu')}
NN_model = MLPClassifier(random_state=101)
clf_GS_AL = GridSearchCV(NN_model, parameters_GS_NN,cv = 5)
clf_GS_AL = clf_GS_AL.fit(X_train_AL, y_train_AL)
print(clf_GS_AL.best_estimator_)
clf_GS_AL_refit = clf_GS_AL.best_estimator_
clf_GS_AL_refit = clf_GS_AL_refit.fit(X_train_AL, y_train_AL)
y_pred_GS_AL = clf_GS_AL_refit.predict(X_test_AL)
print("GridSearchCV NN Model: Airline")
get_metrics(y_test_AL, y_pred_GS_AL)
"""
# # ------------------------------------------------------------- #
# # -------------------------- SVM ------------------------------ #
# # ------------------------------------------------------------- #
# # Make a copy of raw_data to implement SVM
# svm_data = raw_data.copy(deep=True)
#
# # independent and dependent variables/features for SVM Airport selection
# svm_airport_data = svm_data[["Airline", "DepartureTime", "Age", "Destination", "Province"]]
# y_airport = svm_data["Airport"].astype('category').cat.codes
#
# # independent and dependent variables/features for SVM Airline selection
# svm_airline_data = svm_data[["Airport", "DepartureTime", "Airfare", "Destination", "Province", "Nationality", "Age"]]
# y_airline = svm_data["Foreign_Airline"].astype('category').cat.codes
#
# # ------------------------ Airport Model ---------------------- #
# # Convert all categorical variables to a matrix of zeros and ones
# svm_airport_data = pd.get_dummies(svm_airport_data)
# print(svm_airport_data.head())
#
# # Standardizing data improves computations and makes sure all features are weighted equally for SVMs and NN
# scaler = StandardScaler()
# svm_airport_data = scaler.fit_transform(svm_airport_data)
#
# # Split the dataset into training and testing data
# X_train_AP, X_test_AP, y_train_AP, y_test_AP = train_test_split(svm_airport_data, y_airport, test_size=0.30, random_state=109)
#
# print("\n **** SVM with Kernel = Linear ****")
# svclassifier = SVC(kernel='linear')    	# Linear SVM
# svclassifier.fit(X_train_AP, y_train_AP)
# y_pred_svm_linear_AP = svclassifier.predict(X_test_AP)  	# predict test set
# print("Support Vector Machine Linear Model: Airport")
# get_metrics(y_test_AP, y_pred_svm_linear_AP)
#
# print("\n **** SVM with Kernel = RBF ****")
# svclassifier = SVC(kernel='rbf')    	# RBF SVM
# svclassifier.fit(X_train_AP, y_train_AP)
# y_pred_svm_rbf_AP = svclassifier.predict(X_test_AP)  	# predict test set
# print("Support Vector Machine RBF Model: Airport")
# get_metrics(y_test_AP, y_pred_svm_rbf_AP)
#
#
# print("\n **** SVM with Kernel = Poly with degree 3 ****")
# svclassifier = SVC(kernel='poly', degree=3)    	# cubic polynomial SVM
# svclassifier.fit(X_train_AP, y_train_AP)
# y_pred_svm_poly_AP = svclassifier.predict(X_test_AP)  	# predict test set
# print("Support Vector Machine Poly(3) Model: Airport")
# get_metrics(y_test_AP, y_pred_svm_poly_AP)
#
# # ------------------------ Airline Model ---------------------- #
# # Convert all categorical variables to a matrix of zeros and ones
# svm_airline_data = pd.get_dummies(svm_airline_data)
# print(svm_airline_data.head())
#
# # Standardizing data improves computations and makes sure all features are weighted equally for SVMs and NN
# scaler = StandardScaler()
# svm_airline_data = scaler.fit_transform(svm_airline_data)
#
# # Split the dataset into training and testing data
# X_train_AL, X_test_AL, y_train_AL, y_test_AL = train_test_split(svm_airline_data, y_airline, test_size=0.30, random_state=109)
#
# print("\n **** SVM with Kernel = Linear ****")
# svclassifier = SVC(kernel='linear')    	# Linear SVM
# svclassifier.fit(X_train_AL, y_train_AL)
# y_pred_svm_linear_AL = svclassifier.predict(X_test_AL)  	# predict test set
# print("Support Vector Machine Linear Model: Airline")
# get_metrics(y_test_AL, y_pred_svm_linear_AL)
#
# print("\n **** SVM with Kernel = RBF ****")
# svclassifier = SVC(kernel='rbf')    	# RBF SVM
# svclassifier.fit(X_train_AL, y_train_AL)
# y_pred_svm_rbf_AL = svclassifier.predict(X_test_AL)  	# predict test set
# print("Support Vector Machine RBF Model: Airline")
# get_metrics(y_test_AL, y_pred_svm_rbf_AL)
#
# print("\n **** SVM with Kernel = Poly with degree 3 ****")
# svclassifier = SVC(kernel='poly', degree=3)    	# cubic polynomial SVM
# svclassifier.fit(X_train_AL, y_train_AL)
# y_pred_svm_poly_AL = svclassifier.predict(X_test_AL)  	# predict test set
# print("Support Vector Machine Poly(3) Model: Airline")
# get_metrics(y_test_AL, y_pred_svm_poly_AL)
