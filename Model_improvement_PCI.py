import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import scipy.io as sio
from math import factorial
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics.ranking import roc_auc_score


def combination(n, r):
    """
    Helper function

    Parameters
    ----------
    n : int
    r : int

    Returns
    -------
    TYPE : int
        Returns the number of combinations for n choose r.

    """
    return factorial(n) // factorial(r) // factorial(n-r)


def model_improvement_result(X_aces, y_aces, X_aces_filtered, y_aces_filtered, X_nki, y_nki, mdl, mdl_name):
    """
    A helper function for showing the improvement in AUC for a specific model after 
    filtering some instances in ACES and tested on NKI dataset

    Parameters
    ----------
    X_aces : A 2-D array
        Whole ACES training data.
    y_aces : An array
        Class labels for X_aces.
    X_aces_filtered : A 2-D array
        ACES training data after filtering instances.
    y_aces_filtered : An Array
        Class labels for X_aces_filtered.
    X_nki : A 2-D array
        Whole training data for NKI.
    y_nki : An Array
        Class labels for X_nki.
    mdl : A sklearn machine learning model
    mdl_name : String
        Name of the model.

    Returns
    -------
    None.

    """
    mdl.fit(X_aces, y_aces)
    auc_using_all_instance = roc_auc_score(y_nki, mdl.predict_proba(X_nki)[:,1])
    mdl.fit(X_aces_filtered, y_aces_filtered)
    auc_using_filtered_instance = roc_auc_score(y_nki, mdl.predict_proba(X_nki)[:,1])
    print( mdl_name + "  :" + str(auc_using_all_instance) + "---" + str(auc_using_filtered_instance))
    


def get_interpretation_stability_score(interpretations):
    """
    Helper function for generating average inter-classifier stability scores

    Parameters
    ----------
    interpretations : A 3-D array
        Contains interpretations for different models.

    Returns
    -------
    ics_scores : An Array
        Returns average inter-classifier stability scores (ICS scores) for each instance.

    """
    ics_score_comb = np.zeros(shape=(interpretations.shape[1], combination(interpretations.shape[0], 2)))
    
    ics_score_col_idx = 0
    
    for i in range(interpretations.shape[0]):
        interpretation_x = interpretations[i,:,:]
        for j in range(i+1, interpretations.shape[0]):
            interpretation_y = interpretations[j,:,:]
                
            cc = np.zeros(shape=(interpretations.shape[1]))
            for k in range(interpretations.shape[1]):
                cc[k], pval = pearsonr(interpretation_x[k,:], interpretation_y[k,:])
                
            ics_score_comb[:, ics_score_col_idx] = abs(cc)
            ics_score_col_idx +=  1
    ics_scores = np.mean(ics_score_comb, axis=1)
    return ics_scores


## Signed interpretations loading for RF, LR, linear kenrnel SVM, and rbf kernel SVM models
signed_interpretations = np.zeros(shape=(4, 1616, 37))
signed_interpretations[0,:,:] = np.loadtxt('./result/' + 'ACES_whole_dataset_signed_interpretation_RF.txt')
signed_interpretations[1,:,:] = np.loadtxt('./result/' + 'ACES_whole_dataset_signed_interpretation_LR.txt')
signed_interpretations[2,:,:] = np.loadtxt('./result/' + 'ACES_whole_dataset_signed_interpretation_SVM.txt')
signed_interpretations[3,:,:] = np.loadtxt('./result/' + 'ACES_whole_dataset_signed_interpretation_RBF.txt')

## Get the average inter-classifier stability (ICS scores) for six pairwise combinations
avg_ics_scores = get_interpretation_stability_score(signed_interpretations)

## View the histogram of the avg_ics_scores
n, bins, patches = plt.hist(x=avg_ics_scores, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Inter-Classifier stability')
plt.ylabel('Patient count')
plt.title('Distribution  of  average  correlation  between  PCI  interpretations  from  six  pairwise  classifier combinations for each instance in ACES.')


## Find index for lowest avg ICS scores
idx = np.argsort(avg_ics_scores)

## Generate filtered index for ACES by removing 100 samples
filtered_idx = idx[100:]

## Load ACES data for 37 features
expr = sio.loadmat('./data/' + 'ACES_RefinedCommunity_AVG.mat')
X_aces = expr['data']
label = sio.loadmat('./data/' + 'ACESLabel.mat')
y_aces = label['label'].ravel()

## Create ACES filtered data
X_aces_filtered = X_aces[filtered_idx, :]
y_aces_filtered = y_aces[filtered_idx].ravel()

## Load NKI data for 37 features
expr = sio.loadmat('./data/' + 'Vijver_RefinedCommunity_AVG.mat')
X_nki = expr['data']
label = sio.loadmat('./data/' + 'VijverLabel.mat')
y_nki = label['label'].ravel()


print("     AUC using all instances---AUC using filtered instances")

## Model Improvement using RF
rf_mdl = RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced', n_jobs=-1)
model_improvement_result(X_aces, y_aces, X_aces_filtered, y_aces_filtered, X_nki, y_nki, rf_mdl, 'RF  ')


## Model Improvement using LR
lr_mdl = linear_model.LogisticRegression(C = 1.0, class_weight='balanced')
model_improvement_result(X_aces, y_aces, X_aces_filtered, y_aces_filtered, X_nki, y_nki, lr_mdl, 'LR  ')


## Model Improvement using linear kernel SVM
lsvm_mdl = SVC(C=1, kernel='linear', class_weight='balanced', probability=True)
model_improvement_result(X_aces, y_aces, X_aces_filtered, y_aces_filtered, X_nki, y_nki, lsvm_mdl, 'lSVM')


## Model Improvement using rbf kernel SVM
rsvm_mdl = SVC(C=1, kernel='rbf', class_weight='balanced', probability=True)
model_improvement_result(X_aces, y_aces, X_aces_filtered, y_aces_filtered, X_nki, y_nki, rsvm_mdl, 'rSVM')