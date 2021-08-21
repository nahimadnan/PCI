from PCI import PCI
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import scipy.io as sio
from sklearn.metrics.ranking import roc_auc_score



def generate_and_save_all_interpretations_for_model(pci_obj, X, y, mdl, mdl_name):
    """
    A helper function to generate signed, unsigned and dataset-level interpretations for a dataset.

    Parameters
    ----------
    pci_obj : PCI object

    X : A 2-D array
        A dataset.
    y : An array
        Class labels for X.
    mdl : A sklearn machine learning model
    mdl_name : String
        Name of the model.

    Returns
    -------
    None.

    """
    
    ## Generate dataset-level interpretation for Logistic Regression model
    print('Dataset-level interpretation generation is in process for ' + mdl_name + '.')
    dataset_level_interpretation = pci_obj.dataset_level_interpretation(X, y, mdl)
    np.savetxt('./result/' + 'ACES_dataset_level_interpretation_' + mdl_name + '.txt', dataset_level_interpretation, fmt='%1.8e')
    print('Dataset-level interpretation generation is finished for' + mdl_name + '.')
    
    
    ## Generate whole dataset signed interpretation for Logistic Regression model
    print('Whole dataset signed interpretation generation is in process for ' + mdl_name + '.')
    whole_dataset_signed_interpretation = pci_obj.whole_dataset_signed_interpretation(X, y, mdl)
    np.savetxt('./result/' + 'ACES_whole_dataset_signed_interpretation_' + mdl_name + '.txt', whole_dataset_signed_interpretation, fmt='%1.8e')
    print('Whole dataset signed interpretation generation is finished for  ' + mdl_name + '.')
    
    ## Generate whole dataset signed interpretation for Logistic Regression model
    print('Whole dataset unsigned interpretation generation is in process for  ' + mdl_name + '.')
    whole_dataset_unsigned_interpretation = pci_obj.whole_dataset_unsigned_interpretation(X, y, mdl)
    np.savetxt('./result/' + 'ACES_whole_dataset_unsigned_interpretation_' + mdl_name + '.txt', whole_dataset_unsigned_interpretation, fmt='%1.8e')
    print('Whole dataset unsigned interpretation generation is finished for  ' + mdl_name + '.')
    



## Locading ACES expression data for 37 features and class labels
X = sio.loadmat('./data/' + 'ACES_RefinedCommunity_AVG.mat')
X = X['data']
y = sio.loadmat('./data/' + 'ACESLabel.mat')
y = y['label']


## PCI object initialization
pci_obj = PCI()

## Logistic regression model initialization and save interpretations
mdl = linear_model.LogisticRegression(C = 1.0, class_weight='balanced')
generate_and_save_all_interpretations_for_model(pci_obj, X, y, mdl, 'LR')


## Random Forest model initialization and save interpretations
mdl = RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced', n_jobs=-1)
generate_and_save_all_interpretations_for_model(pci_obj, X, y, mdl, 'RF')


## Linear kernel based support vector machine model initialization and save interpretations
mdl = SVC(C=1, kernel='linear', class_weight='balanced', probability=True)
generate_and_save_all_interpretations_for_model(pci_obj, X, y, mdl, 'SVM')


## Radial basis kernel based support vector machine model initialization and save interpretations
mdl = SVC(C=1, kernel='rbf', class_weight='balanced', probability=True)
generate_and_save_all_interpretations_for_model(pci_obj, X, y, mdl, 'RBF')

## Multilayer neural Network model initialization and save interpretations
## Hidden Layer size is initialized to the number of features in the dataset
mdl = MLPClassifier(hidden_layer_sizes=(X.shape[1]), max_iter=10000, alpha=0.0001, solver='adam', random_state=21, tol=1e-4)
generate_and_save_all_interpretations_for_model(pci_obj, X, y, mdl, 'NN')


