import scipy.io as sio
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn import linear_model
from  numpy.matlib import repmat

class PCI():
    """
    Class for Probability Change-based Interpretation method (PCI)
    """
    def __init__(self):
        """
        Deafult PCI object initialization

        Returns
        -------
        None.

        """
        print('PCI object initialization.')
        
    def perturbation(self, test_instance):
        """
        
        
        Parameters
        ----------
        test_instance : An Array
            A single test instance

        Returns
        -------
        perturbed_test_instance : A 2 dimensional matrix 
            A 2 dimensional square matrix NxN, whose N = "# of features" in test_instance 
            and the diagonal are set to 0.

        """
        
        perturbed_test_instance = repmat(test_instance, test_instance.shape[1], 1)
        np.fill_diagonal(perturbed_test_instance, 0)
        return perturbed_test_instance
    
    def unsigned_individual_interpretation(self, train_X, train_y, test_instance, mdl):
        """
        
        
        Parameters
        ----------
        train_X : A 2 dimensional matrix
            Training data for generating unsigned interpretation
        train_y : A 1 dimensional array
            Training class labels for train_X
        test_instance : An array of features
            Test instance whose interpretation needs to be generated
        mdl : A sklearn machine learning model

        Returns
        -------
        unsigned_interpretation : An array
            Returns the unsigned interpretation of the test_instance using mdl.

        """
        
        perturbed_test_instance = self.perturbation(test_instance)
        mdl.fit(train_X, train_y)
        unsigned_interpretation = abs(mdl.predict_proba(perturbed_test_instance)[:,1] - mdl.predict_proba(test_instance)[:,1])
        return unsigned_interpretation
    
    def signed_individual_interpretation(self, train_X, train_y, test_instance, mdl):
        """

        Parameters
        ----------
        train_X : A 2 dimensional matrix
            Training data for generating unsigned interpretation
        train_y : A 1 dimensional array
            Training class labels for train_X
        test_instance : An array of features
            Test instance whose interpretation needs to be generated
        mdl : A sklearn machine learning model

        Returns
        -------
        signed_interpretation : An array
            Returns the signed interpretation of the test_instance using mdl.

        """
        signed_interpretation = np.sign(test_instance)*self.unsigned_individual_interpretation(train_X, train_y, test_instance, mdl)
        return signed_interpretation
    
    def dataset_level_interpretation(self, X, y, mdl):
        """
        

        Parameters
        ----------
        X : A 2 dimensional matrix
            The whole dataset.
        y : An 1 dimensional array
            Class labels for X
        mdl : A sklearn machine learning model            

        Returns
        -------
        dataset_interpretation : An array
            Returns the dataset-level interpretation for X using mdl.

        """
        unsigned_interpretation = np.zeros(shape=(X.shape[0], X.shape[1]))
        
        # Generating LeaveOneOut cross validation to get unsigned individual level interpretation
        # for each instance belonging to the X.
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index].ravel(), y[test_index].ravel()
            
            unsigned_interpretation[test_index[0], :] = self.unsigned_individual_interpretation(X_train, y_train, X_test, mdl)
            # print('current interpretation generation: ' + str(test_index[0]) + ' from ' + str(X.shape[0]))
        dataset_interpretation = np.mean(np.abs(unsigned_interpretation), axis=0)
        return dataset_interpretation
    
    def whole_dataset_unsigned_interpretation(self, X, y, mdl):
        """
        

        Parameters
        ----------
        X : A 2 dimensional matrix
            The whole dataset.
        y : An 1 dimensional array
            Class labels for X
        mdl : A sklearn machine learning model  

        Returns
        -------
        unsigned_interpretation : A 2 dimensional array
            Returns the unsigned interpretation for each instance in X using mdl.

        """
        
        unsigned_interpretation = np.zeros(shape=(X.shape[0], X.shape[1]))
        
        # Generating LeaveOneOut cross validation to get unsigned individual level interpretation
        # for each instance belonging to the X.
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index].ravel(), y[test_index].ravel()
            
            unsigned_interpretation[test_index[0], :] = self.unsigned_individual_interpretation(X_train, y_train, X_test, mdl)
            # print('current interpretation generation: ' + str(test_index[0]) + ' from ' + str(X.shape[0]))
        return unsigned_interpretation
    
    def whole_dataset_signed_interpretation(self, X, y, mdl):
        """
        

        Parameters
        ----------
        X : A 2 dimensional matrix
            The whole dataset.
        y : An 1 dimensional array
            Class labels for X
        mdl : A sklearn machine learning model 

        Returns
        -------
        signed_interpretation : A 2 dimensional array
            Returns the signed interpretation for each instance in X using mdl.

        """
        
        signed_interpretation = np.zeros(shape=(X.shape[0], X.shape[1]))
        
        # Generating LeaveOneOut cross validation to get unsigned individual level interpretation
        # for each instance belonging to the X.
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index].ravel(), y[test_index].ravel()
            
            signed_interpretation[test_index[0], :] = self.signed_individual_interpretation(X_train, y_train, X_test, mdl)
            # print('current interpretation generation: ' + str(test_index[0]) + ' from ' + str(X.shape[0]))
        return signed_interpretation
        
        
        
