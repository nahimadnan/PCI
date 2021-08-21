import numpy as np
from scipy.stats import spearmanr
from math import factorial

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

class ICS():
    """
    A helper class for calculating inter-classifier stability for different model combinations. 
    """
    
    def __init__(self):
        print('ICS object initialization.')
        
    def ICS_for_all_combinations(self, interpretations):
        """
        

        Parameters
        ----------
        interpretations : A 3-D array
            Contains interpretations (i.e., signed and unsigned interpretations) for different models.

        Returns
        -------
        ics_res : An array
            Inter-classifier stability for different model combinations.

        """
        
        ics_res = np.zeros(shape=(combination(interpretations.shape[0], 2)))
        ics_res_idx = 0
        for i in range(interpretations.shape[0]):
            interpretation_x = interpretations[i,:,:]
            for j in range(i+1, interpretations.shape[0]):
                interpretation_y = interpretations[j,:,:]
                
                cc = np.zeros(shape=(interpretations.shape[1]))
                for k in range(interpretations.shape[1]):
                    cc[k], pval = spearmanr(interpretation_x[k,:], interpretation_y[k,:])
                
                ics_res[ics_res_idx] = np.mean(cc)
                ics_res_idx += 1
        
        return ics_res
    
    def ICS_for_dataset_level_all_combinations(self, interpretations):
        """
        

        Parameters
        ----------
        interpretations : A 2-D array
            Contains dataset-level interpretations for different models.

        Returns
        -------
        ics_res : An array
            Dataset-level inter-classifier stability for different model combinations.

        """
        ics_res = np.zeros(shape=(combination(interpretations.shape[1], 2)))
        ics_res_idx = 0
        for i in range(interpretations.shape[1]):
            interpretation_x = interpretations[:,i]
            for j in range(i+1, interpretations.shape[1]):
                interpretation_y = interpretations[:,j]                                
                ics_res[ics_res_idx], pval = spearmanr(interpretation_x, interpretation_y)
                ics_res_idx += 1
        
        return ics_res
        
                    
        