from ICS import ICS 
import numpy as np


def print_result(model_combs, res_ics):
    """
    

    Parameters
    ----------
    model_combs : A list of strings
        Strings of model combinations
    res_ics : An array
        Array of inter-classifier stability scores

    Returns
    -------
    None.

    """
    for i in range(len(model_combs)):
        print(model_combs[i] + " : " + str(res_ics[i]))        
    print("-----------------------------")
    print("AVG  : " + str(np.mean(res_ics)))
    print("")
        


## ICS object initialization
ics_obj = ICS()

model_combs = ['RF-LR','RF-lSVM','RF-rSVM','RF-NN','LR-lSVM','LR-rSVM','LR-NN','lSVM-rSVM','lSVM-NN','rSVM-NN']

## Signed interpretations loading for RF, LR, linear kenrnel SVM, rbf kernel SVM and NN models
signed_interpretations = np.zeros(shape=(5, 1616, 37))
signed_interpretations[0,:,:] = np.loadtxt('./result/' + 'ACES_whole_dataset_signed_interpretation_RF.txt')
signed_interpretations[1,:,:] = np.loadtxt('./result/' + 'ACES_whole_dataset_signed_interpretation_LR.txt')
signed_interpretations[2,:,:] = np.loadtxt('./result/' + 'ACES_whole_dataset_signed_interpretation_SVM.txt')
signed_interpretations[3,:,:] = np.loadtxt('./result/' + 'ACES_whole_dataset_signed_interpretation_RBF.txt')
signed_interpretations[4,:,:] = np.loadtxt('./result/' + 'ACES_whole_dataset_signed_interpretation_NN.txt')

## Get ICS score for all model combinations for signed interpretations
signed_interpretation_vector = ics_obj.ICS_for_all_combinations(signed_interpretations)

## Print the result for PCI
print("Result for inter-classifier stability score (ICS) for signed interpretation for PCI")
print_result(model_combs, signed_interpretation_vector)


## Unsigned interpretations loading for RF, LR, linear kenrnel SVM, rbf kernel SVM and NN models
unsigned_interpretations = np.zeros(shape=(5, 1616, 37))
unsigned_interpretations[0,:,:] = np.loadtxt('./result/' + 'ACES_whole_dataset_unsigned_interpretation_RF.txt')
unsigned_interpretations[1,:,:] = np.loadtxt('./result/' + 'ACES_whole_dataset_unsigned_interpretation_LR.txt')
unsigned_interpretations[2,:,:] = np.loadtxt('./result/' + 'ACES_whole_dataset_unsigned_interpretation_SVM.txt')
unsigned_interpretations[3,:,:] = np.loadtxt('./result/' + 'ACES_whole_dataset_unsigned_interpretation_RBF.txt')
unsigned_interpretations[4,:,:] = np.loadtxt('./result/' + 'ACES_whole_dataset_unsigned_interpretation_NN.txt')

## Get ICS score for all model combinations for unsigned interpretations
unsigned_interpretation_vector = ics_obj.ICS_for_all_combinations(unsigned_interpretations)

## Print the result for PCI
print("Result for inter-classifier stability score (ICS) for unsigned interpretation for PCI")
print_result(model_combs, unsigned_interpretation_vector)


## Dataset-level interpretations loading for RF, LR, linear kenrnel SVM, rbf kernel SVM and NN models
dataset_level_interpretations = np.zeros(shape=(37, 5))
dataset_level_interpretations[:, 0] = np.loadtxt('./result/' + 'ACES_dataset_level_interpretation_RF.txt')
dataset_level_interpretations[:, 1] = np.loadtxt('./result/' + 'ACES_dataset_level_interpretation_LR.txt')
dataset_level_interpretations[:, 2] = np.loadtxt('./result/' + 'ACES_dataset_level_interpretation_SVM.txt')
dataset_level_interpretations[:, 3] = np.loadtxt('./result/' + 'ACES_dataset_level_interpretation_RBF.txt')
dataset_level_interpretations[:, 4] = np.loadtxt('./result/' + 'ACES_dataset_level_interpretation_NN.txt')

## Get ICS score for all model combinations for dataset-level interpretations
dataset_level_interpretation_vector = ics_obj.ICS_for_dataset_level_all_combinations(dataset_level_interpretations)

## Print the result for PCI
print("Result for inter-classifier stability score (ICS) for dataset-level interpretation for PCI")
print_result(model_combs, dataset_level_interpretation_vector)
