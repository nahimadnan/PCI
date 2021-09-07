# Probability Change-based Interpretation (PCI)

**PCI.py** is the class for generating signed, unsigned and dataset-level interpretation for any classifier. It can provide single interpretation as well as interpretations for each instance belonging to the dataset.

**PCI_interpretation_generation.py** generates signed, unsigned and dataset-level interpretation for ACES 37 gene clusters dataset for Random Forest (RF), Logistic Regression (LR), linear kernel Support Vector Machine (lSVM), rbf kernel Support Vector Machine (RSVM) and Neural Network (NN).

**ICS.py** is a class for generating inter-classifier stability (ICS) for all combinations of different classifiers.

**PCI_ICS_generation.py** generates the ICS score using the interpretations generated from PCI. This script outputs the ICS score such as RF-LR for signed , unsigned and dataset-level ICS for PCI.

**Model_improvement_PCI.py** generates inter-classifier stability score for each instance belonging to ACES 37 features dataset. The histogram of the inter-classifier stability score is shown in the script. 100 instances with lowest stability scores were removed from the dataset. Then classifiers were trained with the filtered instances of ACES 37 features dataset and tested on NKI 37 features dataset. This script outputs the AUC improvement on NKI dataset using inter-classifier stability of the interpretations generated from PCI.

**'data'** folder contains the dataset and class labels for ACES 37 features and NKI 37 features.

**'result'** folder contains the signed, unsigned and dataset-level interpretations from RF, LR, lSVM, rSVM and NN models.
