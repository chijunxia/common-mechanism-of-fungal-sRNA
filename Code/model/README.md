3_abstract_false_saple.py：Extract negative samples

MTE.py:Free energy extraction

sample_feature.py:Extract the features of the model

feature.py:
Some information about the features.
feature: Feature name and class.
details: The detailed name and details and class of the feature.
cols_data:  Feature name.
all_model_10：The cross corresponding accuracy, recall, F1 and precision of each model
all_model_parm: Optimal parameters of all models.

feature_select.py、feature_selector.py :Filter out unimportant features

sample_to_onehot.py:Onehot encoding of features

model_xin.py:
import feature
Model cross validation selects the optimal parameters and carries out the classification prediction of the model
The accuracy, recall, accuracy and F1 value of the model were calculated
ROC-AUC further evaluates the model.

ModlePre.py:
Import feature.py The purpose is to import the first line of the feature, which is the feature name
Reading Feature and model/ OptMfe.txt file
Reading Feature and model/ Optsample_ seq.fasta
Function: feature extraction and onehot coding
Write Feature and model

plot.py:
from feature import all_model_parm
Comparing the classification effect of the three models, the accuracy, recall, accuracy and F1 value of the selected optimal parameters are plotted respectively
 
