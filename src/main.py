from __future__ import print_function
import numpy as np
import pandas as pd

from helperfunctions import RBF
from vgpmil import vgpmil

from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
import tensorflow as tf
from time import time

# =============================================================================
# DUMMY
# Xtrain = pd.DataFrame(pd.read_excel('prueba_dummy_Train_xin.xlsx')).to_numpy().astype(np.float32)
# InstBagLabel = np.transpose(pd.DataFrame(pd.read_excel('prueba_dummy_Train_yin.xlsx')).to_numpy().astype(int))[0]
# Bags2= np.transpose(pd.DataFrame(pd.read_excel('prueba_dummy_Train_bags.xlsx')).to_numpy().astype(int))[0]
# Xtest = pd.DataFrame(pd.read_excel('prueba_dummy_Test_xin.xlsx')).to_numpy().astype(np.float32)
# =============================================================================

DataFrame = pd.DataFrame(pd.read_csv('CT_feature_train.csv'))

Xtrain = DataFrame.iloc[:, 6:14].to_numpy().astype("float32")# features columns

instance_label =  (DataFrame['groundtruth (instance)'].to_numpy().astype("int"))# instance_label column
pi = np.random.uniform(0, 0.1, size = len(DataFrame)) # -1 for untagged
pi = np.where( (0 == instance_label), 0, pi)
pi = np.where( ( 0 < instance_label), 1, pi)

# wsi_labels = DataFrame['Scan_label (bag)'].to_numpy().astype('str')
# InstBagLabel = np.where( wsi_labels == '[-2 -2]', 0, 1)

wsi_labels = DataFrame['Scan_label (bag)'].to_numpy().astype('int')
InstBagLabel = wsi_labels

mask = np.where(instance_label > -1, False, True)

Bags = DataFrame['Scan'].to_numpy().astype('str')# wsi column 



Xtest = pd.DataFrame(pd.read_csv('CT_feature_test.csv')).iloc[:, 6:14].to_numpy().astype("float32")# features columns

#kernel and parameters
kernel = RBF()

inducing = 800
iterations = 1200
norm = True
verb = False
Z = None

print('')
print('num_inducing = ', inducing)
print('max_iter = ', iterations)
print('normalize = ', norm)


start = time()


#lclass vgpmil()
mil = vgpmil(kernel, num_inducing=inducing, max_iter=iterations, normalize=norm, verbose=verb)

#mil.initialize(Xtrain, InstBagLabel, Bags)
mil.train(Xtrain, InstBagLabel, Bags, Z, pi, mask)

pred_VGPMIL = mil.predict(Xtest)




#Confusion MAtrix (VGPMIL vs True)
true_labels = pd.DataFrame(pd.read_csv('CT_feature_test.csv'))['groundtruth (instance)'].to_numpy().astype("float32")
#true_labels = np.transpose(np.where(true_labelsDF > 0, 1, true_labelsDF))
#del InstTrueAux

Predictions_VGPMIL = np.where(pred_VGPMIL >= 0.5, 1, 0).astype("float32")


ConfMatrix_VGPMILvsTrue = confusion_matrix(true_labels, Predictions_VGPMIL)

print('')
print('groundtruth (instance) vs Pred_VGPMIL:')

print('Recall =', round(ConfMatrix_VGPMILvsTrue[0][0] / (ConfMatrix_VGPMILvsTrue[0][0] + ConfMatrix_VGPMILvsTrue[1][0]), 3 ))

print('Precision =', round(ConfMatrix_VGPMILvsTrue[0][0] / (ConfMatrix_VGPMILvsTrue[0][0] + ConfMatrix_VGPMILvsTrue[0][1]) ,3))

print('Accuracy =', round((ConfMatrix_VGPMILvsTrue[0][0] + ConfMatrix_VGPMILvsTrue[1][1]) / (ConfMatrix_VGPMILvsTrue[0][0] + ConfMatrix_VGPMILvsTrue[1][0] + ConfMatrix_VGPMILvsTrue[0][1] + ConfMatrix_VGPMILvsTrue[1][1]) ,3)) 

print('F1 score =', round(f1_score(true_labels, Predictions_VGPMIL),3) )

print('Cohen Kappa score =', round(cohen_kappa_score(true_labels, Predictions_VGPMIL),3) )


#F1 score
#cohen_kappa_score(.., weights='quadratic') para multiclass


#Confusion MAtrix (CNN vs True)
CNNPred = pd.DataFrame(pd.read_csv('CT_feature_test.csv'))['prediction (instance)'].to_numpy().astype("float32")
#CNNPred = np.transpose(np.where(CNNPredAux > 0, 1, CNNPredAux))
#del CNNPredAux

ConfMatrix_CNNvsTrue = confusion_matrix(true_labels, CNNPred)

print('')
print('groundtruth (instance) vs prediction (instance)):')

print('Recall =', round(ConfMatrix_CNNvsTrue[0][0] / (ConfMatrix_CNNvsTrue[0][0] + ConfMatrix_CNNvsTrue[1][0]) ,3))

print('Precision =', round(ConfMatrix_CNNvsTrue[0][0] / (ConfMatrix_CNNvsTrue[0][0] + ConfMatrix_CNNvsTrue[0][1]) ,3))

print('Accuracy =', round((ConfMatrix_CNNvsTrue[0][0] + ConfMatrix_CNNvsTrue[1][1]) / (ConfMatrix_CNNvsTrue[0][0] + ConfMatrix_CNNvsTrue[1][0] + ConfMatrix_CNNvsTrue[0][1] + ConfMatrix_CNNvsTrue[1][1]) ,3)) 

print('F1 score =', round(f1_score(true_labels, CNNPred) ,3))

print('Cohen Kappa score =', round(cohen_kappa_score(true_labels, CNNPred) ,3))



#comparacion
Comp = np.array([true_labels.astype("float32"), Predictions_VGPMIL.astype("float32")]).T
CNNcomp = np.array([true_labels.astype("float32"), CNNPred.astype("float32")]).T



stop = time()
print(' ')
print("\tMinutes needed:\t", (stop - start) / 60.)

Results = np.array([true_labels.astype("float32"), Predictions_VGPMIL.astype("float32"), CNNPred.astype("float32")]).T
pd.DataFrame(Results).to_csv("Results.csv", header=["groundtruth", "VGPMIL", "prediction (instance)"], index=False)
