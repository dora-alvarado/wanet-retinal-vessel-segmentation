from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np

def performance(y_scores, y_true, threshold = 0.5):
    # AUC-ROC
    fpr, tpr, thresholds = roc_curve((y_true), y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    # AUC-Prec-Rec
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    y_pred = np.asarray(y_scores >= threshold, dtype=np.int)  # np.empty((y_scores.shape[0]))

    confusion = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion.ravel()
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy += float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    sensitivity = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    precision = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])

    acc_index = accuracy_score(y_true, y_pred)

    # F1 score
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)

    metrics = {
        'AUC_ROC': AUC_ROC,
        'AUC_PrR': AUC_prec_rec,
        'F1_score': F1_score,
        #'Confusion_matrix': confusion,
        'Accuracy': acc_index,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision
    }

    return metrics