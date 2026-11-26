import math
import numpy as np
from sklearn.metrics import mean_squared_error
from math import log10, sqrt

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def net_evaluation_(sp, act):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(p.shape[0]):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = np.squeeze(sum(Tp))
    fp = np.squeeze(sum(Fp))
    tn = np.squeeze(sum(Tn))
    fn = np.squeeze(sum(Fn))

    Dice = (2 * tp) / ((2 * tp) + fp + fn) * 100
    Jaccard = tp / (tp + fp + fn) * 100
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    psnr = PSNR(sp, act)
    mse = mean_squared_error(sp, act)
    IoU = tp / (tp + fp + fn) * 100
    sensitivity = tp / (tp + fn) * 100
    specificity = tn / (tn + fp) * 100
    precision = tp / (tp + fp) * 100
    FPR = fp / (fp + tn) * 100
    FNR = fn / (tp + fn) * 100
    NPV = tn / (tn + fp) * 100
    FDR = fp / (tp + fp) * 100
    F1_score = (2 * tp) / (2 * tp + fp + fn) * 100
    MCC = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    EVAL = [tp, tn, fp, fn, Dice, Jaccard, accuracy, psnr, mse, IoU, sensitivity, specificity, precision, FPR, FNR, NPV, FDR, F1_score,
            MCC]
    return EVAL


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def DSC(TP, FP, FN):
    # Dice Similarity Coefficient
    dsc = (2 * TP) / (2 * TP + FP + FN)
    return dsc * 100

def LE(TP, FP, FN):
    # Localization Error
    iou = TP / (TP + FP + FN)
    le = 1 - iou
    return le * 100

def CLE(actual, predict):
    actual = np.asarray(actual)
    predict = np.asarray(predict)
    actual_points = np.argwhere(actual == 1)
    predict_points = np.argwhere(predict == 1)
    # No foreground object
    if len(actual_points) == 0 and len(predict_points) == 0:
        return 0.0
    if len(actual_points) == 0 or len(predict_points) == 0:
        return float("inf")
    # Centroid of actual mask
    actual_center = np.mean(actual_points, axis=0)
    # Centroid of predicted mask
    predict_center = np.mean(predict_points, axis=0)
    # Euclidean distance
    cle = np.sqrt(
        np.sum((actual_center - predict_center) ** 2)
    )
    return cle

def NWD(actual, predict, C=12.8):

    actual = np.asarray(actual)
    predict = np.asarray(predict)

    actual_points = np.argwhere(actual == 1)
    predict_points = np.argwhere(predict == 1)

    # Both masks are empty
    if len(actual_points) == 0 and len(predict_points) == 0:
        return 1.0

    # One mask is empty
    if len(actual_points) == 0 or len(predict_points) == 0:
        return 0.0

    # Centroids
    actual_center = np.mean(actual_points, axis=0)
    predict_center = np.mean(predict_points, axis=0)

    # Euclidean Wasserstein-like distance between centroids
    distance = np.sqrt(
        np.sum((actual_center - predict_center) ** 2)
    )

    # Normalized Wasserstein similarity
    nwd = np.exp(-distance / C)

    return nwd

from sklearn.metrics import average_precision_score

def AP(actual, predict_probability):
    actual = np.asarray(actual).flatten()
    predict_probability = np.asarray(predict_probability).flatten()

    ap = average_precision_score(
        actual,
        predict_probability
    )
    return ap * 100

def mAP(actual, predict_probability):
    actual = np.asarray(actual)
    predict_probability = np.asarray(predict_probability)
    aps = []
    for i in range(actual.shape[-1]):
        y_true = actual[..., i].flatten()
        y_score = predict_probability[..., i].flatten()

        ap = average_precision_score(
            y_true,
            y_score
        )
        aps.append(ap)
    return np.mean(aps) * 100


def net_evaluation(act, sp):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(p.shape[0]):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = np.squeeze(sum(Tp))
    fp = np.squeeze(sum(Fp))
    tn = np.squeeze(sum(Tn))
    fn = np.squeeze(sum(Fn))

    Dice = (2 * tp) / ((2 * tp) + fp + fn) * 100
    Jaccard = tp / (tp + fp + fn) * 100
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    psnr = PSNR(sp, act)
    mse = mean_squared_error(sp, act)
    IoU = tp / (tp + fp + fn) * 100
    sensitivity = tp / (tp + fn) * 100
    specificity = tn / (tn + fp) * 100
    precision = tp / (tp + fp) * 100
    FPR = fp / (fp + tn) * 100
    FNR = fn / (tp + fn) * 100
    NPV = tn / (tn + fp) * 100
    FDR = fp / (tp + fp) * 100
    F1_score = (2 * tp) / (2 * tp + fp + fn) * 100
    MCC = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    dsc = DSC(tp, fp, fn)
    le = LE(tp, fp, fn)
    cle = CLE(act, sp)
    nwd = NWD(act, sp)
    MAP = mAP(act, sp)
    # EVAL = [tp, tn, fp, fn, Dice, Jaccard, accuracy, psnr, mse, IoU, sensitivity, specificity, precision, FPR, FNR, NPV,
    #         FDR, F1_score, MCC, dsc, le, cle, nwd, MAP]
    EVAL = [tp, tn, fp, fn, Dice, Jaccard, accuracy, psnr, mse, IoU, sensitivity, specificity, precision, FPR, FNR, NPV,
            FDR, F1_score, MCC]  # , dsc, le, cle, nwd, MAP]
    return EVAL


def NegativeLivelihoodRatio(tnr, fnr):
    # Negative Likelihood Ratio (LR-)
    lrminus = tnr / fnr
    return lrminus


def DOR(lrplus, lrminus):
    # Diagnostic Odds Ratio (DOR)
    dor = lrplus / lrminus
    return dor


def evaluation(sp, act):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(act.shape[0]):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(len(p)):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = np.squeeze(sum(Tp))
    fp = np.squeeze(sum(Fp))
    tn = np.squeeze(sum(Tn))
    fn = np.squeeze(sum(Fn))

    accuracy = ((tp + tn) / (tp + tn + fp + fn)) * 100
    sensitivity = (tp / (tp + fn)) * 100
    specificity = (tn / (tn + fp)) * 100
    precision = (tp / (tp + fp)) * 100
    FPR = (fp / (fp + tn)) * 100
    FNR = (fn / (tp + fn)) * 100
    NPV = (tn / (fn + tn)) * 100
    TNR = (tn / (fp + tn)) * 100
    FOR = (fn / (fn + tn)) * 100
    FDR = (fp / (tp + fp)) * 100
    F1_score = ((2 * tp) / (2 * tp + fp + fn)) * 100
    MCC = (((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) * 100
    pt = math.sqrt(FPR) / (math.sqrt(sensitivity) + math.sqrt(sensitivity))
    ba = (sensitivity + specificity) / 2
    fm = math.sqrt(sensitivity * precision)
    bm = sensitivity + specificity - 100
    mk = precision + NPV - 100
    PLHR = sensitivity / FPR
    lrminus = NegativeLivelihoodRatio(specificity, FNR)
    dor = (tp * tn) / (fp * fn)
    prevalence = ((tp + fn) / (tp + fp + tn)) * 100
    TS = (tp / (tp + fn + fp)) * 100
    EVAL = [tp, tn, fp, fn, accuracy, sensitivity, specificity, precision, FPR, FNR, FOR, NPV, FDR, F1_score, MCC, pt,
            ba, fm, bm, mk, PLHR, lrminus, dor, prevalence, TS]
    return EVAL