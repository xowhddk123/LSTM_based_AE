import numpy as np
from sklearn.metrics import precision_score, recall_score, r2_score

# precision, recall 구하기


def get_evaluation(label: np.ndarray, anomaly_score: list, threshold: float) -> float:
    '''
    anomaly score로 abnormal
    '''
    anomaly_score = np.array(anomaly_score)
    anomaly_score_length = anomaly_score.shape[0]

    label = label[:anomaly_score_length]

    pred = np.where(anomaly_score > threshold, 1, 0)  # abnormal 예측값
    precision = precision_score(label, pred)
    recall = recall_score(label, pred)
    return precision, recall

# f_score 구하기


def get_fscore(precision: list, recall: list, beta=0.05):
    if (precision == 0) & (recall == 0):
        return 0
    f_score = (1+beta**2)*precision*(recall/((beta**2)*precision+recall))
    return f_score
