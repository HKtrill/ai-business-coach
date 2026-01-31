import numpy as np
from sklearn.metrics import recall_score

def compute_hybrid_weights(lr_cal, ebm_cal, glass_cal,
                           y_train, lr_prob, ebm_prob,
                           glass_pred, lr_thresh, ebm_thresh,
                           alpha=0.5):

    lr_rec = recall_score(y_train, (lr_prob >= lr_thresh).astype(int), zero_division=0)
    ebm_rec = recall_score(y_train, (ebm_prob >= ebm_thresh).astype(int), zero_division=0)
    glass_rec = recall_score(y_train, glass_pred, zero_division=0)

    ece = np.array([lr_cal['ece'], ebm_cal['ece'], glass_cal['ece']])
    calib = 1 / (ece + 0.01)
    calib /= calib.sum()

    recall = np.array([lr_rec, ebm_rec, glass_rec])
    recall /= recall.sum()

    hybrid = alpha * calib + (1 - alpha) * recall
    hybrid /= hybrid.sum()

    return {
        'lr': hybrid[0],
        'ebm': hybrid[1],
        'glass': hybrid[2],
        'details': {
            'calibration': calib.tolist(),
            'recall': recall.tolist()
        }
    }
