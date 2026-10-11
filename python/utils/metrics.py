import numpy as np
from utils.validation import grades


def ratio(numerator, denominator):
    """Undefined rates are NaN internally and JSON null in published artifacts."""
    return float(numerator / denominator) if denominator else float('nan')

def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa (QWK) from scratch.
    """
    y_true = grades(y_true, 'y_true')
    y_pred = grades(y_pred, 'y_pred')

    if y_true.size != y_pred.size:
        raise ValueError('y_true and y_pred must have the same length')
    if y_true.size == 0:
        return float('nan')

    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError('ratings must be finite numbers')

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
        
    min_rating = int(min(y_true.min(), y_pred.min()))
    max_rating = int(max(y_true.max(), y_pred.max()))
    
    num_ratings = int(max_rating - min_rating + 1)

    # A constant, perfectly matching batch has no disagreement.  Handling it
    # explicitly avoids division by zero in the weight matrix below.
    if num_ratings == 1:
        return float('nan')
    
    conf_mat = np.zeros((num_ratings, num_ratings))
    for a, p in zip(y_true, y_pred):
        conf_mat[int(a - min_rating)][int(p - min_rating)] += 1
        
    num_scored_items = float(y_true.size)
    
    hist_true = np.zeros(num_ratings)
    hist_pred = np.zeros(num_ratings)
    for a, p in zip(y_true, y_pred):
        hist_true[int(a - min_rating)] += 1
        hist_pred[int(p - min_rating)] += 1
        
    expected_mat = np.outer(hist_true, hist_pred) / num_scored_items
    
    weight_mat = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weight_mat[i][j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)
            
    num = np.sum(weight_mat * conf_mat)
    den = np.sum(weight_mat * expected_mat)
    
    if den == 0:
        return float('nan')
    
    return 1.0 - (num / den)

def compute_metrics(confusion_matrix):
    """
    Computes sensitivity, specificity, precision, F1 per class from a confusion matrix.
    """
    confusion_matrix = np.asarray(confusion_matrix)
    if confusion_matrix.ndim != 2 or confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError('confusion_matrix must be a square 2-D array')
    if not np.isfinite(confusion_matrix).all() or (confusion_matrix < 0).any() or (
            confusion_matrix != np.floor(confusion_matrix)).any():
        raise ValueError('confusion_matrix must contain nonnegative integer counts')

    metrics = {}
    num_classes = confusion_matrix.shape[0]
    
    sensitivities = []
    specificities = []
    precisions = []
    f1_scores = []
    
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fn = np.sum(confusion_matrix[i, :]) - tp
        fp = np.sum(confusion_matrix[:, i]) - tp
        tn = np.sum(confusion_matrix) - (tp + fp + fn)
        
        sensitivity = ratio(tp, tp + fn)
        specificity = ratio(tn, tn + fp)
        precision = ratio(tp, tp + fp)
        f1 = ratio(2 * tp, 2 * tp + fp + fn)
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        precisions.append(precision)
        f1_scores.append(f1)
        
    metrics['sensitivity'] = sensitivities
    metrics['specificity'] = specificities
    metrics['precision'] = precisions
    metrics['f1'] = f1_scores
    
    return metrics

def referable_dr_metrics(y_true, y_pred):
    """
    Binary metrics for level 0-1 vs 2-4.
    """
    y_true, y_pred = grades(y_true, 'y_true'), grades(y_pred, 'y_pred')
    if len(y_true) != len(y_pred):
        raise ValueError('y_true and y_pred must have the same length')

    y_true_binary = [1 if y >= 2 else 0 for y in y_true]
    y_pred_binary = [1 if y >= 2 else 0 for y in y_pred]
    
    tp = sum([1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 1 and yp == 1])
    tn = sum([1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 0 and yp == 0])
    fp = sum([1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 0 and yp == 1])
    fn = sum([1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 1 and yp == 0])
    
    sensitivity = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    precision = ratio(tp, tp + fp)
    accuracy = ratio(tp + tn, tp + tn + fp + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'accuracy': accuracy,
        'f1': f1
    }
