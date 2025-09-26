import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(pred, gt):
    """
    Computes precision, recall, and F1 score for binary classification.

    Parameters:
    - pred (numpy array): Predicted labels (binary).
    - gt (numpy array): Ground truth labels (binary).

    Returns:
    - dict: Precision, Recall, and F1 Score.
    """
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    precision = precision_score(gt_flat, pred_flat, average="binary", zero_division=0)
    recall = recall_score(gt_flat, pred_flat, average="binary", zero_division=0)
    f1 = f1_score(gt_flat, pred_flat, average="binary", zero_division=0)

    return precision, recall, f1

def evaluate_all_cities(pred_dir, gt_dir, cities):
    """
    Evaluates precision, recall, and F1 for each city and computes overall scores.

    Parameters:
    - pred_dir (str): Directory where predicted .npy files are stored.
    - gt_dir (str): Directory where ground truth .npy files are stored.
    - cities (list): List of city names.

    Returns:
    - per_city_metrics (dict): Precision, Recall, and F1 for each city.
    - overall_metrics (dict): Micro-averaged Precision, Recall, and F1 across all cities.
    """
    per_city_metrics = {}
    all_preds = []
    all_gts = []

    for city in cities:
        pred_path = os.path.join(pred_dir, f"{city}.npy")
        gt_path = os.path.join(gt_dir, f"{city}.npy")

        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            print(f"Skipping {city}: Missing prediction or ground truth file.")
            continue

        # Load files
        pred = np.load(pred_path)
        gt = np.load(gt_path)

        # Compute per-city metrics
        precision, recall, f1 = compute_metrics(pred, gt)
        per_city_metrics[city] = {"Precision": precision, "Recall": recall, "F1 Score": f1}

        # Store all predictions and ground truths for micro-averaging
        all_preds.append(pred.flatten())
        all_gts.append(gt.flatten())

    # Concatenate all city predictions for micro-averaging
    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)

    # Compute overall (micro-averaged) metrics
    overall_precision = precision_score(all_gts, all_preds, average="binary", zero_division=0)
    overall_recall = recall_score(all_gts, all_preds, average="binary", zero_division=0)
    overall_f1 = f1_score(all_gts, all_preds, average="binary", zero_division=0)

    overall_metrics = {
        "Overall Precision": overall_precision,
        "Overall Recall": overall_recall,
        "Overall F1 Score": overall_f1
    }

    return per_city_metrics, overall_metrics


