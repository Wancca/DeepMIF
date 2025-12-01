# process_dataset/DTA/utils/metrics.py
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
# Optional: Use lifelines for CI or implement manually
try:
    # Note: Event indicator is needed for censoring, DTA usually doesn't have it.
    # Assuming y_true are the actual affinity values to be ordered.
    from lifelines.utils import concordance_index as lifelines_ci
    def calculate_ci(y_true, y_pred):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        # lifelines CI expects higher score = higher event time (or higher affinity here)
        # If your y_true (e.g., pKd) is higher=better, this is correct.
        # If your y_true (e.g., Kd) is lower=better, negate y_true: concordance_index(-y_true, y_pred)
        try:
             return lifelines_ci(y_true, y_pred)
        except Exception as e:
             print(f"Error calculating CI with lifelines: {e}. Returning NaN.")
             return np.nan
except ImportError:
    print("Warning: lifelines not installed. Using basic pairwise CI calculation.")
    def calculate_ci(y_true, y_pred):
        # Simplified pairwise comparison implementation (Handles ties as 0.5)
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        n_pairs = 0
        n_correct_pairs = 0.0
        for i in range(len(y_true)):
            for j in range(i + 1, len(y_true)):
                if y_true[i] != y_true[j]: # Only consider pairs with different true values
                    n_pairs += 1
                    # Check concordance
                    if (y_pred[i] < y_pred[j] and y_true[i] < y_true[j]) or \
                       (y_pred[i] > y_pred[j] and y_true[i] > y_true[j]):
                        n_correct_pairs += 1.0
                    # Handle ties in prediction
                    elif y_pred[i] == y_pred[j]:
                        n_correct_pairs += 0.5
        return n_correct_pairs / n_pairs if n_pairs > 0 else np.nan # Return NaN if no comparable pairs


def calculate_rmse(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_pearson(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    valid_idx = ~np.isnan(y_true) & ~np.isnan(y_pred) # Handle potential NaNs
    if np.sum(valid_idx) < 2: return np.nan # Need at least 2 points
    try:
      corr, _ = pearsonr(y_true[valid_idx], y_pred[valid_idx])
      return corr
    except ValueError: # Handle cases like constant input
      return np.nan


def get_dta_metrics(y_true_list, y_pred_list):
    # Flatten lists of batch results
    y_true = np.concatenate([arr.flatten() for arr in y_true_list])
    y_pred = np.concatenate([arr.flatten() for arr in y_pred_list])

    # Remove potential NaN labels from evaluation
    valid_idx = ~np.isnan(y_true)
    if not np.any(valid_idx):
        print("Warning: All labels are NaN in evaluation.")
        return {"rmse": np.nan, "pearson": np.nan, "ci": np.nan}

    y_true_valid = y_true[valid_idx]
    y_pred_valid = y_pred[valid_idx]

    if len(y_true_valid) == 0:
         print("Warning: No valid labels found after NaN removal.")
         return {"rmse": np.nan, "pearson": np.nan, "ci": np.nan}


    rmse = calculate_rmse(y_true_valid, y_pred_valid)
    pcc = calculate_pearson(y_true_valid, y_pred_valid)
    ci = calculate_ci(y_true_valid, y_pred_valid) # Ensure CI handles potential NaNs in preds if not already filtered

    return {"rmse": rmse, "pearson": pcc, "ci": ci}