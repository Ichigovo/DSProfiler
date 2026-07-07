import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
    confusion_matrix, average_precision_score,
    precision_recall_curve, auc
)

# 读取数据
df = pd.read_csv('./example/score/example_results.csv')
    
targets = df['label'].values
scores = df['score'].values

threshold = 0.5
preds = (scores >= threshold).astype(int)

precision, recall, _ = precision_recall_curve(targets, scores)
auprc = auc(recall, precision)

metrics = {
    'accuracy': accuracy_score(targets, preds),
    'f1_score': f1_score(targets, preds),
    'mcc': matthews_corrcoef(targets, preds),
    'precision': precision_score(targets, preds),
    'recall': recall_score(targets, preds),
    'auc_roc': roc_auc_score(targets, scores),
    'auc_prc': auprc,
    'confusion_matrix': confusion_matrix(targets, preds).tolist()
}

# 输出结果
print("\n" + "="*50)
print("Model Evaluation Metrics:")
print("="*50)
for key, value in metrics.items():
    if key != 'confusion_matrix':
        print(f"{key:25}: {value:.4f}")
    else:
        print(f"\nConfusion Matrix:")
        cm = np.array(value)
        print(f"                  Pred Negative  Pred Positive")
        print(f"Actual Negative:  {cm[0][0]:13}  {cm[0][1]:13}")
        print(f"Actual Positive:  {cm[1][0]:13}  {cm[1][1]:13}")

