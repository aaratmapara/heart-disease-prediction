# Heart Disease Prediction with Machine Learning
Using Python (pandas, seaborn, scikit-learn) for EDA, classification models, and evaluation.

## Overview
This project applies machine learning models to the **Cleveland Heart Disease dataset** (303 patients) to predict the presence of heart disease.  
We perform exploratory data analysis (EDA), train multiple models, and compare their performance to identify the most effective approach for this binary classification task.

## Dataset
- Source: [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)  
- 303 patient records with 14 attributes (age, cholesterol, blood pressure, max heart rate, etc.)  
- Target: presence (1) or absence (0) of heart disease  

## Project Structure
- `Heart Disease Prediction with Machine Learning.ipynb` → Notebook with EDA, model training, and evaluation  
- `requirements.txt` → Dependencies to reproduce the results  

## Models Used
- **Logistic Regression** → simple, interpretable baseline  
- **Random Forest** → ensemble model with strong predictive power and feature importance  
- **K-Nearest Neighbors (KNN)** → instance-based model capturing local data patterns  

## Results (Test Set)
| Model               | Accuracy | Precision | Recall | F1  |
|----------------------|----------|-----------|--------|-----|
| Logistic Regression | 0.869    | 0.872     | 0.873  | 0.869 |
| Random Forest       | 0.902    | 0.901     | 0.904  | 0.901 |
| KNN (k=5)           | 0.902    | 0.912     | 0.909  | 0.902 |

**Conclusion:** Random Forest performed best overall, with the highest ROC-AUC (~0.96). KNN achieved similar performance, while Logistic Regression remains a useful transparent baseline.

## Reproducibility
All models were trained with `random_state=42` for consistency.

## Limitations & Next Steps
- Dataset size is small, limiting generalization  
- Basic tuning only; cross-validation and hyperparameter optimization could improve results  
- Future work: try advanced models such as XGBoost or LightGBM  

## References
- Detrano, R. et al. (1989). *Cleveland Heart Disease Dataset*. UCI Machine Learning Repository  
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830  