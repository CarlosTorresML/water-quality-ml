# Water Quality Prediction - Classification & Regression

This project applies machine learning to the Water Potability dataset (Kadiwal, 2021), tackling both classification and regression tasks.  
The goal was to predict water potability (binary classification) and pH level (regression) using Random Forest and Gradient Boosting models, including proper data preprocessing and hyperparameter tuning.

## Objectives

- Preprocess the dataset (handle missing values, standardize numerical precision)
- Train baseline models for classification and regression
- Tune models using GridSearchCV and compare results
- Detect and address overfitting
- Evaluate models using test accuracy and RMSE
- Draw conclusions based on performance and practical limitations

## Preprocessing

The dataset contained missing values in three columns: `ph`, `Sulfate`, and `Trihalomethanes`. The chosen imputation strategy was:
- `ph`: filled with the median (to reduce influence of potential outliers)
- `Sulfate` and `Trihalomethanes`: filled with the mean

All values (except the target) were rounded to 3 decimals for consistency. `ph` was rounded to 1 decimal, as the pH scale has a limited range and high precision is unnecessary.

## Classification (Potability)

The classification task used `RandomForestClassifier`.

### Challenges Encountered

- **Overfitting**: The base Random Forest model reached 100% accuracy on the training set but performed poorly on test data (~62%).
- **Tuning attempts**: Various combinations of `n_estimators`, `max_depth`, and `min_samples_leaf` were tested using GridSearchCV, but they did not improve generalisation.
- **Evaluation frustration**: Even optimized models sometimes performed worse than the base model.
- **Class imbalance**: Uneven distribution of the target variable was suspected to impact results.

### Solution

After several tuning experiments, the most effective step was to enable `class_weight='balanced'` in the classifier.  
This adjustment slightly improved test accuracy to ~63.7%, suggesting better handling of the minority class and a small gain in generalisation.

## Regression (pH)

For regression, `GradientBoostingRegressor` was used.

### Results

- The RMSE of the Gradient Boosting model was **1.36**, which outperformed the previously tested Random Forest regressor (~1.57).
- The pH values range from 0 to 14, so an RMSE of ~1.36 is considered a solid result.

## Key Findings

- Hyperparameter tuning does not guarantee better performance. In this project, the default Random Forest classifier was already close to optimal.
- Boosting techniques proved more effective in regression than classification for this dataset.
- Class imbalance can affect classifier performance and should not be ignored.
- The limitations of the dataset (e.g., weak feature signals, real-world noise) capped the maximum achievable accuracy.

## Dataset

Water quality data downloaded from [Kaggle - Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)


