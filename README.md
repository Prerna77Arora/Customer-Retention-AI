# Customer-Retention-AI
Machine Learning Model for Churn Prediction

Overview

This project aims to predict customer churn using a machine learning pipeline. The dataset includes various customer attributes, and the target variable is churn (1 for churned customers, 0 for non-churned). The solution involves data preprocessing, exploratory data analysis (EDA), feature engineering, and training a Random Forest classifier with hyperparameter tuning.

Steps in the Workflow

1. Feature Engineering

Handling Missing Values:

Missing values were filled using the median of each column.

This approach ensures robustness against outliers.

Dropping Irrelevant Columns:

Columns such as id were removed as they do not contribute to predictive power.

Handling Class Imbalance:

Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the target variable distribution.

2. Exploratory Data Analysis (EDA)

Understanding Data:

Loaded the dataset and displayed the first few rows to understand its structure.

Checked for missing values and class distribution.

Correlation Analysis:

Identified correlations between features using a heatmap to spot highly correlated features.

Feature Importance:

Used the feature importance scores from the Random Forest model to identify key predictors of churn.

Visualization Tools:

Utilized seaborn and matplotlib to create:

Histograms and boxplots for feature distribution.

Bar plots for categorical feature analysis.

Heatmaps for correlation analysis.

3. Machine Learning Model

Model Selection:

Chose a Random Forest Classifier for its interpretability, robustness, and performance on tabular data.

Data Splitting:

Split the dataset into training (75%) and testing (25%) sets.

Hyperparameter Tuning:

Used GridSearchCV to optimize hyperparameters such as:

n_estimators (number of trees).

max_depth (maximum depth of trees).

min_samples_split (minimum samples required to split an internal node).

min_samples_leaf (minimum samples required at a leaf node).

max_features (number of features to consider when looking for the best split).

Model Evaluation Metrics:

Confusion Matrix: True positives, false positives, true negatives, and false negatives.

Accuracy: Overall performance of the model.

Precision: Proportion of correctly predicted positive observations.

Recall: Ability of the model to detect all positive observations.

AUC-ROC: Evaluates model performance in distinguishing classes.

Feature Importance Visualization:

Bar plot showing the importance of each feature in predicting churn.

4. Output

Predictions and churn probabilities were saved to out_of_sample_data_with_predictions.csv.

Future Enhancements

Implement feature selection to reduce dimensionality and improve efficiency.

Experiment with other algorithms like Gradient Boosting, XGBoost, or LightGBM for improved performance.

Add cross-validation for robust model evaluation.

Automate EDA using tools like pandas-profiling or sweetviz.

Acknowledgments

Special thanks to the developers of Python libraries such as pandas, numpy, seaborn, sklearn, and imblearn, which made this project possible.

