# ML-Project-Compilence-Radar-321551

# Compliance Radar – Machine Learning Project
Kebab, Samba, Baguette.
Deniz Mehmet Taylan 321551,
Gustavo Depieri Fİoravanti 321841,
Diego Diaz 314781. 

## 1. Introduction

The Compliance Radar project focuses on analyzing organizational compliance behavior using a rich dataset of operational, risk-related, and audit-related variables.  
The goal is to build an analytical foundation that will allow the team to:

- understand compliance patterns,
- prepare the data for machine learning,
- explore structural relationships between variables,
- and later build models that estimate or classify compliance risk.

This repository contains the initial stages of the project, including data loading, cleaning, preprocessing, and exploratory analysis.

---

## 2. Methods (Current Progress)

### 2.1 Data Source

The data is stored in a SQLite database (`org_compliance_data.db`), which contains four tables:

- **departments** — the primary dataset used for analysis.
- **risk_summary_by_division** — supplementary aggregated metrics.
- **high_risk_departments** — external list of flagged departments.
- **data_dictionary** — definitions and descriptions of all variables.

The **departments** table (709 rows) is the main focus of the analysis.

---

### 2.2 Data Loading

The SQLite database was loaded using Python’s `sqlite3` module.  
All tables were inspected, and the `departments` table was selected as the main analytical dataset.

Additional information about feature meanings was taken from the `data_dictionary` table.

---

### 2.3 Data Cleaning

#### **Handling Missing Values**

The dataset contained substantial missingness (up to ~40% in several variables).  
Since dropping rows would result in severe data loss, the following imputation strategy was applied:

- **Median imputation** for numerical columns  
- **Most frequent category (mode)** imputation for categorical columns  
- Identifier fields (`dept_id`, `dept_name`) were left untouched

After imputation, the dataset contained **no missing values**.

---

### 2.4 Outlier Analysis

Outlier detection was performed using the **IQR (Interquartile Range)** method.  
Many variables showed large numbers of outliers due to the nature of compliance and operational behavior (risk exposure, reporting delays, violations, audit results, etc.).

These values represent real departmental behavior, not errors.  
**Therefore, no outlier removal or clipping was applied.**

---

### 2.5 Encoding Categorical Variables

Categorical features such as:

- department type  
- division  
- location type  
- team size  
- reporting structure  
- functional classifications  

were encoded using **one-hot encoding**.  
Identifier fields (`dept_id`, `dept_name`) were excluded from encoding.

The resulting dataset is entirely numeric (except for the two ID fields).

---

### 2.6 Scaling

All numerical variables were standardized using a manual z-score transformation:

\[
z = \frac{x - \text{mean}}{\text{std}}
\]

Scaling was applied to:

- operational metrics  
- risk indicators  
- audit scores  
- reporting metrics  
- binary flags  
- one-hot encoded features  

`dept_id` and `dept_name` were excluded.

This prepares the dataset for models that are sensitive to feature magnitude, such as clustering or logistic regression.

---

### 2.7 Implementation Environment

All experiments in this project were carried out in Python.
The main libraries used are:

- **pandas** and **numpy** for data processing and manipulation  
- **sqlite3** for loading the database  
- **scikit-learn** for model training, preprocessing, and evaluation  
- **matplotlib** and **seaborn** for visualization  

To ensure reproducibility, all package versions used in the project are listed
in the `requirements.txt` file included in the repository.  
Installing these packages (e.g., via `pip install -r requirements.txt`)
will recreate the same environment used for our experiments.

---

## 3. Exploratory Data Analysis (Current Progress)

Initial EDA has been performed, including:

### **3.1 Correlation Analysis**
A correlation matrix and heatmap were generated to evaluate linear relationships among numerical variables.

### **Distribution Analysis**
Histograms were created for all numerical columns to inspect:

- skewness  
- spread  
- multimodality  
- extreme values  

### **Boxplots**
Boxplots were generated for:

- Risk indicators  
- Audit performance metrics  
- Reporting delay and gap metrics  

This helps visualize variability and highlight departments with extreme behavior.

---

## 4. The 3 ML Problems

Based on the structure of the dataset and the results of the exploratory data analysis, the Compliance Radar project involves three complementary ML tasks:

1. Unsupervised Clustering

We use clustering (e.g., K-Means) to identify natural groupings of departments based on operational, audit, and risk-related features.
This helps reveal behavioral patterns and high-risk clusters without predefined labels.

2. Risk Score Prediction (Regression)

The **overall_risk_score** serves as a continuous target variable.
We apply regression models to estimate risk levels from operational metrics, allowing us to quantify the drivers of compliance risk.

3. Feature Importance & Risk Drivers

Tree-based models and correlation analysis are used to determine which factors most influence compliance behavior.
This supports interpretability and contributes directly to the final recommendations.

---

## 5. Experimental Design

In this section, we use the cleaned and scaled `scaled` dataframe to train and evaluate machine learning models that classify high-risk departments.

---

### 5.1 Creating the Target Variable

We start from the cleaned and scaled dataframe `scaled`.
Using the `high_risk_departments` table in the database, we create a binary label:

- 1 = high-risk department  
- 0 = not high-risk

---
### 5.2 Feature Selection and Train/Validation/Test Split

In this step, we prepare the feature matrix **X** and the target vector **y** from the cleaned and scaled dataset (`model_df`).

We remove the following columns from the feature matrix:
- `dept_id` and `dept_name`: identifier fields that carry no predictive value,
- `overall_risk_score` and `compliance_score_final`: high-level outcome variables
  that would cause data leakage if included,
- the target column `is_high_risk` (kept only in `y`).

We then split the dataset into:
- **60% training set** — used for baseline model evaluation and GridSearchCV fitting,
- **20% validation set** — used to evaluate the default models (baseline step),
- **20% test set** — kept fully untouched until the end for final evaluation.

The split is done in two stages:
1. First: **train+validation vs test** split (80% / 20%)  
2. Second: split the 80% portion into **train vs validation** (75% / 25%)  
   → resulting in **60% train**, **20% validation**, **20% test**.
---

### 5.3 Baseline Models with Default Hyperparameters

Before tuning any hyperparameters, we first train each model with its default
settings. The goal is to see how well the models perform “out of the box”
on the validation set.

We use three models:
- Logistic Regression
- Random Forest
- Histogram Gradient Boosting

Each model is fitted on the training set (`X_train`, `y_train`) and evaluated
on the validation set (`X_val`, `y_val`) using the F1-score.
These baseline F1 values will serve as a reference point when we later
apply GridSearchCV.

---

### 5.4 Models and Hyperparameters

We train three different models on the scaled features:

1. Logistic Regression  
2. Random Forest  
3. HistGradientBoosting  

For each model we define a small hyperparameter grid.
In the next step we will use GridSearchCV with 3-fold cross-validation
to search over these grids using F1-score as the main metric.

---

### 5.4.1 Explanation of Tuned Hyperparameters

Before running GridSearchCV, we define a set of hyperparameters for each model.
Below is a description of every hyperparameter we tuned and why it matters.
This section is required by the assignment to demonstrate understanding of
how each parameter affects model behavior.

#### Logistic Regression
- **C**  
  Controls the strength of regularization.  
  - Smaller values → stronger regularization (simpler model, less overfitting)  
  - Larger values → weaker regularization (more flexibility)

- **class_weight**  
  Adjusts the weight given to each class.  
  `"balanced"` increases focus on the minority class (high-risk departments).

#### Random Forest
- **n_estimators**  
  Number of trees in the forest. More trees improve stability but increase computation.

- **max_depth**  
  Maximum depth of each decision tree. Controls model complexity.  
  - Deeper trees → may overfit  
  - Shallower trees → more generalizable

- **min_samples_split**  
  Minimum number of samples required to split a node.  
  Higher values reduce overfitting by preventing deep splits.

- **min_samples_leaf**  
  Minimum number of samples at a leaf node.  
  Larger values smooth the model and increase robustness.

- **class_weight**  
  Helps handle class imbalance by giving more weight to the minority class.

- **max_features** *(in some grids)*  
  Controls how many features each tree considers when splitting.  
  `"sqrt"` helps decorrelate trees and often improves performance.

#### HistGradientBoosting
- **learning_rate**  
  Determines how quickly the boosting algorithm learns.  
  Smaller values → safer, slower learning  
  Larger values → faster learning but risk of overfitting.

- **max_depth**  
  Maximum depth of individual trees.  
  Controls how complex each boosting stage can become.

- **max_leaf_nodes**  
  Sets an upper limit on the number of leaf nodes in each tree.  
  Acts as an alternative to max_depth for controlling complexity.

- **min_samples_leaf**  
  Minimum samples per leaf.  
  Larger values → more smoothing, less variance.

These hyperparameters are chosen because they directly control model
complexity, generalization, and the balance between overfitting and underfitting.
They also help address class imbalance, which is essential for correctly
identifying high-risk departments.

---

### 5.5 Running GridSearchCV

We run GridSearchCV for each model on the combined training+validation set 
(`X_trainval`, `y_trainval`) and keep the best estimator according to the
mean cross-validated F1-score.

---

### 5.6 Model Evaluation

We evaluate each tuned model on the held-out test set using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- Confusion matrix and ROC curve

---

### 5.7 Test Results and Model Comparison

We now evaluate all three models on the test set and summarize the metrics in a comparison table.

### *Table 1 – Model Performance Comparison (Test Set)*  
(Values shown here correspond to the output generated by the notebook as a pandas DataFrame.)

| Model                     | F1-score | Accuracy | ROC-AUC |
|--------------------------|----------|----------|---------|
| HistGradientBoosting     | *0.83* | *0.89* | *0.93* |
| Random Forest            | 0.81     | 0.87     | 0.90    |
| Logistic Regression      | 0.73     | 0.84     | 0.88    |

This table is generated directly from the code inside the notebook using the model comparison DataFrame.

---

### 5.8 Feature Importance (Random Forest)

Finally, we inspect which features are most important in the Random Forest model.
Since the input data is already scaled and fully numeric, we can directly use
`feature_importances_` together with the original column names.

---

### 6 Results and Discussion

The model analysis in our notebook demonstrated how each supervised learning approach's performance in classifying high-risk departments is illustrated.

---

### 6.1 Overall Model Performance

Three models were trained and tuned by GridSearchCV:

- **Logistic Regression**
- **Random Forest**

- **HistGradientBoosting**

Every model did a really good job. The most interesting thing that we discovered was that **HistGradientBoosting** got the best test-set results overall with 
- Accuracy ≈ 0.89.

- The approximate **F1-score**  0.83.

- **ROC-AUC** ≈ 0.93

This shows us that it has the most ability to choose between a high-risk and a low-risk department.
Analysing **logistic regression**, it was good too: 
- F1 ≈ **0.73** and ROC-AUC reaching **0.88**

**Random Forest** got a good test-set result with ROC-AUC almost **0.90** and F1 reaching **0.81**. 


<img width="1920" height="1440" alt="logistic_regression_roc_curve" src="https://github.com/user-attachments/assets/3da432a5-8d18-4878-b3b1-835364893dd3" />

<img width="1920" height="1440" alt="random_forest_roc_curve" src="https://github.com/user-attachments/assets/6c432005-ee32-4551-aedd-8f63a61e1123" />

<img width="1920" height="1440" alt="histgradientboosting_roc_curve" src="https://github.com/user-attachments/assets/31464a58-5cae-44f1-af25-ab75d4f19579" />


---

### 6.2 Class Balance and Error Analysis

Target variable `is_high_risk` is moderately imbalanced, with roughly 70% class 0 and 30% class 1.

These models represent:

- **High recall for high-risk cases**: The model identifies most of the departments actually falling under the high-risk category.

- Slightly lower precision, meaning a few departments were incorrectly marked as high-risk.

The latter is an acceptable trade-off in compliance contexts:

- **False negatives or missed high-risk cases** are more dangerous than

- **Extra Departments taken to reviewals (false positives).
---

### 6.3 ROC Curves and Confusion Matrices

The ROC curves below reflect strong separability between classes, with particular emphasis on HistGradientBoosting.

Confusion matrices confirm that:

The misclassification rates are small.
The majority of the high-risk cases are correctly identified.
Non-high-risk cases are rarely misclassified.

---

### 6.4 Feature Importance and Risk Drivers

Random Forest feature importance analysis generally identifies the following as the strongest predictors of risk:

Indicators of historical violations

Audrey's audit results: `audit_score_q1`, `audit_score_q2`, `compliance_score_final

- Risk measures of exposure 

- Reporting gaps and delays

- Remediation and oversight-each of these characteristics

These agree with intuitive domain expectations and, hence, reinforce the model interpretability.

---
### 7 Ethical and Organizational Considerations

While these models provide strong predictive performance, the actual deployment of a risk classification system within an organization demands due attention to ethics, governance, and transparency.

-

### 7.1 Fairness and Possible Bias

There may be embedded biases in historical compliance data; for instance, there may have been more frequent audits in some divisions compared to others.

If not caught, a model could inadvertently act out these same patterns.
Countermeasures include:

Performance tracking across subgroups: division, department type and location

- Tracking the rates of fake positives and fake negatives by group

Now, adjust thresholds or add fairness constraints where it is needed.

---

### 7.2 Accountability, Transparency, and Explainability

Compliance decisions have regulatory implications.

Therefore, the model should be **explainable**, not a "black box".

Best practices:

- Record every single preprocessing step (imputation, encoding, scaling)
- Use feature importance and modern interpretability tools, such as SHAP values.

- Clearly explanations of predictions to the users

---

### 7.3 Human-in-the-Loop Design

Its objective is to support, not substitute for, compliance experts.

- Predictions should only trigger **human review** and not automatic penalties

- Expert must be allowed to override the model outputs.

- Investigators' feedback should be recorded and included in retraining cycles

---

### 7.4 Data Governance

Because the underlying data contains sensitive operational and audit information:

- Access and utilization should be restricted and policed.

- Retention policies should be established

- Model's predictions are to be logged for accountability

All of these considerations ensures that the Compliance Radar remains a responsible and trustful tool.

---

### 8. Conclusion and Next Steps

---

### 8.1 Operational Use Cases

The existing prototype already has a few high-value use cases:

- **Risk-based audit prioritization

- Identify the departments that may require intervention.

- Ongoing monitoring


Recompute risk levels quarterly or monthly to track improvement or deterioration.

- **Management insights
Use risk-driver analysis to identify structural weaknesses across the organization.

---

### 8.2 Threshold Strategy

The decision threshold can be tuned depending on the business needs to emphasize:

- Higher recall when missing high-risk cases is not acceptable

- **Higher accuracy** in cases of limited investigative resources

This threshold process should be documented as part of internal policy.

---

### 8.3 Model Monitoring and Retraining

When deployed, the models need to be monitored for:
- **Performance drift** (reduction in accuracy, recall, F1)

- **Concept drift** (feature importance shifts)

- **Data drift** changes in the distribution over time.

Regular cycles of retraining should be scheduled based on the indicators showed above.

---

### 8.4 Future Enhancements

Possible recommendations for improvement might be:

- Adding **time-series features** (quarterly trends in audits, violations, etc.)
- Implementing **SHAP-based explainability dashboards**
- Could be tested alternative models such as the XGBoost and LightGBM
- Additional contextual datasets. This can include HR, vendor risk and even financial stress indicators
- Performing **unsupervised clustering** to detect hidden behavioral patterns

When implemented, these steps will help the Compliance Radar become a robust, scalable, interpretable decision-support system.
