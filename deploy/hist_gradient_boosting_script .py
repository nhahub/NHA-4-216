## main
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, argparse
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import pickle

RANDOM_STATE = 42


## --------------------- Data Preparation ---------------------------- ##

## Read the Dataset
TRAIN_PATH = os.path.join(os.getcwd(), 'Loan_Default.csv')
df = pd.read_csv(TRAIN_PATH)

## Drop columns with no predictive value (ID) or that can't generalise (year)
df.drop(columns=['ID', 'year'], axis=1, inplace=True)


## Clean the 'age' column: convert ranges like '25-34' into numeric midpoints,
## and open-ended buckets ('<25', '>74') into representative values
def edit_age(age):
    if pd.isna(age):
        return np.nan
    if '-' in age:
        lo, hi = age.split('-')
        return (int(lo) + int(hi)) / 2
    elif '<' in age:
        return 20
    elif '>' in age:
        return 75
    return age

df['age'] = df['age'].apply(edit_age)


## Drop only the two columns whose missingness (~25%+) required extensive
## imputation and hurt generalisation. `rate_of_interest` is kept: it's used
## by the feature-engineering step below.
df.drop(columns=['Upfront_charges', 'Interest_rate_spread'], axis=1, inplace=True)


## Identify categorical columns, split into binary vs nominal
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
binary_cat = [col for col in cat_cols if df[col].nunique() == 2]
nominal_cat = [col for col in cat_cols if col not in binary_cat]

## Label-encode binary categorical columns (2 categories -> 0/1)
le = LabelEncoder()
for col in binary_cat:
    df[col] = le.fit_transform(df[col].astype(str))


## To features and target
X = df.drop(columns=['Status'], axis=1)
y = df['Status']

## Split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE, stratify=y)


## --------------------- Feature Engineering ---------------------------- ##

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds domain-specific ratio & log features for loan risk."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        eps = 1e-6

        X['Loan_to_Income'] = X['loan_amount'] / (X['income'] + 1)
        X['Loan_to_Property'] = X['loan_amount'] / (X['property_value'] + 1)
        X['Debt_to_Income'] = X['dtir1'] / (X['income'] + 1)
        X['Score_LTV_ratio'] = X['Credit_Score'] / X['LTV'].clip(lower=eps)
        X['Interest_x_LTV'] = X['rate_of_interest'] * X['LTV'] / 100

        log_cols = ['loan_amount', 'property_value', 'income', 'rate_of_interest']
        for col in log_cols:
            X[col] = np.log1p(X[col].clip(lower=0))

        return X


## --------------------- Data Processing ---------------------------- ##

## Apply feature engineering first so column types are known for the ColumnTransformer
fe = FeatureEngineer()
X_train_fe = fe.transform(X_train)
X_test_fe = fe.transform(X_test)

## Slice the lists (detected after feature engineering)
num_cols = X_train_fe.select_dtypes(include=['int64', 'float64']).columns.tolist()
categ_cols = [col for col in nominal_cat if col in X_train_fe.columns]


## For Numerical
num_pipeline = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ])

## For Categorical
categ_pipeline = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                    ])

## combine all
all_pipeline = ColumnTransformer(transformers=[
                                    ('numerical', num_pipeline, num_cols),
                                    ('categorical', categ_pipeline, categ_cols)
                                ], remainder='passthrough')

## apply
X_train_final = all_pipeline.fit_transform(X_train_fe)
X_test_final = all_pipeline.transform(X_test_fe)


## --------------------- Imbalancing ---------------------------- ##

# The dataset is moderately imbalanced (~75.4% Rejected / ~24.6% Accepted, ~3:1).
# SMOTE was tested during development and dropped from the pipeline: the
# synthetic minority samples it generated hurt generalisation on the real
# test set. `class_weight=class_weights` gave more stable, honest results
# without inventing data. HistGradientBoostingClassifier has no `class_weight`
# parameter, so the same weights are applied as per-sample weights at fit time.

class_weights = {0: 0.66, 1: 2.03}
sample_weight_balanced = compute_sample_weight(class_weight=class_weights, y=y_train)


## --------------------- Modeling ---------------------------- ##

def train_model(X_train, y_train, plot_name, max_iter, max_depth, learning_rate, sample_weight=None):

    mlflow.set_experiment('loan-default-detection')
    with mlflow.start_run() as run:
        mlflow.set_tag('clf', 'hist_gradient_boosting')
        pickle.dump(all_pipeline, open("pipeline.pkl", "wb"))
        mlflow.log_artifact("pipeline.pkl")

        # Try Hist Gradient Boosting
        clf = HistGradientBoostingClassifier(max_iter=max_iter, max_depth=max_depth,
                                              learning_rate=learning_rate, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        y_pred_test = clf.predict(X_test_final)
        y_proba_test = clf.predict_proba(X_test_final)[:, 1]

        ## metrics
        f1_test = f1_score(y_test, y_pred_test)
        acc_test = accuracy_score(y_test, y_pred_test)
        roc_auc_test = roc_auc_score(y_test, y_proba_test)

        # Log params, metrics, and model
        mlflow.log_params({'max_iter': max_iter, 'max_depth': max_depth,
                            'learning_rate': learning_rate,
                            'sample_weight': 'class_weights' if sample_weight is not None else 'none'})
        mlflow.log_metrics({'accuracy': acc_test, 'f1_score': f1_test, 'roc_auc': roc_auc_test})
        mlflow.sklearn.log_model(clf, name=f'{clf.__class__.__name__}_{plot_name}')

        ## Plot the confusion matrix and save it to mlflow
        plt.figure(figsize=(10, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, cbar=False, fmt='.2f', cmap='Blues')
        plt.title(f'{plot_name}')
        plt.xticks(ticks=np.arange(2) + 0.5, labels=[False, True])
        plt.yticks(ticks=np.arange(2) + 0.5, labels=[False, True])

        # Save the plot to MLflow
        conf_matrix_fig = plt.gcf()
        mlflow.log_figure(figure=conf_matrix_fig, artifact_file=f'{plot_name}_conf_matrix.png')
        plt.close()

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_proba_test)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve and save it to mlflow
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")

        # Save the plot to MLflow
        roc_fig = plt.gcf()
        mlflow.log_figure(figure=roc_fig, artifact_file=f'{plot_name}_roc_curve.png')
        plt.close()



def main(max_iter: int, max_depth: int, learning_rate: float):

    # ---------------- Calling the above function -------------------- ##

    ## 1. without considering the imbalance
    train_model(X_train=X_train_final, y_train=y_train, plot_name='without-imbalance',
                max_iter=max_iter, max_depth=max_depth, learning_rate=learning_rate,
                sample_weight=None)

    ## 2. with considering the imbalance using class weights (as sample_weight)
    train_model(X_train=X_train_final, y_train=y_train, plot_name='with-class-weights',
                max_iter=max_iter, max_depth=max_depth, learning_rate=learning_rate,
                sample_weight=sample_weight_balanced)




if __name__ == '__main__':
    ## Take input from user via CLI using argparser library
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iter', '-n', type=int, default=150)
    parser.add_argument('--max_depth', '-d', type=int, default=10)
    parser.add_argument('--learning_rate', '-l', type=float, default=0.1)
    args = parser.parse_args()

    ## Call the main function
    main(max_iter=args.max_iter, max_depth=args.max_depth, learning_rate=args.learning_rate)
