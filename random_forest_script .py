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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import pickle

RANDOM_STATE = 42
MODEL_OUTPUT_PATH = "model.pkl"   # the file app.py expects to find


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
    """Adds domain-specific ratio & log features for loan risk.

    NOTE: this class must stay IDENTICAL to the one defined in app.py —
    it gets pickled as part of `model.pkl`, and app.py needs the same
    class definition available to unpickle it.
    """

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

## Feature-engineer a throwaway copy of X_train just to discover the final
## column layout (which columns are numeric vs. nominal AFTER engineering).
## This is only used to build the ColumnTransformer's column lists — the
## actual pipeline fitted below re-runs FeatureEngineer on the real data.
_X_train_fe_preview = FeatureEngineer().transform(X_train)
num_cols = _X_train_fe_preview.select_dtypes(include=['int64', 'float64']).columns.tolist()
categ_cols = [col for col in nominal_cat if col in _X_train_fe_preview.columns]


def build_preprocessor():
    """Returns a fresh (unfitted) ColumnTransformer. Built fresh per model
    so every Pipeline below gets its own independently-fitted copy."""
    num_pipeline = Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())
                        ])

    categ_pipeline = Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='most_frequent')),
                            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                        ])

    return ColumnTransformer(transformers=[
                                    ('numerical', num_pipeline, num_cols),
                                    ('categorical', categ_pipeline, categ_cols)
                                ], remainder='passthrough')


## --------------------- Imbalancing ---------------------------- ##

# The dataset is moderately imbalanced (~75.4% Rejected / ~24.6% Accepted, ~3:1).
# SMOTE was tested during development and dropped from the pipeline: the
# synthetic minority samples it generated hurt generalisation on the real
# test set. RandomForestClassifier supports `class_weight` natively, so it's
# passed straight to the model.

class_weights = {0: 0.66, 1: 2.03}


## --------------------- Modeling ---------------------------- ##

def train_model(X_train, y_train, plot_name, n_estimators, max_depth, min_samples_leaf, class_weight=None):

    mlflow.set_experiment('loan-default-detection')
    with mlflow.start_run() as run:
        mlflow.set_tag('clf', 'random_forest')

        # Try Random Forest
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf, class_weight=class_weight,
                                      random_state=RANDOM_STATE, n_jobs=-1)

        # --------------------------------------------------------------
        # Build ONE end-to-end Pipeline: raw DataFrame in -> prediction out.
        # This is what gets saved as model.pkl and loaded by app.py, so the
        # UI can call full_pipeline.predict(raw_input_df) directly without
        # doing any manual feature engineering or preprocessing itself.
        # --------------------------------------------------------------
        full_pipeline = Pipeline(steps=[
            ('feature_engineering', FeatureEngineer()),
            ('preprocessor', build_preprocessor()),
            ('classifier', clf),
        ])

        full_pipeline.fit(X_train, y_train)

        y_pred_test = full_pipeline.predict(X_test)
        y_proba_test = full_pipeline.predict_proba(X_test)[:, 1]

        ## metrics
        f1_test = f1_score(y_test, y_pred_test)
        acc_test = accuracy_score(y_test, y_pred_test)
        roc_auc_test = roc_auc_score(y_test, y_proba_test)

        # Log params, metrics, and the full pipeline as the MLflow model
        mlflow.log_params({'n_estimators': n_estimators, 'max_depth': max_depth,
                            'min_samples_leaf': min_samples_leaf,
                            'class_weight': class_weight if class_weight is not None else 'none'})
        mlflow.log_metrics({'accuracy': acc_test, 'f1_score': f1_test, 'roc_auc': roc_auc_test})

        # Save the local copy FIRST so app.py always gets a usable model.pkl,
        # even if the MLflow logging step below has trouble.
        with open(MODEL_OUTPUT_PATH, "wb") as f:
            pickle.dump(full_pipeline, f)
        mlflow.log_artifact(MODEL_OUTPUT_PATH)

        # Newer MLflow versions save sklearn models with `skops` instead of
        # raw pickle, and skops refuses to (de)serialize custom classes it
        # doesn't recognise as "trusted" — like our FeatureEngineer. Since we
        # wrote that class ourselves and trust it, we explicitly allow it.
        mlflow.sklearn.log_model(
            full_pipeline,
            name=f'{clf.__class__.__name__}_{plot_name}',
            skops_trusted_types=["__main__.FeatureEngineer", "numpy.dtype"],
        )

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



def main(n_estimators: int, max_depth: int, min_samples_leaf: int):

    # ---------------- Calling the above function -------------------- ##

    ## 1. without considering the imbalance
    train_model(X_train=X_train, y_train=y_train, plot_name='without-imbalance',
                n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                class_weight=None)

    ## 2. with considering the imbalance using class_weight
    ## This one runs LAST, so the model.pkl left on disk after the script
    ## finishes is this (better) class-weighted version — the one app.py
    ## will actually load.
    train_model(X_train=X_train, y_train=y_train, plot_name='with-class-weights',
                n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                class_weight=class_weights)

    print(f"\nDone. '{MODEL_OUTPUT_PATH}' now holds the with-class-weights Random Forest "
          f"pipeline — copy it next to app.py to use it in the UI.")




if __name__ == '__main__':
    ## Take input from user via CLI using argparser library
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', '-n', type=int, default=150)
    parser.add_argument('--max_depth', '-d', type=int, default=18)
    parser.add_argument('--min_samples_leaf', '-m', type=int, default=10)
    args = parser.parse_args()

    ## Call the main function
    main(n_estimators=args.n_estimators, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf)
