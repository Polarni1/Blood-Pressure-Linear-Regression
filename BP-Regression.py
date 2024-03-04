import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from statsmodels.graphics.regressionplots import plot_regress_exog

def read_and_preprocess_data(file_path):
    """
    Reads CSV data into a DataFrame and preprocesses it by filling missing values,
    filtering based on medical criteria, and converting data types.
    """
    df = pd.read_csv(file_path)
    df['DIAGNOSIS'] = df['DIAGNOSIS'].fillna('no_diagnosis')
    criteria = {
        'PATIENT_AGE': (0, 120),
        'BLOOD_PRESSURE': (0, 200),
        'CHOLESTEROL_LEVEL': (100, 500),
        'TEST_RESULT_TIME': (1, 240)
    }
    for column, (min_val, max_val) in criteria.items():
        df = df[df[column].between(min_val, max_val)]
    df = df.dropna(subset=['TEST_RESULT_TIME'])
    df[['GLUCOSE_LEVEL', 'TEST_RESULT_TIME', 'BLOOD_PRESSURE']] = df[['GLUCOSE_LEVEL', 'TEST_RESULT_TIME', 'BLOOD_PRESSURE']].apply(pd.to_numeric, errors='coerce')
    df['TEST_DATE'] = pd.to_datetime(df['TEST_DATE'])
    df['TEST_MONTH'] = df['TEST_DATE'].dt.to_period('M').astype(str)
    return df

def enrich_data(df):
    """
    Enriches DataFrame by adding computed columns, dummy variables, and filtering based on new medical criteria.
    """
    df['POSITIVE_RATE'] = df['POSITIVE_RESULTS'] / df['TOTAL_TESTS']
    df = df[df['POSITIVE_RATE'].between(0, 1)]
    df['BLOOD_PRESSURE_SQ'] = df['BLOOD_PRESSURE'] ** 2
    interaction_columns = ['GLUCOSE_LEVEL', 'PATIENT_AGE', 'CHOLESTEROL_LEVEL']
    for col1, col2 in combinations(interaction_columns, 2):
        df[f'{col1}*{col2}'] = df[col1] * df[col2]
    diagnosis_dummies = pd.get_dummies(df['DIAGNOSIS'], drop_first=True)
    df = pd.concat([df, diagnosis_dummies], axis=1)
    df_filtered = df[df['DIAGNOSTIC_ACCURACY'] >= 0.9]
    return df_filtered

def prepare_regression_data(df_filtered):
    """
    Prepares data for regression by selecting required columns, adding constants, and handling missing values.
    """
    required_columns = ['BLOOD_PRESSURE', 'BLOOD_PRESSURE_SQ'] + [col for col in df_filtered.columns if col.startswith('test_month_')]
    X = df_filtered[required_columns]
    X = sm.add_constant(X)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    Y = df_filtered.loc[X.index, 'TEST_RESULT_TIME']
    return X, Y

def perform_regression(X, Y):
    """
    Performs a multiple linear regression and returns the model summary.
    """
    X_np, Y_np = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    model = sm.OLS(Y_np, X_np).fit()
    print(model.summary(xname=X.columns.tolist()))
    return model

def main():
    file_path = 'LAB_TEST_RESULTS_RAW.csv'
    df = read_and_preprocess_data(file_path)
    df_filtered = enrich_data(df)
    X, Y = prepare_regression_data(df_filtered)
    model = perform_regression(X, Y)
    # Additional analysis and plotting can be done here.

if __name__ == "__main__":
    main()
