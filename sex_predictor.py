#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os

try:
    import pandas as pd
    import xgboost as xgb
except:
    print('Requirements not installed, installing!')
    os.system('pip install -r requirements.txt')
    
import pandas as pd
import xgboost as xgb
import os
import argparse


# In[ ]:


base_dir = os.path.dirname(os.path.realpath(__file__))


# In[ ]:


def self_test():

    from sklearn.metrics import accuracy_score

    # Loading model
    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(base_dir, 'data/0.83_acc.model'))

    # Fetching dataset
    filename = os.path.join(base_dir, 'test_data_CANDIDATE.csv')
    df = pd.read_csv(filename, index_col=0)

    mapping = {'trestbps':'blood_pressure', 'chol': 'cholesterol', 'fbs': 'blood_sugar',
                'restecg': 'ekg_results', 'thalach': 'max_heart_rate', 'nar': 'n_arms',
                'sk':'skin_colour', 'hc':'hair_colour', 'trf':'time_in_traffic', 'oldpeak':'stress_test', 'slope':'stress_test_slope'}

    df.rename(columns=mapping, inplace=True)

    df = df[df['cholesterol'] > 50]
    df = df[df['cholesterol'] < 400]

    y_true = df['sex'].str.upper()

    y_true.replace('F', 0, inplace = True)
    y_true.replace('M', 1, inplace = True)

    cols = open(os.path.join(base_dir, 'data/cols.bin'), 'r').read().split(',')
    df = df[cols]


    try:
        threshold = float(open('data/threshold.bin', 'r').read())
    except:
        threshold = 0.5

    df = df.loc[:, df.columns != 'sex']

    # Classification
    y_pred = clf.predict_proba(df)[:,1] > threshold
    print('ACC', accuracy_score(y_true, y_pred))


# In[ ]:


if __name__ == '__main__':

    # Getting inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    filename = vars(parser.parse_args())['input_file']

    # Loading model
    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(base_dir,'data/0.83_acc.model'))

    # Fetching dataset
    filename = os.path.join(base_dir, filename)
    df = pd.read_csv(filename, index_col=0)

    cols = open('data/cols.bin', 'r').read().split(',')
    cols.remove('sex')

    mapping = {'trestbps':'blood_pressure', 'chol': 'cholesterol', 'fbs': 'blood_sugar',
            'restecg': 'ekg_results', 'thalach': 'max_heart_rate', 'nar': 'n_arms',
            'sk':'skin_colour', 'hc':'hair_colour', 'trf':'time_in_traffic', 'oldpeak':'stress_test', 'slope':'stress_test_slope'}

    df.rename(columns=mapping, inplace=True)

    df = df[cols]

    try:
        threshold = float(open('data/threshold.bin', 'r').read())
    except:
        threshold = 0.5

    # Classification
    y_pred = clf.predict_proba(df)[:,1] > threshold

    # Dataframe creation
    df_out = (pd.DataFrame(y_pred, columns=['sex'])).astype('uint8')

    # Returning to labels
    df_out.replace(0, 'F', inplace = True)
    df_out.replace(1, 'M', inplace = True)

    # Outputing the dataframe
    file_output = os.path.join(base_dir, 'newsample_PREDICTIONS_{}.csv'.format('Felipe Bueno Hernandez'))
    df_out.to_csv(file_output, index=False)

    print('File {} created sucessfully!'.format(file_output))

