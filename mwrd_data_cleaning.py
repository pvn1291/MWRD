#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MWRD Data Cleaning

"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as matplot
import seaborn as sns

sns.set(style = 'darkgrid', palette = 'muted')

DIRPATH = '/Users/pvn1291/WorkSpace/MWRD/Test'
FILEPATH = '/Users/pvn1291/WorkSpace/MWRD/Test/mwrd_test.csv'


# --------------------------------------- Utility Functions ------------------------------------------
def convert_datatypes(dataframe, columns, dtype):
    """
        Function converts datatypes of columns in given dataframe by new datatype.
        Returns the dataframes with input columns (with new datatypes).
        
        Parameters,
            dataframe (pandas dataframe) = reference of the dataframe
            columns (list) = input columns to be converted
            dtype (str) = required datatype, valid values for ['category', 'numeric', 'datetime']                
    """
    import pandas as pd
    import numpy as np
    valid_dtypes = ['category', 'numeric', 'datetime']
    
    assert (dataframe.empty == False), 'Dataframe is empty.' 
    assert (len(columns) > 0), 'Column list is empty.'
    assert (dtype in valid_dtypes), 'Datatype specified is not valid.'
    
    if dtype == 'category':
        dataframe[columns] = dataframe[columns].astype('category')
        return dataframe[columns]        
            
    for col in columns:
        if dtype == 'datetime':
            dataframe[col] = pd.to_datetime(dataframe[col])
        if dtype == 'numeric':
            dataframe[col] = pd.to_numeric(dataframe[col])
            
    return dataframe[columns]            
        
def count_NA_values(dataframe, plot = True):
    """
        Function counts the NA values across the dataframe and plots the heatmap.
        Returns new dataframe with count of NA values for eah column.
        
        Parameters,
            dataframe (pandas dataframe) = reference of the dataframe
            plot (boolean) = True if plot is required, false otherwise.
                             Default is true.
    """
    import pandas as pd
    import numpy as np
    
    import matplotlib.pyplot as matplot
    import seaborn as sns

    assert (dataframe.empty == False), 'Dataframe is empty.'
    
    na_df = pd.DataFrame(data = {'NA Count': dataframe.isna().apply(sum),
            '% of total records': (round(dataframe.isna().apply(sum) / dataframe.shape[0] * 100, 2))},
            index = dataframe.columns)
    
    if plot == True:
        print('\n\n'
              + '------------------------------ Heatmap of NA values ------------------------------'
              + '\n\t\t\t Red ticks indicate the NA values')
        matplot.figure(figsize = (15, 8))
        sns.heatmap(data = dataframe.isna(), yticklabels = False, 
                cbar = False, cmap = 'coolwarm')
   
    return na_df

def is_value_nan(value):
    """
    Function checks if value is NAN.
    Takes value parameter and return true if value is NAN.
    
    Logic:
        NAN == NAN is not a valid comparison and returns False
        Number == Number is a valid comparison and returns True
    """
    return value == value

def is_value_nan(value):
    """
    Function checks if value is NAN.
    Takes value parameter and return false if value is NAN.
    
    Logic:
        NAN == NAN is not a valid comparison and returns False
        Number == Number is a valid comparison and returns True
    """
    return value == value

def bucket(x):
    minutes = (x[0] * 60) + x[1] + (x[2] / 60)
    return round(minutes, 2)
    
def convert_int(value):
    if is_value_nan(value) == True:
        x = value.split(":")
        tmp = []
        for ele in x:
            tmp.append(np.int64(ele))
    
        return bucket(tmp)

# ----------------------------------- Data Cleaning -------------------------------------------

# ------------------- Step 1: Read Dataset -----------------------
    dataset = pd.read_csv(FILEPATH)

# --------------- Step 2: Handle Missing Values ------------------
    na_info = count_NA_values(dataframe = dataset, plot = False)
    
    thresh_value = round(dataset.shape[0] * 0.15)
    
    df = dataset.dropna(axis = 1, thresh = thresh_value)
    
    if df.empty == True:
        print('Step 2 Failure')
    else:
        print('Step 2 Success')
    
# --------------------- Step 2: Drop Columns ---------------------
    var_to_drop = ['Title', 'Description', 'Created At', 'Updated At', 'Resolution', 'CC', 'Comments', 'Associated Sla Names', 'CC', 'Total Time Spent', 'To First Response (Elapsed)', 'To Resolution (Elapsed)', 'Tags']
    labels = [label for label in var_to_drop if label in df.columns]
    df = df.drop(labels = labels, axis = 1)
     
# --------------------- Step 3: Recode Variables ---------------------    
    dummies = '10081 10085 10086 10088 10089'.split()    
    
    for i in df.index:
        if df.loc[i, 'Resolution Code'] in dummies:
            df.loc[i, 'Resolution Code'] = 'Other'      
            
    y = df['To First Response (Business)'].apply(convert_int)
    labels = ['Within 15 minutes', 'Within an hour', 'Within a day', 'Within 3 days', 'More than 3 days']
    df['To First Response (Business) Categories'] = pd.cut(y, bins = [0, 15, 60, 480, 1440, max(y)], labels = labels, right = False)

    y = df['To Resolution (Business)'].apply(convert_int)
    labels = ['Within 15 minutes', 'Within an hour', 'Within a day', 'Within 3 days', 'More than 3 days']
    df['To Resolution (Business) Categories'] = pd.cut(y, bins = [0, 15, 60, 480, 1440, max(y)], labels = labels, right = False)            

# --------------------- Step 4: Convert datatypes ---------------------
    cat_features = ['State', 'Priority', 'Incident Origin', 
                'Category', 'Site', 'Department', 'Walk Up', 
                'Customer Satisfied?', 'Resolution Code']
    
    date_features = ['Created At (Timestamp)', 'Updated At (Timestamp)', 'Resolved At', 'Closed At']
    
    df[cat_features] = convert_datatypes(dataframe = df, columns = cat_features, dtype = 'category')
    df[date_features] = convert_datatypes(dataframe = df, columns = date_features, dtype = 'datetime')
    
    df['Walk Up'].cat.rename_categories({0.0 : 'No', 1.0 : 'Yes'}, inplace = True)

    
# --------------------- Step 5: Write Cleaned File ---------------------    
    df.to_csv(DIRPATH + '/mwrd_cleaned.csv', index = False)
    
# ------------------------ Incident Subincident Mapping ---------------------------------------
    if ('incident ids' in df.columns) == True:
        print('Working on incident to subincident mappings...')
        new_df = df[df['incident ids'].notnull()]
        new_df = new_df.reset_index()
        new_df.drop('index', axis = 1, inplace = True)
        
        labels_to_skip = ['id', 'incident ids', 'SLM Breaches']
        new_df_cols = [label for label in df.columns if label not in labels_to_skip]
        id_df = pd.DataFrame(columns = ['Incident ID', 'Subincident ID'] + new_df_cols)
        
        index_cntr = 0
        for i in new_df.index:
            print('Working on row #{} out of {} => {}%'.format(index_cntr, new_df.shape[0], round(100* index_cntr/new_df.shape[0], 2)))
            inc_id = new_df.loc[i, 'id']
            sub_inc_ids = new_df.loc[i, 'incident ids'].split(';')
            
            for j in sub_inc_ids:
                new_row = ([inc_id, j] + list(new_df.loc[i, new_df_cols]))
                id_df.loc[index_cntr] = new_row
                index_cntr = index_cntr + 1
                
        l2 = lambda item: item[1:-1]
        id_df['Subincident ID'] = id_df['Subincident ID'].apply(l2)
        id_df['Subincident ID'] = pd.to_numeric(id_df['Subincident ID'])
                
        id_df.to_csv(DIRPATH + '/mwrd_inc_subinc.csv', index = False)
        print('Incident subincident mapping is processed...')                                
    else:
        print('No incident to subincident mappings')

# ------------------------ Incident SLA Breaches Mapping --------------------------------------
    if ('SLM Breaches' in df.columns) == True:       
        print('Working on incident to SLA breach mappings...')
        labels_to_skip = ['id', 'incident ids', 'SLM Breaches']   
        new_df_cols = [label for label in df.columns if label not in labels_to_skip]
        sla_breaches_df = pd.DataFrame(columns = ['Incident ID', 'SLA Breaches'] + new_df_cols)
        index_cntr = 0
        for i in df.index:
            print('Working on row #{} out of {} => {}%'.format(index_cntr, df[df['SLM Breaches'].notnull()].shape[0], round(100* index_cntr/df[df['SLM Breaches'].notnull()].shape, 2)))
            if is_value_nan(df.loc[i, 'SLM Breaches']) != False:
                first_split = df.loc[i, 'SLM Breaches'].split(';')
            
                for j in first_split:
                    second_split = j.split('SLA Name: ')[1][2 : -3]            
                    sla_breaches_df.loc[index_cntr] = [df.loc[i, 'id'], second_split] + list(df.loc[i, new_df_cols])
                    index_cntr = index_cntr + 1                                
                    
        sla_breaches_df.to_csv(DIRPATH + '/mwrd_sla_breaches.csv', index = False) 
        print('Incident to SLA breach mapping is processed...')               
    else:
        print('No SLA Breaches')        