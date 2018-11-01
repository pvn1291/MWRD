#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 13:00:55 2018

Library Functions,
    1. Convert Datatypes
    2. Count NA values & plot heatmap
    3. Correation matrix with corr plot
    4. Univariate plots

@author: pvn1291
"""
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as matplot
import seaborn as sns
"""
import os

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


def compute_correlation_matrix(dataframe, plot = True):     
    """
        Function computes the correlation values for numeric columns and plots the heatmap.
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
    
    corr_mat = dataframe.corr()
    
    #Mask the upper triangle of matrix
    mask = np.zeros_like(corr_mat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    if plot == True:
        print('\n\n'
              + '------------------------------ Heatmap of correlation values ------------------------------'
              + '\n\t\t\t Annotations describe degree of correlation')
        matplot.figure(figsize = (15, 8))
        sns.heatmap(data = dataframe.corr(), cmap = 'coolwarm', linewidths = 0.1, 
                    annot = True, mask = mask) 

    corr_df = pd.DataFrame(corr_mat)
    
    return corr_df

def generate_univariate_plots(dataframe, columns, coltype, save, path):
    """
    Function generates univariate plots and save plots in given location.
    
    Parameters,
        dataframe (pandas dataframe) = reference of the dataframe
        columns (list) = input columns to be converted
        coltype (str) = type of column, valid values = 'numeric', 'category'
        save (boolean) = True if you want to save the plot, False otherwise
        path (str) = specify path if save is True
    """
    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as matplot
    import seaborn as sns

    valid_dtypes = ['category', 'numeric']
    
    assert (dataframe.empty == False), 'Dataframe is empty.'
    assert (len(columns) > 0), 'Column list is empty.'
    assert (coltype in valid_dtypes), 'Datatype specified is not valid.'
    
    
    for feature in columns:
        if coltype == 'numeric':
            matplot.figure(figsize = (15,8))
            sns.distplot(a = dataframe[feature], rug = True, rug_kws = {'height': 0.05})
            matplot.xlabel(xlabel = feature)
            matplot.ylabel(ylabel = 'Frequency')
            matplot.title(label = str('Histogram: ' + feature.upper()))
            
            if save == True:
                fname = str(path + '/Histogram - ' + feature + '.jpeg')
                matplot.savefig(fname, dpi = 200)
        
        if coltype == 'category':
            matplot.figure(figsize = (15,8))
            sns.countplot(feature, data = dataframe)
            matplot.xlabel(xlabel = feature)
            matplot.ylabel(ylabel = 'Frequency')
            matplot.title(label = str('Barplot: ' + feature.upper()))
            
            if save == True:
                fname = str(path + '/Barplot - ' + feature + '.jpeg')
                matplot.savefig(fname, dpi = 200)
    

import pandas as pd
x = pd.DataFrame({0: [1,2,3,4,5], 1: [6,7,8,9,0], 2: ['a', 'b','a','a','b']})    
x.columns = ['num1', 'num2', 'cat']
    
path = '/Users/pvn1291/WorkSpace/MWRD'

generate_univariate_plots(x, ['cat'], 'category', True, path)

    
    
    
    
    
