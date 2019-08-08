'''
Generate low rank matrix for RPCA testing

Update: functions to save tables and submit tables to cas server
Update: Two other versions of generating tables. 
    One is to generate data from the same base subspace. Useful if one intends to conduct multiple runs on same data.
    Another is to generate low rank data where all entries have specified mean/var. 
'''

import numpy as np
import csv
import random
from numpy import genfromtxt
import pandas as pd
import swat.cas.datamsghandlers as dmh
import copy

# Generate low dimension data by setting all entries
def generate_2d(dimensionX) :
    data = np.zeros((dimensionX, 3))
    for i in range(dimensionX) :
        data[i][0] = i * 1.0
        data[i][1] = i * 1.0
        data[i][2] = i * 10.0
    ramdom_data = np.random.randn(dimensionX,2)
    data[:, 1] += 1 * ramdom_data[:, 0]
    data[:, 2] += 3 * ramdom_data[:, 1]
    return data
    
# Generate high dimension data M*N by multiplying two matrices with dimension M*p and p*N. rank <= p.
# First col is index col
def generate_lowrank(dimensionX, dimensionY, rank, multiplier, random_multiplier) :
    matrix1 = np.random.rand(dimensionX, rank) * multiplier
    matrix2 = np.random.rand(rank, dimensionY+1) * multiplier
    data = np.dot(matrix1, matrix2)
    data[:,0] = np.linspace(1, dimensionX, dimensionX)
    noise = np.random.randn(dimensionX, dimensionY) * random_multiplier
    data[:,1:] += noise
    return data

# Construct outlier_matrix
def construct_outlier_matrix(dimensionX, dimensionY, offset, num) :
    random_X_idx = random.sample(range(dimensionX), num)
    random_Y_idx = np.random.randint(dimensionY, size=num)
    outlier_matrix = np.zeros((dimensionX, dimensionY))
    for i in range(num) :
        outlier_matrix[random_X_idx[i]][random_Y_idx[i]] = offset
    return outlier_matrix, random_X_idx

# Print data in .csv file
def write_data_to_file(data, dimensionY, file_name) :
    label_list = []
    label_list.append('index')
    for i in range(1, dimensionY+1):
        label_list.append('label' + str(i))
        with open(file_name, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(label_list)
            for row in data:
                writer.writerow(row)

# Read matrix data from .txt files
def read_data_from_file(file_name):
    data = genfromtxt(file_name, delimiter=',')
    data = data[1:,:]
    return data

def upload_dataframe_to_castable(df, output):
    handler = dmh.PandasDataFrame(df)
    myCAS.addtable(table=output, replace = True, **handler.args.addtable)
 
# Generate low rank matrix from existing space
def generate_low_rank_from_space(subspace, dimensionY, multiplier, random_SD):
    dimensionX, rank = subspace.shape
    matrix = np.random.rand(rank, dimensionY+1)
    data = np.dot(subspace, matrix) * multiplier
    data[:,0] = np.linspace(1, dimensionX, dimensionX)
    noise = np.random.randn(dimensionX, dimensionY) * random_SD
    data[:,1:] += noise
    return data

# Generate low rank matrix with each entry satisfying specifying mean/variance. Use this!
def generate_lowrank_fixed_entry_dist(dimensionX, dimensionY, rank, mean, var, random_SD) :
    dist_mean = np.sqrt(mean/rank)
    dist_var = (-mean + np.sqrt(mean*mean + var*rank)) / rank
    dist_len = np.sqrt(12 * dist_var) / 2
    matrix1 = np.random.uniform(dist_mean - dist_len, dist_mean + dist_len, (dimensionX, rank))
    matrix2 = np.random.uniform(dist_mean - dist_len, dist_mean + dist_len, (rank, dimensionY+1))
    data = np.dot(matrix1, matrix2)
    data[:,0] = np.linspace(1, dimensionX, dimensionX)
    noise = np.random.randn(dimensionX, dimensionY) * random_SD
    data[:,1:] += noise
    return data

######################################################################################################################################
# main function
# inputs for training
dimensionX, dimensionY = 500, 12800
scoring_dimension = 100
rank = 20
base_multiplier = 100
random_SD = np.sqrt(10)
outlier_row_num = 50

# Create base table and dataframe of dimension (training_dimensionX + scoring_dimensionX, dimensionY) 
col_labels = ['c' + str(idx) for idx in range (-1, dimensionY)]
col_labels[0] = "row_index"
data = generate_lowrank(dimensionX + scoring_dimension, dimensionY, rank, base_multiplier, random_SD)
dataframe = pd.DataFrame(data, columns = col_labels)
dataframe.to_csv("base_row_" + str(dimensionX + scoring_dimension) + "_col_" + str(dimensionY) + "_rank_" + str(rank) ,index = False)

data = pd.read_csv("base_row_" + str(dimensionX + scoring_dimension) + "_col_" + str(dimensionY) + "_rank_" + str(rank))
data = data.values
# Training data from data[:dimensionX, :]
training_data = copy.deepcopy(data[:dimensionX, :])
dataframe = pd.DataFrame(training_data, columns = col_labels)

# Write table to .csv files
dataframe.to_csv("training_data_row_" + str(dimensionX) + "_col_" + str(dimensionY) + "_rank_" + str(rank),index = False)

# Upload table to CAS as CAStable
upload_dataframe_to_castable(dataframe, 'training_data_row_' + str(dimensionX) + '_col_' + str(dimensionY) + "_rank_" + str(rank))

##############################################################################
# Scoring
# Training data from data[dimensionX:, :]
outlier_offset = 500

scoring_data = copy.deepcopy(data[dimensionX:, :])
outlier_matrix = construct_outlier_matrix(scoring_dimension, dimensionY, outlier_offset, outlier_row_num)
scoring_data[:,1:] += outlier_matrix
dataframe = pd.DataFrame(scoring_data, columns = col_labels)

# Write table to .csv files
dataframe.to_csv("scoring_data_row_" + str(scoring_dimension) + "_col_" + str(dimensionY) + "_rank_" + str(rank) + "_outlier_" + str(outlier_offset),index = False)

# Upload table to CAS as CAStable
upload_dataframe_to_castable(dataframe, 'scoring_data_row_' + str(scoring_dimension) + '_col_' + str(dimensionY) + "_rank_" + str(rank) + "_outlier_" + str(outlier_offset))

######################################################################################################################################
# Generate low rank matrix from the same subspace
data_space = np.random.rand(dimensionX+scoring_dimension, rank)
dataframe = pd.DataFrame(data_space)
dataframe.to_csv("base_space" ,index = False)

# Read data_space from existing .csv file
dataframe = pd.read_csv("base_space")
data_space = dataframe.values

# inputs for training
dimensionX, dimensionY = 500, 800
scoring_dimension = 100
rank = 20
base_multiplier = 100
random_SD = np.sqrt(10)
outlier_row_num = 50

# Create base table and dataframe 
col_labels = ['c' + str(idx) for idx in range (-1, dimensionY)]
col_labels[0] = "row_index"

data = generate_low_rank_from_space(data_space, dimensionY, base_multiplier, random_SD)
dataframe = pd.DataFrame(data, columns = col_labels)
dataframe.to_csv("base_row_" + str(dimensionX + scoring_dimension) + "_col_" + str(dimensionY) + "_rank_" + str(rank) ,index = False)

data = pd.read_csv("base_row_" + str(dimensionX + scoring_dimension) + "_col_" + str(dimensionY) + "_rank_" + str(rank))
data = data.values
training_data = data[:dimensionX, :]
dataframe = pd.DataFrame(training_data, columns = col_labels)

# Write table to .csv files
dataframe.to_csv("training_data_row_" + str(dimensionX) + "_col_" + str(dimensionY) + "_rank_" + str(rank),index = False)

# Upload table to CAS as CAStable
upload_dataframe_to_castable(dataframe, 'training_data_row_' + str(dimensionX) + '_col_' + str(dimensionY) + "_rank_" + str(rank))

######################################################################################################################################
# Scoring
# Create table and dataframe 
outlier_offset = 500

scoring_data = data[dimensionX:, :]
outlier_matrix = construct_outlier_matrix(scoring_dimension, dimensionY, outlier_offset, outlier_row_num)
scoring_data[:,1:] += outlier_matrix
dataframe = pd.DataFrame(scoring_data, columns = col_labels)

# Write table to .csv files
dataframe.to_csv("scoring_data_row_" + str(scoring_dimension) + "_col_" + str(dimensionY) + "_rank_" + str(rank) + "_outlier_" + str(outlier_offset),index = False)

# Upload table to CAS as CAStable
upload_dataframe_to_castable(dataframe, 'scoring_data_row_' + str(scoring_dimension) + '_col_' + str(dimensionY) + "_rank_" + str(rank) + "_outlier_" + str(outlier_offset))

####################################################################################################################################
data = pd.read_csv("training_data_row_500_col_6400_rank_20")
data = data.values
print(np.mean(data[:,1:]))
print(np.var(data[:,1:]))