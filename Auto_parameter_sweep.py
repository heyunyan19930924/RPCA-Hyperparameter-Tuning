'''
Automatically do parameter sweep for multiple runs. 
Import data generation functions before running.
'''

import numpy as np
import csv
import random
from numpy import genfromtxt
import pandas as pd
import swat.cas.datamsghandlers as dmh

# Data propert
dimensionX= 500
scoring_dimension = 100
rank = 20
base_multiplier = 100
random_SD = np.sqrt(10)
outlier_row_num = 50
outlier_offset = 500
number_of_runs = 30
total_num_outliers = 50
outlier_num_row = 1
# Specify the parameter to be tweaked, and the varying range of this parameter.
parameter_to_tweak = 'sigmaCoef'
parameter_list = np.linspace(0.15,0.3,number_of_runs)
which_to_score = 2 # 1: lowrank(background); 2: sparse(foreground)
test_parameter_list = np.linspace(0.15,0.3,1)
outlier_indices = created_outlier_indices(total_num_outliers, outlier_num_row)

dataframe = pd.read_csv("base_space")
data_space = dataframe.values

dimension_range = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
output = np.zeros((len(dimension_range), 10))
for row_count, dimensionY in enumerate(dimension_range):
    print("Start working on dimension " + str(dimensionY))
    for counter in range(10):
        print("run " + str(counter))
        col_labels = ['c' + str(idx) for idx in range (-1, dimensionY)]
        col_labels[0] = "row_index"
        data = generate_lowrank(dimensionX + scoring_dimension, dimensionY, rank, base_multiplier, random_SD)
        # Construct training data
        training_data = data[:dimensionX, :]
        dataframe = pd.DataFrame(training_data, columns = col_labels)
        training = 'training_data_row_' + str(dimensionX) + '_col_' + str(dimensionY) + "_rank_" + str(rank)
        upload_dataframe_to_castable(dataframe, training)
        # Construct scoring data
        scoring_data = data[dimensionX:, :]
        outlier_matrix = construct_outlier_matrix(scoring_dimension, dimensionY, outlier_offset, outlier_row_num)
        scoring_data[:,1:] += outlier_matrix
        dataframe = pd.DataFrame(scoring_data, columns = col_labels)
        scoring = 'scoring_data_row_' + str(scoring_dimension) + '_col_' + str(dimensionY) + "_rank_" + str(rank) + "_outlier_" + str(outlier_offset)
        upload_dataframe_to_castable(dataframe, scoring)
        
        # To control RPCA rank, we decrease CumEigPctTol until rank matches the generating data rank.
        Eigtol = 1
        rpca_output, outlier_result = image_parameter_sweep(training, scoring, parameter_to_tweak, test_parameter_list, 1, 10, outlier_num_row, 0, False, Eigtol, which_to_score)
        cur_rank = get_rank_from_rpca_output(rpca_output)[0]
        while (cur_rank > rank):
            Eigtol -= 0.01
            rpca_output, outlier_result = image_parameter_sweep(training, scoring, parameter_to_tweak, test_parameter_list, 1, 10, outlier_num_row, 0, False, Eigtol, which_to_score)
            cur_rank = get_rank_from_rpca_output(rpca_output)[0]
        if cur_rank != rank:
            continue
        rpca_output, outlier_result = image_parameter_sweep(training, scoring, parameter_to_tweak, parameter_list, 1, 10, outlier_num_row, 0, False, Eigtol, which_to_score)
        # Compute accuracy matrix by comparing scoring result with created outlier matrix.
        accuracy_matrix = get_accuracy_matrix(outlier_result, outlier_indices)
        
        check = accuracy_matrix[0,0]
        count = 0
        while (check < 1):
            if (count == (accuracy_matrix.shape[0] - 1)):
                break
            count += 1
            check = accuracy_matrix[count,0]
        output[row_count, counter] = parameter_list[count]

# Write output into log.txt
with open("output_log.txt", "a") as writer:
    writer.write("\n")
    writer.write("#########################################scoring sparse###########################################")
    writer.write("\n")
    writer.write("DimensionY\n")
    np.savetxt(writer, dimension_range, fmt="%0.5f",newline='\n', delimiter=',')
    writer.write(parameter_to_tweak+"\n")
    np.savetxt(writer, parameter_list, fmt="%0.5f",newline='\n', delimiter=',')
    writer.write("\n")
    np.savetxt(writer, output, fmt="%0.5f",newline='\n', delimiter=',')