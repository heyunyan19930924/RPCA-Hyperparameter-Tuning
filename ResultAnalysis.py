'''
Do RPCA sweep on a list of parameters. Which parameter to sweep and what is the sweeping range is specified in the function
'''

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

# This function sweeps through parameter_range for parameter. 
def rpca_parameter_sweep(data_file, _parameter, parameter_range, _lambdaWeight, _sigmaCoef, _numSigVars, _anomalyDetectionMethod, _useMatrix, eig_tol):
    counter = 0
    myCAS.loadactionset(actionset="robustPca")
    myCAS.loadactionset(actionset="astore")
    output = []
    for i in parameter_range:
        sparsematName = "sparsemat_" + str(i)
        if (_parameter == 'lambdaWeight'):
            result = myCAS.robustPca(
                table={"name" : data_file},
                id={"index"},
                method="ALM",
                decomp="svd", 
                scale=True, 
                center=True,
                lambdaWeight= i,
                cumEigPctTol=eig_tol,
                saveState={"name":"store", "replace":True}, 
                outmat={"sparseMat":{"name":sparsematName, "replace":True}},
                anomalyDetection=True,
                anomalyDetectionMethod=_anomalyDetectionMethod, # 0 : SIGVAR method; 1: R4S methods
                sigmaCoef=_sigmaCoef, # Threshold to identify an observation to be outlier
                numSigVars=_numSigVars, # SIGVAR ONLY: Number of outliers for data to be identified as anomaly
                useMatrix=_useMatrix # False: Use standard deviation of original data True: Use standard deviation of sparse data   
                )
        elif (_parameter == 'sigmaCoef') :
            myCAS.robustPca(
                table={"name" : data_file},
                id={"index"},
                method="ALM",
                decomp="svd", 
                scale=True, 
                center=True,
                lambdaWeight= _lambdaWeight,
                cumEigPctTol=eig_tol,
                saveState={"name":"store", "replace":True}, 
                outmat={"sparseMat":{"name":sparsematName, "replace":True}},
                anomalyDetection=True,
                anomalyDetectionMethod=_anomalyDetectionMethod, # 0 : SIGVAR method; 1: R4S methods
                sigmaCoef=i, # Threshold to identify an observation to be outlier
                numSigVars=_numSigVars, # SIGVAR ONLY: Number of outliers for data to be identified as anomaly
                useMatrix=_useMatrix # False: Use standard deviation of original data True: Use standard deviation of sparse data   
                )
        elif (_paramete == 'numSigVars') :
            myCAS.robustPca(
                table={"name" : data_file},
                id={"index"},
                method="ALM",
                decomp="svd", 
                scale=True, 
                center=True,
                lambdaWeight= _lambdaWeight,
                cumEigPctTol=eig_tol,
                saveState={"name":"store", "replace":True}, 
                outmat={"sparseMat":{"name":sparsematName, "replace":True}},
                anomalyDetection=True,
                anomalyDetectionMethod=_anomalyDetectionMethod, # 0 : SIGVAR method; 1: R4S methods
                sigmaCoef=_sigmaCoef, # Threshold to identify an observation to be outlier
                numSigVars=i, # SIGVAR ONLY: Number of outliers for data to be identified as anomaly
                useMatrix=_useMatrix # False: Use standard deviation of original data True: Use standard deviation of sparse data   
                )
        else :
            print("Syntax unknown")
            break
        output.append(result)
        myCAS.score(
                table={"name" : data_file},
                options=[{"name":"RPCA_PROJECTION_TYPE","value":2}],
                rstore={"name":"store"}, 
                out={"name":"scored"+_parameter+str(i),"replace":True}
                )
        counter += 1
    return output

# Output accuracy matrix by comparing scoring results with created outlier matrix.
def analyze_scoring(_parameter, parameter_range, outlier_row):
    num_runs = parameter_range.size
    output = np.zeros(num_runs)
    counter = 0
    for i in parameter_range:
        cur_score = myCAS.fetch(table = {"name":"scored"+_parameter+str(i)}, to=10000, maxRows = 100000)['Fetch']
        cur_score_data = cur_score.values
        detected_outlier_row = np.nonzero(cur_score_data[:, -1])[0]
        intersect_row = np.intersect1d(detected_outlier_row, outlier_row)
        output[counter] = detected_outlier_row.size + outlier_row.size - 2*intersect_row.size
        counter += 1
    return output

# Write result into a .txt file
def write_output(data_to_write, file_name, _parameter, parameter_list, rank) :
    with open(file_name, "a") as writer:
        writer.write("\n")
        writer.write("###############################################################\n")
        writer.write(" RPCA data2\n")
        writer.write(_parameter+":\n ")
        np.savetxt(writer, np.transpose(parameter_list), fmt="%0.5f",newline=',')
        writer.write("\nRank of low_rank matrix\n")
        np.savetxt(writer, np.transpose(rank), fmt="%d",newline=',')
        writer.write("\nnumber of different outliers:\n")
        np.savetxt(writer, np.transpose(data_to_write), fmt="%d",newline=',')

# Get rank. Because difference in rank does effect on the scoring result, the goal for this function is to control the rank. 
def get_rank_from_rpca_output(rpca_output):
    num_of_runs = len(rpca_output)
    output = np.zeros(num_of_runs)
    for i in range(num_of_runs):
        output[i] = rpca_output[i]['Summary']['Value'][1]
    return output

def plot_ss_parameter_sweep(_parameter, parameter_range, outlier_label) :
    num_of_runs = parameter_range.size
    to_return_normal = np.zeros(num_of_runs)
    to_return_outlier = np.zeros(num_of_runs)
    for i in range(num_of_runs):
        cur_score = myCAS.fetch(table = {"name":"scored"+_parameter+str(parameter_range[i])}, to=10000, maxRows = 100000)['Fetch']
        cur_score_val = cur_score.values
        (num_row, num_col) = tmp.shape
        cur_score_val = cur_score_val[:, 1:num_col-1]
        ss = [np.sqrt(sum(x*x)) for x in cur_score_val]
        ss_outlier = [ss[j] for j in outlier_label]
        ss_normal = [ss[j] for j, _ in enumerate(ss) if j not in outlier_label]
        ss_outlier = np.array(ss_outlier)
        ss_normal = np.array(ss_normal)
        to_return_normal[i] = np.mean(ss_normal)
        to_return_outlier[i] = np.mean(ss_outlier)
    return to_return_outlier, to_return_normal

# Main function
# Read file 
data_file = 'RPCA\RPCAdata' # File name
full_path = 'U:\\'+data_file+'.csv'
read_file_cvs(full_path)
data = read_data_from_file('RPCAdata.csv')
outlier_matrix = read_data_from_file('outlier.csv')

number_of_runs = 40
parameter_to_tweak = 'lambdaWeight'
parameter_list = np.linspace(1.0,1.1,number_of_runs)
output_nonzero = np.zeros(number_of_runs)
(nonzero_outlier_row, nonzero_outlier_col) = np.nonzero(outlier_matrix)
rpca_output = rpca_parameter_sweep(data_file, parameter_to_tweak, parameter_list, 1, 10, 1, 1, False, 0.99)
rpca_rank = get_rank_from_rpca_output(rpca_output)
output = analyze_scoring(parameter_to_tweak, parameter_list, nonzero_outlier_row)
write_output(output, "Score analysis against lambdaWeight.txt", parameter_to_tweak, parameter_list, rpca_rank)

# Check single output sparse matrix
tmp = myCAS.fetch(table = {"name":"scored"+parameter_to_tweak+str(parameter_list[10])}, to=10000, maxRows = 100000)['Fetch']
tmp_value = tmp.values
(num_row, num_col) = tmp.shape
tmp_value = tmp_value[:, 1:num_col - 1]
# Compute R4S: sum of square of rows
R4S = [np.sqrt(sum(x*x)) for x in tmp_value]
tmp_value[:,-1]
########################################################### SS analysis #################################################################

(ss_outlier, ss_normal) = plot_ss_parameter_sweep('lambdaWeight', parameter_list, nonzero_outlier_row)
plt.plot(parameter_list, ss_outlier, 'ro')
plt.plot(parameter_list, ss_normal, 'bo')
plt.xlabel('lambdaWeight')
plt.ylabel('RSSS of standardized sparse matrix')
plt.title('plot RSSSS against lambdaWeight')
plt.savefig('plot_RSSSS_against_lambdaWeight_dim_10000_100_low_10_R4S_sigma_1_OV_300.png')
########################################################### Further analysis ############################################################
          
working_table = myCAS.fetch(table = {"name":sparsematName}, to=100)['Fetch']
working_table = working_table.values
# Now working array is a nparray.
working_table = working_table[:,1:]
# print(working_table)
(nonzero_sparsemat_row, nonzero_sparsemat_col) = np.nonzero(working_table)
intersect_row = np.intersect1d(nonzero_sparsemat_row, nonzero_outlier_row)
output_nonzero[counter] = nonzero_outlier_row.size + nonzero_sparsemat_row.size - 2*intersect_row.size
# output_L1[counter] = LA.norm(diff,1)
# output_L2[counter] = LA.norm(diff,2)
counter = counter + 1
  
# Plot the output against varying parameter
plt.scatter(lambda_weight_list, output_nonzero)
plt.axis([0, 10, 0, 4])
plt.show