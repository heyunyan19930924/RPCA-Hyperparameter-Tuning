'''
Generate plots for dimension analysis. Each section prints one specific plot.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_histogram_from_table(data, title, save_location):
    plt.hist(data, bins=10)
    plt.title(title)
    plt.savefig(save_location)
    
generate_histogram_from_table(data, "Raw data histogram", "Raw data histogram.png")

######################################################################################

x_axis = [250, 500, 750, 1000]
f1_lower = [0.19, 0.19, 0.19, 0.19]
f1_upper = [2.34, 4.5, 11.5, 21.8]

plt.plot(x_axis, f1_lower, 'bs', label = "Lower bound")
for i, value in enumerate(f1_lower):
    plt.annotate(value, (x_axis[i], value-0.5))
plt.plot(x_axis, f1_upper, 'rs', label = "Upper bound")
for i, value in enumerate(f1_upper):
    plt.annotate(value, (x_axis[i], value+0.25))
plt.legend()
plt.title("Range of best sigmaCoef")
plt.xlabel("Magnitude of outlier")
plt.ylabel("sigmaCoef")
plt.savefig("Range of best sigmaCoef numSigVar=1.png")

######################################################################################

x_axis = [250, 500, 750, 1000]
f2_lower = [1.41, 2.85, 7.1, 12.8]
f2_upper = [1.5, 3, 7.5, 13.5]

plt.plot(x_axis, f2_lower, 'bs', label = "Lower bound")
for i, value in enumerate(f2_lower):
    plt.annotate(value, (x_axis[i], value-0.5))
plt.plot(x_axis, f2_upper, 'rs', label = "Upper bound")
for i, value in enumerate(f2_upper):
    plt.annotate(value, (x_axis[i], value+0.25))
plt.legend()
plt.title("Range of best sigmaCoef")
plt.xlabel("Magnitude of outlier")
plt.ylabel("sigmaCoef")
plt.savefig("Range of best sigmaCoef numSigVar=20.png")

######################################################################################

x_axis = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
x_axis = np.sqrt(np.log(x_axis))
f3 = [0.16842, 0.17895, 0.19737, 0.22632, 0.24211, 0.26316, 0.25263, 0.27632]
plt.plot(x_axis, f3)
plt.title("SigmaCoef threshold against dimensions")
plt.xlabel("Square root of log of dimensions")
plt.ylabel("SigmaCoef threshold")

########################################################################################
x_axis = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
x_axis = np.sqrt(np.log(x_axis))
dim_data = np.genfromtxt("output_log_rank20.txt", delimiter = ",")
mean = np.mean(dim_data, axis = 1)
plt.plot(x_axis, mean)
for i in range(10):
    plt.scatter(x_axis, dim_data[:,i], c = "black", s = 10)
plt.title("SigmaCoef threshold against dimensions")
plt.xlabel("Square root of log of dimensions")
plt.ylabel("SigmaCoef threshold")
plt.savefig("SigmaCoef threshold against dimensions rank 20.png")

########################################################################################
x_axis = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
x_axis = np.sqrt(np.log(x_axis))
dim_data = np.genfromtxt("output_log_sparse.txt", delimiter = ",")
mean = np.mean(dim_data, axis = 1)
plt.plot(x_axis, mean)
for i in range(10):
    plt.scatter(x_axis, dim_data[:,i], c = "black", s = 10)
plt.title("SigmaCoef threshold against dimensions")
plt.xlabel("Square root of log of dimensions")
plt.ylabel("SigmaCoef threshold")
plt.savefig("SigmaCoef threshold against dimensions.png")

################################ For R4S Method ##########################################################
x_axis = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
noise_data = np.genfromtxt("output_log_noise_sweep.txt", delimiter = ",")
noise_data[4,7] = 5.00
mean = np.mean(noise_data, axis = 1)
plt.plot(x_axis, mean)
for i in range(10):
    plt.scatter(x_axis, noise_data[:,i], c = "black", s = 10)
axes = plt.gca()
axes.set_ylim([3,6])
plt.title("SigmaCoef threshold against noise variance")
plt.xlabel("Variance of noise")
plt.ylabel("SigmaCoef threshold")
plt.savefig("SigmaCoef threshold against noise.png")