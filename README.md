# RPCA-Hyperparameter-Tuning
RPCA hyperparameter tuning schemes exploration

Fundamental statistics: Standardized sparse matrix entry
𝑆𝑆_𝑠𝑐𝑜𝑟𝑒^𝑗=(𝑆_𝑠𝑐𝑜𝑟𝑒^𝑗)/𝜎_𝑗 
	Where 𝜎_𝑗 can be SD of j-th variable either from training input matrix 𝑀 or sparse matrix 𝑆.
SIGVARS method
	Determine each variable is outlier by comparing |〖𝑆𝑆〗_𝑠𝑐𝑜𝑟𝑒^𝑗 | with SIGMACOEF. Count number of outliers. If exceeds NUMSIGVARS, this observation is anomaly.
R4S method
𝑅𝑆𝑆𝑆𝑆_𝑠𝑐𝑜𝑟𝑒=√(∑([𝑆𝑆_𝑠𝑐𝑜𝑟𝑒^𝑗^2 )
Compared with the SD of the same measure calculated on OUTSPARSE multiplied by SIGMACOE.

Hyperparameters:

Anomaly detection method: SIGVARS or R4S.
SIGMACOEF: threshold for critical statistics of variables to be considered as outlier.
NUMSIGVARS: minimum number of significant variables in the sparse part of an observation for it to be classified as an anomaly.

Goal: to find consistent guidelines for hyperparameter tuning which is relatively stable against various properties of data. 

Code:
Generates test datasets with specified distribution. Then do grid search on sigmaCoef and numSigVars.

Findings:
1. The range of ideal sigmaCoef (sigmaCoef that makes anomaly detection accuracy to be 100%) expands as outlier magnitude increases, but the lower bound is fixed.
The range of ideal sigmaCoef shrinks as we increase NumSigVars.
2. When input data attains similar properties (overall mean, variance and noise), the lower bound of ideal SigmaCoef is independent of the number of observations, outlier magnitude and rank of input data.
If we standardize 𝑆𝑆_𝑗^𝑠𝑐𝑜𝑟𝑒 by the SD of training sparse matrix, the ideal SigmaCoef lower bound is also independent of the strength of noise.
3. The lower bound of ideal sigmaCoef depends on the number of variables of input data. 
4. Hypothesis: for SIGVARS method the lower bound of ideal sigmaCoef is proportional to √(2ln⁡𝑛 ), where 𝑛 is the number of variables. Our simulation experiments support the hypothesis with Gaussian noise. (Inspired by Extreme Value Theory) 
