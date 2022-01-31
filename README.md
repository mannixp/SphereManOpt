# KinematicDynamoDiscrete
Optimisation code to search the "best" kinematic dynamo in a periodic box following (A.P. Willis, PRL, 2012). In contrast to the common approach of deriving the adjoint of the continuous forward equations, this code uses the adjoint of the discrete forward equations & objective function. 

Combing this approach with an efficient line-search and both Wolfe conditions superior convergence is achieved, reaching a relative max/min of the objective function with a smaller residual in less iterations. 


Running the code requires an anaconda environment with the parallelised spectral code Dedalus installed. Having installed Dedalus and activated the relevant conda environment the optimisation code can be run and the results plotted by executing:

'''
./run_PBox_problem.sh
'''
