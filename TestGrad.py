import time,sys,os;
os.environ["OMP_NUM_THREADS"] = "1" # Improves performance apparently ????

##from mpi4py import MPI # Import this before numpy
import numpy as np
from numpy import pi
import logging


def Adjoint_Gradient_Test(X0,dX0, *Other_args):


	"""
	Preform the Taylor remainder test (see Farrell P. Cotter C. SIAM JSC 2014), that is 

	|J(Bx0 + h*dBx0) - J(Bx0)| -> 0 at O(h),
	
	|J(Bx0 + h*dBx0) - J(Bx0) - h*<dBx0,dJ/dB>| -> 0 at O(h^2),

	by repeating evaluations for h,h/2,h/4,... to determine convergence order.

	Do so on the functions:

	FWD_Solve_PBox_IND_MHD.FWD_Solve
	FWD_Solve_PBox_IND_MHD.ADJ_Solve

	Using the inner product as defined by:

	FWD_Solve_PBox_IND_MHD.Integrate_Field

	~~~~~~~~~~~~~~~~~~~~~~

	Inputs:
	Bx0   - 1D np.array initial vector satisfying bcs + divergence condition
	dBx0  - 1D np.array perturb vector satisfying bcs + divergence condition

	*Other_args - see DAL_PCF_MAIN.py for definition

	Returns:
	None

	"""
	
	# Codes Written
	from FWD_Solve_PBox_IND_MHD import FWD_Solve_IVP_Lin
	from FWD_Solve_PBox_IND_MHD import ADJ_Solve_IVP_Lin
	from FWD_Solve_PBox_IND_MHD import Inner_Prod

	# Set to info level rather than the debug default
	root = logging.root
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");
		#h.setLevel("DEBUG")
	logger = logging.getLogger(__name__)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# (1) Compute J(B_0), dJ/dB_0
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	logger.info("JB0 FWD_Solve running .... \n")
	start_time = time.time()
	J_ref 	   = FWD_Solve_IVP_Lin(X0,	*Other_args);
	end_time   = time.time()
	print('Total time fwd: %f' %(end_time-start_time))
	

	logger.info("dJ Adjoint_Solve running .... \n")
	start_time = time.time()
	dJdX 	   = ADJ_Solve_IVP_Lin(X0,	*Other_args);
	end_time   = time.time()
	print('Total time adjoint: %f' %(end_time-start_time))

	logger.info("Computing Inner product <dL/dB,dB >_adj  .... \n")
	domain = Other_args[0];
	W_ADJ  = Inner_Prod(domain,dX0,dJdX);

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# (2) Loop for (A) W_fd = <dL/dB,dB >_fd, (B) W_adj = <dL/dB,dB >_adj  
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	epsilon = 1e-04; #0.75;
	logger.info('epsilon = %e'%epsilon)

	N_test = 5;
	EPSILON     = np.zeros(N_test);
	TEST_SUM_R  = np.zeros(N_test); 
	TEST_SUM_R2 = np.zeros(N_test);

			
	# Include a for loop to compute a range of epsilon
	for test in range(N_test):

		TAY_R  = 0.0; # Compute the Taylor remainder -- checks if we've a gradient
		TAY_R2 = 0.0; # Compute the 2^nd Taylor remainder -- checks convergence of adjoint

		J_fd = FWD_Solve_IVP_Lin( X0 + epsilon*dX0, *Other_args);
		
		TAY_R  = abs(J_fd - J_ref); # Should go like O(h);
		TAY_R2 = abs(J_fd - J_ref - epsilon*W_ADJ); # Should go like O(h^2);

		# ~~~~~~~~~# (B.2) 2nd order Central differencing ~~~~~~~~~~	
		#J_fd = FWD_Solve(Bx0 + epsilon*dBx0,*Other_args);
		#J_bck = FWD_Solve(Bx0 - epsilon*dBx0,*Other_args);
		#TAY_R = abs(J_fd - J_bck); # Should go like O(h);
		#TAY_R2 = abs(J_fd - J_bck - 2.*epsilon*W_ADJ); # Should go like O(h^3), but it doesn't ?????

		logger.info('#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~')

		logger.info('epsilon = %e'%epsilon)
		logger.info('|J(B + eps*db) - J(B)| = %e'%TAY_R);
		logger.info('|J(B + eps*db) - J(B) - eps*dJ.dB| = %e'%TAY_R2);

		tay_r_div = TAY_R/epsilon
		logger.info('|J(B + eps*db) - J(B)|/eps = %e'%tay_r_div);
		logger.info('|dJ.dB| = %e'%W_ADJ);

		logger.info('#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~')

		# ~~~~~~~~~~~ Log errors and decrement epsilon ~~~~~~~~~
		EPSILON[test] = epsilon;
		TEST_SUM_R[test] = TAY_R;
		TEST_SUM_R2[test] = TAY_R2;

		epsilon = 0.5*epsilon;

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# 3) Compute Scalings
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# Arrays to store the slopes

	AA = np.zeros( (5,N_test) )
	AA[0,:] = EPSILON;
	AA[1,:] = TEST_SUM_R
	AA[2,:] = TEST_SUM_R2

	logger.info('#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~')

	SUM_UP = 0.0; exp = 0.0;
	for i in range(N_test-1):
		exp = np.log(TEST_SUM_R[i]/TEST_SUM_R[i+1])/np.log(EPSILON[i]/EPSILON[i+1])
		logger.info('exponent = %e'%exp);
		SUM_UP+= np.log(TEST_SUM_R[i]/TEST_SUM_R[i+1])/np.log(EPSILON[i]/EPSILON[i+1]);
		AA[3,i] = np.log(TEST_SUM_R[i]/TEST_SUM_R[i+1])/np.log(EPSILON[i]/EPSILON[i+1]);

	GT1 = SUM_UP/(N_test-1);
	logger.info('Gamma TAYLOR = %d'%GT1);

	logger.info('#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~')


	SUM_UP = 0.0; exp = 0.0;
	for i in range(N_test-1):
		exp = np.log(TEST_SUM_R2[i]/TEST_SUM_R2[i+1])/np.log( EPSILON[i]/EPSILON[i+1] )
		logger.info('exponent = %e'%exp);
		SUM_UP+= np.log(TEST_SUM_R2[i]/TEST_SUM_R2[i+1])/np.log(EPSILON[i]/EPSILON[i+1]);
		AA[4,i] = np.log(TEST_SUM_R2[i]/TEST_SUM_R2[i+1])/np.log(EPSILON[i]/EPSILON[i+1]);

	GT2 = SUM_UP/(N_test-1)
	logger.info('Gamma TAYLOR_2 = %e'%GT2);

	logger.info('#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~')


	np.save("eps_TestR_TestR2_h_h2.npy",AA);

	return None;