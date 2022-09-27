import time
import numpy as np
import logging

def Adjoint_Gradient_Test(X0,dX0, FWD_Solve,ADJ_Solve,Inner_Prod,args_f,args_IP,epsilon = 1e-04):


	"""
	Preform the Taylor remainder test (see Farrell P. Cotter C. SIAM JSC 2014), that is

	|J(Bx0 + h*dBx0) - J(Bx0)| -> 0 at O(h),

	|J(Bx0 + h*dBx0) - J(Bx0) - h*<dBx0,dJ/dB>| -> 0 at O(h^2),

	by repeating evaluations for h,h/2,h/4,... to determine convergence order.

	Inputs:
	X0  - initial condition
	dX0 - perturbation

	FWD_Solve - forward code callable
	ADJ_Solve - adjoint code callable
	Inner_Prod- inner product code callable

	args_f - positional arguments to pass to FWD_Solve & ADJ_Solve
	args_IP- positional arguments to pass to Inner_Prod

	Returns:
	None

	"""

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
	if isinstance(X0, list):
		J_ref 	   = FWD_Solve(X0  ,	*args_f);
	else:
		J_ref 	   = FWD_Solve([X0],	*args_f);
	end_time   = time.time()
	print('Total time fwd: %f' %(end_time-start_time))


	logger.info("dJ Adjoint_Solve running .... \n")
	start_time = time.time()
	if isinstance(X0, list):
		dJdX 	   = ADJ_Solve(X0  ,	*args_f);
	else:
		dJdX 	   = ADJ_Solve([X0],	*args_f);
	end_time   = time.time()
	print('Total time adjoint: %f' %(end_time-start_time))

	logger.info("Computing Inner product <dL/dB,dB >_adj  .... \n")
	if isinstance(dX0, list):
		W_ADJ = 0.
		for f,g in zip(dX0,dJdX):
			W_ADJ += Inner_Prod(f,g,*args_IP);
	else:
		W_ADJ = Inner_Prod(dX0,dJdX[0],*args_IP)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# (2) Loop for (A) W_fd = <dL/dB,dB >_fd, (B) W_adj = <dL/dB,dB >_adj
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	logger.info('epsilon = %e'%epsilon)

	N_test = 5;
	EPSILON     = np.zeros(N_test);
	TEST_SUM_R  = np.zeros(N_test);
	TEST_SUM_R2 = np.zeros(N_test);


	# Include a for loop to compute a range of epsilon
	for test in range(N_test):

		TAY_R  = 0.0; # Compute the Taylor remainder -- checks if we've a gradient
		TAY_R2 = 0.0; # Compute the 2^nd Taylor remainder -- checks convergence of adjoint
		if isinstance(X0, list) and isinstance(dX0, list):
			Pert = [f + epsilon*g  for f,g in zip(X0,dX0)];
			J_fd = FWD_Solve( Pert,	*args_f);
		else:
			J_fd = FWD_Solve([X0 + epsilon*dX0],	*args_f);

		TAY_R  = abs(J_fd - J_ref); # Should go like O(h);
		TAY_R2 = abs(J_fd - J_ref - epsilon*W_ADJ); # Should go like O(h^2);

		logger.info('#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~')

		logger.info('epsilon = %e'%epsilon)
		logger.info('|J(B + eps*db) - J(B)| = %e'%TAY_R);
		logger.info('|J(B + eps*db) - J(B) - eps*dJ.dB| = %e'%TAY_R2);

		tay_r_div = TAY_R/epsilon
		logger.info('|J(B + eps*db) - J(B)|/eps = %e'%tay_r_div);
		logger.info('|dJ.dB| = %e'%W_ADJ);

		logger.info('#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~')

		# ~~~~~~~~~~~ Log errors and decrement epsilon ~~~~~~~~~
		EPSILON[test]     = epsilon;
		TEST_SUM_R[test]  = TAY_R;
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
