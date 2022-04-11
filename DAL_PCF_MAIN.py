import time,sys,os;
os.environ["OMP_NUM_THREADS"] = "1" # Improves performance apparently ????

from mpi4py import MPI # Import this before numpy
import numpy as np
import h5py, logging

## Dedalus Libraries
#import dedalus.public as de

################################################################################################################
# Perform Optimisation using Direct Adjoint Looping (DAL) 
# !! Yet to be re-written + commented !!
################################################################################################################
#'''
def DAL_LOOP(X0, Constraints, *Other_args):

	# !! Yet to be commented !!
	from Normed_LineSearch import Norm_constraint, line_search_armijo, line_search_wolfe2
	#from Normed_LineSearch import Inner_Prod

	# Self-Written Functions
	from FWD_Solve_PBox_IND_MHD import FWD_Solve_IVP_Lin as FWD_Solve
	from FWD_Solve_PBox_IND_MHD import ADJ_Solve_IVP_Lin as ADJ_Solve
	from FWD_Solve_PBox_IND_MHD import Inner_Prod, Inner_Prod_3

	import copy, math

	# Set to info level rather than the debug default
	root = logging.root
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");
	logger = logging.getLogger(__name__)

	# 1) Optimiser - defaults to armijo for first iteration is this critical??
	LS_method = "Scipy_armijo"; # Uses Quad/cubic minizer determining a good step-length starting from previous
	#LS_method = "wolfe12"; # Requires J(B_0),dJ(B_0) => at least 3 x armijo cost 

	# 2) Termination Codition Parameters
	Terminate_error = 1e-06;
	Grad_error = 1.; Cost_error = 1.

	MAX_DAL_LOOPS = 20 + 2;
	DAL_LOOP_COUNTER = 0;
	LS_COUNTER_f,LS_COUNTER_g = 0,0;
	alpha_k = np.ones(2)*(np.pi/2.0) - 1e-5;

	domain = Other_args[0];

	# 3) Initialise line-search/gradient-descent
	B_k = X0; # Depends on optimisation
	dJdB_k = np.zeros(X0.shape);
	J_k = 0.;

	dJdB_k_old = np.zeros(X0.shape);
	alpha_k_old = alpha_k;
	J_k_old = 0.;

	# Buffers to store progress of gradient optimisation
	J_k_objective = []; J_k_error = []; dJ_k_error = []; theta_k_progB = []; theta_k_progU = [];

	# Grad-Descent Run
	# ~~~~~~~~~~~~~~~~~~~~` # ~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~
	logger.info(' \n\n A) ----> Fwd_solve t = [0 -> T] + Gradient Decent')	
	
	# Compute J(B_k,U_k)
	J_k_old = FWD_Solve(B_k, *Other_args);
	J_k_objective.append(J_k_old);
	LS_COUNTER_f += 1

	# Compute dJ(B_k) & dJ(U_k)
	dJdB_k = ADJ_Solve(B_k,	*Other_args);
	gfk    = copy.deepcopy(dJdB_k); # Reqd. for Armijo condition
	LS_COUNTER_g += 1;

	File_Manips(DAL_LOOP_COUNTER);  #Include to check first noisy IC
	DAL_LOOP_COUNTER+=1;
	
	# need to update this such that dJdB_k is the tangent vector otherwise mistake!!!!
	THETA_K = np.zeros(2)
	XK     = np.split(B_k,     2);
	dXK    = np.split(dJdB_k,  2);
	for i in range(len(Constraints)):

		# 1) Project the gradient(s) to be orthogonal to B_k
		# 1.A) Bx0 Vector
		IP_Xhatk_gk = Inner_Prod_3(domain, (-1.)*dXK[i],XK[i] )
		Norm_Xhatk  = np.sqrt( Inner_Prod_3(domain, XK[i] ,XK[i] ) );
		Norm_gk     = np.sqrt( Inner_Prod_3(domain, dXK[i],dXK[i]) );
		
		THETA_K[i] = math.acos( IP_Xhatk_gk/(Norm_gk*Norm_Xhatk) );

		C1 = Inner_Prod_3(domain, dXK[i],XK[i])/Inner_Prod_3(domain,XK[i],XK[i]);
		dXK[i]    =    dXK[i] -    XK[i]*C1; 

		C2 = np.sqrt(Constraints[i]/Inner_Prod_3(domain,dXK[i],dXK[i]))
		dXK[i]    =    dXK[i]*C2;

	# Re-combine gradient vector after applying
	dJdB_k = np.concatenate( (dXK[0],dXK[1]) ); 

	# ~~~~~~~~~~ # ~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~
	theta_k_progB.append(THETA_K[0]); theta_k_progU.append(THETA_K[1]);		

	# Update B_k, returns tangent hg_k but not the one on the unit hypersphere 
	#for con in range(len(Constraints)):
		
	while np.tan(alpha_k[0]/2.) > np.tan(THETA_K[0]):
		# reduce the step-size to satisfy the constraint
		alpha_k[0] = 0.95*alpha_k[0];

		if alpha_k[0] < Terminate_error:
			logger.info("No step size gives descent for con = %i !!!Terminating!!! "%con)
			sys.exit()	

		logger.info("alpha_k[con] = %e"%alpha_k[0]); con =3;
		OUT = line_search_armijo(FWD_Solve, con , Constraints,		B_k,dJdB_k,gfk,	J_k_objective[-1],args=(Other_args),alpha0=alpha_k[0]);
		alpha_k[:], J_k  = OUT[0],np.ones(2)*OUT[2]; 
		LS_COUNTER_f += OUT[1];
		logger.info('--> J(Bx0)_evals %i, alpha_k=%e'%(OUT[1],alpha_k[0]))

	XK  = np.split(B_k,   2);
	PK  = np.split(dJdB_k,2);
	EQs = Constraints;
	
	for i in range(len(EQs)):
		
		XK[i],PK[i] = Norm_constraint(domain, XK[i],PK[i], alpha_k[i], EQs[i]);
		'''
		if (THETA_K[i] < .5*np.pi):

			if (np.tan(alpha_k[i]/2.) < np.tan(THETA_K[i])):
				logger.info('Constraint i=%i     updated as tan(alpha/2) = %e  < tan(theta_k) = %e'%(i,np.tan(alpha_k[i]/2.),np.tan(THETA_K[i])) )
				XK[i],PK[i] = Norm_constraint(domain, XK[i],PK[i], alpha_k[i], EQs[i]);
			else:
				logger.info('Constraint i=%i not updated as tan(alpha/2) = %e !< tan(theta_k) = %e'%(i,np.tan(alpha_k[i]/2.),np.tan(THETA_K[i])) )
				##XK[i],PK[i] = Norm_constraint(domain, XK[i],PK[i],         0., EQs[i]);	
			
		else:
			logger.info('Constraint i=%i not updated as theta_k > pi/2 = %e'%(i,THETA_K[i]) )
			##XK[i],PK[i] = Norm_constraint(domain, XK[i],PK[i], 0., EQs[i]);
		'''

	# ~~~~~~~~~~~~~~~~~~~~` # ~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~

	B_k  = np.concatenate( (XK[0],XK[1]) );
	hg_k = np.concatenate( (PK[0],PK[1]) ); # Tangent vector - but not on the hyper-sphere

	# Compute J(B_k,U_k)
	J_k_old = FWD_Solve(B_k, *Other_args);
	J_k_objective.append(J_k_old);
	LS_COUNTER_f += 1;
	logger.info('FWD solve with both alpha_k J(X) = %e \n\n'%J_k_old);

	File_Manips(DAL_LOOP_COUNTER); #Include to check first update from gradient
	DAL_LOOP_COUNTER+=1;

	while (DAL_LOOP_COUNTER < MAX_DAL_LOOPS) and (Grad_error > Terminate_error):

		for h in root.handlers:
			#h.setLevel("WARNING");
			h.setLevel("INFO");

		##########################################################################
		# 1) Progress output
		##########################################################################	
		
		logger.info("\n\n --> DAL_LOOP: %i of %i with %i J(B_0) evals, %i dJ/dB(B_0) evals"%(DAL_LOOP_COUNTER,MAX_DAL_LOOPS,LS_COUNTER_f,LS_COUNTER_g) );

		logger.info("--> Grad_error   = %e "%Grad_error);
		logger.info("--> J(Bx0)_error = %e "%Cost_error);
		logger.info("--> Step-size's alpha_B, alpha_U = %e, %e \n\n"%(alpha_k[0],alpha_k[1]));

		##########################################################################	
		# 2) Adjoint solve + CG_descent
		##########################################################################

		logger.info(' \n\n B) ----> Adjoint_solve t = [T -> 0]: Compute Gradient')
		# Compute dJ(B_k) & dJ(U_k)
		if   LS_method == "Scipy_armijo":
			dJdB_k = ADJ_Solve(B_k, *Other_args);
			gfk = copy.deepcopy(dJdB_k); # Reqrd. for Armijo condition & Convergence estimate
		
		elif LS_method == "wolfe12":
			dJdB_k = dJdB_k_new;
			gfk = copy.deepcopy(dJdB_k); # Reqrd. for Armijo condition & Convergence estimate

		dJdB_km1  =    hg_k; # Using the prev tangent vector i.e. grad orthog to B_{k-1}, as this is normlaised we must do so with dJdB_k also
		LS_COUNTER_g += 1;	

		# Implement Conj-Grad method PR^+, verifying descent direction
		# Do so for each constraint separately to ensure consistency
		
		THETA_K = np.zeros(2)
		XK     = np.split(B_k,     2);
		dXK    = np.split(dJdB_k,  2);
		dXKm1  = np.split(dJdB_km1,2);
		EQs = Constraints;
		GRAD_IP_TANG = 0.; # Because we do this for both constraints
		for i in range(len(EQs)):

			# 1) Project the gradient(s) to be orthogonal to B_k
			# 1.A) Bx0 Vector
			IP_Xhatk_gk = Inner_Prod_3(domain, (-1.)*dXK[i],XK[i] )
			Norm_Xhatk  = np.sqrt( Inner_Prod_3(domain, XK[i] ,XK[i] ) );
			Norm_gk     = np.sqrt( Inner_Prod_3(domain, dXK[i],dXK[i]) );
			
			THETA_K[i] = math.acos( IP_Xhatk_gk/(Norm_gk*Norm_Xhatk) );

			C1 = Inner_Prod_3(domain, dXK[i],XK[i])/Inner_Prod_3(domain,XK[i],XK[i]);
			dXK[i]    =    dXK[i] -    XK[i]*C1;  GRAD_IP_TANG += Inner_Prod_3(domain,dXK[i],dXK[i]);

			C2 = np.sqrt(EQs[i]/Inner_Prod_3(domain,dXK[i],dXK[i]))
			dXK[i]    =    dXK[i]*C2;

			# 1.B) d(Bx0)/dr Vector
			C1_m1 = Inner_Prod_3(domain, dXKm1[i],XK[i])/Inner_Prod_3(domain,XK[i],XK[i]);		
			dXKm1[i]  =    dXKm1[i] -  XK[i]*C1_m1;

			C2_m2 = np.sqrt(EQs[i]/Inner_Prod_3(domain,dXKm1[i],dXKm1[i]))
			dXKm1[i]  =    dXKm1[i]*C2_m2;

			# Using the rescaled version of tangent gradient's as these are the decent directions we use
			beta_k = 0. #Inner_Prod_3(domain,dXK[i],dXK[i] - dXKm1[i])/Inner_Prod_3(domain,dXKm1[i],dXKm1[i]);
			
			# 1.C) Condition applied to ensure descent direction generated
			if (1. > beta_k > 0.):
				logger.info('Beta_k = %e',beta_k);
				dXK[i]    =    dXK[i] +    beta_k*dXKm1[i];

		# Re-combine gradient vector after applying
		dJdB_k = np.concatenate( (dXK[0],dXK[1]) ); 

		##########################################################################
		# 3) Fwd Solve + Line-search
		##########################################################################
		logger.info(' \n\n A) ----> Fwd_solve t = [0 -> T]: Line-search functional J_k + Fill Checkpoints');
		
		
		theta_k_progB.append(THETA_K[0]); theta_k_progU.append(THETA_K[1])		
		
		for con in range(len(Constraints)):
			
			OUT = line_search_armijo(FWD_Solve,con, Constraints,	B_k,dJdB_k,gfk,		J_k_objective[-1],args=(Other_args),alpha0=alpha_k[con]);
			alpha_k[con], J_k  = OUT[0],OUT[2]; 
			LS_COUNTER_f += OUT[1];
			logger.info('--> J(Bx0)_evals %i, alpha_k=%e'%(OUT[1],alpha_k[con]))

		'''
		alpha_k = np.ones(2)*(np.pi/2.0) - 1e-5;
		for con in range(len(Constraints)):
			
			while np.tan(alpha_k[con]/2.) > np.tan(THETA_K[con]):
				# reduce the step-size to satisfy the constraint
				alpha_k[con] = 0.95*alpha_k[con];
				
				if alpha_k[con] < Terminate_error:
					logger.info("No step size gives descent for con = %i !!!Terminating!!! "%con)
					alpha_k[con] = 0.
					break

			if alpha_k[con] != 0.:	
				logger.info("alpha_k[con] = %e "%alpha_k[con])
				OUT = line_search_armijo(FWD_Solve,con, Constraints,	B_k,dJdB_k,gfk,		J_k_objective[-1],args=(Other_args),alpha0=alpha_k[con]);
				alpha_k[con], J_k  = OUT[0],OUT[2]; 
				LS_COUNTER_f += OUT[1];
				logger.info('--> J(Bx0)_evals %i, alpha_k=%e'%(OUT[1],alpha_k[con]))
			'''

		##########################################################################	
		# 4) Update parameter vector B_k, such that ||B_k+1||_2 =M_0^(1/2)
		##########################################################################
		XK  = np.split(B_k,2);
		PK  = np.split(dJdB_k,2);
		EQs = Constraints;
		for i in range(len(EQs)):
			XK[i],PK[i] = Norm_constraint(domain, XK[i],PK[i], alpha_k[i], EQs[i]);
			'''
			if (THETA_K[i] < .5*np.pi):

				if (np.tan(alpha_k[i]/2.) < np.tan(THETA_K[i])):
					logger.info('Constraint i=%i     updated as tan(alpha/2) = %e  < tan(theta_k) = %e'%(i,np.tan(alpha_k[i]/2.),np.tan(THETA_K[i])) )
					XK[i],PK[i] = Norm_constraint(domain, XK[i],PK[i], alpha_k[i], EQs[i]);
				else:
					logger.info('Constraint i=%i not updated as tan(alpha/2) = %e !< tan(theta_k) = %e'%(i,np.tan(alpha_k[i]/2.),np.tan(THETA_K[i])) )
					##XK[i],PK[i] = Norm_constraint(domain, XK[i],PK[i], 0., EQs[i]);	
				
			else:
				logger.info('Constraint i=%i not updated as theta_k > pi/2 = %e'%(i,THETA_K[i]) )
				##XK[i],PK[i] = Norm_constraint(domain, XK[i],PK[i], 0., EQs[i]);
			'''
		
		B_k  = np.concatenate( (XK[0],XK[1]) );
		hg_k = np.concatenate( (PK[0],PK[1]) ); 

		# 5) Compute J(B_k,U_k)
		J_k_old = FWD_Solve(B_k, *Other_args);
		J_k_objective.append(J_k_old);
		LS_COUNTER_f += 1;
		logger.info('FWD solve with both alpha_k J(X) = %e \n\n'%J_k_old);
		
		##########################################################################		
		# 6) Compute errors & update counters
		##########################################################################	
		GRAD_IP_FULL = Inner_Prod(domain,gfk,gfk);
		Grad_error = np.sqrt(GRAD_IP_TANG/GRAD_IP_FULL);
		Cost_error = abs(J_k - J_k_old)/abs(J_k);
		
		File_Manips(DAL_LOOP_COUNTER);

		#J_k_objective.append(J_k); # adds elements to the end = right hand side
		J_k_error.append(Cost_error);
		dJ_k_error.append(Grad_error);

		##########################################################################
		# 7) Update all vars
		##########################################################################
		J_k_old = J_k;
		dJdB_k_old = dJdB_k;
		alpha_k_old = alpha_k;

		DAL_LOOP_COUNTER+=1;

		if MPI.COMM_WORLD.rank == 0:

			# Plot out the progress :)
			#Plot_MAX_DivUB('FWD_Solve_IVP_DIV_UB.h5',DAL_LOOP_COUNTER-1);	

			# Save the different errors	
			DAL_file = h5py.File('DAL_PROGRESS.h5', 'w')
			
			# Problem Params
			DAL_file['J_k'] = J_k_objective;
			DAL_file['J_k_error'] = J_k_error;
			DAL_file['dJ_k_error'] = dJ_k_error;

			DAL_file['theta_k_U'] = theta_k_progU;
			DAL_file['theta_k_B'] = theta_k_progB;

			# Save the last B_k to allow restarts
			DAL_file['B_k'] = B_k;
			
			DAL_file.close();
			
			# Plot out the progress :)
			if DAL_LOOP_COUNTER >= 2:
				Plot_DAL_Progress('DAL_PROGRESS.h5');

	return None;

#'''

################################################################################################################
# Perform Optimisation using Direct Adjoint Looping (DAL) 
# !! Yet to be re-written + commented !!
################################################################################################################
def DAL_LOOP_OLD_VER(X0, Constraints, *Other_args):

	# !! Yet to be commented !!
	from Normed_LineSearch import Norm_constraint,line_search_wolfe2
	from Normed_LineSearch import line_search_armijo_old_ver as line_search_armijo

	# Self-Written Functions
	from FWD_Solve_PBox_IND_MHD import FWD_Solve_IVP_Lin as FWD_Solve
	from FWD_Solve_PBox_IND_MHD import ADJ_Solve_IVP_Lin as ADJ_Solve
	from FWD_Solve_PBox_IND_MHD import Inner_Prod, Inner_Prod_3

	import copy, math

	# Set to info level rather than the debug default
	root = logging.root
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");
		#h.setLevel("DEBUG")
	logger = logging.getLogger(__name__)

	# 1) Optimiser - defaults to armijo for first iteration is this critical??
	LS_method = "Scipy_armijo"; # Uses Quad/cubic minizer determining a good step-length starting from previous
	#LS_method = "wolfe12"; # Requires J(B_0),dJ(B_0) => at least 3 x armijo cost 

	# 2) Termination Codition Parameters
	Terminate_error = 1e-06;
	Grad_error = 1.; Cost_error = 1.

	MAX_DAL_LOOPS = 20 + 2;
	DAL_LOOP_COUNTER = 0;
	LS_COUNTER_f = 0;
	LS_COUNTER_g = 0;

	alpha_k = (np.pi/2.0) - 1e-5;

	# 3) Initialise line-search/gradient-descent
	B_k = X0; # Depends on optimisation
	
	domain = Other_args[0];
	'''
	# Check ||Bx0|| = M_0 
	if np.round(Inner_Prod(domain,B_k,B_k),12) == np.round(M_0,12):
		
		logger.info(      'Verified <B_0,B_0> = %e  = M_0 = %e'%( Inner_Prod(domain,B_k,B_k) ,M_0) );
	else:
		logger.info('Not verifiable <B_0,B_0> = %e != M_0 = %e'%( Inner_Prod(domain,B_k,B_k) ,M_0) );
		sys.exit();
	'''
	dJdB_k = np.zeros(X0.shape);
	J_k = 0.;

	dJdB_k_old = np.zeros(X0.shape);
	alpha_k_old = alpha_k;
	J_k_old = 0.;

	# Buffers to store progress of gradient optimisation
	J_k_objective = []; J_k_error = []; dJ_k_error = []; theta_k_progB = []; theta_k_progU = [];

	# Grad-Descent Run
	# ~~~~~~~~~~~~~~~~~~~~` # ~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~
	logger.info(' \n\n A) ----> Fwd_solve t = [0 -> T] + Gradient Decent')	
	
	# Compute J(B_k,U_k)
	J_k_old = FWD_Solve(B_k, *Other_args);
	J_k_objective.append(J_k_old);

	# Compute dJ(B_k) & dJ(U_k)
	dJdB_k = ADJ_Solve(B_k,	*Other_args);
	gfk    = copy.deepcopy(dJdB_k); # Reqd. for Armijo condition

	File_Manips(DAL_LOOP_COUNTER);  #Include to check first noisy IC
	DAL_LOOP_COUNTER+=1;
	
	# need to update this such that dJdB_k is the tangent vector otherwise mistake!!!!
	THETA_K = np.zeros(2)
	XK     = np.split(B_k,     2);
	dXK    = np.split(dJdB_k,  2);
	for i in range(len(Constraints)):

		# 1) Project the gradient(s) to be orthogonal to B_k
		# 1.A) Bx0 Vector
		IP_Xhatk_gk = Inner_Prod_3(domain, (-1.)*dXK[i],XK[i] )
		Norm_Xhatk  = np.sqrt( Inner_Prod_3(domain, XK[i] ,XK[i] ) );
		Norm_gk     = np.sqrt( Inner_Prod_3(domain, dXK[i],dXK[i]) );
		
		THETA_K[i] = math.acos( IP_Xhatk_gk/(Norm_gk*Norm_Xhatk) );

		C1 = Inner_Prod_3(domain, dXK[i],XK[i])/Inner_Prod_3(domain,XK[i],XK[i]);
		dXK[i]    =    dXK[i] -    XK[i]*C1; 

		C2 = np.sqrt(Constraints[i]/Inner_Prod_3(domain,dXK[i],dXK[i]))
		dXK[i]    =    dXK[i]*C2;

	# Re-combine gradient vector after applying
	dJdB_k = np.concatenate( (dXK[0],dXK[1]) ); 

	# ~~~~~~~~~~ # ~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~
	theta_k_progB.append(THETA_K[0]);
	theta_k_progU.append(THETA_K[1]);		


	while np.tan(alpha_k/2.) > np.tan(THETA_K[0]):
		# reduce the step-size to satisfy the constraint
		alpha_k = 0.95*alpha_k;

		if alpha_k < Terminate_error:
			sys.exit();	

	# Check Step for alpha
	if 	 LS_method == "Scipy_armijo":
	
		OUT = line_search_armijo(FWD_Solve,Constraints,		B_k,dJdB_k,gfk,		J_k_objective[-1],args=(Other_args),alpha0=alpha_k);
		alpha_k, J_k  = OUT[0],OUT[2]; 
		LS_COUNTER_f += OUT[1];
		LS_COUNTER_g += 1;
		logger.info('--> J(Bx0)_evals %i, alpha_k=%e'%(OUT[1],alpha_k))
		J_k_objective.append(OUT[2]); # Add new J(X)

	elif LS_method == "wolfe12":
	
		OUT = line_search_wolfe2(FWD_Solve, ADJ_Solve, Constraints, B_k, dJdB_k, gfk=gfk,old_fval=J_k_objective[-1],old_old_fval=None,args=(Other_args),amax=alpha_k)
		
		alpha_k, J_k  = OUT[0],OUT[3]; 
		LS_COUNTER_f += OUT[1];
		LS_COUNTER_g += OUT[2];

		logger.info('--> J(Bx0)_evals     %i, alpha_k=%e'%(OUT[1],alpha_k))
		logger.info('--> dJ/dB(Bx0)_evals %i'%OUT[2]);
		
		dJdB_k_new = OUT[-1]; # Grab gradient evaluated at the new step
		J_k_objective.append(J_k); # Add new J(X)

	File_Manips(DAL_LOOP_COUNTER); #Include to check first update from gradient
	DAL_LOOP_COUNTER+=1;

	# Update B_k, returns tangent hg_k but not the one on the unit hypersphere 
	XK  = np.split(B_k,   2);
	PK  = np.split(dJdB_k,2);
	EQs = Constraints;
	for i in range(len(EQs)):
		
		XK[i],PK[i] = Norm_constraint(domain, XK[i],PK[i], alpha_k, EQs[i]);
		'''
		if (THETA_K[i] < .5*np.pi):

			if (np.tan(alpha_k/2.) < np.tan(THETA_K[i])):
				logger.info('Constraint i=%i     updated as tan(alpha/2) = %e  < tan(theta_k) = %e'%(i,np.tan(alpha_k/2.),np.tan(THETA_K[i])) )
				XK[i],PK[i] = Norm_constraint(domain, XK[i],PK[i], alpha_k, EQs[i]);
			else:
				logger.info('Constraint i=%i not updated as tan(alpha/2) = %e !< tan(theta_k) = %e'%(i,np.tan(alpha_k/2.),np.tan(THETA_K[i])) )
				XK[i],PK[i] = Norm_constraint(domain, XK[i],PK[i],      0., EQs[i]);	
			
		else:
			logger.info('Constraint i=%i not updated as theta_k > pi/2 = %e'%(i,THETA_K[i]) )
			XK[i],PK[i] = Norm_constraint(domain, XK[i],PK[i], 0., EQs[i]);
		'''
		#XK[i],PK[i] = Norm_constraint(domain, XK[i],PK[i], 0., EQs[i]);	
	B_k  = np.concatenate( (XK[0],XK[1]) );
	hg_k = (-1.)*np.concatenate( (PK[0],PK[1]) ); # ?? Tangent vector - but not on the hyper-sphere

	# ~~~~~~~~~~~~~~~~~~~~` # ~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~

	while (DAL_LOOP_COUNTER < MAX_DAL_LOOPS) and (Grad_error > Terminate_error):

		for h in root.handlers:
			#h.setLevel("WARNING");
			h.setLevel("INFO");

		##########################################################################
		# 1) Progress output
		##########################################################################	
		
		logger.info("\n\n --> DAL_LOOP: %i of %i with %i J(B_0) evals, %i dJ/dB(B_0) evals"%(DAL_LOOP_COUNTER,MAX_DAL_LOOPS,LS_COUNTER_f,LS_COUNTER_g) );

		logger.info("--> Grad_error = %e "%Grad_error);
		logger.info("--> J(Bx0)_error = %e "%Cost_error);
		logger.info("--> Step-size alpha = %e \n\n"%alpha_k);

		##########################################################################	
		# 2) Adjoint solve + CG_descent
		##########################################################################

		logger.info(' \n\n B) ----> Adjoint_solve t = [T -> 0]: Compute Gradient')
		# Compute dJ(B_k) & dJ(U_k)
		if   LS_method == "Scipy_armijo":
			dJdB_k = ADJ_Solve(B_k, *Other_args);
			gfk = copy.deepcopy(dJdB_k); # Reqrd. for Armijo condition & Convergence estimate
		
		elif LS_method == "wolfe12":
			dJdB_k = dJdB_k_new;
			gfk = copy.deepcopy(dJdB_k); # Reqrd. for Armijo condition & Convergence estimate

		dJdB_km1  =    hg_k; # Using the prev tangent vector i.e. grad orthog to B_{k-1}, as this is normlaised we must do so with dJdB_k also

		# Implement Conj-Grad method PR^+, verifying descent direction
		# Do so for each constraint separately to ensure consistency
		
		THETA_K = np.zeros(2)
		XK     = np.split(B_k,     2);
		dXK    = np.split(dJdB_k,  2);
		dXKm1  = np.split(dJdB_km1,2);
		EQs = Constraints;
		GRAD_IP_TANG = 0.; # Because we do this for both constraints
		for i in range(len(EQs)):

			# 1) Project the gradient(s) to be orthogonal to B_k
			# 1.A) Bx0 Vector
			IP_Xhatk_gk = Inner_Prod_3(domain, (-1.)*dXK[i],XK[i] )
			Norm_Xhatk  = np.sqrt( Inner_Prod_3(domain, XK[i] ,XK[i] ) );
			Norm_gk     = np.sqrt( Inner_Prod_3(domain, dXK[i],dXK[i]) );
			
			THETA_K[i] = math.acos( IP_Xhatk_gk/(Norm_gk*Norm_Xhatk) );

			C1 = Inner_Prod_3(domain, dXK[i],XK[i])/Inner_Prod_3(domain,XK[i],XK[i]);
			dXK[i]    =    dXK[i] -    XK[i]*C1;  GRAD_IP_TANG += Inner_Prod_3(domain,dXK[i],dXK[i]);

			C2 = np.sqrt(EQs[i]/Inner_Prod_3(domain,dXK[i],dXK[i]))
			dXK[i]    =    dXK[i]*C2;

			# 1.B) d(Bx0)/dr Vector
			C1_m1 = Inner_Prod_3(domain, dXKm1[i],XK[i])/Inner_Prod_3(domain,XK[i],XK[i]);		
			dXKm1[i]  =    dXKm1[i] -  XK[i]*C1_m1;

			C2_m2 = np.sqrt(EQs[i]/Inner_Prod_3(domain,dXKm1[i],dXKm1[i]))
			dXKm1[i]  =    dXKm1[i]*C2_m2;

			# Using the rescaled version of tangent gradient's as these are the decent directions we use
			beta_k = 0. #Inner_Prod_3(domain,dXK[i],dXK[i] - dXKm1[i])/Inner_Prod_3(domain,dXKm1[i],dXKm1[i]);
			
			# 1.C) Condition applied to ensure descent direction generated
			if (1. > beta_k > 0.):
				logger.info('Beta_k = %e',beta_k);
				dXK[i]    =    dXK[i] +    beta_k*dXKm1[i];

		# Re-combine gradient vector after applying
		dJdB_k = np.concatenate( (dXK[0],dXK[1]) ); 

		##########################################################################
		# 3) Fwd Solve + Line-search
		##########################################################################
		logger.info(' \n\n A) ----> Fwd_solve t = [0 -> T]: Line-search functional J_k + Fill Checkpoints');
		
		#'''
		theta_k_progB.append(THETA_K[0]);
		theta_k_progU.append(THETA_K[1])		
		
		while np.tan(alpha_k/2.) > np.tan(THETA_K[0]):
			# reduce the step-size to satisfy the constraint
			alpha_k = 0.95*alpha_k;

			if alpha_k < Terminate_error:
				sys.exit();
		#'''

		# Check Step for alpha
		if 	 LS_method == "Scipy_armijo":

			OUT = line_search_armijo(FWD_Solve,Constraints,		B_k,dJdB_k,gfk,		J_k_objective[-1],args=(Other_args),alpha0=alpha_k);
			alpha_k, J_k  = OUT[0],OUT[2]; 
			LS_COUNTER_f += OUT[1];
			LS_COUNTER_g += 1;
			logger.info('--> J(Bx0)_evals %i, alpha_k=%e'%(OUT[1],alpha_k));
			J_k_objective.append(J_k); # Add new J(X)

		elif LS_method == "wolfe12":

			OUT = line_search_wolfe2(FWD_Solve, ADJ_Solve, Constraints, B_k, dJdB_k, gfk=gfk,old_fval=J_k_objective[-1], old_old_fval=J_k_objective[-2],args=(Other_args),amax=alpha_k)
			alpha_k, J_k  = OUT[0],OUT[3]; 
			LS_COUNTER_f += OUT[1];
			LS_COUNTER_g += OUT[2];

			logger.info('--> J(Bx0)_evals     %i, alpha_k=%e'%(OUT[1],alpha_k))
			logger.info('--> dJ/dB(Bx0)_evals %i'%OUT[2]);

			dJdB_k_new = OUT[-1]; # Grab gradient evaluated at the new step
			J_k_objective.append(J_k); # Add new J(X)
		
		if alpha_k == None:
			logger.warning('Line_search did not converge!! Terminating .... ');
			sys.exit();


		##########################################################################	
		# 4) Update parameter vector B_k, such that ||B_k+1||_2 =M_0^(1/2)
		##########################################################################
		XK  = np.split(B_k,2);
		PK  = np.split(dJdB_k,2);
		EQs = Constraints;
		for i in range(len(EQs)):
		
			if (THETA_K[i] < .5*np.pi):

				if (np.tan(alpha_k/2.) < np.tan(THETA_K[i])):
					logger.info('Constraint i=%i     updated as tan(alpha/2) = %e  < tan(theta_k) = %e'%(i,np.tan(alpha_k/2.),np.tan(THETA_K[i])) )
					XK[i],PK[i] = Norm_constraint(domain, XK[i],PK[i], alpha_k, EQs[i]);
				else:
					logger.info('Constraint i=%i not updated as tan(alpha/2) = %e !< tan(theta_k) = %e'%(i,np.tan(alpha_k/2.),np.tan(THETA_K[i])) )
					XK[i],PK[i] = Norm_constraint(domain, XK[i],PK[i], 0., EQs[i]);	
				
			else:
				logger.info('Constraint i=%i not updated as theta_k > pi/2 = %e'%(i,THETA_K[i]) )
				XK[i],PK[i] = Norm_constraint(domain, XK[i],PK[i], 0., EQs[i]);

		B_k  = np.concatenate( (XK[0],XK[1]) );
		hg_k = (-1.0)*np.concatenate( (PK[0],PK[1]) ); 

		##########################################################################		
		# 6) Compute errors & update counters
		##########################################################################	
		GRAD_IP_FULL = Inner_Prod(domain,gfk,gfk);
		Grad_error = np.sqrt(GRAD_IP_TANG/GRAD_IP_FULL);
		Cost_error = abs(J_k - J_k_old)/abs(J_k);
		
		File_Manips(DAL_LOOP_COUNTER);

		#J_k_objective.append(J_k); # adds elements to the end = right hand side
		J_k_error.append(Cost_error);
		dJ_k_error.append(Grad_error);

		##########################################################################
		# 7) Update all vars
		##########################################################################
		J_k_old = J_k;
		dJdB_k_old = dJdB_k;
		alpha_k_old = alpha_k;

		DAL_LOOP_COUNTER+=1;

		if MPI.COMM_WORLD.rank == 0:

			# Plot out the progress :)
			#Plot_MAX_DivUB('FWD_Solve_IVP_DIV_UB.h5',DAL_LOOP_COUNTER-1);	

			# Save the different errors	
			DAL_file = h5py.File('DAL_PROGRESS.h5', 'w')
			
			# Problem Params
			DAL_file['J_k'] = J_k_objective;
			DAL_file['J_k_error'] = J_k_error;
			DAL_file['dJ_k_error'] = dJ_k_error;

			DAL_file['theta_k_U'] = theta_k_progU;
			DAL_file['theta_k_B'] = theta_k_progB;

			# Save the last B_k to allow restarts
			DAL_file['B_k'] = B_k;
			
			DAL_file.close();
			
			# Plot out the progress :)
			if DAL_LOOP_COUNTER >= 2:
				Plot_DAL_Progress('DAL_PROGRESS.h5');

	return None;



def File_Manips(k):

	"""
	Takes files generated by the adjoint loop solve, orgainises them into respective directories
	and removes unwanted data

	Each iteration of the DAL loop code is stored under index k
	"""		

	import shutil
	# Copy bits into it renaming according to iteration

	# Shouldn't need to be done by all processes!

	# A) Contains all scalar data
	#Scalar  = "".join([Local_dir,'scalar_data_iter_%i.h5'%k])
	shutil.copyfile('scalar_data/scalar_data_s1.h5','scalar_data_iter_%i.h5'%k);

	#shutil.copyfile('Adjoint_scalar_data/Adjoint_scalar_data_s1.h5','Adjoint_scalar_data_iter_%i.h5'%k);
	# B) Contains: Angular Momentum Profile, E(K,m), first and last full system state
	#
	#Radial  = "".join([Local_dir,'radial_profiles_iter_%i.h5'%k])
	shutil.copyfile('CheckPoints/CheckPoints_s1.h5','CheckPoints_iter_%i.h5'%k);
	##MPI.COMM_WORLD.Barrier();

	#time.sleep(2); # Allow files to be copied

	# 3) Remove Surplus files at the end
	#shutil.rmtree('snapshots');
	#shutil.rmtree('Adjoint_CheckPoints');
	#shutil.rmtree('Checkpoints');

	return None;		

def GEN_BUFFER(Nx, Ny, Nz, domain, N_SUB_ITERS, WRITE_KIND, Rm):

	"""
	Given the resolution and number of N_SUB_ITERS

	# 1) Estimate the memory usage per proc/core

	# 2) If possible create a memory buffer as follows:

	# The buffer created is a Python dictionary - X_FWD_DICT = {'u(x,t)','v(x,t)','w(x,t)', 'b_x(x,t)','b_y(x,t)','b_z(x,t)'}
	# Chosen so as to allow pass by reference & thus avoids copying
	# All arrays to which these "keys" map must differ as we are otherwise mapping to the same memory!!
	
	Returns:
	Memory Buffer - Type Dict { 'Key':Value } where Value is 4D np.array

	"""

	WRITE_SCALE = 1;

	################################################################################################################
	# A) Choose rescaling factors for the objective function
	################################################################################################################

	# 3) Objective function Inner product weights
	IP_weights = np.ones(3,dtype=float); 
	# [0,1,2] the First three elements are A_11,A_22,A_33 of the diagonal weight matrix
	
	# Steady-rescaling
	IP_weights[0] = 1.0/Rm;  #B_tot/B^2_x; # Experiences the Omega-effect most
	IP_weights[1] = 1.0 #B_tot/B^2_y;
	IP_weights[2] = 1.0;#B_tot/B^2_z;	
	
	################################################################################################################
	# B) Build memory buffer
	################################################################################################################
	if WRITE_KIND == 'c':
		
		# -Total  = (0.5 complex to real)*Nx*Ny*Nz*(6 fields)*(64 bits)*N_SUB_ITERS*(1.25e-10)/MPI.COMM_WORLD.Get_size()
		# -float64 = 64bits # Data-type used # -1 bit = 1.25e-10 GB
		Total  = ( ((Nx/2)*Ny*Nz*6)*64*N_SUB_ITERS*(1.25e-10) )/float( MPI.COMM_WORLD.Get_size() )
		if MPI.COMM_WORLD.rank == 0:
			print("Total memory =%f GB, and memory/core = %f GB"%(MPI.COMM_WORLD.Get_size()*Total,Total));

		gshape = tuple( domain.dist.coeff_layout.global_shape(scales=WRITE_SCALE) );
		lcshape = tuple( domain.dist.coeff_layout.local_shape(scales=WRITE_SCALE) );
		SNAPS_SHAPE = (lcshape[0],lcshape[1],lcshape[2],N_SUB_ITERS+1);
		
		U_SNAPS = np.zeros(SNAPS_SHAPE,dtype=complex);
		V_SNAPS = np.zeros(SNAPS_SHAPE,dtype=complex); 
		W_SNAPS = np.zeros(SNAPS_SHAPE,dtype=complex);

		A_SNAPS = np.zeros(SNAPS_SHAPE,dtype=complex);
		B_SNAPS = np.zeros(SNAPS_SHAPE,dtype=complex);
		C_SNAPS = np.zeros(SNAPS_SHAPE,dtype=complex);

		#X_FWD_DICT = {'u_fwd':U_SNAPS,'v_fwd':V_SNAPS,'w_fwd':W_SNAPS,	'A_fwd':A_SNAPS,'B_fwd':B_SNAPS,'C_fwd':C_SNAPS, 'IP_weights':IP_weights};

	elif WRITE_KIND == 'g':
		
		Total  = ( (Nx*Ny*Nz*6)*64*N_SUB_ITERS*(1.25e-10) )/float( MPI.COMM_WORLD.Get_size() )
		if MPI.COMM_WORLD.rank == 0:
			print("Total memory =%f GB, and memory/core = %f GB"%(MPI.COMM_WORLD.Get_size()*Total,Total));

		gshape = tuple( domain.dist.grid_layout.global_shape(scales=WRITE_SCALE) );
		lgshape = tuple( domain.dist.grid_layout.local_shape(scales=WRITE_SCALE) );
		SNAPS_SHAPE = (lgshape[0],lgshape[1],lgshape[2],N_SUB_ITERS + 1);
		
		U_SNAPS = np.zeros(SNAPS_SHAPE,dtype=float);
		V_SNAPS = np.zeros(SNAPS_SHAPE,dtype=float);
		W_SNAPS = np.zeros(SNAPS_SHAPE,dtype=float);

		A_SNAPS = np.zeros(SNAPS_SHAPE,dtype=float);
		B_SNAPS = np.zeros(SNAPS_SHAPE,dtype=float);
		C_SNAPS = np.zeros(SNAPS_SHAPE,dtype=float);

	return {'u_fwd':U_SNAPS,'v_fwd':V_SNAPS,'w_fwd':W_SNAPS,	'A_fwd':A_SNAPS,'B_fwd':B_SNAPS,'C_fwd':C_SNAPS,'IP_weights':IP_weights};

def Plot_DAL_Progress(filename):



	"""
	Plot the progress of the Direct Adjoint Looping (DAL) routine in-terms of....
	J_k 	   - the objective function evaluation of the k^th iteration
	J_k_error  - the relative error between evaluations
	dJ_k_error - the L2 norm of the tangent gradient/full gradient 

	Input Parameters:

	filename - hdf5 file with dict 'key':value structure, all 'keys' are defined as above

	Returns: None

	"""

	import matplotlib.pyplot as plt

	outfile = 'J_Objective.pdf'; 

	dpi = 400
	fig, ax1 = plt.subplots(figsize=(8,6)); # 1,2,

	# Grab Data
	DAL_file = h5py.File(filename,"r")
	J_k = (-1.)*DAL_file['J_k'][()];
	J_k_error = DAL_file['J_k_error'][()];
	dJ_k_error = DAL_file['dJ_k_error'][()];
	theta_k_B = DAL_file['theta_k_B'][()];
	theta_k_U = DAL_file['theta_k_U'][()];
	DAL_file.close();

	# Plot figures
	color = 'tab:red'
	x = np.arange(0,len(J_k),1)
	ax1.plot(x,J_k,color=color, linestyle=':',linewidth=1.5, markersize=3);
	#ax1.set_title(r'Objective function');
	ax1.tick_params(axis='y', labelcolor=color)
	ax1.set_ylabel(r'Objective function $J_k$',color=color,fontsize=18)
	ax1.set_xlabel(r'Iteration $k$',fontsize=18)

	ax1.set_xlim([0,np.max(x)])

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	color = 'tab:blue'
	x = np.arange(0,len(dJ_k_error),1)
	ax2.plot(x,np.log10(dJ_k_error),color=color, linestyle='-',linewidth=1.5, markersize=3);
	#ax2.set_title(r'Gradient residual error')
	ax2.tick_params(axis='y', labelcolor=color)
	ax2.set_ylabel(r'Gradient residual error $log10(r)$',color=color,fontsize=18)
	#ax2.set_xlabel(r'Iteration $k$',fontsize=15)

	'''
	x = np.arange(0,len(J_k_error),1)
	ax3.plot(x[1:],np.log10(J_k_error[1:]),marker='o', linestyle='dashed',linewidth=1.5, markersize=3);
	ax3.set_title(r'Cost-function residual error')
	ax3.set_ylabel(r'$log10( ||J_k - J_{k-1}||/||J_k|| )$')
	ax3.set_xlabel(r'Iteration $k$',fontsize=15)
	'''
	plt.grid()
	plt.tight_layout(pad=1, w_pad=1.5)
	fig.savefig(outfile, dpi=dpi);
	#plt.show();


	outfile = 'Angle_theta_k.pdf'; 

	dpi = 1200; fig = plt.figure(figsize=(8,6));
	
	#if theta_k_U !=None:
	x = np.arange(0,len(theta_k_U),1);
	plt.semilogy(x,theta_k_U,'r:',label=r"$\theta_k^U$")
	

	x = np.arange(0,len(theta_k_B),1);
	plt.semilogy(x,theta_k_B,'b:',label=r"$\theta_k^B$")
	import math
	plt.semilogy(x,0.5*math.pi*np.ones(len(x)),'k-')
	plt.xlim([0,max(x)]);
	plt.xlabel(r"Iterations",fontsize=18);
	plt.ylabel(r"Angle $\theta_k rads < \pi/2$",fontsize=18);
	plt.grid()
	plt.tight_layout(pad=1, w_pad=1.5)
	fig.savefig(outfile, dpi=dpi);

	return None;
