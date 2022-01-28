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
def DAL_LOOP(X0, Constraints, *Other_args):

	# !! Yet to be commented !!
	from Normed_LineSearch import Norm_constraint, line_search_armijo
	#from Normed_LineSearch import Inner_Prod

	# Self-Written Functions
	from FWD_Solve_PBox_IND_MHD import FWD_Solve_IVP_Lin as FWD_Solve
	from FWD_Solve_PBox_IND_MHD import ADJ_Solve_IVP_Lin as ADJ_Solve
	from FWD_Solve_PBox_IND_MHD import Inner_Prod, Inner_Prod_3

	import copy

	# Set to info level rather than the debug default
	root = logging.root
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");
		#h.setLevel("DEBUG")
	logger = logging.getLogger(__name__)

	# 1) Optimiser - defaults to armijo for first iteration is this critical??
	Line_search_method = "Scipy_armijo"; # Uses Quad/cubic minizer determining a good step-length starting from previous
	#Line_search_method = "wolfe12"; # Requires J(B_0),dJ(B_0) => at least 3 x armijo cost 

	# 2) Termination Codition Parameters
	Terminate_error = 1e-06;
	Grad_error = 1.; Cost_error = 1.

	MAX_DAL_LOOPS = 30 + 2;
	DAL_LOOP_COUNTER = 0;
	LS_COUNTER = 0;

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
	alpha_k = np.pi/4.0; # Initial step tries equal proportion of soln B_k & Grad dJdk
	J_k = 0.;

	dJdB_k_old = np.zeros(X0.shape);
	alpha_k_old = alpha_k;
	J_k_old = 0.;

	# Buffers to store progress of gradient optimisation
	J_k_objective = []; J_k_error = []; dJ_k_error = [];

	# Grad-Descent Run
	# ~~~~~~~~~~~~~~~~~~~~` # ~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~
	logger.info(' \n\n A) ----> Fwd_solve t = [0 -> T] + Gradient Decent')	
	
	# Compute J(B_k,U_k)
	J_k_old = FWD_Solve(B_k, *Other_args);
	J_k_objective.append(J_k_old);

	File_Manips(DAL_LOOP_COUNTER);  #Include to check first noisy IC
	DAL_LOOP_COUNTER+=1;

	# Compute dJ(B_k) & dJ(U_k)
	dJdB_k = ADJ_Solve(B_k,	*Other_args);
	gfk    = copy.deepcopy(dJdB_k); # Reqd. for Armijo condition

	# Check Step for alpha
	OUT = line_search_armijo(FWD_Solve,Constraints,		B_k,dJdB_k,gfk,		J_k_objective[-1],args=(Other_args),alpha0=alpha_k);
	alpha_k, J_k = OUT[0],OUT[2]; LS_COUNTER += OUT[1];
	logger.info('--> J(Bx0)_evals %i, alpha_k=%e'%(OUT[1],alpha_k))

	File_Manips(DAL_LOOP_COUNTER); #Include to check first update from gradient
	DAL_LOOP_COUNTER+=1;

	# Update B_k, returns tangent hg_k but not the one on the unit hypersphere 
	XK  = np.split(B_k,2);
	PK  = np.split(dJdB_k,2);
	EQs = Constraints;
	for i in range(len(EQs)):
		XK[i],PK[i] = Norm_constraint(domain, XK[i],PK[i], alpha_k, EQs[i]);

	B_k  = np.concatenate( (XK[0],XK[1]) );
	hg_k = np.concatenate( (PK[0],PK[1]) ); # Tangent vector - but not on the hyper-sphere

	# ~~~~~~~~~~~~~~~~~~~~` # ~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~

	while (DAL_LOOP_COUNTER < MAX_DAL_LOOPS) and (Grad_error > Terminate_error):

		for h in root.handlers:
			#h.setLevel("WARNING");
			h.setLevel("INFO");

		##########################################################################
		# 1) Progress output
		##########################################################################	
		
		logger.info("\n\n --> DAL_LOOP: %i of %i with %i J(B_0) evals"%(DAL_LOOP_COUNTER,MAX_DAL_LOOPS,LS_COUNTER) );

		logger.info("--> Grad_error = %e "%Grad_error);
		logger.info("--> J(Bx0)_error = %e "%Cost_error);
		logger.info("--> Step-size alpha = %e \n\n"%alpha_k);

		##########################################################################	
		# 2) Adjoint solve + CG_descent
		##########################################################################

		logger.info(' \n\n B) ----> Adjoint_solve t = [T -> 0]: Compute Gradient')
		# Compute dJ(B_k) & dJ(U_k)
		dJdB_k = ADJ_Solve(B_k, *Other_args);
		gfk = copy.deepcopy(dJdB_k); # Reqrd. for Armijo condition & Convergence estimate

		dJdB_km1  =    hg_k; # Using the prev tangent vector i.e. grad orthog to B_{k-1}, as this is normlaised we must do so with dJdB_k also

		# Implement Conj-Grad method PR^+, verifying descent direction
		# Do so for each constraint separately to ensure consistency
		
		XK     = np.split(B_k,     2);
		dXK    = np.split(dJdB_k,  2);
		dXKm1  = np.split(dJdB_km1,2);
		EQs = Constraints;
		GRAD_IP_TANG = 0.;
		for i in range(len(EQs)):

			# 1) Project the gradient(s) to be orthogonal to B_k
			# 1.A) Bx0 Vector
			C1 = Inner_Prod_3(domain, dXK[i],XK[i])/Inner_Prod_3(domain,XK[i],XK[i]);
			dXK[i]    =    dXK[i] -    XK[i]*C1; 
			GRAD_IP_TANG += Inner_Prod_3(domain,dXK[i],dXK[i]);

			C2 = np.sqrt(EQs[i]/Inner_Prod_3(domain,dXK[i],dXK[i]))
			dXK[i]    =    dXK[i]*C2;

			# 1.B) d(Bx0)/dr Vector
			C1_m1 = Inner_Prod_3(domain, dXKm1[i],XK[i])/Inner_Prod_3(domain,XK[i],XK[i]);		
			dXKm1[i]  =    dXKm1[i] -  XK[i]*C1_m1;

			C2_m2 = np.sqrt(EQs[i]/Inner_Prod_3(domain,dXKm1[i],dXKm1[i]))
			dXKm1[i]  =    dXKm1[i]*C2_m2;

			# Using the rescaled version of tangent gradient's as these are the decent directions we use
			beta_k = Inner_Prod_3(domain,dXK[i],dXK[i] - dXKm1[i])/Inner_Prod_3(domain,dXKm1[i],dXKm1[i]);
			
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
			
		#Returns: alpha_k,f_evals,J_k
		OUT = line_search_armijo(FWD_Solve,Constraints, 	B_k,dJdB_k,gfk,		J_k_objective[-1],args=(Other_args),c1=1e-04,alpha0=alpha_k);
		alpha_k, J_k = OUT[0],OUT[2]; LS_COUNTER += OUT[1];
		logger.info('--> J(Bx0)_evals %i, alpha_k=%e'%(OUT[1],alpha_k))
		
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
		    XK[i],PK[i] = Norm_constraint(domain, XK[i],PK[i], alpha_k, EQs[i]);

		B_k  = np.concatenate( (XK[0],XK[1]) );
		hg_k = np.concatenate( (PK[0],PK[1]) ); 

		##########################################################################		
		# 6) Compute errors & update counters
		##########################################################################	
		GRAD_IP_FULL = Inner_Prod(domain,gfk,gfk);
		Grad_error = np.sqrt(GRAD_IP_TANG/GRAD_IP_FULL);
		Cost_error = abs(J_k - J_k_old)/abs(J_k);
		
		File_Manips(DAL_LOOP_COUNTER);

		J_k_objective.append(J_k); # adds elements to the end = right hand side
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

	return None;

##########################################################################
# called from command line - why is this done ?
##########################################################################
if __name__ == "__main__":

	##########################################################################
	# Numerical Configuration Options
	##########################################################################
	# Write type & scale 
	WRITE_KIND = 'c'; WRITE_SCALE = 1; # Default ncc_cutoff is 1e-6, I've set 1e-12

	# Time-stepper
	#ts = "RK222"; 
	ts = "MCNAB2"; 
	ADJ_ts = ts; NCC_TOL = 1e-12;

	# Functional
	#Cost_function = 0; # J = <B_T,B_T>;
	Cost_function = 1; # J = int_t <B,B> dt

	# Formulation - Treat base state
	#CODE_FORMULATION = "Explicit"; # Currently set to mimick Spherical TC flow
	CODE_FORMULATION = "Implicit"; # Base state def -> IC_TC_MHD.TC_Base_State

	LOAD_SIM = None;	
	#Nx, Ny, Nz = 12,24,24; Noise = 0.5;
	Nx, Ny, Nz = 12,48,32; Noise = 0.5;
	#Nx, Ny, Nz = 36,96,48; Noise = 0.5;
	#MESH_SIZE = [6,5]; 
	MESH_SIZE = None;
	HOME_DIR = "/Users/pmannix/Desktop/Nice_CASTOR/DAL_PCF_KEP_MHD/"; 
	#HOME_DIR = "/workspace/pmannix/DAL_PCF_KEP_MHD/"; 
	#HOME_DIR = "/workdir/pmannix/DAL_PCF_KEP_MHD/"
	#STR = "Pm75_Re_Vs_M_Matrix/Results_DAL_Pm75_M1e-03_Re18.0/CheckPoints_iter_1.h5"
	STR = "Results_DAL_Re12Pm75_M1e-03_T0.25_Nx12Ny48Nz32/CheckPoints_iter_4.h5"
	LOAD_SIM = HOME_DIR + STR; # + "/CheckPoints/CheckPoints_s1.h5";	

	##########################################################################
	# (1) Load Parameters + set parameters
	##########################################################################

	# Geometry
	depth = 1.0
	alpha = 0.375; beta = 1.0; # Rincon
	X = (0.,(2.*np.pi)/alpha); # Stream-wise
	Y = (0.,(2.*np.pi)/beta ); # Span-wise
	Z = (-1.,1.);			   # Shear-wise

	# Control
	Re = 12.;   # Re = |S|d^2/\nu as per Rincon 2007
	mu = 4./3.; # = -2*Omega/S where S is the shear rate
	Pm = 75.0;  # Magnetic Prandtl Number #Pm = float(sys.argv[1]);print('Prandtl Number = %e'%Pm);

	# Min-Seeds
	M_0 = 5e-04; # = <B_0,B_0> Amplitude of Magentic initial condition
	E_0 = 0.;  # = <U_0,U_0> Amplitude of velocity initial condition

	# Time-step dt, Total number of iterations N_ITERS, Cadence at which to write/print N_SUB_ITERS
	dt = .05;
	N_ITERS = int(.05*(Pm*Re)/dt); # Total Number of time-steps => Total_time = N_ITERS*dt
	N_SUB_ITERS = N_ITERS//1; #//Num_checkpoints; # Divide by an integer for memory demanding cases

	if MPI.COMM_WORLD.rank == 0:
		print("Total forward iterations = %i \n"%N_ITERS);
	
	# Create a results file & parameters file
	try:
		os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
		atmosphere_file = h5py.File('Params.h5', 'w')

		# Problem Params
		atmosphere_file['Re'] = Re;
		atmosphere_file['Pm'] = Pm;
		atmosphere_file['mu'] = mu;

		# Domain Params
		atmosphere_file['alpha'] = alpha;
		atmosphere_file['beta']  = beta;

		# Numerical Params
		atmosphere_file['Nx'] = Nx;
		atmosphere_file['Ny'] = Ny;
		atmosphere_file['Nz'] = Nz;
		atmosphere_file['dt'] = dt;
		
		# Min seed Params
		atmosphere_file['M_0'] = M_0;
		atmosphere_file['E_0'] = E_0;
		atmosphere_file['T'] = N_ITERS*dt;
		atmosphere_file.close();
	except:	
		
		pass
	
	##########################################################################
	# (2) Generate IC's, Ux0 = {u,v,w}, Bx0 = {B_r,B_phi,B_z}
	##########################################################################
	from GEN_PCF_INIT_COND  import INIT_IVP_FWD_DAL
	
	Re_IC_Prep = 1.; mu_IC_Prep = mu; Pm_IC_Prep = 1. # Are these optimal to ensure a noisy IC?

	dt_IC = dt/1000.; IC_N_ITERS = 50; # Larger dt implies a smoother IC, previously I had mu_IC_Prep = mu; dt_IC=dt/50., IC_N_ITERS = 50;
	Solver_args = [Re_IC_Prep,Pm_IC_Prep,mu_IC_Prep, dt_IC, "RK443" ,NCC_TOL,CODE_FORMULATION];
	DAL_args 	= [IC_N_ITERS,IC_N_ITERS,WRITE_KIND,WRITE_SCALE,Cost_function];
	IC_args = Solver_args + DAL_args;

	# Vectors have norm:
	# <Ux0,Ux0> = E_0 with smooth radial derivative UZx0 = d/dz(Ux0), scaled accordingly
	# <Bx0,Bx0> = M_0 with smooth radial derivative BZx0 = d/dz(Bx0), scaled accordingly
	#Ux0,UZx0, 	Bx0,BZx0,	domain = INIT_IVP_FWD_DAL(Nx,Ny,Nz,  X,Y,Z, LOAD_SIM,  M_0,E_0,MESH_SIZE, *IC_args);
	
	from IC_PCF_MHD import INIT_IVP_FWD
	Ux0,UZx0,    Bx0, BZx0, domain = INIT_IVP_FWD(Nx,Ny,Nz,  X,Y,Z,  LOAD_SIM,  M_0,E_0,MESH_SIZE);
	#Ux0,UZx0,   dBx0,dBZx0,	domain = INIT_IVP_FWD_DAL(Nx,Ny,Nz,  X,Y,Z,  None,  M_0_pert,E_0_pert,MESH_SIZE, *IC_args);
	#sys.exit()
	##########################################################################
	# ~~~~~ Restart using old min-seed	FIX THIS
	##########################################################################
	'''
	from IC_PCF_MHD import Field_to_Bvec, Integrate_Field, new_ncc
	Bx  = new_ncc(domain);  By = new_ncc(domain);  Bz = new_ncc(domain);
	BZx = new_ncc(domain); BZy = new_ncc(domain); BZz = new_ncc(domain);

	file = h5py.File(LOAD_SIM ,"r");
	
	# Set this explicitly
	LOAD_SCALE = domain.dealias; KIND = 'g'; 
	slices = domain.dist.grid_layout.slices(scales=LOAD_SCALE); # Easier Debug

	vars = [Bx,By,Bz, BZx,BZy,BZz];
	for f in vars:
		f.set_scales(LOAD_SCALE, keep_data=False);
		f['g'] = 0.;

	if MPI.COMM_WORLD.rank == 0:
		#print(file['scales/'].keys()); print(file['tasks/'].keys()) #useful commands
		print(file['scales/iteration'][()]);
		print(file['scales/sim_time'][()]);
		print("Loaded B shape        ",file.get('tasks/B')[-1].shape);
		print("Place-holder shape B ",domain.dist.grid_layout.global_shape(scales=LOAD_SCALE));

		if file.get('tasks/B')[0].shape != tuple( domain.dist.grid_layout.global_shape(scales=LOAD_SCALE) ):
			print('\n Change LOAD_SCALE line 440 DAL_MAIN.py to resolve this...')
			print('Fatal: Scales mis-match !! \n');
			sys.exit()

	CheckPoint_Time = 0;
	Bx[KIND] = file.get('tasks/A')[CheckPoint_Time][slices];
	By[KIND] = file.get('tasks/B')[CheckPoint_Time][slices];
	Bz[KIND] = file.get('tasks/C')[CheckPoint_Time][slices];

	BZx[KIND] = file.get('tasks/Az')[CheckPoint_Time][slices];
	BZy[KIND] = file.get('tasks/Bz')[CheckPoint_Time][slices];
	BZz[KIND] = file.get('tasks/Cz')[CheckPoint_Time][slices];
	##MPI.COMM_WORLD.Barrier()

	for f in vars:
		f.set_scales(domain.dealias);

	# Check OLD amplitude - integrate field
	SUM = Integrate_Field(domain, (Bx**2)+(By**2)+(Bz**2) );
	if MPI.COMM_WORLD.rank == 0:
		print('OLD IC (1/V)<B,B> = %e'%SUM);

	Rescale_B = np.sqrt(M_0/SUM);
	Bx['g'] = Rescale_B*Bx['g']; 
	By['g'] = Rescale_B*By['g']; 
	Bz['g'] = Rescale_B*Bz['g'];

	BZx['g'] = Rescale_B*BZx['g']; 
	BZy['g'] = Rescale_B*BZy['g']; 
	BZz['g'] = Rescale_B*BZz['g'];
	##MPI.COMM_WORLD.Barrier();

	# Check NEW amplitude - integrate field
	V1 = Integrate_Field(domain, (Bx**2)+(By**2)+(Bz**2) );
	if MPI.COMM_WORLD.rank == 0:
		print('NEW IC (1/V)<B,B> = %e'%V1);
	
	Bx0  = Field_to_Bvec(domain, Bx,By,Bz);
	BZx0 = Field_to_Bvec(domain, BZx,BZy,BZz);
	'''
	##########################################################################


	# Generate buffers in memory to store the forward solution state
	X_FWD_DICT  = GEN_BUFFER(Nx, Ny, Nz, domain, N_SUB_ITERS, WRITE_KIND, Pm*Re)

	# Modify args before Pass
	Solver_args = [domain, Re,Pm,mu, dt,ts,ADJ_ts,NCC_TOL,CODE_FORMULATION];
	Solver_kwargs = {"domain":domain, "Re":Re,"Pm":Pm,"mu":mu, "dt":dt,"ts":ts,"ADJ_ts":ADJ_ts,"NCC_TOL":NCC_TOL,"CODE_FORMULATION":CODE_FORMULATION};

	DAL_args 	= [N_ITERS,N_SUB_ITERS,WRITE_KIND,WRITE_SCALE,Cost_function];
	DAL_kwargs 	= {'N_ITERS':N_ITERS,'N_SUB_ITERS':N_SUB_ITERS,'WRITE_KIND':WRITE_KIND,'WRITE_SCALE':WRITE_SCALE,'Cost_function':Cost_function};
	DAL_kwargs.update(**Solver_kwargs);

	Other_args  = [X_FWD_DICT,Ux0,UZx0] + Solver_args + DAL_args;
	Other_kwargs  = {'X_FWD_DICT':X_FWD_DICT,'Ux0':Ux0,'UZx0':UZx0};
	Other_kwargs.update(**DAL_kwargs);

	Solver_keys = ['domain', 'Re','Pm','mu', 'dt','ts','ADJ_ts','NCC_TOL','CODE_FORMULATION'];
	DAL_keys 	= ['N_ITERS','N_SUB_ITERS','WRITE_KIND','WRITE_SCALE','Cost_function'];
	Other_keys  = ['X_FWD_DICT','Ux0','UZx0'] + Solver_keys + DAL_keys;

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# (3) Taylor Test the gradient
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#'''
	from DAL_PCF_GRAD_Test import Adjoint_Gradient_Test, PLOT_GRADIENT_TEST
	# Generate a perturbation vector to supply to Taylor-Test
	Ux0,UZx0,   dBx0,dBZx0,	domain = INIT_IVP_FWD_DAL(Nx,Ny,Nz,  X,Y,Z,  None,  M_0,E_0,MESH_SIZE, *IC_args);
	Adjoint_Gradient_Test(Bx0,BZx0,		dBx0,dBZx0,		*Other_args)
	PLOT_GRADIENT_TEST(); # Requires editing the file-name within
	sys.exit()	
	#'''

	if MPI.COMM_WORLD.rank == 0:
		print("########################################################################## \n\n")	
		#print(dir())
		#print("\n")
		#print(dir(INIT_IVP_FWD))
		print("Re=",Re);
		print("Pm=",Pm);
		print("mu=",mu);
		print("dt=",dt);
		print("Lx=",X)
		print("Ly=",Y)
		print("Lz=",Z)
		print("########################################################################## \n\n")	

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# (4) Perform DAL & Gradient descent for a minimal seed
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	DAL_LOOP(Bx0,BZx0, M_0,*Other_args);

	'''	
	from FWD_Solve_PCF_MHD import FWD_Solve_IVP
	FWD_Solve_IVP(Bx0,BZx0, Ux0,UZx0, None,
	domain, Re,Pm,mu, dt,ts,NCC_TOL,CODE_FORMULATION,
	N_ITERS,N_SUB_ITERS,WRITE_KIND,WRITE_SCALE,Cost_function);
	sys.exit()
	'''
