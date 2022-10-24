import sys,os,time;
os.environ["OMP_NUM_THREADS"] = "1" 
from mpi4py import MPI 
import numpy as np
import h5py,logging

'''
Solve the Swift-Hohenberg optimisation problem

max J(u) = ∫_t ∫_x |u(x,t)|^2 dx dt,
u_0

s.t. ∫_x (1/2)*|u0|^2 dx = E0,

	 ∂_t u + (1 + ∂_x^2)^2 u − au = 1.8u^2 − u^3,

on a periodic domain L_x = 12π, using a Fourier basis.

To run this code :        mpiexec -np 1 python3 FWD_Solve_SH23.py 
& to plot the solution :  python3 plot_figure_SH23_FULL.py
'''


##########################################################################
# ~~~~~ General Routines ~~~~~~~~~~~~~
##########################################################################

def filter_field(field,frac=0.5):
    
	"""
	Given a dedalus field object, set "frac" of the highest wave_number coefficients to zero

	Useful when initialising simulations with differentiated noise.

	Input: 
	dedlaus field object
	frac = float in (0,1)

	Returns:
	None - field passed by reference in effect
	"""

	field.require_coeff_space()
	dom = field.domain                                                                                                                                                      
	local_slice = dom.dist.coeff_layout.slices(scales=dom.dealias)                                                                                                          
	coeff = []                                                                                                                                                              
	for n in dom.global_coeff_shape:
		coeff.append(np.linspace(0,1,n,endpoint=False))                                                                                             
	cc = np.meshgrid(*coeff,indexing='ij')
	field_filter = np.zeros(dom.local_coeff_shape,dtype='bool')                                                                                                             
	for i in range(dom.dim):                                                                                                                                                
		field_filter = field_filter | (cc[i][local_slice] > frac)                                                                                                           
	field['c'][field_filter] = 0j

def new_ncc(domain, scales=True, keep_data=False, const=[]):
	"""
	Generate a new non-constant-coefficient field (ncc) on the prescribed domain.
	"""
	field = domain.new_field()
	for direction in const:
		field.meta[direction]['constant'] = True
	if (scales):
		field.set_scales(domain.dealias, keep_data=keep_data)
	return field

def Integrate_Field(domain,F):

	"""
	Performs the Volume integral of a given field F as 

	KE(t) = 1/L_x int_x F(x) dx, where F = u^2

	where KE is Kinetic Enegry
	"""
	# Dedalus Libraries
	from dedalus.extras import flow_tools 
	import dedalus.public as de

	# 1) Multiply by the integration weights r*dr*d_phi*dz for cylindrical domain
	flow_red = flow_tools.GlobalArrayReducer(MPI.COMM_WORLD);
	INT_ENERGY = de.operators.integrate( F ,'x');
	SUM = INT_ENERGY.evaluate();
	
	# Divide by volume size
	VOL = 1./domain.hypervolume;

	return VOL*flow_red.global_max(SUM['g']); # Using this as it's what flow-tools does for volume average

def Field_to_Vec(domain,Fx    ):
	
	"""
	Convert from: field to numpy 1D vector-
	
	Takes the LOCAL portions of:
	Inputs:
	- GLOBALLY distributed fields Fx
	- domain object 

	Creates locally available numpy arrays of a global size &
	makes this array is available on all cores using MPI gather_all function
	
	This function assumes all arrays can fit in memory!!!

	Returns:
	- 1D np.array of Fx
	"""

	# 1) Create local array of the necessary dimension
	#lshape = domain.dist.grid_layout.local_shape(scales=domain.dealias)
	gshape = tuple( domain.dist.grid_layout.global_shape(scales=domain.dealias) );
	
	Fx.set_scales(domain.dealias,keep_data=True);

	# 2) Gather all data onto EVERY processor!!
	Fx_global = MPI.COMM_WORLD.allgather(Fx['g']);
	
	# Gathered slices
	G_slices = MPI.COMM_WORLD.allgather( domain.dist.grid_layout.slices(scales=domain.dealias) )

	# 3) Declared an array of GLOBAL shape on every proc
	FX = np.zeros(gshape);

	# Parse distributed fields into arrays on every proc
	for i in range( MPI.COMM_WORLD.Get_size() ):
		FX[G_slices[i]] = Fx_global[i];

	# 4) Merge them together at the end!
	return FX.flatten();

def Vec_to_Field(domain,A, Bx0):

	"""
	Convert from: numpy 1D vector to field - 
	Takes a 1D array Bx0 and distributes it into field A 
	num_procs = MPI.COMM_WORLD.size

	Inputs:
	- domain object 
	- GLOBALLY distributed dedalus field A
	- Bx0 1D np.array

	Returns:
	- None
	"""

	# 1) Split the 1D array into 1D arrays A
	gshape = tuple( domain.dist.grid_layout.global_shape(scales=domain.dealias) )
	slices = domain.dist.grid_layout.slices(scales=domain.dealias)
	
	A.set_scales(domain.dealias,keep_data=False);
	A['g']=0.

	# 2) Reshape and parse relevant portion into A
	A['g'] = Bx0.reshape( gshape )[slices]

	return None;

def Inner_Prod(x,y,domain,rand_arg=None):

	# The line-search requires the IP
	# m = <\Nabla J^T, P_k >, where P_k = - \Nabla J(X)^T
	# Must be evaluated using an integral consistent with our objective function
	# i.e. <,> = (1/V) int_V x*y dV
	# To do this we transform back to fields and integrate using a consistent routine

	dA = new_ncc(domain);
	Vec_to_Field(domain,dA,x);

	dB = new_ncc(domain);
	Vec_to_Field(domain,dB,y);

	return Integrate_Field(domain, dA*dB );

def Generate_IC(E_0=1.0, Npts = 256, X = (0.,12.*np.pi)):
	"""
	Generate a domain object and initial conditions from which the optimisation can proceed

	Input:
	- Npts - integer resolution size
	- X    - interval/domain scale
	- E_0  - initial condition amplitude
	
	Returns: 
	- domain object
	- initial cond U0 , as a field obj U
	"""
	
	import dedalus.public as de

	# Set to info level rather than the debug default
	root = logging.root
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO"); #h.setLevel("DEBUG")
	logger = logging.getLogger(__name__)


	logger.info('\n\n Generating initial conditions ..... \n\n');

	# Part 1) Generate domain
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	dealias_scale = 2;
	x_basis = de.Fourier('x', Npts, interval=X, dealias=dealias_scale); # x
	domain  = de.Domain([x_basis], grid_dtype=np.float64);

	# Part 2) Generate initial condition U = phi
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	phi = domain.new_field();
	phi.set_scales(domain.dealias, keep_data=False)
	gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
	slices = domain.dist.grid_layout.slices(scales=domain.dealias)
	rand = np.random.RandomState(seed=42)
	noise = rand.standard_normal(gshape)[slices]; #Slicing globally generated noise here!!
	phi['g'] = noise; # Could scale this ???
	filter_field(phi)   # Filter the noise, modify this for less noise

	#x = domain.grid(0,scales=domain.dealias);
	#phi['g'] = np.cos(1*x);

	# 3) Normalise it
	SUM = Integrate_Field(domain, phi**2 );
	logger.info('Pre-scale (1/V)<U,U> = %e'%SUM);
	phi['g'] = np.sqrt(E_0/SUM)*phi['g'];
	SUM = Integrate_Field(domain, phi**2 );
	logger.info('Scaled (1/V)<U*,U*> = %e'%SUM);

	# INTEGRATE IT a few steps
	phi['g'] = FWD_Solve_IVP_PREP([phi], domain)['g']

	SUM = Integrate_Field(domain, phi**2 );
	logger.info('Re-scale (1/V)<U,U> = %e'%SUM);
	phi['g'] = np.sqrt(E_0/SUM)*phi['g'];
	SUM = Integrate_Field(domain, phi**2 );
	logger.info('Re-Scaled (1/V)<U*,U*> = %e'%SUM);

	return domain, Field_to_Vec(domain,phi);

def GEN_BUFFER(domain, N_SUB_ITERS,Npts=256):

	"""
	Given the resolution and number of N_SUB_ITERS

	# 1) Estimate the memory usage per proc/core

	# 2) If possible create a memory buffer as follows:

	# The buffer created is a Python dictionary - X_FWD_DICT = {'U(x,t)'}
	# Chosen so as to allow pass by reference & thus avoids copying
	# All arrays to which these "keys" map must differ as we are otherwise mapping to the same memory!!
	
	Returns:
	Memory Buffer - Type Dict { 'Key':Value } where Value is 4D np.array

	"""
	
	################################################################################################################
	# B) Build memory buffer
	################################################################################################################
		
	# -Total  = (0.5 complex to real)*Npts*(1 fields)*(64 bits)*N_SUB_ITERS*(1.25e-10)/MPI.COMM_WORLD.Get_size()
	# -float64 = 64bits # Data-type used # -1 bit = 1.25e-10 GB
	Total  = ( 0.5*Npts*64*N_SUB_ITERS*(1.25e-10) )/float( MPI.COMM_WORLD.Get_size() )
	if MPI.COMM_WORLD.rank == 0:
		print("Total memory =%f GB, and memory/core = %f GB"%(MPI.COMM_WORLD.Get_size()*Total,Total));

	gshape  = tuple( domain.dist.coeff_layout.global_shape(scales=1) );
	lcshape = tuple( domain.dist.coeff_layout.local_shape(scales=1) );
	SNAPS_SHAPE = (lcshape[0],N_SUB_ITERS+1);

	A_SNAPS = np.zeros(SNAPS_SHAPE,dtype=complex);

	return {'A_fwd':A_SNAPS};


##########################################################################
# ~~~~~ FWD Solvers ~~~~~~~~~~~~~
##########################################################################

def FWD_Solve_Build_Lin(domain):
	
	"""
	Driver program for the Swift-Hohenberg SH23 equation, which builds the forward solver object with options:

	Inputs:
	domain (dedalus object) 

	Returns:
	Dedalus object to solve the SH23 equation

	"""

	# Dedalus Libraries
	import dedalus.public as de

	# Set to info level rather than the debug default
	root = logging.root
	for h in root.handlers:
		h.setLevel("WARNING");
		#h.setLevel("INFO"); 
		#h.setLevel("DEBUG")
	logger = logging.getLogger(__name__)

	#######################################################
	# initialize the problem
	#######################################################

	# Equations Variables 
	PCF = de.IVP(domain, variables=['u'], time='t');
	PCF.parameters['a'] = -0.3;
	PCF.parameters['inv_Vol'] = 1./domain.hypervolume;


	#######################################################
	# Substitutions (1 + dx^2)^2
	#######################################################
	PCF.substitutions['Lap(f)'] = "f + 2.*dx(dx( f )) + dx(dx(  dx(dx( f  ))  )) ";

	#######################################################
	# add equations
	#######################################################
	logger.info("--> Adding Equations");
	PCF.add_equation("dt(u) + Lap(u) - a*u = 1.8*(u**2) - (u**3)"); #

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	IVP_FWD = PCF.build_solver(de.timesteppers.SBDF1);

	# Set to info level rather than the debug default
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");

	return IVP_FWD;

def FWD_Solve_IVP_PREP(X_k, domain, dt=1e-02,  N_ITERS=100, N_SUB_ITERS=100):
	
	"""
	Integrates the initial condition X(t=0) = u(x,t=0) -> u(x,T);
	using the SH23 equation,

	Input:
	X_k - list of a numpy vector initial condition in grid-space
	domain 

	Returns:
		X_k - smoothed initial condition in grid-space
	"""
	from dedalus.extras import flow_tools
	from dedalus.tools  import post
	import dedalus.public as de

	# Set to info level rather than the debug default
	root = logging.root
	for h in root.handlers:
		h.setLevel("WARNING");
		#h.setLevel("INFO");
		#h.setLevel("DEBUG")
	logger = logging.getLogger(__name__)

	#######################################################
	# initialize the problem
	#######################################################

	X0_field = X_k[0];

	IVP_FWD = FWD_Solve_Build_Lin(domain);

	u = IVP_FWD.state['u']; 
	u.set_scales(domain.dealias, keep_data=False)
	u['g'] = X0_field['g']

	#######################################################
	# evolution parameters
	######################################################

	#IVP_FWD.stop_sim_time = tstop;
	#IVP_FWD.stop_wall_time = tstop_wall*3600.
	IVP_FWD.stop_iteration = np.inf
	IVP_FWD.stop_iteration = N_ITERS+1; # Total Foward Iters + 1, to grab last point

	IVP_FWD.sim_tim   = IVP_FWD.initial_sim_time = 0.
	IVP_FWD.iteration = IVP_FWD.initial_iteration = 0	


	#######################################################	
	logger.info("\n\n --> Timestepping FWD_Solve IC PREP ");
	#######################################################

	N_PRINTS = N_SUB_ITERS//2;		
	flow = flow_tools.GlobalFlowProperty(IVP_FWD, cadence=1);
	flow.add_property("inv_Vol*integ( u**2 )", name='J(u)');

	while IVP_FWD.ok:
		IVP_FWD.step(dt);
		'''
		if IVP_FWD.iteration % N_PRINTS == 0:

			logger.info('Iterations: %i' %IVP_FWD.iteration)
			logger.info('Sim time:   %f' %IVP_FWD.sim_time )
			
			logger.info('KE = %e'%flow.volume_average('J(u)') );
		'''

	for h in root.handlers:
		h.setLevel("WARNING");
		#h.setLevel("INFO");

	return u;

def FWD_Solve_IVP_Lin(X_k, domain, dt,  N_ITERS, N_SUB_ITERS, X_FWD_DICT,filename=None, Adjoint_type = "Discrete"):
	
	"""
	Integrates the initial condition X(t=0) = u(x,t=0) -> u(x,T);
	using the SH23 equation,

	Input:
	domain (dedalus object)
	dt - float numerical integration time-step
	N_ITERS,N_SUB_ITERS - int number of iterations to complete
	X_FWD_DICT - fWD checkpoints buffer
	
	Returns:
		time integrated kinetic energy
	
	- Writes the following to disk:
	1) FILE Scalar-data (every 20 iters): kinetic Enegry, etc. 

	2) FILE Checkpoints (every N_SUB_ITERS): full system state in grid space

	"""
	from dedalus.extras import flow_tools
	from dedalus.tools  import post
	import dedalus.public as de

	# Set to info level rather than the debug default
	root = logging.root
	for h in root.handlers:
		h.setLevel("WARNING");
		#h.setLevel("INFO");
		#h.setLevel("DEBUG")
	logger = logging.getLogger(__name__)

	#######################################################
	# initialize the problem
	#######################################################

	IVP_FWD = FWD_Solve_Build_Lin(domain);

	u = IVP_FWD.state['u']; 
	u.set_scales(domain.dealias, keep_data=False)
	u['g'] = 0.

	#######################################################
	# set initial conditions
	#######################################################

	X0 = X_k[0];
	Vec_to_Field(domain,u,X0)

	if filename != None:
		IVP_FWD.load_state(filename,index=-1)

	#######################################################
	# evolution parameters
	######################################################

	#IVP_FWD.stop_sim_time = tstop;
	#IVP_FWD.stop_wall_time = tstop_wall*3600.
	IVP_FWD.stop_iteration = np.inf
	IVP_FWD.stop_iteration = N_ITERS+1; # Total Foward Iters + 1, to grab last point

	IVP_FWD.sim_tim   = IVP_FWD.initial_sim_time = 0.
	IVP_FWD.iteration = IVP_FWD.initial_iteration = 0	

	
	#######################################################
	# analysis tasks
	#######################################################
	analysis_CPT = IVP_FWD.evaluator.add_file_handler('CheckPoints', iter=N_SUB_ITERS, mode='overwrite');
	analysis_CPT.add_system(IVP_FWD.state, layout='g', scales=3/2); 
	analysis_CPT.add_task("u", name="u_hat", layout='c', scales=1);

	analysis1 = IVP_FWD.evaluator.add_file_handler("scalar_data", iter=20, mode='overwrite');
	analysis1.add_task("inv_Vol*integ(u**2)", name="Kinetic energy")

	#######################################################	
	logger.info("\n\n --> Timestepping FWD_Solve ");
	#######################################################

	N_PRINTS = N_SUB_ITERS//2;
		
	flow = flow_tools.GlobalFlowProperty(IVP_FWD, cadence=1);
	flow.add_property("inv_Vol*integ( u**2 )", name='J(u)');

	J_TRAP = 0.;
	snapshot_index = 0;
	while IVP_FWD.ok:

		# 1) Fill Dictionary
		if (IVP_FWD.iteration >= (N_ITERS - N_SUB_ITERS)) and (snapshot_index <= N_SUB_ITERS ):

			#X_FWD_DICT = {'A_fwd':A_SNAPS}
			X_FWD_DICT['A_fwd'][:,snapshot_index] = u['c'];
			snapshot_index+=1;


		IVP_FWD.step(dt);
		if IVP_FWD.iteration % N_PRINTS == 0:
			
			logger.info('Iterations: %i' %IVP_FWD.iteration)
			logger.info('Sim time:   %f' %IVP_FWD.sim_time )

			logger.info('KE = %e'%flow.volume_average('J(u)') );

		# 3) Evaluate Cost_function using flow tools, # J = int_t KE(t) dt 
		# flow tools value is that of ( IVP_FWD.iteration-1 )

		'''
		# Trapezoidal rule
		if ( (IVP_FWD.iteration-1) == 0):
			J_TRAP += dt*0.5*flow.volume_average('J(u)')
		elif (0 < (IVP_FWD.iteration-1) < N_ITERS):
			J_TRAP += dt*flow.volume_average('J(u)')
		elif ( (IVP_FWD.iteration-1) == N_ITERS): 
			J_TRAP += dt*0.5*flow.volume_average('J(u)');
		'''	
		
		# Simple Euler integration 1st order dt	
		if ((IVP_FWD.iteration-1) >= 0) and ((IVP_FWD.iteration-1) <= N_ITERS):	
			J_TRAP += dt*flow.volume_average('J(u)')	
	
	#######################################################

	# final statistics
	post.merge_process_files("CheckPoints", cleanup=True, comm=MPI.COMM_WORLD);
	post.merge_process_files("scalar_data", cleanup=True, comm=MPI.COMM_WORLD);
	time.sleep(1);
	logger.info("\n\n--> Complete <--\n")

	logger.info('J(u) = %e'%J_TRAP );

	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");

	return (-1.)*J_TRAP;


##########################################################################
# ~~~~~ ADJ Solvers + Comptability Condition ~~~~~~~~~~~~~
##########################################################################

def Compatib_Cond(X_FWD_DICT, domain, dt):

	"""
	Implements the discrete compatibility condition
	
	"""

	root = logging.root
	for h in root.handlers:
		h.setLevel("WARNING");
		#h.setLevel("INFO"); #h.setLevel("DEBUG")
	logger = logging.getLogger(__name__)

	logger.info("\n\n Inverting linear operator to solve compatibility condition ....")

	# Dedalus Libraries
	import dedalus.public as de

	prob = de.LBVP(domain, variables=['q']);
	
	fx = new_ncc(domain);
	fx['c'] = X_FWD_DICT['A_fwd'][:,-1];
	
	prob.parameters['fx']= fx;
	prob.parameters['a']=-0.3;
	
	# SBDF1 Parameters
	prob.parameters['a_0']=1./dt;
	prob.parameters['b_0']=1./1.;
	
	prob.substitutions['Lap(f)'] = "f + 2.*dx(dx( f )) + dx(dx(  dx(dx( f  ))  )) ";
	
	prob.add_equation('a_0*q + b_0*(Lap(q) -a*q) = -2.*fx');
		
	solver = prob.build_solver();
	solver.solve();

	q_init = solver.state['q'];
	q_init.set_scales(domain.dealias, keep_data=True);

	for h in root.handlers:
		h.setLevel("WARNING");
		#h.setLevel("INFO"); #h.setLevel("DEBUG")

	return {'q_init':q_init};

def ADJ_Solve_IVP_Lin(X_k, domain, dt,  N_ITERS,N_SUB_ITERS, X_FWD_DICT, filename=None, Adjoint_type = "Discrete"):	
	
	"""
	Computes gradient of SH23 equation

	Inputs:
	domain (dedalus object)
	dt - float numerical integration time-step
	N_ITERS,N_SUB_ITERS - int number of iterations to complete
	X_FWD_DICT - fWD checkpoints buffer

	Returns:
	[dJ/du_0] - gradient list containing a vector

	"""

	# Dedalus Libraries
	import dedalus.public as de

	# Set to info level rather than the debug default
	root = logging.root
	for h in root.handlers:
		h.setLevel("WARNING");
		#h.setLevel("INFO"); #h.setLevel("DEBUG")
	logger = logging.getLogger(__name__)

	#######################################################
	# initialize the problem
	#######################################################

	X0 = X_k[0];
	
	logger.info("--> Adding Equations");
	
	PBox = de.IVP(domain, variables=['q'], time='t');
	PBox.parameters['a'] = -0.3;
	PBox.parameters['inv_Vol'] = 1./domain.hypervolume;

	u_f = new_ncc(domain);
	PBox.parameters['uf'] = u_f

	PBox.substitutions['Lap(f)'] = "f + 2.*dx(dx( f )) + dx(dx(  dx(dx( f  ))  )) ";
	PBox.add_equation("dt(q) + Lap(q) - a*q =  (3.6*uf - 3.*(uf**2) )*q - 2.*uf");

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	IVP_ADJ = PBox.build_solver(de.timesteppers.SBDF1);

	q = IVP_ADJ.state['q'];
	q.set_scales(domain.dealias, keep_data=False)
	q['g'] = 0.	

	#######################################################
	# set initial conditions
	#######################################################
	
	#X_FWD_DICT = {'A_fwd':A_SNAPS}
	if   Adjoint_type == "Continuous":
		
		snapshot_index = -1; # Continuous 

	elif Adjoint_type == "Discrete":

		X_DICT = Compatib_Cond(X_FWD_DICT, domain, dt);
		q['g'] = X_DICT['q_init']['g'];
		
		snapshot_index = -2; # Discrete due to shifting ofindicies

	#######################################################
	# evolution parameters
	######################################################

	IVP_ADJ.stop_iteration = np.inf
	IVP_ADJ.stop_iteration = N_ITERS; #+1; # Total Foward Iters + 1, to grab last point, /!\ only if checkpointing

	IVP_ADJ.sim_tim   = IVP_ADJ.initial_sim_time  = 0.
	IVP_ADJ.iteration = IVP_ADJ.initial_iteration = 0.
	
	#######################################################	
	logger.info("\n\n --> Timestepping ADJ_Solve ");
	#######################################################

	N_PRINTS = N_SUB_ITERS//10;
	from dedalus.extras import flow_tools
	flow = flow_tools.GlobalFlowProperty(IVP_ADJ, cadence=1);
	flow.add_property("inv_Vol*integ( q**2  )", name='J(q)');

	while IVP_ADJ.ok:

		# Must leave this as it's needed for the dL/dU-gradient
		#X_FWD_DICT = {'A_fwd':A_SNAPS}
		IVP_ADJ.problem.namespace['uf']['c'] = X_FWD_DICT['A_fwd'][:,snapshot_index]
		snapshot_index-=1; # Move back in time (<-t


		IVP_ADJ.step(dt);

		if IVP_ADJ.iteration % N_PRINTS == 0:
			logger.info('Iterations: %i' %IVP_ADJ.iteration)
			logger.info('Sim time: %f, Time-step dt: %f' %(IVP_ADJ.sim_time,dt));
			logger.info('KE = %e'%( flow.volume_average('J(q)') 	) );

	#######################################################

	# For discrete adjoint undo LHS inversion of the last-step
	if Adjoint_type == "Discrete":
		
		u_f.set_scales(domain.dealias, keep_data=False)	
		u_f['g']=0.;

		a_0, b_0 = 1./dt, 1./1.; # SBDF1 Parameters
		param_a = -0.3;
		
		q_xx    = q.differentiate('x').differentiate('x')
		q_xx_xx = q_xx.differentiate('x').differentiate('x')

		u_f['g'] = dt*(  a_0*q['g'] + b_0*( (1.-param_a)*q['g'] + 2.*q_xx['g'] + q_xx_xx['g'] )  ); 

		Ux0 = Field_to_Vec(domain,u_f )
	
	else:
		
		Ux0 = Field_to_Vec(domain,q )
	

	logger.info("\n\n--> Complete <--\n")

	# Set to info level rather than the debug default
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");

	return [Ux0];

def File_Manips(k):

	"""
	Takes files generated by the adjoint loop solve, orgainises them into respective directories
	and removes unwanted data

	Each iteration of the DAL loop code is stored under index k
	"""		

	import shutil

	shutil.copyfile('scalar_data/scalar_data_s1.h5','scalar_data_iter_%i.h5'%k);

	shutil.copyfile('CheckPoints/CheckPoints_s1.h5','CheckPoints_iter_%i.h5'%k);

	return None;		



if __name__ == "__main__":

	E_0 = 0.0725;
	dt  = 0.05;
	N_ITERS     = int(50./dt);
	N_SUB_ITERS = N_ITERS//1;

	Adjoint_type = "Discrete";
	#Adjoint_type = "Continuous";


	# 1) Generate an initial condition of norm E_0 
	domain, X_0  = Generate_IC(E_0);
	X_FWD_DICT   = GEN_BUFFER(domain, N_SUB_ITERS)


	sys.path.insert(0,'../../../')
	
	args_IP = (domain,None);
	args_f  = [domain, dt,  N_ITERS, N_SUB_ITERS, X_FWD_DICT, None, Adjoint_type];


	# 2) Test the gradient
	#'''
	from TestGrad import Adjoint_Gradient_Test
	domain, X_0  = Generate_IC(1.);
	_     , dX_0 = Generate_IC(1.);
	Adjoint_Gradient_Test(X_0,dX_0,FWD_Solve_IVP_Lin,ADJ_Solve_IVP_Lin,Inner_Prod,args_f,args_IP,epsilon=1e-04)
	sys.exit()
	#'''

	# 3) Optimise the initial perturbation u(x,t=0)
	from Sphere_Grad_Descent import Optimise_On_Multi_Sphere, plot_optimisation
	RESIDUAL, FUNCT, X_opt = Optimise_On_Multi_Sphere([X_0],[E_0],FWD_Solve_IVP_Lin,ADJ_Solve_IVP_Lin,Inner_Prod,args_f,args_IP,max_iters=200,alpha_k = np.pi,LS = 'LS_wolfe', CG = True,callback=File_Manips)
	plot_optimisation(RESIDUAL,FUNCT);

	
	'''
	# Save the different errors 
	DAL_file = h5py.File('DAL_PROGRESS.h5', 'r+')

	# Problem Params
	RESIDUAL = DAL_file['Residual'][()];
	FUNCT    = DAL_file['Function_Value'][()]; 
	X_0 	 = DAL_file['X_opt'][0];

	DAL_file.close();

	#RESIDUAL, FUNCT, X_opt = Optimise_On_Multi_Sphere([X_0],[E_0],FWD_Solve_IVP_Lin,ADJ_Solve_IVP_Lin,Inner_Prod,args_f,args_IP,max_iters=200,LS = 'LS_armijo', CG = False,callback=File_Manips)
	plot_optimisation(RESIDUAL,FUNCT)
	'''
	###
