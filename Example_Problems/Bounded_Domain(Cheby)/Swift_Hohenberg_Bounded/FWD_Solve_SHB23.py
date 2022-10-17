import sys,os,time;
os.environ["OMP_NUM_THREADS"] = "1" 
from mpi4py import MPI 
import numpy as np
import h5py,logging

from scipy.fftpack import fft, dct
from dedalus.core import field
from dedalus.core import system

'''
Solve the Swift-Hohenberg optimisation problem

max J(u) = ∫_t ∫_x |u(x,t)|^2 dx dt,
u_0

s.t. ∫_z (1/2)*|u0|^2 dz = E0,
	 ∂_t u + (1 + ∂_z^2)^2 u − au = 2u^2 − u^3,

on a bounded domain Z = [-20,20] using a Chebyshev basis wth bcs

∂_z(u) = ∂_zzz(u) =0 for z = -20

	 u = ∂_zz(u)  =0 for z =  20

To run this code :        mpiexec -np 1 python3 FWD_Solve_SHB23.py 
& to plot the solution :  python3 plot_figure_SHB23.py
'''



##########################################################################
# ~~~~~ Chebyshev transforms and their discrete adjoints ~~~~~~~~~~~~~
##########################################################################

def transform(x):
    b = dct(x,type=2)/len(x)
    b[0] *= 0.5
    b[1::2] *= -1
    return b

def transformAdjoint(x):
    c = x.copy()
    c[0] *= 0.5
    c[1::2] *= -1
    c[0]  *= np.sqrt((4*len(x)))
    c[1:] *= np.sqrt((2*len(x)))
    b = dct(c,type=3,norm='ortho')/len(x)
    return b

def transformInverse(x):
    c = x.copy()
    c[1::2] *= -1
    c[1:] *= 0.5
    b = dct(c,type=3)

    return b

def transformInverseAdjoint(x):

    b = dct(x,type=2,norm='ortho')
    b[0]  *= np.sqrt(len(x))
    b[1:] *= 2.*np.sqrt(len(x)/2.)
    b[1:] *= 0.5
    b[1::2] *= -1

    return b

def weightMatrixDisc(domain):
	
	z = domain.grid(0, scales=1)
	W = np.zeros(Npts);
	for i in range(Npts):
		if(i==0):
			W[i] = 0.5*(z[1]-z[0]);
		elif(i==Npts-1):
			W[i] = 0.5*(z[Npts-1]-z[Npts-2]);
		else:
			W[i] = 0.5*(z[i]-z[i-1]) + 0.5*(z[i+1]-z[i]);

	return W;

##########################################################################
# ~~~~~ General Routines ~~~~~~~~~~~~~
##########################################################################

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
	Takes a 1D array Bx0 and distributes it into fields A
	num_procs = MPI.COMM_WORLD.size

	Inputs:
	- domain object
	- GLOBALLY distributed dedalus fields A
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

def Inner_Prod_Cnts(x,y,domain,Type_xy='np_vector'):

	"""
	Performs the Volume integral of a given product of fields fg as

	<f,g> = 1/L_z int_z F(z)G(z) dz,


	"""

	# Dedalus Libraries
	from dedalus.extras import flow_tools
	import dedalus.public as de

	if Type_xy == 'np_vector':
		dA = domain.new_field()
		Vec_to_Field(domain,dA,x);

		dB = domain.new_field()
		Vec_to_Field(domain,dB,y);
	else:
		dA = x; dB = y;


	flow_red   = flow_tools.GlobalArrayReducer(MPI.COMM_WORLD);
	INT_ENERGY = de.operators.integrate( dA*dB ,'z');
	SUM        = INT_ENERGY.evaluate();

	# Divide by volume size
	VOL = 1./domain.hypervolume;

	return VOL*flow_red.global_max(SUM['g']); # Using this as it's what flow-tools does for volume average

def Inner_Prod_Discrete(x,y,domain,Type_xy='np_vector'):
	
	W=weightMatrixDisc(domain)

	return np.dot(x,W*y)/domain.hypervolume;

def Generate_IC(Npts, Z = (-20.,20.), M_0=1.0,Type_IP = 'Field'):
	"""
	Generate a domain object and initial conditions from which the optimisation can proceed

	Input:
	- Npts - integer resolution size
	- Z    - interval/domain scale
	- M_0  - initial condition amplitude

	Returns:
	- domain object
	- initial cond U0 , as a field obj U
	"""

	import dedalus.public as de

	# Part 1) Generate domain
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	if Adjoint_type == "Discrete":
		z_basis = de.Chebyshev('z',  Npts, interval=Z, dealias=1)
	else:
		z_basis = de.Chebyshev('z',  Npts, interval=Z, dealias=2)
	domain = de.Domain([z_basis], np.float64)

	# Part 2) Generate initial condition U = phi
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
		field['c'][field_filter] = 0. # as coefficients are real not 1j

	phi = domain.new_field();
	phi.set_scales(domain.dealias, keep_data=False)
	gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
	slices = domain.dist.grid_layout.slices(scales=domain.dealias)
	rand = np.random.RandomState(seed=42)
	noise = rand.standard_normal(gshape)[slices];
	phi['g'] = noise;
	if(Adjoint_type=='Discrete'):
		filter_field(phi,frac=0.25)
	else:
		filter_field(phi)

	phi['g'] = FWD_Solve_IVP_PREP([phi], domain)['g']

	if Adjoint_type =='Discrete':
		SUM  = Inner_Prod(phi['g'],phi['g'],domain,Type_IP );
	else:
		SUM  = Inner_Prod(phi,phi,domain,Type_IP );	
	phi['g'] = np.sqrt(M_0/SUM)*phi['g'];

	return domain, Field_to_Vec(domain,phi);

def GEN_BUFFER(Npts, domain, N_SUB_ITERS):

	"""
	Given the resolution and number of N_SUB_ITERS

	# 1) Estimate the memory usage per proc/core

	# 2) If possible create a memory buffer as follows:

	# The buffer created is a Python dictionary - X_FWD_DICT = {'U(z,t)'}
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
	Total  = ( Npts*64*N_SUB_ITERS*(1.25e-10) )/float( MPI.COMM_WORLD.Get_size() )
	if MPI.COMM_WORLD.rank == 0:
		print("Total memory =%f GB, and memory/core = %f GB"%(MPI.COMM_WORLD.Get_size()*Total,Total));

	if	(Adjoint_type == "Discrete"):

		gshape  = tuple( domain.dist.grid_layout.global_shape(scales=1) );
		lgshape = tuple( domain.dist.grid_layout.local_shape( scales=1) );

		SNAPS_SHAPE = (lgshape[0],N_SUB_ITERS+1);

	elif (Adjoint_type == "Continuous"):

		gshape  = tuple( domain.dist.coeff_layout.global_shape(scales=1) );
		lcshape = tuple( domain.dist.coeff_layout.local_shape( scales=1) );

		SNAPS_SHAPE = (lcshape[0],N_SUB_ITERS+1);

	A_SNAPS = np.zeros(SNAPS_SHAPE,dtype=np.float64);

	return {'A_fwd':A_SNAPS};


##########################################################################
# ~~~~~ FWD Solvers ~~~~~~~~~~~~~
##########################################################################

def FWD_Solve_Build_Lin(domain):

	"""
	Driver program for the 1D SH23 in a bounded domain

	Inputs:
	domain (dedalus object)

	Returns:
	Dedalus object to solve the SH23 equation

	"""

	# Dedalus Libraries
	import dedalus.public as de

	# Problem
	problem = de.IVP(domain, variables=['u', 'uz', 'uzz','uzzz'], time='t')
	problem.parameters['a'] = -0.1;
	problem.parameters['inv_Vol'] = 1./domain.hypervolume;

	problem.add_equation("dt(u) + (1-a)*u + 2*uzz + dz(uzzz) = 2*(u**2) - u**3");
	problem.add_equation("uz   - dz(u)   = 0");
	problem.add_equation("uzz  - dz(uz)  = 0");
	problem.add_equation("uzzz - dz(uzz) = 0");


	problem.add_bc("left(uz)   = 0");
	problem.add_bc("left(uzzz) = 0");

	problem.add_bc("right(u)   = 0");
	problem.add_bc("right(uzz) = 0");

	# Build solver
	return problem.build_solver(de.timesteppers.SBDF1);

def FWD_Solve_IVP_PREP(X_k, domain, dt=1e-02,  N_ITERS=100):

	"""
	Integrates the initial condition X(t=0) = Ux0 -> U(x,T);
	to clean the data and satisfy the bcs

	Input:
	X_k - list of a numpy vector initial condition in grid-space
	domain

	Returns:
		X_k - smoothed initial condition in grid-space

	"""
	from dedalus.extras import flow_tools
	from dedalus.tools  import post
	import dedalus.public as de

	#######################################################
	# initialize the problem
	#######################################################

	X0_field = X_k[0];
	IVP_FWD  = FWD_Solve_Build_Lin(domain);

	u = IVP_FWD.state['u'];
	u.set_scales(domain.dealias, keep_data=False)
	u['g'] = X0_field['g']

	#######################################################
	# evolution parameters
	######################################################

	IVP_FWD.stop_iteration = np.inf
	IVP_FWD.stop_iteration = N_ITERS+1; # Total Foward Iters + 1, to grab last point

	while IVP_FWD.ok:
		IVP_FWD.step(dt);

	return u;

def FWD_Solve_IVP_Cnts(X_k, domain, X_FWD_DICT, N_ITERS, dt =1e-02,filename=None):

	"""
	Integrates the initial condition X(t=0) = Ux0 -> U(x,T);
	using the SH23 equation,

	Input:
	X_k - list of a numpy vector initial condition in grid-space
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
		h.setLevel("WARNING"); #
		#h.setLevel("INFO"); #h.setLevel("DEBUG")
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

	Vec_to_Field(domain,u,X_k[0])

	if filename != None:
		IVP_FWD.load_state(filename,index=-1)

	#######################################################
	# evolution parameters
	######################################################

	IVP_FWD.stop_iteration = np.inf
	IVP_FWD.stop_iteration = N_ITERS+1; # Total Foward Iters + 1, to grab last point

	IVP_FWD.sim_tim   = IVP_FWD.initial_sim_time = 0.
	IVP_FWD.iteration = IVP_FWD.initial_iteration = 0


	#######################################################
	# analysis tasks
	#######################################################
	
	analysis_CPT = IVP_FWD.evaluator.add_file_handler('CheckPoints', iter=N_ITERS, mode='overwrite');
	analysis_CPT.add_system(IVP_FWD.state, layout='g', scales=3/2);
	analysis_CPT.add_task("u", name="u_hat", layout='c', scales=1);
	analysis_CPT.add_task("u**2", name="KE_hat", layout='c', scales=1);

	analysis1 = IVP_FWD.evaluator.add_file_handler("scalar_data", iter=20, mode='overwrite');
	analysis1.add_task("inv_Vol*integ(u**2)", name="Kinetic energy")


	#######################################################
	logger.info("\n\n --> Timestepping FWD_Solve ");
	#######################################################

	N_PRINTS = N_ITERS//2;

	flow = flow_tools.GlobalFlowProperty(IVP_FWD, cadence=1);
	flow.add_property("inv_Vol*integ( u**2 )", name='J(u)');

	J_TRAP = 0.;
	snapshot_index = 0;
	N_SUB_ITERS = N_ITERS;
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

		# Simple Euler integration 1st order dt
		if ((IVP_FWD.iteration-1) >= 0) and ((IVP_FWD.iteration-1) <= N_ITERS):
			J_TRAP += dt*flow.volume_average('J(u)');
	#######################################################

	# final statistics
	post.merge_process_files("CheckPoints", cleanup=True, comm=MPI.COMM_WORLD);
	post.merge_process_files("scalar_data", cleanup=True, comm=MPI.COMM_WORLD);
	time.sleep(.1);
	logger.info("\n\n--> Complete <--\n")

	logger.info('J(u) = %e'%J_TRAP );

	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");

	return (-1.)*J_TRAP;

def FWD_Solve_IVP_Discrete(X_k,domain, X_FWD_DICT, N_ITERS, dt =1e-02,filename=None):
	
	"""
	Integrates the initial condition X(t=0) = Ux0 -> U(x,T);
	using the SH23 equation,

	Input:
	X_k - list of a numpy vector initial condition in grid-space
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
		h.setLevel("WARNING"); #
		#h.setLevel("INFO"); #h.setLevel("DEBUG")
	logger = logging.getLogger(__name__)

	#######################################################
	# initialize the problem
	#######################################################

	# Linear Part MX_t = LX
	problem = de.LBVP(domain, variables=['u', 'uz', 'uzz','uzzz'])
	problem.parameters['a']  = -0.1;
	problem.parameters['inv_Vol'] = 1./domain.hypervolume;
	problem.parameters['dt'] = dt
	problem.add_equation("u/dt + (1-a)*u + 2*uzz + dz(uzzz) = 0")
	problem.add_equation("uz   - dz(u)   = 0");
	problem.add_equation("uzz  - dz(uz)  = 0");
	problem.add_equation("uzzz - dz(uzz) = 0");

	problem.add_bc("left(uz)   = 0");
	problem.add_bc("left(uzzz) = 0");

	problem.add_bc("right(u)   = 0");
	problem.add_bc("right(uzz) = 0");

	# Non-linear part F(x)
	def NLterm(vec):
		a = transformInverse(vec)
		a = 2.*(a**2) - a**3
		a = transform(a)
		# Dealias
		a[int(len(a)//2):] = 0
		return a;

	IVP_FWD = problem.build_solver();
	##########################################################

	rhsD   = field.Field(domain, name='rhsD')
	rhsD1  = field.Field(domain, name='rhsD1')
	rhsD2  = field.Field(domain, name='rhsD2')
	rhsD3  = field.Field(domain, name='rhsD3')
	rhsD4  = field.Field(domain, name='rhsD4')
	rhsD5  = field.Field(domain, name='rhsD5')
	rhsD6  = field.Field(domain, name='rhsD6')
	rhsD7  = field.Field(domain, name='rhsD7')
	fields = [rhsD,rhsD1,rhsD2,rhsD3,rhsD4,rhsD5,rhsD6,rhsD7]
	equ_rhs = system.FieldSystem(fields)

	#######################################################
	# Analysis tasks
	#######################################################
	file = h5py.File('scalar_data_s1.h5', 'w');

	scalars_tasks  = file.create_group('tasks');
	scalars_scales = file.create_group('scales');
	sim_time = [];
	Kinetic_energy = [];

	file = h5py.File('CheckPoints_s1.h5', 'w');

	CheckPt_tasks  = file.create_group('tasks');
	CheckPt_scales = file.create_group('scales');
	z_save = CheckPt_scales.create_group('z');
	z_save['1.5'] = domain.grid(0, scales=1);
	u_save = np.zeros( (2,IVP_FWD.state['u']['g'].shape[0]) );

	#######################################################
	# evolution parameters
	######################################################

	# Set the initial condition
	IVP_FWD.state['u']['c'] = transform(X_k[0])

	snapshot_index = 0
	for i in range(N_ITERS):
		
		# 1) Pass data to h5 file
		#~~~~~~~~~~~	
		if i == 0:
			u_save[0,:] = IVP_FWD.state['u']['g'];

			X_FWD_DICT['A_fwd'][:,snapshot_index] = IVP_FWD.state['u']['g'][:];
			snapshot_index+=1;
			cost = Inner_Prod(IVP_FWD.state['u']['g'],IVP_FWD.state['u']['g'],domain)*dt
		elif i == (N_ITERS-1):
			
			u_save[1,:] = IVP_FWD.state['u']['g'];	

		Kinetic_energy.append( Inner_Prod(IVP_FWD.state['u']['g'],IVP_FWD.state['u']['g'],domain) );
		sim_time.append(i*dt);	
		#~~~~~~~~~~~	

		# 2)   Form RHS of eqn (A.2) 
		# i.e.  B =        c_1*F(X^{i-1})           - (a_1*M + b_1*L)*X^{i-1}
		rhsD['c'] = NLterm(IVP_FWD.state['u']['c']) +   IVP_FWD.state['u']['c']/dt 
		
		######################## Solve the LBVP ########################
		## P^L(a_1*M + b_1*L)*(P^R)*Y^i = P^L*B
		## where Y^i = P^{-R}*X^i
		equ_rhs.gather()
		for p in IVP_FWD.pencils:
			b = p.pre_left @ equ_rhs.get_pencil(p)     # Left pre-condition RHS P^L*B
			x = IVP_FWD.pencil_matsolvers[p].solve(b)  # Invert for Y^i
			if p.pre_right is not None:
				x = p.pre_right @ x
			IVP_FWD.state.set_pencil(p, x)			   # Recover X^i = P^R*Y^i
			IVP_FWD.state.scatter()
		################################################################
		
		# 3) Compute the objective function
		X_FWD_DICT['A_fwd'][:,snapshot_index] = IVP_FWD.state['u']['g'][:];
		snapshot_index+=1;
		cost += Inner_Prod(IVP_FWD.state['u']['g'],IVP_FWD.state['u']['g'],domain)*dt
	
	# Save the files
	scalars_tasks['Kinetic energy']  = Kinetic_energy
	scalars_scales['sim_time'] = sim_time
	CheckPt_tasks['u']  = u_save;
	file.close(); 

	# Set to info level rather than the debug default
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");

	return (-1.)*cost;


##########################################################################
# ~~~~~ ADJ Solvers + Comptability Condition ~~~~~~~~~~~~~
##########################################################################

def ADJ_Solve_IVP_Cnts(X_k, domain,  X_FWD_DICT, N_ITERS, dt=1e-02, filename=None):
	"""
	Solve the adjoint of the SH23 equation in a bounded domain:

	Input:
	domain (dedalus object)
	dt - float numerical integration time-step
	N_ITERS,N_SUB_ITERS - int number of iterations to complete
	X_FWD_DICT - fWD checkpoints buffer
	
	Returns:
		[dJ/du0] gradient of the time integrated kinetic energy
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

	logger.info("--> Adding Equations");

	problem = de.IVP(domain, variables=['q', 'qz', 'qzz','qzzz'], time='t')
	problem.parameters['a'] = -0.1;
	problem.parameters['inv_Vol'] = 1./domain.hypervolume;

	u_f = domain.new_field(); #new_ncc(domain);
	problem.parameters['uf'] = u_f

	problem.add_equation("dt(q) + (1-a)*q + 2*qzz + dz(qzzz) = (4*uf - 3.*(uf**2) )*q - 2.*uf");
	problem.add_equation("qz   - dz(q)   = 0");
	problem.add_equation("qzz  - dz(qz)  = 0");
	problem.add_equation("qzzz - dz(qzz) = 0");

	problem.add_bc("left(qz)   = 0");
	problem.add_bc("left(qzzz) = 0");

	problem.add_bc("right(q)   = 0");
	problem.add_bc("right(qzz) = 0");

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	IVP_ADJ = problem.build_solver(de.timesteppers.SBDF1);

	q = IVP_ADJ.state['q'];
	q.set_scales(domain.dealias, keep_data=False)
	q['g'] = 0.

	#######################################################
	# set initial conditions
	#######################################################

	#X_FWD_DICT = {'A_fwd':A_SNAPS}
	snapshot_index = -1; # Continuous

	#######################################################
	# evolution parameters
	######################################################

	#IVP_ADJ.stop_sim_time = tstop;
	#IVP_ADJ.stop_wall_time = tstop_wall*3600.
	IVP_ADJ.stop_iteration = np.inf
	IVP_ADJ.stop_iteration = N_ITERS; #+1; # Total Foward Iters + 1, to grab last point, /!\ only if checkpointing

	IVP_ADJ.sim_tim   = IVP_ADJ.initial_sim_time  = 0.
	IVP_ADJ.iteration = IVP_ADJ.initial_iteration = 0.

	#######################################################
	logger.info("\n\n --> Timestepping ADJ_Solve ");
	#######################################################

	N_PRINTS = N_ITERS//10;
	from dedalus.extras import flow_tools
	flow = flow_tools.GlobalFlowProperty(IVP_ADJ, cadence=1);
	flow.add_property("inv_Vol*integ( q**2  )", name='J(q)');

	while IVP_ADJ.ok:

		# Must leave this as it's needed for the dL/dU-gradient
		#X_FWD_DICT = {'A_fwd':A_SNAPS}
		IVP_ADJ.problem.namespace['uf']['c'] = X_FWD_DICT['A_fwd'][:,snapshot_index]
		snapshot_index-=1; # Move back in time <-t

		IVP_ADJ.step(dt);

		if IVP_ADJ.iteration % N_PRINTS == 0:
			logger.info('Iterations: %i' %IVP_ADJ.iteration)
			logger.info('Sim time: %f, Time-step dt: %f' %(IVP_ADJ.sim_time,dt));
			logger.info('KE = %e'%( flow.volume_average('J(q)') 	) );

	#######################################################

	# For discrete adjoint undo LHS inversion of the last-step
	Ux0 = Field_to_Vec(domain,q )

	logger.info("\n\n--> Complete <--\n")

	# Set to info level rather than the debug default
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");

	return [Ux0];

def ADJ_Solve_IVP_Discrete(X_k, domain,  X_FWD_DICT, N_ITERS, dt=1e-02, filename=None):

	"""
	Solve the adjoint of the SH23 equation in a bounded domain:

	Input:
	domain (dedalus object)
	dt - float numerical integration time-step
	N_ITERS,N_SUB_ITERS - int number of iterations to complete
	X_FWD_DICT - fWD checkpoints buffer
	
	Returns:
		[dJ/du0] gradient of the time integrated kinetic energy
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

	# Linear Part MX_t = LX
	problem = de.LBVP(domain, variables=['u', 'uz', 'uzz','uzzz'])
	problem.parameters['a']  = -0.1;
	problem.parameters['dt'] = dt
	problem.add_equation("u/dt + (1-a)*u + 2*uzz + dz(uzzz) = 0")
	problem.add_equation("uz   - dz(u)   = 0");
	problem.add_equation("uzz  - dz(uz)  = 0");
	problem.add_equation("uzzz - dz(uzz) = 0");

	problem.add_bc("left(uz)   = 0");
	problem.add_bc("left(uzzz) = 0");

	problem.add_bc("right(u)   = 0");
	problem.add_bc("right(uzz) = 0");

	# Adjoint of Jacobian (∂F/∂X)^H * q
	def NLtermAdj(vec,vecb):
		a = vec.copy()
		# Dealias
		a[int(len(a)//2):] = 0
		a = transformAdjoint(vec)
		a = (4.*vecb - 3.*(vecb**2))*a
		a = transformInverseAdjoint(a)
		return a

	
	IVP_ADJ = problem.build_solver();

	############### Build the adjoint matrices ###############
	# # Adjoint of LHS Matrix 
	# L^H = [P^L*(a_0*M + b_0*L)*P^R]^H
	# #
	IVP_ADJ.pencil_matsolvers_transposed = {}
	for p in IVP_ADJ.pencils:
		IVP_ADJ.pencil_matsolvers_transposed[p] = IVP_ADJ.matsolver(np.conj(p.L_exp).T, IVP_ADJ)
	##########################################################
	

	uadj  = field.Field(domain, name='uadj')
	uzadj  = field.Field(domain, name='uzadj')
	uzzadj  = field.Field(domain, name='uzzadj')
	uzzzadj  = field.Field(domain, name='uzzzadj')
	fields    = [uadj,uzadj ,uzzadj ,uzzzadj]
	state_adj = system.FieldSystem(fields)

	rhsA   = field.Field(domain, name='rhsA')
	rhsA1  = field.Field(domain, name='rhsA1')
	rhsA2  = field.Field(domain, name='rhsA2')
	rhsA3  = field.Field(domain, name='rhsA3')
	rhsA4  = field.Field(domain, name='rhsA4')
	rhsA5  = field.Field(domain, name='rhsA5')
	rhsA6  = field.Field(domain, name='rhsA6')
	rhsA7  = field.Field(domain, name='rhsA7')
	fields = [rhsA,rhsA1,rhsA2,rhsA3,rhsA4,rhsA5,rhsA6,rhsA7]
	equ_adj = system.FieldSystem(fields)
		

	# Set the initial conditions/compatibility condition

	W = weightMatrixDisc(domain); # Inner product weight matrix
	
	base_state = X_FWD_DICT['A_fwd'][:,-1]
	uadj['c']  = transformInverseAdjoint(X_k[0]*0) + 2*transformInverseAdjoint(W*base_state)*dt; # Form RHS of (1.4a)
	
	for i in range(N_ITERS):
		
		######################## Solve the adjoint LBVP ########################
		## L^H*Y^i = (P^R)^H*(∂ƒ^H/∂x)*(P^L)^H*q^i
		## where Y^i = P^{-R}*q^i
		state_adj.gather()
		# Solve system for each pencil, updating state
		for p in IVP_ADJ.pencils:
			if p.pre_right is not None:								# Left pre-condition RHS by (P^R)^H
				vec = state_adj.get_pencil(p)
				b = np.conj(p.pre_right).T @ vec
			else:
				b = state_adj.get_pencil(p)
			x = IVP_ADJ.pencil_matsolvers_transposed[p].solve(b); 	# Invert for Y^i
			x = np.conj(p.pre_left).T @ x;							# Recover q^i = (P^L)^H*Y^i
			equ_adj.set_pencil(p, x)
			equ_adj.scatter()
		#########################################################################
		

		base_state = X_FWD_DICT['A_fwd'][:,-i-2];
		uadj['c']  = rhsA['c'].copy();
		uadj['c']  = uadj['c']/dt + NLtermAdj(uadj['c'],base_state) + 2*transformInverseAdjoint(W*base_state)*dt; # Form RHS of (1.4b-c)
	
	Ux0 = transformAdjoint(uadj['c']);

	# Set to info level rather than the debug default
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");

	return [-Ux0/W];

def File_Manips(k):

	"""
	Takes files generated by the adjoint loop solve, orgainises them into respective directories
	and removes unwanted data

	Each iteration of the DAL loop code is stored under index k
	"""

	import shutil

	if(Adjoint_type == "Continuous"):
		# A) Contains all scalar data
		shutil.copyfile('scalar_data/scalar_data_s1.h5','scalar_data_iter_%i.h5'%k);

		# B) Contains all state data
		shutil.copyfile('CheckPoints/CheckPoints_s1.h5','CheckPoints_iter_%i.h5'%k);

	
	else:	
		# A) Contains all scalar data
		shutil.copyfile('scalar_data_s1.h5','scalar_data_iter_%i.h5'%k);

		# B) Contains all state data
		shutil.copyfile('CheckPoints_s1.h5','CheckPoints_iter_%i.h5'%k);	

	return None;


Adjoint_type = "Discrete";
#Adjoint_type = "Continuous";

if Adjoint_type == "Discrete":

	# Inner_Prod = Inner_Prod_Discrete
	Inner_Prod = Inner_Prod_Discrete
	FWD_Solve  = FWD_Solve_IVP_Discrete
	ADJ_Solve  = ADJ_Solve_IVP_Discrete

elif Adjoint_type == "Continuous":

	Inner_Prod = Inner_Prod_Cnts
	FWD_Solve  = FWD_Solve_IVP_Cnts
	ADJ_Solve  = ADJ_Solve_IVP_Cnts;

if __name__ == "__main__":

	dt   = 0.01;
	Npts = 256;
	L_z  = 40.0;
	M_0  = 0.0019;

	if Adjoint_type == "Discrete":
		dealias = 2
		Npts *= dealias

	Z_Domain = (-L_z/2.,L_z/2.);
	N_ITERS  = int(20./dt);

	domain, X0  = Generate_IC(Npts,Z_Domain,M_0);
	X_FWD_DICT  = GEN_BUFFER(Npts, domain, N_ITERS)

	args_IP = (domain,'np_vector');
	args_f  = (domain, X_FWD_DICT, N_ITERS);

	sys.path.insert(0,'../../../')	
	
	# 1) Test the Gradient
	from TestGrad import Adjoint_Gradient_Test
	_, dX0  = Generate_IC(Npts,Z_Domain,M_0);
	Adjoint_Gradient_Test([X0],[dX0], FWD_Solve,ADJ_Solve,Inner_Prod,args_f,args_IP,epsilon=1e-04)


	# 2) Call the optimisation
	from Sphere_Grad_Descent import Optimise_On_Multi_Sphere, plot_optimisation
	RESIDUAL, FUNCT, X_opt = Optimise_On_Multi_Sphere([X0],[M_0],FWD_Solve,ADJ_Solve,Inner_Prod,args_f,args_IP,err_tol = 1e-05,max_iters=50,LS = 'LS_wolfe', CG = True,callback=File_Manips)
	plot_optimisation(RESIDUAL,FUNCT);
