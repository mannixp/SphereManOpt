import sys,os;
os.environ["OMP_NUM_THREADS"] = "1" # Improves performance apparently ????

from mpi4py import MPI # Import this before numpy
import numpy as np
import h5py,logging

import dedalus.public as de
from dedalus.core import field
from dedalus.core import system
from scipy.fftpack import fft, dct

ts = de.timesteppers.SBDF1;
#ts = de.timesteppers.MCNAB2
#ts = de.timesteppers.RK222

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

##########################################################################
# ~~~~~ Direct and adjoint transforms ~~~~~~~~~~~~~
##########################################################################

def transform(x):
    ag  = field.Field(domain, name='ag') # tmp field for transforms
    ag['g'] = x
    b = ag[domain.dist.layouts[1]] ##FFT
    b = dct(b,type=2,axis=1)/b.shape[-1]
    b[:,0] *= 0.5
    b[:,1::2] *= -1
    return b

def transformAdjoint(x):
    ag  = field.Field(domain, name='ag') # tmp field for transforms
    c = x.copy()
    c[:,0] *= 0.5
    c[:,1::2] *= -1
    c[:,0]  *= np.sqrt((4*c.shape[-1]))
    c[:,1:] *= np.sqrt((2*c.shape[-1]))
    b = dct(c,type=3,norm='ortho',axis=1)/c.shape[-1]

    ag[domain.dist.layouts[1]] = b
    b = ag['g']/Nx

    return b

def transformInverse(x):
    ag  = field.Field(domain, name='ag') # tmp field for transforms
    c = x.copy()
    c[:,1::2] *= -1
    c[:,1:] *= 0.5
    b = dct(c,type=3,axis=1)

    ag[domain.dist.layouts[1]] = b
    b = ag['g']
    return b

def transformInverseAdjoint(x):
    ag  = field.Field(domain, name='ag') # tmp field for transforms
    ag['g'] = x

    b = ag[domain.dist.layouts[1]]*Nx

    b = dct(b,type=2,norm='ortho',axis=1)*np.sqrt(Nz)
    b[:,1:] *= np.sqrt(2)
    b[:,1:] *= 0.5
    b[:,1::2] *= -1

    return b

def weightMatrixDisc(domain):

	x = domain.grid(0, scales=1);
	y = domain.grid(1, scales=1);



	NxL = x.shape[0]
	NyL = y.shape[1]

	W = np.zeros((NxL,NyL))
	
	# Mult by dy
	for i in range(NyL):
		if(i==0):
			W[:,i] = y[0,i+1]-y[0,i]
		else:
			W[:,i] = y[0,i]-y[0,i-1]

	dx = x[1,0] - x[0,0]

	W = W*dx;
	#M = np.sqrt(W);

	return W; #,M;


##########################################################################
# ~~~~~ General Routines ~~~~~~~~~~~~~
##########################################################################

def filter_field(field,frac=0.25):

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

def Field_to_Vec(domain,Fx,Fz):

	"""
	Convert from: field to numpy 1D vector-

	Takes the LOCAL portions of:
	Inputs:
	- GLOBALLY distributed fields Fx,Fz
	- domain object

	Creates locally available numpy arrays of a global size &
	makes this array is available on all cores using MPI gather_all function

	This function assumes all arrays can fit in memory!!!

	Returns:
	- 1D np.array of (Fx,Fz)
	"""

	# 1) Create local array of the necessary dimension
	#lshape = domain.dist.grid_layout.local_shape(scales=domain.dealias)
	gshape = tuple( domain.dist.grid_layout.global_shape(scales=domain.dealias) );

	for f in [Fx,Fz]:
		f.set_scales(domain.dealias,keep_data=True);

	# 2) Gather all data onto EVERY processor!!
	Fx_global = MPI.COMM_WORLD.allgather(Fx['g']);
	Fz_global = MPI.COMM_WORLD.allgather(Fz['g']);

	# Gathered slices
	G_slices = MPI.COMM_WORLD.allgather( domain.dist.grid_layout.slices(scales=domain.dealias) )

	# 3) Declared an array of GLOBAL shape on every proc
	FX = np.zeros(gshape);
	FZ = np.zeros(gshape);

	# 3) Declared an array of GLOBAL shape on every proc
	#FX = np.zeros(gshape,dtype=np.complex128);
	#FZ = np.zeros(gshape,dtype=np.complex128);

	# Parse distributed fields into arrays on every proc
	for i in range( MPI.COMM_WORLD.Get_size() ):
		FX[G_slices[i]] = np.real(Fx_global[i]);
		FZ[G_slices[i]] = np.real(Fz_global[i]);

	# 4) Merge them together at the end!
	return np.concatenate( (FX.flatten(),FZ.flatten()) );

def Vec_to_Field(domain,A,B,U0):

	"""
	Convert from: numpy 1D vector to field -
	Takes a 1D array U0 and distributes it into fields A,B on
	num_procs = MPI.COMM_WORLD.size

	Inputs:
	- domain object
	- GLOBALLY distributed dedalus fields A,B
	- U0 1D np.array

	Returns:
	- None
	"""

	# 1) Split the 1D array into 1D arrays A,B
	a1,a2 = np.split(U0,2); #Passed in dealiased scale
	gshape = tuple( domain.dist.grid_layout.global_shape(scales=domain.dealias) )
	slices = domain.dist.grid_layout.slices(scales=domain.dealias)
	#lshape = domain.dist.grid_layout.local_shape(scales=domain.dealias)
	#slices = domain.dist.grid_layout.slices(scales=domain.dealias)

	for f in [A,B]:
		f.set_scales(domain.dealias,keep_data=False);
		f['g']=0.

	# 2) Reshape and parse relevant portion into A,B
	A['g'] = a1.reshape( gshape )[slices]
	B['g'] = a2.reshape( gshape )[slices]
	return None;

def Integrate_Field(domain,F):

		"""
		Performs the Volume integral of a given field F as

		KE(t) = 1/V int_v F(x,z) dV, where F = u^2 + w^2 & dV = dx*dz

		where KE is Kinetic Enegry
		"""
		# Dedalus Libraries
		from dedalus.extras import flow_tools
		import dedalus.public as de

		# 1) Multiply by the integration weights r*dr*d_phi*dz for cylindrical domain
		flow_red = flow_tools.GlobalArrayReducer(MPI.COMM_WORLD);
		INT_ENERGY = de.operators.integrate( F ,'x','z');
		SUM = INT_ENERGY.evaluate();

		# Divide by volume size
		VOL = 1./domain.hypervolume;

		return VOL*flow_red.global_max(SUM['g']); # Using this as it's what flow-tools does for volume average

def Inner_Prod_Cnts(x,y,domain,rand_arg=None):

	# The line-search requires the IP
	# m = <\Nabla J^T, P_k >, where P_k = - \Nabla J^T
	# Must be evaluated using an integral consistent with our objective function
	# i.e. <,> = (1/V)int_v x*y dV
	# To do this we transform back to fields and integrate using a consistent routine

	dA = new_ncc(domain);
	dB = new_ncc(domain);
	Vec_to_Field(domain,dA,dB, x);

	du = new_ncc(domain);
	dv = new_ncc(domain);
	Vec_to_Field(domain,du,dv, y);

	return Integrate_Field(domain, (dA*du) + (dB*dv) );

# @ Calum add the discrete IP here
def Inner_Prod_Discrete(x,y,domain,W=None):


	# Filter the noise, modify this for less noise
	W = weightMatrixDisc(domain)

	dA = new_ncc(domain);
	dB = new_ncc(domain);
	Vec_to_Field(domain,dA,dB, x);

	du = new_ncc(domain);
	dv = new_ncc(domain);
	Vec_to_Field(domain,du,dv, y);

	inner_prod = ( np.vdot(dA['g'],W*du['g']) + np.vdot(dB['g'],W*dv['g']) );
	inner_prod = comm.allreduce(inner_prod,op=MPI.SUM)
	
	return inner_prod/domain.hypervolume;

# Works for U_vec,Uz_vec currently
def Generate_IC(Nx,Nz, X_domain=(0.,4.*np.pi),Z_domain=(-1.,1.), E_0=0.02,dealias_scale=3/2,W=None):
	"""
	Generate a domain object and initial conditions from which the optimisation can proceed

	Input:
	- Nx,Nz - integer resolution size
	- X_dom - interval/domain scale
	- Z_dom - interval/domain scale
	- B_0   - initial condition amplitude buoyancy
	- E_0	- initial condition amplitude velocity

	Returns:
	- domain object
	- initial cond U0 , as a vector U
	- initial cond Uz0, as a vector dU/dz
	"""

	import dedalus.public as de

	# Set to info level rather than the debug default
	root = logging.root
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO"); #
		#h.setLevel("DEBUG")
	logger = logging.getLogger(__name__)


	logger.info('\n\n Generating initial conditions ..... \n\n');

	# Part 1) Generate domain
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# dealias_scale = 3/2;
	x_basis = de.Fourier(  'x', Nx, interval=X_domain, dealias=dealias_scale); # x

	if(Adjoint_type=="Discrete"):
		y_basis     = de.Chebyshev('z', Nz, interval=(-1, 1) )
		domain  = de.Domain([x_basis, y_basis], grid_dtype=np.complex128);
	else:
		#zb1     = de.Chebyshev('z1', Nz//2, interval=(-1, 0.) )
		#zb2     = de.Chebyshev('z2', Nz//2, interval=(0., 1.) )
		#z_basis = de.Compound('z', (zb1,zb2), dealias=dealias_scale)

		z_basis = de.Chebyshev('z', Nz, interval=Z_domain, dealias=dealias_scale); # z
		domain  = de.Domain([x_basis, z_basis], grid_dtype=np.float64);

	# Part 2) Generate initial condition U0 = {u, w}
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	ψ = domain.new_field();
	ψ.set_scales(domain.dealias, keep_data=False)
	gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
	slices = domain.dist.grid_layout.slices(scales=domain.dealias)
	rand = np.random.RandomState(seed=42)
	noise = rand.standard_normal(gshape)[slices]; #Slicing globally generated noise here!!

	z = domain.grid(1,scales=domain.dealias);
	x = domain.grid(0,scales=domain.dealias);
	z_bot = domain.bases[1].interval[0];
	z_top = domain.bases[1].interval[1];
	ψ['g'] = noise; #( (z - z_bot)*(z - z_top) )*(np.sin(x)**2);#*noise; # Could scale this ??? #noise*
	filter_field(ψ) # Filter the noise, modify this for less noise

	U0  = Field_to_Vec(domain,ψ,ψ);
	'''
	u = domain.new_field(name='u'); uz = domain.new_field(name='uz');
	w = domain.new_field(name='w'); wz = domain.new_field(name='wz');

	ψ.differentiate('z',out=u); u.differentiate('z',out=uz);
	ψ.differentiate('x',out=w); w.differentiate('z',out=wz);
	u['g']  *= -1;

	U0  = Field_to_Vec(domain,u,w);
	'''

	# Create vector
	U0 = FWD_Solve_IVP_Prep(U0,domain);
	SUM = Inner_Prod(U0,U0,domain);
	logger.info('Pre-scale (1/V)<U,U> = %e'%SUM);

	U0  = np.sqrt(E_0/SUM)*U0;

	logger.info('Created a vector (U,Uz) \n\n');
	return domain, U0;

def GEN_BUFFER(Nx,Nz, domain, N_SUB_ITERS):

	"""
	Given the resolution and number of N_SUB_ITERS

	# 1) Estimate the memory usage per proc/core

	# 2) If possible create a memory buffer as follows:

	# The buffer created is a Python dictionary - X_FWD_DICT = {'b_x(x,t)','b_y(x,t)','b_z(x,t)'}
	# Chosen so as to allow pass by reference & thus avoids copying
	# All arrays to which these "keys" map must differ as we are otherwise mapping to the same memory!!

	Returns:
	Memory Buffer - Type Dict { 'Key':Value } where Value is 4D np.array

	"""

	################################################################################################################
	# B) Build memory buffer
	################################################################################################################

	# -Total  = (0.5 complex to real)*Npts*Npts*Npts*(3 fields)*(64 bits)*N_SUB_ITERS*(1.25e-10)/MPI.COMM_WORLD.Get_size()
	# -float64 = 64bits # Data-type used # -1 bit = 1.25e-10 GB
	Total  = ( 0.5*(Nx*Nz)*64*N_SUB_ITERS*(1.25e-10) )/float( MPI.COMM_WORLD.Get_size() )
	if MPI.COMM_WORLD.rank == 0:
		print("Total memory =%f GB, and memory/core = %f GB"%(MPI.COMM_WORLD.Get_size()*Total,Total));

	gshape  = tuple( domain.dist.coeff_layout.global_shape(scales=1) );
	lcshape = tuple( domain.dist.coeff_layout.local_shape(scales=1)  );
	SNAPS_SHAPE = (lcshape[0],lcshape[1],N_SUB_ITERS+1);

	u_SNAPS  = np.zeros(SNAPS_SHAPE,dtype=complex);
	w_SNAPS  = np.zeros(SNAPS_SHAPE,dtype=complex);
	b_SNAPS  = np.zeros(SNAPS_SHAPE,dtype=complex);

	return {'u_fwd':u_SNAPS,'w_fwd':w_SNAPS,'b_fwd':b_SNAPS};

##########################################################################
# ~~~~~ FWD Solvers ~~~~~~~~~~~~~
##########################################################################

def FWD_Solve_Build_Lin(domain, Reynolds, Richardson, Prandtl=1.,Sim_Type = "Non_Linear"):

	"""
	Driver program for RBC, which builds the forward solver object with options:

	Inputs:
	domain (dedalus object) returned by ??
	flow paramaters Rayleigh & Prandtl numbers

	Returns:
	Dedalus object to solve the 2D RBC in a slab geometry

	"""

	# Dedalus Libraries
	import dedalus.public as de

	# Set to info level rather than the debug default
	root = logging.root
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");
		#h.setLevel("DEBUG")
	logger = logging.getLogger(__name__)

	#######################################################
	# initialize the problem
	#######################################################

	problem = de.IVP(domain, variables=['p','b','u','w','bz','uz','wz'])
	problem.meta['p','bz','u','w']['z']['dirichlet'] = True
	problem.parameters['Re'] = Reynolds
	problem.parameters['Pe'] = Reynolds*Prandtl
	problem.parameters['Ri'] = Richardson
	problem.parameters['inv_Vol'] = 1./domain.hypervolume;

	z = domain.grid(1)
	U = domain.new_field()
	U.meta['x']['constant'] = True
	U['g'] = (1. - z**2);
	problem.parameters['U' ] = U

	z = domain.grid(1)
	Uz = domain.new_field()
	Uz.meta['x']['constant'] = True
	Uz['g'] = -2.*z;
	problem.parameters['Uz'] = Uz

	problem.substitutions['Omega'] = "dx(w) - uz";

	if Sim_Type == "Linear":

		problem.add_equation("dt(b) - (1./Pe)*(dx(dx(b)) + dz(bz))         + U*dx(b)        = 0.")
		problem.add_equation("dt(u) - (1./Re)*(dx(dx(u)) + dz(uz)) - dx(p) + U*dx(u) + w*Uz = 0.")
		problem.add_equation("dt(w) - (1./Re)*(dx(dx(w)) + dz(wz)) - dz(p) + U*dx(w) + b*Ri = 0.")

	else:
		problem.add_equation("dt(b) - (1./Pe)*(dx(dx(b)) + dz(bz))         + U*dx(b)        = -(u*dx(b) + w*bz)")
		problem.add_equation("dt(u) - (1./Re)*(dx(dx(u)) + dz(uz)) - dx(p) + U*dx(u) + w*Uz = -(u*dx(u) + w*uz)")
		problem.add_equation("dt(w) - (1./Re)*(dx(dx(w)) + dz(wz)) - dz(p) + U*dx(w) + b*Ri = -(u*dx(w) + w*wz)")

	problem.add_equation("dx(u) + wz = 0")
	problem.add_equation("bz - dz(b) = 0")
	problem.add_equation("uz - dz(u) = 0")
	problem.add_equation("wz - dz(w) = 0")

	# No-Flux
	problem.add_bc("left(bz)  = 0");
	problem.add_bc("right(bz) = 0");

	# No-Slip
	problem.add_bc("left(u)  = 0")
	problem.add_bc("left(w)  = 0")
	problem.add_bc("right(u) = 0")
	problem.add_bc("right(w) = 0", condition="(nx != 0)");
	#problem.add_bc("right(p) = 0", condition="(nx == 0)");
	problem.add_bc("integ(p,'z') = 0", condition="(nx == 0)")

	# Build solver
	solver = problem.build_solver(ts);
	logger.info('Solver built')

	# Set to info level rather than the debug default
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");

	return solver;

def FWD_Solve_IVP_Prep(U0, domain, Reynolds=500., Richardson=0.05, N_ITERS=100., dt=1e-04, Prandtl=1., δ  = 0.25):

	"""
	Integrates the initial conditions to satisfy bcs,

	Input:
	-initial condition Bx0 & derivative d/dz(Bx0)
	- domain object
	- default parameters Re,Ri,Pr,dt,N_ITERS

	Returns:
		field objects b,d/dz(b)

	"""
	from dedalus.extras import flow_tools
	from dedalus.tools  import post
	import dedalus.public as de

	# Set to info level rather than the debug default
	root = logging.root
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");
		#h.setLevel("DEBUG")
	logger = logging.getLogger(__name__)

	#######################################################
	# initialize the problem
	#######################################################
	IVP_FWD = FWD_Solve_Build_Lin(domain, Reynolds, Richardson, Prandtl, Sim_Type = "Linear");

	p = IVP_FWD.state['p'];
	b = IVP_FWD.state['b'];	bz = IVP_FWD.state['bz'];
	u = IVP_FWD.state['u']; uz = IVP_FWD.state['uz'];
	w = IVP_FWD.state['w']; wz = IVP_FWD.state['wz'];
	for f in [p, b,u,w, bz,uz,wz]:
		f.set_scales(domain.dealias, keep_data=False)
		f['g'] = 0.

	#######################################################
	# set initial conditions
	#######################################################
	Vec_to_Field(domain,u ,w ,U0 );
	#Vec_to_Field(domain,uz,wz,Uz0);

	from scipy.special import erf
	z       = domain.grid(1,scales=domain.dealias);
	b['g']  = -(1./2.)*erf(z/δ);
	bz['g'] = -np.exp(-(z/δ)**2)/(δ*np.sqrt(np.pi));

	#######################################################
	# evolution parameters
	######################################################

	IVP_FWD.stop_iteration = np.inf
	IVP_FWD.stop_iteration = N_ITERS+1; # Total Foward Iters + 1, to grab last point

	IVP_FWD.sim_tim   = IVP_FWD.initial_sim_time = 0.
	IVP_FWD.iteration = IVP_FWD.initial_iteration = 0

	#######################################################
	logger.info("\n\n --> Timestepping to prepare IC's for FWD_Solve ");
	#######################################################

	#N_PRINTS = N_ITERS//2;
	#flow = flow_tools.GlobalFlowProperty(IVP_FWD, cadence=1);
	#flow.add_property("inv_Vol*integ( u**2 + w**2 )", name='Kinetic' );
	#flow.add_property("inv_Vol*integ( b**2 	      )", name='buoyancy');

	while IVP_FWD.ok:

		IVP_FWD.step(dt);

		#logger.info('Iterations: %i' %IVP_FWD.iteration)
		#logger.info('Sim time:   %f' %IVP_FWD.sim_time )

		#logger.info('Kinetic  (1/V)<U,U> = %e'%flow.volume_average('Kinetic') );
		#logger.info('Buoynacy (1/V)<b,b> = %e'%flow.volume_average('buoyancy'));

	#######################################################

	logger.info("--> Complete <--\n\n")

	return Field_to_Vec(domain,u ,w );

def FWD_Solve_Cnts(    U0, domain, Reynolds, Richardson, N_ITERS, X_FWD_DICT,  dt=1e-04, α = 0, ß = 0, Prandtl=1., δ  = 0.25, filename=None):

	"""
	Integrates the initial conditions FWD N_ITERS using RBC code

	Input:
	- initial condition Bx0 & derivative d/dz(Bx0)
	- domain object
	- default parameters Ra,Pr,dt (float)
	- N_ITERS, N_SUB_ITERS (int)
	- X_FWD_DICT (python dictnary buffer to store values)
	- cost_function (str)

	Returns:
		objective function J(Bx0)

	- Writes the following to disk:
	1) FILE Scalar-data (every 10 iters): Kinetic Enegry, buoyancy etc

	2) FILE Checkpoints (every N_SUB_ITERS): full system state in grid space

	"""
	from dedalus.extras import flow_tools
	from dedalus.tools  import post
	import dedalus.public as de

	# Set to info level rather than the debug default
	root = logging.root
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");
		#h.setLevel("DEBUG")
	logger = logging.getLogger(__name__)

	#######################################################
	# initialize the problem
	#######################################################
	IVP_FWD = FWD_Solve_Build_Lin(domain, Reynolds, Richardson, Prandtl);

	p = IVP_FWD.state['p'];
	b = IVP_FWD.state['b'];	bz = IVP_FWD.state['bz'];
	u = IVP_FWD.state['u']; uz = IVP_FWD.state['uz'];
	w = IVP_FWD.state['w']; wz = IVP_FWD.state['wz'];

	for f in [p, b,u,w, bz,uz,wz]:
		f.set_scales(domain.dealias, keep_data=False)
		f['g'] = 0.

	#######################################################
	# set initial conditions
	#######################################################
	Vec_to_Field(domain,u ,w ,U0[0] );
	#Vec_to_Field(domain,uz,wz,Uz0);

	if filename != None:
		IVP_FWD.load_state(filename,index=0)

	from scipy.special import erf
	z       = domain.grid(1,scales=domain.dealias);
	b['g']  = -(z + (0.9*z)**3 + (0.9*z)**5 + (0.9*z)**7);  #-(1./2.)*erf(z/δ);
	#bz['g'] = -np.exp(-(z/δ)**2)/(δ*np.sqrt(np.pi));

	#kk = 5.
	#b['g']  = -(1./2.)*np.tanh(kk*z)
	#bz['g'] = -kk/(np.cosh(2*kk*z) + 1.)


	#######################################################
	# evolution parameters
	######################################################
	IVP_FWD.stop_iteration = N_ITERS+1; # Total Foward Iters + 1, to grab last point

	IVP_FWD.sim_tim   = IVP_FWD.initial_sim_time = 0.
	IVP_FWD.iteration = IVP_FWD.initial_iteration = 0

	#######################################################
	# analysis tasks
	#######################################################
	analysis_CPT = IVP_FWD.evaluator.add_file_handler('CheckPoints', iter=N_ITERS/10, mode='overwrite');
	analysis_CPT.add_system(IVP_FWD.state, layout='g', scales=3/2);

	analysis_CPT.add_task("Omega"							, layout='g', name="vorticity",scales=3/2);
	analysis_CPT.add_task("inv_Vol*integ( u**2 + w**2, 'z')", layout='c', name="kx Kinetic  energy");
	analysis_CPT.add_task("inv_Vol*integ( b**2		 , 'z')", layout='c', name="kx Buoyancy energy");

	analysis_CPT.add_task("inv_Vol*integ( u**2 + w**2, 'x')", layout='c', name="Tz Kinetic  energy");
	analysis_CPT.add_task("inv_Vol*integ( b**2		 , 'x')", layout='c', name="Tz Buoyancy energy");


	analysis1 	= IVP_FWD.evaluator.add_file_handler("scalar_data", iter=20, mode='overwrite');
	analysis1.add_task("inv_Vol*integ( u**2 + w**2 )", name="Kinetic  energy")
	analysis1.add_task("inv_Vol*integ( b**2 	   )", name="Buoyancy energy")

	#######################################################
	logger.info("\n\n --> Timestepping FWD_Solve ");
	#######################################################

	N_PRINTS = N_ITERS//2;
	if α == 0:
		flow = flow_tools.GlobalFlowProperty(IVP_FWD, cadence=1);
	else:
		flow = flow_tools.GlobalFlowProperty(IVP_FWD, cadence=N_PRINTS);
	flow.add_property("inv_Vol*integ( u**2 + w**2 )", name='Kinetic' );
	flow.add_property("inv_Vol*integ( b**2 	      )", name='buoyancy');


	J_TRAP = 0.; snapshot_index = 0;
	while IVP_FWD.ok:

		# 1) Fill Dictionary
		#X_FWD_DICT = {'u_fwd':u_SNAPS,'w_fwd':w_SNAPS,'b_fwd':b_SNAPS};
		X_FWD_DICT['u_fwd' ][:,:,snapshot_index] = u[ 'c'];
		X_FWD_DICT['w_fwd' ][:,:,snapshot_index] = w[ 'c'];
		X_FWD_DICT['b_fwd' ][:,:,snapshot_index] = b[ 'c'];
		snapshot_index+=1;

		IVP_FWD.step(dt);
		if IVP_FWD.iteration % N_PRINTS == 0:
			logger.info('Iterations: %i' %IVP_FWD.iteration)
			logger.info('Sim time:   %f' %IVP_FWD.sim_time )
			logger.info('Kinetic  (1/V)<U,U> = %e'%flow.volume_average('Kinetic') );
			logger.info('Buoynacy (1/V)<b,b> = %e'%flow.volume_average('buoyancy'));

		# 3) Evaluate Cost_function using flow tools,
		# flow tools value is that of ( IVP_FWD.iteration-1 )
		IVP_iter = IVP_FWD.iteration-1;
		if (IVP_iter >= 0) and (IVP_iter <= N_ITERS) and (α == 0): # J = int_t <B,B> dt
			J_TRAP +=    dt*flow.volume_average('Kinetic');

	# final statistics
	#######################################################
	post.merge_process_files("CheckPoints", cleanup=True, comm=MPI.COMM_WORLD);
	post.merge_process_files("scalar_data", cleanup=True, comm=MPI.COMM_WORLD);
	logger.info("\n\n--> Complete <--\n")


	if   α == 1:

		rho = domain.new_field();
		rho['c'] = X_FWD_DICT['b_fwd'][:,:,-1];
		if   ß == 1:

			#||∇^(−β) ρ(x,T) ||^2
			J_obj =  (1./2.)*Norm_and_Inverse_Second_Derivative(rho,domain)[0];

		elif ß == 0:

			#|| ρ(x,T) ||^2
			J_obj =  (1./2.)*Integrate_Field(domain,rho**2);

	elif α == 0:

		T = 1.;#N_ITERS*dt;
		J_obj = -(1./(2.*T))*J_TRAP; # Add a (-1) to maximise this

	logger.info('J(U) = %e'%J_obj);

	return J_obj;

# @ Calum add the discrete forward here - if neccessary ?
def FWD_Solve_Discrete(U0, domain, Reynolds, Richardson, N_ITERS, X_FWD_DICT,  dt=1e-04, α = 0, ß = 0, Prandtl=1., δ  = 0.25, filename=None):

	"""
	Integrates the initial conditions FWD N_ITERS using RBC code

	Input:
	- initial condition U0 type list U0[0] = [u,v] 
	- domain object
	- parameters Reynolds,Richardson (float)
	- N_ITERS (int)
	- X_FWD_DICT (python dictnary buffer to store values)
	- default parameters 

	Returns:
		objective function J(U0)

	- Writes the following to disk:
	1) FILE Scalar-data (every iters): Kinetic Enegry, buoyancy etc

	2) FILE Checkpoints : full system state in grid space of IC and X(t=T)
	"""


	# Set to info level rather than the debug default
	root = logging.root
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO"); #h.setLevel("DEBUG")
	logger = logging.getLogger(__name__)


	# Time-stepping by (1) creating the Linear boundary value problem (LBVP)
	# i.e. [ P^L*(a_0*M + b_0*L)*P^R ]*(P^{-R}*X^n) = P^L*F
	# 					           A  *     Y^n     = B
	# used to form the matrix A

	Re = Reynolds
	Pe = Reynolds*Prandtl
	Ri = Richardson

	problem = de.LBVP(domain, variables=['u','v','ρ',	'uz','vz','ρz',		'p'])
	problem.parameters['dt'] = dt
	problem.parameters['ReInv'] = 1./Re
	problem.parameters['Ri'] = Ri
	problem.parameters['PeInv'] = 1./Pe

	problem.add_equation("u/dt - ReInv*(dx(dx(u)) + dz(uz)) + dx(p) + (1. - z*z)*dx(u) + v*(-2.*z) = 0.")
	problem.add_equation("v/dt - ReInv*(dx(dx(v)) + dz(vz)) + dz(p) + (1. - z*z)*dx(v) + ρ*Ri      = 0.")
	problem.add_equation("ρ/dt - PeInv*(dx(dx(ρ)) + dz(ρz))         + (1. - z*z)*dx(ρ)             = 0.")

	problem.add_equation("dx(u) + vz = 0")
	problem.add_equation("uz - dz(u) = 0");
	problem.add_equation("vz - dz(v) = 0");
	problem.add_equation("ρz - dz(ρ) = 0");

	problem.add_bc("left(u) = 0");
	problem.add_bc("left(v) = 0");

	problem.add_bc("right(u) = 0");
	problem.add_bc("right(v) = 0",condition="(nx != 0)")
	problem.add_bc("integ(p,'z') = 0", condition="(nx == 0)")

	problem.add_bc("left( ρz) = 0");
	problem.add_bc("right(ρz) = 0");

	solver = problem.build_solver()
	
	############### (1.b) Build the adjoint matrices A^H ###############
	solver.pencil_matsolvers_transposed = {}
	for p in solver.pencils:
	    solver.pencil_matsolvers_transposed[p] = solver.matsolver(np.conj(p.L_exp).T, solver)
	##########################################################

	# (1.c) Allocate all Field variables = number of eqns + bcs 
	u  = solver.state['u']
	v  = solver.state['v']
	ρ  = solver.state['ρ']
	uz = solver.state['uz']
	vz = solver.state['vz']
	ρz = solver.state['ρz']
	#p  = solver.state['p']

	rhsU   = field.Field(domain, name='rhsU')
	rhsV   = field.Field(domain, name='rhsV')
	rhsρ   = field.Field(domain, name='rhsρ')
	rhsD4  = field.Field(domain, name='rhsD4')
	rhsD5  = field.Field(domain, name='rhsD5')
	rhsD6  = field.Field(domain, name='rhsD6')
	rhsD7  = field.Field(domain, name='rhsD7')
	rhsD8  = field.Field(domain, name='rhsD8')
	rhsD9  = field.Field(domain, name='rhsD9')
	rhsD10  = field.Field(domain, name='rhsD10')
	rhsD11  = field.Field(domain, name='rhsD11')
	rhsD12  = field.Field(domain, name='rhsD12')
	rhsD13  = field.Field(domain, name='rhsD13')
	rhsD14  = field.Field(domain, name='rhsD14')
	fields = [rhsU,rhsV,rhsρ,	rhsD4,rhsD5,rhsD6,rhsD7,rhsD8,rhsD9,rhsD10,rhsD11,rhsD12,rhsD13,rhsD14]
	equ_rhs = system.FieldSystem(fields)

	################################################################################

	# (2) Create the Linear boundary value problem 
	# i.e. [ P^L*∆*P^R ]*ψ = P^L*ρ
	# 			 L      *X = F
	# used to solve for the mix-norm.

	problemMN = de.LBVP(domain, variables=['ψ','ψz'])
	problemMN.add_equation("dx(dx(ψ)) + dz(ψz) = 0")
	problemMN.add_equation("ψz - dz(ψ)=0")
	problemMN.add_bc("left( ψ) = 0");
	problemMN.add_bc("right(ψ) = 0");

	solverMN = problemMN.build_solver()
	############### Build the adjoint matrices ###############
	solverMN.pencil_matsolvers_transposed = {}
	for p in solverMN.pencils:
	    solverMN.pencil_matsolvers_transposed[p] = solverMN.matsolver(np.conj(p.L_exp).T, solverMN)
	##########################################################

	MN1   = field.Field(domain, name='MN1')
	MN2   = field.Field(domain, name='MN2')
	MN3   = field.Field(domain, name='MN3')
	MN4   = field.Field(domain, name='MN4')
	fields = [MN1,MN2,MN3,MN4]
	MN_rhs = system.FieldSystem(fields)
	################################################################################

	# Create the de-aliaising matrix
	NxCL = u['c'].shape[0]
	NzCL = u['c'].shape[1]

	elements0 = domain.elements(0)
	elements1 = domain.elements(1)

	DA = np.zeros((NxCL,NzCL))
	Lx = abs(domain.bases[0].interval[0] - domain.bases[0].interval[1]);
	Nx0 = 2*Nx//3
	Nz0 = 2*Nz//3
	for i in range(NxCL):
		for j in range(NzCL):
			if(np.abs(elements0[i,0]) < (2.*np.pi/Lx)*(Nx0//2) and elements1[0,j] < Nz0):
				DA[i,j] = 1

	# Create an evaluator for the nonlinear terms			
	def NLterm(u,ux,uz,	v,vx,vz,	ρ,ρx,ρz):
		NLu = -transformInverse(u)*transformInverse(ux) - transformInverse(v)*transformInverse(uz)
		NLv = -transformInverse(u)*transformInverse(vx) - transformInverse(v)*transformInverse(vz)
		NLρ = -transformInverse(u)*transformInverse(ρx) - transformInverse(v)*transformInverse(ρz)
		return DA*transform(NLu),DA*transform(NLv),DA*transform(NLρ)

	# Function for taking derivatives in Fourier space	
	def derivativeX(vec):
		for i in range(vec.shape[0]):
			vec[i,:] *= elements0[i]*1j
		return vec

	# Prescribe the base state and set the ICs	
	from scipy import special
	z = domain.grid(1)
	ρ['g']  = -(z + (0.9*z)**3 + (0.9*z)**5 + (0.9*z)**7);  #-0.5*special.erf(z/δ)
	#ρz['g'] = -np.exp(-(z/δ)**2)/(δ*np.sqrt(np.pi));	
	ρ.differentiate(1, out=ρz)

	Vec_to_Field(domain,u ,v ,U0[0]);
	u.differentiate(1, out=uz)
	v.differentiate(1, out=vz)

	#######################################################
	# Analysis tasks
	#######################################################
	if MPI.COMM_WORLD.Get_rank() == 0:
		
		file1 		   = h5py.File('scalar_data_s1.h5', 'w');
		scalars_tasks  = file1.create_group('tasks');
		scalars_scales = file1.create_group('scales');
		
		file2 		   = h5py.File('CheckPoints_s1.h5', 'w');
		CheckPt_tasks  = file2.create_group('tasks');
		CheckPt_scales = file2.create_group('scales');
		
		x_save = CheckPt_scales.create_group('x');
		scales = domain.remedy_scales(scales=1)
		x_save['1.5'] = domain.bases[0].grid(scales[0]);
		
		z_save = CheckPt_scales.create_group('z');
		scales = domain.remedy_scales(scales=1)
		z_save['1.5'] = domain.bases[1].grid(scales[1]);

	sim_time = [];
	Kinetic_energy = [];
	Density_energy = [];	

	gshape  = tuple( domain.dist.grid_layout.global_shape(scales=1) );
	slices  = domain.dist.grid_layout.slices(scales=1)
	
	SHAPE  = (2,gshape[0],gshape[1])
	Ω_save = np.zeros( SHAPE );
	ρ_save = np.zeros( SHAPE );

	W = weightMatrixDisc(domain)

	# (3) Time-step the equations forwards T = N_ITERS*dt
	# performed by inverting a LVBP at each time-step
	snapshot_index = 0
	for i in range(N_ITERS):
		
		ux = derivativeX(u['c'].copy())
		vx = derivativeX(v['c'].copy())
		ρx = derivativeX(ρ['c'].copy())

		X_FWD_DICT['u_fwd'][:,:,snapshot_index] = u['c'].copy()
		X_FWD_DICT['w_fwd'][:,:,snapshot_index] = v['c'].copy()
		X_FWD_DICT['b_fwd'][:,:,snapshot_index] = ρ['c'].copy()
		snapshot_index+=1;


		#~~~~~~~~~~~
		KE_p = (np.vdot(u['g'],W*u['g']) + np.vdot(v['g'],W*v['g']) )/domain.hypervolume
		KE   = comm.allreduce(KE_p,op=MPI.SUM)
		
		DE_p = np.vdot(ρ['g'],W*ρ['g'])/domain.hypervolume
		DE   = comm.allreduce(DE_p,op=MPI.SUM)

		Kinetic_energy.append( KE );
		Density_energy.append( DE );
		sim_time.append(i*dt);

		if i == 0:
			
			Ω_save[0,:,:][slices] = np.real(transformInverse(vx) - uz['g']);
			ρ_save[0,:,:][slices] = np.real(ρ['g']);

			Ω_save[0,:,:]   = comm.allreduce(Ω_save[0,:,:],op=MPI.SUM)
			ρ_save[0,:,:]   = comm.allreduce(ρ_save[0,:,:],op=MPI.SUM)

		elif i == (N_ITERS-1):
			
			Ω_save[1,:,:][slices] = np.real(transformInverse(vx) - uz['g']);
			ρ_save[1,:,:][slices] = np.real(ρ['g']);

			Ω_save[1,:,:]   = comm.allreduce(Ω_save[1,:,:],op=MPI.SUM)
			ρ_save[1,:,:]   = comm.allreduce(ρ_save[1,:,:],op=MPI.SUM)
		#~~~~~~~~~~~

		NLu,NLv,NLrho = NLterm(u['c'],ux,uz['c'],	v['c'],vx,vz['c'],	ρ['c'],ρx,ρz['c'])
		rhsU['c'] = solver.state['u']['c']/dt + NLu
		rhsV['c'] = solver.state['v']['c']/dt + NLv
		rhsρ['c'] = solver.state['ρ']['c']/dt + NLrho
		
		######################## Solve the LBVP ########################
		equ_rhs.gather()
		for p in solver.pencils:
			b = p.pre_left @ equ_rhs.get_pencil(p)
			x = solver.pencil_matsolvers[p].solve(b)
			if p.pre_right is not None:
				x = p.pre_right @ x
			solver.state.set_pencil(p, x)
			solver.state.scatter()
		################################################################

	# Save the files
	if MPI.COMM_WORLD.Get_rank() == 0:

		scalars_tasks['Kinetic  energy']  = Kinetic_energy
		scalars_tasks['Buoyancy energy']  = Density_energy
		scalars_scales['sim_time'] = sim_time
		file1.close();

		CheckPt_tasks['vorticity']  = Ω_save;
		CheckPt_tasks['b']  = ρ_save;
		file2.close(); 	


	######################## (4) Solve the Mix Norm LBVP ########################
	ψ 		 = solverMN.state['ψ'];
	dρ_inv_dz= solverMN.state['ψz'];
	MN1['c'] = ρ['c'];

	MN_rhs.gather()
	for p in solverMN.pencils:
		b = p.pre_left @ MN_rhs.get_pencil(p)
		x = solverMN.pencil_matsolvers[p].solve(b)
		if p.pre_right is not None:
			x = p.pre_right @ x
		solverMN.state.set_pencil(p, x)
		solverMN.state.scatter()
	################################################################

	# (5) Evaluate the cost function and pass the adjoint equations
	# initial conditions into the checkpointing buffer

	dρ_inv_dx = field.Field(domain, name='dρ_inv_dx')
	ψ.differentiate('x', out=dρ_inv_dx);

	X_FWD_DICT['u_fwd'][:,:,snapshot_index] = dρ_inv_dx['c'].copy()
	X_FWD_DICT['w_fwd'][:,:,snapshot_index] = dρ_inv_dz['c'].copy()
	X_FWD_DICT['b_fwd'][:,:,snapshot_index] =  		  ψ['c'].copy()
	
	dρ_inv_dX  = Field_to_Vec(domain,dρ_inv_dx,dρ_inv_dz);
	cost       = (1./2)*Inner_Prod(dρ_inv_dX,dρ_inv_dX,domain);

	return cost;

##########################################################################
# ~~~~~ ADJ Solvers  ~~~~~~~~~~~~~
##########################################################################

def ADJ_Solve_Cnts(    U0, domain, Reynolds, Richardson, N_ITERS, X_FWD_DICT,  dt=1e-04, α = 0, ß = 0, Prandtl=1., δ  = 0.25, Sim_Type = "Non_Linear"):

	"""
	Driver program for Rayleigh Benard code, which builds the adjoint solver object with options:

	Inputs:
	-domain (dedalus object)
	-flow paramaters Rayleigh, Prandtl number (float)
	-dt (float)
	-N_ITERS,N_SUB_ITERS(int)
	- X_FWD_DICT (python dictnary buffer to store values)
	- cost_function (str)

	Returns:
		Gradient of the objective function dJ(Bx0)/dBx0
	"""

	# Dedalus Libraries
	import dedalus.public as de

	# Set to info level rather than the debug default
	root = logging.root
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO"); #h.setLevel("DEBUG")
	logger = logging.getLogger(__name__)

	#######################################################
	# initialize the problem
	#######################################################

	# 2D Boussinesq hydrodynamics
	problem = de.IVP(domain, variables=['p_adj','b_adj','u_adj','w_adj','bz_adj','uz_adj','wz_adj'])
	problem.meta['p_adj','bz_adj','u_adj','w_adj']['z']['dirichlet'] = True
	problem.parameters['Re'] = Reynolds
	problem.parameters['Pe'] = Reynolds*Prandtl
	problem.parameters['Ri'] = Richardson
	problem.parameters['inv_Vol'] = 1./domain.hypervolume;

	z = domain.grid(1)
	U = domain.new_field()
	U.meta['x']['constant'] = True
	U['g'] = (1. - z**2);
	problem.parameters['U' ] = U

	z = domain.grid(1)
	Uz = domain.new_field()
	Uz.meta['x']['constant'] = True
	Uz['g'] = -2.*z;
	problem.parameters['Uz'] = Uz


	uf = domain.new_field(); problem.parameters['uf' ] = uf
	wf = domain.new_field(); problem.parameters['wf' ] = wf
	bf = domain.new_field(); problem.parameters['bf' ] = bf

	if (α == 0):

		problem.add_equation("dt(b_adj) - (1./Pe)*(dx(dx(b_adj)) + dz(bz_adj))             - U*dx(b_adj) + Ri*w_adj  =  								 (uf*dx(b_adj) + wf*bz_adj)   				   ");
		problem.add_equation("dt(u_adj) - (1./Re)*(dx(dx(u_adj)) + dz(uz_adj)) - dx(p_adj) - U*dx(u_adj) 		     = -(u_adj*dx(uf) + w_adj*dx(wf) ) + (uf*dx(u_adj) + wf*uz_adj) - b_adj*dx(bf) - uf");
		problem.add_equation("dt(w_adj) - (1./Re)*(dx(dx(w_adj)) + dz(wz_adj)) - dz(p_adj) - U*dx(w_adj) + u_adj*Uz  = -(u_adj*dz(uf) + w_adj*dz(wf) ) + (uf*dx(w_adj) + wf*wz_adj) - b_adj*dz(bf) - wf");

	elif (α == 1):

		problem.add_equation("dt(b_adj) - (1./Pe)*(dx(dx(b_adj)) + dz(bz_adj))             - U*dx(b_adj) + Ri*w_adj  =  								 (uf*dx(b_adj) + wf*bz_adj)   			  ");
		problem.add_equation("dt(u_adj) - (1./Re)*(dx(dx(u_adj)) + dz(uz_adj)) - dx(p_adj) - U*dx(u_adj)             = -(u_adj*dx(uf) + w_adj*dx(wf) ) + (uf*dx(u_adj) + wf*uz_adj) - b_adj*dx(bf)");
		problem.add_equation("dt(w_adj) - (1./Re)*(dx(dx(w_adj)) + dz(wz_adj)) - dz(p_adj) - U*dx(w_adj) + u_adj*Uz  = -(u_adj*dz(uf) + w_adj*dz(wf) ) + (uf*dx(w_adj) + wf*wz_adj) - b_adj*dz(bf)");



	problem.add_equation("dx(u_adj) + wz_adj = 0")
	problem.add_equation("bz_adj - dz(b_adj) = 0")
	problem.add_equation("uz_adj - dz(u_adj) = 0")
	problem.add_equation("wz_adj - dz(w_adj) = 0")

	# No-Flux
	problem.add_bc("left( bz_adj) = 0");
	problem.add_bc("right(bz_adj) = 0");

	# No-Slip
	problem.add_bc("left(u_adj)  = 0")
	problem.add_bc("left(w_adj)  = 0")
	problem.add_bc("right(u_adj) = 0")
	problem.add_bc("right(w_adj) = 0", condition="(nx != 0)");
	#problem.add_bc("right(p_adj) = 0", condition="(nx == 0)");
	problem.add_bc("integ(p_adj,'z') = 0", condition="(nx == 0)")

	# Build solver
	IVP_ADJ = problem.build_solver(ts);
	logger.info('Solver built')

	p_adj = IVP_ADJ.state['p_adj'];
	b_adj = IVP_ADJ.state['b_adj']; bz_adj = IVP_ADJ.state['bz_adj'];
	u_adj = IVP_ADJ.state['u_adj']; uz_adj = IVP_ADJ.state['uz_adj'];
	w_adj = IVP_ADJ.state['w_adj']; wz_adj = IVP_ADJ.state['wz_adj'];

	for f in [p_adj, b_adj,u_adj,w_adj, bz_adj,uz_adj,wz_adj]:
		f.set_scales(domain.dealias, keep_data=False)
		f['g'] = 0.

	#######################################################
	# set initial conditions
	#######################################################

	if (α == 1):

		rho = domain.new_field();
		rho['c'] = X_FWD_DICT['b_fwd'][:,:,-1];

		if   (ß == 1):
			b_adj['c'] = (-1.)*Norm_and_Inverse_Second_Derivative(rho,domain)[1]['c'];
		elif (ß == 0):
			b_adj['c'] = rho['c'];

	#######################################################
	# evolution parameters
	######################################################

	IVP_ADJ.stop_iteration = N_ITERS; #+1; # Total Foward Iters + 1, to grab last point, /!\ only if checkpointing

	IVP_ADJ.sim_tim   = IVP_ADJ.initial_sim_time  = 0.
	IVP_ADJ.iteration = IVP_ADJ.initial_iteration = 0.

	#######################################################
	logger.info("\n\n --> Timestepping ADJ_Solve ");
	#######################################################

	#from dedalus.extras import flow_tools
	#N_PRINTS = N_SUB_ITERS//2;
	#flow = flow_tools.GlobalFlowProperty(IVP_ADJ, cadence=1);
	#flow.add_property("inv_Vol*integ( u†**2 + w†**2 )", name='Kinetic');
	#flow.add_property("inv_Vol*integ( b†**2 	      )", name='J(B)'   );

	for f in [uf,wf,bf]:
		f.set_scales(domain.dealias, keep_data=False);	#keep_data = False -> clears data out of the fields,

	snapshot_index = -1; # Continuous
	while IVP_ADJ.ok:

		# Must leave this as it's needed for the dL/dU-gradient
		#X_FWD_DICT = {'u_fwd':u_SNAPS, 'w_fwd':w_SNAPS}
		IVP_ADJ.problem.namespace['uf']['c'] = X_FWD_DICT['u_fwd'][:,:,snapshot_index]
		IVP_ADJ.problem.namespace['wf']['c'] = X_FWD_DICT['w_fwd'][:,:,snapshot_index]
		IVP_ADJ.problem.namespace['bf']['c'] = X_FWD_DICT['b_fwd'][:,:,snapshot_index]
		snapshot_index-=1; # Move back in time (<-t

		IVP_ADJ.step(dt);

	#######################################################

	logger.info("\n\n--> Complete <--\n")

	# Set to info level rather than the debug default
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");

	# Return the gradient δL_δu0
	return [ Field_to_Vec(domain,u_adj ,w_adj) ];

# @ Calum add the discrete forward here
def ADJ_Solve_Discrete(U0, domain, Reynolds, Richardson, N_ITERS, X_FWD_DICT,  dt=1e-04, α = 0, ß = 0, Prandtl=1., δ  = 0.25, Sim_Type = "Non_Linear"):

	# Set to info level rather than the debug default
	root = logging.root
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO"); #h.setLevel("DEBUG")
	logger = logging.getLogger(__name__)


	# Time-stepping by (1) creating the Linear boundary value problem (LBVP)
	# i.e. [ P^L*(a_0*M + b_0*L)*P^R ]*(P^{-R}*X^n) = P^L*F
	# 					           A  *     Y^n     = B
	# used to form the matrix A

	Re = Reynolds
	Pe = Reynolds*Prandtl
	Ri = Richardson

	problem = de.LBVP(domain, variables=['u','v','ρ','p','uz','vz','ρz'])
	problem.parameters['dt'] = dt
	problem.parameters['ReInv'] = 1./Re
	problem.parameters['Ri'] = Ri
	problem.parameters['PeInv'] = 1./Pe

	problem.add_equation("u/dt - ReInv*(dx(dx(u)) + dz(uz)) + dx(p) + (1. - z*z)*dx(u) + v*(-2.*z) = 0.")
	problem.add_equation("v/dt - ReInv*(dx(dx(v)) + dz(vz)) + dz(p) + (1. - z*z)*dx(v) + ρ*Ri      = 0.")
	problem.add_equation("ρ/dt - PeInv*(dx(dx(ρ)) + dz(ρz))         + (1. - z*z)*dx(ρ)             = 0.")

	problem.add_equation("dx(u) + vz = 0")
	problem.add_equation("uz - dz(u) = 0");
	problem.add_equation("vz - dz(v) = 0");
	problem.add_equation("ρz - dz(ρ) = 0");

	problem.add_bc("left(u) = 0");
	problem.add_bc("left(v) = 0");

	problem.add_bc("right(u) = 0");
	problem.add_bc("right(v) = 0",condition="(nx != 0)")
	problem.add_bc("integ(p,'z') = 0", condition="(nx == 0)")

	problem.add_bc("left( ρz) = 0");
	problem.add_bc("right(ρz) = 0");

	solver = problem.build_solver()
	############### (1.b) Build the adjoint matrices A^H ###############
	solver.pencil_matsolvers_transposed = {}
	for p in solver.pencils:
	    solver.pencil_matsolvers_transposed[p] = solver.matsolver(np.conj(p.L_exp).T, solver)
	##########################################################

	# (1.c) Allocate all fwd + adj Field variables = number of eqns + bcs 
	u  = solver.state['u']
	v  = solver.state['v']
	ρ  = solver.state['ρ']
	uz = solver.state['uz']
	vz = solver.state['vz']
	ρz = solver.state['ρz']


	uadj   = field.Field(domain, name='uadj')
	uzadj  = field.Field(domain, name='uzadj')
	vadj   = field.Field(domain, name='vadj')
	vzadj  = field.Field(domain, name='vzadj')
	padj   = field.Field(domain, name='padj')
	ρadj   = field.Field(domain, name='ρadj')
	ρzadj  = field.Field(domain, name='ρzadj')
	fields = [uadj,vadj,ρadj,padj,uzadj,vzadj,ρzadj]
	state_adj = system.FieldSystem(fields)

	rhsUA   = field.Field(domain, name='rhsUA')
	rhsVA   = field.Field(domain, name='rhsVA')
	rhsRhoA = field.Field(domain, name='rhsRhoA')
	rhsPA   = field.Field(domain, name='rhsPA')
	rhsuzA  = field.Field(domain, name='rhsuzA')
	rhsvzA  = field.Field(domain, name='rhsvzA')
	rhsRhozA= field.Field(domain, name='rhsRhozA')
	rhsA8   = field.Field(domain, name='rhsA8')
	rhsA9   = field.Field(domain, name='rhsA9')
	rhsA10  = field.Field(domain, name='rhsA10')
	rhsA11  = field.Field(domain, name='rhsA11')
	rhsA12  = field.Field(domain, name='rhsA12')
	rhsA13  = field.Field(domain, name='rhsA13')
	rhsA14  = field.Field(domain, name='rhsA14')
	fields  = [rhsUA,rhsVA,rhsRhoA,rhsPA,rhsuzA,rhsvzA,rhsRhozA,rhsA8,rhsA9,rhsA10,rhsA11,rhsA12,rhsA13,rhsA14  ]
	equ_adj = system.FieldSystem(fields)

	##########################################################


	# (2) Create the Linear boundary value problem 
	# i.e. [ P^L*∆*P^R ]*ψ = P^L*ρ
	# 			 L      *X = F
	# used to solve for the mix-norm.

	problemMN = de.LBVP(domain, variables=['ψ','ψz'])
	problemMN.add_equation("dx(dx(ψ)) + dz(ψz) = 0")
	problemMN.add_equation("ψz - dz(ψ)=0")
	problemMN.add_bc("left( ψ) = 0");
	problemMN.add_bc("right(ψ) = 0");

	solverMN = problemMN.build_solver()
	############### (2.b) Build the adjoint matrices L^H ###############
	solverMN.pencil_matsolvers_transposed = {}
	for p in solverMN.pencils:
	    solverMN.pencil_matsolvers_transposed[p] = solverMN.matsolver(np.conj(p.L_exp).T, solverMN)
	##########################################################

	# (2.c) Allocate all adj Field variables = number of eqns + bcs 
	MN1adj = field.Field(domain, name='MN1adj')
	MN2adj = field.Field(domain, name='MN2adj')
	fields = [MN1adj,MN2adj]
	MNadj_rhs = system.FieldSystem(fields)

	MN1L   = field.Field(domain, name='MN1L')
	MN2L   = field.Field(domain, name='MN2L')
	MN3L   = field.Field(domain, name='MN3L')
	MN4L   = field.Field(domain, name='MN4L')
	fields = [MN1L,MN2L,MN3L,MN4L]
	MNadj_lhs = system.FieldSystem(fields)
	################################################################################

	# (3) Build all the matrices for taking derivatives and performing dealiasing
	elements0 = domain.elements(0)
	elements1 = domain.elements(1)

	NxCL = uadj['c'].shape[0]
	NzCL = uadj['c'].shape[1]
	Nx0 = 2*Nx//3
	Nz0 = 2*Nz//3
	DA = np.zeros((NxCL,NzCL))

	Lx = abs(domain.bases[0].interval[0] - domain.bases[0].interval[1]);
	for i in range(NxCL):
		for j in range(NzCL):
			if(np.abs(elements0[i,0]) <  (2.*np.pi/Lx)*(Nx0//2) and elements1[0,j] < Nz0):
				DA[i,j] = 1

	def diffMat():
		Dz = np.zeros((Nz,Nz))
		for i in range(Nz):
			for j in range(Nz):
				if(i<j):
					Dz[i,j] = 2*j*((j-i) % 2)
		Dz[0,:] /= 2
		return Dz;
	Dz = diffMat()
		
	def adjointDerivativeX(vec):
		for i in range(vec.shape[0]):
			vec[i,:] *= -elements0[i]*1j
		return vec

	def derivativeX(vec):
		for i in range(vec.shape[0]):
			vec[i,:] *= elements0[i]*1j
		return vec

	def derivativeZ(vec):
		for i in range(vec.shape[0]):
			vec[i,:] = Dz@vec[i,:]
		return vec

	def derivativeZAdjoint(vec):
		for i in range(vec.shape[0]):
			vec[i,:] = Dz.T@vec[i,:]
		return vec

	# (4) Build the term which computes the action
	# of the adjoint of the jacobian nonlinear term
	# i.e. (∂F/∂x)^†*X^† 
	def NLtermAdj(vec1adj,vec2adj,vec3adj,statess):
		vec1adj = transformAdjoint(DA*vec1adj)
		vec2adj = transformAdjoint(DA*vec2adj)
		vec3adj = transformAdjoint(DA*vec3adj)
		adju  = transformInverseAdjoint(-statess[1]*vec1adj - statess[4]*vec2adj - statess[7]*vec3adj)
		adjux = transformInverseAdjoint(-statess[0]*vec1adj)
		adjuz = transformInverseAdjoint(-statess[3]*vec1adj)
		adjv  = transformInverseAdjoint(-statess[2]*vec1adj - statess[5]*vec2adj - statess[8]*vec3adj)
		adjvx = transformInverseAdjoint(-statess[0]*vec2adj)
		adjvz = transformInverseAdjoint(-statess[3]*vec2adj)
		adjρ = transformInverseAdjoint(0*vec2adj)
		adjρx = transformInverseAdjoint(-statess[0]*vec3adj)
		adjρz = transformInverseAdjoint(-statess[3]*vec3adj)
		return adju,adjux,adjuz,adjv,adjvx,adjvz,adjρ,adjρx,adjρz
	

	# (5) Set the adjoint equation ICs/compatibility condition
	# this requires taking account of the weight matrices,
	# transforms & LBVP
	snapshot_index = -1
	
	W = weightMatrixDisc(domain);
	vecx = transformInverse(X_FWD_DICT['u_fwd'][:,:,snapshot_index])*(W/domain.hypervolume)
	vecz = transformInverse(X_FWD_DICT['w_fwd'][:,:,snapshot_index])*(W/domain.hypervolume)
	snapshot_index -= 1

	vecxhat = transformInverseAdjoint(vecx)
	veczhat = transformInverseAdjoint(vecz)

	vecxhatdx = adjointDerivativeX(vecxhat.copy())
	veczhatdz = derivativeZAdjoint(veczhat.copy())

	MN1adj['c'] = vecxhatdx + veczhatdz
	MNadj_rhs.gather()
	# Solve system for each pencil, updating state
	for p in solverMN.pencils:
		if p.pre_right is not None:
			vec = MNadj_rhs.get_pencil(p)
			b = np.conj(p.pre_right).T @ vec
		else:
			b = state_adj.get_pencil(p)

		x = solverMN.pencil_matsolvers_transposed[p].solve(b)
		x = np.conj(p.pre_left).T @ x
		MNadj_lhs.set_pencil(p, x)
		MNadj_lhs.scatter()
	#########################################################################


	uadj['c']  = 0
	uzadj['c'] = 0
	vadj['c']  = 0
	vzadj['c'] = 0
	padj['c']  = 0
	ρadj['c']  = MN1L['c']
	ρzadj['c'] = 0

	# (6) Solve the adjoint equations bckwards
	for i in range(N_ITERS):
		
		######################## Solve the adjoint LBVP ########################
		state_adj.gather()
		# Solve system for each pencil, updating state
		for p in solver.pencils:
			if p.pre_right is not None:
				vec = state_adj.get_pencil(p)
				b = np.conj(p.pre_right).T @ vec
			else:
				b = state_adj.get_pencil(p)

			x = solver.pencil_matsolvers_transposed[p].solve(b)
			x = np.conj(p.pre_left).T @ x
			equ_adj.set_pencil(p, x)
			equ_adj.scatter()
		#########################################################################
		uadj['c']  = rhsUA['c'].copy()
		vadj['c']  = rhsVA['c'].copy()
		ρadj['c']  = rhsRhoA['c'].copy()
		uzadj['c'] = rhsuzA['c'].copy()
		vzadj['c'] = rhsvzA['c'].copy()
		ρzadj['c'] = rhsRhozA['c'].copy()

		uDir = X_FWD_DICT['u_fwd'][:,:,snapshot_index]
		vDir = X_FWD_DICT['w_fwd'][:,:,snapshot_index]
		ρDir = X_FWD_DICT['b_fwd'][:,:,snapshot_index]

		uxDir = transformInverse(derivativeX(uDir.copy()))
		uzDir = transformInverse(derivativeZ(uDir.copy()))

		vxDir = transformInverse(derivativeX(vDir.copy()))
		vzDir = transformInverse(derivativeZ(vDir.copy()))

		ρxDir = transformInverse(derivativeX(ρDir.copy()))
		ρzDir = transformInverse(derivativeZ(ρDir.copy()))

		uDir = transformInverse(uDir.copy())
		vDir = transformInverse(vDir.copy())

		states = [uDir,uxDir,uzDir,vDir,vxDir,vzDir,ρDir,ρxDir,ρzDir]

		snapshot_index -= 1

		adju,adjux,adjuz,adjv,adjvx,adjvz,adjρ,adjρx,adjρz = NLtermAdj(uadj['c'].copy(),vadj['c'].copy(),ρadj['c'].copy(),states)
		uadj['c']  = uadj['c']/dt + adju + adjointDerivativeX(adjux)
		vadj['c']  = vadj['c']/dt + adjv + adjointDerivativeX(adjvx)
		ρadj['c']  = ρadj['c']/dt + adjρ + adjointDerivativeX(adjρx)
		uzadj['c'] = adjuz
		vzadj['c'] = adjvz
		ρzadj['c'] = adjρz

	uadj['c'] += derivativeZAdjoint(uzadj['c'])
	vadj['c'] += derivativeZAdjoint(vzadj['c'])

	uadj['g'] = (domain.hypervolume/W)*transformAdjoint(uadj['c'])
	vadj['g'] = (domain.hypervolume/W)*transformAdjoint(vadj['c'])

	# Set to info level rather than the debug default
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");

	return [ Field_to_Vec(domain,uadj ,vadj) ];

#########################################################################
# ~~~~~ Negative derivatives Solvers  ~~~~~~~~~~~~~
##########################################################################

def Norm_and_Inverse_Second_Derivative(rho,domain):
	"""
	Build a BVP that takes the final density field
	and returns its inverse first derivative or
	inverse laplacian. We enforce only that the
	solution must have zero-mean
	"""

	import dedalus.public as de

	# Poisson equation
	problem = de.LBVP(domain, variables=['Ψ','Ψz']);
	problem.meta['Ψ']['z']['dirichlet'] = True;
	problem.parameters['f' ] = rho;

	problem.add_equation("dx(dx(Ψ)) + dz(Ψz) = f")
	problem.add_equation("Ψz - dz(Ψ) = 0")
	problem.add_bc("left(Ψ)  = 0");
	problem.add_bc("right(Ψ) = 0");

	# Build solver
	solver = problem.build_solver()
	solver.solve()

	# Differentiate solution
	Ψ  = solver.state['Ψ']
	fz = solver.state['Ψz']

	fx = domain.new_field(name='fx'); Ψ.differentiate('x',out=fx);
	#fz = domain.new_field(name='fz'); Ψ.differentiate('z',out=fz);

	return Integrate_Field(domain,fx**2 + fz**2), Ψ;


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

	Inner_Prod = Inner_Prod_Discrete
	FWD_Solve  = FWD_Solve_Discrete
	ADJ_Solve  = ADJ_Solve_Discrete

elif Adjoint_type == "Continuous":

	Inner_Prod = Inner_Prod_Cnts
	FWD_Solve  = FWD_Solve_Cnts
	ADJ_Solve  = ADJ_Solve_Cnts;


if __name__ == "__main__":


	Re = 500.;  Ri = 0.05;
	#Nx = 256; Nz = 2*48; T_opt = 10; dt = 5e-04;
	Nx = 128; Nz = 64; T_opt = 5; dt = 5e-03;
	E_0 = 0.02

	N_ITERS = int(T_opt/dt);

	if(Adjoint_type=="Discrete"):
		Nx = 3*Nx//2
		Nz = 3*Nz//2
		dealias_scale = 1
	else:
		dealias_scale = 3/2
	#α = 0; ß = 0; # (A) time-averaged-kinetic-energy maximisation (α = 0)
	α = 1; ß = 1; # (B) mix-norm minimisation (α = 1, β = 1)

	domain, Ux0  = Generate_IC(Nx,Nz,dealias_scale=dealias_scale);
	X_FWD_DICT   = GEN_BUFFER( Nx,Nz,domain,N_ITERS);

	Prandtl=1.; δ  = 0.125
	args_f  = [domain, Re,Ri, N_ITERS, X_FWD_DICT,dt, α,ß, Prandtl,δ];
	args_IP = [domain,None];


	#FWD_Solve([Ux0],*args_f);
	#sys.exit()

	#sys.path.insert(0,'/Users/pmannix/Desktop/Nice_CASTOR')

	# Test the gradient
	#from TestGrad import Adjoint_Gradient_Test
	#_, dUx0  = Generate_IC(Nx,Nz,dealias_scale=dealias_scale);
	#Adjoint_Gradient_Test(Ux0,dUx0, FWD_Solve,ADJ_Solve,Inner_Prod,args_f,args_IP,epsilon=1e-04)
	#sys.exit()
	
	# Run the optimisation
	from Sphere_Grad_Descent import Optimise_On_Multi_Sphere, plot_optimisation
	RESIDUAL,FUNCT,U_opt = Optimise_On_Multi_Sphere([Ux0], [E_0], FWD_Solve,ADJ_Solve,Inner_Prod,args_f,args_IP, err_tol = 1e-06, max_iters = 10, alpha_k = 10., LS = 'LS_armijo', CG = False, callback=File_Manips)
	plot_optimisation(RESIDUAL,FUNCT);

	####
