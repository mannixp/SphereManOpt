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
    b[1:,:] *= 1./2
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
    # b[1:,:] *= 1./2
    b = dct(b,type=2,norm='ortho',axis=1)*np.sqrt(Ny)
    b[:,1:] *= np.sqrt(2)
    b[:,1:] *= 0.5
    b[:,1::2] *= -1

    return b

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

	# Parse distributed fields into arrays on every proc
	for i in range( MPI.COMM_WORLD.Get_size() ):
		FX[G_slices[i]] = Fx_global[i];
		FZ[G_slices[i]] = Fz_global[i];

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
def Inner_Prod_Discrete(x,y,domain,W):

	dA = new_ncc(domain);
	dB = new_ncc(domain);
	Vec_to_Field(domain,dA,dB, x);

	du = new_ncc(domain);
	dv = new_ncc(domain);
	Vec_to_Field(domain,du,dv, y);

	inner_prod = (np.vdot(dA['g'],W*du['g']) + np.vdot(dB['g'],W*dv['g']))/domain.hypervolume
	inner_prod = comm.allreduce(inner_prod,op=MPI.SUM)
	return inner_prod;


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
	#z_basis = de.Chebyshev('z', Nz, interval=Z_domain, dealias=dealias_scale); # z

	if(Adjoint_type=="Discrete"):
		y_basis     = de.Chebyshev('z', Nz, interval=(-1, 1) )
		domain  = de.Domain([x_basis, y_basis], grid_dtype=np.complex128);
	else:
		zb1     = de.Chebyshev('z1', Nz, interval=(-1, 0.) )
		zb2     = de.Chebyshev('z2', Nz, interval=(0., 1.) )
		z_basis = de.Compound('z', (zb1,zb2), dealias=dealias_scale)

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

	U0  = Field_to_Vec(domain,ψ,ψ);
	'''
	filter_field(ψ)   # Filter the noise, modify this for less noise
	u = domain.new_field(name='u'); uz = domain.new_field(name='uz');
	w = domain.new_field(name='w'); wz = domain.new_field(name='wz');

	ψ.differentiate('z',out=u); u.differentiate('z',out=uz);
	ψ.differentiate('x',out=w); w.differentiate('z',out=wz);
	u['g']  *= -1;

	U0  = Field_to_Vec(domain,u,w);
	'''

	# Create vector
	U0 = FWD_Solve_IVP_Prep(U0,domain);
	if Adjoint_type == "Discrete":
		x = domain.grid(0, scales=1)
		y = domain.grid(1, scales=1)
		NxL = x.shape[0]
		NyL = y.shape[1]
		def weightMatrixDisc():
			W = np.zeros((NxL,NyL))
			# Mult by dy
			for i in range(NyL):
				if(i==0):
					W[:,i] = y[0,i+1]-y[0,i]
				else:
					W[:,i] = y[0,i]-y[0,i-1]
			dx = 4*np.pi/Nx
			return W*dx

		W = weightMatrixDisc()
		M = np.sqrt(W)
		SUM = Inner_Prod(U0,U0,domain,W);
	else:
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

def FWD_Solve_IVP_Prep(U0, domain, Reynolds=500., Richardson=0.5, N_ITERS=100., dt=1e-04, Prandtl=1., δ  = 0.025):

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

def FWD_Solve_Cnts( U0, domain, Reynolds, Richardson, N_ITERS, X_FWD_DICT,   dt=1e-04, α = 0, ß = 0,filename=None, Prandtl=1., δ  = 0.025):

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
	b['g']  = -(1./2.)*erf(z/δ);
	bz['g'] = -np.exp(-(z/δ)**2)/(δ*np.sqrt(np.pi));

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
def FWD_Solve_Discrete(U0, domain, Reynolds, Richardson, N_ITERS, X_FWD_DICT,M,   dt=1e-04, α = 0, ß = 0,filename=None, Prandtl=1., δ  = 0.025):

	Re = Reynolds
	Pe = Reynolds*Prandtl
	Ri = Richardson

	ReInv = 1./Re

	problem = de.LBVP(domain, variables=['u','v','rho','p','uy','vy','rhoy'])
	problem.parameters['dt'] = dt
	problem.parameters['ReInv'] = 1./Re
	problem.parameters['Ri'] = Ri
	problem.parameters['PeInv'] = 1./Pe

	problem.add_equation("u/dt   - ReInv*(dx(dx(u)) + dz(uy)) + dx(p) = 0")
	problem.add_equation("v/dt   - ReInv*(dx(dx(v)) + dz(vy)) + dz(p) + Ri*rho  = 0")
	problem.add_equation("rho/dt - PeInv*(dx(dx(rho)) + dz(rhoy)) = 0")
	problem.add_equation("dx(u) + vy = 0")
	problem.add_equation("uy   - dz(u)   = 0");
	problem.add_equation("vy   - dz(v)   = 0");
	problem.add_equation("rhoy - dz(rho)   = 0");

	problem.add_bc("left(u) = 0");
	problem.add_bc("left(v) = 0");

	problem.add_bc("right(u) = 0");
	problem.add_bc("right(v) = 0",condition="(nx != 0)")
	problem.add_equation("integ(p) = 0",condition="(nx == 0)");

	problem.add_bc("left(rhoy) = 0");
	problem.add_bc("right(rhoy) = 0");

	solver = problem.build_solver()
	############### Build the adjoint matrices ###############
	solver.pencil_matsolvers_transposed = {}
	for p in solver.pencils:
	    solver.pencil_matsolvers_transposed[p] = solver.matsolver(np.conj(p.L_exp).T, solver)
	##########################################################

	u   = solver.state['u']
	v   =  solver.state['v']
	rho = solver.state['rho']
	uy = solver.state['uy']
	vy = solver.state['vy']
	rhoy = solver.state['rhoy']

	##########################################################

	problemMN = de.LBVP(domain, variables=['rho_inv','rho_inv_y'])
	problemMN.add_equation("dx(dx(rho_inv))+dz(rho_inv_y)=0")
	problemMN.add_equation("rho_inv_y - dz(rho_inv)=0")
	problemMN.add_bc("left(rho_inv) = 0");
	problemMN.add_bc("right(rho_inv) = 0");

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

	MN1adj   = field.Field(domain, name='MN1adj')
	MN2adj   = field.Field(domain, name='MN2adj')

	fields = [MN1adj,MN2adj]
	MNadj_rhs = system.FieldSystem(fields)

	MN1L   = field.Field(domain, name='MN1L')
	MN2L   = field.Field(domain, name='MN2L')
	MN3L   = field.Field(domain, name='MN3L')
	MN4L   = field.Field(domain, name='MN4L')
	fields = [MN1L,MN2L,MN3L,MN4L]
	MNadj_lhs = system.FieldSystem(fields)

	rho_inv   = solverMN.state['rho_inv']
	################################################################################

	rhsU   = field.Field(domain, name='rhsU')
	rhsV   = field.Field(domain, name='rhsV')
	rhsRho = field.Field(domain, name='rhsRho')
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
	fields = [rhsU,rhsV,rhsRho,rhsD4,rhsD5,rhsD6,rhsD7,rhsD8,rhsD9,rhsD10,rhsD11,rhsD12,rhsD13,rhsD14]
	equ_rhs = system.FieldSystem(fields)

	x = domain.grid(0)
	y = domain.grid(1)

	uBase = field.Field(domain, name='uBase')
	uBase['g'] = 1-y**2

	uyBase = field.Field(domain, name='uyBase')
	uBase.differentiate(1, out=uyBase)

	rhoBase = field.Field(domain, name='rhoBase')
	from scipy import special
	rhoBase['g'] = -0.5*special.erf(y/δ)

	rhoyBase = field.Field(domain, name='rhoyBase')
	rhoBase.differentiate(1, out=rhoyBase)

	NxCL = u['c'].shape[0]
	NyCL = u['c'].shape[1]

	Nx = u['g'].shape[0]
	Ny = u['g'].shape[1]

	elements0 = domain.elements(0)
	elements1 = domain.elements(1)

	DA = np.zeros((NxCL,NyCL))
	Nx0 =2*Nx//3
	Ny0 = 2*Ny//3
	for i in range(NxCL):
		for j in range(NyCL):
			if(elements0[i,0] < Nx0//2 and elements1[0,j] < Ny0):
				DA[i,j] = 1

	def NLterm(u,ux,uy,v,vx,vy,rho,rhox,rhoy):
		NLu   = -transformInverse(u)*transformInverse(ux)   - transformInverse(v)*transformInverse(uy) + 2*ReInv
		NLv   = -transformInverse(u)*transformInverse(vx)   - transformInverse(v)*transformInverse(vy)
		NLrho = -transformInverse(u)*transformInverse(rhox) - transformInverse(v)*transformInverse(rhoy)
		return DA*transform(NLu),DA*transform(NLv),DA*transform(NLrho)

	def derivativeX(vec):
		for i in range(vec.shape[0]):
			vec[i,:] *= elements0[i]*1j
		return vec

	Vec_to_Field(domain,u ,v ,U0[0]);
	# vecU = vec.reshape((2,NxL,NyL))[0,:,:]
	# vecV = vec.reshape((2,NxL,NyL))[1,:,:]

	u['c'] += uBase['c']
	# states = []
	# u['c'] = uBase['c'] + transform(vecU)
	# v['c'] = transform(vecV)
	rho['c'] = rhoBase['c']
	u.differentiate(1, out=uy)
	v.differentiate(1, out=vy)
	rhoy['c'] = rhoyBase['c']

	snapshot_index = 0

	for i in range(N_ITERS):
		ux   = derivativeX(u['c'].copy())
		vx   = derivativeX(v['c'].copy())
		rhox = derivativeX(rho['c'].copy())

		X_FWD_DICT['u_fwd'][:,:,snapshot_index] = u['c']
		X_FWD_DICT['w_fwd'][:,:,snapshot_index] = v['c']
		X_FWD_DICT['b_fwd'][:,:,snapshot_index] = rho['c']
		snapshot_index+=1;

		NLu,NLv,NLrho = NLterm(u['c'],ux,uy['c'],v['c'],vx,vy['c'],rho['c'],rhox,rhoy['c'])
		rhsU['c'] = solver.state['u']['c']/dt + NLu
		rhsV['c'] = solver.state['v']['c']/dt + NLv
		rhsRho['c'] = solver.state['rho']['c']/dt + NLrho
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


	######################## Solve the Mix Norm LBVP ########################
	MN1['c'] = rho['c']
	MN_rhs.gather()
	for p in solverMN.pencils:
		b = p.pre_left @ MN_rhs.get_pencil(p)
		x = solverMN.pencil_matsolvers[p].solve(b)
		if p.pre_right is not None:
			x = p.pre_right @ x
		solverMN.state.set_pencil(p, x)
		solverMN.state.scatter()
		################################################################

	X_FWD_DICT['u_fwd'][:,:,snapshot_index] = u['c']
	X_FWD_DICT['w_fwd'][:,:,snapshot_index] = v['c']
	X_FWD_DICT['b_fwd'][:,:,snapshot_index] = rho_inv['c']
	# states.append(transformInverse(rho_inv['c']).real)
	# states.append(transformInverse(rho['c']))
	cost = np.linalg.norm(M*rho_inv['g'])**2
	cost = comm.allreduce(cost,op=MPI.SUM)
	return cost;
##########################################################################
# ~~~~~ ADJ Solvers  ~~~~~~~~~~~~~
##########################################################################

def ADJ_Solve_Cnts(U0, domain, Reynolds, Richardson, N_ITERS, X_FWD_DICT,   dt=1e-04, α = 0, ß = 0, Prandtl=1., δ  = 0.025, Sim_Type = "Non_Linear"):

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
def ADJ_Solve_Discrete(X_k,some_args):

	return None;

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
	Ψ = solver.state['Ψ']

	fx = domain.new_field(name='fx'); Ψ.differentiate('x',out=fx);
	fz = domain.new_field(name='fz'); Ψ.differentiate('z',out=fz);

	return Integrate_Field(domain,fx**2 + fz**2), Ψ;


def File_Manips(k):

	"""
	Takes files generated by the adjoint loop solve, orgainises them into respective directories
	and removes unwanted data

	Each iteration of the DAL loop code is stored under index k
	"""

	import shutil

	# A) Contains all scalar data
	shutil.copyfile('scalar_data/scalar_data_s1.h5','scalar_data_iter_%i.h5'%k);

	# B) Contains all field data
	shutil.copyfile('CheckPoints/CheckPoints_s1.h5','CheckPoints_iter_%i.h5'%k);

	return None;


Adjoint_type = "Discrete";

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
	dt = 5e-04;
	Nx = 256;
	Nz = 48; # Using a compound basis in z to resolve the erf(z) so the resolution will be double this
	T_opt = 10.; E_0 = 0.02
	T_opt = 10e-3
	N_ITERS = int(T_opt/dt);

	if(Adjoint_type=="Discrete"):
		Nx = 3*Nx//2
		Nz = 3*Nz//2
		dealias_scale = 1
	else:
		dealias_scale = 3/2
	α = 0; ß = 0; # (A) time-averaged-kinetic-energy maximisation (α = 0)
	#α = 1; ß = 1; # (B) mix-norm minimisation (α = 1, β = 1)

	domain, Ux0  = Generate_IC(Nx,Nz,dealias_scale=dealias_scale);

	x = domain.grid(0, scales=1)
	y = domain.grid(1, scales=1)
	NxL = x.shape[0]
	NyL = y.shape[1]
	def weightMatrixDisc():
		W = np.zeros((NxL,NyL))
		# Mult by dy
		for i in range(NyL):
			if(i==0):
				W[:,i] = y[0,i+1]-y[0,i]
			else:
				W[:,i] = y[0,i]-y[0,i-1]
		dx = 4*np.pi/Nx
		return W*dx

	W = weightMatrixDisc()
	M = np.sqrt(W)

	X_FWD_DICT   = GEN_BUFFER( Nx,Nz,domain,N_ITERS);

	if(Adjoint_type=="Discrete"):
		args_f  = [domain, Re,Ri, N_ITERS, X_FWD_DICT,M,dt, α,ß];
		args_IP = [domain,W];
	else:
		args_f  = [domain, Re,Ri, N_ITERS, X_FWD_DICT,dt, α,ß];
		args_IP = [domain,None];

	#sys.path.insert(0,'/Users/pmannix/Desktop/Nice_CASTOR')

	# Test the gradient
	#from TestGrad import Adjoint_Gradient_Test
	#_, dUx0  = Generate_IC(Nx,Nz);
	#Adjoint_Gradient_Test(Ux0,dUx0, FWD_Solve,ADJ_Solve,Inner_Prod,args_f,args_IP)
	#sys.exit()

	# Run the optimisation
	from Sphere_Grad_Descent import Optimise_On_Multi_Sphere, plot_optimisation
	RESIDUAL,FUNCT,U_opt = Optimise_On_Multi_Sphere([Ux0], [E_0], FWD_Solve,ADJ_Solve,Inner_Prod,args_f,args_IP, err_tol = 1e-06, max_iters = 200, alpha_k = 1., LS = 'LS_wolfe', CG = True, callback=File_Manips)

	plot_optimisation(RESIDUAL,FUNCT);

	####
