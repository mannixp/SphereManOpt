import sys,os,time;
os.environ["OMP_NUM_THREADS"] = "1" # Improves performance apparently ????

from mpi4py import MPI # Import this before numpy
import numpy as np
import h5py,logging

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

def Integrate_Field(domain,F):

	"""
	Performs the Volume integral of a given field F as 

	KE(t) = 1/V int_v F(r,z,phi) dV, where F = u^2 + v^2 + w^2 & dV = r*dr*d_phi*dz

	where KE is Kinetic Enegry
	"""
	# Dedalus Libraries
	from dedalus.extras import flow_tools 
	import dedalus.public as de

	# 1) Multiply by the integration weights r*dr*d_phi*dz for cylindrical domain
	flow_red = flow_tools.GlobalArrayReducer(MPI.COMM_WORLD);
	INT_ENERGY = de.operators.integrate( F ,'x', 'y','z');
	SUM = INT_ENERGY.evaluate();
	
	# Divide by volume size
	VOL = 1./domain.hypervolume;

	return VOL*flow_red.global_max(SUM['g']); # Using this as it's what flow-tools does for volume average

def Field_to_Vec(domain,Fx,Fy,Fz ):
	
	"""
	Convert from: field to numpy 1D vector-
	
	Takes the LOCAL portions of:
	Inputs:
	- GLOBALLY distributed fields Fx,Fy,Fz
	- domain object 

	Creates locally available numpy arrays of a global size &
	makes this array is available on all cores using MPI gather_all function
	
	This function assumes all arrays can fit in memory!!!

	Returns:
	- 1D np.array of (Fx,Fy,Fz)
	"""

	# 1) Create local array of the necessary dimension
	#lshape = domain.dist.grid_layout.local_shape(scales=domain.dealias)
	gshape = tuple( domain.dist.grid_layout.global_shape(scales=domain.dealias) );
	
	for f in [Fx,Fy,Fz]:
		f.set_scales(domain.dealias,keep_data=True);

	# 2) Gather all data onto EVERY processor!!
	Fx_global = MPI.COMM_WORLD.allgather(Fx['g']);
	Fy_global = MPI.COMM_WORLD.allgather(Fy['g']);
	Fz_global = MPI.COMM_WORLD.allgather(Fz['g']);

	# Gathered slices
	G_slices = MPI.COMM_WORLD.allgather( domain.dist.grid_layout.slices(scales=domain.dealias) )

	# 3) Declared an array of GLOBAL shape on every proc
	FX = np.zeros(gshape);
	FY = np.zeros(gshape); 
	FZ = np.zeros(gshape);

	# Parse distributed fields into arrays on every proc
	for i in range( MPI.COMM_WORLD.Get_size() ):
		FX[G_slices[i]] = Fx_global[i];
		FY[G_slices[i]] = Fy_global[i];
		FZ[G_slices[i]] = Fz_global[i]; 

	# 4) Merge them together at the end!
	return np.concatenate( (FX.flatten(),FY.flatten(),FZ.flatten()) );

def Vec_to_Field(domain,A,B,C,Bx0):

	"""
	Convert from: numpy 1D vector to field - 
	Takes a 1D array Bx0 and distributes it into fields A,B,C on 
	num_procs = MPI.COMM_WORLD.size

	Inputs:
	- domain object 
	- GLOBALLY distributed dedalus fields A,B,C
	- Bx0 1D np.array

	Returns:
	- None
	"""

	# 1) Split the 1D array into 1D arrays A,B,C
	a1,a2,a3 = np.split(Bx0,3); #Passed in dealiased scale
	gshape = tuple( domain.dist.grid_layout.global_shape(scales=domain.dealias) )
	slices = domain.dist.grid_layout.slices(scales=domain.dealias)
	#lshape = domain.dist.grid_layout.local_shape(scales=domain.dealias)
	#slices = domain.dist.grid_layout.slices(scales=domain.dealias)

	for f in [A,B,C]:
		f.set_scales(domain.dealias,keep_data=False);
		f['g']=0.

	# 2) Reshape and parse relevant portion into A,B,C
	A['g'] = a1.reshape( gshape )[slices]
	B['g'] = a2.reshape( gshape )[slices]
	C['g'] = a3.reshape( gshape )[slices]

	return None;

def Inner_Prod(domain,x,y):

	# The line-search requires the IP
	# m = <\Nabla J^T, P_k >, where P_k = - \Nabla J(B_0)^T
	# Must be evaluated using an integral consistent with our objective function
	# i.e. <,> = (1/V)int_v x*y r dr ds dz
	# To do this we transform back to fields and integrate using a consistent routine

	#Split the 6 vector into three vectors 
	X = np.split(x,2);
	Y = np.split(y,2);

	Sum = 0.;
	for i in range(2):

		dA = new_ncc(domain); dB = new_ncc(domain); dC = new_ncc(domain);
		Vec_to_Field(domain,dA,dB,dC, X[i]);

		du = new_ncc(domain); dv = new_ncc(domain); dw = new_ncc(domain);
		Vec_to_Field(domain,du,dv,dw, Y[i]);

		Sum += Integrate_Field(domain, (dA*du) + (dB*dv) + (dC*dw) );

	return Sum;

def Inner_Prod_3(domain,x,y):

	dA = new_ncc(domain); dB = new_ncc(domain); dC = new_ncc(domain);
	Vec_to_Field(domain,dA,dB,dC, x);

	du = new_ncc(domain); dv = new_ncc(domain); dw = new_ncc(domain);
	Vec_to_Field(domain,du,dv,dw, y);

	return Integrate_Field(domain, (dA*du) + (dB*dv) + (dC*dw) );

def Generate_IC(Npts, X = (0.,2.*np.pi), M_0=1.0, U_Noise = False):
	"""
	Generate a domain object and initial conditions from which the optimisation can proceed

	Input:
	- Npts - integer resolution size
	- X    - interval/domain scale
	- M_0  - initial condition amplitude
	
	Returns: 
	- domain object
	- initial cond Bx0, as a field obj {Bx,By,Bz}
	- initial cond U0 , as a field obj {Ux,Uy,Uz}
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
	dealias_scale = 3/2;
	x_basis = de.Fourier('x', Npts, interval=X, dealias=dealias_scale); # x
	y_basis = de.Fourier('y', Npts, interval=X, dealias=dealias_scale); # y
	z_basis = de.Fourier('z', Npts, interval=X, dealias=dealias_scale); # z
	domain  = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64);

	# Part 2) Generate initial condition B = {Bx, By, Bz}
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	phi = domain.new_field();
	phi.set_scales(domain.dealias, keep_data=False)
	gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
	slices = domain.dist.grid_layout.slices(scales=domain.dealias)
	rand = np.random.RandomState(seed=42)
	noise = rand.standard_normal(gshape)[slices]; #Slicing globally generated noise here!!
	phi['g'] = noise; # Could scale this ???
	filter_field(phi)   # Filter the noise, modify this for less noise

	phi_x = domain.new_field(name='phi_x'); phi.differentiate('x',out=phi_x)
	phi_y = domain.new_field(name='phi_y'); phi.differentiate('y',out=phi_y)
	phi_z = domain.new_field(name='phi_z'); phi.differentiate('z',out=phi_z)

	Bx = domain.new_field();
	By = domain.new_field();
	Bz = domain.new_field();

	for f in [Bx,By,Bz]:
		f.set_scales(domain.dealias, keep_data=False)

	# 2) Take curl - could be a better way of doing this above so as to avoid creating new fields?
	Bx['g'] += (phi_y['g'] - phi_z['g']);
	By['g'] += (phi_z['g'] - phi_x['g']);
	Bz['g'] += (phi_x['g'] - phi_y['g']);

	# Part 3) Generate initial condition B = {Bx, By, Bz}
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	Ux = domain.new_field();
	Uy = domain.new_field();
	Uz = domain.new_field();

	if U_Noise == False:
		
		x = domain.grid(0);
		y = domain.grid(1);
		z = domain.grid(2);

		Ux['g'] = 0.5*np.sin(y)*np.cos(z)/np.sqrt(3.);
		Uy['g'] = 0.5*np.sin(z)*np.cos(x)/np.sqrt(3.);
		Uz['g'] = 0.5*np.sin(x)*np.cos(y)/np.sqrt(3.);
	
	elif U_Noise == True:	
		
		for f in [Ux,Uy,Uz]:
			f.set_scales(domain.dealias, keep_data=False);
			f['g'] = 0.

		phi = domain.new_field();
		phi.set_scales(domain.dealias, keep_data=False)
		gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
		slices = domain.dist.grid_layout.slices(scales=domain.dealias)
		rand = np.random.RandomState(seed=42)
		noise = rand.standard_normal(gshape)[slices]; #Slicing globally generated noise here!!
		phi['g'] = noise; # Could scale this ???
		filter_field(phi)   # Filter the noise, modify this for less noise

		phi_x = domain.new_field(name='phi_x'); phi.differentiate('x',out=phi_x)
		phi_y = domain.new_field(name='phi_y'); phi.differentiate('y',out=phi_y)
		phi_z = domain.new_field(name='phi_z'); phi.differentiate('z',out=phi_z)

		# 2) Take curl - could be a better way of doing this above so as to avoid creating new fields?
		Ux['g'] += (phi_y['g'] - phi_z['g']);
		Uy['g'] += (phi_z['g'] - phi_x['g']);
		Uz['g'] += (phi_x['g'] - phi_y['g']);

	# 3) Normalise it
	SUM = Integrate_Field(domain, (Ux**2)+(Uy**2)+(Uz**2) );
	logger.info('Pre-scale (1/V)<U,U> = %e'%SUM);
	
	Rescale_B = np.sqrt(1./SUM);
	for f in [Ux,Uy,Uz]:
		f['g'] = Rescale_B*f['g'];

	logger.info('Created a vector (Ux,Uy,Uz) \n\n');

	# Part 4) Smoothen initial condition B = {Bx, By, Bz}
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	Rm_IC = 1.0; dt_IC = 0.001; N_ITERS = 100;
	Bx0 = Field_to_Vec(domain,Bx,By,Bz);
	Ux0 = Field_to_Vec(domain,Ux,Uy,Uz);
	Bx,By,Bz = FWD_Solve_IVP_Prep(Bx0,Ux0, domain,Rm,dt,  N_ITERS)
	
	# 3) Normalise it
	SUM = Integrate_Field(domain, (Bx**2)+(By**2)+(Bz**2) );
	logger.info('Pre-scale (1/V)<B,B> = %e'%SUM);
	
	Rescale_B = np.sqrt(M_0/SUM);
	for f in [Bx,By,Bz]:
		f['g'] = Rescale_B*f['g'];

	logger.info('Created a vector (Bx,By,Bz) \n\n');

	for f in [Bx,By,Bz, Ux,Uy,Uz]:
		f.set_scales(domain.dealias, keep_data=True)

	return domain, Field_to_Vec(domain,Bx,By,Bz ), Field_to_Vec(domain,Ux,Uy,Uz );

def GEN_BUFFER(Npts, domain, N_SUB_ITERS):

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
	Total  = ( 0.5*(Npts**3)*64*N_SUB_ITERS*(1.25e-10) )/float( MPI.COMM_WORLD.Get_size() )
	if MPI.COMM_WORLD.rank == 0:
		print("Total memory =%f GB, and memory/core = %f GB"%(MPI.COMM_WORLD.Get_size()*Total,Total));

	gshape  = tuple( domain.dist.coeff_layout.global_shape(scales=1) );
	lcshape = tuple( domain.dist.coeff_layout.local_shape(scales=1) );
	SNAPS_SHAPE = (lcshape[0],lcshape[1],lcshape[2],N_SUB_ITERS+1);

	A_SNAPS = np.zeros(SNAPS_SHAPE,dtype=complex);
	B_SNAPS = np.zeros(SNAPS_SHAPE,dtype=complex);
	C_SNAPS = np.zeros(SNAPS_SHAPE,dtype=complex);

	return {'A_fwd':A_SNAPS,'B_fwd':B_SNAPS,'C_fwd':C_SNAPS};

##########################################################################
# ~~~~~ FWD Solvers ~~~~~~~~~~~~~
##########################################################################

def FWD_Solve_Build_Lin(domain, Rm, dt, Ux0):
	
	"""
	Driver program for Periodic Box Dynamo, which builds the forward solver object with options:

	Inputs:
	domain (dedalus object) returned by ??
	flow paramaters Rm - Magnetic Reynolds number

	dt, ts - float numerical integration time-step & dedalus timestepper object

	Returns:
	Dedalus object to solve the MHD equations in a periodic box geometry

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
	# \vec{v}  = u x^ + v y^ + w z^ 	&		 \vec{B}  = A x^ + B y^ + C z^
	PCF = de.IVP(domain, variables=['Pi', 'A', 'B', 'C'], time='t');
	PCF.parameters['Rm'] = Rm;
	PCF.parameters['inv_Vol'] = 1./domain.hypervolume;

	U = new_ncc(domain);
	V = new_ncc(domain);
	W = new_ncc(domain);

	Vec_to_Field(domain,U,V,W,Ux0);

	# Fields for linearisation
	PCF.parameters['uf'] = U
	PCF.parameters['vf'] = V
	PCF.parameters['wf'] = W

	#######################################################
	# Substitutions
	#######################################################
	# 1.A) ~~~~~~ Laplacian^2 ~~~~~~~~~~~~~~
	PCF.substitutions['Lap(f)'] = "dx(dx(f)) + dy(dy(f)) + dz(dz(f))";

	# 1.C) ~~~~~~ (NAB x (A x B) ~~~~~~~~~~ Correct
	PCF.substitutions['EMF_x(A2,A3,B2,B3)'] = "A2*B3 - A3*B2";
	PCF.substitutions['EMF_y(A1,A3,B1,B3)'] = "A3*B1 - A1*B3";
	PCF.substitutions['EMF_z(A1,A2,B1,B2)'] = "A1*B2 - A2*B1";

	PCF.substitutions['INDx(A1,A2,A3,B1,B2,B3)'] = "dy( EMF_z(A1,A2,B1,B2) ) - dz( EMF_y(A1,A3,B1,B3) )"; 
	PCF.substitutions['INDy(A1,A2,A3,B1,B2,B3)'] = "dz( EMF_x(A2,A3,B2,B3) ) - dx( EMF_z(A1,A2,B1,B2) )"; 
	PCF.substitutions['INDz(A1,A2,A3,B1,B2,B3)'] = "dx( EMF_y(A1,A3,B1,B3) ) - dy( EMF_x(A2,A3,B2,B3) )";

	#######################################################
	# add equations
	#######################################################
	logger.info("--> Adding Equations");

	# Apply divergence free condition
	PCF.add_equation("dx(A) + dy(B) + dz(C)  = 0", 							   condition="(nx != 0) or (ny != 0) or (nz != 0)");
	PCF.add_equation("dt(A) - (1./Rm)*Lap(A) - dx(Pi) = INDx(uf,vf,wf,A,B,C)", condition="(nx != 0) or (ny != 0) or (nz != 0)");
	PCF.add_equation("dt(B) - (1./Rm)*Lap(B) - dy(Pi) = INDy(uf,vf,wf,A,B,C)", condition="(nx != 0) or (ny != 0) or (nz != 0)");   
	PCF.add_equation("dt(C) - (1./Rm)*Lap(C) - dz(Pi) = INDz(uf,vf,wf,A,B,C)", condition="(nx != 0) or (ny != 0) or (nz != 0)");

	# Zero the constant part
	PCF.add_equation("A  = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)");
	PCF.add_equation("B  = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)");
	PCF.add_equation("C  = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)");
	PCF.add_equation("Pi = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)");	

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	IVP_FWD = PCF.build_solver(de.timesteppers.CNAB1);

	# Set to info level rather than the debug default
	for h in root.handlers:
		h.setLevel("WARNING");
		#h.setLevel("INFO");

	return IVP_FWD;

def FWD_Solve_IVP_Prep(Bx0,Ux0, domain,Rm,dt,  N_ITERS):
	
	"""
	Integrates the initial condition X(t=0) = Bx0 -> B(x,T);
	using the induction equation,

	Input:
	# Initial conditions
	if filename == None:
		Bx0  - dict, Inital magnetic field
		#Ux0  - dict, Inital Velocity field
	else:
		load IC's from filename	

	*args - list of solver arguments, see "main of the file" (below) for there definition

	args = [domain, IC, Rm, dt];
	
	Returns:
		final magnetic energy/time integrated magnetic energy
	
	- Writes the following to disk:
	1) FILE Scalar-data (every 20 iters): Magnetic Enegry, etc. 

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
	IVP_FWD = FWD_Solve_Build_Lin(domain, Rm, dt, Ux0); #Base_Bx0,Base_Ux0);

	Pi = IVP_FWD.state['Pi']; 
	A  = IVP_FWD.state['A'];	B  = IVP_FWD.state['B'];	C  = IVP_FWD.state['C']
	for f in [Pi, A,B,C]:
		f.set_scales(domain.dealias, keep_data=False)
		f['g'] = 0.

	#######################################################
	# set initial conditions
	#######################################################
	Vec_to_Field(domain,A,B,C,Bx0);

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
	logger.info("\n\n --> Timestepping to prepare IC's for FWD_Solve ");
	#######################################################

	while IVP_FWD.ok:

		IVP_FWD.step(dt);

	#######################################################

	logger.info("--> Complete <--\n\n")

	return A,B,C;

def FWD_Solve_IVP_Lin(X0, domain,Rm,dt,  N_ITERS,N_SUB_ITERS, X_FWD_DICT, Cost_function= "Final"):
	
	"""
	Integrates the initial condition X(t=0) = Bx0 -> B(x,T);
	using the induction equation,

	Input:
	# Initial conditions
	if filename == None:
		Bx0  - dict, Inital magnetic field
		#Ux0  - dict, Inital Velocity field
	else:
		load IC's from filename	

	*args - list of solver arguments, see "main of the file" (below) for there definition

	args = [domain, IC, Rm, dt];
	
	Returns:
		final magnetic energy/time integrated magnetic energy
	
	- Writes the following to disk:
	1) FILE Scalar-data (every 20 iters): Magnetic Enegry, etc. 

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
	
	Bx0,Ux0 = np.split(X0,2)

	IVP_FWD = FWD_Solve_Build_Lin(domain, Rm, dt, Ux0); #Base_Bx0,Base_Ux0);

	Pi = IVP_FWD.state['Pi']; 
	A  = IVP_FWD.state['A'];	B  = IVP_FWD.state['B'];	C  = IVP_FWD.state['C']
	for f in [Pi, A,B,C]:
		f.set_scales(domain.dealias, keep_data=False)
		f['g'] = 0.

	#######################################################
	# set initial conditions
	#######################################################

	Vec_to_Field(domain,A,B,C,Bx0)

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
	analysis_CPT.add_task("uf", name='u-velocity',layout='g', scales=3/2); 
	analysis_CPT.add_task("vf", name='v-velocity',layout='g', scales=3/2);  
	analysis_CPT.add_task("wf", name='w-velocity',layout='g', scales=3/2); 

	analysis1 = IVP_FWD.evaluator.add_file_handler("scalar_data", iter=20, mode='overwrite');
	analysis1.add_task("inv_Vol*integ( A**2 + B**2 + C**2 )", name="Magnetic energy")

	#######################################################	
	logger.info("\n\n --> Timestepping FWD_Solve ");
	#######################################################

	N_PRINTS = N_SUB_ITERS//2;
		
	flow = flow_tools.GlobalFlowProperty(IVP_FWD, cadence=1);
	flow.add_property("inv_Vol*integ( A*A + B*B + C*C )", name='J(B)');
	#flow.add_property("abs(dx(A) + dy(B) + dz(C) 	  )", name='div_B');
	#flow.add_property("inv_Vol*integ( A + B + C 	  )", name='Flux_B');

	#dt = cfl.compute_dt();
	J_TRAP = 0.;
	snapshot_index = 0;
	while IVP_FWD.ok:

		# 1) Fill Dictionary
		if (IVP_FWD.iteration >= (N_ITERS - N_SUB_ITERS)) and (snapshot_index <= N_SUB_ITERS ):

			#X_FWD_DICT = {'A_fwd':A_SNAPS, 'B_fwd':B_SNAPS, 'C_fwd':C_SNAPS}
			X_FWD_DICT['A_fwd'][:,:,:,snapshot_index] = A['c'];
			X_FWD_DICT['B_fwd'][:,:,:,snapshot_index] = B['c'];
			X_FWD_DICT['C_fwd'][:,:,:,snapshot_index] = C['c'];
			snapshot_index+=1;


		IVP_FWD.step(dt);
		#'''
		if IVP_FWD.iteration % N_PRINTS == 0:
			
			logger.info('Iterations: %i' %IVP_FWD.iteration)
			logger.info('Sim time:   %f' %IVP_FWD.sim_time )

			logger.info('Mag (1/V)<B,B> = %e'%flow.volume_average('J(B)') 		  );
			#logger.info('Max |div(B)| 	= %e'%( flow.max('div_B')				) );
			#logger.info('Flux <B> 		= %e'%( flow.volume_average('Flux_B') 	) );
		#'''

		# 3) Evaluate Cost_function using flow tools, 
		# flow tools value is that of ( IVP_FWD.iteration-1 )
		if Cost_function == "Integrated": # J = int_t <B,B> dt 

			'''
			# Trapezoidal rule
			if ( (IVP_FWD.iteration-1) == 0):
				J_TRAP += dt*0.5*flow.volume_average('J(B)')
			elif (0 < (IVP_FWD.iteration-1) < N_ITERS):
				J_TRAP += dt*flow.volume_average('J(B)')
			elif ( (IVP_FWD.iteration-1) == N_ITERS): 
				J_TRAP += dt*0.5*flow.volume_average('J(B)');
			'''	
			
			# Simple Euler integration 1st order dt	
			if ((IVP_FWD.iteration-1) >= 0) and ((IVP_FWD.iteration-1) <= N_ITERS):	
				J_TRAP += dt*flow.volume_average('J(B)')	
			#'''	
		elif ( Cost_function == "Final") and ( (IVP_FWD.iteration-1) == N_ITERS): # J = <B_T,B_T>

			J_TRAP = flow.volume_average('J(B)');

		'''
		p = 1
		print(dir(IVP_FWD.pencils[p]))


		print( type(IVP_FWD.pencils[p].pre_left) )
		print( type(IVP_FWD.pencils[p].pre_right) )
		
		print(IVP_FWD.pencils[p].pre_left.shape)
		print(IVP_FWD.pencils[p].pre_right.shape)
		
		import matplotlib.pyplot as plt

		plt.spy(IVP_FWD.pencils[p].pre_left)
		plt.show()

		plt.spy(IVP_FWD.pencils[p].pre_right)
		plt.show()

		sys.exit();	
		'''
	#######################################################

	# final statistics
	post.merge_process_files("CheckPoints", cleanup=True, comm=MPI.COMM_WORLD);
	post.merge_process_files("scalar_data", cleanup=True, comm=MPI.COMM_WORLD);
	time.sleep(1);
	logger.info("\n\n--> Complete <--\n")

	logger.info('J(Bx0) = %e'%J_TRAP );

	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");

	return (-1.)*J_TRAP;

##########################################################################
# ~~~~~ ADJ Solvers + Comptability Condition ~~~~~~~~~~~~~
##########################################################################

def Compatib_Cond(X_FWD_DICT, domain, Rm, dt, Cost_function= "Final"):

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

	prob = de.LBVP(domain, variables=['Pi', 'A','B','C']);
	
	fx = new_ncc(domain);
	fy = new_ncc(domain);
	fz = new_ncc(domain);

	fx['c'] = X_FWD_DICT['A_fwd'][:,:,:,-1];
	fy['c'] = X_FWD_DICT['B_fwd'][:,:,:,-1];
	fz['c'] = X_FWD_DICT['C_fwd'][:,:,:,-1];

	prob.parameters['fx']= fx;
	prob.parameters['fy']= fy;
	prob.parameters['fz']= fz;

	prob.parameters['Rm']=Rm;
	prob.parameters['dt']=dt;
	
	prob.substitutions['Lap(f)'] = "dx(dx(f)) + dy(dy(f)) + dz(dz(f))";
	
	# Note minus on the RHS is due to the fact we chose to minimise rather than maximise	
	if Cost_function == "Final":
		prob.add_equation('A - dt*(.5/Rm)*Lap(A) - dx(Pi) = -2.*fx', condition="(nx != 0) or (ny != 0) or (nz != 0)");
		prob.add_equation('B - dt*(.5/Rm)*Lap(B) - dy(Pi) = -2.*fy', condition="(nx != 0) or (ny != 0) or (nz != 0)");
		prob.add_equation('C - dt*(.5/Rm)*Lap(C) - dz(Pi) = -2.*fz', condition="(nx != 0) or (ny != 0) or (nz != 0)");
		prob.add_equation('dx(A) + dy(B) + dz(C) = 0'			   , condition="(nx != 0) or (ny != 0) or (nz != 0)");

	elif Cost_function == "Integrated":
		prob.add_equation('A/dt - (.5/Rm)*Lap(A) - dx(Pi) = -2.*fx', condition="(nx != 0) or (ny != 0) or (nz != 0)");
		prob.add_equation('B/dt - (.5/Rm)*Lap(B) - dy(Pi) = -2.*fy', condition="(nx != 0) or (ny != 0) or (nz != 0)");
		prob.add_equation('C/dt - (.5/Rm)*Lap(C) - dz(Pi) = -2.*fz', condition="(nx != 0) or (ny != 0) or (nz != 0)");
		prob.add_equation('dx(A) + dy(B) + dz(C) = 0'			   , condition="(nx != 0) or (ny != 0) or (nz != 0)");
	

	prob.add_equation("A  = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)");
	prob.add_equation("B  = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)");
	prob.add_equation("C  = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)");
	prob.add_equation("Pi = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)");
	
	solver = prob.build_solver();
	solver.solve();

	A = solver.state['A'];
	B = solver.state['B'];
	C = solver.state['C'];

	for f in [A,B,C]:
		f.set_scales(domain.dealias, keep_data=True);

	for h in root.handlers:
		h.setLevel("WARNING");
		#h.setLevel("INFO"); #h.setLevel("DEBUG")

	return {'Bx':A,'By':B,'Bz':C};

Adjoint_type = "Discrete";
#Adjoint_type = "Continuous";

def ADJ_Solve_IVP_Lin(X0, domain,Rm,dt,  N_ITERS,N_SUB_ITERS, X_FWD_DICT, Cost_function= "Final"):	
	"""
	Driver program for Periodic Box Dynamo, which builds the forward solver object with options:

	Inputs:
	domain (dedalus object) returned by ??
	flow paramaters Rm - Magnetic Reynolds number

	dt, ts - float numerical integration time-step & dedalus timestepper object

	Returns:
	Dedalus object to solve the MHD equations in a periodic box geometry

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

	# Equations Variables 
	# \vec{v}  = nu_u x^ + nu_v y^ + nu_w z^ 	&		 \vec{B}  = G_A x^ + G_B y^ + G_C z^
	#PBox = de.IVP(domain, variables=['Pi', 'G_A', 'G_B', 'G_C'], time='t');
	PBox = de.IVP(domain, variables=['Pi', 'G_A', 'G_B', 'G_C',		'P', 'nu_u', 'nu_v', 'nu_w'], time='t');
	PBox.parameters['Rm'] = Rm;
	PBox.parameters['inv_Vol'] = 1./domain.hypervolume;

	Bx0,Ux0 = np.split(X0,2)

	# Fields for linearisation
	#~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~
	U = new_ncc(domain);
	V = new_ncc(domain);
	W = new_ncc(domain);

	Vec_to_Field(domain,U,V,W,Ux0);

	PBox.parameters['uf'] = U
	PBox.parameters['vf'] = V
	PBox.parameters['wf'] = W
	
	#~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~
	A_f = new_ncc(domain);	
	B_f = new_ncc(domain);
	C_f = new_ncc(domain);
	
	PBox.parameters['Af'] = A_f; 
	PBox.parameters['Bf'] = B_f; 
	PBox.parameters['Cf'] = C_f;

	#######################################################
	# Substitutions
	#######################################################
	# 1.A) ~~~~~~ Laplacian^2 ~~~~~~~~~~~~~~
	PBox.substitutions['Lap(f)'] = "dx(dx(f)) + dy(dy(f)) + dz(dz(f))";

	# 1.B) ~~~~~~ (NAB x A) 	~~~~~~~~~~ Correct
	PBox.substitutions['W_x(A_y ,A_z )'] = "dy(A_z) - dz(A_y)"; 
	PBox.substitutions['W_y(A_x ,A_z )'] = "dz(A_x) - dx(A_z)"; 
	PBox.substitutions['W_z(A_x ,A_y )'] = "dx(A_y) - dy(A_x)";

	# 1.B) ~~~~~~ (NAB x A) X B ~~~~~~~~~~ Correct
	PBox.substitutions['F_x(A_x,A_y,A_z,	B_x,B_y,B_z)'] = "W_y(A_x ,A_z)*B_z - W_z(A_x ,A_y)*B_y"; 
	PBox.substitutions['F_y(A_x,A_y,A_z,	B_x,B_y,B_z)'] = "W_z(A_x ,A_y)*B_x - W_x(A_y ,A_z)*B_z"; 
	PBox.substitutions['F_z(A_x,A_y,A_z,	B_x,B_y,B_z)'] = "W_x(A_y ,A_z)*B_y - W_y(A_x ,A_z)*B_x";

	#######################################################
	# add equations
	#######################################################
	logger.info("--> Adding Equations");

	PBox.add_equation("dx(G_A) + dy(G_B) + dz(G_C) = 0", 								      condition="(nx != 0) or (ny != 0) or (nz != 0)");
	if Cost_function == "Final":
		PBox.add_equation("dt(G_A) - (1./Rm)*Lap(G_A) - dx(Pi) = F_x(G_A,G_B,G_C, uf,vf,wf)", condition="(nx != 0) or (ny != 0) or (nz != 0)");
		PBox.add_equation("dt(G_B) - (1./Rm)*Lap(G_B) - dy(Pi) = F_y(G_A,G_B,G_C, uf,vf,wf)", condition="(nx != 0) or (ny != 0) or (nz != 0)");   
		PBox.add_equation("dt(G_C) - (1./Rm)*Lap(G_C) - dz(Pi) = F_z(G_A,G_B,G_C, uf,vf,wf)", condition="(nx != 0) or (ny != 0) or (nz != 0)");
	
	elif Cost_function == "Integrated":	
		PBox.add_equation("dt(G_A) - (1./Rm)*Lap(G_A) - dx(Pi) = -2.*Af + F_x(G_A,G_B,G_C, uf,vf,wf)", condition="(nx != 0) or (ny != 0) or (nz != 0)");
		PBox.add_equation("dt(G_B) - (1./Rm)*Lap(G_B) - dy(Pi) = -2.*Bf + F_y(G_A,G_B,G_C, uf,vf,wf)", condition="(nx != 0) or (ny != 0) or (nz != 0)");   
		PBox.add_equation("dt(G_C) - (1./Rm)*Lap(G_C) - dz(Pi) = -2.*Cf + F_z(G_A,G_B,G_C, uf,vf,wf)", condition="(nx != 0) or (ny != 0) or (nz != 0)");

	# Zero the constant part
	PBox.add_equation("G_A  = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)");
	PBox.add_equation("G_B  = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)");
	PBox.add_equation("G_C  = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)");
	PBox.add_equation("Pi   = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)");

	# For the gradient dL/du
	#~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~
	PBox.add_equation("dx(nu_u) + dy(nu_v) + dz(nu_w) = 0", 		    condition="(nx != 0) or (ny != 0) or (nz != 0)");	
	PBox.add_equation("dt(nu_u) + dx(P) = -F_x(G_A,G_B,G_C, Af,Bf,Cf)", condition="(nx != 0) or (ny != 0) or (nz != 0)");
	PBox.add_equation("dt(nu_v) + dy(P) = -F_y(G_A,G_B,G_C, Af,Bf,Cf)", condition="(nx != 0) or (ny != 0) or (nz != 0)");   
	PBox.add_equation("dt(nu_w) + dz(P) = -F_z(G_A,G_B,G_C, Af,Bf,Cf)", condition="(nx != 0) or (ny != 0) or (nz != 0)");

	PBox.add_equation("nu_u  = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)");
	PBox.add_equation("nu_v  = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)");
	PBox.add_equation("nu_w  = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)");
	PBox.add_equation("P     = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)");
	#~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	IVP_ADJ = PBox.build_solver(de.timesteppers.CNAB1);

		
	Pi   = IVP_ADJ.state['Pi'];
	G_A  = IVP_ADJ.state['G_A']; G_B  = IVP_ADJ.state['G_B']; G_C  = IVP_ADJ.state['G_C'];
	
	P     = IVP_ADJ.state['P'];
	nu_u  = IVP_ADJ.state['nu_u']; nu_v  = IVP_ADJ.state['nu_v']; nu_w  = IVP_ADJ.state['nu_w'];
	
	for f in [Pi, G_A,G_B,G_C,		P, nu_u,nu_v,nu_w]:
		f.set_scales(domain.dealias, keep_data=False)
		f['g'] = 0.	

	#######################################################
	# set initial conditions
	#######################################################
	
	#X_FWD_DICT = {'A_fwd':A_SNAPS, 'B_fwd':B_SNAPS, 'C_fwd':C_SNAPS}
	if Adjoint_type == "Continuous":
	
		G_A['c'] = -2.*X_FWD_DICT['A_fwd'][:,:,:,-1];
		G_B['c'] = -2.*X_FWD_DICT['B_fwd'][:,:,:,-1]; 
		G_C['c'] = -2.*X_FWD_DICT['C_fwd'][:,:,:,-1];
		
		snapshot_index = -1; # Continuous 

	elif Adjoint_type == "Discrete":

		X_DICT = Compatib_Cond(X_FWD_DICT, domain, Rm, dt,Cost_function);

		G_A['g'] = X_DICT['Bx']['g'];
		G_B['g'] = X_DICT['By']['g'];
		G_C['g'] = X_DICT['Bz']['g'];
		
		snapshot_index = -2; # Discrete due to shifting ofindicies

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
	#'''
	N_PRINTS = N_SUB_ITERS//10;
	from dedalus.extras import flow_tools
	flow = flow_tools.GlobalFlowProperty(IVP_ADJ, cadence=1);
	flow.add_property("inv_Vol*integ( G_A**2  + G_B**2  + G_C**2  )", name='J(B)');
	flow.add_property("abs(			  dx(G_A) + dy(G_B) + dz(G_C) )", name='div_B');
	flow.add_property("inv_Vol*integ( G_A + G_B + G_C 	          )", name='Flux_B');

	flow.add_property("abs(		   dx(nu_u) + dy(nu_v) + dz(nu_w) )", name='div_U');
	#'''

	# Outlined above
	#snapshot_index = -2; # Discrete due to shifting ofindicies
	#snapshot_index = -1; # Continuous 
	while IVP_ADJ.ok:

		# Must leave this as it's needed for the dL/dU-gradient
		#X_FWD_DICT = {'A_fwd':A_SNAPS, 'B_fwd':B_SNAPS, 'C_fwd':C_SNAPS}
		IVP_ADJ.problem.namespace['Af']['c'] = X_FWD_DICT['A_fwd'][:,:,:,snapshot_index]
		IVP_ADJ.problem.namespace['Bf']['c'] = X_FWD_DICT['B_fwd'][:,:,:,snapshot_index]
		IVP_ADJ.problem.namespace['Cf']['c'] = X_FWD_DICT['C_fwd'][:,:,:,snapshot_index]
		snapshot_index-=1; # Move back in time (<-t


		IVP_ADJ.step(dt);

		#print(IVP_ADJ.timestepper.RHS.data.shape)
		#'''
		if IVP_ADJ.iteration % 1 == 0:
			
			logger.info('Iterations: %i' %IVP_ADJ.iteration)
			logger.info('Sim time: %f, Time-step dt: %f' %(IVP_ADJ.sim_time,dt));

			logger.info('Mag (1/V)<B,B> = %e'%( flow.volume_average('J(B)') 		) );
			logger.info('Max |div(B)| 	= %e'%( flow.max('div_B')				) );
			logger.info('Max |div(U)| 	= %e'%( flow.max('div_U')				) );
			logger.info('Flux <B> 		= %e'%( flow.volume_average('Flux_B') 	) );
		#'''

	#######################################################

	# For discrete adjoint undo LHS inversion of the last-step
	if Adjoint_type == "Discrete":
		
		for f in [A_f,B_f,C_f]:
			f.set_scales(domain.dealias, keep_data=False)	
			f['g']=0.;

		A_f['g'] = dt*( G_A['g']/dt - (.5/Rm)*( G_A.differentiate('x').differentiate('x')['g'] + G_A.differentiate('y').differentiate('y')['g'] + G_A.differentiate('z').differentiate('z')['g'] ) );
		B_f['g'] = dt*( G_B['g']/dt - (.5/Rm)*( G_B.differentiate('x').differentiate('x')['g'] + G_B.differentiate('y').differentiate('y')['g'] + G_B.differentiate('z').differentiate('z')['g'] ) );
		C_f['g'] = dt*( G_C['g']/dt - (.5/Rm)*( G_C.differentiate('x').differentiate('x')['g'] + G_C.differentiate('y').differentiate('y')['g'] + G_C.differentiate('z').differentiate('z')['g'] ) );
		
		Bx0 = Field_to_Vec(domain,A_f ,B_f ,C_f )
	
	else:
		
		Bx0 = Field_to_Vec(domain,G_A ,G_B ,G_C )
	
	Ux0 = Field_to_Vec(domain,nu_u,nu_v,nu_w); #

	logger.info("\n\n--> Complete <--\n")

	# Set to info level rather than the debug default
	for h in root.handlers:
		#h.setLevel("WARNING");
		h.setLevel("INFO");

	return np.concatenate( (Bx0,Ux0) );
	#return np.concatenate( (Bx0,np.zeros(Ux0.shape)) );


if __name__ == "__main__":


	Rm = 1.; dt = 1e-03; Npts = 24; 
	N_ITERS = int(Rm/dt); N_SUB_ITERS = N_ITERS//1;
	X_domain = (0.,2.*np.pi); M_0 = 1.; E_0 = 1.; Noise = True;

	Cost_function = "Final"; 
	#Cost_function = "Integrated"

	domain, Bx0, Ux  = Generate_IC(Npts,X_domain,M_0,Noise);
	X0 = np.concatenate((Bx0,Ux));

	X_FWD_DICT = GEN_BUFFER(Npts, domain, N_SUB_ITERS)

	args = [domain,Rm,dt,   N_ITERS,N_SUB_ITERS, X_FWD_DICT, Cost_function];

	#'''
	# FWD Solve
	#J_obj = FWD_Solve_IVP_Lin(X0,*args);

	# ADJ Solve
	#dJdB0 = ADJ_Solve_IVP_Lin(X0,*args);
	#sys.exit()
	#'''

	from DAL_PCF_MAIN import DAL_LOOP_OLD_VER as DAL_LOOP
	DAL_LOOP(X0,[M_0,E_0], *args); # Remember to return (dJ_dB0, 0) only
	
	#from DAL_PCF_MAIN import DAL_LOOP
	#DAL_LOOP(X0,[M_0,E_0], *args)

	'''
	from TestGrad import Adjoint_Gradient_Test
	Noise = True;
	XdomainX, dBx0, dUx  = Generate_IC(Npts,X_domain,M_0,Noise);
	ZEROS = np.zeros(dUx.shape)
	#dX0 = np.concatenate((dBx0 ,ZEROS));
	dX0 = np.concatenate((ZEROS,dUx  ));
	Adjoint_Gradient_Test(X0,dX0,	*args)
	'''
