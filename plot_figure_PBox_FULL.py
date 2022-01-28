import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, cm
import h5py, sys, os

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#import logging
#logger = logging.getLogger(__name__)

#plt.rcParams['pcolor.shading']
#plt.style.use('./methods_paper.mplstyle'); # Used to set font_size ticks size font type defaults etc

##########################################################################
# Scalar Data Plot functions
##########################################################################

def Plot_KinematicB_scalar_data(file_names,LEN,CAD,Normalised_B,Logscale_B):

	"""
	Plot the Magnetic fields integrated energies, as determined by the Magnetic Iduction equation 

	Input Parameters:

	file_names = ['one_file','second_file'] - array like, with file-types .hdf5
	LEN - integer: Length of the filenames array
	CAD - int: the cadence at which to plot of the details of files contained
	Normalied_B - bool: Show the magnetic fields evolution in terms of amplification or its magnitude

	Returns: None
	
	"""
	if Logscale_B == True:
		outfile = 'Linear_Kinetic_B_Logscale.pdf'; 
	else:	
		outfile = 'Linear_Kinetic_B.pdf'; 
	
	dpi = 400;
	fig,a =  plt.subplots(1,2,figsize=(8,6));
	
	for i in range(0,LEN,CAD):

		file = h5py.File(file_names[i],"r")
		#print(file['scales/'].keys()); print(file['tasks/'].keys()) #useful commands	

		# Set the time interval we take
		index = 0; index_end = -1;
		time = file['scales/sim_time'][:];
		x = time - time[0]*np.ones(len(time)); # Modify time so that it's zero'd
		x = x[index:index_end]
		
		BE = file['tasks/Magnetic energy'][index:index_end,0,0,0];
		'''
		if Normalised_B == True:

			BE = BE/BE[0];
			LABEL = r'$<B,B>/<B_0,B_0>$';

		elif Logscale_B == True:
			
			BE = np.log10(BE);
			LABEL = r'$\ln_{10} (<B,B>) $'

		else:
			LABEL = r'$<B,B>$'	
		'''
		LABEL = r'$<B,B>$'	
		a[0].plot(x,np.log10(BE),'-' ,label=r'$<B^2>_{i=%i}$'%i);#,fontsize=25);
		a[1].plot(x,BE          ,'-.',label=r'$<B^2>_{i=%i}$'%i);#,fontsize=25);

	# Set their labels
	a[0].set_title(r'$\log10( B(x,t) )$ Magentic Energy')
	a[0].set_ylabel(LABEL);
	a[0].legend()
	a[0].set_xlim([np.min(x),np.max(x)])
	a[0].grid()

	a[1].set_title(r'$B(x,t)$ Magentic Energy');
	a[1].set_ylabel(LABEL);
	a[1].legend()
	a[1].set_xlim([np.min(x),np.max(x)])
	a[1].grid()

	plt.tight_layout(pad=1, w_pad=1.5)
	fig.savefig(outfile, dpi=dpi);

	plt.show()

	return None;

##########################################################################
# Plot a vec=U,B pair
##########################################################################

# CHECK
def Plot_UB_pair(file_names,times,LEN,CAD,Just_B):

	"""
	Plot the Magnetic & Velocity (Optional) fields, as determined by:
	1) the Magnetic Iduction equation 
	2) Full MHD equations

	Input Parameters:

	file_names = ['one_file','second_file'] - array like, with file-types .hdf5
	times - integer: indicies of the CheckPoint(s) you wish to plot.... Depends on how the analysis_tasks are defined in FWD_Solve_TC_MHD
	LEN - integer: Length of the filenames array
	CAD - int: the cadence at which to plot of the details of files contained
	Just_B - bool: If True Plots only the magnetic field components
	
	Returns: None
	
	"""

	for k in range(0,LEN,CAD):

		file = h5py.File(file_names[k],"r")
		print(file['scales/'].keys()); print(file['tasks/'].keys()) #useful commands

		x = file['scales/x/1.5']; y = file['scales/y/1.5']; z = file['scales/z/1.5']
		#x = file['scales/x/1.0']; y = file['scales/y/1.0']; z = file['scales/z/1.0']
		#(time,x,y,z)
		
		if Just_B == False:
			#u   = file['tasks/u']; v   = file['tasks/v']; w   = file['tasks/w'];
			u   = file['tasks/u-velocity']; v   = file['tasks/v-velocity']; w   = file['tasks/w-velocity'];
		
		B_x = file['tasks/A']; B_y = file['tasks/B']; B_z = file['tasks/C'];
		'''
		if Just_B == False:
			u   = file['tasks/nu_u']; v   = file['tasks/nu_v']; w   = file['tasks/nu_w'];
		
		B_x = file['tasks/G_A']; B_y = file['tasks/G_B']; B_z = file['tasks/G_C'];
		'''
		SLICE = 12; # This needs some modification

		for i in range(len(times)):
			
			index = times[i]; # The instant at which we plot out the vector

			outfile_U = "".join(['U_PLOTS_Iter_i%i_Time_t%i.pdf'%(k,index) ]);	
			outfile_B = "".join(['B_PLOTS_Iter_i%i_Time_t%i.pdf'%(k,index) ]);	
			
			##########################################################################
			# ~~~~~~~~~~~~~~~~~~~ plotting Magnetic B ~~~~~~~~~~~~~~~~~~~~~
			##########################################################################

			fig = plt.figure(figsize=(8,6))
			plt.suptitle("B Field Iter=i%i Time=t%i"%(k,index) ); dpi = 400;

			'''
			print("B_x shape =",B_x[index,SLICE,:,:].shape);
			#print("y =",y.shape);
			#print("z =",z.shape);
			maxBx = np.amax(B_x[i,:,:,:]);
			minBx = np.amin(B_x[i,:,:,:]);
			
			maxBy = np.amax(B_y[i,:,:,:]);
			minBy = np.amin(B_y[i,:,:,:]);

			maxBz = np.amax(B_z[i,:,:,:]);
			minBz = np.amin(B_z[i,:,:,:]);
			
			print("max(Bx) =",maxBx)
			print("min(Bx) =",minBx)
			
			print("max(Bz) =",maxBz)
			print("min(Bz) =",minBz)
			
			print("min(By) =",minBy)
			print("max(By) =",maxBy,"\n")
			'''

			#------------------------------------ #------------------------------------
			ax1 = plt.subplot(221)
			Y,Z = np.meshgrid(y,z);
			cs = ax1.contourf(Z,Y,B_x[index,SLICE,:,:].T,cmap='PuOr',levels=30)
			
			skip=(slice(None,None,2),slice(None,None,2))
			ax1.quiver(Z[skip],Y[skip],B_z[index,SLICE,:,:][skip].T,B_y[index,SLICE,:,:][skip].T,width=0.005);
			fig.colorbar(cs,ax=ax1);

			ax1.set_title(r'$B_x$, vecs - $(B_z,B_y)$');
			ax1.set_xlabel(r'Shearwise - $z$')#, fontsize=18)
			ax1.set_ylabel(r'Spanwise  - $y$')#, fontsize=18)

			#------------------------------------ #------------------------------------
			ax2 = plt.subplot(222)
			X,Z = np.meshgrid(x,z);
			SLICE_y = 12
			cs = ax2.contourf(Z,X,B_y[index,:,SLICE_y,:].T,cmap='PuOr',levels=30)
			
			skip=(slice(None,None,4),slice(None,None,4))
			ax2.quiver(Z[skip],X[skip], B_z[index,:,SLICE_y,:][skip].T,B_x[index,:,SLICE_y,:][skip].T,width=0.005);
			fig.colorbar(cs,ax=ax2);

			ax2.set_title(r'$B_y$, vecs - ($B_z,B_x$)');
			ax2.set_xlabel(r'Shearwise  - $z$')
			ax2.set_ylabel(r'Streamwise - $x$')

			#------------------------------------ #------------------------------------
			ax3 = plt.subplot(212)
			Y,X = np.meshgrid(y,x);
			#print("X =",X.shape);
			#print("Y =",Y.shape);
			#print("Bz =",B_z[index,:,:,SLICE].shape)
			cs = ax3.contourf(X,Y,B_y[index,:,:,SLICE],cmap='PuOr',levels=30)
			
			skip=(slice(None,None,4),slice(None,None,4))
			ax3.quiver(X[skip],Y[skip], B_x[index,:,:,SLICE][skip],B_y[index,:,:,SLICE][skip],width=0.005);
			fig.colorbar(cs,ax=ax3)#,loc='right');

			ax3.set_title(r'$B_y$, vecs - ($B_x,B_y$)');
			ax3.set_xlabel(r'Streamwise - $x$');#, fontsize=18)
			ax3.set_ylabel(r'Spanwise   - $y$')

			#------------------------------------ #------------------------------------
			# Save figure
			plt.tight_layout(pad=1, w_pad=1.5)
			fig.savefig(outfile_B, dpi=dpi)
			#plt.show()


			##########################################################################
			# ~~~~~~~~~~~~~~~~~~~ plotting Velocity U ~~~~~~~~~~~~~~~~~~~~~
			##########################################################################
			if Just_B == False:

				fig = plt.figure(figsize=(8,6))
				plt.suptitle("U Field Iter=i%i Time=t%i"%(k,index) ); dpi = 400;

				#------------------------------------ #------------------------------------
				ax1 = plt.subplot(221)
				Y,Z = np.meshgrid(y,z);
				cs = ax1.contourf(Z,Y,u[index,SLICE,:,:].T,cmap='PuOr',levels=10)

				skip=(slice(None,None,2),slice(None,None,2))
				ax1.quiver(Z[skip],Y[skip],w[index,SLICE,:,:][skip].T,v[index,SLICE,:,:][skip].T,width=0.005);
				fig.colorbar(cs,ax=ax1);

				ax1.set_title(r'$u$, vecs - $(w,v)$');
				ax1.set_xlabel(r'Shearwise - $z$')#, fontsize=18)
				ax1.set_ylabel(r'Spanwise  - $y$')#, fontsize=18)

				#------------------------------------ #------------------------------------
				ax2 = plt.subplot(222)
				X,Z = np.meshgrid(x,z);
				cs = ax2.contourf(Z,X,w[index,:,SLICE,:].T,cmap='PuOr',levels=10)
				
				skip=(slice(None,None,4),slice(None,None,4))
				ax2.quiver(Z[skip],X[skip], w[index,:,SLICE,:][skip].T,u[index,:,SLICE,:][skip].T,width=0.005);
				fig.colorbar(cs,ax=ax2);

				ax2.set_title(r'$w$, vecs - ($w,u$)');
				ax2.set_xlabel(r'Shearwise  - $z$')
				ax2.set_ylabel(r'Streamwise - $x$')

				#------------------------------------ #------------------------------------
				ax3 = plt.subplot(212)
				Y,X = np.meshgrid(y,x);
				#print("X =",X.shape);
				#print("Y =",Y.shape);
				#print("Bz =",B_z[index,:,:,SLICE].shape)
				cs = ax3.contourf(X,Y,w[index,:,:,SLICE],cmap='PuOr',levels=10)
				
				skip=(slice(None,None,4),slice(None,None,4))
				ax3.quiver(X[skip],Y[skip], u[index,:,:,SLICE][skip],v[index,:,:,SLICE][skip],width=0.005);
				fig.colorbar(cs,ax=ax3)#,loc='right');

				ax3.set_title(r'$w$, vecs - ($u,v$)');
				ax3.set_xlabel(r'Streamwise - $x$');#, fontsize=18)
				ax3.set_ylabel(r'Spanwise   - $y$')

				#------------------------------------ #------------------------------------
				# Save figure
				plt.tight_layout(pad=1, w_pad=1.5)
				fig.savefig(outfile_U, dpi=dpi)
				#plt.show()

	return None;

##########################################################################
# Plot Kinetc <u,u>(k,m) & Magentic <B,B>(k,m) as 3D surfaces
########################################################################## 
'''def Plot_KUKB_pair(file_names,times,LEN,CAD,Just_B):

	"""
	Plot Kinetc <u,u>(k,m) & Magentic <B,B>(k,m) as 3D surfaces

	The Integrals are indexed by the waveniumbers as follows
	# k - axial wavenumber, 
	# m - azim  wavenumber, 
	# Radial dependancy has been integrated out

	Input Parameters:

	file_names = ['one_file','second_file'] - array like, with file-types .hdf5
	times - integer: indicies of the CheckPoint(s) you wish to plot.... Depends on how the analysis_tasks are defined in FWD_Solve_TC_MHD
	LEN - integer: Length of the filenames array
	CAD - int: the cadence at which to plot of the details of files contained
	Just_B - bool: If True Plots only the magnetic field components

	Returns: None
	
	"""
	
	#from mpl_toolkits.mplot3d import Axes3D
	#from matplotlib import cm
	#from matplotlib.ticker import LinearLocator, FormatStrFormatter

	for k in range(0,LEN,CAD):
	
		# Make data.
		file = h5py.File(file_names[k],"r")
		#print(file['tasks/'].keys()); print(file['scales/'].keys()); 

		# Get wavenumbers and create a Mesh-grid for plotting
		ks = file['scales/ks']; kz = file['scales/kz']
		Nz = int( 1 + (len(kz[:]) - 1)/2);
		kz = kz[0:Nz];  ks = ks[:];
		X, Y = np.meshgrid(ks, kz);

		for i in range(len(times)):

			index = times[i]; # The instant at which we plot out the vector

			outfile_U = "".join(['KE_PLOTS_Iter_i%i_Time_t%i.pdf'%(k,index) ]);	
			outfile_B = "".join(['KB_PLOTS_Iter_i%i_Time_t%i.pdf'%(k,index) ]);

			##########################################################################
			# ~~~~~~~~~~~~~~~~~~~ plotting Velocity U ~~~~~~~~~~~~~~~~~~~~~
			##########################################################################

			if Just_B == False:
				
				# KE(k,m) = FFT{ int_r U**2 dr }
				U = file['tasks/KE per k'][index,:,0:Nz,0]; #(time,phi,z,0)
				KE = np.log10(abs(U) + 1e-16);

				fig = plt.figure(figsize=(8,6)); dpi = 400;
				ax = fig.gca(projection='3d');
				plt.title("Energy <U,U>(k,m) Field Iter=i%i Time=t%i"%(k,index) );

				# Plot the surface.
				surf = ax.plot_surface(X, Y, KE[:,::-1].T, cmap=cm.coolwarm,linewidth=0, antialiased=False);

				# Customize the z axis.
				ax.set_zlim(-15.,2.)
				ax.set_xlabel(r'$m_{\phi}$ - azimuthal',fontsize=18);
				ax.set_ylabel(r'$k_{z}$ - axial rev',fontsize=18);
				ax.set_zlabel(r'$log10(\hat{E}_U(m_{\phi},k_{z}))$',fontsize=18);

				# Save figure
				plt.tight_layout(pad=1, w_pad=1.5)
				fig.savefig(outfile_U, dpi=dpi)
				#plt.show()

			##########################################################################
			# ~~~~~~~~~~~~~~~~~~~ plotting Magnetic B ~~~~~~~~~~~~~~~~~~~~~
			##########################################################################

			# BE(k,m) = FFT{ int_r B**2 dr }
			B = file['tasks/BE per k'][index,:,0:Nz,0]; #(time,phi,z,0)
			BE = np.log10(abs(B) + 1e-16);

			fig = plt.figure(figsize=(8,6)); 
			ax = fig.gca(projection='3d'); dpi = 400;
			plt.title("Energy <B,B>(k,m) Field Iter=i%i Time=t%i"%(k,index) );

			# Plot the surface.
			surf = ax.plot_surface(X, Y, BE[:,::-1].T, cmap=cm.coolwarm,linewidth=0, antialiased=False);

			# Customize the z axis.
			ax.set_zlim(-15.,2.)
			#ax.zaxis.set_major_locator(LinearLocator(10))
			#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
			ax.set_xlabel(r'$m_{\phi}$ - azimuthal',fontsize=18);
			ax.set_ylabel(r'$k_{z}$ - axial rev',fontsize=18);
			ax.set_zlabel(r'$log10(\hat{E}_B(m_{\phi},k_{z}))$',fontsize=18);

			# Save figure
			plt.tight_layout(pad=1, w_pad=1.5)
			fig.savefig(outfile_B, dpi=dpi)
			#plt.show()

	return None;'''


##########################################################################
# Plot Kinetc <u,u>(k,m) & Magentic <B,B>(k,m) as 3D surfaces
########################################################################## 
def Plot_KUKB_pair(file_names,times,LEN,CAD,Just_B):

	"""
	Plot Kinetc <u,u>(k,m) & Magentic <B,B>(k,m) as 3D surfaces

	The Integrals are indexed by the waveniumbers as follows
	# k - axial wavenumber, 
	# m - azim  wavenumber, 
	# Radial dependancy has been integrated out

	Input Parameters:

	file_names = ['one_file','second_file'] - array like, with file-types .hdf5
	times - integer: indicies of the CheckPoint(s) you wish to plot.... Depends on how the analysis_tasks are defined in FWD_Solve_TC_MHD
	LEN - integer: Length of the filenames array
	CAD - int: the cadence at which to plot of the details of files contained
	Just_B - bool: If True Plots only the magnetic field components

	Returns: None
	
	"""
	
	#from mpl_toolkits.mplot3d import Axes3D
	#from matplotlib import cm
	#from matplotlib.ticker import LinearLocator, FormatStrFormatter

	for k in range(0,LEN,CAD):
	
		# Make data.
		file = h5py.File(file_names[k],"r")
		#print(file['tasks/'].keys()); print(file['scales/'].keys()); 
		
		# Get wavenumbers and create a Mesh-grid for plotting
		kx = file['scales/kx']; ky = file['scales/ky']
		print(len(kx[:]));
		print(len(ky[:]));

		print("kx = ",kx[:],"\n")

		Ny = int( 1 + (len(ky[:]) - 1)/2);
		inds_z = np.r_[0:Ny]; #Nz+1:len(kz[:])
		print(inds_z)
		print(ky[inds_z])
		ky = ky[inds_z];#[::-1];
		kx = kx[:];
		X, Y = np.meshgrid(kx, ky);
		print("X = ",X.shape)
		print("Y = ",Y.shape)	
		#sys.exit();

		for i in range(len(times)):

			index = times[i]; # The instant at which we plot out the vector

			outfile_U = "".join(['KE_PLOTS_Iter_i%i_Time_t%i.pdf'%(k,index) ]);	
			outfile_B = "".join(['KB_PLOTS_Iter_i%i_Time_t%i.pdf'%(k,index) ]);

			outfile_UB = "".join(['KE_and KB_PLOTS_Iter_i%i_Time_t%i.pdf'%(k,index) ]);	
			##########################################################################
			# ~~~~~~~~~~~~~~~~~~~ plotting Magnetic B ~~~~~~~~~~~~~~~~~~~~~
			##########################################################################

			if Just_B == True:

				# BE(kx,ky) = FFT{ int_z B**2 dz }
				B = file['tasks/BE per k'][index,:,inds_z,0]; #(time,k_x,k_y,0)
				BE = np.log10(abs(B) + 1e-16);

				fig = plt.figure(figsize=(8,6)); 
				ax = fig.gca(projection='3d'); dpi = 400;
				plt.title("Energy <B,B>(k,m) Field Iter=i%i Time=t%i"%(k,index) );

				# Plot the surface.
				surf = ax.plot_surface(X, Y, BE[:,:].T, cmap=cm.Greys,linewidth=0, antialiased=False);

				# Customize the z axis.
				ax.set_zlim(-15.,2.)
				#ax.zaxis.set_major_locator(LinearLocator(10))
				#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
				ax.set_xlabel(r'$k_x$ - Streamwise',fontsize=18);
				ax.set_ylabel(r'$k_y$ - Spanwise  ',fontsize=18);
				ax.set_zlabel(r'$log10(\hat{E}_B(k_x,k_y))$',fontsize=18);

				# Save figure
				plt.tight_layout(pad=1, w_pad=1.5)
				fig.savefig(outfile_B, dpi=dpi)
				#plt.show()
			elif Just_B == False:	
				##########################################################################
				# ~~~~~~~~~~~~~~~~~~~ plotting Velocity B ~~~~~~~~~~~~~~~~~~~~~
				##########################################################################
				
				# BE(kx,ky) = FFT{ int_z B**2 dz }
				B = file['tasks/BE per k'][index,:,inds_z,0]; #(time,k_x,k_y,0)
				BE = np.log10(abs(B) + 1e-16);

				#fig = plt.figure(figsize=(8,6)); 
				fig = plt.figure(figsize=plt.figaspect(0.5))
				ax = fig.add_subplot(1, 2, 1, projection='3d'); dpi = 1200;
				#ax = fig.gca(projection='3d'); 
				plt.title(r'Energy $\hat{E}_B = <B,B>(k_x,k_y)$',fontsize=12)# Field Iter=i%i Time=t%i'%(k,index) );

				#cmaps['Perceptually Uniform Sequential'] = ['viridis', 'plasma', 'inferno', 'magma', 'cividis'];
				# Plot the surface.
				surf = ax.plot_surface(X, Y, BE[:,:].T, cmap=cm.Greys,linewidth=0, antialiased=False);

				# Customize the z axis.
				ax.set_zlim(np.min(BE),np.max(BE))
				ax.set_xlim(0,np.max(kx))
				ax.set_ylim(0,np.max(ky));

				#ax.zaxis.set_major_locator(LinearLocator(10))
				#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
				ax.set_xlabel(r'$k_x$ - Streamwise',fontsize=12);
				ax.set_ylabel(r'$k_y$ - Spanwise  ',fontsize=12);
				#ax.set_zlabel(r'$log_{10}(\hat{E}_B(m_{\phi},k_{z}))$',fontsize=12);

				ax.view_init(30, 30)

				##########################################################################
				# ~~~~~~~~~~~~~~~~~~~ plotting Magnetic U ~~~~~~~~~~~~~~~~~~~~~
				##########################################################################

				# KE(k,m) = FFT{ int_r U**2 dr }
				U = file['tasks/KE per k'][index,:,inds_z,0]; #(time,phi,z,0)
				KE = np.log10(abs(U) + 1e-16);

				ax = fig.add_subplot(1, 2, 2, projection='3d')
				plt.title(r'Energy $\hat{E}_U = <U,U>(k_x,k_y)$',fontsize=12)# Field Iter=i%i Time=t%i'%(k,index) );

				# Plot the surface.
				surf = ax.plot_surface(X, Y, KE[:,:].T, cmap=cm.Greys,linewidth=0, antialiased=False);
				#ax.plot_wireframe(X, Y, KE[:,::-1].T, rstride=10, cstride=10)

				# Customize the z axis.
				ax.set_zlim(np.min(KE),np.max(KE))
				ax.set_xlim(0,np.max(kx))
				ax.set_ylim(0,np.max(ky));

				ax.set_xlabel(r'$k_x$ - Streamwise',fontsize=12);
				ax.set_ylabel(r'$k_y$ - Spanwise  ',fontsize=12);
				#ax.set_zlabel(r'$log_{10}(\hat{E}_U(m_{\phi},k_{z}))$',fontsize=12);

				ax.view_init(30,30)

				# Save figure
				plt.tight_layout(pad=1, w_pad=1.5)
				fig.savefig(outfile_UB, dpi=dpi)
				#plt.show()

			

	return None;


##########################################################################
# Plot Kinetc <u,u>(t) & Magentic <B,B>(t) scalar data
########################################################################## 
def Plot_UB_scalar_data_General(file_names,Legend_Names,LEN,CAD,Normalised_B,Logscale,outfile):

	"""
	Plot the Velocity field & Magnetic fields integrated energies 

	Input Parameters:

	file_names = ['one_file','second_file'] - array like, with file-types .hdf5
	LEN - integer: Length of the filenames array
	CAD - int: the cadence at which to plot of the details of files contained
	Normalied_B - bool: Show the magnetic fields evolution in terms of amplification or its magnitude

	Returns: None

	"""
	fig,a =  plt.subplots(2,2,figsize=(8,6)); dpi=400;
	
	for i in range(0,LEN,CAD):

		linestyle = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted'];

		file = h5py.File(file_names[i],"r")
		#print(file['scales/'].keys()); print(file['tasks/'].keys()) #useful commands	
		print(file_names[i])
		# Set the time interval we take
		index = 0; index_end = -1; 	
		time = file['scales/sim_time'][index:index_end];

		# All of these are <u,u> = (1/V)*(int_v u*u dV) where dV = rdr*ds*dz
		u2 = file['tasks/u_x total kinetic energy'][index:index_end,0,0,0];
		v2 = file['tasks/v kinetic energy'][index:index_end,0,0,0];
		w2 = file['tasks/w kinetic energy'][index:index_end,0,0,0];
		KE = u2 + v2 + w2;
		ME = v2 + w2;

		A2 = file['tasks/B_x   magnetic energy'][index:index_end,0,0,0];
		B2 = file['tasks/B_y magnetic energy'][index:index_end,0,0,0];
		C2 = file['tasks/B_z magnetic energy'][index:index_end,0,0,0];
		BKE = A2 + B2 + C2;
		BME = B2 + C2;
		
		x = time; #np.log10( time);# - time[0]*np.ones(len(time)) ); # Modify time so that it's zero'd

		if (Logscale == True) and (Normalised_B == False):
			a[0][0].plot(x,np.log10(KE),linestyle=linestyle[i],label=Legend_Names[i]);#,fontsize=25);
			a[1][0].plot(x,np.log10(ME),linestyle=linestyle[i],label=Legend_Names[i]);#,fontsize=25);
			
			a[0][1].plot(x,np.log10(BKE),linestyle=linestyle[i],label=Legend_Names[i]);#,fontsize=25);
			a[1][1].plot(x,np.log10(BME),linestyle=linestyle[i],label=Legend_Names[i]);#,fontsize=25);

		elif (Logscale == False) and (Normalised_B == True):
			a[0][1].plot(x,BKE/BKE[0],linestyle=linestyle[i],label=Legend_Names[i]);#,fontsize=25)
			a[1][1].plot(x,BME/BME[0],linestyle=linestyle[i],label=Legend_Names[i]);#,fontsize=25)

			a[0][0].plot(x,KE,linestyle=linestyle[i],label=Legend_Names[i]);#,fontsize=25);
			a[1][0].plot(x,ME,linestyle=linestyle[i],label=Legend_Names[i]);#,fontsize=25);

		elif (Normalised_B == True) and (Logscale == True):
			a[0][1].plot(x,np.log10(BKE/BKE[0]),linestyle=linestyle[i],label=Legend_Names[i]);#,fontsize=25)
			a[1][1].plot(x,np.log10(BME/BME[0]), linestyle=linestyle[i],label=Legend_Names[i]);#,fontsize=25)
			
			a[0][0].plot(x,np.log10(KE),linestyle=linestyle[i],label=Legend_Names[i]);#,fontsize=25);
			a[1][0].plot(x,np.log10(ME),linestyle=linestyle[i],label=Legend_Names[i]);#,fontsize=25);

		elif (Normalised_B == False) and (Logscale == False):	
			a[0][1].plot(x,BKE,linestyle=linestyle[i],label=Legend_Names[i]);#,fontsize=25)
			a[1][1].plot(x,BME,linestyle=linestyle[i],label=Legend_Names[i]);#,fontsize=25)

			a[0][0].plot(x,KE,linestyle=linestyle[i],label=Legend_Names[i]);#,fontsize=25);
			a[1][0].plot(x,ME,linestyle=linestyle[i],label=Legend_Names[i]);#,fontsize=25);
				
	LIM = [min(x),max(x)]
	#LIM = [0,(3./16.)*1500.]
	a[1][0].set_title(r'$U(x,t)^*$ Poloidal Energy')
	a[1][0].set_xlabel(r'$t_{shear} \sim S^{-1} t^*$')
	a[1][0].set_ylabel(r'$<U,U>^*$')
	a[1][0].set_xlim(LIM)
	#a[1][0].set_ylim([-4.65,-4.8])
	a[1][0].grid()

	a[1][1].set_title(r'$B(x,t)^*$ Poloidal Energy')
	a[1][1].set_xlabel(r'$t_{shear} \sim S^{-1} t^*$')
	a[1][1].set_ylabel(r'$<B,B>^*$')
	a[1][1].set_xlim(LIM)
	#a[1][1].set_ylim([-4.,-4.5])
	a[1][1].grid()

	a[0][0].set_title(r'$U(x,t)$ Flow Energy')
	a[0][0].set_ylabel(r'$<U,U>$');
	a[0][0].legend()
	a[0][0].set_xlim(LIM)
	#a[0][0].set_ylim([-0.478,-0.481])
	a[0][0].grid()

	a[0][1].set_title(r'$B(x,t)$ Magentic Energy');
	a[0][1].set_ylabel(r'$<B,B>$');
	a[0][1].set_xlim(LIM)
	#a[0][1].set_ylim([-0.36,-0.38])
	a[0][1].grid()

	#~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~

	if (Logscale == True):
		a[1][0].set_ylabel(r'$\log10(<U,U>^*)$')
		a[0][0].set_ylabel(r'$\log10(<U,U>)$');

	elif (Logscale == False):
		a[1][0].set_ylabel(r'$<U,U>^*$')
		a[0][0].set_ylabel(r'$<U,U>$');

	#~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~
	if (Normalised_B == False) and (Logscale == True):
		a[1][1].set_ylabel(r'$\log10(<B,B>^*)$')
		a[0][1].set_ylabel(r'$\log10(<B,B>)$');

	elif (Normalised_B == True) and (Logscale == False):
		a[1][1].set_ylabel(r'$<B,B>^*/<B_0,B_0>$')
		a[0][1].set_ylabel(r'$<B,B>/<B_0,B_0>$');	

	#~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~	
	
	plt.tight_layout(pad=1, w_pad=1.5)
	fig.savefig(outfile, dpi=dpi);
	plt.show()

	return None;


def Plot_wave_probes_General(file_names,Legend_Names,LEN,CAD,outfile):

	"""
	Plot the magnitude of the azimuthal velocity V_phi at the annulus centerline
	Useful for detecting periodic solutions and travelling wave solutions

	Input Parameters:

	file_names = 'one_file' with file-types .hdf5

	Returns: None
	
	"""
	fig, (ax1,ax2) =  plt.subplots(1,2,figsize=(8,6)); dpi = 400;
	for i in range(0,LEN,CAD):

		linestyle = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted'];

		file = h5py.File(file_names[i],"r")
		#print(file['scales/'].keys()); print(file['tasks/'].keys()) #useful commands	
		print(file_names[i])
		# Set the time interval we take

		index = 0; index_end = -1
		time = file['scales/sim_time'][index:index_end];
		time = time - time[0]*np.ones(len(time));
		x = time;

		Vp0 = file['tasks/u_x p0'][index:index_end,0,0,0];
		Vp1 = file['tasks/u_x p1'][index:index_end,0,0,0]
		Vp2 = file['tasks/u_x p2'][index:index_end,0,0,0];

		Vp3 = file['tasks/u_x p3'][index:index_end,0,0,0];
		Vp4 = file['tasks/u_x p4'][index:index_end,0,0,0]
		#Vp1 = file['tasks/V_phi p1'][index:index_end,0,0,0];

		Bp0 = file['tasks/b_x p0'][index:index_end,0,0,0];
		Bp1 = file['tasks/b_x p1'][index:index_end,0,0,0]
		Bp2 = file['tasks/b_x p2'][index:index_end,0,0,0];

		Bp3 = file['tasks/b_x p3'][index:index_end,0,0,0];
		Bp4 = file['tasks/b_x p4'][index:index_end,0,0,0]
		##Bp1 = file['tasks/B_phi p1'][index:index_end,0,0,0];


		ax1.plot(x,Vp0,'r',label='u_x=0'+ Legend_Names[i],linestyle=linestyle[i]);
		ax1.plot(x,Vp1,'b',label='u_x=2'+ Legend_Names[i],linestyle=linestyle[i])
		ax1.plot(x,Vp2,'k',label='u_x=4'+ Legend_Names[i],linestyle=linestyle[i])
		#ax1.plot(x,Vp3,'b-.',label=r'$u_x=6$')
		#ax1.plot(x,Vp4,'k-.',label=r'$u_x=8$')

		ax2.plot(x,Bp0,'r',label='B_phi=0'+ Legend_Names[i],linestyle=linestyle[i]);
		ax2.plot(x,Bp1,'b',label='B_phi=2'+ Legend_Names[i],linestyle=linestyle[i])
		ax2.plot(x,Bp2,'k',label='B_phi=4'+ Legend_Names[i],linestyle=linestyle[i])
		#ax2.plot(x,Bp3,'b-.',label='B_phi=6')
		#ax2.plot(x,Bp4,'k-.',label='B_phi=8')

	ax1.set_title(r'$U_x$ probes');
	ax1.set_xlabel(r'Time $t$',fontsize=15)
	ax1.legend()

	ax2.set_title(r'$B_x$ probes')
	ax2.set_xlabel(r'Time $t$',fontsize=15)
	ax2.legend()

	plt.tight_layout(pad=1, w_pad=1.5)
	fig.savefig(outfile, dpi=dpi);

	#plt.show()

	return None;


##########################################################################
# Plot Magentic <B,B>(t) scalar data
########################################################################## 
def Plot_KinematicB_scalar_data_General(file_names,Legend_Names,LEN,CAD,Normalised_B,Logscale,outfile):

	"""
	Plot the Magnetic fields integrated energies, as determined by the Magnetic Iduction equation 

	Input Parameters:

	file_names = ['one_file','second_file'] - array like, with file-types .hdf5
	LEN - integer: Length of the filenames array
	CAD - int: the cadence at which to plot of the details of files contained
	Normalied_B - bool: Show the magnetic fields evolution in terms of amplification or its magnitude

	Returns: None
	
	"""
	#outfile = 'Linear_Kinetic.pdf'; 
	fig,a =  plt.subplots(2,2,figsize=(8,6));dpi = 400;
	
	for i in range(0,LEN,CAD):

		file = h5py.File(file_names[i],"r")
		print(file['scales/'].keys()); print(file['tasks/'].keys()) #useful commands	

		# Set the time interval we take
		index = 0; index_end = -1;
		 	
		time = file['scales/sim_time'][:];
		x = time;# - time[0]*np.ones(len(time)); # Modify time so that it's zero'd
		x = x[index:index_end]
		
		# All of these are <u,u> = (1/V)*(int_v u*u dV) where dV = rdr*ds*dz
		A2 = file['tasks/B_x   magnetic energy'][index:index_end,0,0,0];
		B2 = file['tasks/B_y magnetic energy'][index:index_end,0,0,0];
		C2 = file['tasks/B_z magnetic energy'][index:index_end,0,0,0];
		BKE = A2 + B2 + C2;
		BME = B2 + C2;

		B_x = A2 - file['tasks/B_xM0 magnetic energy'][index:index_end,0,0,0];
		B_y = B2
		B_z = C2

		#Bxm0 = A2/Rm
		
		if Normalised_B == True:
			B_x = B_x/B_x[0];
			B_y = B_y/B_y[0];
			B_z = B_z/B_z[0] ;

			BKE = BKE/BKE[0];
			BKE = BME/BME[0];

			LABEL = r'$<B,B>/<B_0,B_0>$'
		elif Normalised_B == False:
			LABEL = r'$<B,B>$'	
		
		if Logscale == True:
			a[1][0].plot(x,np.log10(B_x),'-',label=Legend_Names[i]);#,fontsize=25);
			a[1][1].plot(x,np.log10(B_y),'-.',label=Legend_Names[i]);#,fontsize=25);
			a[0][0].plot(x,np.log10(B_z),'-.',label=Legend_Names[i]);#,fontsize=25);

			a[0][1].plot(x,np.log10(BKE),'-',label=Legend_Names[i]);#,fontsize=25)
			#a[0][1].plot(x,BME,':',label=r'$<B,B>^*_{i=%i}$'%i);#,fontsize=25)
			LABEL = r'$ln_10$' + LABEL;
		elif Logscale == False:

			a[1][0].plot(x,B_x,'-',label=Legend_Names[i]);#,fontsize=25);
			a[1][1].plot(x,B_y,'-.',label=Legend_Names[i]);#,fontsize=25);
			a[0][0].plot(x,B_z,'-.',label=Legend_Names[i]);#,fontsize=25);

			a[0][1].plot(x,BKE,'-',label=Legend_Names[i]);#,fontsize=25)
			#a[0][1].plot(x,BME,':',label=r'$<B,B>^*_{i=%i}$'%i);#,fontsize=25)

	# Set their labels
	#LIM = [np.min(x),max(x)];
	#LIM = [0,3000.]	
	a[1][0].set_title(r'$B_x(x,t)$')
	a[1][0].set_xlabel(r'$t_{shear} \sim S^{-1} t^*$')
	a[1][0].set_ylabel(LABEL)
	a[1][0].set_xlim(LIM)
	a[1][0].grid()

	a[1][1].set_title(r'$B_y(x,t)$')
	a[1][1].set_xlabel(r'$t_{shear} \sim S^{-1} t^*$')
	a[1][1].set_ylabel(LABEL)
	#a[1][1].set_ylim([7.0,7.5])
	a[1][1].set_xlim(LIM)
	a[1][1].grid()

	a[0][0].set_title(r'$B_x(x,t)$')
	a[0][0].set_ylabel(LABEL);
	#a[0][0].legend()
	#a[0][0].set_ylim([-2.0,0.5])
	a[0][0].set_xlim(LIM)
	a[0][0].grid()

	a[0][1].set_title(r'$B(x,t)$ Magentic Energy');
	a[0][1].set_ylabel(LABEL);
	a[0][1].legend(loc=2)
	#a[0][1].set_ylim([7.0,7.5])
	a[0][1].set_xlim(LIM)
	a[0][1].grid()

	plt.tight_layout(pad=1, w_pad=1.5)
	fig.savefig(outfile, dpi=dpi);

	#plt.show()

	return None;

def Check_BCS(file_names,times,LEN,CAD):
	

	for k in range(0,LEN,CAD):

		file = h5py.File(file_names[k],"r")
		#print(file['scales/s'].keys()); print(file['tasks/'].keys()) #useful commands

		x = file['scales/x/1.5']; y = file['scales/y/1.5']; z = file['scales/z/1.5']

		#(time,x,y,z)
		#if Just_B == False:
		u   = file['tasks/u']; v   = file['tasks/v']; w   = file['tasks/w'];
		
		#B_x = file['tasks/A']; B_y = file['tasks/B']; 
		#B_z = file['tasks/C'];
		#BZ_x = file['tasks/Az']; BZ_y = file['tasks/Bz']; #B_z = file['tasks/C'];

		SLICE = 10; # This needs some modification

		for i in range(len(times)):
			
			index = times[i]; # The instant at which we plot out the vector

			print("\n Iteration k=%i, time t=%e"%(k,times[i]))
			##########################################################################
			# Code segments for checking boundary conditions
			##########################################################################

			#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~

			print("\n")
			print("Slip u-streamwise")
			G1 = file['tasks/u'][index,0,10,0];
			print(G1)
			G1 = file['tasks/u'][index,0,10,-1];
			print(G1)
			#r = file['scales/r/1.0']
			print("dz",abs(z[1] - z[0]));

			print("\n")
			print("Slip v-spanwise")
			G1 = file['tasks/v'][index,0,10,0];
			print(G1)
			G1 = file['tasks/v'][index,0,10,-1];
			print(G1)
			print("dy",abs(y[1] - y[0]))

			print("\n")
			print("Slip w-shearwise")
			G1 = file['tasks/w'][index,0,10,0];
			print(G1)
			G1 = file['tasks/w'][index,0,10,-1];
			print(G1)
			print("dx",abs(x[1] - x[0]))

			print("\n")
			#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~

			print("\n")
			print("Slip dz(Bx)-streamwise")
			G1 = file['tasks/Az'][index,0,10,0];
			print(G1)
			G1 = file['tasks/Az'][index,0,10,-1];
			print(G1)

			print("\n")
			print("Slip dz(By)-spanwise")
			G1 = file['tasks/Bz'][index,0,10,0];
			print(G1)
			G1 = file['tasks/Bz'][index,0,10,-1];
			print(G1)

			print("\n")
			#print("Slip Div.B-shearwise")
			print("Slip Bz-shearwise")
			G1 = file['tasks/C'][index,0,10,0];
			print(G1)
			G1 = file['tasks/C'][index,0,10,-1];
			print(G1)
			print("\n")
			#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~

	return None;

def Plot_MAX_DivUB(filename,k=0):

	"""
	Plot the max( abs(div(B)) ) & max( abs(div(U)) ) during a Direct Adjoint Looping (DAL) routine in-terms of....
	divB = DAL_file['max|div(B)|'][()];
	divU = DAL_file['max|div(U)|'][()];

	Input Parameters:

	filename - hdf5 file with dict 'key':value structure, all 'keys' are defined as above

	Returns: None

	"""
	import matplotlib.pyplot as plt
	outfile = 'MaxDivU_MaxDivB_vs_iterations_k=%i.pdf'%k; 

	dpi = 400
	fig, ax1 = plt.subplots(figsize=(8,6)); # 1,2,

	# Grab Data
	DAL_file = h5py.File(filename,"r")

	divB = DAL_file['max|div(B)|'][()];
	divU = DAL_file['max|div(U)|'][()];

	FLUX_B = DAL_file['FLUX_B'][()];
	#'''
	#MAX_Bx = DAL_file['max|abs(Bx)|'][()];
	#MAX_By = DAL_file['max|abs(By)|'][()];

	#ZER0_Bx = DAL_file['ZER0_Bx'][()];
	#ZER0_By = DAL_file['ZER0_By'][()];
	#'''
	DAL_file.close();

	# Plot figures
	color = 'tab:red'
	x = np.arange(0,len(divB),1)
	ax1.plot(x,np.log10(divB),color=color, linestyle=':',linewidth=1.5, markersize=3);
	ax1.tick_params(axis='y', labelcolor=color)
	ax1.set_ylabel(r'log10( Max|div(B)| )',color=color,fontsize=18)
	ax1.set_xlabel(r'Iteration k',fontsize=18)

	ax1.set_xlim([0,np.max(x)])

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	color = 'tab:blue'
	x = np.arange(0,len(divU),1)
	ax2.plot(x,np.log10(divU),color=color, linestyle='-',linewidth=1.5, markersize=3);
	
	ax2.tick_params(axis='y', labelcolor=color)
	ax2.set_ylabel(r'log10( Max|div(U)| )',color=color,fontsize=18)
	#ax2.set_ylabel(r'Gradient residual error $log10(r)$',color=color,fontsize=18)
	#ax2.set_xlabel(r'Iteration $k$',fontsize=15)

	plt.grid()
	plt.tight_layout(pad=1, w_pad=1.5)
	fig.savefig(outfile, dpi=dpi);
	plt.show();

	#'''
	outfile1 = 'FluxB_vs_iterations_k=%i.pdf'%k; 
	fig1, ax = plt.subplots(figsize=(8,6));
	x = np.arange(0,len(FLUX_B),1)

	#ax.plot(x,np.log10(ZER0_Bx),'k:',linewidth=1.5, markersize=3);
	ax.plot(x,np.log10(FLUX_B),'k-',linewidth=1.5, markersize=3);
	
	#ax.plot(x,np.log10(MAX_Bx),'r:',linewidth=1.5, markersize=3);
	#ax.plot(x,np.log10(MAX_By),'r-',linewidth=1.5, markersize=3);
	
	ax.set_ylabel(r'log10( |<B>| )',color='k',fontsize=18)
	ax.set_xlabel(r'Iteration k',fontsize=18)

	plt.grid()
	plt.xlim([min(x),max(x)])
	plt.tight_layout(pad=1, w_pad=1.5)
	fig1.savefig(outfile1, dpi=dpi);
	plt.show();
	#'''
	return None;

#####################################

if __name__ == "__main__":

	
	##########################################################################
	# Change into Results-Directory
	##########################################################################
	'''
	Home_Dir = "/Users/pmannix/Desktop/Nice_CASTOR/DAL_TC_MHD/";
	os.chdir(Home_Dir)
	import glob
	Files  = glob.glob('./MinSeed_FullRuns_Pm9/Test_DAL_Re100_Pm9*');
	Files += glob.glob('./MinSeed_FullRuns_Pm9/Test_Pm9*');
	
	#print(Files);
	#sys.exit()
	
	for f in Files:
	#os.chdir("/workdir/pmannix/DAL_TC_MHD/Results_File_M1e-03_J1_3Re/")
	os.chdir(Home_Dir + f);
	'''
	#Home_Dir = "/Users/pmannix/Desktop/Nice_CASTOR/DAL_TC_MHD/";
	#os.chdir(Home_Dir + "Results_Lin_Test2");#_Omega"); #+ "Test_DAL_Re100_Pm9_M0.2_dt2.5e-04_MCNAB2_N128")
	#os.chdir(Home_Dir + "Results_Test_U0_t5.5");#_Omega"); #+ "Test_DAL_Re100_Pm9_M0.2_dt2.5e-04_MCNAB2_N128")
	
	'''
	# Incase files unmerged
	from dedalus.tools  import post
	from mpi4py import MPI
	comm = MPI.COMM_WORLD
	#Checkpoints_filenames = ['CheckPoints/CheckPoints_s1.h5','CheckPoints/CheckPoints_s2.h5','CheckPoints/CheckPoints_s3.h5','CheckPoints/CheckPoints_s4.h5']
	#post.merge_sets("CheckPoints_Merged", Checkpoints_filenames,cleanup=False, comm=MPI.COMM_WORLD)
	post.merge_process_files("CheckPoints", cleanup=True, comm=MPI.COMM_WORLD);
	post.merge_process_files("scalar_data", cleanup=True, comm=MPI.COMM_WORLD);
	comm.Barrier();
	#sys.exit()
	'''

	try:
		atmosphere_file = h5py.File('./Params.h5', 'r+')
		print(atmosphere_file.keys())

		Re = atmosphere_file['Re'][()];
		Pm = atmosphere_file['Pm'][()];
		mu = atmosphere_file['mu'][()];

		alpha = atmosphere_file['alpha'][()];
		beta = atmosphere_file['beta'][()];

		# Numerical Params
		Nx = atmosphere_file['Nx'][()];
		Ny = atmosphere_file['Ny'][()];
		Nz = atmosphere_file['Nz'][()];
		dt = atmosphere_file['dt'][()];

		print("Nx = ",Nx);
		print("Ny = ",Ny);
		print("Nz = ",Nz);
		print("dt = ",dt,"\n");

		M_0 = atmosphere_file['M_0'][()];
		E_0 = atmosphere_file['M_0'][()];
		T   = atmosphere_file['T'][()]

		atmosphere_file.close()
	except:

		pass;		

	##########################################################################
	# Scalar_data filenames
	##########################################################################	
	#'''
	#Scalar_data_filenames = ['Adjoint_scalar_data/Adjoint_scalar_data_s1.h5']
	Scalar_data_filenames = ['scalar_data/scalar_data_s1.h5']
	#Scalar_data_filenames = ['scalar_data_iter_9.h5']
	
	LEN = len(Scalar_data_filenames);
	Plot_Cadence = 1;	
	Normalised_B = False;
	Logscale = True;

	# Useful for examing the energetics of the magnetic fields different components
	Plot_KinematicB_scalar_data(Scalar_data_filenames,LEN,Plot_Cadence,Normalised_B,Logscale)
	
	##########################################################################
	# Full Checkpoints filenames
	##########################################################################

	#Checkpoints_filenames = ['Adjoint_CheckPoints/Adjoint_CheckPoints_s1.h5']
	Checkpoints_filenames = ['CheckPoints/CheckPoints_s1.h5']

	LEN = len(Checkpoints_filenames);
	Plot_Cadence = 1;
	times = [0,-1]; #First and Last Checkpoints
	Just_B = False; # If True only plots B-field

	Plot_UB_pair(Checkpoints_filenames,times,LEN,Plot_Cadence,Just_B)
	
	#Plot_KUKB_pair(Checkpoints_filenames,times,LEN,Plot_Cadence,Just_B)

	print("\n ----> Vector Field Plots Complete <------- \n")

