import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, cm
import h5py, sys, os

##########################################################################
# Scalar Data Plot functions
##########################################################################

def Plot_scalar_data(file_names,LEN,CAD,Normalised_B,Logscale_B):

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
		outfile = 'Linear_Kinetic_UB_Logscale.pdf'; 
	else:	
		outfile = 'Linear_Kinetic_UB.pdf'; 
	
	dpi = 1200;
	fig,a =  plt.subplots(1,2,figsize=(8,6));
	
	for i in range(0,LEN,CAD):

		file = h5py.File(file_names[i],"r")
		#print(file['scales/'].keys()); print(file['tasks/'].keys()) #useful commands	

		# Set the time interval we take
		index = 0; index_end = -1;
		time = file['scales/sim_time'][:];
		x = time - time[0]*np.ones(len(time)); # Modify time so that it's zero'd
		x = x[index:index_end]
		
		KE = file['tasks/Kinetic  energy'][index:index_end,0,0];
		BE = file['tasks/Buoyancy energy'][index:index_end,0,0];

		a[0].semilogy(x,KE,label=r'$<u^2 + w^2>_{i=%i}$'%i);#,fontsize=25);
		a[1].semilogy(x,BE,label=r'$<b^2>_{i=%i}$'%i);#,fontsize=25);

	# Set their labels
	a[0].set_title(r'$<u^2 + w^2>$ Kinetic  Energy')
	#a[0].set_ylabel(LABEL);
	a[0].legend()
	a[0].set_xlim([np.min(x),np.max(x)])
	a[0].grid()

	a[1].set_title(r'$<b,b>$ Buoyancy Energy');
	#a[1].set_ylabel(LABEL);
	a[1].legend()
	a[1].set_xlim([np.min(x),np.max(x)])
	a[1].grid()

	plt.tight_layout(pad=1, w_pad=1.5)
	fig.savefig(outfile, dpi=dpi);

	#plt.show()

	return None;

##########################################################################
# Plot a vec=U,B pair
##########################################################################
def Plot_U_and_B(file_names,times,LEN,CAD,Just_B):

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

		#(time,x,z)
		x = file['scales/x/1.5']; z = file['scales/z/1.5'];
		u = file['tasks/u']; w = file['tasks/w']; b = file['tasks/b'];
		Ω = file['tasks/vorticity']; 
		
		for i in range(len(times)):
			
			index = times[i]; # The instant at which we plot out the vector

			outfile = "".join(['PLOTS_Iter_i%i_Time_t%i.pdf'%(k,index) ]);	
			
			##########################################################################
			# ~~~~~~~~~~~~~~~~~~~ plotting Magnetic B ~~~~~~~~~~~~~~~~~~~~~
			##########################################################################

			fig, ax =  plt.subplots(2,1,figsize=(8,6));
			plt.suptitle("B Field Iter=i%i Time=t%i"%(k,index) ); dpi = 1200;

			#------------------------------------ #------------------------------------
			#ax1 = plt.subplot(221)
			X,Z = np.meshgrid(x,z);

			#KE = u[index,:,:]**2 + w[index,:,:]**2;
			#cs = ax[0].contourf(X,Z,KE.T,cmap='RdBu',levels=30)
			cs = ax[0].contourf(X,Z,Ω[index,:,:].T,cmap='RdBu',levels=30)
			
			skip=(slice(None,None,2),slice(None,None,2))
			#ax[0].quiver(X[skip],Z[skip],u[index,:,:][skip].T,w[index,:,:][skip].T,width=0.005);
			fig.colorbar(cs,ax=ax[0]);

			#ax[0].set_title(r'$u^2 + w^2$, vecs - $(u,w)$');
			ax[0].set_title(r'$Ω$, vecs - $(u,w)$');
			ax[0].set_xlabel(r'Shearwise - $x$')#, fontsize=18)
			ax[0].set_ylabel(r'Spanwise  - $z$')#, fontsize=18)

			#------------------------------------ #------------------------------------
			#ax2 = plt.subplot(222)
			cs = ax[1].contourf(X,Z,b[index,:,:].T,cmap='RdBu',levels=30)
			
			#skip=(slice(None,None,4),slice(None,None,4))
			#ax2.quiver(Z[skip],X[skip], B_z[index,:,SLICE_y,:][skip].T,B_x[index,:,SLICE_y,:][skip].T,width=0.005);
			fig.colorbar(cs,ax=ax[1]);

			ax[1].set_title(r'$b$');
			ax[1].set_xlabel(r'Shearwise  - $x$')
			ax[1].set_ylabel(r'Streamwise - $z$')

			#------------------------------------ #------------------------------------
			# Save figure
			plt.tight_layout(pad=1, w_pad=1.5)
			fig.savefig(outfile, dpi=dpi)
			#plt.show()

	return None;

##########################################################################
# Plot Kinetc <u,u>(k,m) & Magentic <B,B>(k,m) as 3D surfaces
########################################################################## 
def Plot_Spectra(file_names,times,LEN,CAD,Just_B):

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
		print(file['tasks/'].keys()); print(file['scales/'].keys()); 
		
		# Get wavenumbers and create a Mesh-grid for plotting
		kx = file['scales/kx'][()];
		KE = file['tasks/kx Kinetic  energy'][:,:,0]; 
		KB = file['tasks/kx Buoyancy energy'][:,:,0];

		try:
			Tz = file['scales/(T,T)z'][()];
		except: 
			Tz = file['scales/Tz'][()];

		TE = file['tasks/Tz Kinetic  energy'][:,0,:]; 
		TB = file['tasks/Tz Buoyancy energy'][:,0,:];

		for i in range(len(times)):

			index = times[i]; # The instant at which we plot out the vector

			outfile = "".join(['KE_PLOTS_Iter_i%i_Time_t%i.pdf'%(k,index) ]);	
		
			##########################################################################
			# ~~~~~~~~~~~~~~~~~~~ plotting Magnetic B ~~~~~~~~~~~~~~~~~~~~~
			##########################################################################

			dpi = 1200;
			fig, a =  plt.subplots(1,2,figsize=(8,6));

			LABEL = r'$<X,X>$'	
			a[0].semilogy(kx,abs(KE[index,:]),'b.',label=r'$<u^2 + w^2>_{i=%i}$'%i);#,fontsize=25);
			a[1].semilogy(kx,abs(KB[index,:]),'k.',label=r'$<b^2>_{i=%i}$'%i);#,fontsize=25);

			# Set their labels
			a[0].set_title(r'$<u^2 + w^2>(kx)$ Kinetic  Energy')
			a[0].set_ylabel(LABEL);
			a[0].legend()
			a[0].set_xlim([np.min(kx),np.max(kx)])
			a[0].grid()

			a[1].set_title(r'$<b,b>(kx)$ Buoyancy Energy');
			a[1].set_ylabel(LABEL);
			a[1].legend()
			a[1].set_xlim([np.min(kx),np.max(kx)])
			a[1].grid()

			plt.tight_layout(pad=1, w_pad=1.5)
			fig.savefig(outfile, dpi=dpi);
			#plt.show()


		for i in range(len(times)):

			index = times[i]; # The instant at which we plot out the vector

			outfile = "".join(['Cheb_PLOTS_Tz_Iter_i%i_Time_t%i.pdf'%(k,index) ]);	
		
			##########################################################################
			# ~~~~~~~~~~~~~~~~~~~ plotting Magnetic B ~~~~~~~~~~~~~~~~~~~~~
			##########################################################################

			dpi = 1200;
			fig, a =  plt.subplots(1,2,figsize=(8,6));

			LABEL = r'$<X,X>$'	
			a[0].semilogy(Tz,abs(TE[index,:]),'b.',label=r'$<u^2 + w^2>_{i=%i}$'%i);#,fontsize=25);
			a[1].semilogy(Tz,abs(TB[index,:]),'k.',label=r'$<b^2>_{i=%i}$'%i);#,fontsize=25);

			# Set their labels
			a[0].set_title(r'$<u^2 + w^2>(Tz)$ Kinetic  Energy')
			a[0].set_ylabel(LABEL);
			a[0].legend()
			a[0].set_xlim([np.min(Tz),np.max(Tz)])
			a[0].grid()

			a[1].set_title(r'$<b,b>(Tz)$ Buoyancy Energy');
			a[1].set_ylabel(LABEL);
			a[1].legend()
			a[1].set_xlim([np.min(Tz),np.max(Tz)])
			a[1].grid()

			plt.tight_layout(pad=1, w_pad=1.5)
			fig.savefig(outfile, dpi=dpi);
			#plt.show()	
			

	return None;

#####################################

if __name__ == "__main__":

	
	import glob

	##########################################################################
	# Scalar_data filenames
	##########################################################################	
	
	Scalar_data_filenames = glob.glob('./scalar_data_iter_*.h5');
	#Scalar_data_filenames = glob.glob('./RES_*/scalar_data/scalar_data_s1.h5');
	#Scalar_data_filenames = ['scalar_data/scalar_data_s1.h5']
	
	print(Scalar_data_filenames)
	LEN = len(Scalar_data_filenames);
	Plot_Cadence = 1;	
	Normalised_B = False;
	Logscale = True;

	# Useful for examing the energetics of the magnetic fields different components
	Plot_scalar_data(Scalar_data_filenames,LEN,Plot_Cadence,Normalised_B,Logscale)
	
	##########################################################################
	# Full Checkpoints filenames
	##########################################################################

	Checkpoints_filenames = glob.glob('./CheckPoints_iter_*.h5');
	#Checkpoints_filenames = glob.glob('./Test_*/CheckPoints/CheckPoints_s1.h5');
	#Checkpoints_filenames = ['CheckPoints/CheckPoints_s1.h5']

	LEN = len(Checkpoints_filenames);
	Plot_Cadence = 1;
	times = [0,-1]; # First and Last Checkpoints
	Just_B = False; # If True only plots B-field

	Plot_U_and_B(Checkpoints_filenames,times,LEN,Plot_Cadence,Just_B)
	Plot_Spectra(Checkpoints_filenames,times,LEN,Plot_Cadence,Just_B)

	print("\n ----> Vector Field Plots Complete <------- \n")

