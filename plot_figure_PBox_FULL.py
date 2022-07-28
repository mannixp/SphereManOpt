import numpy as np
import matplotlib.pyplot as plt
import h5py, sys, os, glob

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

	#plt.show()

	return None;

##########################################################################
# Plot a vec=U,B pair
##########################################################################
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

if __name__ == "__main__":
	
	#filenames = ['Test_M1T1_N24_dt1e-03_Continuous/DAL_PROGRESS.h5','Test_M1T1_N24_dt1e-03_Discrete/DAL_PROGRESS.h5'];
	#filenames = ['Test_M1T1_N24_dt1e-03_Continuous_2Wolfe/DAL_PROGRESS.h5','Test_M1T1_N24_dt1e-03_Discrete_2Wolfe/DAL_PROGRESS.h5']
	#filenames = ['Test_M1T1_N24_dt1e-03_Discrete_2Wolfe_WOLFEI/DAL_PROGRESS.h5','Test_M1T1_N24_dt1e-03_Discrete_2Wolfe_WOLFEI_CG/DAL_PROGRESS.h5'];
	
	##########################################################################
	# Scalar_data
	##########################################################################	

	Scalar_data_filenames = glob.glob('./scalar_data_iter_*.h5');#['scalar_data/scalar_data_s1.h5']

	LEN = len(Scalar_data_filenames);
	Plot_Cadence = 1;	
	Normalised_B = False;
	Logscale = True;

	Plot_KinematicB_scalar_data(Scalar_data_filenames,LEN,Plot_Cadence,Normalised_B,Logscale)
	
	##########################################################################
	# Checkpoints
	##########################################################################
	
	Checkpoints_filenames = glob.glob('./CheckPoints_iter_*.h5'); #['CheckPoints/CheckPoints_s1.h5']

	LEN = len(Checkpoints_filenames);
	Plot_Cadence = 1;
	times = [0,-1]; #First and Last Checkpoints
	Just_B = False; # If True only plots B-field

	Plot_UB_pair(Checkpoints_filenames,times,LEN,Plot_Cadence,Just_B)

	print("\n ----> Vector Field Plots Complete <------- \n")

