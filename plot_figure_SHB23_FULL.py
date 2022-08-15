import numpy as np
import matplotlib.pyplot as plt
import h5py, sys, os, glob

##########################################################################
# Scalar Data Plot functions
##########################################################################
def Plot_KinematicB_scalar_data(file_names,LEN,CAD,Logscale_B):

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
		outfile = 'Linear_Kinetic_Logscale.pdf'; 
	else:	
		outfile = 'Linear_Kinetic.pdf'; 
	
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
		
		BE = file['tasks/Kinetic energy'][index:index_end,0];
		
		LABEL = r'$<u,u>$'	
		a[0].plot(x,np.log10(BE),'-' ,label=r'$<u^2>_{i=%i}$'%i);#,fontsize=25);
		a[1].plot(x,BE          ,'-.',label=r'$<u^2>_{i=%i}$'%i);#,fontsize=25);

	# Set their labels
	a[0].set_title(r'$\log10(<u^2>(t) )$ Kinetic Energy')
	a[0].set_ylabel(LABEL);
	a[0].legend()
	a[0].set_xlim([np.min(x),np.max(x)])
	a[0].grid()

	a[1].set_title(r'$<u^2>(t)$ Kinetic Energy');
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
def Plot_UB_pair(file_names,times,LEN,CAD):

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
	fig = plt.figure(figsize=(14,4))
	plt.xticks(fontsize=26 )
	plt.yticks(fontsize=26 )

	ax = plt.gca()
	ax.yaxis.set_major_locator(plt.MaxNLocator(4));

	for time in times:

		outfile = "".join(['U_PLOTS_Time_t=%i_SH23.pdf'%time ]); dpi=1200


		for k in range(0,LEN,CAD):

			file = h5py.File(file_names[k],"r")
			#print(file['scales/'].keys()); print(file['tasks/'].keys()) #useful commands

			#(time,x)
			x = file['scales/z/1.5'][()];
			u = file['tasks/u']; 

			#kx = file['scales/kx'][()]#print(kx)
			if (k == LEN - 1):
				u_hat = file['tasks/u_hat'][time,:]; 
				
				u_hat = np.real(u_hat[:]*np.conj(u_hat[:]))
				u_hat = 2.*np.pi*( 0.5*u_hat[0] + np.sum(u_hat[1:-1]) );
				
				print("\n\n E_t(u)=%e, time=%i \n\n"%(u_hat,time) )

			if time == 0:	
				plt.plot(x,u[time,:],'-',label=r'$M_2$',linewidth=2.)
			elif time == -1:
				plt.plot(x,u[time,:],':',label=r'$S_2$',linewidth=2.)	

	#------------------------------------ #------------------------------------
	# Save figure
	plt.xlabel(r'$z$',fontsize=26)
	plt.ylabel(r'$u(z)$',fontsize=26)
	plt.xlim([np.min(x),np.max(x)])
	plt.legend(fontsize=26)
	#plt.grid()
	plt.tight_layout(pad=1, w_pad=1.5)
	fig.savefig(outfile, dpi=dpi)
	#plt.show()

	return None;

if __name__ == "__main__":

	
	##########################################################################
	# Scalar_data filenames
	##########################################################################	
	Scalar_data_filenames = glob.glob('./scalar_data_iter_*.h5');
	#Scalar_data_filenames = ['scalar_data_iter_51.h5']

	LEN = len(Scalar_data_filenames);
	Plot_Cadence = 4;	
	Logscale = True;

	Plot_KinematicB_scalar_data(Scalar_data_filenames,LEN,Plot_Cadence,Logscale)
	
	##########################################################################
	# Full Checkpoints filenames
	##########################################################################

	#Checkpoints_filenames = ['CheckPoints_iter_51.h5']
	Checkpoints_filenames = glob.glob('./CheckPoints_iter_*.h5');
	
	LEN = len(Checkpoints_filenames);
	Plot_Cadence = 4;
	times = [0]; #First and Last Checkpoints
	Just_B = False; # If True only plots B-field

	Plot_UB_pair(Checkpoints_filenames,times,LEN,Plot_Cadence)
	
	print("\n ----> Vector Field Plots Complete <------- \n")

