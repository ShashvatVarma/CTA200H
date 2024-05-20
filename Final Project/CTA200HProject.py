#basic imports
from turtle import left
from matplotlib.bezier import check_if_parallel
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import time
import os
import shutil
import h5py

#update matplotlib latex preamble
params = {'text.usetex': True, 'text.latex.preamble': [r'\usepackage{physics}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)

#video generation import
from subprocess import call
from PIL import Image
import cv2

#parallelization import
from multiprocessing import Pool

#These are private libraries used in my project that are just meant to properly load the simulation files
#I will provide the final data in a .txt format so that it can be used to run the code instead.
#========================================================================================================
import FLASHtools.aux_funcs.derived_var_funcs as dv
import FLASHtools.analysis_functions as funcs
#========================================================================================================
#IF YOU WANT TO RUN THE CODE, EDIT THE NECESSARY VARIABLES HERE
#directory is the location of the data files, and where you wish the outputs to be placed
#========================================================================================================
directory = 'C:\\Users\\shash\\Desktop\\Assignments\\AST425\\Short_144\\'    #make sure to use double slashes as shown
i_am_the_grader = False      #set to true here, will not run the data loader functions, instead uses downloaded data
question_being_graded = 1    #set to the question number here (only options are 1,2,3 (if you enter anything else it will delete system32))
n_cores = 6                  #set the number of processor cores you wish to use here
temp_directory = directory + 'temp_images\\'

if not os.path.exists(temp_directory):
    os.mkdir(temp_directory) #makes temporary directory for temporary images to be created
#if this doesn't work, make sure to create the directory manually. Not sure if this works on MacOS system.
#========================================================================================================
#FUNCTIONS FOR VIDEO GENERATION
def create_video(start, stop, skip, temp_dir, output_directory, output_name, fps):
    """this is for creating a video given a set of .png images, only need to supply it the folder name
    now use the temporary images to make a video used in the parallelized generation of animations. This function uses ffmpeg
    to generate the video, so make sure you have it installed on your computer.

    Parameters
    ----------
    start : int
        file identifier start value
    stop : int
        file identifier stop value
    skip : list
        list of integers to skip in the file identifier. Use in case of missing images due to runtime errors
    temp_dir : string
        full path location to temporary directory containing images. Make sure to add double slashes, and
         end path name with double slash e.g. 'C:\\Users\\user\\folder\\'. The function will
         create this directory automatically, and similarly will delete it after the video is created. DO NOT ENTER
         A DIRECTORY THAT CONTAINS OS FILES AS THAT COULD BE DELETED.
    output_name : string
        full path of output file, including file name and extension (only .avi supported for now)
    output_directory : string
        full path of output directory where the .mp4 video will be saved. Make sure to end the path with a double slash.
    fps : int
        frames per second of the video
    """
    def create_mp4(video_name):
        """Generates .mp4 video from the temporary .avi video. Must have ffmpeg on your computer.

        Parameters
        ----------
        video_name : string
            name of the video file to be generated
        """
        call(['ffmpeg', '-i', f'{temp_dir}temp_video.avi', f'{output_directory}{video_name}.mp4'])        

    #generate the image names to make the video from
    name_array = []
    for i in np.arange(start,stop+1):
        if i in skip:
            pass
        else:
            #create the image names
            name_array.append(str(temp_dir) + 'image'+str(i)+'.png')

    #find width/height for an image. If size in video not defined to this, then the video output will be corrupt
    img = Image.open(name_array[0])
    width, height = img.size
    img.close()
    #define the video writer
    video = cv2.VideoWriter(f'{temp_dir}temp_video.avi', 0, fps, (width,height))   #ensures videowriter width/height matches image

    #use the .png files in the video
    for image in name_array:
        video.write(cv2.imread(image))

    #write the .avi file
    cv2.destroyAllWindows()
    video.release()

    #convert the .avi file to .mp4
    create_mp4(output_name)

    #once video is made, delete temp_directory
    shutil.rmtree(temp_directory)    
#=======================================================================================
#QUESTION 1 FUNCTIONS

#Computing the eigenvalues of tensor fields efficently
#We will parallelize using the multiprocessing library to run many computations with multiple cores

def symmetric_eigvals(matrix, find_vectors = False):
    """Finds the eigenvalues and eigenvectors of real Hermitian 3x3 matrices defined at each point
     in some grid. Methods from Deledalle et al. 2017 https://hal.science/hal-01501221/document . 

    Parameters
    ----------
    matrix : np.ndarry
        Must be an array of shape (3,3,Nx,Ny,Nz) where Nx, Ny, Nz are the dimensions of the grid.
    find_vectors : bool, optional
        If True, the function will also compute the eigenvectors of the matrix. Default is False to save time on
        eigenvalue-only calculations if needed.

    Returns
    -------
    eig_array : np.ndarray
        Array containing the three eigenvalues of the matrix, which will always be real for symmetric matrices.
        The eigenvalues are sorted in ascending order. Has shape (3,Nx,Ny,Nz).
    vec_array : np.ndarray, optional
        Only returns if find_vectors = True. Array containing the three eigenvectors
        of the matrix defined at each grid point. The eigenvalue and eigenvector arrays are such that the same indexed
        element in each matrix correspond to each other (i.e. they are the eigenvalue/vector pair). Has shape (3,3,Nx,Ny,Nz)
        where indexing is [a,b,i,j,k] where a is the eigenvector identifier (i.e. a=1 is the second eigenvector), b is
        the b-th component of the eigenvector, and i,j,k are the grid indices. (vec_array[0,:,Nx,Ny,Nz] will give the
        eigenvector with the smallest eigenvalue at the grid point (Nx,Ny,Nz) printed as a row vector).
    """
    A = matrix   #a symmetric matrix
    #define some quantities for computations given by Deledalle et al. 2017
    a = A[0,0,:,:,:]
    b = A[1,1,:,:,:]
    c = A[2,2,:,:,:]
    d = A[0,1,:,:,:]
    e = A[1,2,:,:,:]
    f = A[0,2,:,:,:]       

    #begin the computations (Since matrices are real hermitian, we do not need to worry about complex numbers)
    x1 = a**2 + b**2 + c**2 - a*b - a*c - b*c+3*(d**2 + f**2 + e**2)
    x2 = (-1)*(2*a-b-c)*(2*b-a-c)*(2*c-a-b) + 9*((2*c-a-b)*d**2 + (2*b-a-c)*f**2 + (2*a-b-c)*e**2) - 54*d*e*f

    #define what phi is conditional to previous variables
    condition_list = [x2>0, x2==0, x2<0]
    choice_list = [np.arctan((np.sqrt(4*x1**3-x2**2))/(x2)), np.pi/2, np.arctan((np.sqrt(4*x1**3-x2**2))/(x2))+np.pi]
    phi = np.select(condition_list, choice_list)         #vectorizes the computation rather than using if statements

    #calculate the eigenvalues
    lambda1 = (a+b+c-2*np.sqrt(x1)*np.cos(phi/3))/3
    lambda2 = (a+b+c+2*np.sqrt(x1)*np.cos((phi-np.pi)/3))/3
    lambda3 = (a+b+c+2*np.sqrt(x1)*np.cos((phi+np.pi)/3))/3
    eig_array = np.array([lambda1, lambda2, lambda3])   #defines the eigenvalue list

    #sort the eigenvalues
    idx = np.argsort(eig_array, axis=0)  #save the sort indices to sort the eigenvectors as well if needed
    eig_array = np.take_along_axis(eig_array, idx, axis=0)  #perform the sort

    if find_vectors:
        #compute the eigenvectors
        #first define some values
        m1 = (d*(c-lambda1) - e*f) / (f*(b-lambda1) - d*e)
        m2 = (d*(c-lambda2) - e*f) / (f*(b-lambda2) - d*e)
        m3 = (d*(c-lambda3) - e*f) / (f*(b-lambda3) - d*e)

        #compute the vectors, putting them in row vector form in a list
        vec1 = [(lambda1 - c - e * m1)/f, m1, np.ones(np.shape(m1))]
        vec2 = [(lambda2 - c - e * m2)/f, m2, np.ones(np.shape(m2))]
        vec3 = [(lambda3 - c - e * m3)/f, m3, np.ones(np.shape(m3))]
        vec_array = np.array([vec1, vec2, vec3])

        #do the corresponding sort on the vec array
        #here, we have to sort row vectors, so we must vertically sort each component
        vec_array[:,0,:,:,:] = np.take_along_axis(vec_array[:,0,:,:,:], idx, axis=0)    #makes use of the sort indices idx
        vec_array[:,1,:,:,:] = np.take_along_axis(vec_array[:,1,:,:,:], idx, axis=0)
        vec_array[:,2,:,:,:] = np.take_along_axis(vec_array[:,2,:,:,:], idx, axis=0)

        return eig_array, vec_array
    else:
        return eig_array

def symmetric_eigvalsh(tensor_field, find_vectors = False):
    """Uses the symmetric_eigvals function to compute the eigenvalues of the hermitian tensor field.
    This is the alternate algorithm to np.linalg.eigvalsh() that is more efficient

    Parameters
    ----------
    tensor_field : np.ndarray
        The velocity gradient field used to find the stretching tensor, but any tensor field may be
        used here.
    """    
    #Decompose tensor field to find the symmetric component
    tensor_sym, _, _ = dv.orthogonal_tensor_decomposition(tensor_field)

    #Compute eigenvalues of the tensor
    #check whether we wish to find eigenvectors or not
    if find_vectors:
        #compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = symmetric_eigvals(tensor_sym, find_vectors)
        print("Eigenvalues and eigenvectors computed.")
        return eigenvalues, eigenvectors
    else:
        eigenvalues = symmetric_eigvals(tensor_sym)
        print("Eigenvalues computed.")
        return eigenvalues


def do_work_question1(i):
    """Computes the eigenvalues and eigenvectors for the velocity gradient field for the data at timestep i.
    Will pass to the multiprocessing library in order to do the computation faster.

    Parameters
    ----------
    i : int
        Identifier for the timestep. This is an argument that eventually defined by the multiprocessing function
    """
    print(f"Analyzing Frame: {i}")    
    if i_am_the_grader == True:
        file = h5py.File(directory + 'velocity_gradient.h5', 'r')
        vel_gradient = file['File' + str(i)][:]   #load in the velocity gradient field
        file.close()
        pass
    else:
        data = funcs.DataLoader(directory, i)   #this loads in the data file for timestep i, where its in units of (i/100)t_0
        turb = data.turb
        turb.read('vel')   #read in velocity and magnetic fields, gridsize for this data is 128^3

        #compute the velocity gradient tensor, performing a standard 4th order finite difference computation
        vel_gradient = dv.gradient_tensor(np.array([turb.velx, turb.vely, turb.velz]), order = 4)
    
    #compute the eigenvalues and eigenvectors of symmetric component of the velocity gradient tensor
    eigenvalues, eigenvectors = symmetric_eigvalsh(vel_gradient, find_vectors=True)

    #separate the eigenvalues into the compression, null, and stretching eigenvalues. These are defines simply based
    #on their value, which is why they were initially sorted in ascending order
    eig_compress = eigenvalues[0,:,:,:]
    eig_null = eigenvalues[1,:,:,:]
    eig_stretch = eigenvalues[2,:,:,:]
    
    #we can make an animation that shows their evolution through time with a histogram. 
    #Will generate a video of the eigenvalues by stringing together many images.
    fig= plt.figure()
    plt.hist(eig_compress.flatten(), density=True, bins = 50, histtype = 'step', color = 'r',label = r'$\lambda_{\rm c}$')
    plt.hist(eig_null.flatten(), density=True, bins = 50, histtype = 'step', color = 'g',label = r'$\lambda_{\rm n}$')
    plt.hist(eig_stretch.flatten(), density=True, bins = 50, histtype = 'step', color = 'b',label = r'$\lambda_{\rm s}$')
    plt.xlabel(r'$\lambda_i$')
    plt.ylabel(r'$p(\lambda_i)$')
    plt.legend()
    plt.xlim(-3,3)
    #to make scales reasonable
    if i>47:
        plt.ylim(0,6)
        plt.text(-0.5, 6.35, r'$t = $' + str(i/100) + r'$t_0$', fontsize=12, color='black')
    else:
        plt.text(-0.5, 32, r'$t = $' + str(i/100) + r'$t_0$', fontsize=12, color='black')
        plt.ylim(0,30)
    
    #save this plot in the temporary images directory
    #save the image figures
    image_name = "image" + str(i)
    plt.savefig(temp_directory + image_name + '.png')    
    #clear figures to save on RAM for each iteration
    fig.clear()
    plt.close(fig)

#QUESTION 2 FUNCTIONS

def orthogonal_tensor_decomposition(tensor_field : np.ndarray ):
    """
    Compute the symmetric, anti-symmetric and bulk components of a tensor field.
    
    Author: James Beattie

    Args:
        tensor_field (np.ndarray): 
        The tensor field to be decomposed.

    Returns:
        tensor_sym, tensor_anti, tensor_trace: np.ndarray
        These are the symmetric, antisymmetric, and bulk components of the tensor field.
    """
    
    # transpose
    tensor_transpose = np.einsum("ij... -> ji...",tensor_field)
    
    # bulk component
    tensor_trace = (1./3.) * np.einsum("ii...",tensor_field)
    
    # symmetric component
    tensor_sym = 0.5 * (tensor_field + tensor_transpose) -  np.einsum('...,ij...->ij...',tensor_trace,np.identity(3))
    
    # anti-symmetric component
    tensor_anti = 0.5 * (tensor_field - tensor_transpose)
    
    return tensor_sym, tensor_anti, tensor_trace

#QUESTION 3 FUNCTIONS
def dot_product(vector_a, vector_b):
    """Will compute the dot product of two vectors

    Parameters
    ----------
    vector_a : array
        3D spacial array of vector components
    vector_b : array
        3D spacial array of vector components
    """
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    return (np.sum(vector_a*vector_b, axis = 0))

def vector_angle(vector_a, vector_b):
    """Will compute the angle between two vectors

    Parameters
    ----------
    vector_a : array
        3D spacial array of vector components
    vector_b : array
        3D spacial array of vector components
    """
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    return np.arccos(dot_product(vector_a, vector_b)/(np.sqrt(dot_product(vector_a, vector_a))*np.sqrt(dot_product(vector_b, vector_b))))

def do_work_question3(i):
    """Will compute the angle between the eigenvectors of the stretching tensor and the magnetic field 
    at each point within the grid for timestep i.

    Parameters
    ----------
    i : int
        the timestep of analysis
    """
    print(f"Analyzing Frame: {i}")
    if i_am_the_grader == True:
        file = h5py.File(directory + 'velocity_gradient.h5', 'r')
        vel_gradient = file['File' + str(i)][:]   #load in the velocity gradient field
        file.close()
        file = h5py.File(directory + 'mag_field.h5', 'r')
        mag_field = file['File' + str(i)][:]   #load in the magnetic field
        file.close()
        pass
    else:
        data = funcs.DataLoader(directory, i)
        turb = data.turb
        turb.read('vel')
        turb.read('mag')
        vel_gradient = dv.gradient_tensor(np.array([turb.velx, turb.vely, turb.velz]), order = 4)
        mag_field = np.array([turb.magx, turb.magy, turb.magz])
    #we will first find the eigenvalues/vectors of the symmetric component of the velocity gradient tensor
    eigenvalues, eigenvectors = symmetric_eigvalsh(vel_gradient, find_vectors=True)

    #separate eigenvalues into different arrays
    eig_compress = eigenvalues[0,:,:,:]
    eig_null = eigenvalues[1,:,:,:]
    eig_stretch = eigenvalues[2,:,:,:]

    #define the eigenvector with the largest eigenvalue to be the stretching eigenvector
    stretching_vector = eigenvectors[2,:,:,:,:]
    
    #compute the angle between eigenvector and magnetic vector
    angle = vector_angle(stretching_vector, mag_field)
    #convert angle to absolute angle, where its angle if less than pi/2, and its pi - angle if more than pi/2
    condition_list = [angle < np.pi/2, angle >= np.pi/2]
    choice_list = [angle, np.pi - angle]
    angle = np.select(condition_list, choice_list)

    #now, lets plot the angle distribution and eigenvalues
    #1X2 SUBPLOT FOR THETA, AN ALL THREE EIG PDFS
    #=============================
    #FOR FORMATTING
    ticklen = np.pi/4
    
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    axes[0].hist(angle.flatten(), density=True, bins = 50, histtype = 'step', color = 'r')
    axes[0].set_xlabel(r'$\theta_{\hat{\vb*{b}},\xi_{\rm s}}$')
    axes[0].set_ylabel(r'$p(\theta_{\hat{\vb*{b}},\xi_{\rm s}})$')

    # setting ticks labels
    axes[0].xaxis.set_major_formatter(FuncFormatter(funcs.pi_axis_formatter))
    # setting ticks at proper numbers
    axes[0].xaxis.set_major_locator(MultipleLocator(base=ticklen))

    axes[1].hist(eig_compress.flatten(), density=True, bins = 50, histtype = 'step', color = 'r',label = r'$\lambda_{\rm c}$')
    axes[1].hist(eig_null.flatten(), density=True, bins = 50, histtype = 'step', color = 'g',label = r'$\lambda_{\rm n}$')
    axes[1].hist(eig_stretch.flatten(), density=True, bins = 50, histtype = 'step', color = 'b',label = r'$\lambda_{\rm s}$')
    axes[1].set_xlabel(r'$\lambda_i$')
    axes[1].set_ylabel(r'$p(\lambda_i)$')
    axes[1].legend()


    #fix limits for purpose of animation
    axes[0].set_xlim(0,np.pi/2)
    axes[0].set_ylim(0,1)
    axes[1].set_xlim(-3,3)
    if i > 41:
        axes[1].set_ylim(0,10)
        axes[1].text(-4.5,10.7, r'$t = ' + str(i/100) + r't_0$', fontsize=18, color='black')
    else:
        axes[1].text(-4.5,235, r'$t = ' + str(i/100) + r't_0$', fontsize=18, color='black')
        axes[1].set_ylim(0,225)

    #save this plot in the temporary images directory
    #save the image figures
    image_name = "image" + str(i)
    plt.savefig(temp_directory + image_name + '.png')    
    #clear figures to save on RAM for each iteration
    fig.clear()
    plt.close(fig)

#========================================================================================================
#This runs the parallelized computations
#Note: If video generation encounters errors, you can rerun this code for the timesteps that are missing image files
if __name__ == '__main__':
    pool = Pool(n_cores)
    if question_being_graded == 1:
        pool.map(do_work_question1, np.arange(1,400+1))    #the timesteps of the data range from 1 to 400
        #This code generates the video from the temporary images, and then clears temporary images
        create_video(1,400,[],temp_directory, directory, 'eigenvalue_evolution',10)
    if question_being_graded == 2:
        #In question two, we will test the speed of the symmetric_eigvals function compared to other standard
        #ways of computing eigenvalues, including scripts from numpy and scipy
        #use one data file as a testcase
        if i_am_the_grader == True:
            file = h5py.File(directory + 'velocity_gradient.h5', 'r')
            vel_gradient = file['File1'][:]   #load in the velocity gradient field
            file.close()
            pass
        else:
            data = funcs.DataLoader(directory, 123)
            turb = data.turb
            turb.read('vel')

            #compute the velocity gradient tensor, performing a standard 4th order finite difference computation
            vel_gradient = dv.gradient_tensor(np.array([turb.velx, turb.vely, turb.velz]), order = 4)
        #the numpy function np.linalg.eigvalsh() cannot be vectorized, so we expect it to be much slower
        #as we have to loop through the entire array

        print("Calculating with np.linalg.eigh()...")
        start_np_eigh = time.time()         #begin measuring time

        #initialize eigenvalue array
        eigenvalues_numpy = np.zeros((3,128,128,128))
        eigenvectors_numpy = np.zeros((3,3,128,128,128))
        
        #define symmetric component of the field
        tensor_sym,_,_ = orthogonal_tensor_decomposition(vel_gradient)
        #loop through array
        for i in np.arange(128):
            for j in np.arange(128):
                for k in np.arange(128):
                    symmetric_matrix = tensor_sym[:,:,i,j,k]   #define the matrix
                    #compute the eigenvalues
                    eigenvalues_numpy[:,i,j,k], eigenvectors_numpy[:,:,i,j,k] = np.linalg.eigh(symmetric_matrix)
        end_np_eigh = time.time()           #end measuring time
        print(f'The time taken to compute all eigenvalues using np.linalg.eigh() is: \
              {end_np_eigh - start_np_eigh} seconds. \n')
        
        #what if we compute with scipy
        print("Calculating with scipy.linalg.eigh()...")
        start_sp_eigh = time.time()         #begin measuring time

        #initialize eigenvalue array
        eigenvalues_scipy = np.zeros((3,128,128,128))
        eigenvectors_scipy = np.zeros((3,3,128,128,128))

        #loop through array
        for i in np.arange(128):
            for j in np.arange(128):
                for k in np.arange(128):
                    #compute the eigenvalues
                    symmetric_matrix = tensor_sym[:,:,i,j,k]   #define the symmetric matrix
                    eigenvalues, eigenvectors_scipy[:,:,i,j,k]= scipy.linalg.eig(symmetric_matrix,check_finite = False)
                    eigenvalues_scipy[:,i,j,k] = np.sort(eigenvalues, axis=0)

        end_sp_eigh = time.time()           #end measuring time
        print(f'The time taken to compute all eigenvalues using scipy.linalg.eigh() is: \
              {end_sp_eigh - start_sp_eigh} seconds. \n')
        
        #what if we compute with our function
        print("Calculating with symmetric_eigvals()...")

        start_sym_eigh = time.time()         #begin measuring time
        eigenvalues_our_method, eigenvectors_our_method = symmetric_eigvalsh(vel_gradient, find_vectors=True)
        end_sym_eigh = time.time()           #end measuring time
        print(f'The time taken to compute all eigenvalues using symmetric_eigvals() is: \
              {end_sym_eigh - start_sym_eigh} seconds.')

        #lets see how similar the values are with our method and the numpy/scipy methods
        numpy_diff = eigenvalues_numpy- eigenvalues_our_method
        scipy_diff = eigenvalues_scipy - eigenvalues_our_method
        
        #plot histograms to see the difference distribution
        plt.figure()
        plt.hist(numpy_diff.flatten(), bins = 50, histtype = 'step', color = 'r')
        plt.ylabel('Probability Density')
        plt.xlabel('Eigenvalue Difference')
        plt.title('Difference with numpy.eigh()')
        plt.show()

        plt.figure()
        plt.hist(scipy_diff.flatten(), bins = 50, histtype = 'step', color = 'g')
        plt.ylabel('Probability Density')
        plt.xlabel('Eigenvalue Difference')
        plt.title('Difference with scipy.linalg.eigh()')
        plt.show()        

    if question_being_graded == 3:
        #In this question, we will take the eigenvectors of the stretching tensor and compute the angle between them
        pool.map(do_work_question3, np.arange(1,400+1))    #the timesteps of the data range from 1 to 400
        #generate video
        create_video(1,400,[],temp_directory, directory, 'angle_eigval_evolution',10)
'''
    if question_being_graded == 4:
        #Generate the datafiles for the grader
        #open the hdf5 
        hf5 = h5py.File(directory + 'mag_field.h5', 'w')
        #loop through the data
        for i in np.arange(1,400+1):
            print(f"Generating data for File{i}...")
            data = funcs.DataLoader(directory, i)
            turb = data.turb
            turb.read('mag')
            mag_field = np.array([turb.magx, turb.magy, turb.magz])
            hf5.create_dataset(f'File{i}', data=mag_field)
        hf5.close()
'''


        

    























