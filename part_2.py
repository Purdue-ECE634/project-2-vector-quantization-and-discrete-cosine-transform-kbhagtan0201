## Make necessary imports

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import cv2
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from PIL import Image
import math
import time
from sklearn.cluster import KMeans
import skimage
from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import scc

##############################################################################
######### Metrics to verify how well is the reconstruction ###################
##############################################################################


##############################################################################
## Borrowed from ECE634 Project 1
## Function to compute PSNR of the reconstructed image relative to its original

# This code is inspired and adapted from: 
# https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
def compute_psnr(orig,recons):
    
    '''
        Input - orig : original image
              - recons : predicted image 

        Output - psnr value between the two in dB
    '''

    # PSNR = 20log10(Imax/(sqrt(MSE)))
    psnr_db = 20*math.log10(255/math.sqrt(np.mean((orig-recons)**2)))
    
    return psnr_db

##############################################################################
## Function to compute MSE of the reconstructed image relative to its original

# Source of equations: https://towardsdatascience.com/deep-image-quality-assessment-30ad71641fac
def compute_mse(orig,recons):
    
    '''
        Input - orig : original image
              - recons : predicted image 

        Output - mean squared error between the two 
    '''

    s1 = orig.shape[0]
    s2 = orig.shape[1]
    mse = (np.sum((orig - recons)**2))/(s1*s2)
    
    return mse

##############################################################################
## Function to compute Structural Similarity Index Measure (SSIM)
## of the reconstructed image relative to its original

# Source of equations: https://towardsdatascience.com/deep-image-quality-assessment-30ad71641fac
def compute_ssim(orig,recons):
    
    '''
        Input - orig : original image
              - recons : predicted image 

        Output - SSIM between the two 
    '''
    
    SSIM = ssim(orig,recons,data_range=orig.max() - orig.min())
    
    return SSIM

##############################################################################
## Function to compute Spatial Correlation Coefficient (SCC)
## of the reconstructed image relative to its original

# Source of equations: 
# https://towardsdatascience.com/measuring-similarity-in-two-images-using-python-b72233eb53c6
def compute_scc(orig,recons):
    
    '''
        Input - orig : original image
              - recons : predicted image 

        Output - SCC between the two 
    '''
    SCC = scc(orig,recons)
    
    return SCC

##############################################################################
## Function to select top K DCT cofficients and set the rest to zero

## idea and inspiration of this code has been derived from:
## https://github.com/getsanjeev/compression-DCT/blob/master/zigzag.py
## But the code is very different and entirely my own implementation

def select_K_dct(array,K,s1,s2):
    
    
    '''
        Input - array : DCT transformed image block
              - K : Number of DCT coefficients to be selected
              - s1 : first element of the shape of block
              - s1 : second element of the shape of block

        Output - DCT transformed block with only selected K DCT coefficients,
                 rest all set to zero
    '''
    
    # declare final 8X8 output containing only K DCT coefficients
    output_array = np.zeros((s1,s2))
    
    # to keep track of directions in zig-zag ordering, we use a dummy variable
    # from (0,0) we start in the right direction using zigzag
    # dummy_var = 34 stands for right
    # dummy_var = 35 stands for down
    # dummy_var = 36 stands for top-right direction
    # dummy_var = 37 stands for bottom-left direction
    # any dummy values can be chosen
    
    dummy_var = 34
    
    # assign the first value in the top-left corner of the matrix
    output_array[0][0] = array[0][0]
    i = 0
    j = 0
    for coeff in range(K-1):
        
        # if it goes down, it can either go top-right or bottom-left
        # based on if it is at a border or not
        if dummy_var == 35 :
            i = i+1
            j = j
            if j != 0:
                dummy_var = 37
            elif j == 0:
                dummy_var = 36
                
            # assign the value after 1 update
            output_array[i][j] = array[i][j]

        # if it goes bottom-left, it can either go right or down
        # based on if it is at a border or not
        elif dummy_var == 37:
            i = i+1
            j = j-1
            if i == 7:
                dummy_var = 34
            elif j == 0:
                dummy_var = 35
                
            # assign the value after 1 update
            output_array[i][j] = array[i][j]
            
        # if it goes top-right, it can either go right or down
        # based on if it is at a border or not
        elif dummy_var == 36:
            i = i-1
            j = j+1
            if i == 0:  
                dummy_var = 34
            elif j == 7:
                dummy_var = 35
                
            # assign the value after 1 update
            output_array[i][j] = array[i][j]
            
        # if it goes right, it can either go top-right or bottom-left
        # based on if it is at a border or not
        else:
            i = i
            j = j+1
            if i != 7:
                dummy_var = 37
            elif i == 7:
                dummy_var = 36
                
            # assign the value after 1 update
            output_array[i][j] = array[i][j]
        
    # return the processed DCT block
    return output_array


# equations and ideas inspired from:
# https://cs.marlboro.college/cours/spring2019/algorithms/code/discrete_cosine_transform/dct.html
# but the entire implementation is different and my own

##############################################################################
## Function to get 8X8 DCT basis matrix
def get_dct_basis():
    
    dct_basis = np.zeros((8,8))

    for i in range(8):
        for j in range(8):

            # first row of the matrix
            if i == 0:
                dct_basis[i,j] = np.sqrt(2/8)/np.sqrt(2)
                
            # other rows are filled by cosing function
            else:
                dct_basis[i,j] = np.sqrt(2/8) * np.cos((np.pi/8)*i*(j+0.5))
    return dct_basis 

dct_basis = get_dct_basis()

# equations inspired from 
# https://eeweb.engineering.nyu.edu/~yao/EE3414/ImageCoding_DCT.pdf
# and
# https://cs.marlboro.college/cours/spring2019/algorithms/code/discrete_cosine_transform/dct.html

##############################################################################
## Function to apply 8X8 DCT basis matrix on whole image block by block
def apply_dct_whole_image(image,dct_basis,K):
    
    # shape of the original image
    s1 = image.shape[0]
    s2 = image.shape[1]
    
    # will store the DCT transformed image with selected K coefficients
    selected_dct = np.zeros((s1,s2))
    
    # will store the final reconstructed image
    reconstruction = np.zeros((s1,s2))
    
    for i in range(0,s1,8):
        for j in range(0,s2,8):
            
            # get a 8X8 block from image
            image_array = image[i:i+8,j:j+8]
            
            # apply DCT to the block
            conv_dct = np.matmul(dct_basis,image_array)
            dct_only_array = np.matmul(conv_dct,np.transpose(dct_basis))
            
            # select only K coefficients
            dct_select_coeff = select_K_dct(dct_only_array,K,8,8)
            selected_dct[i:i+8,j:j+8] = dct_select_coeff
            
            # Apply inverse DCT to the block and store the block in the final
            # reconstruction image
            inv_conv_dct = np.matmul(np.transpose(dct_basis),dct_select_coeff)
            reconstruction[i:i+8,j:j+8] = np.matmul(inv_conv_dct,dct_basis)
            
    return selected_dct, reconstruction


# Main function to experiment with number of DCT coefficients needed
K_array = [2,4,8,16,32]
print("Please enter relative image path to be used for vector quantization testing:")
image_path = input()
for k in K_array:
    print("Number of coefficients:", k)
    grayscale_image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    time1 = time.time()
    selected_dct, reconstruction = apply_dct_whole_image(grayscale_image,dct_basis,k) 
    PSNR = compute_psnr(grayscale_image,reconstruction)
    print('PSNR in dB: ',PSNR)
    MSE = compute_mse(grayscale_image,reconstruction)
    print('MSE: ',MSE)
    SSIM = compute_ssim(grayscale_image,reconstruction)
    print('SSIM: ',SSIM)
    SCC = compute_scc(grayscale_image,reconstruction)
    print('SCC: ',SCC)
    plt.figure()
    plt.imshow(selected_dct,cmap="gray")
    plt.title('DCT Transformed Image after selecting {0} coefficients'.format(k))
    plt.axis('off')
    plt.figure()
    plt.imshow(reconstruction,cmap="gray")
    plt.axis('off')
    plt.title('Reconstructed Image after selecting {0} DCT coefficients'.format(k))
    time2 = time.time()
    print("time difference in seconds",time2-time1)

# # plot for monarch

# K_array = [2,4,8,16,32]
# PSNR_array = [22.425855668950256,25.437414399835284,28.41316527366112,32.19811595464756,37.692644775091864]
# plt.figure()
# plt.plot(K_array,PSNR_array,'o-',color="green")
# plt.title('PSNR values vs number of DCT coefficients')
# plt.xlabel('Number of DCT coefficients (K)')
# plt.ylabel('PSNR Values (in dB)')

# MSE_array = [371.9587623463605,185.92548407313947,93.7052305438562,39.19843836402017,11.061547366376956]
# plt.figure()
# plt.plot(K_array,MSE_array,'*-',color="orange")
# plt.title('MSE values vs number of DCT coefficients')
# plt.xlabel('Number of DCT coefficients (K)')
# plt.ylabel('MSE Values')


# SSIM_array = [0.7451578450988654,0.8501520670385986,0.9137306613776752,0.9526926245614442,0.9808265660189316]
# plt.figure()
# plt.plot(K_array,SSIM_array,'+-',color="blue")
# plt.title('SSIM values vs number of DCT coefficients')
# plt.xlabel('Number of DCT coefficients (K)')
# plt.ylabel('SSIM Values')

# SCC_array = [0.02120225886977402,0.07851435970421446,0.20246852553955405,0.43836738947505,0.7333736082833089]
# plt.figure()
# plt.plot(K_array,SCC_array,'+-',color="magenta")
# plt.title('SCC values vs number of DCT coefficients')
# plt.xlabel('Number of DCT coefficients (K)')
# plt.ylabel('SCC Values')


# # plot for sails

# K_array = [2,4,8,16,32]
# PSNR_array = [22.28247831629206,23.996548617859823,25.876973739336776,28.74532877750401,33.502649690819204]
# plt.figure()
# plt.plot(K_array,PSNR_array,'o-',color="green")
# plt.title('PSNR values vs number of DCT coefficients')
# plt.xlabel('Number of DCT coefficients (K)')
# plt.ylabel('PSNR Values (in dB)')

# MSE_array = [384.4435060150037,259.07499538004043,168.0285141218966,86.80554800113846,29.027884861037165]
# plt.figure()
# plt.plot(K_array,MSE_array,'*-',color="orange")
# plt.title('MSE values vs number of DCT coefficients')
# plt.xlabel('Number of DCT coefficients (K)')
# plt.ylabel('MSE Values')


# SSIM_array = [0.41128426994032724,0.5870252797754392,0.7100170510972165,0.8534514626852846,0.9493239382203662]
# plt.figure()
# plt.plot(K_array,SSIM_array,'+-',color="blue")
# plt.title('SSIM values vs number of DCT coefficients')
# plt.xlabel('Number of DCT coefficients (K)')
# plt.ylabel('SSIM Values')

# SCC_array = [0.033118954392629255,0.09994333735232268,0.21478314877989158,0.48197978467964253,0.7882265937606306]
# plt.figure()
# plt.plot(K_array,SCC_array,'+-',color="magenta")
# plt.title('SCC values vs number of DCT coefficients')
# plt.xlabel('Number of DCT coefficients (K)')
# plt.ylabel('SCC Values')
    