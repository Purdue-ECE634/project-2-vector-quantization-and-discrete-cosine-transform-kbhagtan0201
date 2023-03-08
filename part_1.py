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
import itertools
import skimage
from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import scc

##############################################################################
############ Training using 1 image and quantize it ##########################
##############################################################################

##############################################################################
## Function to read an image, convert to grayscale & partition into 4X4 blocks

def partition_image(image_path):
    
    '''
    Input - image_path : provide relative image path
          
    Output - A dictionary and a list of arrays of 4X4 blocks of image
        
    '''
    
    # initialize the list and dictionary
    # both are required because dictionary is used for tracking 
    # position of the 4X4 block in the original image
    vector_dict = {}
    vector_array = []
    
    # read image
    img = mpimg.imread(image_path)
    
    # convert it to grayscale
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # show the grayscale version of original image
    plt.imshow(grayscale_image,cmap="gray")
    plt.axis('off')
    
    # number of partitions in x direction
    num_x = int(grayscale_image.shape[0]/4)
    
    # number of partitions in y direction
    num_y = int(grayscale_image.shape[1]/4)
    
    # partition the image and keep appending to the dictionary and list
    for i in range(num_x):
        for j in range(num_y):
            
            # 4X4 block
            block = grayscale_image[i*4 : (i + 1)*4, j*4 : (j + 1)*4]
            
            # flatten to form a 1-D vector
            block_vector = block.flatten()
            
            # append to dict and list
            vector_dict[i,j] = block_vector
            vector_array.append(block_vector) 
            
    return vector_dict,vector_array

##############################################################################
####### Implementation of VQ using inbuilt K-means from sklearn ##############
##############################################################################

##############################################################################
## Function to obtain partitions of image and apply VQ through KMeans
## Implementation syntax has been taken from:
## https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

def inbuilt_kmeans(image_path,L):
    
    # L is the size of the codebook (number of vector codes in the codebook)
    # obtain sample vector blocks from training image
    # X stores the training array of vectors
    vector_dict,vector_array = partition_image(image_path)
    X = np.array(vector_array)
    
    # Apply KMeans, number of clusters = L because we want L codes
    kmeans = KMeans(n_clusters=L, random_state=0).fit(X)
    
    # centers of the clusters obtained, these act as codes for us
    centroids = kmeans.cluster_centers_
    
    return centroids,kmeans

##############################################################################
## Function to quantize the image using codebook

def quantize_images(image_path,im_shape,codebook,vec_size,L):
    
    
    '''
    Input - image_path : provide relative image path
            im_shape : shape of the grayscale image as a tuple
            codebook: Array of L codewords which are vectors of size 16
            vec_size: 16 in our case as we use 4X4 blocks
            L: size of the the codebook (number of codewords)
          
    Output - quantized_image: image obtained after testing using the codebook
        
    '''
    
    # obtain sample vector blocks from testing image
    vector_dict,vector_array = partition_image(image_path)
    
    # declare empty quantization image
    quantized_image = np.zeros((im_shape[0],im_shape[1]))
    
    # these keys help in deciding locations of each 4X4 block in the original image
    keys = list(vector_dict.keys())
    
    # obtain codebook after training
    codebook,kmeans = inbuilt_kmeans(image_path,L)
    
    for i in range(len(keys)):
        
        numpy_array = np.array([vector_dict[keys[i]]])
        
        # predict each 4X4 block one by one based on a distance metric
        # each 4X4 block is assigned to the code which is closest to it
        assigned_code = vector_array[kmeans.predict(numpy_array)[0]]
        
        # reshape test vector as 4X4 block and put it into the output quantization image
        assigned_block = assigned_code.reshape((4,4))
        quantized_image[4*keys[i][0]:4*(keys[i][0] + 1),4*keys[i][1]:4*(keys[i][1] + 1)] = assigned_block
    
    return quantized_image


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
############ Training using 10 images and quantization #######################
##############################################################################


##############################################################################
## Function to obtain partitions of 10 images for training and apply VQ through KMeans
## Implementation syntax has been taken from:
## https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

def train_10_images(image_path,L):
    
    # these are the 10 images used for training
    image_paths = ['sample_image/pool.png','sample_image/watch.png','sample_image/mountain.png',
               'sample_image/tulips.png','sample_image/baboon.png','sample_image/fruits.png',
               'sample_image/peppers.png','sample_image/cat.png','sample_image/img03.tif',
               'sample_image/img12.tif']

    # will store training vectors from all 10 images
    A = []
    for im in image_paths:
        print(im,'\n')
        vector_dict,vector_array = partition_image(im)
        A.append(vector_array)

    F = [i for sub in A for i in sub]
    
    # training vector from all 10 images
    X = np.array(F)
    
    # Apply KMeans for training
    # Keep higher number of codewords for less lossy quantization
    kmeans = KMeans(n_clusters=L*5, random_state=0).fit(X)
    
    # return the codewords/codebook
    centroids = kmeans.cluster_centers_
    
    return centroids,kmeans


##############################################################################
## Function for testing the codebook obtained using 10 training images
def quantize_images_10(image_path,im_shape,codebook,vec_size,L):
    
    vector_dict,vector_array = partition_image(image_path)
    
    # will store the final quantized image
    quantized_image = np.zeros((im_shape[0],im_shape[1]))
    
    # to keep track of location of 4X4 blocks in the testing image
    keys = list(vector_dict.keys())
    
    # codebook
    codebook,kmeans = train_10_images(image_path,L)
    for i in range(len(keys)):
        numpy_array = np.array([vector_dict[keys[i]]])
        
        # predict code for each block in the test image
        # the closest code is assigned to it
        assigned_code = vector_array[kmeans.predict(numpy_array)[0]]
        
        # reshape the vector as 4X4 block and put it into the image
        assigned_block = assigned_code.reshape((4,4))
        quantized_image[4*keys[i][0]:4*(keys[i][0] + 1),4*keys[i][1]:4*(keys[i][1] + 1)] = assigned_block
    
    return quantized_image

##############################################################################
## Main code for training using 1 image and testing

print("Please enter codebook size (L):")
L = int(input())
print("Please enter vectorsize (for 4X4 blocks, vectorsize = 16):")
vec_size = int(input())
print("Please enter relative image path to be used for vector quantization testing:")
image_path = input()
img = mpimg.imread(image_path)
grayscale_image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
time1 = time.time()
codebook,kmeans = inbuilt_kmeans(image_path,L)
quantized_image = quantize_images(image_path,(img.shape[0],img.shape[1]),codebook,vec_size,L)
plt.figure()
plt.imshow(quantized_image,cmap="gray")
plt.axis('off')
time2 = time.time()
print("Time taken to run the code in seconds",time2-time1)
PSNR = compute_psnr(grayscale_image,quantized_image)
print('PSNR in dB: ',PSNR)
MSE = compute_mse(grayscale_image,quantized_image)
print('MSE: ',MSE)
SSIM = compute_ssim(grayscale_image,quantized_image)
print('SSIM: ',SSIM)
SCC = compute_scc(grayscale_image,quantized_image)
print('SCC: ',SCC)

##############################################################################
## Main code for training using 10 images and testing

# print("Please enter codebook size (L):")
# L = int(input())
# print("Please enter vectorsize (for 4X4 blocks, vectorsize = 16):")
# vec_size = int(input())
# print("Please enter relative image path to be used for vector quantization testing:")
# image_path = input()
# img = mpimg.imread(image_path)
# grayscale_image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
# time1 = time.time()
# codebook,kmeans = train_10_images(image_path,L)
# quantized_image = quantize_images_10(image_path,(img.shape[0],img.shape[1]),codebook,vec_size,L)
# plt.figure()
# plt.imshow(quantized_image,cmap="gray")
# plt.axis('off')
# time2 = time.time()
# print("Time taken to run the code in seconds",time2-time1)
# PSNR = compute_psnr(grayscale_image,quantized_image)
# print('PSNR in dB: ',PSNR)
# MSE = compute_mse(grayscale_image,quantized_image)
# print('MSE: ',MSE)
# SSIM = compute_ssim(grayscale_image,quantized_image)
# print('SSIM: ',SSIM)
# SCC = compute_scc(grayscale_image,quantized_image)
# print('SCC: ',SCC)





##############################################################################
## My own partially working implementation of Generalized Lloyd algorithm ####
##############################################################################
########## (not complete because not efficient) ##############################
## was taking long time to run

## NOTE: I tried to partially implement Generalized Lloyd algorithm without using
## inbuilt KMeans function from sklearn. But the following piece of code
## is not used in the outputs provided.


def choosing_initial_codewords(L,vec_size,vector_dict,vector_array):
    
    code_book = []
    for i in range(L):
        code_book.append(vector_array[i])
    return code_book

def compute_mse_distortion_measure(vector_1,vector_2,vec_size):
    
    vec_sum = 0
    for i in range(vec_size):
        vec_sum = vec_sum + (vector_1[i] - vector_2[i])**2
    mse = vec_sum/vec_size
    return mse

def find_initial_partition(image_path,L,vec_size):
    
    vector_dict,vector_array = partition_image(image_path)
    init_code = choosing_initial_codewords(L,vec_size,vector_dict,vector_array)
    init_coded_vec_samples = []
    
    D0 = 0
    
    for i in range(len(vector_array)):
        
        min_dis = float('inf')
        best_code = None
        for j in range(len(init_code)):
            
            distor = compute_mse_distortion_measure(vector_array[i],init_code[j],vec_size)
            D0 = D0 + distor
            if distor < min_dis:
                
                min_dis = distor
                best_code = init_code[j]
        init_coded_vec_samples.append(best_code)
    
    print("coded samples",len(init_coded_vec_samples))
    D0 = D0/len(vector_array)
    print("initial distortion",D0)
    
    return init_coded_vec_samples, D0

L = 128
vec_size = 16
find_initial_partition('sample_image/sails.png',L,vec_size)