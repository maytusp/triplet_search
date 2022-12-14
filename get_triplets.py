import numpy as np
import random
import pdb
import pandas as pd
from numpy import inf
from scipy import stats as st
import sys
import h5py
import os
import gc
import argparse
import pickle


# For example
# python get_triplets.py --upper_corr_path corr_mat/corr_CORnet-s_IT.npy --intermediate_corr_path corr_mat/corr_CORnet-s_V2.npy

parser = argparse.ArgumentParser()
parser.add_argument('--upper_corr_path', type=str, nargs='?', default='')
parser.add_argument('--intermediate_corr_path', type=str, nargs='?', default='')
parser.add_argument('--triplet_saved_path', type=str, nargs='?', default='saved_files')
parser.add_argument('--num_loops', type=int, nargs='?', default=10)

args = parser.parse_args()


def make_sym_matrix(n,vals):
    m = np.zeros([n,n], dtype=np.double)
    xs,ys = np.triu_indices(n,k=1)
    m[xs,ys] = vals
    m[ys,xs] = vals
    m[ np.diag_indices(n) ] = 0
    return m

# A sorting based program to count pairs with difference k 

def countPairs_new(arr_ori,arr2_ori,n1,n2,k1,k2,error_,ori_, prev_idx): 
    #The goal is to find image indices that their difference in correlation with the "root" image is k1 and k2 for layer 1 and layer 2 respectively. 
    #Input: 
    #ori_, this is the index of the "root" image
    #arr_ori, the correlation array of the "root" image with the rest of the images in terms of layer 1
    #arr2_ori, the correlation array of the "root" image with the rest of the images in terms of layer 2
    #n1,n2, are the lengths of arr_or, arr2_ori 
    #k1, k2 are the desired differences in correlation for layer 1, layer 2 (e.g., IT and V2)

    count =0
    indarr1= []
    indarr2=[]
    ori_array =[]
    ori_array_new = []
    ori_array_new1 = []
    ori_array_2 = []
    indarr11 = []
    indarr22 = []
    ind_s11 = []
    ind_s22 = []
    indarr1_new = []
    indarr2_new = []
    inter_ind = []
    inter2_ind = []
    ind_s1 = []
    ind_s2 = []

    # Sort array elements 
    ind_s = np.argsort(arr_ori)
    arr = np.sort(arr_ori) 
    arr_des  = np.sort(arr_ori)[::-1]
    arr2 = np.sort(arr2_ori)


    # 1st step.
    # Find indices i, j in the layer 1 array such that their correlation difference satisfies k1 (within error).
    # Store them.
    # Note: The way this is coded implies that, although abs(corr(j, root_)-corr(i, root_)) ~= k1, the correlation of element j with the root_ will always be bigger than the one of element i, i.e., corr(j, root_) > corr(i, root_)
    # 
    for i in range(n1):  
        j = i+1
        while j < n1 and  (((arr[j]-arr[i]))<error_+k1):
            if (arr[j]-arr[i]>k1-error_) and ind_s[i] != ori_ and ind_s[j] != ori_:
                inter_ind.append(ind_s[i])
                inter2_ind.append(ind_s[j])
                ind_s1.append(i)
                ind_s2.append(j)
                ori_array.append(ori_)
            j += 1

    # 2nd step. 
    # Out of these indices, select those that their correlation difference in layer 2 is k2 (within error).
    # Store them.
    # Note: The way this is coded, correlation in layer 2 of inter_ind[i] with the root_ is always bigger than that of inter2_ind[i]. This is a problem because eventually you will always return indices inter_ind[i], inter2_ind[i] such that corr(inter_ind[i], root_)>corr(inter2_ind[i], root_) for V2 and those indices will ALSO have corr(inter_ind[i], root_) > corr(inter2_ind[i] for IT because of step 1!! Thus you need to run this script again with the ordering reversed i.e., change the code to abs((arr2_ori[inter2_ind[i]] - arr2_ori[inter_ind[i]])-k2)<error_  How can we do this in one script?
    n2_new = len(inter_ind)

    if len(inter_ind) != 0:
        for i in range(n2_new):
            if k2-error_<=abs((arr2_ori[inter_ind[i]] - arr2_ori[inter2_ind[i]]))<=k2+error_:
                if not(inter_ind[i] in prev_idx) and not(inter2_ind[i] in prev_idx):
                    prev_idx[inter_ind[i]] = True
                    prev_idx[inter2_ind[i]] = True                
                    indarr1_new.append(inter_ind[i])
                    indarr2_new.append(inter2_ind[i])
                    ori_array_new.append(ori_array[i])
    else:
        ori_array_new =[]
        indarr1_new =[]
        indarr2_new = []

        # So eventually, you get pairs of images i,j (stored in indarr1_new, indarr2_new) that their difference with the root_ i.e., abs(corr(i,root_)-corr(j,root_)) is about k1 and k2 for IT and V2. Again, note that that the way is coded is that corr(j,root_) > corr(i,root_) for IT and corr(i, root_) > corr(j, root_) for V2. This will mess up when we want to randomly choose triplets. Thus we need to balance it out by also choosing corr(inter_ind[i], root_) < corr(inter2_ind[i], root_). See note above!
    return (ori_array_new,indarr1_new,indarr2_new)


def check_delta(delta_p1, delta_p2, mat_diff_upper, mat_diff_intermediate, prev_idx, num_bins=10, bin_size=0.5):
    # Input: 
    # delta_p1: this is a parameter for defining the desired difference in correlations for layer 1 (see below)
    # delta_p2: this is a parameter for defining the desired difference in correlations for layer 1 (see below)
    # mat_diff_upper: the images x images matrix of correlations for layer 1
    # mat_diff_intermediate: the images x images matrix of correlations for layer 2

    num_images = mat_diff_upper.shape[0]

    # Vectorize them
    xx2_vec=np.reshape(mat_diff_intermediate,mat_diff_intermediate.shape[0]*mat_diff_intermediate.shape[0])
    xx_vec=np.reshape(mat_diff_upper,mat_diff_upper.shape[0]*mat_diff_upper.shape[0])

    std1 = np.std(xx_vec)
    std2 = np.std(xx2_vec)

    # Shuffle the indices of the matrices
    arr = np.arange(num_images)
    np.random.shuffle(arr)
    arr = np.random.permutation(num_images)

    # allocation stuff (not sure I need all of these : ))
    keep_pic1 = [] 
    keep_mat11 = []
    keep_mat12  = []

    ori_keep = []
    ct = 0
    ct2=0
    ct3=0
    ct4=0
    ct5=0
    failed = 0
    keep_int1 =0 

    
    # Create map for determining k1 and k2
    map_delta = {}
    for bin_idx in range(1, num_bins+1):
        map_delta[str(bin_idx)] = bin_idx * bin_size
    k1 = std1 * map_delta[str(delta_p1)]
    k2 = std2 * map_delta[str(delta_p2)]

    # calculation error (hard coded for now)
    error_ = 0.01

    ############## Layer 1 level differences in correlations, i.e. we will calculate abs(corr(root_,image1)-corr(root_,image2)=DIFF) where corr(a,b) is the correlation of images at layer 1 and DIFF will span the entire distribution of correlations. I chose 10 buckets for DIFF based on the std of the correlations. Thus for each delta_p, I am choosing a different buck to extract differences from.
    # Layer 2 level difference in correlations, i.e., we will calculate abs(corr(root_,image1)-corr(root_,image2)=DIFF) where corr(a,b) is the correlation of images at layer 2 and DIFF will span the entire distribution of correlations. I chose 10 buckets for DIFF based on the std of the correlations.
    #########
    # So now k1_ has determined the desired correlation difference for layer 1, and err_X has determined the entire space of correlation differences for layer 2 (where X is a bucket). (and k2 for the layer 2)
    #########
    #########

    keep_pic1_ref =[]
    keep_mat11_ref =[]
    keep_mat12_ref =[]


    keep_pic1_ref_f = []
    keep_mat11_ref_f = []
    keep_mat12_ref_f =[]


    ori_mat = np.random.choice(num_images-1, 10, replace=False)
    ori_ = ori_mat[0]

    # This is the "root" image
    acc = []
    acc2 =[]
    ori_keep.append(ori_)
    slice_ = mat_diff_upper[ori_,:]
    slice_2 = mat_diff_intermediate[ori_,:]
    ln1 = len(slice_)
    ln2 = len(slice_2)
    # Call the function to return indices such that they satisfy abs(corr(root_,image1)-corr(root_,image2) = delta_) for layer 1 AND abs(corr(root_,image1)-corr(root_,image2) = err_1) for layer 2 (within error_)
    out= countPairs_new(slice_, slice_2, ln1, ln2, k1, k2, error_, ori_, prev_idx)
    # Unroll the output
    out1 = out[0] #root image
    out2 = out[1] #image 1 of the triplet
    out3 = out[2] #image 2 of the triplet
    if len(out1) != 0 and len(out2) !=0:
        keep_pic1_ref.append(out1)
        keep_mat11_ref.append(out2)
        keep_mat12_ref.append(out3)


    # After the loop is done, I am unrolling the previous lists. Eventually, keep_picX_ref will contain the root image (only one), keep_matX1_ref will contain all the potential image1's, keep_matX2_ref will contain all the potential image2's
    keep_pic1_ref = [item for sublist in keep_pic1_ref for item in sublist]
    keep_mat11_ref = [item for sublist in keep_mat11_ref for item in sublist]
    keep_mat12_ref = [item for sublist in keep_mat12_ref for item in sublist]

    #################


    ###############
    ############### In case you have > 1 image1's and image2's for each list, just choose a random one. Eventually, 
    # keep_picX_ref will contain the root image (only one), keep_matX1_ref will contain all the potential image1's, keep_matX2_ref will contain all the potential image2's
    
    if len(keep_pic1_ref) != 0:
        len_img = len(keep_pic1_ref) # = len(keep_mat11_ref) = len(keep_mat12_ref)
        rand_idx = np.random.choice([i for i in range(len_img)])

        keep_pic1_ref = keep_pic1_ref[rand_idx]
        keep_mat11_ref = keep_mat11_ref[rand_idx]
        keep_mat12_ref = keep_mat12_ref[rand_idx]

    else:

        keep_pic1_ref = []
        keep_mat11_ref = []
        keep_mat12_ref = []

    return (keep_pic1_ref, keep_mat11_ref, keep_mat12_ref)

# num_bins is the number of bins for the layer 1 and layer 2

def generate_triplet_table(mat_diff_upper, mat_diff_intermediate, num_bins=10, bin_size=0.5):
    triplet_table = {}
    prev_idx = {}
    for i1 in range(1, num_bins+1): # Loop over layer1's bins
        for i2 in range(1, num_bins+1): # Loop over layer2's bins
            delta_p1 = i1
            delta_p2 = i2
            triplet_table[str(i1)+'_'+str(i2)] = check_delta(delta_p1, delta_p2, mat_diff_upper, mat_diff_intermediate,
                                                        prev_idx, num_bins, bin_size)
    return triplet_table
            
def save_triplet_table(saved_path, triplet_table):
    with open(saved_path, 'wb') as f:
        pickle.dump(triplet_table, f)
        
def load_triplet_table(loaded_path):
    with open(loaded_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict
def create_new_dir(dir_name):
    if not(os.path.exists(dir_name)):
        # Create a new directory because it does not exist
        os.makedirs(dir_name)
        print("The new directory is created!")
    else:
        print("The directory already exists!")
    
if __name__ == '__main__':  
    #The main driver function. All you need is the two correlation matrices for layer 1 and layer 2. 
    #The way to call this function is python get_triplets_public_part1.py RANDOMNUMBER where RANDOMNUMBER can be any number. This basically creates separate folders with the root directory name corresponding to RANDOMNUMBER (I did this because I am running this script multiple times).

    # Load the correlation matrices for the two layers from VGG. These were derived by inputting about 26,000 images through a pretrained VGG-16 neural network and extracting features at each layer (see separate code for more information).
    mat_diff_upper = np.load(args.upper_corr_path)
    mat_diff_intermediate = np.load(args.intermediate_corr_path)

    create_new_dir(args.triplet_saved_path)
      

    #######################
    #### The goal of this program is to find triplets of images (root_, image1, image2) such that abs(corr(root_,image1)-corr(root_,image2)) = k1 for layer 1, and abs(corr(root_,image1)-corr(root_,image2) = k2) for layer 2. In other words, for a random image "root_" we are trying to find two additional images "image1" and "image2" such that their difference in terms of correlation with the root image satisfies a value. The correlation is defined based on the layer 1 correlation matrices, or layer 2 correlations matrices.
    # This external loop will run over all the desired correlation differences for layer 1-these have been clustered in 10 buckets corresponding to different k1 values (see internal function check_delta). Each loop will create different folder. Within each run you will have 10 subfolders. These correspond to 10 desired correlation differences for layer 2 (see internal function check_delta). Thus for each call of this program you will have 10 folders with 10 subfolder each. Each folder will have a triplet file .txt.

    #######################
    for loop_ in range(args.num_loops): 

        loop_saved_dir = os.path.join(args.triplet_saved_path,'run'+str(loop_))
        
        create_new_dir(loop_saved_dir)

        triplet_table = generate_triplet_table(mat_diff_upper, mat_diff_intermediate)
        saved_path = os.path.join(loop_saved_dir, 'triplet_table.pkl')
        save_triplet_table(saved_path, triplet_table)
