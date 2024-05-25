import glob
import numpy as np
from numpy import inf
from scipy import stats as st
from PIL import Image
from matplotlib import pyplot as plt
import random
import pdb
import pandas as pd
import sys
import h5py
import os
import gc
import pickle
import pandas as pd
import argparse

def load_triplet_table(loaded_path):
    with open(loaded_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

def plot_triplet(triplet, file_list):
    image_ids = [triplet[0], triplet[1], triplet[2]]
    plt_titles = ['Root Images', 'Image 1', 'Image 2']
    # Show subplots | shape: (1,3) 
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
    for i, ax in enumerate(axs.flatten()):
        plt.sca(ax)
        plt.imshow(np.array(Image.open(file_list[image_ids[i]])))
        #plt.colorbar()
        plt.title(plt_titles[i])

    #plt.tight_layout()
    plt.suptitle('Triplet')
    plt.show()    




# 1. Check if there exisits repetitive image in all triplets
# 2. Check if there is an empty bin (i.e., contain no triplet that satisfies the similarity) in the triplet table
def check_empty_and_repeat(triplet_table, num_bins=10):
    empty_bin = [] # Determine an empty bin that has no triplet in the triplet table
    idx_counter = {} # Count image occurrence in the triplet table
    max_triplets = -100
    for i1 in range(1, num_bins+1): # Loop over layer1's bins
        for i2 in range(1, num_bins+1): # Loop over layer2's bins
            bin_name = "L1="+str(i1)+"_"+'L2='+str(i2)

            if bin_name in triplet_table:
                for num_triplet, triplet in enumerate(triplet_table[bin_name]):
                    
                    for img_num, idx in enumerate(triplet):
                        if idx in idx_counter:
                            if img_num == 0:
                                print("Found repetitive root image")
                            else:
                                print("Found repetitive images")
                            idx_counter[idx] += 1
                        else:
                            idx_counter[idx] = 1
            else:
                empty_bin.append(bin_name)
            max_triplets = max(num_triplet+1, max_triplets)
    return empty_bin, idx_counter, max_triplets


def get_report_from_triplet(saved_dir, triplet_tables, layer1_corr_mat, layer2_corr_mat, image_names, layer1_name='IT', layer2_name='V2', num_bins=10):
    data = []
    empty_bins = []
    for bin1 in range(1, num_bins+1):
        for bin2 in range(1, num_bins+1):
            bin_name = "L1="+str(bin1)+"_"+'L2='+str(bin2)
            if bin_name in triplet_tables:
                for triplet_idx, triplet in enumerate(triplet_tables[bin_name]):
                    root_image_idx, image1_idx, image2_idx = triplet
                    layer1_root_image1 = layer1_corr_mat[root_image_idx, image1_idx]
                    layer1_root_image2 = layer1_corr_mat[root_image_idx, image2_idx]
                    layer1_image1_image2 = layer1_corr_mat[image1_idx, image2_idx]
                    
                    layer2_root_image1 = layer2_corr_mat[root_image_idx, image1_idx]
                    layer2_root_image2 = layer2_corr_mat[root_image_idx, image2_idx]
                    layer2_image1_image2 = layer2_corr_mat[image1_idx, image2_idx]

                    layer1_corr_diff =  layer1_root_image1 - layer1_root_image2
                    layer2_corr_diff = layer2_root_image1 - layer2_root_image2

                    layer1_corr_abs_diff = np.abs(layer1_corr_diff)
                    layer2_corr_abs_diff = np.abs(layer2_corr_diff)

                    root_image_name, image1_name, image2_name = image_names[root_image_idx], image_names[image1_idx], image_names[image2_idx]

                    data.append([bin1, bin2, 
                                layer1_corr_abs_diff, layer2_corr_abs_diff, 
                                layer1_corr_diff, layer2_corr_diff,
                                root_image_name, image1_name, image2_name,
                                layer1_root_image1, layer1_root_image2, layer1_image1_image2,
                                layer2_root_image1, layer2_root_image2, layer2_image1_image2
                                ])
            else:
                empty_bins.append(f"{layer1_name} bin {bin1}, {layer2_name} bin {bin2}")
    columns = [f'{layer1_name} bin', f'{layer2_name} bin', 
            f'abs({layer1_name}(root,img1)-{layer1_name}(root,img2))', f'abs({layer2_name}(root,img1)-{layer2_name}(root,img2))', 
            f'{layer1_name}(root,img1)-{layer1_name}(root,img2)', f'{layer2_name}(root,img1)-{layer2_name}(root,img2)', 
            'root image', 'image1', 'image2',
            f'{layer1_name}(root,img1)', f'{layer1_name}(root,img2)', f'{layer1_name}(img1,img2)',
            f'{layer2_name}(root,img1)', f'{layer2_name}(root,img2)', f'{layer2_name}(img1,img2)'
            ]
    report_df = pd.DataFrame(data=data, columns=columns)
    return report_df, empty_bins

def get_histogram(dataframe, saved_dir, layer1_name='IT', layer2_name='V2', num_bins=10):
    layer1_diff_title = f'{layer1_name}(root,img1)-{layer1_name}(root,img2)'
    layer2_diff_title = f'{layer2_name}(root,img1)-{layer2_name}(root,img2)'
    layer1_diff = dataframe[layer1_diff_title]
    layer2_diff = dataframe[layer2_diff_title]

    ax1 = layer1_diff.plot.hist(bins=num_bins, figsize=(20,10))
    plt.title(layer1_diff_title)
    plt.savefig(os.path.join(saved_dir, f"{layer1_name}_diff.png"))
    plt.clf()

    ax2 = layer2_diff.plot.hist(bins=num_bins, figsize=(20,10))
    plt.title(layer2_diff_title)
    plt.savefig(os.path.join(saved_dir, f"{layer2_name}_diff.png"))
    plt.clf()
    return None

if __name__ == "__main__":
    '''
    image names are a list of image names which each index of the list corresponds to each index in correlation matrix
    For example, image_names = ['cat01.png', 'dog02.png', 'bird03.png'], index 0 means 'cat01.png' and so on
    The index pair (0,2) of a correlation matrix means the correlation between 'cat01.png' and 'dog02.png' responses.

    layer1 here is IT (upper layer)
    layer2 here is V2 (intermediate layer)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--upper_corr_path', type=str, nargs='?', default='')
    parser.add_argument('--intermediate_corr_path', type=str, nargs='?', default='')
    parser.add_argument('--triplet_path', type=str, nargs='?', default='triplet_table.pkl')
    parser.add_argument('--image_names_path', type=str, nargs='?', default='triplet_table.pkl')
    parser.add_argument('--saved_path', type=str, nargs='?', default='reports/report_vgg16_IT_f43_V2_f27_20_samples')


    args = parser.parse_args()

    layer1_corr_mat_dir = args.upper_corr_path
    layer2_corr_mat_dir = args.intermediate_corr_path 
    triplet_dir = args.triplet_path
    image_names_path = args.image_names_path
    saved_dir = args.saved_path



    if not(os.path.exists(saved_dir)):
        os.makedirs(saved_dir)
    triplet_tables = load_triplet_table(triplet_dir)
    empty_bin, idx_counter, max_triplets = check_empty_and_repeat(triplet_tables, num_bins=10)
    print('empty bins', empty_bin)
    # print("number of empty bins", len(empty_bin))
    # print('idx counter:', idx_counter)

    layer1_corr_mat = np.load(layer1_corr_mat_dir)
    layer2_corr_mat = np.load(layer2_corr_mat_dir)

    image_names = np.load(image_names_path) 
    
    report_df, empty_bins = get_report_from_triplet(saved_dir, triplet_tables, layer1_corr_mat, layer2_corr_mat, image_names, 'IT', 'V2')
    report_df.to_csv(os.path.join(saved_dir, 'report.csv'), index=False)
    
    print(empty_bins)
    print(report_df)
    get_histogram(report_df, saved_dir)
    
