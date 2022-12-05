# Triplet Search
 
### Pre-requisite
1. Install BrainScore from https://github.com/brain-score/brain-score
2. Install Candidate models from https://github.com/brain-score/candidate_models
\
\
This project is for searching a triplet contained of 3 images (root_img, img_1, img_2) such that
the similarity between img_1 and img_2 is equal to k1 calculated from the neural responses from the layer 1 
and the similarity between img_1 and img_2 is equal to k2 calculated from the neural responses from the layer 2. \
The similarity score is between img_1 and img_2 is calculated from the difference in the correlation of neural responses of between root_img and img_1.
and root_img and img_2. 
\
\
i.e., 
\
|corr(F(root_img), F(img_1)) - corr(F(root_img), F(img_2))| = k1 for the layer 1
\
|corr(G(root_img), G(img_1)) - corr(G(root_img), G(img_2))| = k2 for the layer 2  
\
\
Where F and G are the functions that transform input images to the responses of the layer 1 and 2 respectively.
\

### Example
1. Getting correlation matrices from the layer X of the neural network Y 
```
python get_corr_mat.py --stim_data_path object_images/*/*.jpg --saved_path corr_mat --use_pca
```
2. Creating triplets
```
python get_triplets.py --upper_corr_path corr_mat/corr_CORnet-s_IT.npy --intermediate_corr_path corr_mat/corr_CORnet-s_V2.npy
```
