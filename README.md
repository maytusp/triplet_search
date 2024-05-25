# Triplet Search
Contributors: [Ioannis Pappas (USC)](https://scholar.google.co.uk/citations?user=M-zFg4kAAAAJ&hl=en) and [Maytus Piriyajitakonkij (VISTEC)](https://maytusp.com)
### Description
The code is for creating a triplet contained of 3 images (root_img, img_1, img_2) such that
the similarity between img_1 and img_2 is equal to k1 calculated from the neural responses from the layer 1 
and the similarity between img_1 and img_2 is equal to k2 calculated from the neural responses from the layer 2. \
The similarity score is between img_1 and img_2 is calculated from the difference in the correlation of neural responses between root_img and img_1
and the correlation of neural responses between root_img and img_2. 
\
\
i.e., 
\
|| corr(F(root_img), F(img_1)) - corr(F(root_img), F(img_2)) || = k1 for the layer 1
\
|| corr(G(root_img), G(img_1)) - corr(G(root_img), G(img_2)) || = k2 for the layer 2  
\
\
Where F and G are the functions that transform input images to the responses of the layer 1 and 2 respectively.
\

### Pre-requisites
1. Pytorch 2.x (This is for get_corr_mat.py, you can use your own script to get correlation of model layers)

If you want to add new neural network models, you may want to compute their Brain-Score to select the most suitable layers for predicitng neural responses.

3. (Optional) Install BrainScore from https://github.com/brain-score/brain-score
4. 
5. (Optional) Install Candidate models from https://github.com/brain-score/candidate_models


### Example
1. Getting correlation matrices from the layer X of the neural network Y 
```
python get_corr_mat.py --stim_data_path object_images/*/*.jpg --saved_path corr_mat --use_pca
```
2. Creating triplets
```
python get_triplets.py --upper_corr_path corr_mat/corr_CORnet-s_IT.npy --intermediate_corr_path corr_mat/corr_CORnet-s_V2.npy
```
