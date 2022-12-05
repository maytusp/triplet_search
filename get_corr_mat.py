from candidate_models.base_models import base_model_pool
from candidate_models.base_models.cornet import cornet
from candidate_models.model_commitments import model_layers
from sklearn.decomposition import PCA
import os
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--stim_data_path', type=str, nargs='?', default='object_images/*/*.jpg')
parser.add_argument('--saved_path', type=str, nargs='?', default='corr_mat')
parser.add_argument('--use_pca', default=False, action='store_true')
args = parser.parse_args()
if not os.path.exists(args.saved_path):
    # Create a new directory because it does not exist
    os.makedirs(args.saved_path)
    print("The new directory is created!")

file_list = glob.glob(args.stim_data_path)
stimuli_path_list = [file for file in file_list]
stimuli_path_list = stimuli_path_list[0:1000]

# The library "candidate_models" (part of brainscore) has many models including VGGs, ResNets, AlexNet, etc.
# We can load a model and select its layer from candidates model
# The list of models can be found in the link below
# https://github.com/brain-score/candidate_models/blob/master/candidate_models/model_commitments/model_layer_def.py
model_name_list = ['CORnet-S', 'vgg-16', 'vgg-19', 'voneresnet-50', 'alexnet'] # This is for example, you can pick any models in the link above
print("Please select the model: \n")
print("Examples:", model_name_list, "\n")
model_name = str(input()) # Input

assert model_name != '', 'No model is selected'

if model_name == 'CORnet-S':
    model = cornet('CORnet-S',  separate_time=False)
    layer_name = ['V1', 'V2', 'V4', 'IT']
else:
    model = base_model_pool[model_name]  # (1)
    layer_name = model_layers[model_name]  # (2)
    
print("Please select the layer from the list: \n")
print(layer_name, "\n")

selected_layer = str(input())

assert model_name != '', 'No layer is selected'

activations = np.array(model(stimuli=stimuli_path_list, layers=[selected_layer]))  # (3)
act_dims = activations.shape[1]
# Compte Correlation Matrix
if args.use_pca:
    pca = PCA(n_components=0.95)
    pca.fit(activations)
    reduced_act = pca.transform(activations)
    print('Reduce dimensions from ', act_dims, ' to ', reduced_act.shape[1])
    corr_mat = np.corrcoef(reduced_act)
else:
    corr_mat = np.corrcoef(activations)

saved_file_activations = os.path.join(args.saved_path, 'activation_'+model_name + "_" + selected_layer + ".npy")
saved_file_corr = os.path.join(args.saved_path, 'corr_'+model_name + "_" + selected_layer + ".npy")
np.save(saved_file_activations , activations)
np.save(saved_file_corr , corr_mat)
print("Done")