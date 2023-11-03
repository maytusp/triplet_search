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
import torch
from get_activation import vgg16_activation
parser = argparse.ArgumentParser()
parser.add_argument('--stim_data_path', type=str, nargs='?', default="../../Images/few_thing_object_images")
#default='../../Images/thing_object_images/*/*.jpg') # For brainscore activation
parser.add_argument('--model_name', type=str, default='vgg-16', choices=['CORnet-S', 'vgg-16', 'vgg-19', 'voneresnet-50', 'alexnet'])
parser.add_argument('--model_from_brainscore', default=False, action='store_true')
parser.add_argument('--saved_path', type=str, nargs='?', default='corr_mat')
parser.add_argument('--use_pca', default=False, action='store_true')
parser.add_argument('--batch_size', default=32)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Configuring Pytorch
if not os.path.exists(args.saved_path):
    # Create a new directory because it does not exist
    os.makedirs(args.saved_path)
    print("The new directory is created!")

file_list = glob.glob(args.stim_data_path)
stimuli_path_list = [file for file in file_list]
# stimuli_path_list = stimuli_path_list[0:1000]
print("Num images:", len(stimuli_path_list))

# The library "candidate_models" (part of brainscore) has many models including VGGs, ResNets, AlexNet, etc.
# We can load a model and select its layer from candidates model
# The list of models can be found in the link below
# https://github.com/brain-score/candidate_models/blob/master/candidate_models/model_commitments/model_layer_def.py
model_name_list = ['CORnet-S', 'vgg-16', 'vgg-19', 'voneresnet-50', 'alexnet'] # This is for example, you can pick any models in the link above
print("Please select the model: \n")
print("Examples:", model_name_list, "\n")
model_name = args.model_name # str(input()) # Input

assert model_name != '', 'No model is selected'
if args.model_from_brainscore:
    if model_name == 'CORnet-S':
        model = cornet('CORnet-S',  separate_time=False)
        layer_name = ['V1', 'V2', 'V4', 'IT']
    else:
        model = base_model_pool[model_name]  # (1)
        layer_name = model_layers[model_name]  # (2)
else:
    if model_name == 'vgg-16':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=False)
        model.load_state_dict(torch.load('../../models/checkpoints/vgg16_bn-6c64b313.pth'))
        model.eval()
        features_layer_list = [str(i) for i in range(0, 44)]
        classifier_layer_list = [str(i) for i in range(7)]
        layer_name = ['features.'+i for i in features_layer_list] + ['classifiers.'+i for i in classifier_layer_list]
    else:
        raise Exception("Only vgg-16 is available for models which are not from BrainScore")
    
print("Please select the layer from the list: \n")
print(layer_name, "\n")

selected_layer = str(input())

assert model_name != '', 'No layer is selected'
if args.model_from_brainscore:
    activations = np.array(model(stimuli=stimuli_path_list, layers=[selected_layer]))  # (3)
    filenames = stimuli_path_list
else:
    activations, filenames = vgg16_activation(model, selected_layer, batch_size=args.batch_size, dataset_dir=args.stim_data_path)
    activations = activations.detach().cpu().numpy()
    
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

saved_file_activations = os.path.join(args.saved_path, f'activation_{model_name}_{selected_layer}.npy')
saved_file_corr = os.path.join(args.saved_path, f'corr_{model_name}_{selected_layer}.npy')
saved_file_fname = os.path.join(args.saved_path, f'filename_{model_name}_{selected_layer}.npy')
np.save(saved_file_activations , activations) # Save activations of a layer
np.save(saved_file_corr , corr_mat) # Save the correlatio of the activations
np.save(saved_file_fname, np.array(filenames)) # Save corresponding image names
print("Done")