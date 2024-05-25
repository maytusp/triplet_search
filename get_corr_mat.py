from sklearn.decomposition import PCA
import os
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot
import argparse
import torch
from get_activation import model_activation
import vonenet
def print_log(printlog, logger_output):
    logger_output.write(printlog + '\n')
    logger_output.flush()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--stim_data_path', type=str, nargs='?', default="")
    parser.add_argument('--filename_order_path', type=str, nargs='?', default="")
    parser.add_argument('--model_name', type=str, default='vgg-16', choices=['CORnet-S', 'vgg-16', 'vgg-19', 'voneresnet-50', 'alexnet'])
    parser.add_argument('--selected_layer', type=str)
    parser.add_argument('--saved_path', type=str, nargs='?', default='corr_mat')
    parser.add_argument('--use_pca', default=False, action='store_true')
    parser.add_argument('--batch_size', default=32)
    args = parser.parse_args()
    logger_output = open(os.path.join(args.saved_path, f"output_{args.selected_layer}.txt"), 'a')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Configuring Pytorch
    if not os.path.exists(args.saved_path):
        # Create a new directory because it does not exist
        os.makedirs(args.saved_path)
        print("The new directory is created!")
    if args.filename_order_path:
        filename_order = np.load(args.filename_order_path) # checking if filenname list is aligned with the previous runs

    # The library "candidate_models" (part of brainscore) has many models including VGGs, ResNets, AlexNet, etc.
    # We can load a model and select its layer from candidates model
    # The list of models can be found in the link below
    # https://github.com/brain-score/candidate_models/blob/master/candidate_models/model_commitments/model_layer_def.py
    model_name_list = ['CORnet-S', 'vgg-16', 'vgg-19', 'voneresnet-50', 'alexnet'] # This is for example, you can pick any models in the link above
    model_name = args.model_name # str(input()) # Input
    print_log(f"select {model_name}", logger_output)


    if model_name == 'vgg-16':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=False)
        model.load_state_dict(torch.load('../../models/checkpoints/vgg16_bn-6c64b313.pth'))
        features_layer_list = [str(i) for i in range(0, 44)]
        classifier_layer_list = [str(i) for i in range(7)]
        layer_name = ['features.'+i for i in features_layer_list] + ['classifiers.'+i for i in classifier_layer_list]
    elif model_name =="voneresnet-50":
        layer_name =  ['vone_block'] + \
                ['model.layer1.0', 'model.layer1.1', 'model.layer1.2'] + \
                ['model.layer2.0', 'model.layer2.1', 'model.layer2.2', 'model.layer2.3'] + \
                ['model.layer3.0', 'model.layer3.1', 'model.layer3.2', 'model.layer3.3', \
                'model.layer3.4', 'model.layer3.5'] + \
                ['model.layer4.0', 'model.layer4.1', 'model.layer4.2'] + \
                ['model.avgpool']
        model = vonenet.get_model(model_arch='resnet50', pretrained=True).module
    elif model_name == "CORnet-S":
        from cornet_s import CORnet_S
        model = CORnet_S() # Build CORnet-S model
        ckpt = torch.load("./checkpoints/cornet_s-1d3f7974.pth")
        loaded_state_dict = {}
        for k,v in ckpt['state_dict'].items():
            if "module." in k:
                k_new = k.replace("module.", "")
            else:
                k_new = k
            loaded_state_dict[k_new] = v
            
        model.load_state_dict(loaded_state_dict)
        print_log("CORnet-S LOADED", logger_output)
        layer_name = ["V1", "V2", "V4", "IT"]
    else:
        raise Exception("Models are not supported in the default code. Please add your model manually.")
        
    # print("Please select the layer from the list: \n")
    # print(layer_name, "\n")
    print(model)
    model.to(device)
    model.eval()
    print_log(f"use {device}", logger_output)
    selected_layer = args.selected_layer # str(input())
    print_log(f"Start Collect Activation", logger_output)
    # assert selected_layer != '', 'No layer is selected'
    
    activations, filenames = model_activation(model, model_name, selected_layer, batch_size=args.batch_size, dataset_dir=args.stim_data_path, device=device)


        

    activations = activations.detach().cpu().numpy()
    print_log(f"Done Collect Activation", logger_output)
    act_dims = activations.shape[1]

    saved_file_activations = os.path.join(args.saved_path, f'activation_{model_name}_{selected_layer}.npy')
    saved_file_corr = os.path.join(args.saved_path, f'corr_{model_name}_{selected_layer}.npy')
    saved_file_fn = os.path.join(args.saved_path, f'filenames_{model_name}_{selected_layer}.npy')
    # np.save(saved_file_activations , activations) # Save activations of a layer

    
    if args.filename_order_path:
        '''
        Check if the filenames we use to extract responses match the previous filenames (the reference) in the same order
        '''
        error=False
        for idx, fn in enumerate(filenames):
            if filename_order[idx] != fn:
                print(f"new fn {fn} is not aligned with the exisiting {filename_order[idx]}")
                error=True

        if error:
            raise Exception("Filenames mismatch")
        else:
            print("No Mismatch")

    # Compte Correlation Matrix
    if args.use_pca:
        pca = PCA(n_components=0.95)
        pca.fit(activations)
        reduced_act = pca.transform(activations)
        print('Reduce dimensions from ', act_dims, ' to ', reduced_act.shape[1])
        corr_mat = np.corrcoef(reduced_act)
    else:
        print_log(f"Start Compute Corr", logger_output)
        corr_mat = np.corrcoef(activations)



    np.save(saved_file_corr , corr_mat) # Save the correlatio of the activations
    print_log(f"Saved at {saved_file_corr}", logger_output)

    np.save(saved_file_fn , filenames) # Save the correlatio of the activations
    print_log(f"check filename (new) with exising filename: they have to be the same order", logger_output)
