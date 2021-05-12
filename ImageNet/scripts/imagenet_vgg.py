import numpy as np
import sys
import os
from PIL import Image
from tqdm import tqdm

import torch
import torchvision as TV
import torchvision.transforms as T

import pdb

class Hook():
    def __init__(self, module, backward = False):
        if not backward:
            self.hook_ = module.register_forward_hook(self.hook_fn)
        else:
            self.hook_ = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input_ = input
        self.output_ = output

    def close(self):
        self.hook_.remove()

def get_train(directory, processor, feat_network, feature_layer_idx, device, extract_format):
    folder_names = os.listdir(directory)
    folder_names = [fn for fn in folder_names if fn[0] == 'n']
    folder_names.sort()
    image_feats = []
    labels = []
    invalid_dirs = []
    class_dict = {}
    hook = Hook(feat_network.classifier[feature_layer_idx])
    for ifn, fn in enumerate(tqdm(folder_names)):
        class_dict[fn] = ifn
        images_dir = os.path.join(directory, fn, 'images')
        image_names = os.listdir(images_dir)
        image_names = [image for image in image_names if image[-4:] == 'JPEG']
        image_names.sort()
        for image in tqdm(image_names):
            raw_img = Image.open(os.path.join(images_dir, image))
            if raw_img.mode[0: 3] != 'RGB':
                invalid_dirs.append(os.path.join(images_dir, image))
            else:
                img = torch.unsqueeze(processor(raw_img), 0).to(device)
                if extract_format == 'features':
                    feat_network(img)
                    image_feats.append(hook.output_.detach().cpu().numpy())
                elif extract_format == 'image':
                    image_feats.append(img.cpu().numpy().astype('int8'))
                labels.append(ifn)
    
    return np.concatenate(image_feats, axis = 0), np.array(labels), class_dict, invalid_dirs

def get_test(directory, label_file, class_dict, processor, feat_network, feature_layer_idx, device, extract_format):
    label_file = open(label_file, 'r')
    lines = label_file.readlines()
    image_feats = []
    labels = []
    invalid_dirs = []
    hook = Hook(feat_network.classifier[feature_layer_idx])
    for line in tqdm(lines):
        words = line.split()
        raw_img = Image.open(os.path.join(directory, 'images', words[0]))
        if raw_img.mode[0: 3] != 'RGB':
            invalid_dirs.append(os.path.join(directory, words[0]))
        else:
            img = torch.unsqueeze(processor(raw_img), 0).to(device)
            if extract_format == 'features':
                feat_network(img)
                image_feats.append(hook.output_.detach().cpu().numpy())
            elif extract_format == 'image':
                image_feats.append(img.cpu().numpy().astype('int8'))
            labels.append(class_dict[words[1]])# if class_dict.get(words[1]) else -1)

    return np.concatenate(image_feats, axis = 0), np.array(labels), invalid_dirs

def main():
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:
        dev = "cpu"  
    device = torch.device(dev)
    data_folder = '../tiny-imagenet-200'
    
    extract_format = 'image'#"features"
    vgg_indices = [11, 13, 16, 19]
    vgg_idx = vgg_indices[int(sys.argv[1])]
    vgg = eval('TV.models.vgg%d_bn(pretrained = True, progress = True)' % vgg_idx)
    feature_layer_idx = 3
    vgg.classifier[feature_layer_idx + 1].inplace = False # to get unrelued feature
    vgg.eval()
    vgg.to(device)
    normalize = T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    processor = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])

    train_image_feats, train_labels, class_names, _ =\
        get_train(os.path.join(data_folder, 'train'), processor, vgg, feature_layer_idx, device, extract_format)
    test_image_feats, test_labels, _ =\
        get_test(os.path.join(data_folder, 'val'),\
                 os.path.join(data_folder, 'val', 'val_annotations.txt'),\
                 class_names, processor, vgg, feature_layer_idx, device, extract_format)
    if extract_format == 'features':
        np.save('../ImageNet_train_raw_features_%d.npy' % vgg_idx, train_image_feats)
        np.save('../ImageNet_test_raw_features_%d.npy' % vgg_idx, test_image_feats)
    elif extract_format == 'image':
        np.save('../ImageNet_train_raw_images.npy', train_image_feats)
        np.save('../ImageNet_test_raw_images.npy', test_image_feats)
    np.save('../ImageNet_train_labels.npy', train_labels)
    np.save('../ImageNet_test_labels.npy', test_labels)
    return

if __name__ == '__main__':
    main()