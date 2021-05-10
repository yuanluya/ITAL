import numpy as np
import scipy.io as sio
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

def get_raw_feat(img_directory, annotate_directory, label_file,
                 processor, feat_network, device, extract_format):
    image_feats = []
    labels = []
    hook = Hook(feat_network.avgpool)
    for il, line in enumerate(tqdm(label_file[0][0])):
        img_name = str(line[0])
        raw_img = Image.open(os.path.join(img_directory, img_name))
        annot_name = img_name[0: img_name.rfind('.')] + '.mat'
        annot = sio.loadmat(os.path.join(annotate_directory, annot_name))
        if raw_img.mode[0: 3] == 'RGB':
            crop_raw_img = raw_img.crop(box = [int(annot['bbox'][0][0][i]) for i in range(4)])
            img = torch.unsqueeze(processor(raw_img), 0).to(device)
            if extract_format == 'feature':
                feat_network(img)
                image_feats.append(np.squeeze(hook.output_.detach().cpu().numpy(), axis = (2, 3)))
            elif extract_format == 'image':
                image_feats.append(img.cpu().numpy())
            else:
                assert(0)
            labels.append(label_file[1][0][il] - 1)# if class_dict.get(words[1]) else -1)
    return np.concatenate(image_feats, axis = 0), np.array(labels)

def main():
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:
        dev = "cpu"  
    device = torch.device(dev)
    data_folder = 'images'
    annot_folder = 'annotations-mat'
    extract_format = 'image'#'features'
    
    resnet_indices = [34, 50, 101, 152]
    resnet_idx = resnet_indices[int(sys.argv[1])]
    resnet = eval('TV.models.resnet%d(pretrained = True, progress = True)' % resnet_idx)
    resnet.eval()
    resnet.to(device)
    normalize = T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    processor = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])

    index_file = sio.loadmat('lists/splits.mat')['splits'][0][0]
    train_image_feats, train_labels = get_raw_feat(data_folder, annot_folder, (index_file[0], index_file[1]),
                                                   processor, resnet, device, extract_format)
    test_image_feats, test_labels = get_raw_feat(data_folder, annot_folder, (index_file[2], index_file[3]),
                                                 processor, resnet, device, extract_format)
    if extract_format == 'feature':
        np.save('CUB200_train_raw_%ss_%d.npy' % (extract_format, resnet_idx), train_image_feats)
        np.save('CUB200_test_raw_%ss_%d.npy' % (extract_format, resnet_idx), test_image_feats)
    elif extract_format == 'image':
        np.save('CUB200_train_raw_%ss.npy' % extract_format, train_image_feats)
        np.save('CUB200_test_raw_%ss.npy' % extract_format, test_image_feats)
    else:
        assert(0)
    np.save('CUB200_train_labels.npy', train_labels)
    np.save('CUB200_test_labels.npy', test_labels)
    
    assert(np.max(test_labels) == np.max(train_labels))
    train_label_oh = np.zeros((train_labels.size, train_labels.max()+1))
    train_label_oh[np.arange(train_labels.size), train_labels] = 1
    test_label_oh = np.zeros((test_labels.size, test_labels.max()+1))
    test_label_oh[np.arange(test_labels.size),test_labels] = 1
    np.save('CUB200_train_labels_oh.npy', train_label_oh)
    np.save('CUB200_test_labels_oh.npy', test_label_oh)

    return

if __name__ == '__main__':
    main()