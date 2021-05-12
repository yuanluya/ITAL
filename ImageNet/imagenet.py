import sys
import bisect
import torch
import numpy as np
from easydict import EasyDict as edict

from imagenet_vgg import Hook
import torchvision as TV
import torchvision.transforms as T
from tqdm import tqdm

import pdb

class FC_Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, resnet_idx = None):
        super(FC_Net, self).__init__()
        self.dims_ = [input_dim] + hidden_dims
        self.num_classes_ = num_classes
        self.layers_ = []
        for idx in range(1, len(self.dims_)):
            self.layers_.append(torch.nn.Linear(self.dims_[idx - 1], self.dims_[idx]))
            if idx != len(self.dims_) - 1:
                self.layers_.append(torch.nn.ReLU())
                self.layers_.append(torch.nn.Dropout())
            else:
                self.layers_.append(torch.nn.Dropout(0.3))
        self.layers_.append(torch.nn.Linear(self.dims_[-1], self.num_classes_))
        self.new_fc_ = torch.nn.Sequential(*self.layers_)
        if resnet_idx:
            self.processor_ = torch.nn.Sequential(
                T.RandomApply(torch.nn.ModuleList([T.RandomHorizontalFlip()]), p = 0.3),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p = 0.3),
                T.RandomApply(torch.nn.ModuleList([T.RandomAffine(30, (0.25, 0.25), (0.8, 1.2))]), p = 0.6))
            self.resnet_ = eval('TV.models.resnet%d(pretrained = True, progress = True)' % resnet_idx)
            self.resnet_params_ = list(self.resnet_.layer4.parameters())
            self.resnet_.fc = self.new_fc_
            self.network_ = torch.nn.Sequential(self.processor_, self.resnet_)
        else:
            self.processor_ = None
            self.network_ = self.new_fc_
    
    def forward(self, x):
        x = self.network_(x)
        return x

def train(model, config_train, data_package, device, model_ckpt_path):
    if config_train.get('resnet_lr'):
        opt = torch.optim.SGD([{'params': model.new_fc_.parameters()}, {'params': model.resnet_params_, 'lr': config_train.resnet_lr}],
                                lr = config_train.lr, momentum = config_train.momentum, weight_decay = 1e-3)
    else:
        opt = torch.optim.SGD(model.parameters(), lr = config_train.lr,
                              momentum = config_train.momentum, weight_decay = 1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: 0.5 ** (epoch // config_train.decay_epoch))
    
    for te in tqdm(range(config_train.train_epoch)):
        # train
        model.train()
        losses = []
        accuracies = []
        start_idx = 0
        indices = np.arange(data_package.train_label.shape[0])
        np.random.shuffle(indices)
        while start_idx < data_package.train_data.shape[0]:
            opt.zero_grad()
            batch_x = torch.tensor(data_package.train_data[indices[start_idx: start_idx + config_train.batch_size], ...],
                                   dtype = torch.float32).to(device)
            batch_y = torch.tensor(data_package.train_label\
                [indices[start_idx: start_idx + config_train.batch_size]]).type(torch.int64).to(device)
            logits = model(batch_x)
            pred = torch.argmax(logits, axis = 1)
            acc = torch.mean(torch.tensor(pred == batch_y).type(torch.float32))
            
            loss_func = torch.nn.CrossEntropyLoss()
            loss = loss_func(logits, batch_y)
            loss.backward()
            opt.step()

            losses.append(loss.detach().cpu().numpy())
            accuracies.append(acc.cpu().numpy())
            start_idx += config_train.batch_size
        
        if te % 50 == 0:
            # validate
            test_acc_1, test_acc_3, test_acc_5 = test(model, config_train, data_package, device)
            print('[%d/%d] loss: %f, train-acc: %f, test-acc-1: %f, test-acc-3: %f, test-acc-5: %f' %\
                (te + 1, config_train.train_epoch, np.mean(losses), np.mean(accuracies), test_acc_1, test_acc_3, test_acc_5))
        scheduler.step()
        if (te + 1) % 50000 == 0:
            torch.save(model.state_dict(), model_ckpt_path)

def test(model, config_test, data_package, device):
    model.eval()
    test_start_idx = 0
    prediction = []
    while test_start_idx < data_package.test_data.shape[0]:
        batch_x = torch.tensor(data_package.test_data[test_start_idx: test_start_idx + config_test.batch_size, ...],
                               dtype=torch.float32).to(device)
        if config_test.get('resnet_lr'):
            logits = model.resnet_(batch_x)
        else:
            logits = model(batch_x)
        _, pred = torch.topk(logits, 5)
        prediction.append(pred)
        test_start_idx += config_test.batch_size
    prediction = torch.squeeze(torch.cat(prediction, dim = 0))
    test_acc_1 = torch.mean(torch.sum(torch.tensor(prediction[:, 0: 1].cpu() ==\
                    torch.unsqueeze(torch.tensor(data_package.test_label).type(torch.int64), 1)).type(torch.float32), 1))
    test_acc_3 = torch.mean(torch.sum(torch.tensor(prediction[:, 0: 3].cpu() ==\
                    torch.unsqueeze(torch.tensor(data_package.test_label).type(torch.int64), 1)).type(torch.float32), 1))
    test_acc_5 = torch.mean(torch.sum(torch.tensor(prediction[:, 0: 5].cpu() ==\
                    torch.unsqueeze(torch.tensor(data_package.test_label).type(torch.int64), 1)).type(torch.float32), 1))
    return test_acc_1, test_acc_3, test_acc_5

def extract_feat(model, raw_data, device, feature_layer_idx = -2):
    model.eval()
    start_idx = 0
    batch_size = 64
    features = []
    if model.processor_:
        hook = Hook(model.network.resnet_.fc[feature_layer_idx])
    else:
        hook = Hook(model.network_[feature_layer_idx])
    while start_idx < raw_data.shape[0]:
        batch_x = torch.tensor(raw_data[start_idx: start_idx + batch_size, :]).to(device)
        if model.processor_:
            model.resnet_(batch_x)
        else:
            model(batch_x)
        features.append(hook.output_.detach().cpu().numpy())
        start_idx += batch_size

    return np.concatenate(features, axis = 0)

def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:
        dev = "cpu"  
    device = torch.device(dev)
    extract_format = 'image'

    # experiment setup
    resnet_indices = [34, 50, 101, 152]
    resnet_idx = resnet_indices[int(sys.argv[1])]
    data_package = edict({})
    num_classes = 200
    feature_extraction = (len(sys.argv) == 3)

    # read in raw features
    data_package.train_label = np.load('ImageNet_train_labels.npy')
    class_idx = bisect.bisect_left(data_package.train_label, num_classes) - 1
    data_package.train_label = data_package.train_label[0: class_idx + 1]
    if extract_format == 'feature':
        data_package.train_data = np.load('ImageNet_train_raw_%ss_%d.npy' % (extract_format, resnet_idx))[0: class_idx + 1, ...]
        data_package.test_data = np.load('ImageNet_test_raw_%ss_%d.npy' % (extract_format, resnet_idx))
    elif extract_format == 'image':
        data_package.train_data = np.load('ImageNet_train_raw_%ss.npy' % extract_format)[0: class_idx + 1, ...]
        data_package.test_data = np.load('ImageNet_test_raw_%ss.npy' % extract_format)
    data_package.test_label = np.load('ImageNet_test_labels.npy')
    test_indices = np.nonzero(data_package.test_label < num_classes)[0]
    data_package.test_data = data_package.test_data[test_indices, ...]
    data_package.test_label = data_package.test_label[test_indices]

    # initialize network
    raw_feat_dim = 2048
    hidden_dims = [500, 200, 10]
    model_ckpt_path = 'CKPT_%d_%s' % (resnet_idx, '_'.join([str(hd) for hd in hidden_dims]))
    fc_net = FC_Net(raw_feat_dim, hidden_dims, num_classes,
                    resnet_idx if extract_format == 'image' else None)
    try:
        fc_net.load_state_dict(torch.load(model_ckpt_path, map_location = dev))
        print('Load model from %s' % model_ckpt_path)
    except:
        print('Train %s model from scratch' % model_ckpt_path)
    fc_net.to(device)
    print(next(fc_net.parameters()).is_cuda)
    
    # config training
    config_train = edict({})
    config_train.decay_epoch = 2000
    config_train.train_epoch = 1 * config_train.decay_epoch
    config_train.batch_size = 32
    config_train.lr = 1e-3
    if extract_format == 'image':
        config_train.resnet_lr = config_train.lr / 50
    config_train.momentum = 0.9

    if not feature_extraction:
        train(fc_net, config_train, data_package, device, model_ckpt_path)
    else:
        ta1, ta3, ta5 = test(fc_net, config_train, data_package, device)
        print('test-acc-1: %f, test-acc-3: %f, test-acc-5: %f' % (ta1, ta3, ta5))
        train_feat = extract_feat(fc_net, data_package.train_data, device)
        test_feat = extract_feat(fc_net, data_package.test_data, device)
        W, b = list(fc_net.network_[-1].parameters())
        W = W.detach().cpu().numpy()
        b = b.detach().cpu().numpy()
        W = np.concatenate([W, np.expand_dims(b, 1)], axis = 1)
        np.save('ImageNet_train_features%d.npy' % resnet_idx, train_feat)
        np.save('ImageNet_test_features%d.npy' % resnet_idx, test_feat)
        np.save('ImageNet_gt_weights%d.npy' % resnet_idx, W)

    return

if __name__ == '__main__':
    main()
