import torchvision.transforms.functional
import torchvision.transforms.functional_tensor
from pipnet.pipnet import PIPNet, get_network
from util.log import Log
import torch.nn as nn
from util.args import get_args, save_args, get_optimizer_nn
from util.data import get_dataloaders
from util.func import init_weights_xavier
from pipnet.train import train_pipnet
from pipnet.test import get_image_clusters
from pipnet.test import eval_pipnet, get_thresholds, eval_ood
import torch
from util.vis_pipnet import visualize, visualize_topk
from util.visualize_prediction import vis_pred, vis_pred_experiments
import sys, os
import random
import numpy as np
from shutil import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader 
import torchvision
from copy import deepcopy
from sklearn.cluster import KMeans

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):

    def __init__(self, path, args):
        super(ImageFolderWithPaths, self).__init__(path)
        img_size = args.image_size
        shape = (3, img_size, img_size)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        normalize = torchvision.transforms.Normalize(mean=mean,std=std)
        self.transform_no_augment = torchvision.transforms.Compose([
                                torchvision.transforms.Resize(size=(img_size, img_size)),
                                torchvision.transforms.ToTensor(),
                                normalize
                            ])

    def __getitem__(self, index):
  
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)

        img = self.transform_no_augment(img)
        path = self.imgs[index][0]
        
        return (img, label ,path)

def run_pipnet(args=None):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args = args or get_args()
    assert args.batch_size > 1

    # Create a logger
    log = Log(args.log_dir)
    print("Log dir: ", args.log_dir)
    # Log the run arguments
    save_args(args, log.metadata_dir)
    
    gpu_list = args.gpu_ids.split(',')
    device_ids = []
    if args.gpu_ids!='':
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))
    
    global device
    if not args.disable_cuda and torch.cuda.is_available():
        if len(device_ids)==1:
            device = torch.device('cuda:{}'.format(args.gpu_ids))
        elif len(device_ids)==0:
            device = torch.device('cuda')
            print("CUDA device set without id specification")
            device_ids.append(torch.cuda.current_device())
        else:
            print("This code should work with multiple GPU's but we didn't test that, so we recommend to use only 1 GPU.")
            device_str = ''
            for d in device_ids:
                device_str+=str(d)
                device_str+=","
            device = torch.device('cuda:'+str(device_ids[0]))
    else:
        device = torch.device('cpu')
     
    # Log which device was actually used
    print("Device used: ", device, "with id", device_ids)
    
    # Obtain the dataset and dataloaders
    trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(args, device)
    if len(classes)<=20:
        if args.validation_size == 0.:
            print("Classes: ", testloader.dataset.class_to_idx)
        else:
            print("Classes: ", str(classes))
    
    # Create a convolutional network based on arguments and add 1x1 conv layer
    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(len(classes), args)
   
    # Create a PIP-Net
    net = PIPNet(num_classes=len(classes),
                    num_prototypes=num_prototypes,
                    feature_net = feature_net,
                    args = args,
                    add_on_layers = add_on_layers,
                    pool_layer = pool_layer,
                    classification_layer = classification_layer
                    )
    net = net.to(device=device)
    net = nn.DataParallel(net, device_ids = device_ids)    
    
    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args)   

    # Initialize or load model
    with torch.no_grad():
        if args.state_dict_dir_net != '':
            epoch = 0
            checkpoint = torch.load(args.state_dict_dir_net,map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'],strict=True) 
            print("Pretrained network loaded")
            net.module._multiplier.requires_grad = False
            try:
                optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict']) 
            except:
                pass
            if torch.mean(net.module._classification.weight).item() > 1.0 and torch.mean(net.module._classification.weight).item() < 3.0 and torch.count_nonzero(torch.relu(net.module._classification.weight-1e-5)).float().item() > 0.8*(num_prototypes*len(classes)): #assume that the linear classification layer is not yet trained (e.g. when loading a pretrained backbone only)
                print("We assume that the classification layer is not yet trained. We re-initialize it...")
                torch.nn.init.normal_(net.module._classification.weight, mean=1.0,std=0.1) 
                torch.nn.init.constant_(net.module._multiplier, val=2.)
                print("Classification layer initialized with mean", torch.mean(net.module._classification.weight).item())
                if args.bias:
                    torch.nn.init.constant_(net.module._classification.bias, val=0.)
            # else: #uncomment these lines if you want to load the optimizer too
            #     if 'optimizer_classifier_state_dict' in checkpoint.keys():
            #         optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])
            
        else:
            net.module._add_on.apply(init_weights_xavier)
            torch.nn.init.normal_(net.module._classification.weight, mean=1.0,std=0.1) 
            if args.bias:
                torch.nn.init.constant_(net.module._classification.bias, val=0.)
            torch.nn.init.constant_(net.module._multiplier, val=2.)
            net.module._multiplier.requires_grad = False

            print("Classification layer initialized with mean", torch.mean(net.module._classification.weight).item())

    train_with_path = ImageFolderWithPaths(os.environ['TEST_JANES_MARVEL_PATH'], args)

    with torch.no_grad():
        net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)) 
        classification_weights = net.module._classification.weight
        relevant_prototypes_per_class = {}
        for cl in range(classification_weights.size(0)):
            relevant_prototypes_per_class[cl] = torch.nonzero(classification_weights[cl])

        
        
        # print(relevant_prototypes_per_class)
        for i in range(classification_weights.size(1)):
            contains_classes = []
            for j in range(classification_weights.size(0)):
                if i in relevant_prototypes_per_class[j]:
                    contains_classes.append(j)
            if contains_classes != []:
                print(i, contains_classes)
    info = get_image_clusters(net, train_with_path,0 , device)
    xs = []
    ys = list(info.keys())
    for a in info.keys():
        print(a, info[a])
        xs.append([info[a]])

    kmeans = KMeans(5)
    data = np.concatenate(xs)
    kmeans.fit(data)

    sorted = {}
    for x, y, c in zip(xs, ys, kmeans.labels_):
        if not c in sorted.keys():
            sorted[c] = []
        
        sorted[c].append(y)

    for key in sorted.keys():
        for member in sorted[key]:
            print(key,member)

    
    
    # visualize(net, projectloader, len(classes), device, 'visualised_prototypes', args)
    # testset_img0_path = test_projectloader.dataset.samples[0][0]
    # test_path = os.path.split(os.path.split(testset_img0_path)[0])[0]
    # vis_pred(net, test_path, classes, device, args) 
    # if args.extra_test_image_folder != '':
    #     if os.path.exists(args.extra_test_image_folder):   
    #         vis_pred_experiments(net, args.extra_test_image_folder, classes, device, args)


    print("Done!")

if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print_dir = os.path.join(args.log_dir,'out.txt')
    tqdm_dir = os.path.join(args.log_dir,'tqdm.txt')
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = open(print_dir, 'w')
    sys.stderr = open(tqdm_dir, 'w')
    run_pipnet(args)
    
    sys.stdout.close()
    sys.stderr.close()
