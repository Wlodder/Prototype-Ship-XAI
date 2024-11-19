import os
from PIL import Image
import shutil
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import argparse
import re
from util.helpers import makedir
import push, model_cap, train_and_test_cap as tnt
import os
import sys
# Adding the utils to the path to be found

sys.path.append(os.environ['PACKAGE_PATH'])

from joint.data.datasets import create_datasets
from joint.settings.TesNet_settings_MMarvel import get_args, get_optimizer_coeffs
from util import save
from util.log import create_logger
from util.preprocess import mean, std, preprocess_input_function
from MMarvel_Dataset import MARVEL_MILITARY
import settings_CUB
import settings_MMarvel 
import vis_caps

# parser = argparse.ArgumentParser()
# parser.add_argument('-gpuid',type=str, default='0')
# parser.add_argument('-dataset',type=str,default="CUB")
args = get_args().parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
print(os.environ['CUDA_VISIBLE_DEVICES'])

# Base parameters
num_classes = args.num_classes
img_size = args.img_size
add_on_layers_type = args.add_on_layers_type
prototype_activation_function = args.prototype_activation_function
base_architecture = args.base_architecture

#datasets
train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size
train_push_batch_size = args.train_push_batch_size

#optimzer
optimizer_options = get_optimizer_coeffs()
joint_optimizer_lrs = optimizer_options['joint_optimizer_lrs']
joint_lr_step_size = args.joint_lr_step_size
warm_optimizer_lrs = optimizer_options['warm_optimizer_lrs']
last_layer_optimizer_lr = optimizer_options['last_layer_optimizer_lr']
# weighting of different training losses

coefs = optimizer_options['coefs']

# number of training epochs, number of warm epochs, push start epoch, push epochs
num_train_epochs = args.num_train_epochs
num_warm_epochs = args.num_warm_epochs
push_start = args.push_start
push_epochs = range(int(push_start), int(num_train_epochs))
# input for caps
analysis_start = args.analysis_start
cap_width = args.cap_width
k = args.k
model_dir = args.model_dir
prototype_shape=(args.num_prototypes,args.proto_depth,1,1)

# #setting parameter
# dataset_name = args.dataset
# # load the hyper param
# if dataset_name == "CUB":
#     #model param
#     num_classes = settings_CUB.num_classes
#     img_size = settings_CUB.img_size
#     add_on_layers_type = settings_CUB.add_on_layers_type
#     prototype_shape = settings_CUB.prototype_shape
#     prototype_activation_function = settings_CUB.prototype_activation_function
#     base_architecture = settings_CUB.base_architecture
#     #datasets
#     train_dir = settings_CUB.train_dir
#     test_dir = settings_CUB.test_dir
#     train_push_dir = settings_CUB.train_push_dir
#     train_batch_size = settings_CUB.train_batch_size
#     test_batch_size = settings_CUB.test_batch_size
#     train_push_batch_size = settings_CUB.train_push_batch_size
#     #optimzer
#     joint_optimizer_lrs = settings_CUB.joint_optimizer_lrs
#     joint_lr_step_size = settings_CUB.joint_lr_step_size
#     warm_optimizer_lrs = settings_CUB.warm_optimizer_lrs
#     last_layer_optimizer_lr = settings_CUB.last_layer_optimizer_lr
#     # weighting of different training losses
#     coefs = settings_CUB.coefs
#     # number of training epochs, number of warm epochs, push start epoch, push epochs
#     num_train_epochs = settings_CUB.num_train_epochs
#     num_warm_epochs = settings_CUB.num_warm_epochs
#     push_start = settings_CUB.push_start
#     push_epochs = settings_CUB.push_epochs
#     # input for caps
#     analysis_start = settings_CUB.analysis_start
#     cap_width = settings_CUB.cap_width
#     k = settings_CUB.k
# elif dataset_name == 'MMarvel': 
#     num_classes = settings_MMarvel.num_classes
#     img_size = settings_MMarvel.img_size
#     add_on_layers_type = settings_MMarvel.add_on_layers_type
#     prototype_shape = settings_MMarvel.prototype_shape
#     prototype_activation_function = settings_MMarvel.prototype_activation_function
#     base_architecture = settings_MMarvel.base_architecture
#     #datasets
#     train_dir = settings_MMarvel.train_dir
#     test_dir = settings_MMarvel.test_dir
#     train_push_dir = settings_MMarvel.train_push_dir
#     train_batch_size = settings_MMarvel.train_batch_size
#     test_batch_size = settings_MMarvel.test_batch_size
#     train_push_batch_size = settings_MMarvel.train_push_batch_size
#     #optimzer
#     joint_optimizer_lrs = settings_MMarvel.joint_optimizer_lrs
#     joint_lr_step_size = settings_MMarvel.joint_lr_step_size
#     warm_optimizer_lrs = settings_MMarvel.warm_optimizer_lrs
#     last_layer_optimizer_lr = settings_MMarvel.last_layer_optimizer_lr
#     # weighting of different training losses
#     coefs = settings_MMarvel.coefs
#     # number of training epochs, number of warm epochs, push start epoch, push epochs
#     num_train_epochs = settings_MMarvel.num_train_epochs
#     num_warm_epochs = settings_MMarvel.num_warm_epochs
#     push_start = settings_MMarvel.push_start
#     push_epochs = settings_MMarvel.push_epochs
#     # input for caps
#     analysis_start = settings_MMarvel.analysis_start
#     cap_width = settings_MMarvel.cap_width
#     k = settings_MMarvel.k
# else:
#     raise Exception("there are no settings file of datasets {}".format(dataset_name))
base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
model_dir = model_dir + base_architecture + '/'

makedir(model_dir)

logname = f'train_{base_architecture}.log'
log, logclose = create_logger(log_filename=os.path.join(model_dir, logname))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'


print('Use topk clster with k:',k)
print('Increase weight for ss, ortho by 10 and sep loss to be',coefs['sep'])
print('The chosen cap_width is:',cap_width)
print('The cap coef is:', coefs['cap_coef'])

# all datasets
# train set

# Create datasets
train_dataset, test_dataset, train_push_dataset = create_datasets()

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=2, pin_memory=False)

train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=2, pin_memory=False)
# test set
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=2, pin_memory=False)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
print('training set size: {0}'.format(len(train_loader.dataset)))
print('push set size: {0}'.format(len(train_push_loader.dataset)))
print('test set size: {0}'.format(len(test_loader.dataset)))
print('batch size: {0}'.format(train_batch_size))

print("backbone architecture:{}".format(base_architecture))
print("basis concept size:{}".format(prototype_shape))
# construct the model
ppnet = model_cap.construct_TesNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type,
                              cap_width = cap_width)
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)


class_specific = True

# define optimizer
from settings_CUB import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
  {'params': ppnet.cap_width_l2, 'lr': joint_optimizer_lrs['cap_width_l2']}
]

joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from settings_CUB import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
 {'params': ppnet.cap_width_l2, 'lr': warm_optimizer_lrs['cap_width_l2']}
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings_CUB import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

#best acc
best_acc = 0
best_epoch = 0
best_time = 0
# train the model
log('start training')
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))
    #stage 1: Embedding space learning
    #train
    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _,train_results = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                    class_specific=class_specific, coefs=coefs, log=log, topk = k)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        _,train_results = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                    class_specific=class_specific, coefs=coefs, log=log, topk = k)
        joint_lr_scheduler.step()

    #test
    accu,test_results = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)

    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                    target_accu=0.70, log=log)
 
log('Start pruning and finetuning')
# vis 
vis_count = vis_caps.vis_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=None,
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True)
# mask out the non-visualized prototype
ppnet_multi.module.prune(vis_count)
num = (vis_count > 0).sum()
print('The estimated number of visualizable prototypes:')
print(num)
print()
# finetune 
tnt.last_only(model=ppnet_multi, log=log)
for i in range(15):
    log('iteration: \t{0}'.format(i))
    _,train_results= tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                  class_specific=class_specific, coefs=coefs, log=log)

    accu,test_results = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(i) + '_' + str(i) + 'finetuned', accu=accu,
                                target_accu=0.70, log=log)
       
logclose()

