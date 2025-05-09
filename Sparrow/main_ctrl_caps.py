import os
import shutil
import numpy as np
import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import vis_caps
import argparse
import re
import os
import sys
# Adding the utils to the path to be found

sys.path.append(os.environ['PACKAGE_PATH'])

from joint.data.datasets import create_datasets
from joint.settings.ProtoPNet_settings_MMarvel import get_args, get_optimizer_coeffs
from helpers import makedir
import model_ctrl_caps
import train_and_test_ctrl_caps as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

args = get_args().parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

# Extract the optimizer coefficients
optimizer_stats = get_optimizer_coeffs()
coefs = optimizer_stats['coefs']
joint_optimizer_lrs = optimizer_stats['joint_optimizer_lrs']
warm_optimizer_lrs = optimizer_stats['warm_optimizer_lrs']
joint_lr_step_size = optimizer_stats['joint_lr_step_size']  

# Parser the arguments
base_architecture = args.base_architecture
img_size = args.img_size
num_classes = args.num_classes
prototype_activation_function = args.prototype_activation_function
add_on_layers_type = args.add_on_layers_type
experiment_run = args.experiment_run
last_layer_type = args.last_layer_type

train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size
train_push_batch_size = args.train_push_batch_size
model_path = args.model_dir

spstr = 'hardsparse' if args.hard_sparse else 'softsparse'
capstr = 'cap' if args.use_cap else 'nocap'
lstr = 'l2' if args.ltwo else 'hs'
prototype_shape = (args.num_prototypes, args.proto_depth, 1, 1)      

# (Num prototypes, Proto depth,1,1) 
# if add_on_layers_type == 'none': 
#     prototype_shape = (2000, 512, 1, 1)
#     #prototype_shape = (1960, 512, 1, 1)
    
# else: 
#     prototype_shape = (2000, 128, 1, 1)
#     #prototype_shape = (1960, 128, 1, 1)

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
model_dir = args.model_dir + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
# shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
# shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
# shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
# shutil.copy(src=os.path.join(os.getcwd(), 'model_ctrl_caps.py'), dst=model_dir)
# shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test_ctrl_caps.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
log('base architecture: ' + base_architecture)
log('experiment run: ' + experiment_run)
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'
seed = np.random.randint(10, 10000, size=1)[0]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
print('current seed is:', seed)
normalize = transforms.Normalize(mean=mean,
                                 std=std)

# all datasets
train_dataset, test_dataset, caps_push_dataset = create_datasets()


# train set
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=2, pin_memory=False)
# push set
caps_push_loader = torch.utils.data.DataLoader(
    caps_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=2, pin_memory=False)
# test set
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=2, pin_memory=False)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(caps_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

from settings import clst_k, cap_width
print('train with top-k cluster with k:', clst_k)
print('radius initialization is:',cap_width)
# construct the model
ppnet = model_ctrl_caps.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              cap_width=cap_width,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type,
                              last_layer_type=last_layer_type)

ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# define optimizer
joint_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
    {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
    ]

joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)


warm_optimizer_specs = \
    [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
     {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
     {'params': ppnet.cap_width_l2, 'lr': warm_optimizer_lrs['cap_width']},
    ]

warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

# finetune
from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)# finetune



# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

print(torch.cuda.get_device_name())

# train the model
log('start training')
import copy

for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                    class_specific=class_specific, clst_k = clst_k, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                    class_specific=class_specific, clst_k = clst_k, coefs=coefs, log=log)
        joint_lr_scheduler.step()

    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nofinetune', accu=accu,
                                target_accu=0.0, log=log)
# visualization 
with torch.no_grad():
    vis_count = vis_caps.vis_prototypes(
            caps_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=10, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            ltwo=True,
            log=log)

# mask
ppnet_multi.module.mask(vis_count)
num = (vis_count > 0).sum()
#finetune
print('The estimated number of visualizable prototypes:')
print(num)
print()
tnt.last_only(model=ppnet_multi, log=log)
for epoch in range(5):
    log('iteration: \t{0}'.format(epoch))
    _= tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                  class_specific=class_specific, coefs=coefs, log=log)
    print('Accuracy is:')
    accu= tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_topk_' + 'finetuned', accu=accu,
                                target_accu=0.70, log=log)
   
logclose()

