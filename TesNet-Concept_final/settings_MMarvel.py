img_size = 224
prototype_shape = (2000, 64, 1, 1)#(1960, 64, 1, 1)
num_classes = 8
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = 'MMarvel'
base_architecture = 'vgg19'
data_path = '/media/wlodder/T9/Datasets/Experiments/XAI/Military_MARVEL/train_cropped_augmented/data'# where you save the data
train_dir = data_path 
test_dir = data_path 
train_push_dir = data_path 
train_batch_size = 80
test_batch_size = 100
train_push_batch_size =75
joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3,
                       'cap_width_l2': 1e-5}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3,
                      'cap_width_l2': 1e-4}

last_layer_optimizer_lr = 1e-4

k = 3
cap_width = 8.05 

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.2,# by a lot 
    'l1': 1e-4,
    'orth': 5e-3, # by 10
    'sub_sep': -5e-5, #by 10 
    'cap_coef':3e-5
}

num_train_epochs = 10
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
analysis_start = 0
