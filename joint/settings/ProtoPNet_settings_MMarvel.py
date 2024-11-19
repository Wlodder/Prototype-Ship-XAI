import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_architecture',default='resnet152')
    parser.add_argument('--img_size',default=224, type=int)

    parser.add_argument('--num_classes' ,default= 8, type=int)
    parser.add_argument('--prototype_activation_function',default = 'log')
    parser.add_argument('--add_on_layers_type',default= 'regular')
    parser.add_argument('--temp',default= 5.0, type=float)
    parser.add_argument('--experiment_run',default= 'MMarvel')
    parser.add_argument('--hard_sparse',default=True)
    parser.add_argument('--use_cap',default=True)
    parser.add_argument('--ctrl',default=True)
    parser.add_argument('--sub_mean',default=True)
    parser.add_argument('--ltwo',default=True)
    parser.add_argument('--proto_depth',default=512, type=int)
    parser.add_argument('--num_prototypes',default=80, type=int)

    parser.add_argument('--last_layer_type',default='single')

    parser.add_argument('--cap_width',default=6.0, type=float)
    parser.add_argument('--clst_k',default=10, type=int)

    parser.add_argument('--seed',default=0, type=int)

    parser.add_argument('--sep_cost_filter',default='cutoff')
    parser.add_argument('--sep_cost_cutoff',default=-0.05, type=float)

    parser.add_argument('--train_batch_size',default=50, type=int)
    parser.add_argument('--test_batch_size',default=80, type=int)
    parser.add_argument('--train_push_batch_size',default=50,  type=int)

    parser.add_argument('--num_train_epochs',default=15, type=int)
    parser.add_argument('--num_warm_epochs',default=5, type=int)
    parser.add_argument('--push_start',default=10, type=int)
    parser.add_argument('--push_epoch_interval',default=10, type=int)
    parser.add_argument('--model_dir')
    parser.add_argument('--gpuid',default='0', type=str)



    return parser



#none: regular separation cost
#cutoff: don't consider separation cost to classes where a prototype has weight > cutoff

# spstr = 'hardsparse' if hard_sparse else 'softsparse'
# capstr = 'cap' if use_cap else 'nocap'
# lstr = 'l2' if ltwo else 'hs'


def get_optimizer_coeffs():
    joint_optimizer_lrs = {'features': 1e-4,
                        'add_on_layers': 3e-3,
                        'prototype_vectors': 3e-3, 
                        'last_layer': 1e-4}
    joint_lr_step_size = 5


    warm_optimizer_lrs = {'add_on_layers': 3e-3,
                        'prototype_vectors': 3e-3,
                        'cap_width': 5e-4}

    last_layer_optimizer_lr = 1e-4

    coefs = {
        'crs_ent': 1,
        'clst': 0.8,
        'sep': -0.08,
        'l1': 1e-4,
        'relu': 1e-3,
        'dist': 1.0,
        'cap': 0.01,
    }

    return {'joint_optimizer_lrs':joint_optimizer_lrs,
            'joint_lr_step_size':joint_lr_step_size,
            'warm_optimizer_lrs':warm_optimizer_lrs,
            'last_layer_optimizer_lr':last_layer_optimizer_lr,
            'coefs':coefs}