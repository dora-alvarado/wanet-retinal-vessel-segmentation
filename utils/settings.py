import configparser
import argparse


def setup_parser():
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("-c", "--config", help="Specify a configuration file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()

    defaults = {
        'path_dataset' : './datasets/',
        'dataset'      : 'DRIVE',
        'experiment'   : './test_DRIVE/',
        'crop_size'    : 48,
        'num_patches'  : 64,
        'num_epochs'   : 1,
        'batch_size'   : 32,
        'optimizer'    : 'AdamW',
        'learning_rate': 1e-3,
        'resume'       : False,
        'best'         : True,
        'stride'       : 30,
        'num_imgs'     : 3,
        'num_group'    : 3,
    }

    if args.config:
        config = configparser.ConfigParser()
        config.read([args.config])
        data_settings = dict(config.items("DEFAULT"))
        train_settings = dict(config.items("TRAINING"))
        test_settings = dict(config.items("TEST"))
        defaults = dict(data_settings)
        defaults.update(train_settings)
        defaults.update(test_settings)
        defaults['resume'] = False if defaults['resume'] == 'False' else True
        defaults['best'] = False if defaults['best'] == 'False' else True

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[conf_parser],
        # print script description with -h/--help
        description=__doc__,
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.set_defaults(**defaults)
    # DEFAULT
    group = parser.add_argument_group('DEFAULT')
    group.add_argument('-p', '--path_dataset', help="Default path to datasets folder")
    group.add_argument('-d', '--dataset', help="Dataset to use for training/test step")
    group.add_argument('-e', '--experiment', help="Experiment name (a folder is created to save logfile, weights and results")
    group.add_argument('-s', '--crop_size', type=int, help='Patch size to crop training/test images')
    # TRAINING
    group = parser.add_argument_group('TRAINING')
    group.add_argument('-n ', '--num_patches', type=int, help='Number of patches for training step')
    group.add_argument('-i ', '--num_epochs', type=int, help='Number of epochs/iterations for training step')
    group.add_argument('-b ', '--batch_size', type=int, help='Batch size for training step')
    group.add_argument('-o ', '--optimizer', help='Optimizer for training step')
    group.add_argument('-l ', '--learning_rate', type=float,help='Learning rate for training step')
    group.add_argument('-r ', '--resume', type=bool,help='Optimizer for training step')
    # TEST
    group = parser.add_argument_group('TEST')
    group.add_argument('--best', type=bool, help='Checkpoint to use for test step, '
                                                        'if True min-loss model is used, otherwise the last model is used')
    group.add_argument('--stride', type=int, help='Prediction patch stride for test step')
    group.add_argument('--num_imgs', type=int, help='Number of images for test step')
    group.add_argument('--num_group', type=int, help='Number of images per row for visualization')

    args = parser.parse_args(remaining_argv)
    print(args)
    return args


def save_config(args, path):
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'path_dataset': args['path_dataset'],
                         'dataset': args['dataset'],
                         'experiment': args['experiment'],
                         'crop_size': args['crop_size']
                         }
    config['TRAINING'] = {'num_patches': args['num_patches'],
                          'num_epochs': args['num_epochs'],
                          'batch_size': args['batch_size'],
                          'optimizer': args['optimizer'],
                          'learning_rate': args['learning_rate'],
                          'resume': args['resume']
                          }
    config['TEST'] = {'best': args['best'],
                      'stride': args['stride'],
                      'num_imgs': args['num_imgs'],
                      'num_group': args['num_group']
                      }

    with open(path, 'w') as configfile:
        config.write(configfile)