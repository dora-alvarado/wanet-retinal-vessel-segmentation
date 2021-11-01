def get_dataset_settings(name):
    return dict(
        DRIVE = drive_settings,
        STARE = stare_settings,
        CHASEDB = chasedb_settings
    )[name]

def drive_settings(home_dir):
    settings = {}
    settings['path_dataset'] = home_dir + 'DRIVE/'
    settings['path_train_imgs'] = settings['path_dataset'] + 'training/images'
    settings['path_test_imgs'] = settings['path_dataset'] + 'test/images'
    settings['path_train_gts'] = settings['path_dataset'] + 'training/1st_manual'
    settings['path_test_gts'] = settings['path_dataset'] + 'test/1st_manual'
    settings['path_train_fovs'] = settings['path_dataset'] + 'training/mask'
    settings['path_test_fovs'] = settings['path_dataset'] + 'test/mask'
    settings['image_ext'] = '.tif'
    settings['gt_ext'] = '_manual1.gif'
    settings['fov_ext'] = '_mask.gif'
    return settings


def stare_settings(home_dir):
    settings = {}
    settings['path_dataset'] = home_dir + 'STARE/'
    settings['path_train_imgs'] = settings['path_dataset'] + 'training/images'
    settings['path_test_imgs'] = settings['path_dataset'] + 'test/images'
    settings['path_train_gts'] = settings['path_dataset'] + 'training/labels-ah'
    settings['path_test_gts'] = settings['path_dataset'] + 'test/labels-ah'
    settings['path_train_fovs'] = settings['path_dataset'] + 'training/mask'
    settings['path_test_fovs'] = settings['path_dataset'] + 'test/mask'
    settings['image_ext'] = '.ppm'
    settings['gt_ext'] = '.ah.ppm'
    settings['fov_ext'] = '.png'
    return settings


def chasedb_settings(home_dir):
    settings = {}
    settings['path_dataset'] = home_dir + 'CHASEDB/'
    settings['path_train_imgs'] = settings['path_dataset'] + 'training/images'
    settings['path_test_imgs'] = settings['path_dataset'] + 'test/images'
    settings['path_train_gts'] = settings['path_dataset'] + 'training/manual'
    settings['path_test_gts'] = settings['path_dataset'] + 'test/manual'
    settings['path_train_fovs'] = settings['path_dataset'] + 'training/mask'
    settings['path_test_fovs'] = settings['path_dataset'] + 'test/mask'
    settings['image_ext'] = '.jpg'
    settings['gt_ext'] = '_1stHO.png'
    settings['fov_ext'] = '.png'
    return settings

