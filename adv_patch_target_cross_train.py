import old_models
import keras.backend as K
import os
import utils
import numpy as np
from scipy.io import loadmat
import tensorflow as tf


if __name__ == '__main__':
    K.set_image_data_format('channels_first')
    K.set_learning_phase(0) #set learning phase

    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    gpuConfig = tf.ConfigProto(allow_soft_placement=True)
    gpuConfig.gpu_options.allow_growth = True

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
    config.gpu_options.per_process_gpu_memory_fraction = 0.32
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    # data_names = ['EPFL', 'ERN', 'MI4C']
    data_names = ['EPFL']
    # model_list = ['EEGNet', 'DeepConvNet', 'ShallowConvNet']
    model_list = ['ShallowConvNet']

    a = 0.2   # noisy
    attack_type = 'target'  # 'target' or 'nontarget'
    reg = None
    patch_size = (32, 50)

    model_tr = []
    for model_used in model_list:
        for data_name in data_names:
            if data_name == 'EPFL':
                target_classes = [0, 1]
            elif data_name == 'ERN':
                target_classes = [0, 1]
            elif data_name == 'MI4C':
                target_classes = [0, 1, 2, 3]
            else:
                raise Exception('No such dataset:{}'.format(data_name))

            # Load dataset
            data_dir = 'Data/processed_data'
            train_dir = 'runs'
            checkpoint = 'checkpoint'
            model_name = 'model.h5'
            batch_size = 256
            epoches = 1000

            # Build pathes
            data_path = os.path.join(data_dir, '{}.npz'.format(data_name))

            # Load dataset
            data = np.load(data_path)
            # print(data.keys())
            x = data['x']
            y = data['y']
            s = data['s']
            # seed = 2019

            # x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.3, random_state=seed)

            # num_classes = 4
            print(x.shape,y.shape)

            tr_list = []
            for target_class in target_classes:
                raw_accs = []
                adv_accs = []
                rand_accs = []
                rand_bcas = []
                raw_bcas = []
                adv_bcas = []
                raw_trs = []
                rand_trs = []
                adv_trs = []
                datasets_wrt_s = utils.list_leave_one_subject(x, y, s, shuffle=True)
                for dataset_wrt_s in datasets_wrt_s:
                    x_train, y_train = dataset_wrt_s[0]
                    x_test, y_test = dataset_wrt_s[1]
                    subject = dataset_wrt_s[2]

                    nb_classes = len(np.unique(y_test))
                    samples = x_test.shape[3]
                    channels = x_test.shape[2]

                    print(x_train.shape, x_test.shape, nb_classes)



                    # Build pathes
                    # checkpoint_path = os.path.join('runs', '{}_{}'.format(data_name, model_used), '{}'.format(int(subject)), 'checkpoint')
                    # checkpoint_path = os.path.join('runs', '{}_{}'.format(data_name, model_used), '{}'.format(int(subject)), 'checkpoint')
                    # print(checkpoint_path)
                    # checkpoint_path = os.path.join('/mnt/disk2/zhangxiao/AdversarialExamplesOnEEG_cross_subject/old_runs',
                    #                                '{}_{}'.format(data_name, model_used), '{}'.format(int(subject)), 'checkpoint')
                    checkpoint_path = os.path.join('runs', '{}/{}'.format(data_name, model_used),
                                                   '{}'.format(int(subject)))
                    print(checkpoint_path)
                    model_path = os.path.join(checkpoint_path, model_name)

                    # Build Model
                    if model_used == 'EEGNet':
                        model = old_models.EEGNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
                    elif model_used == 'DeepConvNet':
                        model = old_models.DeepConvNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
                    elif model_used == 'ShallowConvNet':
                        model = old_models.ShallowConvNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
                    else:
                        raise Exception('No such model:{}'.format(model_used))

                    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
                    # model.load_weights(model_path)

                    save_path = os.path.join(checkpoint_path, 'adv_patch_v_target{}_size{}_{}.npz'.format(target_class, patch_size[0], patch_size[1]))
                    patch_save_path = os.path.join(checkpoint_path, 'origin_adv_patch_v_target{}_size{}_{}.npz'.format(target_class, patch_size[0], patch_size[1]))


                    v = utils.Adv_patch_target(x_train, model=model, model_used=model_used, model_path=model_path,
                                                save_path=save_path, patch_size=patch_size, patch_scale=None, noise_limit=a, patch_save_path=patch_save_path,
                                               attack_type=attack_type, target_class=target_class, batch_size=64,
                                                nb_classes=nb_classes, channels=channels, samples=samples, regular=reg)
                    # v = np.load(os.path.join(checkpoint_path, 'adv_v_target{}.npz'.format(target_class)))['v']
                    v = np.expand_dims(v, axis=0)
                    # print(v)


                    model.load_weights(model_path)
                    y_test = y_test.astype('int32').squeeze()
                    y_test_pre = np.argmax(model.predict(x_test), axis=1)

                    raw_acc = np.sum(y_test_pre == y_test) / len(y_test_pre)

                    # np.random.seed(2009)
                    shape = x_test.shape
                    # random_v = a * np.random.rand(1, 1, channels, samples)
                    # random_v = a * np.random.uniform(-1, 1, (1, 1, channels, samples))
                    random_v = a * np.clip(np.random.randn(1, 1, channels, samples), -0.2, 0.2)
                    random_x = x_test + random_v

                    y_rand_pre = np.argmax(model.predict(random_x), axis=1)
                    rand_acc = np.sum(y_rand_pre == y_test) / len(y_rand_pre)

                    adv_x = x_test + v
                    # print(v.shape)

                    y_adv_pre = np.argmax(model.predict(adv_x), axis=1)

                    adv_acc = np.sum(y_adv_pre == y_test) / len(y_adv_pre)

                    raw_accs.append(raw_acc)
                    rand_accs.append(rand_acc)
                    adv_accs.append(adv_acc)

                    raw_bca = utils.bca(y_test, y_test_pre)
                    rand_bca = utils.bca(y_test, y_rand_pre)
                    adv_bca = utils.bca(y_test, y_adv_pre)

                    raw_bcas.append(raw_bca)
                    rand_bcas.append(rand_bca)
                    adv_bcas.append(adv_bca)

                    raw_tr = np.sum(y_test_pre == target_class) / len(y_test)
                    rand_tr = np.sum(y_rand_pre == target_class) / len(y_test)
                    adv_tr = np.sum(y_adv_pre == target_class) / len(y_test)

                    raw_trs.append(raw_tr)
                    rand_trs.append(rand_tr)
                    adv_trs.append(adv_tr)

                    print('raw target rate: ', raw_tr)
                    print('rand target rate: ', rand_tr)
                    print('adv target rate: ', adv_tr)
                    K.clear_session()
                        # exit()
                print(raw_trs)
                print(rand_trs)
                print(adv_trs)
                print('\n')
                print(np.mean(raw_trs), np.mean(rand_trs), np.mean(adv_trs))
                tr_list.append([np.mean(raw_trs), np.mean(rand_trs), np.mean(adv_trs)])

            for tr in tr_list:
                print(tr)
        model_tr.append(tr_list)
    for m_tr in model_tr:
        print(m_tr)
