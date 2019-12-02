import models
# import old_models
import os
import numpy as np
import tensorflow.keras.backend as K
import utils
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from cleverhans.attacks import DeepFool
from cleverhans.utils_keras import KerasModelWrapper
from tqdm import tqdm
# import logging
import matplotlib.pyplot as plt
# logging.getLogger("requests").setLevel(logging.WARNING)
# logging.getLogger("urllib3").setLevel(logging.WARNING)


def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi / np.linalg.norm(v.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
        raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v


def universal_perturbation(model, deepfool, x, delta=0.8, max_iter_uni=5, xi=1, p=np.inf, num_classes=10, overshoot=0.02,
                           max_iter_df=10):
    v = 0
    fooling_rate = 0.0
    fool_list = []
    itr = 0
    df_params = {'max_iter': 50, 'nb_candidate': 3,
                 'clip_min': -1., 'clip_max': 1.}

    # np.random.shuffle(x)
    # end = int(0.2 * len(x))
    # data = x[0:end]
    data = x
    v_list = []
    while fooling_rate < delta and itr < max_iter_uni-1:
        np.random.shuffle(data)
        for cur_x in tqdm(data):
            # df_params = {'over_shoot': 0.09,
            #              'max_iter': 10,
            #              'clip_max': 1,
            #              'clip_min': 0,
            #              'nb_candidate': 2}

            cur_x = cur_x.reshape(1, 1, 22, 256)
            if int(np.argmax(np.array(model.predict(cur_x)).flatten())) == int(
                    np.argmax(np.array(model.predict(cur_x + v)).flatten())):
                adv_x = deepfool.generate_np(cur_x, **df_params)
                dr = adv_x - cur_x
                v = v + dr
                # Project on l_p ball
                v = proj_lp(v, xi, p)
                # print(v)
                # print(cur_x)

        v_list.append(v)
        itr = itr + 1
        data_adv = data + v
        num_images = len(data)
        est_labels_orig = np.zeros((num_images))
        est_labels_pert = np.zeros((num_images))

        batch_size = 100
        num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

        # Compute the estimated labels in batches
        for ii in range(0, num_batches):
            m = (ii * batch_size)
            M = min((ii + 1) * batch_size, num_images)
            est_labels_orig[m:M] = np.argmax(model.predict(data[m:M, :, :, :]), axis=1).flatten()
            est_labels_pert[m:M] = np.argmax(model.predict(data_adv[m:M, :, :, :]), axis=1).flatten()

        # Compute the fooling rate
        fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
        print('FOOLING RATE = ', fooling_rate)
        fool_list.append(fooling_rate)
    max_index = fool_list.index(max(fool_list))
    print(max_index)
    v = v_list[max_index]
    return v, fool_list

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    K.set_image_data_format('channels_first')
    K.set_learning_phase(0) #set learning phase

    # data_names = ['EPFL', 'ERN', 'MI4C']
    data_names = ['MI4C']
    # model_list = ['EEGNet', 'DeepConvNet', 'ShallowConvNet']
    model_list = ['DeepConvNet']

    epsilon = 0.1

    for data_name in data_names:
        # Load dataset
        data_dir = 'EEG_data/process_data'
        train_dir = 'runs'
        checkpoint = 'checkpoint'
        model_name = 'model.h5'
        batch_size = 256
        epoches = 1000

        # Build pathes
        data_path = os.path.join(data_dir, '{}.npz'.format(data_name))

        # Load dataset
        data = np.load(data_path)
        print(data.keys())
        x = data['x']
        y = data['y']
        s = data['s']
        # seed = 2019

        # x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.3, random_state=seed)

        # num_classes = 4
        print(x.shape,y.shape)
        raw_accs = []
        adv_accs = []
        raw_bcas = []
        adv_bcas = []
        subject = 4
        datasets_wrt_s = utils.list_leave_one_subject(x, y, s, shuffle=True)
        for dataset_wrt_s in datasets_wrt_s:
            print(subject)
            x_train, y_train = dataset_wrt_s[0]
            x_test, y_test = dataset_wrt_s[1]
            subject = dataset_wrt_s[2]

            nb_classes = len(np.unique(y_test))
            samples = x_test.shape[3]
            channels = x_test.shape[2]

            print(x_train.shape, x_test.shape)

            # for model_used in model_list:
            model_used = 'EEGNet'

            # Build pathes
            # checkpoint_path = os.path.join('runs', '{}_{}'.format(data_name, model_used), '{}'.format(int(subject)), 'checkpoint')
            # checkpoint_path = os.path.join('runs', '{}_{}'.format(data_name, model_used), '{}'.format(int(subject)), 'checkpoint')
            # print(checkpoint_path)
            checkpoint_path = os.path.join('/mnt/disk2/zhangxiao/AdversarialExamplesOnEEG_cross_subject/runs',
                                           data_name, model_used, '{}'.format(int(subject)))
            # checkpoint_path = os.path.join('/mnt/disk2/zhangxiao/AdversarialExamplesOnEEG_cross_subject/runs', '{}/{}'.format(data_name, model_used), '{}'.format(int(subject)))
            print(checkpoint_path)
            model_path = os.path.join(checkpoint_path, 'model.h5')

            # checkpoint_path = 'model'
            # model_path = os.path.join(checkpoint_path, 'model.h5')

            # Build Model
            if model_used == 'EEGNet':
                model = models.EEGNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
            elif model_used == 'DeepConvNet':
                model = models.DeepConvNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
            elif model_used == 'ShallowConvNet':
                model = models.ShallowConvNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
            else:
                raise Exception('No such model:{}'.format(model_used))


            # target = K.placeholder(
            #     ndim=len(model.outputs),
            #     name='my_target',
            #     sparse=K.is_sparse(model.outputs[0]),
            #     dtype=K.dtype(model.outputs[0])
            # )

            # model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc'], target=[target])
            model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
            model.load_weights(model_path)
            # ------------ Unsupervised Attacking -------------
            y_test = y_test.flatten()
            y_test_pre = np.argmax(model.predict(x_test), axis=1)


            ch_model = KerasModelWrapper(model)
            deepfool = DeepFool(ch_model, back='tf', sess=K.get_session())

            raw_acc = np.sum(y_test_pre == y_test) / len(y_test_pre)
            # print(np.sum(y_test_pre == y_test), y_test_pre.shape, y_test.shape)
            print(raw_acc)
            v, fool_list= universal_perturbation(model, deepfool, x_train)
            print(v, '\n', fool_list)
            adv_x = x_test + v
            print(v.shape)
            # np.savez('adv_x_test.npz', x=adv_x)

            # df_params = {'max_iter': 50, 'nb_candidate': 3,
            #              'clip_min': -1., 'clip_max': 1.}
            # adv_x = deepfool.generate_np(x_test, **df_params)
            # v = adv_x - x_test

            y_adv_pre = np.argmax(model.predict(adv_x), axis=1)
            adv_acc = np.sum(y_adv_pre == y_test) / len(y_adv_pre)
            print(raw_acc)
            print(adv_acc)
            raw_accs.append(raw_acc)
            adv_accs.append(adv_acc)
            raw_bca = utils.bca(y_test, y_test_pre)
            adv_bca = utils.bca(y_test, y_adv_pre)
            raw_bcas.append(raw_bca)
            adv_bcas.append(adv_bca)
            subject += 1
            print(raw_bca)
            print(adv_bca)


            utils.plot_data(x_test, v)
            utils.plot_data(x_test,adv_x)
            exit()
        print(raw_accs,'\n',adv_accs,'\n',
              raw_bcas,'\n',adv_bcas,'\n')
        print(np.mean(raw_accs), np.mean(adv_accs), np.mean(raw_bcas), np.mean(adv_bcas))











