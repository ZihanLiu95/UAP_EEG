import old_models
import keras.backend as K
import os
import utils
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold


if __name__ == '__main__':
    K.set_image_data_format('channels_first')
    K.set_learning_phase(0)  # set learning phase

    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    random_seed = None
    # np.random.seed(random_seed)
    data_dir = 'Data/processed_data'
    train_dir = 'within_block_runs'
    model_name = 'model.h5'
    batch_size = 64
    epoches = 1600

    # data_list = ['EPFL', 'ERN', 'MI4C']
    data_list = ['ERN']

    # model_list = ['EEGNet', 'DeepConvNet', 'ShallowConvNet']
    model_list = ['ShallowConvNet']

    a = 0.2  # noisy
    # target_classes = [0, 1, 2, 3]
    k_fold = 5

    attack_type = 'target'  # 'target' or 'nontarget'

    for data_name in data_list:
        if data_name == 'EPFL':
            s_num = 8
            target_classes = [0, 1]
        elif data_name == 'ERN':
            s_num = 16
            target_classes = [0, 1]
        elif data_name == 'MI4C':
            s_num = 9
            target_classes = [0, 1, 2, 3]
        else:
            raise Exception('No such dataset:{}'.format(data_name))

        tr_list = []
        for target_class in target_classes:

            for model_used in model_list:
                raw_trs = np.zeros((s_num, k_fold))
                rand_trs = np.zeros((s_num, k_fold))
                adv_trs = np.zeros((s_num, k_fold))
                for s_id in range(s_num):
                    # Build pathes
                    data_path = os.path.join(data_dir, data_name, 'block', 's{}.mat'.format(s_id))
                    # Load dataset
                    data = loadmat(data_path)
                    x = data['x']
                    y = np.squeeze(data['y'])
                    kfold = KFold(n_splits=k_fold, shuffle=False)
                    kf = 0
                    for train_index, test_index in kfold.split(x, y):
                        # Build checkpoint pathes
                        checkpoint_path = os.path.join(train_dir, data_name, 'gray_{}'.format(model_used), '{}'.format(s_id), '{}'.format(kf))
                        model_path = os.path.join(checkpoint_path, model_name)

                        if not os.path.exists(checkpoint_path):
                            os.makedirs(checkpoint_path)
                        print(checkpoint_path)

                        x_train = x[train_index]
                        y_train = y[train_index]
                        x_test = x[test_index]
                        y_test = y[test_index]

                        data_size = y_train.shape[0]
                        print(x_train.shape)
                        print(y_train.shape)

                        nb_classes = len(np.unique(y_train))
                        samples = x_train.shape[3]
                        channels = x_train.shape[2]

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


                        save_path = os.path.join(checkpoint_path, 'adv_v_target{}.npz'.format(target_class))



                        v = utils.UAP_target(x_train, y_train, model_used=model_used, model_path=model_path, save_path=save_path, noise_limit=a)
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
                        random_v = a * np.random.randn(1, 1, channels, samples)
                        random_x = x_test + random_v

                        y_rand_pre = np.argmax(model.predict(random_x), axis=1)
                        rand_acc = np.sum(y_rand_pre == y_test) / len(y_rand_pre)

                        adv_x = x_test + v
                        # print(v.shape)

                        y_adv_pre = np.argmax(model.predict(adv_x), axis=1)

                        adv_acc = np.sum(y_adv_pre == y_test) / len(y_adv_pre)

                        raw_tr = np.sum(y_test_pre == target_class) / len(y_test)
                        rand_tr = np.sum(y_rand_pre == target_class) / len(y_test)
                        adv_tr = np.sum(y_adv_pre == target_class) / len(y_test)

                        print('raw target rate: ', raw_tr)
                        print('rand target rate: ', rand_tr)
                        print('adv target rate: ', adv_tr)

                        raw_trs[s_id, kf] = raw_tr
                        rand_trs[s_id, kf] = rand_tr
                        adv_trs[s_id, kf] = adv_tr

                        K.clear_session()

                        kf += 1
                        # exit()
                    print('subject {}/{}, target rate:'.format(s_id, s_num), [list(raw_trs[s_id]), list(rand_trs[s_id]), list(adv_trs[s_id])])
                print(np.mean(raw_trs, axis=1))
                print(np.mean(rand_trs, axis=1))
                print(np.mean(adv_trs, axis=1))
                print('\n')
                tr_list.append([np.mean(raw_trs), np.mean(rand_trs), np.mean(adv_trs)])
                print(tr_list[-1])


        for tr in tr_list:
            print(tr)