import old_models
import keras.backend as K
import os
import utils
import numpy as np



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    K.set_image_data_format('channels_first')
    K.set_learning_phase(0) #set learning phase

    # data_names = ['EPFL', 'ERN', 'MI4C']
    data_names = ['MI4C']
    # model_list = ['EEGNet', 'DeepConvNet', 'ShallowConvNet']
    model_list = ['ShallowConvNet']

    a = 0.2   # noisy
    # epsilon = 0.1
    attack_type = 'nontarget'  # 'target' or 'nontarget'

    for data_name in data_names:
        # Load dataset
        data_dir = 'Data/processed_data'
        train_dir = 'runs'
        checkpoint = 'checkpoint'
        model_name = 'model.h5'
        batch_size = 64
        epoches = 1600
        reg = 'l1'

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
        rand_accs = []
        rand_bcas = []
        raw_bcas = []
        adv_bcas = []

        datasets_wrt_s = utils.list_leave_one_subject(x, y, s, shuffle=True)
        for dataset_wrt_s in datasets_wrt_s:
            x_train, y_train = dataset_wrt_s[0]
            x_test, y_test = dataset_wrt_s[1]
            subject = dataset_wrt_s[2]

            nb_classes = len(np.unique(y_test))
            samples = x_test.shape[3]
            channels = x_test.shape[2]

            print(x_train.shape, x_test.shape, nb_classes)

            for model_used in model_list:


                # Build pathes
                # checkpoint_path = os.path.join('runs', '{}_{}'.format(data_name, model_used), '{}'.format(int(subject)), 'checkpoint')
                # checkpoint_path = os.path.join('runs', '{}_{}'.format(data_name, model_used), '{}'.format(int(subject)), 'checkpoint')
                # print(checkpoint_path)
                # checkpoint_path = os.path.join('/mnt/disk2/zhangxiao/AdversarialExamplesOnEEG_cross_subject/old_runs',
                #                                '{}_{}'.format(data_name, model_used), '{}'.format(int(subject)), 'checkpoint')
                checkpoint_path = os.path.join('runs', '{}/{}'.format(data_name, model_used), '{}'.format(int(subject)))
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


                save_path = os.path.join(checkpoint_path, 'adv_v_nontarget_pre_{}_test.npz'.format(reg))



                # v = utils.UAP_target(x_train, y_train, model=model, model_used=model_used, model_path=model_path, save_path=save_path,
                #                noise_limit=a, attack_type=attack_type, batch_size=batch_size, nb_classes=nb_classes, channels=channels, samples=samples, regular=reg)
                v = utils.UAP_target_pre(x_train, model=model, model_used=model_used, model_path=model_path,
                                     save_path=save_path,
                                     noise_limit=a, attack_type=attack_type, batch_size=batch_size,
                                     nb_classes=nb_classes, channels=channels, samples=samples, regular=reg)

                v = np.load(save_path)['v']
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
                random_v = a * np.random.uniform(-1, 1, (1, 1, channels, samples))
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
                K.clear_session()
                # exit()
                print(raw_accs)
                print(rand_accs)
                print(adv_accs)
                print('\n')


        print(np.mean(raw_accs), np.mean(raw_bcas), np.mean(rand_accs), np.mean(rand_bcas), np.mean(adv_accs),
              np.mean(adv_bcas),'\n')