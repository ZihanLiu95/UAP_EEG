import old_models
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
import os
import utils
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from sklearn.model_selection import KFold

K.set_image_data_format('channels_first')


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.32
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

gpuConfig = tf.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth = True

random_seed = None
# np.random.seed(random_seed)
data_dir = 'Data/processed_data'
train_dir = 'within_block_runs'
# model_name = 'model.h5'
model_name = 'model.h5'
batch_size = 64
epoches = 1000

data_list = ['ERN']
s_num = 16
# model_list = ['EEGNet', 'DeepConvNet', 'ShallowConvNet']
model_list = ['EEGNet']
k_fold = 5

for data_name in data_list:
    for model_used in model_list:
        for s_id in range(s_num):
            # Build pathes
            data_path = os.path.join(data_dir, data_name, 'block', 's{}.mat'.format(s_id))
            # Load dataset
            data = loadmat(data_path)
            x = data['x']
            y = np.squeeze(data['y'])
            kfold = KFold(n_splits=k_fold, shuffle=False)
            kf = 0
            # acc_kfold = 0.
            # bca_kfold = 0.
            for train_index, test_index in kfold.split(x, y):
                # Build checkpoint pathes
                checkpoint_path = os.path.join(train_dir, data_name, 'gray_{}'.format(model_used),
                                               '{}'.format(s_id), '{}'.format(kf))
                model_path = os.path.join(checkpoint_path, model_name)

                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                print(checkpoint_path)

                x_train = x[train_index]
                y_train = y[train_index]
                x_test = x[test_index]
                y_test = y[test_index]

                if 'EPFL' in data_name:
                    # downsampling
                    data_size = y_train.shape[0]
                    shuffle_index = utils.shuffle_data(data_size, random_seed=random_seed)
                    # shuffle_index = utils.shuffle_data(data_size)
                    x_train = x_train[shuffle_index]
                    y_train = y_train[shuffle_index]

                    y0_index = np.argwhere(y_train==0)
                    y1_index = np.argwhere(y_train==1)
                    length = min(len(y0_index), len(y1_index))
                    y0_index = y0_index[0:length]
                    y1_index = y1_index[0:length]
                    y_index = np.squeeze(np.concatenate((y0_index, y1_index)))

                    y_train = y_train[y_index]
                    x_train = x_train[y_index]
                elif 'ERN' in data_name:
                    y0_rate = np.mean(np.where(y_train==0, 1, 0))
                    y1_rate = np.mean(np.where(y_train==1, 1, 0))
                    class_weights = {0: y1_rate, 1: y0_rate}

                data_size = y_train.shape[0]
                # shuffle_index = utils.shuffle_data(data_size, random_seed=random_seed)
                shuffle_index = utils.shuffle_data(data_size)
                x_train = x_train[shuffle_index]
                y_train = y_train[shuffle_index]

                print(x_train.shape)
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

                model.summary()
                model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
                early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=30)
                model_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', mode='min', save_best_only=True)

                # Train Model
                his = model.fit(
                    x_train, y_train,
                    batch_size=batch_size,
                    validation_split=0.25,
                    shuffle=False,
                    epochs=epoches,
                    callbacks=[early_stop, model_checkpoint],
                    class_weight=class_weights
                )

                # Test Model
                # y_pred = np.argmax(model.predict(x_test), axis=1)
                # y_test = np.squeeze(y_test)
                # bca = utils.bca(y_test, y_pred)
                # acc = np.sum(y_pred==y_test).astype(np.float32)/len(y_pred)
                # print('{}_{}:bca-{} acc-{}'.format(data_name, model_used, bca, acc))
                K.clear_session()

                kf += 1

