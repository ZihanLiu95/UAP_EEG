import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from keras import Model
from keras.layers import Lambda, Input
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import old_models

def standard_normalize(x, clip_range=None):
    x = (x-np.mean(x))/np.std(x)
    if clip_range is not None:
        x = np.clip(x, a_min=clip_range[0], a_max=clip_range[1])
    return x


def split_data(data_size, split=0.8, random_seed=1000, shuffle=True):
    np.random.seed(random_seed)
    split_index = int(data_size * split)
    indices = np.arange(data_size)
    if shuffle:
        indices = np.random.permutation(indices)
    return indices[:split_index], indices[split_index:]


def shuffle_data(data_size, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    indices = np.arange(data_size).squeeze()
    return np.random.permutation(indices).squeeze()


def bca(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    numb = m.shape[0]
    acc_each_label = 0
    for i in range(numb):
        all = np.sum(m[i, :], keepdims=False).astype(np.float32)
        acc = m[i, i]/all
        acc_each_label += acc
    return acc_each_label/numb


def list_leave_one_subject(x, y, s, shuffle=True):
    subjects = np.unique(s)
    for subject in subjects:
        test_indics = np.argwhere(s == subject).squeeze()
        test_x = x[test_indics]
        test_y = y[test_indics]

        train_indics = np.argwhere(s != subject).squeeze()
        train_x = x[train_indics]
        train_y = y[train_indics]

        if shuffle:
            indices = shuffle_data(len(train_y))
            train_x = train_x[indices]
            train_y = train_y[indices]

        yield (train_x, train_y), (test_x, test_y), subject



def list_cross_subject(x, y, s, shuffle=True):
    subjects = np.unique(s)
    subject = 0
    test_indics = np.argwhere(s == subject).squeeze()
    test_x = x[test_indics]
    test_y = y[test_indics]

    train_indics = np.argwhere(s != subject).squeeze()
    train_x = x[train_indics]
    train_y = y[train_indics]

    if shuffle:
        indices = shuffle_data(len(train_y))
        train_x = train_x[indices]
        train_y = train_y[indices]

    return (train_x, train_y), (test_x, test_y), subject



def plot_data(test_x, adv_x, data='MI'):
    if data == 'EPFL':
        t = np.arange(test_x.shape[3]) * 1. / 250.
    if data == 'MI4C':
        t = np.arange(test_x.shape[3]) * 1. / 128.
    if data == 'ERN':
        t = np.arange(test_x.shape[3]) * 1. / 200.
    i = 0
    channels = 10
    for j in range(channels):
        if j == 0:
            plt.plot(t, adv_x[i, 0, j, :]+j*4, 'r', label='Adversarial Example', linewidth=1)
            plt.plot(t, test_x[i, 0, j, :]+j*4, 'g', label='Original Example', linewidth=1)
        else:
            plt.plot(t, adv_x[i, 0, j, :]+j*4, 'r', linewidth=1)
            plt.plot(t, test_x[i, 0, j, :]+j*4, 'g', linewidth=1)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylim([-4, 46])
    temp_y = np.arange(channels) * 4
    y_names = ['channel {}'.format(int(y_id/4)) for y_id in temp_y]
    plt.yticks(temp_y, y_names, fontsize=12)
    plt.legend(loc='upper center', fontsize=12, ncol=2)
    # plt.title('Original EEG Signal and Adversarial EEG Signal', fontsize=15)
    plt.tight_layout()
    plt.show()

def plot_data_new(test_x, adv_x, data='MI'):
    if data == 'EPFL':
        t = np.arange(test_x.shape[3]) * 1. / 250.
    if data == 'MI4C':
        t = np.arange(test_x.shape[3]) * 1. / 128.
    if data == 'ERN':
        t = np.arange(test_x.shape[3]) * 1. / 200.
    i = 0
    channels = [0, 7, 9, 11, 19]
    for j in range(len(channels)):
        if j == 0:
            plt.plot(t, adv_x[i, 0, j, :]+j*4, 'r', label='Adversarial Example', linewidth=1)
            plt.plot(t, test_x[i, 0, j, :]+j*4, 'g', label='Benign Example', linewidth=1)
        else:
            plt.plot(t, adv_x[i, 0, channels[j], :]+j*4, 'r', linewidth=1)
            plt.plot(t, test_x[i, 0, channels[j], :]+j*4, 'g', linewidth=1)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Channel', fontsize=14)
    plt.ylim([-4, 23])
    temp_y = np.arange(len(channels)) * 4
    y_names = ['$Fz$', '$C3$', '$Cz$', '$C4$', '$Pz$']
    plt.yticks(temp_y, y_names, fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(loc='upper center', fontsize=12, ncol=2)
    # plt.title('Original EEG Signal and Adversarial EEG Signal', fontsize=15)
    plt.tight_layout()
    plt.show()


def plot_regular(adv_x, adv_x_l1, adv_x_l2, adv_x_l3, data='MI'):
    if data == 'EPFL':
        t = np.arange(adv_x.shape[3]) * 1. / 250.
    if data == 'MI4C':
        t = np.arange(adv_x.shape[3]) * 1. / 128.
    if data == 'ERN':
        t = np.arange(adv_x.shape[3]) * 1. / 200.
    i = 0
    channels = 2
    for j in [2]:
        j=2
        if j == 2:
            plt.plot(t, adv_x[i, 0, j, :], 'r', label='No', linewidth=1.5)
            plt.plot(t, adv_x_l1[i, 0, j, :], 'g', label='L1', linewidth=1.5)
            plt.plot(t, adv_x_l2[i, 0, j, :], 'b', label='L2', linewidth=1.5)
            plt.plot(t, adv_x_l3[i, 0, j, :], 'm', label='L1+L2', linewidth=1.5)
            plt.plot(t, [0]*len(t), 'black',linestyle='--', linewidth=0.5)
        else:
            plt.plot(t, adv_x[i, 0, j, :], 'r', linewidth=1.5)
            plt.plot(t, adv_x_l1[i, 0, j, :], 'g', linewidth=1.5)
            plt.plot(t, adv_x_l2[i, 0, j, :], 'b', linewidth=1.5)
            plt.plot(t, adv_x_l3[i, 0, j, :], 'm', label='L1+L2', linewidth=1.5)
    plt.xlabel('time (s)', fontsize=12)
    plt.ylim([-0.5,0.5])
    plt.xlim([0,0.3])
    temp_y = np.arange(channels) * 0.5
    # y_names = ['channel {}'.format(int(y_id/0.5)) for y_id in temp_y]
    y_names = ['channel {}'.format(channels)]
    plt.yticks(temp_y, y_names, fontsize=10)
    plt.legend(loc='upper right', fontsize=10)
    # plt.title('Original EEG Signal and Adversarial EEG Signal', fontsize=15)
    plt.tight_layout()
    plt.show()



def plot_regular_0(adv_x, adv_x_l1, adv_x_l2, adv_x_l3, data='MI'):
    if data == 'EPFL':
        t = np.arange(adv_x.shape[3]) * 1. / 250.
    if data == 'MI4C':
        t = np.arange(adv_x.shape[3]) * 1. / 128.
    if data == 'ERN':
        t = np.arange(adv_x.shape[3]) * 1. / 200.
    i = 0
    channels = 1
    for j in range(channels):
        if j == 0:
            plt.plot(t, adv_x[i, 0, j, :], 'r', label='No', linewidth=1.5)
            plt.plot(t, adv_x_l1[i, 0, j, :], 'g', label='L1', linewidth=1.5)
            plt.plot(t, adv_x_l2[i, 0, j, :], 'b', label='L2', linewidth=1.5)
            plt.plot(t, adv_x_l3[i, 0, j, :], 'm', label='L1+L2', linewidth=1.5)
            plt.plot(t, [0]*len(t), 'black',linestyle='--', linewidth=0.5)
        else:
            plt.plot(t, adv_x[i, 0, j, :], 'r', linewidth=1.5)
            plt.plot(t, adv_x_l1[i, 0, j, :], 'g', linewidth=1.5)
            plt.plot(t, adv_x_l2[i, 0, j, :], 'b', linewidth=1.5)
            plt.plot(t, adv_x_l3[i, 0, j, :], 'm', label='L1+L2', linewidth=1.5)
    plt.xlabel('time (s)', fontsize=12)
    plt.xlim([0, 0.3])
    plt.ylim([-0.5,0.5])
    temp_y = np.arange(channels) * 0.5
    y_names = ['channel {}'.format(int(y_id/0.5)) for y_id in temp_y]
    # y_names = ['channel {}'.format(channels)]
    plt.yticks(temp_y, y_names, fontsize=10)
    plt.legend(loc='upper right', fontsize=10)
    # plt.title('Original EEG Signal and Adversarial EEG Signal', fontsize=15)
    plt.tight_layout()
    plt.show()


def plot_regular_fz(adv_x, adv_x_l1, adv_x_l2, adv_x_l3, data='MI'):
    if data == 'EPFL':
        t = np.arange(adv_x.shape[3]) * 1. / 250.
    if data == 'MI4C':
        t = np.arange(adv_x.shape[3]) * 1. / 128.
    if data == 'ERN':
        t = np.arange(adv_x.shape[3]) * 1. / 200.
    i = 0
    channels = 1
    for j in range(channels):
        if j == 0:
            plt.plot(t, adv_x[i, 0, j, :], 'r', label='No', linewidth=1.5)
            plt.plot(t, adv_x_l1[i, 0, j, :], 'g', label='L1', linewidth=1.5)
            plt.plot(t, adv_x_l2[i, 0, j, :], 'b', label='L2', linewidth=1.5)
            plt.plot(t, adv_x_l3[i, 0, j, :], 'm', label='L1+L2', linewidth=1.5)
            plt.plot(t, [0]*len(t), 'black',linestyle='--', linewidth=0.5)
        else:
            plt.plot(t, adv_x[i, 0, j, :], 'r', linewidth=1.5)
            plt.plot(t, adv_x_l1[i, 0, j, :], 'g', linewidth=1.5)
            plt.plot(t, adv_x_l2[i, 0, j, :], 'b', linewidth=1.5)
            plt.plot(t, adv_x_l3[i, 0, j, :], 'm', label='L1+L2', linewidth=1.5)
    plt.xlabel('time (s)', fontsize=12)
    plt.xlim([0, 0.3])
    plt.ylim([-0.5,0.5])
    temp_y = np.arange(channels) * 0.5
    y_names = ['Channel $Fz$ '.format(int(y_id/0.5)) for y_id in temp_y]
    # y_names = ['channel {}'.format(channels)]
    plt.yticks(temp_y, y_names, fontsize=10)
    plt.legend(loc='upper right', fontsize=10)
    # plt.title('Original EEG Signal and Adversarial EEG Signal', fontsize=15)
    plt.tight_layout()
    plt.show()

def plot_regular_mutil(adv_x, adv_x_l1, adv_x_l2, adv_x_l3, data='MI'):
    if data == 'EPFL':
        t = np.arange(adv_x.shape[3]) * 1. / 250.
    if data == 'MI4C':
        t = np.arange(adv_x.shape[3]) * 1. / 128.
    if data == 'ERN':
        t = np.arange(adv_x.shape[3]) * 1. / 200.
    i = 0
    channels = 20
    for j in range(channels):
        if j == 0:
            plt.plot(t, adv_x[i, 0, j, :], 'r', label='No', linewidth=1.5)
            plt.plot(t, adv_x_l1[i, 0, j, :], 'g', label='L1', linewidth=1.5)
            plt.plot(t, adv_x_l2[i, 0, j, :], 'b', label='L2', linewidth=1.5)
            plt.plot(t, adv_x_l3[i, 0, j, :], 'm', label='L1+L2', linewidth=1.5)
            plt.plot(t, [0]*len(t), 'black',linestyle='--', linewidth=0.5)
        if j == 9:
            plt.plot(t, adv_x[i, 0, j, :]+0.8, 'r', linewidth=1.5)
            plt.plot(t, adv_x_l1[i, 0, j, :]+0.8, 'g', linewidth=1.5)
            plt.plot(t, adv_x_l2[i, 0, j, :]+0.8, 'b', linewidth=1.5)
            plt.plot(t, adv_x_l3[i, 0, j, :]+0.8, 'm', linewidth=1.5)
            plt.plot(t, [0.8] * len(t), 'black', linestyle='--', linewidth=0.5)
        if j == 19:
            plt.plot(t, adv_x[i, 0, j, :]+2*0.8, 'r', linewidth=1.5)
            plt.plot(t, adv_x_l1[i, 0, j, :]+2*0.8, 'g', linewidth=1.5)
            plt.plot(t, adv_x_l2[i, 0, j, :]+2*0.8, 'b', linewidth=1.5)
            plt.plot(t, adv_x_l3[i, 0, j, :]+2*0.8, 'm', linewidth=1.5)
            plt.plot(t, [1.6] * len(t), 'black', linestyle='--', linewidth=0.5)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Channel', fontsize=14)
    plt.xlim([0, 0.3])
    plt.ylim([-0.5,2.3])
    temp_y = [0,0.8,1.6]
    y_names = ['$Fz$', '$Cz$', '$Pz$']
    # y_names = ['channel {}'.format(channels)]
    plt.yticks(temp_y, y_names, fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(loc='upper center', fontsize=12, ncol=4)
    # plt.title('Original EEG Signal and Adversarial EEG Signal', fontsize=15)
    plt.tight_layout()
    plt.show()

def plot_regular_3(adv_x, adv_x_l1, adv_x_l2, data='MI'):
    if data == 'EPFL':
        t = np.arange(adv_x.shape[3]) * 1. / 250.
    if data == 'MI4C':
        t = np.arange(adv_x.shape[3]) * 1. / 128.
    if data == 'ERN':
        t = np.arange(adv_x.shape[3]) * 1. / 200.
    i = 0
    channels = 20
    for j in range(channels):
        if j == 0:
            plt.plot(t, adv_x[i, 0, j, :], 'r', label='No', linewidth=1.5)
            plt.plot(t, adv_x_l1[i, 0, j, :], 'g', label='L1', linewidth=1.5)
            plt.plot(t, adv_x_l2[i, 0, j, :], 'b', label='L2', linewidth=1.5)
            # plt.plot(t, adv_x_l3[i, 0, j, :], 'm', label='L1+L2', linewidth=1.5)
            plt.plot(t, [0]*len(t), 'black',linestyle='--', linewidth=0.5)
        if j == 9:
            plt.plot(t, adv_x[i, 0, j, :]+0.8, 'r', linewidth=1.5)
            plt.plot(t, adv_x_l1[i, 0, j, :]+0.8, 'g', linewidth=1.5)
            plt.plot(t, adv_x_l2[i, 0, j, :]+0.8, 'b', linewidth=1.5)
            # plt.plot(t, adv_x_l3[i, 0, j, :]+0.8, 'm', linewidth=1.5)
            plt.plot(t, [0.8] * len(t), 'black', linestyle='--', linewidth=0.5)
        if j == 19:
            plt.plot(t, adv_x[i, 0, j, :]+2*0.8, 'r', linewidth=1.5)
            plt.plot(t, adv_x_l1[i, 0, j, :]+2*0.8, 'g', linewidth=1.5)
            plt.plot(t, adv_x_l2[i, 0, j, :]+2*0.8, 'b', linewidth=1.5)
            # plt.plot(t, adv_x_l3[i, 0, j, :]+2*0.8, 'm', linewidth=1.5)
            plt.plot(t, [1.6] * len(t), 'black', linestyle='--', linewidth=0.5)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Channel', fontsize=14)
    plt.xlim([0, 0.3])
    plt.ylim([-0.5,2.3])
    temp_y = [0,0.8,1.6]
    y_names = ['$Fz$', '$Cz$', '$Pz$']
    # y_names = ['channel {}'.format(channels)]
    plt.yticks(temp_y, y_names, fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(loc='upper center', fontsize=12, ncol=4)
    # plt.title('Original EEG Signal and Adversarial EEG Signal', fontsize=15)
    plt.tight_layout()
    plt.show()


def plot_v_diversity(test_x, adv_x):
    t = np.arange(test_x.shape[3]) * 1. / 250.
    v = adv_x-test_x
    i = 0
    channels = 3
    for j in range(channels):
        if j == 0:
            plt.plot(t, v[i, 0, j, :]+j*2, 'r', label='UAP', linewidth=2)
            # plt.plot(t, test_x[i, 0, j, :]+j*4, 'g', label='Original Example', linewidth=1)
        else:
            plt.plot(t, v[i, 0, j, :]+j*2, 'r', linewidth=2)
            # plt.plot(t, test_x[i, 0, j, :]+j*4, 'g', linewidth=1)
    plt.xlabel('time (s)', fontsize=26)
    plt.xticks(fontsize=24)
    plt.ylim([-1, 6.6])
    temp_y = np.arange(channels) * 2
    y_names = ['ch {}'.format(int(y_id/2)) for y_id in temp_y]
    plt.yticks(temp_y, y_names, fontsize=26)
    plt.legend(loc='upper right', fontsize=26)
    # plt.title('Original EEG Signal and Adversarial EEG Signal', fontsize=15)
    plt.show()

def plot_alone(x0, x1, title):
    t = np.arange(x0.shape[3]) * 1. / 250.
    i = 0
    channels = 10
    for j in range(channels):
        if j == 0:
            plt.plot(t, x0[i, 0, j, :]+j*4, 'r', label='Adversarial Example', linewidth=1)
            plt.plot(t, x1[i, 0, j, :]+j*4, 'g', label='Original Example', linewidth=1)
        else:
            plt.plot(t, x0[i, 0, j, :]+j*4, 'r', linewidth=1)
            plt.plot(t, x1[i, 0, j, :]+j*4, 'g', linewidth=1)
    plt.xlabel('time (s)', fontsize=12)
    plt.ylim([-4, 46])
    temp_y = np.arange(channels) * 4
    y_names = ['channel {}'.format(int(y_id/4)) for y_id in temp_y]
    plt.yticks(temp_y, y_names, fontsize=10)
    plt.legend(loc='upper right', fontsize=10)
    plt.title(title, fontsize=15)
    plt.show()


def plot_one_channel(x0, x1, title):
    t = np.arange(x0.shape[3]) * 1. / 250.
    i = 0
    channels = 1
    for j in range(channels):
        if j == 0:
            plt.plot(t, x0[i, 0, j, :]+j*4, 'r', label='Adversarial Example', linewidth=1)
            plt.plot(t, x1[i, 0, j, :]+j*4, 'g', label='Original Example', linewidth=1)
        else:
            plt.plot(t, x0[i, 0, j, :]+j*4, 'r', linewidth=1)
            plt.plot(t, x1[i, 0, j, :]+j*4, 'g', linewidth=1)
    plt.xlabel('time (s)', fontsize=12)
    # plt.ylim([-4, 46])
    temp_y = np.arange(channels) * 4
    y_names = ['channel {}'.format(int(y_id/4)) for y_id in temp_y]
    # plt.yticks(temp_y, y_names, fontsize=10)
    plt.legend(loc='upper right', fontsize=10)
    plt.title(title, fontsize=15)
    plt.show()


def UAP_target(x, y, model, model_used, model_path, save_path, noise_limit=0.2, attack_type=None, target_class=None,
               batch_size=None, nb_classes=None, channels=None, samples=None, regular=None):
    x_train, x_val, y_train, y_val = train_test_split(x, y, shuffle=True, test_size=0.2)
    batch_size = min(batch_size, len(x_train))

    universal_noise = tf.Variable(np.zeros((x_train[0].shape)), dtype=tf.float32)
    temp_universal_noise = tf.expand_dims(universal_noise, 0)
    # print(temp_universal_noise)
    x_input = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    x = Lambda(lambda xx: xx + tf.clip_by_value(temp_universal_noise, -noise_limit, noise_limit))(x_input)

    # Model output
    if model_used == 'EEGNet':
        prediction = old_models.EEGNet_output(nb_classes=nb_classes, Chans=channels, Samples=samples, x_input=x)
    elif model_used == 'DeepConvNet':
        prediction = old_models.DeepConvNet_output(nb_classes=nb_classes, Chans=channels, Samples=samples, x_input=x)
    elif model_used == 'ShallowConvNet':
        prediction = old_models.ShallowConvNet_output(nb_classes=nb_classes, Chans=channels, Samples=samples, x_input=x)
    else:
        raise Exception('No such model:{}'.format(model_used))

    # print(prediction)
    u_model = Model(inputs=x_input, outputs=prediction)
    u_model.load_weights(model_path)
    model.load_weights(model_path)

    alpha = tf.placeholder(dtype=tf.float32)
    al = 100
    if regular == 'l1':
        loss = alpha * (tf.reduce_mean(tf.abs(universal_noise)))
        al = 10
    elif regular == 'l2':
        loss = alpha * (tf.reduce_mean(tf.square(universal_noise)))
        al = 100
    elif regular == 'l1+l2':
        loss = alpha * (tf.reduce_mean(10*tf.square(universal_noise) + tf.abs(universal_noise)))
        al = 10
    elif regular == None:
        loss = 0
    else:
        raise Exception('no such loss regularization!')
    # loss = alpha * (tf.reduce_mean(tf.square(universal_noise) + tf.abs(universal_noise)))
    # loss = alpha * (tf.reduce_mean(tf.square(universal_noise) + tf.square(universal_noise)))
    # print(loss)
    target = tf.placeholder(dtype=tf.int32, shape=[None, ])
    if attack_type == 'nontarget':
        # loss += K.mean(K.sparse_categorical_crossentropy(target, 1-prediction, from_logits=False))
        loss += -K.mean(K.sparse_categorical_crossentropy(target, prediction, from_logits=False))
    elif attack_type == 'target':
        loss += K.mean(K.sparse_categorical_crossentropy(target, prediction, from_logits=False))
    else:
        raise Exception('no such attack_type!')

    start_vars = set(x.name for x in tf.global_variables())
    lr_ph = tf.placeholder(shape=[], dtype=tf.float32)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr_ph)
    train = optimizer.minimize(loss, var_list=[universal_noise])

    end_vars = tf.global_variables()
    new_vars = [x for x in end_vars if x.name not in start_vars]
    init = tf.variables_initializer(var_list=[universal_noise] + new_vars)

    sess = K.get_session()
    sess.run(init)

    nb_batch = len(x_train) // batch_size


    end = False

    epochs = 500
    lr = 1e-3
    v = np.zeros((x_train[0].shape))

    patience = 0
    patience_threshold = 10

    idx_list = [m for m in range(len(x_train))]

    # target
    if attack_type == 'target':
        y_true = np.ones(y_val.shape) * target_class
        stop_condition = 1
        acc_best = 0.
    else:
        y_true = np.copy(y_val)
        stop_condition = -1
        acc_best = 1.
        # stop_condition = 1
        # fr_best = 0.

    for epoch in range(epochs):
        idx_list = shuffle(idx_list)
        for i in range(nb_batch):
            target_idx = idx_list[i * batch_size:min((i + 1) * batch_size, len(x_train))]
            x_batch, y_batch = x_train[target_idx], y_train[target_idx]

            if attack_type == 'target':
                y_batch = np.ones(y_batch.shape) * target_class       

            _, losses = sess.run(
                [train, loss],
                {
                    u_model.inputs[0]: x_batch,
                    alpha: al, lr_ph: lr,
                    target: y_batch,
                    # K.learning_phase(): 0
                }
            )

            if (i + epoch * nb_batch) % 100 == 0:
                # if i % 1 == 0:
                pred = np.argmax(u_model.predict(x_val), -1)
                y_pred = pred.squeeze()
                acc = np.sum(np.where(y_pred == y_true, 1, 0)).astype(np.float64) / len(y_pred)
                norm = np.mean(np.square(sess.run(universal_noise)))
                if attack_type == 'target':
                    print('epoch:{}/{}, batch:{}/{}, acc:{}, norm:{}'.format(epoch + 1, epochs, i + 1, nb_batch,
                                                                             acc, norm))
                else:
                    raw_pred = np.argmax(model.predict(x_val), -1).squeeze()
                    fooling_rate = np.sum(np.where(y_pred != raw_pred, 1, 0)).astype(np.float64) / len(y_pred)
                    print('epoch:{}/{}, batch:{}/{}, acc:{}, fooling rate:{}, norm:{}, loss:{}'.format(epoch + 1,
                                                                 epochs, i + 1, nb_batch, acc, fooling_rate, norm, losses))

                # if acc > threshold_acc and norm > threshold_norm:
                #     a = 5e2
                if stop_condition * acc > stop_condition * acc_best:
                    patience = 0
                    acc_best = acc
                    v = K.eval(universal_noise)
                    if save_path == None:
                        print('update v! but not save.')
                    else:
                        print('best acc:{}, now saving adversarial patch to {}.'.format(acc_best, save_path))
                        # np.savez(noise_filename, v=un_no)
                        np.savez(save_path, v=v)
                else:
                    patience += 1

                if patience == patience_threshold:
                    end = True
                    break

        if end:
            break
    return v

def UAP_target_pre(x, model, model_used, model_path, save_path, noise_limit=0.2, attack_type=None, target_class=None,
               batch_size=None, nb_classes=None, channels=None, samples=None, regular=None):
    # x_train, x_val, y_train, y_val = train_test_split(x, y, shuffle=True, test_size=0.2)
    x_train, x_val = train_test_split(x, shuffle=True, test_size=0.2)
    batch_size = min(batch_size, len(x_train))

    universal_noise = tf.Variable(np.zeros((x_train[0].shape)), dtype=tf.float32)
    temp_universal_noise = tf.expand_dims(universal_noise, 0)
    # print(temp_universal_noise)
    x_input = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    x = Lambda(lambda xx: xx + tf.clip_by_value(temp_universal_noise, -noise_limit, noise_limit))(x_input)

    # Model output
    if model_used == 'EEGNet':
        prediction = old_models.EEGNet_output(nb_classes=nb_classes, Chans=channels, Samples=samples, x_input=x)
    elif model_used == 'DeepConvNet':
        prediction = old_models.DeepConvNet_output(nb_classes=nb_classes, Chans=channels, Samples=samples, x_input=x)
    elif model_used == 'ShallowConvNet':
        prediction = old_models.ShallowConvNet_output(nb_classes=nb_classes, Chans=channels, Samples=samples, x_input=x)
    else:
        raise Exception('No such model:{}'.format(model_used))

    # print(prediction)
    u_model = Model(inputs=x_input, outputs=prediction)
    u_model.load_weights(model_path)
    model.load_weights(model_path)

    y_train = np.argmax(model.predict(x_train, batch_size=batch_size), axis=1).flatten()
    y_val = np.argmax(model.predict(x_val, batch_size=batch_size), axis=1).flatten()


    alpha = tf.placeholder(dtype=tf.float32)
    al = 100
    if regular == 'l1':
        loss = alpha * (tf.reduce_mean(tf.abs(universal_noise)))
        al = 5
    elif regular == 'l2':
        loss = alpha * (tf.reduce_mean(tf.square(universal_noise)))
        al = 100
    elif regular == 'l1+l2':
        loss = alpha * (tf.reduce_mean(10*tf.square(universal_noise) + 0.1*tf.abs(universal_noise)))
        al = 10
    elif regular == None:
        loss = 0
    else:
        raise Exception('no such loss regularization!')
    # loss = alpha * (tf.reduce_mean(tf.square(universal_noise) + tf.abs(universal_noise)))
    # loss = alpha * (tf.reduce_mean(tf.square(universal_noise) + tf.square(universal_noise)))
    # print(loss)
    target = tf.placeholder(dtype=tf.int32, shape=[None, ])
    if attack_type == 'nontarget':
        # loss += K.mean(K.sparse_categorical_crossentropy(target, 1-prediction, from_logits=False))
        loss += -K.mean(K.sparse_categorical_crossentropy(target, prediction, from_logits=False))
    elif attack_type == 'target':
        loss += K.mean(K.sparse_categorical_crossentropy(target, prediction, from_logits=False))
    else:
        raise Exception('no such attack_type!')

    start_vars = set(x.name for x in tf.global_variables())
    lr_ph = tf.placeholder(shape=[], dtype=tf.float32)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr_ph)
    train = optimizer.minimize(loss, var_list=[universal_noise])

    end_vars = tf.global_variables()
    new_vars = [x for x in end_vars if x.name not in start_vars]
    init = tf.variables_initializer(var_list=[universal_noise] + new_vars)

    sess = K.get_session()
    sess.run(init)

    nb_batch = len(x_train) // batch_size


    end = False

    epochs = 500
    lr = 1e-3
    v = np.zeros((x_train[0].shape))

    patience = 0
    patience_threshold = 10

    idx_list = [m for m in range(len(x_train))]

    # target
    if attack_type == 'target':
        y_true = np.ones(y_val.shape) * target_class
        stop_condition = 1
        acc_best = 0.
    else:
        y_true = np.copy(y_val)
        stop_condition = -1
        acc_best = 1.
        # stop_condition = 1
        # fr_best = 0.

    for epoch in range(epochs):
        idx_list = shuffle(idx_list)
        for i in range(nb_batch):
            target_idx = idx_list[i * batch_size:min((i + 1) * batch_size, len(x_train))]
            x_batch, y_batch = x_train[target_idx], y_train[target_idx]

            if attack_type == 'target':
                y_batch = np.ones(y_batch.shape) * target_class

            _, losses = sess.run(
                [train, loss],
                {
                    u_model.inputs[0]: x_batch,
                    alpha: al, lr_ph: lr,
                    target: y_batch,
                    # K.learning_phase(): 0
                }
            )

            if (i + epoch * nb_batch) % 100 == 0:
                # if i % 1 == 0:
                pred = np.argmax(u_model.predict(x_val), -1)
                y_pred = pred.squeeze()
                acc = np.sum(np.where(y_pred == y_true, 1, 0)).astype(np.float64) / len(y_pred)
                norm = np.mean(np.square(sess.run(universal_noise)))
                if attack_type == 'target':
                    print('epoch:{}/{}, batch:{}/{}, acc:{}, norm:{}'.format(epoch + 1, epochs, i + 1, nb_batch,
                                                                             acc, norm))
                else:
                    raw_pred = np.argmax(model.predict(x_val), -1).squeeze()
                    fooling_rate = np.sum(np.where(y_pred != raw_pred, 1, 0)).astype(np.float64) / len(y_pred)
                    print('epoch:{}/{}, batch:{}/{}, acc:{}, fooling rate:{}, norm:{}, loss:{}'.format(epoch + 1,
                                                                 epochs, i + 1, nb_batch, acc, fooling_rate, norm, losses))

                # if acc > threshold_acc and norm > threshold_norm:
                #     a = 5e2
                if stop_condition * acc > stop_condition * acc_best:
                    patience = 0
                    acc_best = acc
                    v = K.eval(universal_noise)
                    if save_path == None:
                        print('update v! but not save.')
                    else:
                        print('best acc:{}, now saving adversarial patch to {}.'.format(acc_best, save_path))
                        # np.savez(noise_filename, v=un_no)
                        np.savez(save_path, v=v)
                else:
                    patience += 1
                    if acc == 1:
                        print('best acc:{}, now saving adversarial patch to {}.'.format(acc_best, save_path))
                        np.savez(save_path, v=v)

                if patience == patience_threshold:
                    end = True
                    break

        if end:
            break
    return v



def UAP_target_pre_wide(x, model, model_used, model_path, save_path, noise_limit=0.2, attack_type=None, target_class=None,
               batch_size=None, nb_classes=None, channels=None, samples=None, regular=None, wide=False):
    # x_train, x_val, y_train, y_val = train_test_split(x, y, shuffle=True, test_size=0.2)
    x_train, x_val = train_test_split(x, shuffle=True, test_size=0.2)
    batch_size = min(batch_size, len(x_train))

    universal_noise = tf.Variable(np.zeros((x_train[0].shape)), dtype=tf.float32)
    temp_universal_noise = tf.expand_dims(universal_noise, 0)
    # print(temp_universal_noise)
    x_input = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    x = Lambda(lambda xx: xx + tf.clip_by_value(temp_universal_noise, -noise_limit, noise_limit))(x_input)

    # Model output
    if wide == False:
        if model_used == 'EEGNet':
            prediction = old_models.EEGNet_output(nb_classes=nb_classes, Chans=channels, Samples=samples, x_input=x)
        elif model_used == 'DeepConvNet':
            prediction = old_models.DeepConvNet_output(nb_classes=nb_classes, Chans=channels, Samples=samples, x_input=x)
        elif model_used == 'ShallowConvNet':
            prediction = old_models.ShallowConvNet_output(nb_classes=nb_classes, Chans=channels, Samples=samples, x_input=x)
        else:
            raise Exception('No such model:{}'.format(model_used))
    else:
        if model_used == 'EEGNet':
            prediction = old_models.EEGNet_output(nb_classes=nb_classes, Chans=channels, Samples=samples, x_input=x, numFilters=32)
        elif model_used == 'DeepConvNet':
            prediction = old_models.DeepConvNet_output_wide(nb_classes=nb_classes, Chans=channels, Samples=samples, x_input=x)
        elif model_used == 'ShallowConvNet':
            prediction = old_models.ShallowConvNet_output_wide(nb_classes=nb_classes, Chans=channels, Samples=samples, x_input=x)
        else:
            raise Exception('No such model:{}'.format(model_used))

    # print(prediction)
    u_model = Model(inputs=x_input, outputs=prediction)
    u_model.load_weights(model_path)
    model.load_weights(model_path)

    y_train = np.argmax(model.predict(x_train, batch_size=batch_size), axis=1).flatten()
    y_val = np.argmax(model.predict(x_val, batch_size=batch_size), axis=1).flatten()


    alpha = tf.placeholder(dtype=tf.float32)
    al = 100
    if regular == 'l1':
        loss = alpha * (tf.reduce_mean(tf.abs(universal_noise)))
        al = 5
    elif regular == 'l2':
        loss = alpha * (tf.reduce_mean(tf.square(universal_noise)))
        al = 100
    elif regular == 'l1+l2':
        loss = alpha * (tf.reduce_mean(10*tf.square(universal_noise) + 0.1*tf.abs(universal_noise)))
        al = 10
    elif regular == None:
        loss = 0
    else:
        raise Exception('no such loss regularization!')
    # loss = alpha * (tf.reduce_mean(tf.square(universal_noise) + tf.abs(universal_noise)))
    # loss = alpha * (tf.reduce_mean(tf.square(universal_noise) + tf.square(universal_noise)))
    # print(loss)
    target = tf.placeholder(dtype=tf.int32, shape=[None, ])
    if attack_type == 'nontarget':
        # loss += K.mean(K.sparse_categorical_crossentropy(target, 1-prediction, from_logits=False))
        loss += -K.mean(K.sparse_categorical_crossentropy(target, prediction, from_logits=False))
    elif attack_type == 'target':
        loss += K.mean(K.sparse_categorical_crossentropy(target, prediction, from_logits=False))
    else:
        raise Exception('no such attack_type!')

    start_vars = set(x.name for x in tf.global_variables())
    lr_ph = tf.placeholder(shape=[], dtype=tf.float32)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr_ph)
    train = optimizer.minimize(loss, var_list=[universal_noise])

    end_vars = tf.global_variables()
    new_vars = [x for x in end_vars if x.name not in start_vars]
    init = tf.variables_initializer(var_list=[universal_noise] + new_vars)

    sess = K.get_session()
    sess.run(init)

    nb_batch = len(x_train) // batch_size


    end = False

    epochs = 500
    lr = 1e-3
    v = np.zeros((x_train[0].shape))

    patience = 0
    patience_threshold = 10

    idx_list = [m for m in range(len(x_train))]

    # target
    if attack_type == 'target':
        y_true = np.ones(y_val.shape) * target_class
        stop_condition = 1
        acc_best = 0.
    else:
        y_true = np.copy(y_val)
        stop_condition = -1
        acc_best = 1.
        # stop_condition = 1
        # fr_best = 0.

    for epoch in range(epochs):
        idx_list = shuffle(idx_list)
        for i in range(nb_batch):
            target_idx = idx_list[i * batch_size:min((i + 1) * batch_size, len(x_train))]
            x_batch, y_batch = x_train[target_idx], y_train[target_idx]

            if attack_type == 'target':
                y_batch = np.ones(y_batch.shape) * target_class

            _, losses = sess.run(
                [train, loss],
                {
                    u_model.inputs[0]: x_batch,
                    alpha: al, lr_ph: lr,
                    target: y_batch,
                    # K.learning_phase(): 0
                }
            )

            if (i + epoch * nb_batch) % 100 == 0:
                # if i % 1 == 0:
                pred = np.argmax(u_model.predict(x_val), -1)
                y_pred = pred.squeeze()
                acc = np.sum(np.where(y_pred == y_true, 1, 0)).astype(np.float64) / len(y_pred)
                norm = np.mean(np.square(sess.run(universal_noise)))
                if attack_type == 'target':
                    print('epoch:{}/{}, batch:{}/{}, acc:{}, norm:{}'.format(epoch + 1, epochs, i + 1, nb_batch,
                                                                             acc, norm))
                else:
                    raw_pred = np.argmax(model.predict(x_val), -1).squeeze()
                    fooling_rate = np.sum(np.where(y_pred != raw_pred, 1, 0)).astype(np.float64) / len(y_pred)
                    print('epoch:{}/{}, batch:{}/{}, acc:{}, fooling rate:{}, norm:{}, loss:{}'.format(epoch + 1,
                                                                 epochs, i + 1, nb_batch, acc, fooling_rate, norm, losses))

                # if acc > threshold_acc and norm > threshold_norm:
                #     a = 5e2
                if stop_condition * acc > stop_condition * acc_best:
                    patience = 0
                    acc_best = acc
                    v = K.eval(universal_noise)
                    if save_path == None:
                        print('update v! but not save.')
                    else:
                        print('best acc:{}, now saving adversarial patch to {}.'.format(acc_best, save_path))
                        # np.savez(noise_filename, v=un_no)
                        np.savez(save_path, v=v)
                else:
                    patience += 1
                    if acc == 1:
                        print('best acc:{}, now saving adversarial patch to {}.'.format(acc_best, save_path))
                        np.savez(save_path, v=v)

                if patience == patience_threshold:
                    end = True
                    break

        if end:
            break
    return v


