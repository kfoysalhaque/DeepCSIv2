import argparse
import os
import pickle

from dataset_utility import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from network_utility import *  # provides ConvNormalization + tf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('positions', help='Number of different positions', type=int)
    parser.add_argument('model_name', help='Model name')
    parser.add_argument('M', help='Number of transmitting antennas', type=int)
    parser.add_argument('N', help='Number of receiving antennas', type=int)
    parser.add_argument('tx_antennas', help='Indices of TX antennas to consider (comma separated)')
    parser.add_argument('rx_antennas', help='Indices of RX antennas to consider (comma separated)')
    parser.add_argument('bandwidth', help='Bandwidth in [MHz] to select the subcarriers, can be 10, 20, 40, 80, 160',
                        type=int)
    parser.add_argument('model_type', help='convolutional or attention')
    parser.add_argument('prefix', help='Prefix')
    parser.add_argument('scenario', help='Scenario considered, in {S1, S2, S3, S4, S5}')
    args = parser.parse_args()

    prefix = args.prefix
    model_name = args.model_name
    model_type = args.model_type

    # ---- Mirror training logic: hyper-selection modifies model_name used in saved filename ----
    if model_type[:29] == 'convolutional_hyper_selection':
        hyper_parameters = model_type[30:]
        model_name = model_name + hyper_parameters + '_'
    elif model_type[:25] == 'attention_hyper_selection':
        hyper_parameters = model_type[26:]
        model_name = model_name + hyper_parameters + '_'
    # -----------------------------------------------------------------------------

    # ---- Clean only caches that match this model_name prefix (same spirit as training) ----
    if os.path.exists('./cache_files/'):
        for f in os.listdir('./cache_files/'):
            if f.startswith(model_name):
                try:
                    os.remove(os.path.join('./cache_files/', f))
                except IsADirectoryError:
                    pass
    # -----------------------------------------------------------------------------

    scenario = args.scenario
    if scenario == 'S1':
        pos_train_val = [1, 2, 3]
        train_fraction = [0, 0.64]
        val_fraction = [0.64, 0.8]
        pos_test = [1, 2, 3]
        test_fraction = [0.8, 1]
    elif scenario == 'S2':
        pos_train_val = [1]
        train_fraction = [0, 0.8]
        val_fraction = [0.8, 1]
        pos_test = [1]
        test_fraction = [0, 1]
    elif scenario == 'S3':
        pos_train_val = [2]
        train_fraction = [0, 0.8]
        val_fraction = [0.8, 1]
        pos_test = [2]
        test_fraction = [0, 1]
    elif scenario == 'S4':
        pos_train_val = [3]
        train_fraction = [0, 0.8]
        val_fraction = [0.8, 1]
        pos_test = [3]
        test_fraction = [0, 1]
    elif scenario == 'S5':
        pos_train_val = [1, 2]
        train_fraction = [0, 0.8]
        val_fraction = [0.8, 1]
        pos_test = [3]
        test_fraction = [0, 1]
    else:
        raise ValueError('Scenario must be one of {S1,S2,S3,S4,S5}')

    # Positions and device IDs (same as training)
    num_pos = args.positions  # kept for compatibility; not explicitly used here
    extension = '.txt'
    module_IDs = ['ee', 'd6', '28', '8b', 'c2', '65', 'b3', '2d', 'e6',
                  'c6', '88', '6c', '0d', 'a9', '49', '12', '04', 'bb']
    labels_IDs = np.arange(0, len(module_IDs))

    # TX and RX antennas selection
    M = args.M
    N = args.N

    tx_antennas_list = []
    for a in args.tx_antennas.split(','):
        a = int(a)
        if a >= M:
            raise ValueError('error in the tx_antennas input arg')
        tx_antennas_list.append(a)

    rx_antennas_list = []
    for a in args.rx_antennas.split(','):
        a = int(a)
        if a >= N:
            raise ValueError('error in the rx_antennas input arg')
        rx_antennas_list.append(a)

    # Subcarriers selection (same as training)
    selected_subcarriers_idxs = None  # default, i.e., 160 MHz
    bandwidth = args.bandwidth
    if bandwidth == 160:
        num_selected_subcarriers = 500
    elif bandwidth == 80:
        num_selected_subcarriers = 250
        selected_subcarriers_idxs = np.arange(0, 250)
    elif bandwidth == 40:
        num_selected_subcarriers = 125
        selected_subcarriers_idxs = np.arange(0, 125)
    elif bandwidth == 20:
        num_selected_subcarriers = 62
        selected_subcarriers_idxs = np.arange(0, 62)
    elif bandwidth == 10:
        num_selected_subcarriers = 30
        selected_subcarriers_idxs = np.arange(0, 30)
    else:
        raise ValueError("bandwidth must be one of {10,20,40,80,160}")

    # Build TEST set filenames (same pos_id logic as training, including 'A')
    input_dir = args.dir + '/'

    name_files_test = []
    labels_test = []
    for mod_label, mod_ID in enumerate(module_IDs):
        for pos in pos_test:
            pos_id = pos - 1
            if pos_id == 10:
                pos_id = 'A'
            name_file = input_dir + mod_ID + prefix + str(pos_id) + extension
            name_files_test.append(name_file)
            labels_test.append(mod_label)

    batch_size = 32
    name_cache_test = './cache_files/' + model_name + 'cache_test'
    dataset_test, num_samples_test, labels_complete_test = create_dataset(
        name_files_test, labels_test, batch_size,
        M, tx_antennas_list, N, rx_antennas_list,
        shuffle=False, cache_file=name_cache_test,
        prefetch=True, repeat=True,
        start_fraction=test_fraction[0],
        end_fraction=test_fraction[1],
        selected_subcarriers_idxs=selected_subcarriers_idxs
    )

    # Input shape (same as training)
    IQ_dimension = 2
    N_considered = len(rx_antennas_list)
    M_considered = len(tx_antennas_list)
    input_shape = (N_considered, num_selected_subcarriers, M_considered * IQ_dimension)
    if (M - 1) in tx_antennas_list:
        input_shape = (N_considered, num_selected_subcarriers, M_considered * IQ_dimension - 1)
    print(input_shape)

    num_classes = len(module_IDs)

    # Build model filename exactly like training
    name_save = model_name + \
                'IDTX' + str(tx_antennas_list) + \
                '_RX' + str(rx_antennas_list) + \
                '_posTRAIN' + str(pos_train_val) + \
                '_posTEST' + str(pos_test) + \
                '_bandwidth' + str(bandwidth) + \
                '_MOD' + args.model_type

    # ---- UPDATED: Keras 3 full-model format ----
    name_model = './network_models/' + name_save + 'network.keras'
    # -------------------------------------------

    custom_objects = {'ConvNormalization': ConvNormalization}
    model_net = tf.keras.models.load_model(name_model, custom_objects=custom_objects)

    # TEST
    test_steps_per_epoch = int(np.ceil(num_samples_test / batch_size))
    prediction_test = model_net.predict(dataset_test, steps=test_steps_per_epoch)[:len(labels_complete_test)]
    labels_pred_test = np.argmax(prediction_test, axis=1)

    labels_complete_test_array = np.asarray(labels_complete_test)
    conf_matrix_test = confusion_matrix(
        labels_complete_test_array, labels_pred_test,
        labels=labels_IDs,
        normalize='true'
    )

    precision_test, recall_test, fscore_test, _ = precision_recall_fscore_support(
        labels_complete_test_array, labels_pred_test,
        labels=labels_IDs
    )

    accuracy_test = accuracy_score(labels_complete_test_array, labels_pred_test)
    print('Accuracy test: %.5f' % accuracy_test)

    metrics_dict = {
        'conf_matrix_test': conf_matrix_test,
        'accuracy_test': accuracy_test,
        'precision_test': precision_test,
        'recall_test': recall_test,
        'fscore_test': fscore_test
    }

    name_file = './outputs/' + name_save + '.txt'
    with open(name_file, "wb") as fp:
        pickle.dump(metrics_dict, fp)

    # Optional: print confusion matrix in LaTeX-friendly coordinates
    string_latex = ''
    for row in range(len(module_IDs)):
        for col in range(len(module_IDs)):
            string_latex += f'({row},{col}) [{conf_matrix_test[row, col]}] '
        string_latex += '\n\n'
    print(string_latex)
