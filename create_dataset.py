import argparse
import pickle
from dataset_utility import load_numpy
import numpy as np
import math as mt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('positions', help='Number of different positions', type=int)
    parser.add_argument('num_samples', help='Max number of samples to use', type=int)
    parser.add_argument('prefix', help='Prefix')
    parser.add_argument('save_folder', help='Folder where to save the dataset')
    parser.add_argument('rand', help='Select random indices or sub-sample the trace: type `random` or `sampling`')
    args = parser.parse_args()

    input_dir = args.dir + '/vtilde_matrices/'
    num_pos = args.positions
    max_num_samples = args.num_samples
    prefix = args.prefix
    save_folder = args.save_folder

    module_IDs = ['ee', 'd6', '28', '8b', 'c2', '65', 'b3', '2d', 'e6',
                  'c6', '88', '6c', '0d', 'a9', '49', '12', '04', 'bb']

    for idx in range(len(module_IDs)):
        for pos in range(num_pos):
            if pos == 10:
                pos = 'A'

            name_file = module_IDs[idx] + prefix + str(pos)
            name_v_mat = input_dir + name_file + '.npy'
            try:
                v_matrix = np.load(name_v_mat)
            except FileNotFoundError:
                print('file ', name_v_mat, ' doesn\'t exist')
                continue

            # Select samples to balance the dataset
            num_sampl = v_matrix.shape[0]
            if num_sampl > max_num_samples:
                if args.rand == 'random':
                    selected_idxs = np.random.randint(0, num_sampl, max_num_samples)
                    selected_idxs = np.sort(selected_idxs)
                elif args.rand == 'sampling':
                    selected_idxs = np.arange(0, num_sampl, mt.ceil(num_sampl/max_num_samples))
                v_matrix = v_matrix[selected_idxs, ...]

            v_matrix = np.moveaxis(v_matrix, [3], [1])
            name_file = save_folder + '/' + name_file + '.txt'
            with open(name_file, "wb") as fp:
                pickle.dump(v_matrix, fp)
