import pickle
import numpy as np
import os
import multiprocessing
from itertools import product


def ensemble(weights, logits, label):
    assert isinstance(weights, (list, tuple)) and isinstance(logits, (list, tuple))
    res = np.zeros_like(logits[0])

    for weight, logit in zip(weights, logits):
        res += weight * logit

    acc = (res.argmax(axis=-1) == label).sum() / len(res)
    return acc


def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return np.array(list(data.values()))


def process_chunk(weight_combinations_chunk, predictions, label):
    local_max_accuracy = 0
    best_weights = None
    for i, weights in enumerate(weight_combinations_chunk):
        accuracy = ensemble(weights, predictions, label)
        if accuracy > local_max_accuracy:
            print(f"{i + 1}/{len(weight_combinations_chunk)}: Current max accuracy: {accuracy} with {weights}")
            local_max_accuracy = accuracy
            best_weights = weights
    return local_max_accuracy, best_weights


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


if __name__ == '__main__':
    npz_data = np.load('/mnt/ssd/datasets/NTU_120/NTU120_CSub.npz')
    label = np.where(npz_data['y_test'] > 0)[1]

    root_path = "logits/ntu120/csub"
    files = ["k1.pkl", "k2.pkl", "k8.pkl", "k1_motion.pkl", "k2_motion.pkl", "k8_motion.pkl"]
    files = [os.path.join(root_path, file) for file in files]
    logits = [read_pickle(file) for file in files]

    k1_range = np.arange(0.5, 0.7, 0.1)
    k2_range = np.arange(0.3, 0.5, 0.1)
    k8_range = np.arange(0.2, 0.4, 0.1)
    k1_motion = np.arange(0., 0.2, 0.1)
    k2_motion = np.arange(0.2, 0.4, 0.1)
    k8_motion = np.arange(0., 0.2, 0.1)

    all_weights = [k1_range, k2_range, k8_range, k1_motion, k2_motion, k8_motion]

    # Generate all combinations
    all_combinations = list(product(*all_weights))

    num_processes = multiprocessing.cpu_count()
    chunks = chunkify(all_combinations, num_processes)

    # Using Pool to handle process creation and joining automatically
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            process_chunk, [(chunk, logits, label) for chunk in chunks])

    # Determine global maximum from local maxima
    global_max_accuracy, best_weights_global = max(results, key=lambda x: x[0])
    print(f"Global max accuracy: {global_max_accuracy} with weights: {best_weights_global}")