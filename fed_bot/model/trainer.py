__author__ = 'allentran'

import random
import logging
import json

import numpy as np
import allen

import lstm

def batch_and_load_data(data_path, batch_size=10, n_rates=3):

    def calc_target_rates(rates, days):

        current = rates['0']
        mean_diff = np.mean([rates[day] - current for day in days if day != '0' and day in rates])
        target_rates = []
        for day in days:
            if day in rates:
                target_rates.append(rates[day] - current)
            else:
                target_rates.append(mean_diff)

        return np.array(target_rates)

    def merge(data_to_batch):

        max_length = max([len(obs['vectors']) for obs in data_to_batch])
        batch_size = len(data_to_batch)

        target_rates = np.zeros((batch_size, n_rates))
        word_vectors = np.zeros((max_length, batch_size, data_to_batch[0]['vectors'].shape[1]))
        last_indexes = []
        for data_idx in xrange(batch_size):
            vectors = data_to_batch[data_idx]['vectors']
            length = len(vectors)
            word_vectors[0:length, data_idx, :] = vectors
            last_indexes.append(length - 1)
            target_rates[data_idx, :] = calc_target_rates(data_to_batch[data_idx]['rates'], days=['30', '90', '180'])

        return dict(
            word_vectors=word_vectors.astype('float32'),
            last_indexes=np.array(last_indexes).astype('int32'),
            rates=target_rates.astype('float32')
        )

    with open(data_path, 'r') as json_file:
        paired_data = json.load(json_file)

    paired_data = [obs for obs in paired_data if '0' in obs['rates'] and len(obs['rates'].keys()) > 1]
    for data in paired_data:
        data['vectors'] = np.array(data['vectors'])

    batched_data = []
    paired_data = sorted(paired_data, key=lambda obs: len(obs['vectors']))

    for start_idx in xrange(0, len(paired_data), batch_size):
        end_idx = min([start_idx + batch_size, len(paired_data)])
        batched_data.append(merge(paired_data[start_idx: end_idx]))

    return batched_data


def train(data_path):

    n_epochs = 200
    batch_size = 10
    test_frac = 0.2

    batched_data = batch_and_load_data(data_path, batch_size=batch_size)
    random.shuffle(batched_data)

    test_idx = int(round(len(batched_data) * test_frac))
    test_data = batched_data[:test_idx]
    train_data = batched_data[test_idx:]

    model = lstm.FedLSTM(
        hidden_sizes=[500, 400, 300, 100]
    )

    for obs in train_data:
        print  model.get_cost_and_update(
            obs['word_vectors'],
            obs['rates'],
            obs['last_indexes']
        )

    import IPython
    IPython.embed()

if __name__ == "__main__":
    allen.setup()
    train('data/statements/paired_data.json')
