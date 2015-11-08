__author__ = 'allentran'

import random
import json

from spacy.en import English
import allen_utils
import numpy as np

import lstm

logger = allen_utils.get_logger(__name__)

def load_data(data_path, n_rates=3, batch_size=32):

    def calc_target_rates(rates, days):

        current = rates['0']
        mean_diff = np.mean([rates[day] - current for day in days if day != '0' and day in rates])
        target_rates = []
        for day in days:
            if day in rates:
                target_rates.append(rates[day] - current)
            else:
                target_rates.append(mean_diff)

        return np.array(target_rates) / 100.0

    def merge(data_to_batch, mask_value=-1e5):

        max_n_sentences = max([len(obs['sentences']) for obs in data_to_batch])
        max_length = max([len(sentence) for obs in data_to_batch for sentence in obs['sentences']])
        batch_size = len(data_to_batch)

        target_rates = np.zeros((batch_size, n_rates))
        word_vectors = np.zeros((max_length, max_n_sentences, batch_size))
        max_mask = np.zeros((max_length, max_n_sentences, batch_size))
        regimes = np.zeros(batch_size)
        doc_types = np.zeros(batch_size)
        for data_idx in xrange(batch_size):
            obs = data_to_batch[data_idx]
            sentences = obs['sentences']
            for sentence_number, sentence in enumerate(sentences):
                length = len(sentence)
                word_vectors[0:length, sentence_number, data_idx] = sentence
                max_mask[0:length, sentence_number, data_idx] = mask_value
            target_rates[data_idx, :] = calc_target_rates(obs['rates'], days=['30', '90', '180'])
            if obs['is_minutes']:
                doc_types[data_idx] = 1
            regimes[data_idx] = obs['regime']

        return dict(
            word_vectors=word_vectors.astype('int32'),
            max_mask=max_mask.astype('float32'),
            rates=target_rates.astype('float32'),
            doc_types=np.array(doc_types).astype('int32'),
            regimes=np.array(regimes).astype('int32'),
        )

    with open(data_path, 'r') as json_file:
        paired_data = json.load(json_file)

    paired_data = [obs for obs in paired_data if '0' in obs['rates'] and len(obs['rates'].keys()) > 1]
    random.shuffle(paired_data)

    batched_data = []
    paired_data = sorted(paired_data, key=lambda obs: np.hstack(obs['sentences']).shape[0])

    for start_idx in xrange(0, len(paired_data), batch_size):
        end_idx = min([start_idx + batch_size, len(paired_data)])
        batched_data.append(merge(paired_data[start_idx: end_idx]))

    return batched_data

def build_wordvectors(vocab_dict_path):

    with open(vocab_dict_path, 'r') as json_f:
        vocab = json.load(json_f)

    word_vectors = np.random.normal(size=(len(vocab), 300))
    nlp = English()

    for token, position in vocab.iteritems():
        try:
            vector = nlp(unicode(token))[0].repvec
            if len(vector[vector != 0]) > 0:
                word_vectors[position, :] = vector
        except ValueError:
            logger.info("No init vector for %s", token)


    return word_vectors.astype('float32')


def train(data_path, vocab_path):

    n_epochs = 200
    test_frac = 0.2

    data = load_data(data_path, batch_size=3)

    word_embeddings = build_wordvectors(vocab_path)

    test_idx = int(round(len(data) * test_frac))
    test_data = data[:test_idx]
    train_data = data[test_idx:]

    model = lstm.FedLSTM(
        hidden_sizes=[256, 128, 128, 64, 32],
        l2_penalty=1e-4,
        n_mixtures=2,
        vocab_size=word_embeddings.shape[0],
        word_vectors=word_embeddings,
        truncate=100
    )

    for epoch_idx in xrange(n_epochs):
        train_cost = 0
        random.shuffle(train_data)
        for obs in train_data:
            train_cost += model.get_cost_and_update(
                obs['word_vectors'],
                obs['rates'],
                obs['max_mask'],
                obs['regimes'],
                obs['doc_types']
            )

        if epoch_idx % 5 == 0:
            test_cost = 0
            for obs in test_data:
                test_cost += model.get_cost(
                        obs['word_vectors'],
                        obs['rates'],
                        obs['max_mask'],
                        obs['regimes'],
                        obs['doc_types']
                )
            test_cost /= len(test_data)
            train_cost /= len(train_data)
            logger.info('train_cost=%s, test_cost=%s after %s epochs', train_cost, test_cost, epoch_idx)

if __name__ == "__main__":
    train('data/paired_data.json', 'data/dictionary.json')
