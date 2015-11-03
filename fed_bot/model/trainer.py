__author__ = 'allentran'

import random
import json

from spacy.en import English
import allen_utils
import numpy as np

import lstm

logger = allen_utils.get_logger(__name__)

def load_data(data_path, n_rates=3):

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

    def transform(obs):

        max_length = max([len(sentence) for sentence in obs['sentences']])
        n_sentences = len(obs['sentences'])

        target_rates = calc_target_rates(obs['rates'], days=['30', '90', '180'])
        word_vectors = np.zeros((max_length, n_sentences))
        max_mask = -1e5 * np.ones((max_length, n_sentences))
        doc_types = 1 if obs['is_minutes'] else 0

        for sent_idx, sentence in enumerate(obs['sentences']):
            length = len(sentence)
            word_vectors[0:length, sent_idx] = sentence
            max_mask[0:length, sent_idx] = 1

        return dict(
            word_vectors=word_vectors.astype('int32'),
            max_mask=max_mask.astype('float32'),
            rates=target_rates.astype('float32'),
            doc_types=np.int32(doc_types),
            regimes=np.int32(obs['regime']),
        )

    with open(data_path, 'r') as json_file:
        paired_data = json.load(json_file)

    paired_data = [obs for obs in paired_data if '0' in obs['rates'] and len(obs['rates'].keys()) > 1]
    random.shuffle(paired_data)
    for data in paired_data:
        data['sentences'] = np.array(data['sentences'])

    transformed_data = []
    for obs in paired_data:
        transformed_data.append(transform(obs))

    return transformed_data

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

    data = load_data(data_path)

    word_embeddings = build_wordvectors(vocab_path)

    test_idx = int(round(len(data) * test_frac))
    test_data = data[:test_idx]
    train_data = data[test_idx:]

    model = lstm.FedLSTM(
        hidden_sizes=[500, 400, 300, 100],
        l2_penalty=1e-4,
        n_mixtures=2,
        truncate=100,
        vocab_size=word_embeddings.shape[0],
        word_vectors=word_embeddings
    )

    for epoch_idx in xrange(n_epochs):
        train_cost = 0
        random.shuffle(train_data)
        for obs in train_data:
            cost = model.get_cost_and_update(
                obs['word_vectors'],
                obs['rates'],
                obs['max_mask'],
                obs['regimes'],
                obs['doc_types']
            )
            import IPython
            IPython.embed()
            assert False
            cost /= obs['word_vectors'].shape[1]
            train_cost += cost

        if epoch_idx % 5 == 0:
            test_cost = 0
            for obs in test_data:
                cost = model.get_cost(
                        obs['word_vectors'],
                        obs['rates'],
                        obs['max_mask'],
                        obs['regimes'],
                        obs['doc_types']
                )
                cost /= obs['word_vectors'].shape[1]
                test_cost += cost
            test_cost /= len(test_data)
            train_cost /= len(train_data)
            logger.info('train_cost=%s, test_cost=%s after %s epochs', train_cost, test_cost, epoch_idx)

if __name__ == "__main__":
    train('data/paired_data.json', 'data/dictionary.json')
