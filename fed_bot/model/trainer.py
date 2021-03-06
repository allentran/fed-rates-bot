__author__ = 'allentran'

import random
import json

from spacy.en import English
import allen_utils
import numpy as np
import joblib

import lstm, lstm_lasagne

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

    def merge(data_to_batch, mask_value=0):

        max_n_sentences = max([len(obs['sentences']) for obs in data_to_batch])
        max_length = max([len(sentence) for obs in data_to_batch for sentence in obs['sentences']])
        batch_size = len(data_to_batch)

        target_rates = np.zeros((batch_size, n_rates))
        word_vectors = np.zeros((max_length, max_n_sentences, batch_size))
        max_mask = np.ones((max_length, max_n_sentences, batch_size))
        last_sentence = np.ones(batch_size)
        last_word_in_sentence = np.ones((max_n_sentences, batch_size))
        regimes = np.zeros(batch_size)
        doc_types = np.zeros(batch_size)
        for data_idx in xrange(batch_size):
            obs = data_to_batch[data_idx]
            sentences = obs['sentences']
            last_sentence[data_idx] = len(sentences) - 1
            for sentence_number, sentence in enumerate(sentences):
                length = len(sentence)
                last_word_in_sentence[sentence_number, data_idx] = length - 1
                word_vectors[0:length, sentence_number, data_idx] = sentence
                max_mask[length:, sentence_number, data_idx] = mask_value
            target_rates[data_idx, :] = calc_target_rates(obs['rates'], days=['30', '90', '180'])
            if obs['is_minutes']:
                doc_types[data_idx] = 1
            regimes[data_idx] = obs['regime']

        return dict(
            word_vectors=word_vectors.astype('int32'),
            last_sentence=last_sentence.astype('int32'),
            last_word_in_sentence=last_word_in_sentence.astype('int32'),
            max_mask=max_mask.astype('float32'),
            rates=target_rates.astype('float32'),
            doc_types=np.array(doc_types).astype('int32'),
            regimes=np.array(regimes).astype('int32'),
        )

    def get_shape(list_of_lists):
        max_inside_list = max([len(l) for l in list_of_lists])
        return max_inside_list, len(list_of_lists)

    with open(data_path, 'r') as json_file:
        paired_data = json.load(json_file)

    paired_data = [obs for obs in paired_data if '0' in obs['rates'] and len(obs['rates'].keys()) > 1]
    random.shuffle(paired_data)

    batched_data = []
    paired_data = sorted(paired_data, key=lambda obs: get_shape(obs['sentences']))

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

def evaluate(model, data):
    outputs = []
    for obs in data:
        model_output = model.get_output(obs['word_vectors'], obs['max_mask'], obs['regimes'], obs['doc_types'])
        outputs.append(
            {'rates': obs['rates'], 'model_output': model_output}
        )

    return outputs

def eval_lasagne_on_test(model, test_data):

    test_outputs = []

    for obs in test_data:
        priors, means, stds = model.get_output(
            np.swapaxes(obs['word_vectors'], 0, 2),
            np.swapaxes(obs['last_word_in_sentence'], 0, 1),
            obs['last_sentence'],
            obs['regimes'],
            obs['doc_types']
        )
        batch_size, target_size, n_mixtures = means.shape
        for obs_idx in xrange(batch_size):
            test_output = dict()
            test_output['rates'] = obs['rates'][obs_idx, :]
            test_output['priors'] = priors[obs_idx, :]
            test_output['means'] = means[obs_idx, :, :]
            test_output['stds'] = stds[obs_idx, :]
            test_outputs.append(test_output)

    return test_outputs

def train_lasagne(data_path, vocab_path):

    n_epochs = 500
    test_frac = 0.2

    data = load_data(data_path, batch_size=4)

    word_embeddings = build_wordvectors(vocab_path)

    test_idx = int(round(len(data) * test_frac))
    test_data = data[:test_idx]
    train_data = data[test_idx:]

    model = lstm_lasagne.FedLSTMLasagne(
        hidden_size=32,
        lstm_size=64,
        n_mixtures=2,
        n_regimes=6,
        regime_size=5,
        doc_size=5,
        n_words=word_embeddings.shape[0],
        word_size=word_embeddings.shape[1],
        init_word_vectors=word_embeddings
    )


    for epoch_idx in xrange(n_epochs):
        train_cost = 0
        random.shuffle(train_data)
        for obs in train_data:
            train_cost += model.train(
                obs['rates'],
                np.swapaxes(obs['word_vectors'], 0, 2),
                np.swapaxes(obs['last_word_in_sentence'], 0, 1),
                obs['last_sentence'],
                obs['regimes'],
                obs['doc_types']
            )

        if epoch_idx % 5 == 0:
            test_cost = 0
            for obs in test_data:
                test_cost += model.get_cost(
                    obs['rates'],
                    np.swapaxes(obs['word_vectors'], 0, 2),
                    np.swapaxes(obs['last_word_in_sentence'], 0, 1),
                    obs['last_sentence'],
                    obs['regimes'],
                    obs['doc_types']
                )
            test_cost /= len(test_data)
            train_cost /= len(train_data)
            logger.info('train_cost=%s, test_cost=%s after %s epochs', train_cost, test_cost, epoch_idx)

    test_results = eval_lasagne_on_test(model, train_data)
    joblib.dump(test_results, 'results.pkl')


def train_theano(data_path, vocab_path):

    n_epochs = 1
    test_frac = 0.2

    data = load_data(data_path, batch_size=2)

    word_embeddings = build_wordvectors(vocab_path)

    test_idx = int(round(len(data) * test_frac))
    test_data = data[:test_idx]
    train_data = data[test_idx:]

    model = lstm.FedLSTM(
        hidden_size=32,
        lstm_size=64,
        l2_penalty=1e-4,
        n_mixtures=1,
        vocab_size=word_embeddings.shape[0],
        word_vectors=word_embeddings,
        truncate=200
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
            output = model.get_output(
                obs['word_vectors'],
                obs['max_mask'],
                obs['regimes'],
                obs['doc_types']
            )
            import IPython
            IPython.embed()
            assert False

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

    test_output = evaluate(model, test_data)
    import IPython
    IPython.embed()

if __name__ == "__main__":
    train_lasagne('data/paired_data.json', 'data/dictionary.json')
