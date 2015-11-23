__author__ = 'allentran'

import numpy as np

from ..scraper import Scraper
from ..model import lstm, lstm_lasagne

def model_test():

    n_sentences = 6
    T = 21
    n_batch = 7
    vocab_size=55
    word_vector_size=111

    test_ob = dict(
        word_vectors=np.random.randint(0, vocab_size, size=(T, n_sentences, n_batch)).astype('int32'),
        rates=np.ones((n_batch, 3)).astype('float32'),
        max_mask=np.ones((T, n_sentences, n_batch)).astype('float32'),
        regimes=np.ones(n_batch).astype('int32'),
        doc_types=np.ones(n_batch).astype('int32')
    )

    word_embeddings = np.random.normal(0, 1, size=(vocab_size, word_vector_size)).astype('float32')

    assert word_embeddings[test_ob['word_vectors']].shape == (T, n_sentences, n_batch, word_vector_size)

    model = lstm.FedLSTM(
        vocab_size=vocab_size,
        hidden_size=9,
        lstm_size=12,
        n_mixtures=2,
        word_vectors=word_embeddings,
        doctype_size=7,
        regime_size=4,
        input_size=word_vector_size
    )

    first_cost = model.get_cost_and_update(
        test_ob['word_vectors'],
        test_ob['rates'],
        test_ob['max_mask'],
        test_ob['regimes'],
        test_ob['doc_types']
    )

    for _ in xrange(5):
        last_cost = model.get_cost_and_update(
            test_ob['word_vectors'],
            test_ob['rates'],
            test_ob['max_mask'],
            test_ob['regimes'],
            test_ob['doc_types']
        )

    assert first_cost > last_cost

def lasagne_test():

    fedlstm_model = lstm_lasagne.FedLSTMLasagne(20, 5, 50, 2, 13, 10)

    targets = np.random.randn(5, 3).astype('float32')
    words = np.random.randint(0, 20, size=(5, 2)).astype('int32')

    first_cost = fedlstm_model._train(
        words,
        np.ones(5).astype('int32'),
        np.ones(5).astype('int32'),
        np.ones(5).astype('int32'),
        targets
    )

    for _ in xrange(10):
        last_cost = fedlstm_model._train(
            words,
            np.ones(5).astype('int32'),
            np.ones(5).astype('int32'),
            np.ones(5).astype('int32'),
            targets
        )

    assert first_cost > last_cost

def scraper_test():

    scraper = Scraper()
    assert len(scraper.get_docs(limit=1)) == 1

