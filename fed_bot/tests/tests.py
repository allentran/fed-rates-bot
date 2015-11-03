__author__ = 'allentran'

import numpy as np

from ..scraper import Scraper
from ..model import lstm

def scraper_test():

    scraper = Scraper()
    assert len(scraper.get_docs(limit=1)) == 1

def model_test():

    n_sentences = 6
    T = 21
    vocab_size=55
    word_vector_size=111

    test_ob = dict(
        word_vectors=np.random.randint(0, vocab_size, size=(T, n_sentences)).astype('int32'),
        rates=np.ones(3).astype('float32'),
        max_mask=np.ones((T, n_sentences)).astype('float32'),
        regimes=np.int32(1),
        doc_types=np.int32(1).astype('int32')
    )

    word_embeddings = np.random.normal(0, 1, size=(vocab_size, word_vector_size)).astype('float32')

    assert word_embeddings[test_ob['word_vectors']].shape == (T, n_sentences, word_vector_size)

    model = lstm.FedLSTM(
        vocab_size=vocab_size,
        hidden_sizes=[10, 20, 30, 40],
        n_mixtures=2,
        word_vectors=word_embeddings,
        doctype_size=7,
        regime_size=4,
        input_size=word_vector_size
    )

    model.get_cost_and_update(
        test_ob['word_vectors'],
        test_ob['rates'],
        test_ob['max_mask'],
        test_ob['regimes'],
        test_ob['doc_types']
    )
