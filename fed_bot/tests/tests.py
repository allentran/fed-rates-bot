__author__ = 'allentran'

import numpy as np

from ..scraper import Scraper
from ..model import lstm

def scraper_test():

    scraper = Scraper()
    assert len(scraper.get_docs(limit=1)) == 1

def model_test():

    n_batch = 10
    T = 20

    test_ob = dict(
        word_vectors=np.random.standard_normal((T, n_batch, 300)).astype('float32'),
        rates=np.ones((n_batch, 3)).astype('float32'),
        max_mask=np.ones((T, n_batch)).astype('float32'),
        regimes=np.ones(n_batch).astype('int32'),
        doc_types=np.ones(n_batch).astype('int32')
    )

    model = lstm.FedLSTM(
        hidden_sizes=[10, 10, 10, 10]
    )

    model.get_cost_and_update(
        test_ob['word_vectors'],
        test_ob['rates'],
        test_ob['max_mask'],
        test_ob['regimes'],
        test_ob['doc_types']
    )
