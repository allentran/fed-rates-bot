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
        word_vectors=np.random.standard_normal((T, n_batch, 300)),
        rates=np.ones((n_batch, 3)),
        max_mask=np.ones((T, n_batch)),
        regimes=np.ones(n_batch),
        doc_types=np.ones(n_batch)
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
