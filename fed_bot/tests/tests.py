__author__ = 'allentran'

from ..scraper import Scraper
from ..model import lstm

def scraper_test():

    scraper = Scraper()
    assert len(scraper.get_docs(limit=1)) == 1

def model_test():

    model = lstm.FedLSTM(
        hidden_sizes=[10, 10, 10, 10]
    )
