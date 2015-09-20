__author__ = 'allentran'

from ..scraper import Scraper

def scraper_test():

    scraper = Scraper()
    assert len(scraper.get_docs(limit=1)) == 1
