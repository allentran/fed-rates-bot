__author__ = 'allentran'

import re
import urlparse
import os

from unidecode import unidecode
import requests
from bs4 import BeautifulSoup

class Scraper(object):

    def __init__(self, start_year=2008, end_year=2009):

        self.date_regex = re.compile(r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})')

        self.recent_url = 'http://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'
        self.historical_years = range(start_year, end_year + 1)
        self.historical_url = 'http://www.federalreserve.gov/monetarypolicy/fomchistorical%s.htm'

    def download_minutes(self, urls):

        def download(session, url):

            r = session.get(url)
            soup = BeautifulSoup(r.content, 'lxml')
            text = soup.find('div',{'id':'leftText'}).get_text()
            match = self.date_regex.search(url)
            year, month, day = match.group('year'), match.group('month'), match.group('day')

            return dict(
                date=[year, month, day],
                text=text
            )

        session = requests.session()
        return [download(session, url) for url in urls]

    def get_urls(self, table_class='statement2', link_text='Statement'):

        session = requests.session()

        # get recent
        r = session.get(self.recent_url)
        soup = BeautifulSoup(r.content, 'lxml')
        links = []
        for row in soup.find_all('td', table_class):
            row_links = row.find_all('a')
            links += [urlparse.urljoin(self.recent_url, l['href']) for l in row_links]

        # get historical
        for year in self.historical_years:
            r = session.get(self.historical_url % year)
            soup = BeautifulSoup(r.content, 'lxml')
            row_links = soup.find_all('a')
            links += [urlparse.urljoin(self.recent_url, l['href']) for l in row_links if l.get_text().strip() == link_text]

        links = [link for link in links if link.find('.pdf') == -1]

        return links

    def get_docs(self, minutes=False, limit=1000000000):

        if not minutes:
            statement_urls = self.get_urls()
            return self.download_minutes(statement_urls[:limit])
        minute_urls = self.get_urls(table_class='minutes', link_text='HTML')
        return self.download_minutes(minute_urls[:limit])

if __name__ == "__main__":
    scraper = Scraper()
    statements = scraper.get_docs()
    for statement in statements:
        date = statement['date']
        filename = '%s%s%s.txt' % (date[0], date[1], date[2])
        with open(os.path.join('data/statements', filename), 'w') as f:
            f.write(unidecode(statement['text']))