__author__ = 'allentran'

import re
import urlparse

import requests
from bs4 import BeautifulSoup

class Scraper(object):

    def __init__(self, start_year=2008, end_year=2009):

        self.date_regex = re.compile(r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})')
        self.minutes_regex = re.compile(r'fomcminutes\d{4}\d{2}\d{2}')

        self.recent_url = 'http://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'
        self.historical_years = range(start_year, end_year + 1)
        self.historical_url = 'http://www.federalreserve.gov/monetarypolicy/fomchistorical%s.htm'

        self.minutes = None
        self.statements = None

    def download_minutes(self, urls):

        def download(session, url):

            r = session.get(url)
            soup = BeautifulSoup(r.content, 'lxml')
            text = soup.find('div',{'id':'leftText'}).get_text()
            match = self.date_regex.search(url)
            year, month, day = match.group('year'), match.group('month'), match.group('day')

            return ((year, month, day), text)

        session = requests.session()
        return [download(session, url) for url in urls]

    def get_minute_urls(self):

        session = requests.session()

        # get recent minutes
        r = session.get(self.recent_url)
        soup = BeautifulSoup(r.content, 'lxml')
        links = []
        for row in soup.find_all('td', 'minutes'):
            row_links = row.find_all('a')
            links += [urlparse.urljoin(self.recent_url, l['href']) for l in row_links]

        # get historical minutes
        for year in self.historical_years:
            r = session.get(self.historical_url % year)
            soup = BeautifulSoup(r.content, 'lxml')
            for row in soup.find_all('td', 'minutes'):
                row_links = row.find_all('a')
                links += [urlparse.urljoin(self.recent_url, l['href']) for l in row_links if self.minutes_regex.search(l['href'])]

        links = [link for link in links if link.find('.pdf') == -1]

        return links

    def get_minutes(self):

        minutes_urls = self.get_minute_urls()
        self.minutes = self.download_minutes(minutes_urls)

if __name__ == '__main__':
    scraper = Scraper()
    scraper.get_minutes()

