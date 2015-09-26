__author__ = 'allentran'

import json
import os
import re
import datetime

import unidecode
from spacy.en import English
import requests
import pandas as pd
import numpy as np

class PairedDocAndRates(object):

    def __init__(self, date, vectors):

        self.date = date
        self.vectors = vectors
        self.rates = None

    def match_rates(self, rates_df, days = [30, 90, 180]):

        def get_closest_rate(days_to_add):
            future_date = self.date + datetime.timedelta(days=days_to_add)
            diff = abs(future_date - rates_df['date'])
            if (last_available_date - future_date).total_seconds() >= 0:
                closest_index = diff.argmin()
                return float(rates_df.iloc[closest_index]['value'])
            else:
                return None

        future_rates = {}
        last_available_date = rates_df['date'].iloc[-1]
        current_rate = get_closest_rate(0)
        if current_rate:
            future_rates['0'] = current_rate
        for add_days in days:
            future_rate = get_closest_rate(add_days)
            if future_rate:
                future_rates[str(add_days)] =  future_rate

        self.rates = future_rates

    def to_dict(self):

        return dict(
            date = self.date.strftime('%Y-%m-%d'),
            vectors = self.vectors.tolist(),
            rates = self.rates
        )


class DataTransformer(object):

    def __init__(self, data_dir):

        self.url = 'https://api.stlouisfed.org/fred/series/observations'
        self.data_dir = data_dir

        self.rates = None
        self.docs = None

    def get_rates(self, api_key):

        params = dict(
            api_key=api_key,
            file_type='json',
            series_id='FEDFUNDS'
        )
        r = requests.get(self.url, params=params)
        if r.status_code == 200:
            self.rates = pd.DataFrame(r.json()['observations'])
            self.rates['date'] = self.rates['date'].apply(lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'))
            self.rates.sort('date')

    def get_docs(self, min_sentence_length=8):

        def parse_doc(doc_path):
            with open(doc_path, 'r') as f:
                text = unidecode.unidecode(unicode(f.read().decode('iso-8859-1')))
                text = ' '.join(text.split()).strip()
            if len(text) > 0:
                date = datetime.datetime.strptime(date_re.search(doc_path).group(0), '%Y%m%d')
                doc = nlp(unicode(text))

                vectors = []
                sentences = list(doc.sents)
                for sent in sentences[1:]:
                    if len(sent) > min_sentence_length:
                        for token in sent:
                            try:
                                vectors.append(token.repvec)
                            except ValueError:
                                pass

                paired_doc = PairedDocAndRates(date, np.array(vectors))
                paired_doc.match_rates(self.rates)

                return paired_doc

        date_re = re.compile(r'\d{8}')
        file_re = re.compile(r'\d{8}')
        nlp = English()

        docs = []
        for root, dirs, filenames in os.walk(self.data_dir):
            for filename in filenames:
                if file_re.search(filename):
                    parsed_doc = parse_doc(os.path.join(root, filename))
                    if parsed_doc:
                        docs.append(parsed_doc)

        self.docs = docs

    def save_output(self):

        with open(os.path.join(self.data_dir, 'paired_data.json'), 'w') as f:
            json.dump([doc.to_dict() for doc in self.docs], f, indent=2, sort_keys=True)

if __name__ == "__main__":

    data_transformer = DataTransformer('data')
    data_transformer.get_rates('51c09c6b8aa464671aa8ac96c76a8416')
    data_transformer.get_docs()
    data_transformer.save_output()

