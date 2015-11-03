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
import allen_utils

logger = allen_utils.get_logger(__name__)

class Interval(object):

    def __init__(self, start, end):

        assert isinstance(start, datetime.date) and isinstance(end, datetime.date)

        self.start = start
        self.end = end

    def contains(self, new_date):
        assert isinstance(new_date, datetime.date)
        return (new_date >= self.start) and (new_date <= self.end)

fed_regimes = {
    0: Interval(datetime.date(1951, 4, 2), datetime.date(1970, 1, 31)),
    1: Interval(datetime.date(1970, 2, 1), datetime.date(1978, 3, 7)),
    2: Interval(datetime.date(1978, 3, 8), datetime.date(1979, 8, 6)),
    3: Interval(datetime.date(1979, 8, 7), datetime.date(1987, 8, 11)),
    4: Interval(datetime.date(1987, 8, 12), datetime.date(2006, 1, 31)),
    5: Interval(datetime.date(2006, 2, 1), datetime.date(2020, 1, 31)),
}

def find_regime(date):

    for regime, interval in fed_regimes.iteritems():
        if interval.contains(date):
            return regime
    raise ValueError("Could not find regime for date, %s", date)

class PairedDocAndRates(object):

    def __init__(self, date, sentences, is_minutes):

        self.date = date
        self.sentences = sentences
        self.is_minutes = is_minutes
        self.rates = None
        self.regime = find_regime(date)

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
            sentences = self.sentences,
            rates = self.rates,
            is_minutes = self.is_minutes,
            regime = self.regime
        )

class Vocab(object):

    def __init__(self):

        self.vocab = {}
        self.special_words = [
            '$CARDINAL$',
            '$DATE$',
            '$UNKNOWN$'
        ]

    def update_count(self, word):

        if word not in self.vocab:
            self.vocab[word] = 1
        else:
            self.vocab[word] += 1

    def to_dict(self, min_count=5):

        position_dict = {word: idx for idx, word in enumerate(self.special_words)}
        counter = len(self.special_words)
        for word, word_count in self.vocab.iteritems():
            if word_count >= min_count:
                position_dict[word] = counter
                counter += 1

        return position_dict

class DataTransformer(object):

    def __init__(self, data_dir, min_sentence_length):

        self.url = 'https://api.stlouisfed.org/fred/series/observations'
        self.data_dir = data_dir
        self.min_sentence_length = min_sentence_length
        self.replace_entities = {
            'DATE': '$DATE$',
            'CARDINAL': '$CARDINAL$'
        }

        self.vocab = Vocab()
        self.word_positions = None

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
            self.rates['date'] = self.rates['date'].apply(lambda s: datetime.datetime.strptime(s, '%Y-%m-%d').date())
            self.rates.sort('date')

    def build_vocab(self):

        def process_doc(doc_path):
            with open(doc_path, 'r') as f:
                text = unidecode.unidecode(unicode(f.read().decode('iso-8859-1')))
                text = ' '.join(text.split()).strip()
            if len(text) > 0:
                doc = nlp(unicode(text).lower())
                found_words = set()
                for sent in doc.sents:
                    if len(sent) > self.min_sentence_length:
                        for token in doc:
                            if token.text not in found_words:
                                self.vocab.update_count(token.text)
                                found_words.add(token.text)

        file_re = re.compile(r'\d{8}')
        nlp = English()

        for root, dirs, filenames in os.walk(self.data_dir):
            for filename in filenames:
                if file_re.search(filename):
                    filepath = os.path.join(root, filename)
                    process_doc(filepath)
                    logger.info("Built vocab from: %s", filepath)

        self.word_positions = self.vocab.to_dict()

    def get_docs(self, min_sentence_length=8):

        # custom token replacement
        regexes = [
            (re.compile(r'\d{4}'), '$DATE$'),
            (re.compile(r'\d+[\.,]*\d+'), '$CARDINAL$')
        ]


        def parse_doc(doc_path):
            with open(doc_path, 'r') as f:
                text = unidecode.unidecode(unicode(f.read().decode('iso-8859-1')))
                text = ' '.join(text.split()).strip()
            if len(text) > 0:
                date = datetime.datetime.strptime(date_re.search(doc_path).group(0), '%Y%m%d').date()
                unicode_text = unicode(text).lower()
                doc = nlp(unicode_text)

                # spacy token replacement
                ents_dict = {ent.text: self.replace_entities[ent.label_] for ent in doc.ents if ent.label_ in self.replace_entities.keys()}
                for ent in ents_dict:
                    unicode_text = unicode_text.replace(ent, ents_dict[ent])

                for regex, replacement_token in regexes:
                    unicode_text = regex.sub(replacement_token, unicode_text)

                doc = nlp(unicode_text)
                sentences = list(doc.sents)
                doc_sents = []

                for sent in sentences[1:]:
                    if len(sent) > min_sentence_length:
                        sentence_as_idxes = []
                        for token in sent:
                            try:
                                sentence_as_idxes.append(self.word_positions[token.text])
                            except KeyError:
                                sentence_as_idxes.append(self.word_positions['$UNKNOWN$'])
                        doc_sents.append(sentence_as_idxes)

                paired_doc = PairedDocAndRates(date, doc_sents, doc_path.find('minutes') > -1)
                paired_doc.match_rates(self.rates)

                return paired_doc

        date_re = re.compile(r'\d{8}')
        file_re = re.compile(r'\d{8}')
        nlp = English()

        docs = []
        for root, dirs, filenames in os.walk(self.data_dir):
            for filename in filenames:
                if file_re.search(filename):
                    filepath = os.path.join(root, filename)
                    parsed_doc = parse_doc(filepath)
                    if parsed_doc:
                        logger.info("Parsed %s", filepath)
                        docs.append(parsed_doc)

        self.docs = docs

    def save_output(self):

        with open(os.path.join(self.data_dir, 'paired_data.json'), 'w') as f:
            json.dump([doc.to_dict() for doc in self.docs], f, indent=2, sort_keys=True)

        with open(os.path.join(self.data_dir, 'dictionary.json'), 'w') as f:
            json.dump(self.vocab.to_dict(), f, indent=2, sort_keys=True)

if __name__ == "__main__":

    data_transformer = DataTransformer('data', min_sentence_length=8)
    data_transformer.build_vocab()
    data_transformer.get_rates('51c09c6b8aa464671aa8ac96c76a8416')
    data_transformer.get_docs()
    data_transformer.save_output()

