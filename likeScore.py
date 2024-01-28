from fuzzywuzzy import fuzz
from nlp_rake import Rake
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
import pymorphy2
from pymorphy2 import MorphAnalyzer
import razdel
from razdel import tokenize
import re
import yake
from summa import keywords
import pandas as pd
import evaluate
from rouge import Rouge
from scipy import stats
import numpy as np

from tqdm.auto import tqdm
tqdm.pandas()

import ru_core_news_lg

def get_ner(text):
    doc = nlp(text)
    return doc.ents


class Preprocesser:

    """
    preprocesses text
    """

    def __init__(self) -> None:
        self.cash = {}
        self.tokens = []


    # функция для разбиения текста на токены
    def _tokenize_sent(self, sent: str) -> list:
        """
        returns generator over tokens (uses Radel library)
        """
        for start, stop, token in tokenize(sent):
            yield token


    # функция для лемматизации текста, т.е. приведения токена к стандартной форме (лемме)
    def _morph_analyze(self, token: str) -> str:
        """
        returns Pymorphy2 lemma
        """
        # если мы уже видели такой токен, берём его лемму из сохранённого словаря
        if token in self.cash.keys():
            return self.cash[token]
        # если такой токен встретился впервые, лемматизируем через pymorphy2 и сохраняем лемму в словарь
        else:
            self.cash[token] = morph.parse(token)[0].normal_form
            return self.cash[token]


    # функция для замены чисел на спецтокен digit
    def _replace_digits(self, token: str) -> str:
        """
        returns mask 'digit' if a token consists of numerals only
        """
        if not re.search('[^0-9]', token):
            return 'digit'
        else:
            return token


    # функция для удаления стопслов
    def _remove_stopwords(self, lemma: str) -> str:
        """
        If input token is in stopwords, returns None. Else returns lemma
        """
        if lemma not in STOPWORDS:
            return lemma


    # функция для удаления мусорных символов и очень коротких слов
    def _remove_thrash(self, lemma: str) -> str:
        """
        Returns tokens without special symbols. Keeps cyrillic letters and digits
        """
        if lemma == 'digit':
            return lemma
        elif (not re.search('[^а-яА-ЯЁё-]|^-|--|-$|([а-я])\\1{2}', lemma)) and len(lemma) > 1:
            return lemma


   # вариант препроцессинга, при котором из текста удаляются мусорные символы и стопслова, числа заменяются на спецтокен, слова приводятся к стандартной форме
    def preprocess_full(self, doc: str):
        """
        Takes a sentence, returns a generator of clean tokens
        """
        for token in self._tokenize_sent(doc):
            clean_token = self._remove_thrash(self._replace_digits(token))
            if clean_token:
                lemma = self._remove_stopwords(self._morph_analyze(clean_token))
                if lemma is not None:
                    yield lemma


nlp = ru_core_news_lg.load()
morph = MorphAnalyzer()
STOPWORDS = list(set(stopwords.words('russian')))
prep = Preprocesser()


rouge_score = Rouge()
bertscore = evaluate.load('bertscore')


def like_score(prediction: str, reference: str) -> float:

    """
    Likeness metrics for text and its summary. Works and tuned for Russian language.
    See details: https://github.com/eaklykova/multi_document_summarization_2023
    
    prediction: str
    Text of summary as a raw string without  any preprocessing. Works best for longer texts.
    
    reference: str
    riginal text 
    """

    if not isinstance(prediction, str):
        print(f'prediction should be string type, {type(prediction)} found')
        raise ValueError
    if not isinstance(reference, str):
        print(f'reference should be string type, {type(reference)} found')
        raise ValueError

    # initialize constants
    data = pd.DataFrame(columns=['text', 'text_prep', 'textrank', 'rake', 'ner', 'ner_len', 'len'])
    rake = Rake(stopwords=STOPWORDS, max_words=5)

    # get preprocessing for both objects
    for i, text in enumerate([prediction, reference]):
        if not isinstance(text, str):
            print(f'Prediction and reference should be string type, {type(text)} found')
            raise ValueError
        else:
            data.at[i, 'text'] = text
            data.at[i, 'text_prep'] = [w for w in prep.preprocess_full(text)]
            data.at[i, 'textrank'] = keywords.keywords(" ".join(data.at[i, 'text_prep']), language='russian').split('\n')
            data.at[i, 'rake'] = [item[0] for item in rake.apply(data.at[i, 'text']) if item[1] > 1]
            data.at[i, 'ner'] = data.text.apply(lambda x: get_ner(x))
            data.at[i, 'len'] = data.text_prep.apply(lambda x: len(x))

    # compute metrics
    if len(data.at[0, 'textrank']) and len(data.at[1, 'textrank']):
        bert_tx = bertscore.compute(predictions=[" ".join(data.at[0, 'textrank'])],
                                    references=[" ".join(data.at[1, 'textrank'])], lang='ru')['f1'][0]
        

    else:
         bert_tx = 0

    if len(data.at[0, 'rake']) and len(data.at[1, 'rake']):
        bert_rake = bertscore.compute(predictions=[" ".join(data.at[0, 'rake'])],
                                      references=[" ".join(data.at[1, 'rake'])], lang='ru')['f1'][0]
    else:
        bert_rake = 0

    try:
        ner = len(data.at[0, 'ner']) / len(data.at[1, 'ner'])
    except ZeroDivisionError:
        ner = 1
    
    try:
        ln = data.at[0, 'len'][0] / data.at[1, 'len'][1]
    except ZeroDivisionError:
        ln = 1
    

    rouge = rouge_score.get_scores(data.at[0, 'text'], data.at[1, 'text'])[0]['rouge-l']['f']

    score = bert_tx + bert_rake * (ner * rouge)/(ln * (ner + rouge))
    
    return score
	
	