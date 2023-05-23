import math
from itertools import combinations

import nltk
nltk.download('punkt')

import networkx as nx
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.stem.snowball import RussianStemmer


class TextRank:
    def __init__(self, tokenizer=None) -> None:
        self.graph = nx.Graph()
        if tokenizer is None:
            self.tokenizer = RegexpTokenizer(r'\w+')
            self.lmtzr = RussianStemmer()
        else:
            self.tokenizer = tokenizer
            self.lmtzr = None
        self._text_to_sentences = sent_tokenize

    def similarity(self, s1, s2):
        if not len(s1) or not len(s2):
            return 0.0
        return len(s1.intersection(s2))/(1.0 * (len(s1) + len(s2)))

    def _tokenize(self, text):
        sentences = self._text_to_sentences(text, language='russian')
        words = []
        for sentence in sentences:
            sentence = sentence.lower()
            if self.lmtzr is None:
                w = set(word for word in self.tokenizer.tokenize(sentence))
            else:
                w = set(self.lmtzr.stem(word) for word in self.tokenizer.tokenize(sentence))
            words.append(w)
        return words, sentences

    def _textrank(self, text):
        words, sentences = self._tokenize(text)
            
        pairs = combinations(range(len(sentences)), 2)
        scores = [(i, j, self.similarity(words[i], words[j])) for i, j in pairs]
        scores = filter(lambda x: x[2], scores)
        
        self.graph.add_weighted_edges_from(scores)
        pr = nx.pagerank(self.graph)
        
        return sorted(((i, pr[i], s) for i, s in enumerate(sentences) if i in pr), key=lambda x: pr[x[0]], reverse=True)

    def _extract(self, text):
        tr = self._textrank(text)
        return tr

    def __call__(self, text, ratio=0.2):
        extracted = self._extract(text)
        extract_len = math.floor(len(extracted) * ratio)
        top_n = sorted(extracted[:extract_len])
        result = [
            {
                'id': r[0],
                'score': r[1],
                'sentence': r[2]
            } for r in top_n
        ]
        return result
