import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.probability import FreqDist
import os
from dataclasses import dataclass, field
from typing import Dict
import math
from enum import Enum


class Tokenizer(Enum):
    SPLIT = "split"
    NLTK = "nltk"


class Stemmer(Enum):
    PORTER = "porter"
    LANCASTER = "lancaster"


PATH_TEMPLATE = "results/{file_type}{tokenizer}_{stemmer}.txt"


class TextProcessor:
    @dataclass(slots=True, order=True)
    class Token:
        token: str = field(compare=True, repr=True)
        freq: Dict[int, int] = field(compare=False, repr=False)
        docs: [int] = field(default_factory=list, compare=False, repr=True)
        weight: Dict[int, float] = field(default_factory=dict, compare=False, repr=False)

    def __init__(self, docs: [str]):
        self._tokenizer: Tokenizer | None = None
        self._stemmer: Stemmer | None = None
        self.docs: [str] = docs
        self._tokens: [TextProcessor.Token] = []

    @property
    def tokenizer(self):
        if not self._tokenizer:
            raise Exception("Tokenizer not set")
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer

    @property
    def stemmer(self):
        if not self._stemmer:
            raise Exception("Stemmer not set")
        return self._stemmer

    @stemmer.setter
    def stemmer(self, stemmer: Stemmer):
        self._stemmer = stemmer

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, tokens: list[Token]):
        self._tokens = tokens

    def add_tokens(self, tokens: list[Token]):
        for token in tokens:
            if token in self._tokens:
                self._tokens[self._tokens.index(token)].docs += token.docs
                self._tokens[self._tokens.index(token)].freq.update(token.freq)
            else:
                self._tokens.append(token)

    def calculate_weight(self, token: Token):
        for doc_number in token.docs:
            max_freq = max([token.freq[doc_number] for token in self.tokens if doc_number in token.docs])
            token.weight[doc_number] = (token.freq[doc_number] / max_freq) * math.log10(
                len(self.docs) / len(token.docs) + 1)

    def stem(self, tokens: [str]):
        """
        Stem the tokens
        :return:
        """
        if self.stemmer.value == "porter":
            stemmer = PorterStemmer()
        elif self.stemmer.value == "lancaster":
            stemmer = LancasterStemmer()
        else:
            raise Exception("Invalid stemmer")
        return [stemmer.stem(token) for token in tokens]

    def tokenize(self, text: str):
        """
        Tokenize the text
        :param text:
        :return:
        """
        match self.tokenizer.value:
            case "split":
                return text.split()
            case "nltk":
                return nltk.RegexpTokenizer(r'\d+[a-zA-Z]|\w+-?\d+(?:\.\d+)?|\w+').tokenize(text)
            case _:
                raise Exception("Invalid method")

    @staticmethod
    def get_freq_dist(tokens: [str]):
        """
        Get frequency distribution of tokens
        :param tokens:
        :return:
        """
        return FreqDist(tokens)

    @classmethod
    def remove_stopwords(cls, tokens: [str]):
        """
        Remove stopwords from the text
        :param tokens:
        :return:
        """
        return [word for word in tokens if word not in stopwords.words('english')]

    def write_to_file(self, file_path: str, doc_number: int, file_type: str = "descriptor"):
        """
        Write tokens to file
        :param file_path:
        :param doc_number:
        :param file_type:
        :return:
        """

        tokens = self.get_tokens_by_doc(doc_number)
        with open(file_path, "a") as f:
            for token in sorted(tokens, key=lambda x: x.token):
                if file_type == "descriptor":
                    f.write(f"{doc_number} {token.token} {token.freq[doc_number]} {token.weight[doc_number]:.4f} \n")
                elif file_type == "inverse":
                    f.write(f"{token.token} {doc_number} {token.freq[doc_number]} {token.weight[doc_number]:.4f}\n")
                else:
                    raise Exception("Invalid type")

    def process_text(self, text: str, doc_number: int):
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem(tokens)
        tokens = [self.Token(token, freq={doc_number: freq}, docs=[doc_number])
                  for token, freq in self.get_freq_dist(tokens).items()]
        return tokens

    def get_tokens_by_doc(self, doc_number: int):
        return [token for token in self.tokens if doc_number in token.docs]

    def save(self):
        descriptor_file_path = f"results/descriptor{self.tokenizer.value.capitalize()}_{self.stemmer.value.capitalize()}.txt"
        inverse_file_path = f"results/inverse{self.tokenizer.value.capitalize()}_{self.stemmer.value.capitalize()}.txt"

        if os.path.exists(descriptor_file_path):
            os.remove(descriptor_file_path)
        if os.path.exists(inverse_file_path):
            os.remove(inverse_file_path)

        for i in range(len(self.docs)):
            self.write_to_file(descriptor_file_path, i + 1)
            self.write_to_file(inverse_file_path, i + 1, file_type="inverse")

    def __call__(self, tokenizer: Tokenizer = Tokenizer.SPLIT, stemmer: Stemmer = Stemmer.LANCASTER):
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.tokens = []
        for i, doc in enumerate(self.docs):
            doc_number = i + 1
            with open(doc, "r") as f:
                text = f.read()
                tokens = self.process_text(text, doc_number)
                self.add_tokens(tokens)

        for token in self.tokens:
            self.calculate_weight(token)
