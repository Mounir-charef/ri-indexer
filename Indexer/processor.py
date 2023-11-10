import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.probability import FreqDist
import os
from dataclasses import dataclass, field
from typing import Dict
import math
from enum import Enum
from pathlib import Path


class Tokenizer(Enum):
    SPLIT = "split"
    NLTK = "nltk"


class Stemmer(Enum):
    PORTER = "porter"
    LANCASTER = "lancaster"


class FileType(Enum):
    DESCRIPTOR = "descriptor"
    INVERSE = "inverse"


class SearchType(Enum):
    DOCS = "docs"
    TERM = "term"


PATH_TEMPLATE = "results/{file_type}{tokenizer}_{stemmer}.txt"


class TextProcessor:
    @dataclass(slots=True, order=True)
    class Token:
        token: str = field(compare=True, repr=True)
        freq: Dict[int, int] = field(compare=False, repr=False)
        docs: [int] = field(default_factory=list, compare=False, repr=True)
        weight: Dict[int, float] = field(default_factory=dict, compare=False, repr=False)

    def __init__(self, docs: [str], file_path: Path):
        if file_path.exists() and file_path.is_dir():
            for file in file_path.iterdir():
                file.unlink()
            file_path.rmdir()
        file_path.mkdir()
        self._tokenizer: Tokenizer | None = None
        self._stemmer: Stemmer | None = None
        self.docs: [str] = docs
        self._tokens: [TextProcessor.Token] = []
        self.file_path = file_path

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
    def descriptor_file_path(self):
        return f"{self.file_path}/descriptor{self.tokenizer.value.capitalize()}_{self.stemmer.value.capitalize()}.txt"

    @property
    def inverse_file_path(self):
        return f"{self.file_path}/inverse{self.tokenizer.value.capitalize()}_{self.stemmer.value.capitalize()}.txt"

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

    def stem_word(self, token: str):
        if self.stemmer.value == "porter":
            stemmer = PorterStemmer()
        elif self.stemmer.value == "lancaster":
            stemmer = LancasterStemmer()
        else:
            raise Exception("Invalid stemmer")
        return stemmer.stem(token)

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
                return nltk.RegexpTokenizer(r'\w+(?:[-/,%@\.]\w+)*%?').tokenize(text)
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

    def write_to_file(self, doc_number: int, file_type: FileType = FileType.DESCRIPTOR):
        """
        Write tokens to file
        :param doc_number:
        :param file_type:
        :return:
        """

        tokens = self.get_tokens_by_doc(doc_number)
        file_path = self.descriptor_file_path if file_type == FileType.DESCRIPTOR else self.inverse_file_path
        with open(file_path, "a") as f:
            for token in sorted(tokens, key=lambda x: x.token):
                if file_type == FileType.DESCRIPTOR:
                    f.write(f"{doc_number} {token.token} {token.freq[doc_number]} {token.weight[doc_number]:.4f} \n")
                elif file_type == FileType.INVERSE:
                    f.write(f"{token.token} {doc_number} {token.freq[doc_number]} {token.weight[doc_number]:.4f}\n")
                else:
                    raise Exception("Invalid type")

    def save(self):
        if os.path.exists(self.descriptor_file_path):
            os.remove(self.descriptor_file_path)
        if os.path.exists(self.inverse_file_path):
            os.remove(self.inverse_file_path)

        for i in range(len(self.docs)):
            self.write_to_file(i + 1)
            self.write_to_file(i + 1, file_type=FileType.INVERSE)

    def file_generator(self, file_type: FileType):
        file_path = self.descriptor_file_path if file_type.value == "descriptor" else self.inverse_file_path
        with open(file_path, "r") as f:
            for line in f:
                yield line.split()

    def search_in_file(self, file_type: FileType, query: str, search_type: SearchType):
        if not query:
            file_path = self.descriptor_file_path if file_type.value == "descriptor" else self.inverse_file_path
            with open(file_path, "r") as f:
                data = [line.split() for line in f.readlines()]
            return data

        data = []
        query = self.stem_word(query) if search_type == SearchType.TERM else query
        if search_type == SearchType.DOCS:
            for doc_number, token, freq, weight in self.file_generator(file_type):
                if query == doc_number:
                    data.append([doc_number, token, freq, weight])
        elif search_type == SearchType.TERM:
            for token, doc_number, freq, weight in self.file_generator(file_type):
                if query == token:
                    data.append([doc_number, token, freq, weight])
        else:
            raise Exception("Invalid Search type")
        return data

    def process_text(self, text: str, doc_number: int):
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem(tokens)
        tokens = [self.Token(token, freq={doc_number: freq}, docs=[doc_number])
                  for token, freq in self.get_freq_dist(tokens).items()]
        return tokens

    def get_tokens_by_doc(self, doc_number: int):
        return [token for token in self.tokens if doc_number in token.docs]

    def __call__(self, tokenizer: Tokenizer = Tokenizer.SPLIT, stemmer: Stemmer = Stemmer.LANCASTER):
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.tokens = []
        for i, doc in enumerate(self.docs):
            doc_number = i + 1
            with open(doc, "r") as f:
                text = f.read().lower()
                tokens = self.process_text(text, doc_number)
                self.add_tokens(tokens)

        for token in self.tokens:
            self.calculate_weight(token)
