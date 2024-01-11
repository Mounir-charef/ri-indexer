import math
import os
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import TypedDict
import gc
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer, LancasterStemmer
from tqdm import tqdm


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
    VECTOR = "Vector Space Model"
    PROBABILITY = "Probability Space Model"
    LOGIC = "Logic Model"


class MatchingType(Enum):
    Scalar = "Scalar Product"
    Cosine = "Cosine Measure"
    Jaccard = "Jaccard Measure"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


PATH_TEMPLATE = "results/{file_type}{tokenizer}_{stemmer}.txt"


class TextProcessor:
    class Token(TypedDict):
        freq: dict[int, int]
        docs: list[int]
        weight: dict[int, float]

    def __init__(self, documents_dir: Path, results_dir: Path, *, doc_prefix: str):
        results_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer: Tokenizer | None = None
        self._stemmer: Stemmer | None = None
        self.docs: dict[int, str] = {
            int(file.stem[len(doc_prefix) :]): file.read_text().lower()
            for file in documents_dir.iterdir()
        }
        self.tokens: dict[str, TextProcessor.Token] = {}
        self.file_path = results_dir
        self._tokens_by_doc = defaultdict(dict)

    def cleanup(self):
        self._tokenizer = None
        self._stemmer = None
        self.tokens = None
        self._tokens_by_doc = None

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

    def set_processor(self, tokenizer: Tokenizer, stemmer: Stemmer):
        self.tokenizer = tokenizer
        self.stemmer = stemmer

    @property
    def descriptor_file_path(self):
        return (
            self.file_path
            / f"descriptor{self.tokenizer.value.capitalize()}_{self.stemmer.value.capitalize()}.txt"
        )

    @property
    def inverse_file_path(self):
        return (
            self.file_path
            / f"inverse{self.tokenizer.value.capitalize()}_{self.stemmer.value.capitalize()}.txt"
        )

    @property
    def tokens_by_doc(self):
        if self._tokens_by_doc:
            return self._tokens_by_doc
        else:
            raise Exception("No tokens are yet generated")

    def add_tokens(self, tokens: dict[str, Token]):
        """
        Add tokens to the tokens dictionary
        :param tokens:
        :return:
        """
        for token, value in tokens.items():
            if token not in self.tokens:
                self.tokens[token] = value
            else:
                self.tokens[token]["docs"] += value["docs"]
                for doc_number in value["freq"]:
                    if doc_number not in self.tokens[token]["freq"]:
                        self.tokens[token]["freq"][doc_number] = value["freq"][
                            doc_number
                        ]
                    else:
                        self.tokens[token]["freq"][doc_number] += value["freq"][
                            doc_number
                        ]
            self._tokens_by_doc[value["docs"][0]][token] = value

    def calculate_weight(self, token_key: str):
        """
        Calculate the weight of the token
        :param token_key:
        :return:
        """
        token = self.tokens[token_key]

        for doc_number in token["docs"]:
            max_freq = max(
                [
                    token["freq"][doc_number]
                    for token in self.get_tokens_by_doc(doc_number).values()
                ]
            )
            token["weight"][doc_number] = round(
                (token["freq"][doc_number] / max_freq)
                * math.log10(len(self.docs) / len(token["docs"]) + 1),
                4,
            )

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
        return [
            stemmer.stem(token)
            for token in tokens
            if token not in set(stopwords.words("english"))
        ]

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
                return nltk.RegexpTokenizer(
                    r"(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*"
                ).tokenize(text)
            case _:
                raise Exception("Invalid method")

    def save(self):
        if os.path.exists(self.descriptor_file_path):
            os.remove(self.descriptor_file_path)
        if os.path.exists(self.inverse_file_path):
            os.remove(self.inverse_file_path)

        with open(self.descriptor_file_path, "w") as f:
            for token, value in self.tokens.items():
                for doc_number in value["freq"]:
                    f.write(
                        f"{doc_number} {token} {value['freq'][doc_number]} {value['weight'][doc_number]}\n"
                    )
        with open(self.inverse_file_path, "w") as f:
            for token, value in self.tokens.items():
                for doc_number in value["freq"]:
                    f.write(
                        f"{token} {doc_number} {value['freq'][doc_number]} {value['weight'][doc_number]}\n"
                    )

    def process_text(self, text: str):
        tokens = self.tokenize(text)
        return self.stem(tokens)

    def process_doc(self, text: str, doc_number: int):
        """
        Process the text
        :param text:
        :param doc_number:
        :return:
        """
        tokens = self.process_text(text)
        processed_tokens: dict[str, TextProcessor.Token] = {}
        for token, freq in FreqDist(tokens).items():
            processed_tokens[token] = {
                "freq": {doc_number: freq},
                "docs": [doc_number],
                "weight": {doc_number: 0},
            }
        return processed_tokens

    def get_tokens_by_doc(self, doc_number: int):
        """
        Get tokens by doc number
        :param doc_number:
        :return:
        """
        if doc_number in self.tokens_by_doc:
            return self.tokens_by_doc[doc_number]
        else:
            raise Exception("Invalid doc number")

    def process_docs(self):
        """
        Process the docs and generate all inverted and descriptor files
        :return:
        """
        for tokenizer in Tokenizer:
            for stemmer in Stemmer:
                self.set_processor(tokenizer, stemmer)
                self.tokens = {}
                for doc_number, doc in tqdm(self.docs.items(), desc="Processing docs"):
                    self.add_tokens(self.process_doc(doc, doc_number))
                for token in self.tokens:
                    self.calculate_weight(token)
                self.save()

        self.cleanup()
        gc.collect()
