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
from collections import defaultdict
import re


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
    @dataclass(slots=True, order=True)
    class Token:
        token: str = field(compare=True, repr=True)
        freq: Dict[int, int] = field(compare=False, repr=False)
        docs: [int] = field(default_factory=list, compare=False, repr=True)
        weight: Dict[int, float] = field(default_factory=dict, compare=False, repr=True)

    def __init__(
        self, documents_dir, results_dir: Path, *, judgements_path: Path, queries_path: Path
    ):
        if results_dir.exists() and results_dir.is_dir():
            for file in results_dir.iterdir():
                file.unlink()
            results_dir.rmdir()
        results_dir.mkdir()
        self._tokenizer: Tokenizer | None = None
        self._stemmer: Stemmer | None = None
        self.docs: [str] = [
            file.resolve() for file in documents_dir.iterdir() if file.suffix == ".txt"
        ]
        self._tokens: [TextProcessor.Token] = []
        self.file_path = results_dir
        self.judgements_path = judgements_path
        self.queries_path = queries_path
        self._tokens_by_doc = defaultdict(list)

    @property
    def judgements(self) -> list[[str, str]]:
        try:
            with open(self.judgements_path, "r") as f:
                return [line.split() for line in f.readlines()]
        except FileNotFoundError:
            return []

    @property
    def queries(self) -> list[str]:
        try:
            with open(self.queries_path, "r") as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            return []

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
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, tokens: list[Token]):
        self._tokens = tokens

    @property
    def tokens_by_doc(self):
        if self._tokens_by_doc:
            return self._tokens_by_doc
        else:
            raise Exception("No tokens are yet generated")

    def add_tokens(self, tokens: list[Token]):
        for token in tokens:
            if token in self._tokens:
                self._tokens[self._tokens.index(token)].docs += token.docs
                self._tokens[self._tokens.index(token)].freq.update(token.freq)
            else:
                self._tokens.append(token)
            self._tokens_by_doc[token.docs[0]].append(token)

    def calculate_weight(self, token: Token):
        for doc_number in token.docs:
            max_freq = max(
                [
                    token.freq[doc_number]
                    for token in self.tokens
                    if doc_number in token.docs
                ]
            )
            token.weight[doc_number] = (token.freq[doc_number] / max_freq) * math.log10(
                len(self.docs) / len(token.docs) + 1
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
                return nltk.RegexpTokenizer(r"\w+(?:[-/,%@\.]\w+)*%?").tokenize(text)
            case _:
                raise Exception("Invalid method")

    def get_token_by_value(self, text: str):
        for token in self.tokens:
            if token.token == text:
                return token

    def get_token_by_doc_number(self, doc_number: int):
        return [token for token in self.tokens if doc_number in token.docs]

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
        return [word for word in tokens if word not in stopwords.words("english")]

    def write_to_file(self, doc_number: int, file_type: FileType = FileType.DESCRIPTOR):
        """
        Write tokens to file
        :param doc_number:
        :param file_type:
        :return:
        """

        tokens = self.get_tokens_by_doc(doc_number)
        file_path = (
            self.descriptor_file_path
            if file_type == FileType.DESCRIPTOR
            else self.inverse_file_path
        )
        with open(file_path, "a") as f:
            for token in sorted(tokens, key=lambda x: x.token):
                if file_type == FileType.DESCRIPTOR:
                    f.write(
                        f"{doc_number} {token.token} {token.freq[doc_number]} {token.weight[doc_number]:.4f} \n"
                    )
                elif file_type == FileType.INVERSE:
                    f.write(
                        f"{token.token} {doc_number} {token.freq[doc_number]} {token.weight[doc_number]:.4f}\n"
                    )
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
        file_path = (
            self.descriptor_file_path
            if file_type.value == "descriptor"
            else self.inverse_file_path
        )
        with open(file_path, "r") as f:
            for line in f:
                yield line.split()

    def evaluate(self, query_index: int, results, search_type: SearchType):
        """
            Evaluate the results of the query against the judgements
        :param query_index: the index of the query
        :param results: the results of the query
        :param search_type: the type of search
        """

        # get the relevant docs
        relevant_docs = set()
        for judgement in self.judgements:
            if judgement[0] == str(query_index):
                relevant_docs.add(judgement[1])

        # get the retrieved docs
        retrieved_docs = set()
        if search_type == SearchType.TERM:
            for doc in results:
                retrieved_docs.add(doc[1])
        else:
            for doc in results:
                retrieved_docs.add(doc[0])

        # calculate precision, recall and f1-score
        print(f'Relevant docs: {relevant_docs}, Retrieved docs: {retrieved_docs}, intersection: {relevant_docs.intersection(retrieved_docs)}')
        precision = len(relevant_docs.intersection(retrieved_docs)) / len(retrieved_docs)
        precision_5 = len(relevant_docs.intersection(list(retrieved_docs)[:5])) / 5
        precision_10 = len(relevant_docs.intersection(list(retrieved_docs)[:10])) / 10
        recall = len(relevant_docs.intersection(retrieved_docs)) / len(relevant_docs)
        f1_score = 2 * precision * recall / (precision + recall)

        return {
            "Precision": precision,
            "P@5": precision_5,
            "P@10": precision_10,
            "Recall": recall,
            "F1 score": f1_score,
        }

    def search_in_file(
        self,
        query: str,
        *,
        file_type: FileType,
        search_type: SearchType,
        matching_form=MatchingType.Scalar,
        **kwargs,
    ):
        if not query and search_type not in [SearchType.VECTOR, SearchType.PROBABILITY]:
            file_path = (
                self.inverse_file_path
                if file_type == FileType.INVERSE
                else self.descriptor_file_path
            )
            with open(file_path, "r") as f:
                data = [line.split() for line in f.readlines()]
            return data
        data = []
        match search_type:
            case SearchType.DOCS:
                query = [self.stem_word(word) for word in self.tokenize(query)]
                for doc_number, token, freq, weight in self.file_generator(file_type):
                    if token in query:
                        data.append([doc_number, token, freq, weight])

            case SearchType.TERM:
                query = [self.stem_word(word) for word in self.tokenize(query)]
                for token, doc_number, freq, weight in self.file_generator(file_type):
                    if token in query:
                        data.append([token, doc_number, freq, weight])

            case SearchType.VECTOR:
                query = [self.stem_word(word) for word in self.tokenize(query)]
                query = self.remove_stopwords(query)
                total_weight = defaultdict(list)
                doc_weights = defaultdict(list)
                for doc_number, token, freq, weight in self.file_generator(file_type):
                    doc_weights[doc_number].append(float(weight) ** 2)
                    if token in query:
                        total_weight[doc_number].append(float(weight))
                for doc_number, weights in total_weight.items():
                    match matching_form:
                        case MatchingType.Scalar:
                            weight = sum(weights)
                        case MatchingType.Cosine:
                            weight = sum(weights) / (
                                math.sqrt(len(query))
                                * math.sqrt(sum(doc_weights[doc_number]))
                            )
                        case MatchingType.Jaccard:
                            weight = sum(weights) / (
                                len(query) + sum(doc_weights[doc_number]) - sum(weights)
                            )
                        case _:
                            raise Exception("None valid matching formula")

                    data.append([doc_number, round(weight, 4)])
                data.sort(key=lambda row: row[1], reverse=True)

            case search_type.PROBABILITY:
                query = [self.stem_word(word) for word in self.tokenize(query)]
                query = self.remove_stopwords(query)
                tokens = [
                    self.get_token_by_value(token)
                    for token in query
                    if self.get_token_by_value(token)
                ]
                k, b = float(kwargs["matching_params"].get("K", 2)), float(
                    kwargs["matching_params"].get("B", 1.5)
                )
                docs_size = defaultdict(int)
                for doc in range(1, len(self.docs) + 1):
                    current_tokens = self.get_tokens_by_doc(doc)
                    docs_size[doc] = sum([token.freq[doc] for token in current_tokens])

                average_doc_size = sum(docs_size.values()) / len(self.docs)
                rsv = defaultdict(float)
                for token in tokens:
                    for doc_number in token.docs:
                        rsv[doc_number] += (
                            token.freq[doc_number]
                            / (
                                k
                                * (
                                    (1 - b)
                                    + b * (docs_size[doc_number] / average_doc_size)
                                )
                                + token.freq[doc_number]
                            )
                        ) * math.log10(
                            (len(self.docs) - len(token.docs) + 0.5)
                            / (len(token.docs) + 0.5)
                        )

                for doc_number, weight in rsv.items():
                    data.append([str(doc_number), round(weight, 4)])
                data.sort(key=lambda row: row[1], reverse=True)
            case search_type.LOGIC:

                def is_valid_query(test_query):
                    word_pattern = r"\w+(?:[-/,%@\.]\w+)*%?"
                    test_query = test_query.strip()
                    if test_query in ["AND", "OR", "NOT"]:
                        return False
                    return bool(
                        re.match(
                            rf"^(NOT\s+)?(?!AND|OR|NOT){word_pattern}(?:\s+(AND|OR)\s+(NOT\s+)?(?!AND|OR|NOT){word_pattern})*$",
                            test_query,
                        )
                    )

                if not is_valid_query(query):
                    return data

                must_parts = [part.strip() for part in query.strip().split("OR")]
                results = defaultdict(bool)
                for part in must_parts:
                    part = part.split("AND")
                    positive = [
                        self.stem_word(term.lower())
                        for term in part
                        if not term.strip().startswith("NOT")
                    ]
                    negative = [
                        self.stem_word(term.strip().split()[-1])
                        for term in part
                        if term.strip().startswith("NOT")
                    ]
                    for doc_id in range(1, len(self.docs) + 1):
                        tokens = self.get_tokens_by_doc(doc_id)
                        positive_result = 0
                        negative_result = 0
                        for word in positive:
                            for token in tokens:
                                if token.token == word:
                                    positive_result += 1

                        for word in negative:
                            for token in tokens:
                                if token.token == word:
                                    negative_result += 1
                        results[doc_id] = results.get(doc_id, False) or (
                            positive_result == len(positive) and negative_result == 0
                        )

                for doc in results.keys():
                    if results[doc]:
                        data.append([str(doc), results[doc]])
                data.sort(key=lambda row: row[0])

            case _:
                raise Exception("Invalid Search type")
        return data

    def process_text(self, text: str, doc_number: int):
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem(tokens)
        tokens = [
            self.Token(token, freq={doc_number: freq}, docs=[doc_number])
            for token, freq in self.get_freq_dist(tokens).items()
        ]
        return tokens

    def get_tokens_by_doc(self, doc_number: int):
        return [token for token in self.tokens if doc_number in token.docs]

    def __call__(
        self,
        tokenizer: Tokenizer = Tokenizer.SPLIT,
        stemmer: Stemmer = Stemmer.LANCASTER,
    ):
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
