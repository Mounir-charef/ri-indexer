import math
import os
import re
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import TypedDict
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer, LancasterStemmer


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

    def __init__(
        self,
        documents_dir: Path,
        results_dir: Path,
        *,
        judgements_path: Path,
        queries_path: Path,
        docs_prefix="",
    ):
        if results_dir.exists() and results_dir.is_dir():
            for file in results_dir.iterdir():
                file.unlink()
            results_dir.rmdir()
        results_dir.mkdir()
        self._tokenizer: Tokenizer | None = None
        self._stemmer: Stemmer | None = None
        self.docs: [str] = [
            file.read_text().lower()
            for file in documents_dir.iterdir()
            if file.suffix == f"{docs_prefix}.txt"
        ]
        self.tokens: dict[str, TextProcessor.Token] = {}
        self.file_path = results_dir
        self.judgements_path = judgements_path
        self.queries_path = queries_path
        self._tokens_by_doc = defaultdict(dict)

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
        """
        Get token by value
        :param text:
        :return:
        """
        return self.tokens.get(text)

    def get_token_by_doc_number(self, doc_number: int):
        """
        Get token by doc number
        :param doc_number:
        :return:
        """
        return self.tokens_by_doc.get(doc_number)

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
        retrieved_docs = []
        if search_type == SearchType.TERM:
            for doc in results:
                retrieved_docs.append(doc[1])
        else:
            for doc in results:
                retrieved_docs.append(doc[0])

        # calculate precision, recall and f1-score
        precision = (
            len(relevant_docs.intersection(retrieved_docs)) / len(retrieved_docs)
            if len(retrieved_docs)
            else 0
        )
        precision_5 = len(relevant_docs.intersection(retrieved_docs[:5])) / 5
        precision_10 = len(relevant_docs.intersection(retrieved_docs[:10])) / 10
        recall = (
            len(relevant_docs.intersection(retrieved_docs)) / len(relevant_docs)
            if len(relevant_docs)
            else 0
        )
        f1_score = (
            2 * precision * recall / (precision + recall) if precision + recall else 0
        )

        # get curve

        if len(retrieved_docs) > 10:
            ranks = retrieved_docs[:10]
        else:
            ranks = retrieved_docs + [-1] * (10 - len(retrieved_docs))

        pi = []
        ri = []
        current_relevant = set()
        for i in range(len(ranks)):
            if ranks[i] in relevant_docs:
                current_relevant.add(ranks[i])
            pi.append(len(current_relevant) / (i + 1))
            ri.append(len(current_relevant) / len(relevant_docs))

        pj = []
        rj = [i / 10 for i in range(0, 11)]
        i = 0
        current = max(pi)
        for j in range(len(ranks) + 1):
            if ri[i] >= rj[j]:
                pj.append(current)
            else:
                while i < len(ri) and ri[i] < rj[j]:
                    i += 1
                if i < 10:
                    current = max(pi[i:])
                else:
                    current = 0
                pj.append(current)
        return {
            "Precision": round(precision, 4),
            "P@5": round(precision_5, 4),
            "P@10": round(precision_10, 4),
            "Recall": round(recall, 4),
            "F1 score": round(f1_score, 4),
        }, {"recall": rj, "precision": pj}

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
                    docs_size[doc] = sum(
                        [token["freq"][doc] for token in current_tokens.values()]
                    )

                average_doc_size = sum(docs_size.values()) / len(self.docs)
                rsv = defaultdict(float)
                for token in tokens:
                    for doc_number in token["docs"]:
                        rsv[doc_number] += (
                            token["freq"][doc_number]
                            / (
                                k
                                * (
                                    (1 - b)
                                    + b * (docs_size[doc_number] / average_doc_size)
                                )
                                + token["freq"][doc_number]
                            )
                        ) * math.log10(
                            (len(self.docs) - len(token["docs"]) + 0.5)
                            / (len(token["docs"]) + 0.5)
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
                                if token == word:
                                    positive_result += 1

                        for word in negative:
                            for token in tokens:
                                if token == word:
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
        processed_tokens: dict[str, TextProcessor.Token] = {}
        for token, freq in self.get_freq_dist(tokens).items():
            if token not in processed_tokens:
                processed_tokens[token] = {
                    "freq": {doc_number: freq},
                    "docs": [doc_number],
                    "weight": {doc_number: 0},
                }
            else:
                processed_tokens[token]["freq"][doc_number] = freq
                processed_tokens[token]["docs"].append(doc_number)
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

    def __call__(
        self,
        tokenizer: Tokenizer = Tokenizer.SPLIT,
        stemmer: Stemmer = Stemmer.LANCASTER,
    ):
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.tokens = {}
        for i, text in enumerate(self.docs):
            doc_number = i + 1
            tokens = self.process_text(text, doc_number)
            self.add_tokens(tokens)
        for token in self.tokens:
            self.calculate_weight(token)
        self.save()
