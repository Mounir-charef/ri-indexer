import math
import re
from collections import defaultdict
from pathlib import Path
from functools import cached_property

from Indexer.processor import (
    TextProcessor,
    FileType,
    SearchType,
    MatchingType,
    Tokenizer,
    Stemmer,
)


class Indexer:
    def __init__(
            self,
            documents_dir: Path,
            results_dir: Path,
            *,
            judgements_path: Path,
            queries_path: Path,
            doc_prefix: str = "D",
    ):
        self.processor = TextProcessor(
            documents_dir,
            results_dir,
            doc_prefix=doc_prefix,
        )
        self.judgements_path = judgements_path
        self.queries_path = queries_path

    @property
    def judgements(self) -> list[list[str]]:
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

    @cached_property
    def get_freq_by_doc(self):
        """
            Get the frequency of each token in each document using the descriptor file and regex
        :return: a dictionary of the form {doc_number: {token: freq}}
        """
        freq_by_doc = defaultdict(dict)
        for doc_number, token, freq, _ in self.file_generator(FileType.DESCRIPTOR):
            freq_by_doc[doc_number][token] = int(freq)
        return freq_by_doc

    def file_generator(self, file_type: FileType):
        file_path = (
            self.processor.descriptor_file_path
            if file_type.value == "descriptor"
            else self.processor.inverse_file_path
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
            ri.append(
                len(current_relevant) / len(relevant_docs) if len(relevant_docs) else 0
            )

        pj = []
        rj = [i / 10 for i in range(0, 11)]
        i = 0
        current = max(pi)
        for j in range(len(ranks) + 1):
            if ri[i] >= rj[j]:
                pj.append(current)
            else:
                while i < len(ri) - 1 and ri[i] < rj[j]:
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

    def __call__(
            self,
            query: str,
            search_type: SearchType,
            *,
            file_type: FileType = FileType.INVERSE,
            tokenizer: Tokenizer = Tokenizer.NLTK,
            stemmer: Stemmer = Stemmer.PORTER,
            matching_type: MatchingType = MatchingType.Scalar,
            **kwargs,
    ):
        self.processor.set_processor(tokenizer, stemmer)
        if not query and search_type not in [
            SearchType.VECTOR,
            SearchType.PROBABILITY,
            SearchType.LOGIC,
        ]:
            file_path = (
                self.processor.inverse_file_path
                if file_type == FileType.INVERSE
                else self.processor.descriptor_file_path
            )
            with open(file_path, "r") as f:
                data = [line.split() for line in f.readlines()]
            return data

        # calculate formulas
        results = []
        match search_type:
            case SearchType.DOCS:
                query = self.processor.process_text(query.lower())
                for doc_number, token, freq, weight in self.file_generator(file_type):
                    if doc_number in query:
                        results.append([doc_number, token, freq, weight])
                results.sort(key=lambda row: row[0])

            case SearchType.TERM:
                query = self.processor.process_text(query.lower())
                for token, doc_number, freq, weight in self.file_generator(file_type):
                    if token in query:
                        results.append([token, doc_number, freq, weight])
                results.sort(key=lambda row: row[0])

            case SearchType.VECTOR:
                query = self.processor.process_text(query.lower())
                total_weight = defaultdict(list)
                doc_weights = defaultdict(list)

                for doc_number, token, freq, weight in self.file_generator(file_type):
                    doc_weights[doc_number].append(float(weight) ** 2)
                    if token in query:
                        total_weight[doc_number].append(float(weight))

                for doc_number, weights in total_weight.items():
                    match matching_type:
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
                    results.append([doc_number, round(weight, 4)])
                results.sort(key=lambda row: row[1], reverse=True)

            case SearchType.PROBABILITY:
                query = self.processor.process_text(query.lower())
                k, b = float(kwargs["matching_params"].get("K", 2)), float(
                    kwargs["matching_params"].get("B", 1.5)
                )
                freq_by_doc = self.get_freq_by_doc
                docs_size = {
                    doc_number: sum(freq_by_doc[doc_number].values())
                    for doc_number in freq_by_doc
                }
                average_doc_size = sum(docs_size.values()) / len(docs_size)
                rsv = defaultdict(float)
                num_of_docs = len(self.processor.docs)
                num_doc_with_token = defaultdict(int)
                for token in set(query):
                    for doc_number in freq_by_doc:
                        if token in freq_by_doc[doc_number]:
                            num_doc_with_token[token] += 1
                for doc_number in freq_by_doc:
                    for token in query:
                        if token in freq_by_doc[doc_number]:
                            rsv[doc_number] += math.log10(
                                (num_of_docs - num_doc_with_token[token] + 0.5)
                                / (num_doc_with_token[token] + 0.5)
                            ) * (
                                                       (freq_by_doc[doc_number][token])
                                                       / (
                                                               freq_by_doc[doc_number][token]
                                                               + k
                                                               * (
                                                                       1
                                                                       - b
                                                                       + b * docs_size[doc_number] / average_doc_size
                                                               )
                                                       )
                                               )
                for doc_number, weight in rsv.items():
                    results.append([doc_number, round(weight, 4)])
                results.sort(key=lambda row: row[1], reverse=True)

            case SearchType.LOGIC:

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
                    return results

                or_parts = [part.strip() for part in query.strip().split("OR")]
                bool_results = defaultdict(bool)
                tokens_in_docs = defaultdict(set)
                for doc_number, token, _, _ in self.file_generator(file_type):
                    tokens_in_docs[doc_number].add(token)
                for part in or_parts:
                    must_part = part.split("AND")
                    positive = [
                        self.processor.stem_word(term.lower().strip())
                        for term in must_part
                        if not term.strip().startswith("NOT")
                    ]
                    negative = [
                        self.processor.stem_word(term.strip().split()[-1])
                        for term in must_part
                        if term.strip().startswith("NOT")
                    ]
                    for doc_number, tokens in tokens_in_docs.items():
                        if all(token in tokens for token in positive) and all(
                                token not in tokens for token in negative
                        ):
                            bool_results[doc_number] = True

                results = sorted(
                    list(bool_results.items()), key=lambda row: int(row[0])
                )
            case _:
                raise ValueError(f"Invalid search type: {search_type}")
        return results
