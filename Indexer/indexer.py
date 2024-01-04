import math
from collections import defaultdict
from pathlib import Path
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
        docs_prefix="",
    ):
        self.processor = TextProcessor(
            documents_dir,
            results_dir,
            docs_prefix=docs_prefix,
        )
        self.judgements_path = judgements_path
        self.queries_path = queries_path

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
            ri.append(len(current_relevant) / len(relevant_docs))

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
                    if token in query:
                        results.append([doc_number, token, freq, weight])

            case SearchType.TERM:
                query = self.processor.process_text(query.lower())
                for token, doc_number, freq, weight in self.file_generator(file_type):
                    if token in query:
                        results.append([token, doc_number, freq, weight])

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

            case _:
                raise ValueError(f"Invalid search type: {search_type}")
        return results
