import sys
from PyQt5.QtWidgets import QApplication
from pathlib import Path
from Indexer import MyWindow
from Indexer.indexer import Indexer

current_file = Path(__file__).resolve()
DIR_PATH = current_file.parent


def init_indexer():
    collection_dir = DIR_PATH / "lisa_collection"
    evaluation_dir = DIR_PATH / "lisa_eval"
    judgement = evaluation_dir / "Judgements.txt"
    queries = evaluation_dir / "Queries.txt"

    results_dir = DIR_PATH / "results"

    return Indexer(
        documents_dir=collection_dir,
        results_dir=results_dir,
        judgements_path=judgement,
        queries_path=queries,
        doc_prefix="Doc",
    )


def main():
    indexer = init_indexer()
    # check for argument of reprocess
    if len(sys.argv) > 1 and sys.argv[1] == "reprocess":
        indexer.processor.process_docs()
    app = QApplication(sys.argv)
    window = MyWindow(indexer)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
