from Indexer.processor import TextProcessor, PATH_TEMPLATE, Tokenizer, Stemmer
from pathlib import Path

current_file = Path(__file__).resolve()
DIR_PATH = current_file.parent.parent
collection_dir = DIR_PATH / "Collection"


results_dir = DIR_PATH / "results"

# delete results directory if exists and create a new one
if results_dir.exists() and results_dir.is_dir():
    for file in results_dir.iterdir():
        file.unlink()
    results_dir.rmdir()
results_dir.mkdir()


docs = [file.resolve() for file in collection_dir.iterdir() if file.suffix == ".txt"]
processor = TextProcessor(docs)