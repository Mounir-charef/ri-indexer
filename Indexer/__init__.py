from Indexer.processor import TextProcessor, PATH_TEMPLATE
from pathlib import Path

current_file = Path(__file__).resolve()
DIR_PATH = current_file.parent.parent
collection_dir = DIR_PATH / "Collection"


docs = [file.resolve() for file in collection_dir.iterdir() if file.suffix == ".txt"]
processor = TextProcessor(docs)