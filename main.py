from Indexer.processor import TextProcessor
import os

DIR_PATH = os.path.join(os.path.dirname(__file__), "Collection")


def main():
    docs = [os.path.abspath(os.path.join(DIR_PATH, file)) for file in os.listdir(DIR_PATH) if file.endswith(".txt")]
    proc = TextProcessor(docs)
    proc(stemmer='porter')


if __name__ == '__main__':
    main()
