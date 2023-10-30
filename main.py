import sys
from PyQt5.QtWidgets import QApplication
from Indexer.gui import MyWindow


def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

