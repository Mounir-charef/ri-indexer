from Indexer import PATH_TEMPLATE, DIR_PATH, processor, Stemmer, Tokenizer
from PyQt5.QtWidgets import (
    QDesktopWidget, QMainWindow, QSizePolicy, QVBoxLayout, QWidget, QLineEdit,
    QTableWidget, QTableWidgetItem, QPushButton, QHBoxLayout, QCheckBox, QApplication,
    QGroupBox, QRadioButton, QButtonGroup
)


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("RI Indexer")
        self.setStyleSheet("""
                    QMainWindow {
                        background-color: #f4f4f4;
                    }
                    QGroupBox {
                        background-color: #E0E0E0;
                        border: 1px solid #ccc;
                        border-radius: 8px;
                        margin: 10px;
                        padding: 12px;
                    }
                    QLineEdit {
                        padding: 8px;
                        border: 1px solid #ccc;
                        border-radius: 5px;
                    }
                    QLineEdit:focus {
                        border: 1px solid #3498DB;
                    }
                    QPushButton {
                        background-color: #3498DB;
                        color: white;
                        border: none;
                        padding: 10px 20px;
                        text-align: center;
                        text-decoration: none;
                        display: inline-block;
                        font-size: 14px;
                        margin: 8px 4px;
                        cursor: pointer;
                        border-radius: 5px;
                    }
                    QPushButton:hover {
                        background-color: #2980B9;
                    }
                    QTableWidget {
                        background-color: #FFFFFF;
                        border: 1px solid #ccc;
                        border-radius: 8px;
                        padding: 8px;
                        selection-background-color: #3498DB;
                        selection-color: white;
                    }
                    
                """)
        self.setGeometry(400, 400, 900, 900)

        layout = QVBoxLayout()

        # Search Section
        search_layout = QHBoxLayout()
        self.search_bar = QLineEdit()
        search_layout.addWidget(self.search_bar)
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.search)
        search_layout.addWidget(search_button)
        layout.addLayout(search_layout)

        # Processing and Indexer Sections
        processing_indexer_layout = QHBoxLayout()

        # Processing Section
        processing_group = QGroupBox("Processing")
        processing_group.setStyleSheet("QGroupBox { border: 1px solid gray; border-radius: 3px; margin-top: 10px; padding: 10px; }")
        processing_layout = QVBoxLayout()
        self.tokenization_checkbox = QCheckBox("Tokenization")
        self.porter_stemmer_checkbox = QCheckBox("Porter Stemmer")
        processing_layout.addWidget(self.tokenization_checkbox)
        processing_layout.addWidget(self.porter_stemmer_checkbox)
        processing_group.setLayout(processing_layout)
        processing_indexer_layout.addWidget(processing_group)

        # Indexer Section
        indexer_group = QGroupBox("Indexer")
        indexer_group.setStyleSheet("QGroupBox { border: 1px solid gray; border-radius: 3px; margin-top: 10px; padding: 10px; }")
        indexer_layout = QVBoxLayout()
        self.indexer_docs_radio = QRadioButton("DOCS")
        self.indexer_terms_radio = QRadioButton("Terms")
        self.indexer_docs_radio.setChecked(True)
        self.indexer_radio_group = QButtonGroup()
        self.indexer_radio_group.addButton(self.indexer_docs_radio)
        self.indexer_radio_group.addButton(self.indexer_terms_radio)
        indexer_layout.addWidget(self.indexer_docs_radio)
        indexer_layout.addWidget(self.indexer_terms_radio)
        indexer_group.setLayout(indexer_layout)
        processing_indexer_layout.addWidget(indexer_group)

        layout.addLayout(processing_indexer_layout)

        run_button = QPushButton("Run")
        run_button.clicked.connect(self.run)
        layout.addWidget(run_button)

        self.table = QTableWidget()
        self.table.setSortingEnabled(True)
        self.table.setColumnCount(4)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.table)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.run()
        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def populate_table(self):
        tokenizer = Tokenizer.NLTK if self.tokenization_checkbox.isChecked() else Tokenizer.SPLIT
        stemmer = Stemmer.PORTER if self.porter_stemmer_checkbox.isChecked() else Stemmer.LANCASTER
        file_type = "descriptor" if self.indexer_docs_radio.isChecked() else "inverse"
        file = DIR_PATH / PATH_TEMPLATE.format(file_type=file_type, stemmer=stemmer.value.capitalize(), tokenizer=tokenizer.value.capitalize())
        if file_type == "descriptor":
            self.table.setHorizontalHeaderLabels(['N°doc ', 'Term', 'Frequency', 'Weight'])
        else:
            self.table.setHorizontalHeaderLabels(['Term', 'N°doc', 'Frequency', 'Weight'])
        self.setDisabled(True)
        QApplication.processEvents()

        with open(file, 'r') as f:
            data = [line.split() for line in f.readlines()]

        self.table.setRowCount(len(data))
        for row_index, row_data in enumerate(data):
            for col_index, col_data in enumerate(row_data):
                item = QTableWidgetItem(str(col_data))
                self.table.setItem(row_index, col_index, item)

        self.setEnabled(True)

    def search(self):
        search_text = self.search_bar.text().lower()
        for row in range(self.table.rowCount()):
            row_hidden = True
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item and search_text in item.text().lower():
                    row_hidden = False
                    break

            self.table.setRowHidden(row, row_hidden)

    def run(self):
        tokenizer = Tokenizer.NLTK if self.tokenization_checkbox.isChecked() else Tokenizer.SPLIT
        stemmer = Stemmer.PORTER if self.porter_stemmer_checkbox.isChecked() else Stemmer.LANCASTER

        self.setDisabled(True)
        QApplication.processEvents()

        processor(tokenizer=tokenizer, stemmer=stemmer)
        processor.save()
        self.populate_table()
        self.search()
