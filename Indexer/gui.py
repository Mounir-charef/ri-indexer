from Indexer import processor, Stemmer, Tokenizer
from Indexer.processor import FileType, SearchType, MatchingType
from PyQt5.QtWidgets import (
    QDesktopWidget, QMainWindow, QSizePolicy, QVBoxLayout, QWidget, QLineEdit,
    QTableWidget, QTableWidgetItem, QPushButton, QHBoxLayout, QCheckBox, QApplication,
    QGroupBox, QRadioButton, QButtonGroup, QLabel, QComboBox
)
from PyQt5.QtGui import QDoubleValidator

FILTERS_PARAMS = {
    'DOCS': {
        'file_type': FileType.DESCRIPTOR,
        'search_type': SearchType.DOCS,
        'row_labels': ['N°doc ', 'Term', 'Frequency', 'Weight']
    },
    'Terms': {
        'file_type': FileType.INVERSE,
        'search_type': SearchType.TERM,
        'row_labels': ['Term ', 'N°doc', 'Frequency', 'Weight']
    },
    'Vector Space Model': {
        'file_type': FileType.DESCRIPTOR,
        'search_type': SearchType.VECTOR,
        'row_labels': ['N°doc', 'Relevance']
    },
    'Probability Model': {
        'file_type': FileType.DESCRIPTOR,
        'search_type': SearchType.PROBABILITY,
        'row_labels': ['N°doc', 'Relevance'],
        'matching_params': {'K': 2.0, 'B': 1.5}
    },
    'Logic Model': {
        'file_type': FileType.DESCRIPTOR,
        'search_type': SearchType.LOGIC,
        'row_labels': ['N°doc', 'Relevance']
    }
}


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
        float_validator = QDoubleValidator()

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
        processing_group.setStyleSheet(
            "QGroupBox { border: 1px solid gray; border-radius: 3px; margin-top: 10px; padding: 10px; }")
        processing_layout = QVBoxLayout()
        self.tokenization_checkbox = QCheckBox("Tokenization")
        self.porter_stemmer_checkbox = QCheckBox("Porter Stemmer")
        self.tokenization_checkbox.setChecked(True)
        self.porter_stemmer_checkbox.setChecked(True)
        processing_layout.addWidget(self.tokenization_checkbox)
        processing_layout.addWidget(self.porter_stemmer_checkbox)
        processing_group.setLayout(processing_layout)
        processing_indexer_layout.addWidget(processing_group)

        # Indexer Section
        indexer_group = QGroupBox("Indexer")
        indexer_group.setStyleSheet(
            "QGroupBox { border: 1px solid gray; border-radius: 3px; margin-top: 10px; padding: 10px; }")
        indexer_layout = QVBoxLayout()
        self.indexer_docs_radio = QRadioButton("DOCS")
        self.indexer_terms_radio = QRadioButton("Terms")
        self.indexer_terms_radio.setChecked(True)
        self.indexer_radio_group = QButtonGroup()
        self.indexer_radio_group.addButton(self.indexer_docs_radio)
        self.indexer_radio_group.addButton(self.indexer_terms_radio)
        self.indexer_radio_group.buttonClicked.connect(self.search)
        indexer_layout.addWidget(self.indexer_docs_radio)
        indexer_layout.addWidget(self.indexer_terms_radio)
        indexer_group.setLayout(indexer_layout)
        processing_indexer_layout.addWidget(indexer_group)

        layout.addLayout(processing_indexer_layout)

        matching_group = QGroupBox("Matching")
        matching_group.setStyleSheet(
            "QGroupBox { border: 1px solid gray; border-radius: 3px; margin-top: 10px; padding: 10px; }")
        matching_layout = QVBoxLayout()

        # Checkbox to enable/disable matching parameters
        self.matching_checkbox = QCheckBox("Enable Matching")
        matching_layout.addWidget(self.matching_checkbox)

        self.models_radio_group = QButtonGroup()

        self.vector_model_radio = QRadioButton("Vector Space Model")
        self.vector_model_radio.setChecked(True)
        self.models_radio_group.addButton(self.vector_model_radio)
        matching_layout.addWidget(self.vector_model_radio)

        matching_type_label = QLabel("Matching Type:")
        matching_layout.addWidget(matching_type_label)

        self.matching_form_combobox = QComboBox()
        self.matching_form_combobox.addItems(MatchingType.list())
        matching_layout.addWidget(self.matching_form_combobox)

        self.probability_model_radio = QRadioButton("Probability Model")
        self.models_radio_group.addButton(self.probability_model_radio)
        matching_layout.addWidget(self.probability_model_radio)

        default_k_value = str(FILTERS_PARAMS['Probability Model']['matching_params']['K'])
        self.k_parameter_edit = QLineEdit(default_k_value)
        self.k_parameter_edit.setValidator(float_validator)
        self.k_parameter_edit.setPlaceholderText("Enter K value")
        self.k_parameter_edit.textChanged.connect(self.update_k_parameter)
        matching_layout.addWidget(self.k_parameter_edit)

        default_b_value = str(FILTERS_PARAMS['Probability Model']['matching_params']['B'])
        self.b_parameter_edit = QLineEdit(default_b_value)
        self.b_parameter_edit.setValidator(float_validator)
        self.b_parameter_edit.setPlaceholderText("Enter B value")
        self.b_parameter_edit.textChanged.connect(self.update_b_parameter)
        matching_layout.addWidget(self.b_parameter_edit)

        self.logic_model_radio = QRadioButton("Logic Model")
        self.models_radio_group.addButton(self.logic_model_radio)
        matching_layout.addWidget(self.logic_model_radio)

        matching_group.setLayout(matching_layout)
        processing_indexer_layout.addWidget(matching_group)

        run_button = QPushButton("Run")
        run_button.clicked.connect(self.run)
        layout.addWidget(run_button)

        self.table = QTableWidget()
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

    def update_k_parameter(self):
        try:
            FILTERS_PARAMS['Probability Model']['matching_params']['K'] = float(self.k_parameter_edit.text())
        except ValueError:
            pass

    def update_b_parameter(self):
        try:
            FILTERS_PARAMS['Probability Model']['matching_params']['B'] = float(self.b_parameter_edit.text())
        except ValueError:
            pass

    def search(self):
        query = self.search_bar.text()
        if self.matching_checkbox.isChecked():
            index_type = self.models_radio_group.checkedButton()
            options = FILTERS_PARAMS[index_type.text()]
            if index_type.text() == 'Vector Space Model':
                match_form = MatchingType(self.matching_form_combobox.currentText())
                options['matching_form'] = match_form
        else:
            index_type = self.indexer_radio_group.checkedButton().text()
            options = FILTERS_PARAMS[index_type]
        self.table.setColumnCount(len(options['row_labels']))
        self.table.setHorizontalHeaderLabels(options['row_labels'])
        data = processor.search_in_file(query, **options)
        self.table.setRowCount(len(data))
        for row_index, row_data in enumerate(data):
            for col_index, col_data in enumerate(row_data):
                item = QTableWidgetItem(str(col_data))
                self.table.setItem(row_index, col_index, item)

    def run(self):
        tokenizer = Tokenizer.NLTK if self.tokenization_checkbox.isChecked() else Tokenizer.SPLIT
        stemmer = Stemmer.PORTER if self.porter_stemmer_checkbox.isChecked() else Stemmer.LANCASTER

        self.setDisabled(True)

        QApplication.processEvents()
        processor(tokenizer=tokenizer, stemmer=stemmer)
        processor.save()
        self.search()

        self.setEnabled(True)
