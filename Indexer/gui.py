from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLineEdit, QTableWidget, QTableWidgetItem
from Indexer import DIR_PATH


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Search and Table Display")
        self.setGeometry(200, 200, 600, 400)

        layout = QVBoxLayout()

        # Search Bar
        self.search_bar = QLineEdit()
        # self.search_bar.returnPressed.connect(self.search)
        layout.addWidget(self.search_bar)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['Number', 'Term', 'Frequency', 'Weight'])
        layout.addWidget(self.table)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Sample data for the table
        data_dir = DIR_PATH / "results"
        data = []

        # Populate the table with sample data
        self.populate_table(data)

    def populate_table(self, data):
        self.table.setRowCount(len(data))
        for row_index, row_data in enumerate(data):
            for col_index, col_data in enumerate(row_data):
                item = QTableWidgetItem(str(col_data))
                self.table.setItem(row_index, col_index, item)

    def search(self):
        search_text = self.search_bar.text().lower()
        for row in range(self.table.rowCount()):
            self.table.setRowHidden(row, True)  # Hide all rows
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item and search_text in item.text().lower():
                    self.table.setRowHidden(row, False)  # Show row if search text is found
                    break
