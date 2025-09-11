from PyQt6.QtWidgets import (
	QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QLineEdit,
	QTabWidget, QMessageBox, QTableWidget, QTableWidgetItem, QSizePolicy, QFileDialog, QSplitter, QDialog
)
from PyQt6.QtGui import QTextCursor, QPixmap
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
import sys
import requests
import sseclient
import json
from markdown import markdown as md_to_html
from gguf.gguf_reader import GGUFReader
import numpy as np

url = "http://127.0.0.1:5175/v1/chat/completions"

class LLMWorker(QThread):
    result_ready = pyqtSignal(str)
    token_emit = pyqtSignal(str)

    def __init__(self, request):
        super().__init__()
        self.request = request
        self.reply = ""
        self._is_running = True

    def run(self):
        self._is_running = True
        response = requests.post(url, json=self.request, stream=True)
        client = sseclient.SSEClient(response)
        self.reply = ""
        for event in client.events():
            if event.data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(event.data)
                choices = chunk.get('choices', [])
                if not choices:
                    if "usage" in chunk or "timings" in chunk:
                        continue
                    continue
                delta = choices[0].get('delta', {})
                token = delta.get('content')
                if token:
                    self.reply += token
                    self.token_emit.emit(token)
            except (json.JSONDecodeError, KeyError):
                continue
        self.result_ready.emit(self.reply)

        if not self._is_running:
            return
        
    def stop(self):
        self._is_running = False


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LMnGen App")
        self.chatHistory = [{"role": "system", "content": "You are a helpful assistant."}]
        self.models = {}

        self.mainLayout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.initUI()
        self.initModels()
        self.mainLayout.addWidget(self.tabs)
        self.setLayout(self.mainLayout)

    def initUI(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(6)

        self.chatDisplay = QTextEdit()
        self.chatDisplay.setReadOnly(True)
        self.chatDisplay.setStyleSheet("""
QTextEdit {
    background-color: #000000;
}
QScrollBar:vertical {
    background: #f0f0f0;
    width: 12px;
    margin: 0;
    border-radius: 6px;
}
QScrollBar::handle:vertical {
    background: #6c8cd5;
    min-height: 30px;
    border-radius: 6px;
}
QScrollBar::handle:vertical:hover {
    background: #3a5bbb;
}
QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {
    height: 0px;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}
""")
        layout.addWidget(self.chatDisplay)

        inputLayout = QHBoxLayout()
        self.chatInput = QLineEdit()
        self.chatInput.returnPressed.connect(self.send_prompt)
        self.chatInput.setPlaceholderText("Type your prompt here...")

        sendBtn = QPushButton("Send")
        sendBtn.clicked.connect(self.send_prompt)

        inputLayout.addWidget(self.chatInput)
        inputLayout.addWidget(sendBtn)

        layout.addLayout(inputLayout)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Chat")

    def send_prompt(self):
        print("dooing")
        prompt = self.chatInput.text().strip()
        self.chatHistory.append({"role": "user", "content": prompt})
        self.chatInput.clear()
        self.update_chat_display()
        self.chatDisplay.append("<b>Assistant:</b> ")
        request = {"messages": self.chatHistory, "max_tokens": -1, "n_predict": -1, "stream": True}
        self.worker = LLMWorker(request)
        self.worker.token_emit.connect(lambda token: self.chatDisplay.insertPlainText(token) or self.chatDisplay.moveCursor(QTextCursor.MoveOperation.End))
        self.worker.result_ready.connect(self.handle_reply)
        self.worker.start()
        self.worker.exec()

    def handle_reply(self, reply):
        self.chatHistory.append({"role": "assistant", "content": reply.split("</think>")[1]})
        self.update_chat_display()

    def update_chat_display(self):
        self.chatDisplay.clear()
        parts = []
        for message in self.chatHistory:
            role = message['role']
            content = message['content']
            while content.startswith("\n"):
                content = content[1:]
            content = md_to_html(content, extensions=["extra", "fenced_code", "sane_lists", "nl2br"])
            if role == 'user':
                parts.append(f'<b><span style="color: blue;">User:</span></b> {content}')
            elif role == 'assistant':
                parts.append(f'<b><span style="color: orange;">Assistant:</span></b> {content}')
            elif role == 'system':
                parts.append(f'<b><i><span style="color: green;">System:</span></i></b> {content}')
            else:
                parts.append(f'<b><span style="color: red;">{role.capitalize()}:</span></b> {content}')
            #self.chatDisplay.insertPlainText(content + "\n")
        self.chatDisplay.setMarkdown("\n\n".join(parts))
        self.chatDisplay.verticalScrollBar().setValue(self.chatDisplay.verticalScrollBar().maximum())

    def initModels(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(6)

        buttonsSec = QHBoxLayout()
        addModelBtn = QPushButton("Add Model")
        addModelBtn.clicked.connect(self.add_model)
        removeModelBtn = QPushButton("Remove Model")
        removeModelBtn.clicked.connect(self.remove_model)
        buttonsSec.addWidget(addModelBtn)
        buttonsSec.addWidget(removeModelBtn)
        layout.addLayout(buttonsSec)

        try:
            with open("models.json", "r") as f:
                self.models = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.models = {}
        
        self.modelsTable = QTableWidget()
        self.modelsTable.setColumnCount(5)
        self.modelsTable.setHorizontalHeaderLabels(["Path", "Name", "Parameters","Weights", "Layers"])
        self.modelsTable.horizontalHeader().setStretchLastSection(True)
        self.modelsTable.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.modelsTable.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.modelsTable.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.modelsTable.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        for model in self.models.get("models", []):
            row = self.modelsTable.rowCount()
            self.modelsTable.insertRow(row)
            self.modelsTable.setItem(row, 0, QTableWidgetItem(model.get("path", "")))
            self.modelsTable.setItem(row, 1, QTableWidgetItem(model.get("name", "")))
            self.modelsTable.setItem(row, 2, QTableWidgetItem(str(model.get("parameters", ""))))
            self.modelsTable.setItem(row, 3, QTableWidgetItem(str(model.get("weights", ""))))
            self.modelsTable.setItem(row, 4, QTableWidgetItem(str(model.get("layers", ""))))

        layout.addWidget(self.modelsTable)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Models")

    def add_model(self):
        
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            name = "Unknown"
            parameters = "Unknown"
            weights = "Unknown"
            layers = "Unknown"

            weights_map = {
                0: "F32",
                1: "F16",
                2: "Q4_0",
                3: "Q4_1",
                4: "Q4_1_SOME_F16",
                5: "Q4_2",
                6: "Q4_3",
                7: "Q8_0",
                8: "Q5_0",
                9: "Q5_1",
                10: "Q2_K",
                11: "Q3_K_S",
                12: "Q3_K_M",
                13: "Q3_K_L",
                14: "Q4_K_S",
                15: "Q4_K_M",
                16: "Q5_K_S",
                17: "Q5_K_M",
                18: "Q6_K",
                19: "IQ2_XSS",
                20: "IQ2_XS",
                21: "Q2_K_S",
                22: "IQ3_XS",
                23: "IQ3_XXS",
                24: "IQ1_S",
                25: "IQ4_NL",
                26: "IQ3_S",
                27: "IQ3_M",
                28: "IQ2_S",
                29: "IQ2_M",
                30: "IQ4_XS",
                31: "IQ1_M",
                32: "BF16",
                33: "Q4_0_4_4",
                34: "Q4_0_4_8",
                35: "Q4_0_8_8",
                36: "TQ1_0",
                37: "TQ2_0",
                38: "MXFP4_MOE",
                145: "IQ4_KS",
                147: "IQ2_KS",
                148: "IQ4_KSS",
                150: "IQ5_KS",
                154: "IQ3_KS",
                155: "IQ2_KL",
                156: "IQ1_KT"
            }
            
            try:
                model_info = GGUFReader(file_path)
                for key, field in model_info.fields.items():
                    if key == "general.name":
                        name = str(self.maybe_decode(field.parts[field.data[0]]))
                    elif key == "general.size_label":
                        parameters = str(self.maybe_decode(field.parts[field.data[0]]))
                    elif key == "general.file_type":
                        weights = weights_map.get(self.maybe_decode(field.parts[field.data[0]]), f"Unknown ({field.data[0]})")
                    elif key.endswith("block_count"):
                        layers = str(self.maybe_decode(field.parts[field.data[0]]))

                self.models.setdefault("models", []).append({
                    "path": file_path,
                    "name": name,
                    "parameters": parameters,
                    "weights": weights,
                    "layers": layers
                })
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read model info: {e}")
                return
        self.refresh_models_table()
    
    def maybe_decode(self, value):
        # bytes or bytearray -> try UTF-8
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", errors="replace")

        # numpy array of uint8 -> bytes -> UTF-8
        if isinstance(value, np.ndarray) and value.dtype == np.uint8:
            return value.tobytes().decode("utf-8", errors="replace")

        # list/tuple of ints 0..255 -> bytes -> UTF-8
        if isinstance(value, (list, tuple)) and value and all(isinstance(x, (int, np.integer)) and 0 <= int(x) <= 255 for x in value):
            return bytes(value).decode("utf-8", errors="replace")

        # list/tuple of bytes -> decode each
        if isinstance(value, (list, tuple)) and value and all(isinstance(x, (bytes, bytearray)) for x in value):
            return [x.decode("utf-8", errors="replace") for x in value]

        return value[0]

    def remove_model(self):
        selected_rows = self.modelsTable.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "Warning", "No model selected.")
            return
        row = selected_rows[0].row()
        del self.models["models"][row]
        self.refresh_models_table()

    def refresh_models_table(self):
        self.modelsTable.setRowCount(0)
        if self.models.get("models") is None:
            return
        for model in self.models.get("models", []):
            row = self.modelsTable.rowCount()
            self.modelsTable.insertRow(row)
            self.modelsTable.setItem(row, 0, QTableWidgetItem(model.get("path", "")))
            self.modelsTable.setItem(row, 1, QTableWidgetItem(model.get("name", "")))
            self.modelsTable.setItem(row, 2, QTableWidgetItem(str(model.get("parameters", ""))))
            self.modelsTable.setItem(row, 3, QTableWidgetItem(str(model.get("weights", ""))))
            self.modelsTable.setItem(row, 4, QTableWidgetItem(str(model.get("layers", ""))))
        with open("models.json", "w") as f:
            json.dump(self.models, f, indent=4)

if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = App()
	window.resize(1000, 600)
	window.show()
	sys.exit(app.exec())