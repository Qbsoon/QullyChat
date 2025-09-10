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

        self.mainLayout = QVBoxLayout()
        self.initUI()
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
        self.mainLayout.addWidget(widget)

    def send_prompt(self):
        print("dooing")
        prompt = self.chatInput.text().strip()
        self.chatHistory.append({"role": "user", "content": prompt})
        self.chatInput.clear()
        self.update_chat_display()
        self.chatDisplay.append("<b>Assistant:</b> ")
        #QApplication.processEvents()
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

if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = App()
	window.resize(1000, 600)
	window.show()
	sys.exit(app.exec())