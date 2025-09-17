import os
import signal
from PyQt6.QtWidgets import (
	QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QLineEdit,
	QTabWidget, QMessageBox, QTableWidget, QTableWidgetItem, QSizePolicy, QFileDialog, QSplitter,
    QDialog, QListWidget, QListWidgetItem, QInputDialog, QComboBox, QCheckBox, QSlider, QFrame,
    QRadioButton
)
from PyQt6.QtGui import QTextCursor, QPixmap, QCursor, QIntValidator
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QItemSelectionModel, QEvent, QPoint
import sys
import requests
import sseclient
import json
from markdown import markdown as md_to_html
from gguf.gguf_reader import GGUFReader
import numpy as np
import subprocess
import atexit
import threading

url = "http://127.0.0.1:5175/v1/chat/completions"

class Llama_cpp(QThread):
    def __init__(self, options):
        super().__init__()
        self.options = options
        self._is_running = True

    def run(self):
        self._is_running = True
        command = ["./llama/llama-server", "-m", self.options['model_path'], "--host", "127.0.0.1", "--port", str(self.options['port']), "-n", "-1"]
        if self.options['threads'] > 0:
            command += ["-t", str(self.options['threads'])]
        if self.options['gpu_layers'] > 0:
            command += ["--n-gpu-layers", str(self.options['gpu_layers'])]
        if self.options['batch_size'] > 0:
            command += ["-b", str(self.options['batch_size'])]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True, encoding="utf-8")

        atexit.register(self.process.terminate)

    def stop(self):
        self._is_running = False
        try:
            try:
                self.process.send_signal(signal.SIGINT)
                self.process.wait(timeout=1)
                return
            except subprocess.TimeoutExpired:
                pass
            
            try:
                self.process.terminate()
                self.process.wait(timeout=1)
                return
            except subprocess.TimeoutExpired:
                pass

            self.process.kill()
            self.process.wait(timeout=1)
        except Exception as e:
            print(f"Error terminating llama.cpp server: {e}")

class LLMWorker(QThread):
    result_ready = pyqtSignal(str)
    token_emit = pyqtSignal(str)
    error_emit = pyqtSignal(str)

    def __init__(self, request):
        super().__init__()
        self.request = request
        self.reply = ""
        self._is_running = True

    def run(self):
        self._is_running = True
        try:
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
        except Exception as e:
            self.error_emit.emit(str(e))
        
    def stop(self):
        self._is_running = False

class GGUFInfoWoker(QThread):
    info_ready = pyqtSignal(dict)

    def __init__(self, model_path):
        super().__init__()
        self._is_running = True
        self.model_path = model_path
        self.weights_map = {
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

    def run(self):
        self._is_running = True
        info = {}
        try:
            model_info = GGUFReader(self.model_path)
            info.update({"path": self.model_path})
            for key, field in model_info.fields.items():
                if key == "general.name":
                    info.update({"name": str(self.maybe_decode(field.parts[field.data[0]]))})
                elif key == "general.size_label":
                    info.update({"parameters": str(self.maybe_decode(field.parts[field.data[0]]))})
                elif key == "general.file_type":
                    info.update({"weights": self.weights_map.get(self.maybe_decode(field.parts[field.data[0]]), f"Unknown ({field.data[0]})")})
                elif key.endswith("block_count"):
                    info.update({"layers": str(self.maybe_decode(field.parts[field.data[0]]))})
            if "name" not in info:
                info["name"] = "Unknown"
            if "parameters" not in info:
                info["parameters"] = "Unknown"
            if "weights" not in info:
                info["weights"] = "Unknown"
            if "layers" not in info:
                info["layers"] = "Unknown"
        except Exception as e:
            info = {"path": self.model_path, "name": "Error", "parameters": "Error", "weights": "Error", "layers": "Error"}
        
        self.info_ready.emit(info)

        if not self._is_running:
            return
        
    def stop(self):
        self._is_running = False
    
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

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setMinimumSize(600, 338)
        self.setWindowTitle("Qully Chat")

        self.chatHistory = []
        self.chatLegacyHistory = []
        self.models = []

        self.LLMSettings = {}
        self.bpSettings = [
            {'type': 'radiobutton', 'name': 'model_settings', 'display': 'Use model settings', 'default': False, 'use_case': [1]},
            {'type': 'text', 'name': 'address', 'display': 'Address', 'default': '127.0.0.1', 'use_case': [0]},
            {'type': 'number', 'name': 'port', 'display': 'Port', 'default': '5175', 'use_case': [0]},
            {'type': 'slider', 'name': 'threads', 'display': 'CPU Threads', 'default': "-1", 'min': 1, 'max': os.cpu_count(), 'use_case': [0, 1]},
            {'type': 'combo', 'name': 'gpu_layers', 'display': 'Layers on GPU', 'default': "All", 'options': ["Auto", "All", "0"], 'use_case': [0]},
            {'type': 'slider', 'name': 'gpu_layers', 'display': 'Layers on GPU', 'default': "-1", 'min': 0, 'max': 0, 'use_case': [1]},
            {'type': 'number', 'name': 'batch_size', 'display': 'Batch size', 'default': "512", 'use_case': [0, 1]},
            {'type': 'text', 'name': 'system_prompt', 'display': 'System prompt', 'default': 'You are a helpful assistant.', 'use_case': [0, 1]}
        ]   # 0: llm settings tab; 1: llm model-specific settings

        self.mainLayout = QVBoxLayout()
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)
        self.tabs = QTabWidget()
        self.initTopBar()
        self.initChat()
        self.initModels()
        self.initLLMSettings()
        self.mainLayout.addWidget(self.tabs)
        self.setLayout(self.mainLayout)

        QApplication.instance().installEventFilter(self)
        self.enable_mouse_tracking(self)

    def initTopBar(self):
        self.topBar = QWidget(self)
        self.topBar.setObjectName("Qully Chat")
        self.topBar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self.topBar)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        leftPart = QHBoxLayout()
        label = QLabel("Qully Chat")
        leftPart.addWidget(label)
        layout.addLayout(leftPart)
        layout.addStretch()

        centerPart = QHBoxLayout()
        self.modelSelect = QComboBox(self.topBar)
        self.modelSelect.setToolTip("Select Model")
        self.modelSelect.setFixedHeight(24)
        self.modelSelect.setPlaceholderText("Select LLM Model")
        self.modelSelect.activated.connect(self.model_changed)
        self.modelSelect.installEventFilter(self)
        centerPart.addWidget(self.modelSelect)

        modelStopBtn = QPushButton("⏏")
        modelStopBtn.setToolTip("Stop LLM Server")
        modelStopBtn.setFixedSize(24, 18)
        modelStopBtn.clicked.connect(self.stop_llama_server)
        centerPart.addWidget(modelStopBtn)

        layout.addLayout(centerPart)
        layout.addStretch()

        rightPart = QHBoxLayout()
        self.minimizeBtn = QPushButton("-", self.topBar)
        self.minimizeBtn.setToolTip("Minimize")
        self.minimizeBtn.setFixedSize(32, 24)
        self.minimizeBtn.clicked.connect(self.minimize)
        rightPart.addWidget(self.minimizeBtn)

        self.maximizeBtn = QPushButton("❐", self.topBar)
        self.maximizeBtn.setToolTip("Maximize")
        self.maximizeBtn.setFixedSize(32, 24)
        self.maximizeBtn.clicked.connect(self.toggle_maximize)
        rightPart.addWidget(self.maximizeBtn)

        self.closeBtn = QPushButton("×", self.topBar)
        self.closeBtn.setToolTip("Close")
        self.closeBtn.setFixedSize(32, 24)
        self.closeBtn.clicked.connect(self.closeApp)
        rightPart.addWidget(self.closeBtn)

        layout.addLayout(rightPart)

        self.topBar.installEventFilter(self)
        self.mainLayout.addWidget(self.topBar)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Show and obj.metaObject().className() == "QComboBoxPrivateContainer":
            c = obj.parent()
            if isinstance(c, QComboBox) and c is getattr(self, "modelSelect", None):
                obj.move(c.mapToGlobal(QPoint(0, c.height()))); obj.setMinimumWidth(c.width())
        
        if obj is self.topBar:
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                pos_in_win = self.topBar.mapTo(self, event.position().toPoint())
                if not self.isMaximized() and self.hit_test_edges(pos_in_win):
                    pass
                else:
                    win = self.windowHandle()
                    if win:
                        win.startSystemMove()
                        return True
            if event.type() == QEvent.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton:
                pos_in_win = self.topBar.mapTo(self, event.position().toPoint())
                if not self.hit_test_edges(pos_in_win):
                    self.showNormal() if self.isMaximized() else self.showMaximized()
                return True
            
        if isinstance(obj, QWidget) and obj.window() is self:
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                win = self.windowHandle()
                if win and not self.isMaximized():
                    pos_in_win = obj.mapTo(self, event.position().toPoint())
                    edges = self.hit_test_edges(pos_in_win)
                    if edges:
                        win.startSystemResize(edges)
                        return True

            if event.type() in (QEvent.Type.MouseMove, QEvent.Type.HoverMove):
                if hasattr(event, 'position'):
                    pos_in_win = self.mapTo(self, event.position().toPoint())
                else:
                    pos_in_win = self.mapFromGlobal(QCursor.pos())
                edges = self.hit_test_edges(pos_in_win)
                if self.isMaximized():
                    self.unsetCursor()
                elif edges == (Qt.Edge.LeftEdge | Qt.Edge.TopEdge) or edges == (Qt.Edge.RightEdge | Qt.Edge.BottomEdge):
                    self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                elif edges == (Qt.Edge.RightEdge | Qt.Edge.TopEdge) or edges == (Qt.Edge.LeftEdge | Qt.Edge.BottomEdge):
                    self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                elif edges & (Qt.Edge.LeftEdge | Qt.Edge.RightEdge):
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                elif edges & (Qt.Edge.TopEdge | Qt.Edge.BottomEdge):
                    self.setCursor(Qt.CursorShape.SizeVerCursor)
                else:
                    self.unsetCursor()
                return False

            if event.type() == QEvent.Type.MouseButtonRelease:
                self.unsetCursor()
                return False
            
        return super().eventFilter(obj, event)
    
    def enable_mouse_tracking(self, widget):
        widget.setMouseTracking(True)
        for child in widget.findChildren(QWidget):
            child.setMouseTracking(True)
    
    def hit_test_edges(self, pos):
        if self.isMaximized():
            return Qt.Edge(0)
        r = self.rect(); x, y = pos.x(), pos.y()
        edges = Qt.Edge(0)
        if x <= 6: edges |= Qt.Edge.LeftEdge
        if x >= r.width() - 6: edges |= Qt.Edge.RightEdge
        if y <= 6: edges |= Qt.Edge.TopEdge
        if y >= r.height() - 6: edges |= Qt.Edge.BottomEdge
        return edges
    
    def model_changed(self, index):
        idx = self.modelSelect.itemData(index)
        gpu_layers = self.LLMSettings.get('gpu_layers')
        if gpu_layers == "Auto":
            gpu_layers = int(self.models[int(idx)]['layers'])+1
        elif gpu_layers == "All":
            gpu_layers = int(self.models[int(idx)]['layers'])+1
        elif gpu_layers == "0":
            gpu_layers = 0
        options = {
            'model_path': self.models[int(idx)]['path'],
            'address': self.LLMSettings['address'],
            'port': self.LLMSettings['port'],
            'threads': int(self.LLMSettings['threads']),
            'gpu_layers': gpu_layers,
            'batch_size': int(self.LLMSettings['batch_size'])
        }
        if hasattr(self, 'llama_thread') and self.llama_thread._is_running:
            self.llama_thread.stop()
            self.llama_thread.wait()
        self.llama_thread = Llama_cpp(options)
        self.llama_thread.start()
        self.llama_thread.exec()
        self.llama_thread.run()

    def stop_llama_server(self):
        if hasattr(self, 'llama_thread') and self.llama_thread._is_running:
            self.llama_thread.stop()
            self.llama_thread.wait()
            self.modelSelect.setCurrentIndex(-1)

    def minimize(self):
        self.showMinimized()

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
            self.maximizeBtn.setText("❐")
            self.maximizeBtn.setToolTip("Restore")
        else:
            self.showMaximized()
            self.maximizeBtn.setText("□")
            self.maximizeBtn.setToolTip("Maximize")

    def closeApp(self):
        try:
            self.save_chat()
            self.save_chat_list()
            self.stop_llama_server()
        finally:
            self.close()

    def closeEvent(self, event):
        try:
            self.save_chat()
            self.save_chat_list()
        finally:
            super().closeEvent(event)

    def initChat(self):
        widget = QWidget()
        layout = QHBoxLayout()

        chatLLayout = QVBoxLayout()
        chatLLayout.setSpacing(6)

        chatLTitle = QLabel("List of chats")
        chatLTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chatLLayout.addWidget(chatLTitle)

        chatLButtons = QHBoxLayout()
        createChatBtn = QPushButton("+")
        createChatBtn.setToolTip("Create a new chat session")
        createChatBtn.clicked.connect(self.create_new_chat)
        chatLButtons.addWidget(createChatBtn)

        removeChatsBtn = QPushButton("-")
        removeChatsBtn.setToolTip("Remove selected chat sessions")
        removeChatsBtn.clicked.connect(self.remove_chat)
        chatLButtons.addWidget(removeChatsBtn)

        chatLLayout.addLayout(chatLButtons)

        self.chatList = QListWidget()
        self.chatList.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.chatList.currentItemChanged.connect(self.load_chat)

        chatLLayout.addWidget(self.chatList)
        layout.addLayout(chatLLayout, 20)

        chatWLayout = QVBoxLayout()
        chatWLayout.setSpacing(6)

        chatWButtons = QHBoxLayout()
        spBtn = QPushButton("System Prompt")
        spBtn.setToolTip("Edit System Prompt")
        spBtn.setFixedHeight(24)
        spBtn.clicked.connect(self.edit_system_prompt)
        chatWButtons.addWidget(spBtn)
        chatWLayout.addLayout(chatWButtons)

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
        chatWLayout.addWidget(self.chatDisplay)

        inputLayout = QHBoxLayout()
        self.chatInput = QLineEdit()
        self.chatInput.returnPressed.connect(self.send_prompt)
        self.chatInput.setPlaceholderText("Type your prompt here...")

        sendBtn = QPushButton("Send")
        sendBtn.clicked.connect(self.send_prompt)

        inputLayout.addWidget(self.chatInput)
        inputLayout.addWidget(sendBtn)

        chatWLayout.addLayout(inputLayout)
        layout.addLayout(chatWLayout, 80)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Chat")
        self.load_chat_list()

    def create_new_chat(self, title=None):
        ok = True
        if isinstance(title, bool):
            title = None
        if title is None:
            title, ok = QInputDialog.getText(self, "New Chat", "Enter chat title:")
        if ok and title.strip():
            self.save_chat()
            print(self.LLMSettings)
            self.chatHistory = [{"role": "system", "content": self.LLMSettings['system_prompt']}]
            chat = QListWidgetItem(title.strip())
            chat.setData(Qt.ItemDataRole.UserRole, f"chat_{self.chatList.count()}.json")
            self.chatList.addItem(chat)
            self.save_chat(chat)
            self.chatList.setCurrentItem(chat, QItemSelectionModel.SelectionFlag.ClearAndSelect)
            self.update_chat_display()
            self.save_chat_list()
    
    def load_chat_list(self):
        try:
            with open("chats/chat_list.json", "r") as f:
                chats = json.load(f)
                if chats.get("chats") is None:
                    self.create_new_chat("Default Chat")
                    return
                for chat in chats.get("chats", []):
                    item = QListWidgetItem(chat.get("title", "Untitled Chat"))
                    item.setData(Qt.ItemDataRole.UserRole, chat.get("filename", ""))
                    self.chatList.addItem(item)
        except (FileNotFoundError, json.JSONDecodeError):
            self.create_new_chat("Default Chat")

    def load_chat(self):
        chat = self.chatList.currentItem()
        if chat:
            filename = chat.data(Qt.ItemDataRole.UserRole)
            try:
                with open("chats/" + filename, "r") as f:
                    self.chatHistory = json.load(f).get("history", [{"role": "system", "content": "You are a helpful assistant."}])
            except (FileNotFoundError, json.JSONDecodeError):
                self.chatHistory = [{"role": "system", "content": "You are a helpful assistant."}]
            self.update_chat_display()
    
    def save_chat(self, chat = None):
        if chat is None:
            chat = self.chatList.currentItem()
        if chat:
            filename = chat.data(Qt.ItemDataRole.UserRole)
            try:
                with open("chats/" + filename, "w") as f:
                    json.dump({"title": chat.text(), "history": self.chatHistory}, f, indent=4)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save chat: {e}")
                return
    
    def save_chat_list(self):
        if not os.path.exists("chats"):
            os.makedirs("chats")
        chats = []
        for i in range(self.chatList.count()):
            item = self.chatList.item(i)
            chats.append({"title": item.text(), "filename": item.data(Qt.ItemDataRole.UserRole)})
        if not chats:
            self.chatHistory = []
            self.update_chat_display()
        try:
            with open("chats/chat_list.json", "w") as f:
                json.dump({"chats": chats}, f, indent=4)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save chat list: {e}")
            return
        
    def remove_chat(self):
        selected_items = self.chatList.selectedItems()
        if not selected_items:
            return
        
        for chat in selected_items:
            self.chatList.takeItem(self.chatList.row(chat))
            filename = chat.data(Qt.ItemDataRole.UserRole)
            try:
                os.remove("chats/" + filename)
                self.save_chat_list()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete chat file: {e}")
                return
    
    def edit_system_prompt(self):
        replace = False
        if self.chatList.count() == 0:
            replace = True
        text, ok = QInputDialog.getMultiLineText(self, "Edit System Prompt", "System Prompt:", self.find_last_sp())
        if ok and text.strip():
            if replace == True:
                self.create_new_chat("Default Chat")
                self.chatHistory = []
            self.chatHistory.append({"role": "system", "content": text.strip()})
            self.save_chat()
            self.update_chat_display()

    def find_last_sp(self):
        for msg in self.chatHistory[::-1]:
            if msg['role'] == 'system':
                return msg['content']
        return self.LLMSettings['system_prompt']

    def send_prompt(self):
        prompt = self.chatInput.text().strip()
        if not prompt:
            return
        if self.chatList.count() == 0:
            self.create_new_chat("Default Chat")
        self.chatHistory.append({"role": "user", "content": prompt})
        self.chatInput.clear()
        self.update_chat_display()
        self.chatDisplay.append(f'<b><span style="color: orange;">Assistant:</span></b> ')
        self.convert_chat_toLegacy()
        QApplication.processEvents()
        request = {"messages": self.chatLegacyHistory, "max_tokens": -1, "n_predict": -1, "stream": True}
        self.worker = LLMWorker(request)
        self.worker.token_emit.connect(lambda token: self.chatDisplay.insertPlainText(token) or self.chatDisplay.moveCursor(QTextCursor.MoveOperation.End))
        self.worker.result_ready.connect(self.handle_reply)
        self.worker.error_emit.connect(lambda e: self.chatDisplay.append(f'<b><span style="color: red;">Qully:</span></b> {e}'))
        self.worker.start()
        self.worker.exec()
            
    def convert_chat_toLegacy(self):
        self.chatLegacyHistory = [self.chatHistory[0]]
        for message in self.chatHistory:
            if message['role'] == 'system':
                self.chatLegacyHistory[0] = message
            else:
                self.chatLegacyHistory.append(message)

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
        self.chatDisplay.setMarkdown("\n\n".join(parts))
        self.chatDisplay.verticalScrollBar().setValue(self.chatDisplay.verticalScrollBar().maximum())
        self.save_chat()

    def initModels(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(6)

        buttonsSec = QHBoxLayout()
        addModelBtn = QPushButton("Add Model")
        addModelBtn.clicked.connect(self.add_model)
        removeModelBtn = QPushButton("Remove Model")
        removeModelBtn.clicked.connect(self.remove_model)
        settingsModelBtn = QPushButton("Model Settings")
        settingsModelBtn.clicked.connect(self.settings_model)
        buttonsSec.addWidget(addModelBtn)
        buttonsSec.addWidget(removeModelBtn)
        buttonsSec.addWidget(settingsModelBtn)
        layout.addLayout(buttonsSec)

        try:
            with open("models.json", "r") as f:
                models_json = json.load(f)
                self.models = models_json.get('models', [])
        except (FileNotFoundError, json.JSONDecodeError):
            self.models = []

        modelsLayout = QHBoxLayout()

        self.modelsTable = QTableWidget()
        self.modelsTable.setColumnCount(5)
        self.modelsTable.setHorizontalHeaderLabels(["Path", "Name", "Parameters","Weights", "Layers"])
        self.modelsTable.horizontalHeader().setStretchLastSection(True)
        self.modelsTable.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.modelsTable.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.modelsTable.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.modelsTable.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.modelsTable.itemSelectionChanged.connect(lambda: self.settings_model(change=True))

        for model in self.models:
            row = self.modelsTable.rowCount()
            self.modelsTable.insertRow(row)
            self.modelsTable.setItem(row, 0, QTableWidgetItem(model.get("path", "")))
            self.modelsTable.setItem(row, 1, QTableWidgetItem(model.get("name", "")))
            self.modelsTable.setItem(row, 2, QTableWidgetItem(str(model.get("parameters", ""))))
            self.modelsTable.setItem(row, 3, QTableWidgetItem(str(model.get("weights", ""))))
            self.modelsTable.setItem(row, 4, QTableWidgetItem(str(model.get("layers", ""))))

            self.modelSelect.addItem(model.get("name", "Unknown") + " (" + model.get("weights", "Unknown") + ")", str(row))

        modelsLayout.addWidget(self.modelsTable, 65)

        modelSettingsLayout = QVBoxLayout()

        self.LLMModelSettingsTable = QTableWidget()
        self.LLMModelSettingsTable.setColumnCount(2)
        self.LLMModelSettingsTable.setHorizontalHeaderLabels(["Setting", "Value"])
        self.LLMModelSettingsTable.horizontalHeader().setStretchLastSection(True)
        self.LLMModelSettingsTable.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.LLMModelSettingsTable.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.LLMModelSettingsTable.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.LLMModelSettingsTable.itemChanged.connect(self.llm_setting_changed)
        self.LLMModelSettingsTable.setVisible(False)
        modelsLayout.addWidget(self.LLMModelSettingsTable, 35)

        layout.addLayout(modelsLayout)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Models")

    def add_model(self):
        
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            ggufRead = GGUFInfoWoker(file_path)
            ggufRead.info_ready.connect(self.add_model_postworker)
            ggufRead.start()
            ggufRead.exec()
    
    def add_model_postworker(self, info):
        file_path = info.get("path", "Unknown")
        name = info.get("name", "Unknown")
        parameters = info.get("parameters", "Unknown")
        weights = info.get("weights", "Unknown")
        layers = info.get("layers", "Unknown")

        self.models.append({
            "path": file_path,
            "name": name,
            "parameters": parameters,
            "weights": weights,
            "layers": layers
        })

        self.refresh_models_table()

    def remove_model(self):
        selected_rows = self.modelsTable.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "Warning", "No model selected.")
            return
        row = selected_rows[0].row()
        del self.models[row]
        self.refresh_models_table()

    def settings_model(self, change=False):
        if self.LLMModelSettingsTable.isVisible() and not change:
            self.LLMModelSettingsTable.setVisible(False)
            return
        if not self.LLMModelSettingsTable.isVisible() and change:
            return
        
        self.LLMModelSettingsTable.setVisible(True)
        
        selected_rows = self.modelsTable.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "Warning", "No model selected.")
            return
        row = selected_rows[0].row()
        model = self.models[row]
        for setting in self.bpSettings:
            if setting['name'] == 'gpu_layers' and 1 in setting['use_case']:
                setting['max'] = int(model.get("layers", "0")) + 1
        self.loadLLMSettings(path=model.get("path", ""), type=1)

    def model_settings_switcher(self, checked):
        if checked:
            for row in range(1, self.LLMModelSettingsTable.rowCount()):
                widget = self.LLMModelSettingsTable.cellWidget(row, 1)
                if widget:
                    widget.setDisabled(True)
                else:
                    item = self.LLMModelSettingsTable.item(row, 1)
                    if item:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
        else:
            for row in range(1, self.LLMModelSettingsTable.rowCount()):
                widget = self.LLMModelSettingsTable.cellWidget(row, 1)
                if widget:
                    widget.setDisabled(False)
                else:
                    item = self.LLMModelSettingsTable.item(row, 1)
                    if item:
                        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled)

    def refresh_models_table(self):
        self.modelsTable.setRowCount(0)
        self.modelSelect.clear()
        if not self.models:
            with open("models.json", "w") as f:
                json.dump({"models": []}, f, indent=4)
            return
        for model in self.models:
            row = self.modelsTable.rowCount()
            self.modelsTable.insertRow(row)
            self.modelsTable.setItem(row, 0, QTableWidgetItem(model.get("path", "")))
            self.modelsTable.setItem(row, 1, QTableWidgetItem(model.get("name", "")))
            self.modelsTable.setItem(row, 2, QTableWidgetItem(str(model.get("parameters", ""))))
            self.modelsTable.setItem(row, 3, QTableWidgetItem(str(model.get("weights", ""))))
            self.modelsTable.setItem(row, 4, QTableWidgetItem(str(model.get("layers", ""))))

            self.modelSelect.addItem(model.get("name", "Unknown") + " (" + model.get("weights", "Unknown") + ")", model.get("path", "Unknown"))
        with open("models.json", "w") as f:
            json.dump({"models": self.models}, f, indent=4)

    def initLLMSettings(self):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setSpacing(6)

        llmSLLayout = QVBoxLayout()
        llmSLLayout.setSpacing(6)

        llmSLTitle = QLabel("Settings profiles")
        llmSLTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        llmSLLayout.addWidget(llmSLTitle)

        llmSLButtons = QHBoxLayout()
        createSProfileBtn = QPushButton("+")
        createSProfileBtn.setToolTip("Create a new settings profile")
        createSProfileBtn.clicked.connect(self.create_new_settings)
        llmSLButtons.addWidget(createSProfileBtn)

        removeSProfileBtn = QPushButton("-")
        removeSProfileBtn.setToolTip("Remove selected settings profiles")
        removeSProfileBtn.clicked.connect(self.removeLLMSettings)
        llmSLButtons.addWidget(removeSProfileBtn)

        llmSLLayout.addLayout(llmSLButtons)

        self.llmSettingsList = QListWidget()
        self.llmSettingsList.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.llmSettingsList.currentItemChanged.connect(lambda: self.loadLLMSettings(type=0))

        llmSLLayout.addWidget(self.llmSettingsList)
        layout.addLayout(llmSLLayout, 20)

        llmSWLayout = QVBoxLayout()
        llmSWLayout.setSpacing(6)

        self.LLMSettingsTable = QTableWidget()
        self.LLMSettingsTable.setColumnCount(2)
        self.LLMSettingsTable.setHorizontalHeaderLabels(["Setting", "Value"])
        self.LLMSettingsTable.horizontalHeader().setStretchLastSection(True)
        self.LLMSettingsTable.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.LLMSettingsTable.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.LLMSettingsTable.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.LLMSettingsTable.itemChanged.connect(self.llm_setting_changed)

        llmSWLayout.addWidget(self.LLMSettingsTable)
        layout.addLayout(llmSWLayout, 80)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Settings")
        self.load_settings_list()
        self.llmSettingsList.setCurrentRow(0)
    
    def create_new_settings(self, title=None):
        ok = True
        if isinstance(title, bool):
            title = None
        if title is None:
            title, ok = QInputDialog.getText(self, "New Settings Profile", "Enter settings profile name:")
        if ok and title.strip():
            if self.llmSettingsList.count() == 0:
                self.loadLLMSettings(type=-1)
                settings = QListWidgetItem(title.strip())
                settings.setData(Qt.ItemDataRole.UserRole, f"settings_llm_default.json")
                self.llmSettingsList.addItem(settings)
                self.saveLLMSettings(path=f"settings/{settings.data(Qt.ItemDataRole.UserRole)}", type=0)
                self.llmSettingsList.setCurrentItem(settings, QItemSelectionModel.SelectionFlag.ClearAndSelect)
            else:
                self.saveLLMSettings(type=0)
                self.loadLLMSettings(type=-1)
                settings = QListWidgetItem(title.strip())
                settings.setData(Qt.ItemDataRole.UserRole, f"settings_llm_{self.llmSettingsList.count() - 1}.json")
                self.llmSettingsList.addItem(settings)
                self.saveLLMSettings(path=f"settings/{settings.data(Qt.ItemDataRole.UserRole)}", type=0)
                self.llmSettingsList.setCurrentItem(settings, QItemSelectionModel.SelectionFlag.ClearAndSelect)
            self.save_settings_list()
    
    def llm_setting_changed(self, item, native=True, path=None, type=0):
        if not native:
            self.LLMSettings[item['name']] = item['value']
        else:
            data = item.data(Qt.ItemDataRole.UserRole)
            if data:
                self.LLMSettings[data['name']] = item.text()
                path = data['path']
                type = data['type']
        self.saveLLMSettings(path=path, type=type)


    def load_settings_list(self):
        try:
            with open("settings/settings_list.json", "r") as f:
                settings_json = json.load(f)
                if settings_json.get("settings") is None:
                    self.create_new_settings("Default Settings")
                    return
                for settings in settings_json.get("settings", []):
                    item = QListWidgetItem(settings.get("title", "Untitled Settings"))
                    item.setData(Qt.ItemDataRole.UserRole, settings.get("filename", ""))
                    self.llmSettingsList.addItem(item)
        except (FileNotFoundError, json.JSONDecodeError):
            self.create_new_settings("Default Settings")

    def save_settings_list(self):
        if not os.path.exists("settings"):
            os.makedirs("settings")
        settings = []
        for i in range(self.llmSettingsList.count()):
            item = self.llmSettingsList.item(i)
            settings.append({"title": item.text(), "filename": item.data(Qt.ItemDataRole.UserRole)})
        if not settings:
            self.create_new_settings("Default Settings")
            return
        try:
            with open("settings/settings_list.json", "w") as f:
                json.dump({"settings": settings}, f, indent=4)
        except Exception as e:
            print(f"Error saving settings list: {e}")

    def loadLLMSettings(self, path=None, type=0):
        if type == 0 and self.llmSettingsList.currentItem() is None:
            return
        try:
            if type == 0:
                filename = self.llmSettingsList.currentItem().data(Qt.ItemDataRole.UserRole)
                path = f"settings/{filename}"
            if type == -1:
                path = f"settings/settings_llm_default.json"
            if type == 1:
                path = path[:-5]
                path = f"{path}.json"
            with open(f"{path}", "r") as f:
                settings_json = json.load(f)
                self.LLMSettings = settings_json.get('settings', {})
                for setting in self.bpSettings:
                    if setting['name'] not in self.LLMSettings and type in setting['use_case']:
                        self.LLMSettings[setting['name']] = setting['default']
                        self.saveLLMSettings(path=path, type=type)
        except Exception as e:
            self.LLMSettings = {}
        
        if not self.LLMSettings:
            for setting in self.bpSettings:
                if type in setting['use_case']:
                    self.LLMSettings[setting['name']] = setting['default']
            if type == -1:
                return
            self.saveLLMSettings(path=path, type=type)

        target = None
        if type == 0:
            target = self.LLMSettingsTable
        elif type == 1:
            target = self.LLMModelSettingsTable
        target.setRowCount(0)
        for setting in self.bpSettings:
            if type in setting['use_case']:
                row = target.rowCount()
                target.insertRow(row)
                label = QTableWidgetItem(setting['display'])
                label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
                target.setItem(row, 0, label)
                value = ""
                if setting['type'] == 'text':
                    value = QTableWidgetItem(self.LLMSettings[setting['name']])
                    value.setData(Qt.ItemDataRole.UserRole, {"row": row, "name": setting['name'], "path": path, "type": type})
                elif setting['type'] == 'number':
                    value = QLineEdit()
                    value.setValidator(QIntValidator(1024, 65535, value))
                    value.setText(str(self.LLMSettings[setting['name']]))
                    value.textChanged.connect(lambda text, name=setting['name']: self.llm_setting_changed({'value': text, "name": name}, native=False, path=path, type=type))
                elif setting['type'] == 'slider':
                    value = QFrame()
                    value_layout = QVBoxLayout()
                    value_layout.setContentsMargins(2, 2, 2, 2)
                    value_layout.setSpacing(2)
                    slider = QSlider(Qt.Orientation.Horizontal)
                    slider.setRange(setting['min'], setting['max'])
                    setvalue = int(self.LLMSettings[setting['name']])
                    if setvalue == -1:
                        setvalue = setting['max']
                    slider.setValue(setvalue)
                    slider.setSingleStep(1)
                    slider.setTickPosition(QSlider.TickPosition.TicksBelow)
                    slider.setTickInterval(10)
                    slider.setPageStep(2)
                    value_layout.addWidget(slider)
                    curr_label = QLabel(self.LLMSettings[setting['name']])
                    slider.valueChanged.connect(lambda curr_value, slider_el = slider, label=curr_label: self.update_slider(slider_el,label, curr_value, path=path, type=type))
                    value_layout.addWidget(curr_label)
                    value.setLayout(value_layout)
                    self.update_slider(slider, curr_label, slider.value())
                    value.adjustSize()
                    value.setWhatsThis(setting['name'])
                elif setting['type'] == 'combo':
                    value = QComboBox()
                    for option in setting['options']:
                        value.addItem(option)
                        if option == self.LLMSettings[setting['name']]:
                            value.setCurrentText(option) 
                    value.currentTextChanged.connect(lambda text, name=setting['name']: self.llm_setting_changed({'value': text, "name": name},native=False, path=path, type=type))
                elif setting['type'] == 'checkbox':
                    value = QCheckBox()
                    value.setChecked(bool(self.LLMSettings[setting['name']]))
                    value.stateChanged.connect(lambda state, name=setting['name']: self.llm_setting_changed({'value': state, "name": name}, native=False, path=path, type=type))
                elif setting['type'] == 'radiobutton':
                    value = QRadioButton()
                    value.setChecked(bool(self.LLMSettings[setting['name']]))
                    value.toggled.connect(lambda checked, name=setting['name']: self.llm_setting_changed({'value': checked, "name": name}, native=False, path=path, type=type))
                    if setting['name'] == 'model_settings':
                        value.setToolTip("When turned on, it uses model settings for model instead of profile settings.")
                        value.toggled.connect(lambda checked: self.model_settings_switcher(checked))

                target.setCellWidget(row, 1, value) if not isinstance(value, QTableWidgetItem) else target.setItem(row, 1, value)
                target.resizeRowToContents(row)

    def saveLLMSettings(self, path=None, type=0):
        if self.llmSettingsList.count() == 0:
            return
        if not os.path.exists("settings"):
            os.makedirs("settings")
        if type == 0 and path is None:
            filename = self.llmSettingsList.currentItem().data(Qt.ItemDataRole.UserRole)
            path = f"settings/{filename}"
        if type == 1:
            path = path[:-5]
            path = f"{path}.json"
        try:
            with open(f"{path}", "w") as f:
                json.dump({"settings": self.LLMSettings, "type": type}, f, indent=4)
        except Exception as e:
            print(f"Error saving LLM settings: {e}")

    def removeLLMSettings(self):
        selected_items = self.llmSettingsList.selectedItems()
        if not selected_items:
            return
        
        for settings in selected_items:
            self.llmSettingsList.takeItem(self.llmSettingsList.row(settings))
            filename = settings.data(Qt.ItemDataRole.UserRole)
            try:
                os.remove("settings/" + filename)
                self.save_settings_list()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete settings file: {e}")
                return

    def update_slider(self, slider, label, value, path=None, type=0):
        label.setText(str(value))
        slider_width = slider.width() - slider.style().pixelMetric(slider.style().PixelMetric.PM_SliderLength)
        if slider_width <= 0:
            return

        ratio = (value - slider.minimum()) / (slider.maximum() - slider.minimum())
        handle_x = int(ratio * slider_width)

        label_width = label.fontMetrics().boundingRect(label.text()).width()
        x_offset = max(0, handle_x - label_width // 2)

        label.setContentsMargins(x_offset, 0, 0, 0)

        self.LLMSettings[slider.whatsThis()] = str(value)
        self.saveLLMSettings(path=path, type=type)

if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = App()
	window.resize(1000, 600)
	window.show()
	sys.exit(app.exec())