import os
import signal
import math
from PyQt6.QtWidgets import (
	QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QLineEdit,
	QTabWidget, QMessageBox, QTableWidget, QTableWidgetItem, QSizePolicy, QFileDialog, QSplitter,
    QDialog, QListWidget, QListWidgetItem, QInputDialog, QComboBox, QCheckBox, QSlider, QFrame,
    QRadioButton, QScrollArea, QTextBrowser, QToolTip, QStackedLayout, QMenu, QHeaderView,
    QGridLayout
)
from PyQt6.QtGui import (
    QTextCursor, QPixmap, QCursor, QIntValidator, QPainter, QColor, QBrush, QPen, QMouseEvent,
    QTextOption, QFontDatabase, QFont
)
from PyQt6.QtCore import (
    QTimer, Qt, QThread, pyqtSignal, QItemSelectionModel, QEvent, QPoint, QPropertyAnimation,
    QEasingCurve, pyqtProperty
)
from PyQt6.QtTest import QTest
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
import copy
import time
from pathlib import Path
import psutil

from llama_server import Llama_cpp, is_llama_server_running, kill_llama_server
from llm_worker import LLMWorker
from gguf_worker import GGUFInfoWoker
from toggle_switch import ToggleSwitch
from chat_bubble import ChatBubble

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setMinimumSize(600, 338)
        self.setWindowTitle("Qully Chat")

        self.chat_ids = []
        self.chatHistory = []
        self.chatLegacyHistory = []
        self.models = []

        self.LLMSettings = {"system_prompt": "You are a helpful assistant."}
        self.bpSettings = [
            {'type': 'radiobutton', 'name': 'model_settings', 'display': 'Use model settings', 'default': False, 'use_case': [1]},
            {'type': 'radiobutton', 'name': 'chat_settings', 'display': 'Use chat settings', 'default': False, 'use_case': [2]},
            {'type': 'text', 'name': 'address', 'display': 'Address', 'default': '127.0.0.1', 'use_case': [0]},
            {'type': 'number', 'name': 'port', 'display': 'Port', 'default': '5175', 'use_case': [0]},
            {'type': 'slider', 'name': 'threads', 'display': 'CPU Threads', 'default': "-1", 'min': 1, 'max': os.cpu_count(), 'use_case': [0, 1, 2]},
            {'type': 'combo', 'name': 'gpu_layers', 'display': 'Layers on GPU', 'default': "All", 'options': ["Auto", "All", "0"], 'use_case': [0, 2]},
            {'type': 'slider', 'name': 'gpu_layers', 'display': 'Layers on GPU', 'default': "-1", 'min': 0, 'max': 0, 'use_case': [1]},
            {'type': 'number', 'name': 'batch_size', 'display': 'Batch size', 'default': "512", 'use_case': [0, 1, 2]},
            {'type': 'text', 'name': 'system_prompt', 'display': 'System prompt', 'default': 'You are a helpful assistant.', 'use_case': [0, 1, 2]},
            {'type': 'checkbox_group', 'name': 'statistics_display', 'display': 'Display statistics', 'options': ['Input ms', 'Generation ms', 'Total ms', 'Input tokens', 'Generated tokens', 'Total tokens', 'Tokens per second'], 'max_per_line': 4, 'default': ['Input ms', 'Generation ms', 'Total ms', 'Input tokens', 'Generated tokens', 'Total tokens', 'Tokens per second'], 'use_case': [0, 1, 2]}
        ]   # 0: llm settings tab; 1: llm model-specific settings; 2: chat-specific settings
        self.currentAddress = "http://127.0.0.1:5175/v1/chat/completions"
        self.last_stats = {}

        self._suppress_bubble_pop = False
        self._suppress_input = False
        self._suppress_scroll_down = True

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

        self.profileSelect = QComboBox(self.topBar)
        self.profileSelect.setToolTip("Select Profile")
        self.profileSelect.setFixedHeight(24)
        self.profileSelect.setPlaceholderText("Select Profile")
        self.profileSelect.installEventFilter(self)
        centerPart.addWidget(self.profileSelect)

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
        if event.type() in (QEvent.Type.Show, QEvent.Type.Resize, QEvent.Type.ShowToParent) and obj.metaObject().className() in ("QComboBoxPrivateContainer", "QComboBoxPopupContainer", "QWidgetWindow"):
            c = obj.parent()
            if isinstance(c, QComboBox) and c is getattr(self, "modelSelect", None):
                obj.move(c.mapToGlobal(QPoint(0, c.height()))); obj.setMinimumWidth(c.width())
            elif isinstance(c, QComboBox) and c is getattr(self, "profileSelect", None):
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
        settings_build = {}
        settings_set = 0
        ### if chat has settings
        chat = self.chatList.currentItem()
        if chat:
            filename = chat.data(Qt.ItemDataRole.UserRole)
            filename = filename[:-5]
            path = Path("chats") / f"{filename}_settings.json"
            if path.exists():
                self.loadLLMSettings(path=path, type_f=2, display=0)
                if self.LLMSettings.get('chat_settings', False) == True:
                    if settings_set == 0:
                        settings_set = 1
                        settings_build = self.LLMSettings.copy()
                    else:
                        for key, value in self.LLMSettings.items():
                            if key not in settings_build:
                                settings_build[key] = value
        ### if model has settings
        idx = self.modelSelect.itemData(index)['row']
        self.loadLLMSettings(path=self.models[int(idx)].get("path", ""), type_f=1, display=0)
        if self.LLMSettings.get('model_settings', False) == True:
            if settings_set == 0:
                settings_set = 1
                settings_build = self.LLMSettings.copy()
            else:
                for key, value in self.LLMSettings.items():
                    if key not in settings_build:
                        settings_build[key] = value
        ### else use profile settings
        path = Path("settings") / f"{self.profileSelect.currentData()}"
        self.loadLLMSettings(path=path, type_f=0, display=0)
        if settings_set == 0:
            settings_build = self.LLMSettings.copy()
        else:
            for key, value in self.LLMSettings.items():
                if key not in settings_build:
                    settings_build[key] = value
        gpu_layers = settings_build.get('gpu_layers')
        if gpu_layers == "Auto":
            gpu_layers = int(self.models[int(idx)]['layers'])+1
        elif gpu_layers == "All":
            gpu_layers = int(self.models[int(idx)]['layers'])+1
        elif gpu_layers == "0":
            gpu_layers = 0
        options = {
            'model_path': self.modelSelect.currentData()['path'],
            'address': settings_build['address'],
            'port': settings_build['port'],
            'threads': int(settings_build['threads']),
            'gpu_layers': int(gpu_layers),
            'batch_size': int(settings_build['batch_size'])
        }
        self.currentAddress = f"http://{settings_build['address']}:{settings_build['port']}/v1/chat/completions"
        if hasattr(self, 'llama_thread') and self.llama_thread._is_running:
            self.llama_thread.stop()
            self.llama_thread.wait()
        else:
            while is_llama_server_running():
                kill_llama_server()
                time.sleep(0.1)
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

        exportChatBtn = QPushButton("📤")
        exportChatBtn.setToolTip("Export selected chat session")
        exportMenu = QMenu()
        exportMenu.addAction("Export as JSON", lambda: self.export_chat_json())
        exportMenu.addAction("Export as HTML", lambda: self.export_chat_html())
        exportMenu.addAction("Export as Markdown", lambda: self.export_chat_md())
        exportChatBtn.clicked.connect(lambda _: exportMenu.exec(exportChatBtn.mapToGlobal(QPoint(0, exportChatBtn.height()))))
        chatLButtons.addWidget(exportChatBtn)

        chatLLayout.addLayout(chatLButtons)

        self.chatList = QListWidget()
        self.chatList.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.chatList.itemChanged.connect(self.save_chat_list)
        self.chatList.currentItemChanged.connect(self.load_chat)
        self.chatList.setDragEnabled(True)
        self.chatList.setAcceptDrops(True)
        self.chatList.setDropIndicatorShown(True)
        self.chatList.setDragDropMode(self.chatList.DragDropMode.InternalMove)
        self.chatList.model().rowsMoved.connect(self.save_chat_list)

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

        settingsChatBtn = QPushButton("Chat Settings")
        settingsChatBtn.clicked.connect(self.settings_chat)
        settingsChatBtn.setFixedHeight(24)
        chatWButtons.addWidget(settingsChatBtn)
        chatWLayout.addLayout(chatWButtons)

        chatWLayout2 = QHBoxLayout()
        chatWLayout2S = QVBoxLayout()

        self.chatDisplayScroll = QScrollArea()
        self.chatDisplayScroll.setWidgetResizable(True)
        self.chatDisplayWidget = QWidget()
        self.chatDisplay = QVBoxLayout()
        self.chatDisplay.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chatDisplay.setContentsMargins(0, 0, 0, 0)

        def _bubbles_change_event(w, ev):
            if not self._suppress_scroll_down:
                self.chatDisplayScroll.verticalScrollBar().setValue(self.chatDisplayScroll.verticalScrollBar().maximum())
            if ev.type() == QEvent.Type.ChildAdded:
                if isinstance(ev.child(), ChatBubble):
                    QTimer.singleShot(0, lambda: self.bubbles_change(atype = "add"))
            if ev.type() == QEvent.Type.ChildRemoved:
                QTimer.singleShot(0, lambda: self.bubbles_change(atype = "rem"))
            return QWidget.event(w, ev)
        self.chatDisplayWidget.event = _bubbles_change_event.__get__(self.chatDisplayWidget, QWidget)

        self.chatDisplayWidget.setLayout(self.chatDisplay)
        self.chatDisplayScroll.setWidget(self.chatDisplayWidget)
        chatWLayout2S.addWidget(self.chatDisplayScroll)

        inputLayout = QHBoxLayout()
        self.chatInput = QLineEdit()
        self.chatInput.returnPressed.connect(self.send_prompt)
        self.chatInput.setPlaceholderText("Type your prompt here...")

        sendBtn = QPushButton("Send")
        sendBtn.clicked.connect(lambda _C: self.send_prompt(prompt_t="input"))

        inputLayout.addWidget(self.chatInput)
        inputLayout.addWidget(sendBtn)

        chatWLayout2S.addLayout(inputLayout)
        chatWLayout2.addLayout(chatWLayout2S, 65)

        self.chatSettingsTable = QTableWidget()
        self.chatSettingsTable.setColumnCount(2)
        self.chatSettingsTable.setHorizontalHeaderLabels(["Setting", "Value"])
        self.chatSettingsTable.horizontalHeader().setStretchLastSection(True)
        self.chatSettingsTable.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.chatSettingsTable.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.chatSettingsTable.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.chatSettingsTable.itemChanged.connect(self.llm_setting_changed)
        self.chatSettingsTable.setVisible(False)
        chatWLayout2.addWidget(self.chatSettingsTable, 35)

        chatWLayout.addLayout(chatWLayout2)
        layout.addLayout(chatWLayout, 80)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Chat")
        self.load_chat_list()

    def create_new_chat(self, title=None, history=None):
        ok = True
        if isinstance(title, bool):
            title = None
        if title is None:
            title, ok = QInputDialog.getText(self, "New Chat", "Enter chat title:")
        if ok and title.strip():
            self.save_chat()
            QApplication.processEvents()
            if history is None:
                self.chatHistory = [{"role": "system", "content": self.LLMSettings['system_prompt']}]
            else:
                self.chatHistory = history
            chat = QListWidgetItem(title.strip())
            number = self.chatList.count()
            while number in self.chat_ids:
                number += 1
            self.chat_ids.append(number)
            chat.setData(Qt.ItemDataRole.UserRole, f"chat_{number}.json")
            chat.setFlags(chat.flags() | Qt.ItemFlag.ItemIsEditable)
            self.chatList.addItem(chat)
            self.save_chat(chat)
            self.chatList.setCurrentItem(chat, QItemSelectionModel.SelectionFlag.ClearAndSelect)
            self.update_chat_display()
            self.save_chat_list()
    
    def load_chat_list(self):
        try:
            path = Path("chats")
            if not path.exists():
                path.mkdir(parents=False, exist_ok=True)
            path = path / "chat_list.json"
            with open(path, "r") as f:
                chats = json.load(f)
                if chats.get("chats") is None:
                    self.create_new_chat("Default Chat")
                    return
                self.chat_ids.clear()
                for chat in chats.get("chats", []):
                    item = QListWidgetItem(chat.get("title", "Untitled Chat"))
                    item.setData(Qt.ItemDataRole.UserRole, chat.get("filename", ""))
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                    self.chatList.addItem(item)
                    self.chat_ids.append(int(chat.get("filename", "").removesuffix(".json").removeprefix("chat_")))
        except (FileNotFoundError, json.JSONDecodeError):
            self.create_new_chat("Default Chat")

    def load_chat(self):
        chat = self.chatList.currentItem()
        if chat:
            path = Path("chats")
            if not path.exists():
                path.mkdir(parents=False, exist_ok=True)
            filename = chat.data(Qt.ItemDataRole.UserRole)
            try:
                path = path / f"{filename}"
                with open(path, "r") as f:
                    self.chatHistory = json.load(f).get("history", [{"role": "system", "content": "You are a helpful assistant."}])
            except (FileNotFoundError, json.JSONDecodeError):
                self.chatHistory = [{"role": "system", "content": "You are a helpful assistant."}]
            self.update_chat_display()
            QTimer.singleShot(0, lambda: self.chatDisplayScroll.verticalScrollBar().setValue(self.chatDisplayScroll.verticalScrollBar().maximum()))
    
    def save_chat(self, chat = None):
        if chat is None:
            chat = self.chatList.currentItem()
        if chat:
            path = Path("chats")
            if not path.exists():
                path.mkdir(parents=False, exist_ok=True)
            filename = chat.data(Qt.ItemDataRole.UserRole)
            try:
                path = path / f"{filename}"
                with open(path, "w") as f:
                    json.dump({"title": chat.text(), "history": self.chatHistory}, f, indent=4)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save chat: {e}")
                return
    
    def save_chat_list(self):
        path = Path("chats")
        if not path.exists():
            path.mkdir(parents=False, exist_ok=True)
        chats = []
        for i in range(self.chatList.count()):
            item = self.chatList.item(i)
            chats.append({"title": item.text(), "filename": item.data(Qt.ItemDataRole.UserRole)})
        if not chats:
            self.chatHistory = []
            self.update_chat_display()
        try:
            path = path / "chat_list.json"
            with open(path, "w") as f:
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
                path = Path("chats") / f"{filename}"
                path.unlink(missing_ok=True)
                self.save_chat_list()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete chat file: {e}")
                return
            
    def settings_chat(self, change = False):
        if self.chatSettingsTable.isVisible() and not change:
            self.chatSettingsTable.setVisible(False)
            return
        if not self.chatSettingsTable.isVisible() and change:
            return
        
        self.chatSettingsTable.setVisible(True)

        selected_chats = self.chatList.selectedItems()
        if not selected_chats:
            QMessageBox.information(self, "No Chat Selected", "Please select a chat to edit its settings.")
            return
        chat = selected_chats[0]
        filename = chat.data(Qt.ItemDataRole.UserRole)
        path = Path("chats") / f"{filename}"
        self.loadLLMSettings(path=path, type_f=2)

    def chat_settings_switcher(self, checked):
        if not checked:
            for row in range(1, self.chatSettingsTable.rowCount()):
                widget = self.chatSettingsTable.cellWidget(row, 1)
                if widget:
                    widget.setDisabled(True)
                else:
                    item = self.chatSettingsTable.item(row, 1)
                    if item:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
        else:
            for row in range(1, self.chatSettingsTable.rowCount()):
                widget = self.chatSettingsTable.cellWidget(row, 1)
                if widget:
                    widget.setDisabled(False)
                else:
                    item = self.chatSettingsTable.item(row, 1)
                    if item:
                        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled)
    
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

    def send_prompt(self, prompt_t="input"):
        if self._suppress_input:
            return
        if prompt_t == "input":
            prompt = self.chatInput.text().strip()
            if not prompt:
                return
            if self.chatList.count() == 0:
                self.create_new_chat("Default Chat")
            self.chatHistory.append({"role": "user", "content": prompt})
            bubble_u = ChatBubble(prompt, "user")
            self.chatDisplay.addWidget(bubble_u)
            self.chatInput.clear()
        if self.modelSelect.currentIndex() >= 0:
            self._suppress_input = True
            self._suppress_scroll_down = False
            self.convert_chat_toLegacy()
            QApplication.processEvents()
            bubble_a = ChatBubble("", "assistant")
            self.chatDisplay.addWidget(bubble_a)
            request = {"messages": self.chatLegacyHistory, "max_tokens": -1, "n_predict": -1, "stream": True}
            self.worker = LLMWorker(request, self.currentAddress)
            self.worker.token_emit.connect(lambda token: bubble_a.textbox.insertPlainText(token))
            self.worker.result_ready.connect(self.handle_reply)
            self.worker.error_emit.connect(lambda e: self.error_returned(e))
            self.worker.stats_emit.connect(lambda s: self.connect_stats(s))
            self.worker.start()
            self.worker.exec()
        else:
            QTest.mouseClick(self.modelSelect, Qt.MouseButton.LeftButton)
            
    def convert_chat_toLegacy(self, chat=None):
        if chat is None:
            self.chatLegacyHistory = [self.chatHistory[0]]
            for message in self.chatHistory:
                if message['role'] == 'system':
                    self.chatLegacyHistory[0] = message
                else:
                    self.chatLegacyHistory.append(message)
        else:
            chatLegacy = [chat[0]]
            for message in chat:
                if message['role'] == 'system':
                    chatLegacy[0] = message
                else:
                    chatLegacy.append(message)
            return chatLegacy

    def handle_reply(self, reply):
        print(f'Reply: {reply}')
        reply_think = ""
        if reply.startswith("<think>") and "</think>" in reply:
            reply_think = reply.split("</think>")[0][7:]
            reply = reply.split("</think>")[1]
        QApplication.processEvents()
        self.chatHistory.append({"role": "assistant", "think": reply_think, "content": reply, "llm": self.modelSelect.currentText(), "stats": self.last_stats})
        self.update_chat_display()
        self._suppress_input = False
    
    def error_returned(self, error):
        QMessageBox.warning(self, "Error", f"A server error occurred: {error}")
        self._suppress_input = False
        self._suppress_scroll_down = True

    def update_chat_display(self):
        self.chatDisplayWidget.setVisible(False)
        self._suppress_bubble_pop = True
        self._suppress_scroll_down = False
        while self.chatDisplay.count():
            item = self.chatDisplay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
            
        QApplication.processEvents()
        self._suppress_bubble_pop = False
        for message in self.chatHistory:
            role = message['role']
            content = message['content']
            while content.startswith("\n"):
                content = content[1:]
            if role == 'user':
                bubble = ChatBubble(content, "user")
                bubble.editBranchBtn.clicked.connect(lambda _checked, b=bubble: self.branch_edit_bubble(bubble=b))
                bubble.saveBtn.clicked.connect(lambda _checked, b=bubble: self.save_edit_bubble(bubble=b))
            elif role == 'assistant':
                content = md_to_html(content, extensions=["extra", "fenced_code", "sane_lists", "nl2br"])
                bubble = ChatBubble(content, "assistant", llm=message.get('llm', None), think=message.get('think', None))
                stats = message.get('stats', {})
                stats_html = f'''
                    {'<b>Time</b>' if any(stat in self.LLMSettings.get('statistics_display', []) for stat in ['Input ms', 'Generation ms', 'Total ms']) else ''}
                    {f'<div style="display: block; margin: 0 0 0 1em; padding: 0;"><b>Input (ms):</b> {stats.get('input_ms', "Unavailable")}</div>' if 'Input ms' in self.LLMSettings.get('statistics_display', []) else ''}
                    {f'<div style="display: block; margin: 0 0 0 1em; padding: 0;"><b>Generation (ms):</b> {stats.get('gen_ms', "Unavailable")}</div>' if 'Generation ms' in self.LLMSettings.get('statistics_display', []) else ''}
                    {f'<div style="display: block; margin: 0 0 0 1em; padding: 0;"><b>Total (ms):</b> {stats.get('total_ms', "Unavailable")}</div>' if 'Total ms' in self.LLMSettings.get('statistics_display', []) else ''}
                    {'<b>Tokens</b>' if any(stat in self.LLMSettings.get('statistics_display', []) for stat in ['Input tokens', 'Generated tokens', 'Total tokens']) else ''}
                    {f'<div style="display: block; margin: 0 0 0 1em; padding: 0;"><b>Input:</b> {stats.get('input_t', "Unavailable")}</div>' if 'Input tokens' in self.LLMSettings.get('statistics_display', []) else ''}
                    {f'<div style="display: block; margin: 0 0 0 1em; padding: 0;"><b>Generated:</b> {stats.get('gen_t', "Unavailable")}</div>' if 'Generated tokens' in self.LLMSettings.get('statistics_display', []) else ''}
                    {f'<div style="display: block; margin: 0 0 0 1em; padding: 0;"><b>Total:</b> {stats.get('total_t', "Unavailable")}</div>' if 'Total tokens' in self.LLMSettings.get('statistics_display', []) else ''}
                    {f'<b>Tokens per second:</b> {stats.get('t_s', "Unavailable")}' if 'Tokens per second' in self.LLMSettings.get('statistics_display', []) else ''}
                '''
                bubble.statsBtn.info = stats_html
            elif role == 'system':
                bubble = ChatBubble(content, "system")
            else:
                bubble = ChatBubble(content, role)
            bubble.deleteDownBtn.clicked.connect(lambda _checked, b=bubble: self.delete_down_bubble(bubble=b))
            bubble.branchBtn.clicked.connect(lambda _checked, b=bubble: self.branch_bubble(bubble=b))
            self.chatDisplay.addWidget(bubble)

        #if hasattr(self, "chatDisplayWidget"):
        #    self.chatDisplayWidget.adjustSize()
        #    self.chatDisplayWidget.update()
        #if hasattr(self, "chatDisplayScroll"):
        #    self.chatDisplayScroll.viewport().update()
        self.save_chat()
        self.chatDisplayWidget.setVisible(True)
        QApplication.processEvents()
        self._suppress_scroll_down = True

    def bubbles_change(self, atype):
        vlast = None
        count = self.chatDisplay.count()
        if count == 0:
            return
        last = self.chatDisplay.itemAt(count-1).widget()
        if count > 1:
            vlast = self.chatDisplay.itemAt(count-2).widget()
        try:
            if atype == "add":
                if vlast is not None:
                    vlast.deleteBtn.setVisible(False)
                    vlast.generateBtn.setVisible(False)
                    vlast.deleteDownBtn.setVisible(True)
                last.deleteBtn.setVisible(True)
                last.deleteDownBtn.setVisible(False)
                if last.speaker == 'User':
                    last.generateBtn.setVisible(True)
                    last.generateBtn.clicked.connect(lambda _c: self.send_prompt(prompt_t="manual"))
            elif atype == "rem":
                last.deleteBtn.setVisible(True)
                last.deleteDownBtn.setVisible(False)
                if last.speaker == 'User':
                    last.generateBtn.setVisible(True)
                    last.generateBtn.clicked.connect(lambda _c: self.send_prompt(prompt_t="manual"))
                if not self._suppress_bubble_pop:
                    self.chatHistory.pop()
        except:
            pass

    def connect_stats(self, stats_d):
        self.last_stats = {'input_ms': round(stats_d['timings']['prompt_ms'], 2), 'gen_ms': round(stats_d['timings']['predicted_ms'], 2),
                           'total_ms': round(stats_d['timings']['prompt_ms'] + stats_d['timings']['predicted_ms'], 2),
                           'input_t': stats_d['usage']['prompt_tokens'], 'gen_t': stats_d['usage']['completion_tokens'],
                           'total_t': stats_d['usage']['total_tokens'], 't_s': round(stats_d['timings']['predicted_per_second'], 2)}
        
    def delete_down_bubble(self, bubble):
        index = self.chatDisplay.indexOf(bubble)
        if index >= 0:
            for i in range(self.chatDisplay.count()-1, index, -1):
                item = self.chatDisplay.takeAt(i)
                w = item.widget()
                if w is not None:
                    w.setParent(None)
                    w.deleteLater()

    def branch_bubble(self, bubble):
        index = self.chatDisplay.indexOf(bubble)
        history = []
        if index >= 0:
            history = copy.deepcopy(self.chatHistory[:index+1])
            self.create_new_chat(title=self.chatList.currentItem().text()+" - Branch", history=history)

    def branch_edit_bubble(self, bubble):
        index = self.chatDisplay.indexOf(bubble)
        history = []
        if index >= 0:
            history = copy.deepcopy(self.chatHistory[:index+1])
            history[-1]['content'] = bubble.editbox.toPlainText().strip()
            self.create_new_chat(title=self.chatList.currentItem().text()+" - Edit Branch", history=history)

    def save_edit_bubble(self, bubble):
        self.delete_down_bubble(bubble)
        QApplication.processEvents()
        self.chatHistory[-1]['content'] = bubble.editbox.toPlainText().strip()
        bubble.text = bubble.editbox.toPlainText().strip()
        bubble.layout().setCurrentIndex(0)
        self.update_chat_display()

    def export_chat_json(self):
        selected_chats = self.chatList.selectedItems()
        if not selected_chats:
            QMessageBox.information(self, "No Chat Selected", "Please select a chat to export.")
            return
        chat = selected_chats[0]
        chat_data = {}
        filename = chat.data(Qt.ItemDataRole.UserRole)
        try:
            path = Path("chats") / f"{filename}"
            with open(path, "r") as f:
                chat_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            QMessageBox.critical(self, "Error", "Failed to load the selected chat.")
            return
        chat_data_legacy = self.convert_chat_toLegacy(chat_data['history'])
        options = QFileDialog.Option(0)
        options |= QFileDialog.Option.DontUseNativeDialog
        save_path, _ = QFileDialog.getSaveFileName(self, "Export Chat as JSON", f"{chat.text()}.json", "JSON Files (*.json);;All Files (*)", options=options)
        if not save_path:
            return
        try:
            with open(save_path, "w") as f:
                json.dump({"title": chat.text(), "history": chat_data_legacy}, f, indent=4)
            QMessageBox.information(self, "Success", f"Chat {chat.text()} exported successfully to {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export chat {chat.text()}: {e}")

    def export_chat_html(self):
        selected_chats = self.chatList.selectedItems()
        if not selected_chats:
            QMessageBox.information(self, "No Chat Selected", "Please select a chat to export.")
            return
        chat = selected_chats[0]
        chat_data = {}
        filename = chat.data(Qt.ItemDataRole.UserRole)
        try:
            path = Path("chats") / f"{filename}"
            with open(path, "r") as f:
                chat_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            QMessageBox.critical(self, "Error", "Failed to load the selected chat.")
            return
        chat_html = f"<html><head><meta charset='UTF-8'><title>{chat.text()}</title></head><body>"
        for message in chat_data['history']:
            role = message['role']
            content = message['content']
            if role == 'user':
                chat_html += f"<div style='background-color:#d1e7dd;padding:10px;margin:10px;border-radius:10px;'><b>User:</b><br>{content}</div>"
            elif role == 'assistant':
                name = message.get('llm', 'Assistant')
                stats = message.get('stats', {})
                if not stats:
                    chat_html += f"<div style='background-color:#f8d7da;padding:10px;margin:10px;border-radius:10px;'><b>{name}:</b><br>{md_to_html(content, extensions=['extra', 'fenced_code', 'sane_lists', 'nl2br'])}</div>"
                else:
                    stats_html = f'''
                        <b>Time</b>
                        <div style="display: block; margin: 0 0 0 1em; padding: 0;"><b>Input (ms):</b> {stats.get('input_ms', "Unavailable")}</div>
                        <div style="display: block; margin: 0 0 0 1em; padding: 0;"><b>Generation (ms):</b> {stats.get('gen_ms', "Unavailable")}</div>
                        <div style="display: block; margin: 0 0 0 1em; padding: 0;"><b>Total (ms):</b> {stats.get('total_ms', "Unavailable")}</div>
                        <b>Tokens</b>
                        <div style="display: block; margin: 0 0 0 1em; padding: 0;"><b>Input:</b> {stats.get('input_t', "Unavailable")}</div>
                        <div style="display: block; margin: 0 0 0 1em; padding: 0;"><b>Generated:</b> {stats.get('gen_t', "Unavailable")}</div>
                        <div style="display: block; margin: 0 0 0 1em; padding: 0;"><b>Total:</b> {stats.get('total_t', "Unavailable")}</div>
                        <b>Tokens per second:</b> {stats.get('t_s', "Unavailable")}
                    '''
                    chat_html += f"<div style='background-color:#f8d7da;padding:10px;margin:10px;border-radius:10px;'><b>{name}:</b><br>{md_to_html(content, extensions=['extra', 'fenced_code', 'sane_lists', 'nl2br'])}<br><hr><div style='font-size:small;color:#6c757d;'>{stats_html}</div></div>"
            elif role == 'system':
                chat_html += f"<div style='background-color:#cff4fc;padding:10px;margin:10px;border-radius:10px;'><b>System:</b><br>{content}</div>"
            else:
                chat_html += f"<div style='background-color:#e2e3e5;padding:10px;margin:10px;border-radius:10px;'><b>{role.capitalize()}:</b><br>{content}</div>"
        chat_html += "</body></html>"
        options = QFileDialog.Option(0)
        options |= QFileDialog.Option.DontUseNativeDialog
        save_path, _ = QFileDialog.getSaveFileName(self, "Export Chat as HTML", f"{chat.text()}.html", "HTML Files (*.html);;All Files (*)", options=options)
        if not save_path:
            return
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(chat_html)
            QMessageBox.information(self, "Success", f"Chat {chat.text()} exported successfully to {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export chat {chat.text()}: {e}")

    def export_chat_md(self):
        selected_chats = self.chatList.selectedItems()
        if not selected_chats:
            QMessageBox.information(self, "No Chat Selected", "Please select a chat to export.")
            return
        chat = selected_chats[0]
        chat_data = {}
        filename = chat.data(Qt.ItemDataRole.UserRole)
        try:
            path = Path("chats") / f"{filename}"
            with open(path, "r") as f:
                chat_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            QMessageBox.critical(self, "Error", "Failed to load the selected chat.")
            return
        chat_md = f"# {chat.text()}\n\n"
        for message in chat_data['history']:
            role = message['role']
            content = message['content']
            if role == 'user':
                chat_md += f"**User:**\n\n{content}\n\n"
            elif role == 'assistant':
                name = message.get('llm', 'Assistant')
                stats = message.get('stats', {})
                if not stats:
                    chat_md += f"**{name}:**\n\n{content}\n\n"
                else:
                    stats_md = f'''**{name} - Statistics:**
                        - **Time**: {stats.get('time', "Unavailable")}
                        - **Input (ms)**: {stats.get('input_ms', "Unavailable")}
                        - **Generation (ms)**: {stats.get('gen_ms', "Unavailable")}
                        - **Total (ms)**: {stats.get('total_ms', "Unavailable")}
                        - **Tokens**: {stats.get('total_t', "Unavailable")}
                        - **Input**: {stats.get('input_t', "Unavailable")}
                        - **Generated**: {stats.get('gen_t', "Unavailable")}
                        - **Tokens per second**: {stats.get('t_s', "Unavailable")}
                    '''
                    chat_md += f"**{name}:**\n\n{content}\n\n{stats_md}\n\n"
            elif role == 'system':
                chat_md += f"**System:**\n\n{content}\n\n"
            else:
                chat_md += f"**{role.capitalize()}:**\n\n{content}\n\n"
        options = QFileDialog.Option(0)
        options |= QFileDialog.Option.DontUseNativeDialog
        save_path, _ = QFileDialog.getSaveFileName(self, "Export Chat as Markdown", f"{chat.text()}.md", "Markdown Files (*.md);;All Files (*)", options=options)
        if not save_path:
            return
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(chat_md)
            QMessageBox.information(self, "Success", f"Chat {chat.text()} exported successfully to {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export chat {chat.text()}: {e}")

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

        path = Path("models")
        if not path.exists():
            path.mkdir(parents=False, exist_ok=True)

        try:
            path = path / "models.json"
            with open(path, "r") as f:
                models_json = json.load(f)
                self.models = models_json.get('models', [])
        except (FileNotFoundError, json.JSONDecodeError):
            self.models = []

        modelsLayout = QHBoxLayout()

        self.modelsTable = QTableWidget()
        self.modelsTable.setColumnCount(6)
        self.modelsTable.setHorizontalHeaderLabels(["Name", "Parameters","Weights", "Layers", "Size", "Path"])
        self.modelsTable.horizontalHeader().setStretchLastSection(True)
        self.modelsTable.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.modelsTable.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.modelsTable.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.modelsTable.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.modelsTable.itemSelectionChanged.connect(lambda: self.settings_model(change=True))
        self.modelsTable.setSortingEnabled(True)

        self.refresh_models_table()

        modelsLayout.addWidget(self.modelsTable, 65)

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
        size = info.get("size", "Unknown")

        self.models.append({
            "path": file_path,
            "name": name,
            "parameters": parameters,
            "weights": weights,
            "layers": layers,
            "size": size
        })
        QApplication.processEvents()
        self.refresh_models_table()

    def remove_model(self):
        selected_rows = self.modelsTable.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "Warning", "No model selected.")
            return
        row_n = selected_rows[0].row()
        row = int(self.modelsTable.item(row_n, 0).data(Qt.ItemDataRole.UserRole))
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
        self.loadLLMSettings(path=model.get("path", ""), type_f=1)

    def model_settings_switcher(self, checked):
        if not checked:
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
        self.modelsTable.setSortingEnabled(False)
        self.modelsTable.setRowCount(0)
        self.modelSelect.clear()
        path = Path("models")
        if not path.exists():
            path.mkdir(parents=False, exist_ok=True)
        path = path / "models.json"
        if not self.models:
            with open(path, "w") as f:
                json.dump({"models": []}, f, indent=4)
            return
        for model in self.models:
            row = self.modelsTable.rowCount()
            self.modelsTable.insertRow(row)
            self.modelsTable.setItem(row, 0, QTableWidgetItem(model.get("name", "Unknown")))
            self.modelsTable.setItem(row, 1, QTableWidgetItem(str(model.get("parameters", "Unknown"))))
            self.modelsTable.setItem(row, 2, QTableWidgetItem(str(model.get("weights", "Unknown"))))
            self.modelsTable.setItem(row, 3, QTableWidgetItem(str(model.get("layers", "Unknown"))))
            self.modelsTable.setItem(row, 4, QTableWidgetItem(str(model.get("size", "Unknown"))))
            self.modelsTable.setItem(row, 5, QTableWidgetItem(model.get("path", "Unknown")))
            self.modelsTable.item(row, 0).setData(Qt.ItemDataRole.UserRole, str(row))

            self.modelSelect.addItem(model.get("name", "Unknown") + " (" + model.get("weights", "Unknown") + ")", {"row": str(row), "path": model.get("path", "")})
        with open(path, "w") as f:
            json.dump({"models": self.models}, f, indent=4)
        QTimer.singleShot(0, lambda: self.modelsTable.resizeColumnToContents(0))
        self.modelsTable.setSortingEnabled(True)

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
        self.llmSettingsList.currentItemChanged.connect(lambda: self.loadLLMSettings(type_f=0))
        self.llmSettingsList.itemChanged.connect(self.save_settings_list)
        self.llmSettingsList.itemChanged.connect(lambda: self.reload_settings_select(save_selection=True))

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
            path = Path("settings")
            if self.llmSettingsList.count() == 0:
                self.loadLLMSettings(type_f=-1)
                settings = QListWidgetItem(title.strip())
                settings.setData(Qt.ItemDataRole.UserRole, f"settings_llm_default.json")
                settings.setFlags(settings.flags() | Qt.ItemFlag.ItemIsEditable)
                self.llmSettingsList.addItem(settings)
                path = path / settings.data(Qt.ItemDataRole.UserRole)
                self.saveLLMSettings(path=path, type_f=0)
                self.llmSettingsList.setCurrentItem(settings, QItemSelectionModel.SelectionFlag.ClearAndSelect)
                self.profileSelect.addItem(title, settings.data(Qt.ItemDataRole.UserRole))
                self.profileSelect.setCurrentIndex(0)
            else:
                self.saveLLMSettings(type_f=0)
                self.loadLLMSettings(type_f=-1)
                settings = QListWidgetItem(title.strip())
                settings.setData(Qt.ItemDataRole.UserRole, f"settings_llm_{self.llmSettingsList.count() - 1}.json")
                settings.setFlags(settings.flags() | Qt.ItemFlag.ItemIsEditable)
                self.llmSettingsList.addItem(settings)
                path = path / settings.data(Qt.ItemDataRole.UserRole)
                self.saveLLMSettings(path=path, type_f=0)
                self.llmSettingsList.setCurrentItem(settings, QItemSelectionModel.SelectionFlag.ClearAndSelect)
                self.profileSelect.addItem(title, settings.data(Qt.ItemDataRole.UserRole))
            self.save_settings_list()
    
    def llm_setting_changed(self, item, native=True, path=None, type_f=0):
        if not native:
            self.LLMSettings[item['name']] = item['value']
        else:
            data = item.data(Qt.ItemDataRole.UserRole)
            if data:
                self.LLMSettings[data['name']] = item.text()
                path = data['path']
                type_f = data['type']
        self.saveLLMSettings(path=path, type_f=type_f)


    def load_settings_list(self):
        try:
            path = Path("settings") / "settings_list.json"
            with open(path, "r") as f:
                settings_json = json.load(f)
                if settings_json.get("settings") is None:
                    self.create_new_settings("Default Settings")
                    return
                for settings in settings_json.get("settings", []):
                    item = QListWidgetItem(settings.get("title", "Untitled Settings"))
                    item.setData(Qt.ItemDataRole.UserRole, settings.get("filename", ""))
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                    self.llmSettingsList.addItem(item)
                    self.profileSelect.addItem(settings.get("title", "Untitled Settings"), settings.get("filename", ""))
                if self.profileSelect.count() > 0:
                    self.profileSelect.setCurrentIndex(0)
        except (FileNotFoundError, json.JSONDecodeError):
            self.create_new_settings("Default Settings")

    def reload_settings_select(self, save_selection=False):
        current_selection = None
        if save_selection and self.profileSelect.currentIndex() >= 0:
            current_selection = self.profileSelect.currentIndex()
        self.profileSelect.clear()
        for i in range(self.llmSettingsList.count()):
            item = self.llmSettingsList.item(i)
            self.profileSelect.addItem(item.text(), item.data(Qt.ItemDataRole.UserRole))
        if current_selection is not None and save_selection:
            self.profileSelect.setCurrentIndex(current_selection)

    def save_settings_list(self):
        path = Path("settings")
        if not path.exists():
            path.mkdir(parents=False, exist_ok=True)
        settings = []
        for i in range(self.llmSettingsList.count()):
            item = self.llmSettingsList.item(i)
            settings.append({"title": item.text(), "filename": item.data(Qt.ItemDataRole.UserRole)})
        if not settings:
            self.create_new_settings("Default Settings")
            return
        try:
            path = path / "settings_list.json"
            with open(path, "w") as f:
                json.dump({"settings": settings}, f, indent=4)
        except Exception as e:
            print(f"Error saving settings list: {e}")

    def loadLLMSettings(self, path=None, type_f=0, display=1):
        if type_f == 0 and self.llmSettingsList.currentItem() is None:
            return
        if type(path) == str:
            path = Path(path)
        try:
            if type_f == 0 and path is None:
                filename = self.llmSettingsList.currentItem().data(Qt.ItemDataRole.UserRole)
                path = Path("settings") / f"{filename}"
            if type_f == -1:
                path = path / "settings_llm_default.json"
            if type_f == 1:
                path = path.with_suffix(".json")
            if type_f == 2:
                if not path.name.endswith("_settings.json"):
                    path = path.with_name(path.stem + "_settings.json")
            with open(path, "r") as f:
                settings_json = json.load(f)
                self.LLMSettings = settings_json.get('settings', {})
                for setting in self.bpSettings:
                    if setting['name'] not in self.LLMSettings and type_f in setting['use_case']:
                        self.LLMSettings[setting['name']] = setting['default']
                        self.saveLLMSettings(path=path, type_f=type_f)
        except Exception as e:
            self.LLMSettings = {}
        
        if not self.LLMSettings:
            for setting in self.bpSettings:
                if type_f in setting['use_case']:
                    self.LLMSettings[setting['name']] = setting['default']
            if type_f == -1:
                return
            self.saveLLMSettings(path=path, type_f=type_f)

        if type_f == -1 or display == 0:
            return

        target = None
        if type_f == 0:
            target = self.LLMSettingsTable
        elif type_f == 1:
            target = self.LLMModelSettingsTable
        elif type_f == 2:
            target = self.chatSettingsTable
        target.setRowCount(0)
        for setting in self.bpSettings:
            if type_f in setting['use_case']:
                row = target.rowCount()
                target.insertRow(row)
                label = QTableWidgetItem(setting['display'])
                label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
                target.setItem(row, 0, label)
                value = ""
                if setting['type'] == 'text':
                    value = QTableWidgetItem(self.LLMSettings[setting['name']])
                    value.setData(Qt.ItemDataRole.UserRole, {"row": row, "name": setting['name'], "path": path, "type": type_f})
                elif setting['type'] == 'number':
                    value = QLineEdit()
                    value.setValidator(QIntValidator(1024, 65535, value))
                    value.setText(str(self.LLMSettings[setting['name']]))
                    value.textChanged.connect(lambda text, name=setting['name']: self.llm_setting_changed({'value': text, "name": name}, native=False, path=path, type_f=type_f))
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
                    slider.setWhatsThis(setting['name'])
                    value_layout.addWidget(slider)
                    curr_label = QLabel(self.LLMSettings[setting['name']])
                    slider.valueChanged.connect(lambda curr_value, slider_el = slider, label=curr_label: self.update_slider(slider_el,label, curr_value, path=path, type_f=type_f))
                    value_layout.addWidget(curr_label)
                    value.setLayout(value_layout)
                    self.update_slider(slider, curr_label, slider.value())
                    value.adjustSize()
                elif setting['type'] == 'combo':
                    value = QComboBox()
                    for option in setting['options']:
                        value.addItem(option)
                        if option == self.LLMSettings[setting['name']]:
                            value.setCurrentText(option) 
                    value.currentTextChanged.connect(lambda text, name=setting['name']: self.llm_setting_changed({'value': text, "name": name},native=False, path=path, type_f=type_f))
                elif setting['type'] == 'checkbox':
                    value = QCheckBox()
                    value.setChecked(bool(self.LLMSettings[setting['name']]))
                    value.stateChanged.connect(lambda state, name=setting['name']: self.llm_setting_changed({'value': state, "name": name}, native=False, path=path, type_f=type_f))
                elif setting['type'] == 'checkbox_group':
                    value = QWidget()
                    value_layout = QVBoxLayout()
                    value_part_layout = QHBoxLayout()
                    for i in range(len(setting['options'])):
                        if i % setting['max_per_line'] == 0:
                            value_part_layout = QHBoxLayout()
                            value_part_layout.setContentsMargins(2, 2, 2, 2)
                            value_part_layout.setSpacing(2)
                        option = setting['options'][i]
                        checkbox = QCheckBox(option)
                        checkbox.setChecked(bool(option in self.LLMSettings[setting['name']]))
                        checkbox.stateChanged.connect(lambda state, name=setting['name'], opt=option: self.llm_setting_changed_checkbox_group(state, name, opt, path=path, type_f=type_f))
                        value_part_layout.addWidget(checkbox)
                        if (i + 1) % setting['max_per_line'] == 0 or i == len(setting['options']) - 1:
                            value_layout.addLayout(value_part_layout)
                    value.setLayout(value_layout)
                elif setting['type'] == 'radiobutton':
                    value = ToggleSwitch()
                    value.setChecked(bool(self.LLMSettings[setting['name']]))
                    value.setMaximumWidth(45)
                    value.setMaximumHeight(22)
                    if setting['name'] == 'model_settings':
                        value.setToolTip("When turned on, it uses model settings for model instead of profile settings.")
                        value.toggled.connect(lambda checked: self.model_settings_switcher(checked))
                    if setting['name'] == 'chat_settings':
                        value.setToolTip("When turned on, it uses chat settings for chat instead of profile settings.")
                        value.toggled.connect(lambda checked: self.chat_settings_switcher(checked))
                    value.toggled.connect(lambda checked, name=setting['name']: self.llm_setting_changed({'value': checked, "name": name}, native=False, path=path, type_f=type_f))

                target.setCellWidget(row, 1, value) if not isinstance(value, QTableWidgetItem) else target.setItem(row, 1, value)
                target.resizeRowToContents(row)
        if type_f == 1:
            self.model_settings_switcher(self.LLMSettings.get('model_settings', False))
        elif type_f == 2:
            self.chat_settings_switcher(self.LLMSettings.get('chat_settings', False))

    def saveLLMSettings(self, path=None, type_f=0):
        if self.llmSettingsList.count() == 0:
            return
        path_b = Path("settings")
        if type(path) == str:
            path = Path(path)
        if not path_b.exists():
            path_b.mkdir(parents=False, exist_ok=True)
        if type_f == 0 and path is None:
            filename = self.llmSettingsList.currentItem().data(Qt.ItemDataRole.UserRole)
            path = path_b / f"{filename}"
        if type_f == 1:
            path = path.with_suffix(".json")
        if type_f == 2:
            if not path.name.endswith("_settings.json"):
                path = path.with_name(path.stem + "_settings.json")
        try:
            with open(path, "w") as f:
                json.dump({"settings": self.LLMSettings, "type": type_f}, f, indent=4)
        except Exception as e:
            print(f"Error saving LLM settings: {e}")

    def removeLLMSettings(self):
        selected_items = self.llmSettingsList.selectedItems()
        if not selected_items:
            return
        
        for settings in selected_items:
            self.llmSettingsList.takeItem(self.llmSettingsList.row(settings))
            self.profileSelect.removeItem(self.profileSelect.findData(settings.data(Qt.ItemDataRole.UserRole)))
            filename = settings.data(Qt.ItemDataRole.UserRole)
            try:
                path = Path("settings") / f"{filename}"
                path.unlink(missing_ok=True)
                self.save_settings_list()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete settings file: {e}")
                return

    def update_slider(self, slider, label, value, path=None, type_f=0):
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
        self.saveLLMSettings(path=path, type_f=type_f)
    
    def llm_setting_changed_checkbox_group(self, state, name, option, path=None, type_f=0):
        current_options = self.LLMSettings.get(name, [])
        if state > 0:
            if option not in current_options:
                current_options.append(option)
        else:
            if option in current_options:
                current_options.remove(option)
        self.LLMSettings[name] = current_options
        self.saveLLMSettings(path=path, type_f=type_f)
        self.update_chat_display()

if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = App()
	window.resize(1000, 600)
	window.show()
	sys.exit(app.exec())