import os
import signal
import math
from PyQt6.QtWidgets import (
	QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QLineEdit,
	QTabWidget, QMessageBox, QTableWidget, QTableWidgetItem, QSizePolicy, QFileDialog, QSplitter,
    QDialog, QListWidget, QListWidgetItem, QInputDialog, QComboBox, QCheckBox, QSlider, QFrame,
    QRadioButton, QScrollArea, QTextBrowser, QToolTip
)
from PyQt6.QtGui import (
    QTextCursor, QPixmap, QCursor, QIntValidator, QPainter, QColor, QBrush, QPen, QMouseEvent,
    QTextOption
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
    stats_emit = pyqtSignal(dict)

    def __init__(self, request, url):
        super().__init__()
        self.request = request
        self.reply = ""
        self._is_running = True
        self.url = url

    def run(self):
        self._is_running = True
        try:
            response = requests.post(self.url, json=self.request, stream=True)
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
                            self.stats_emit.emit(chunk)
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

class ToggleSwitch(QRadioButton):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setAutoExclusive(False)
		self.setCursor(Qt.CursorShape.PointingHandCursor)

		self._track_color_off = QColor("#e0e0e0")
		self._track_color_on = QColor("#34c759")
		self._handle_color = QColor("#ffffff")
		self._handle_shadow_color = QColor(0, 0, 0, 50)

		self._handle_position = self._get_target_handle_pos()
		self.animation = QPropertyAnimation(self, b"handle_position", self)
		self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
		self.animation.setDuration(200)  # ms

		self.toggled.connect(self._start_animation)

	def get_handle_position(self):
		return self._handle_position

	def set_handle_position(self, pos):
		self._handle_position = pos
		self.update()

	handle_position = pyqtProperty(float, fget=get_handle_position, fset=set_handle_position)

	def mouseReleaseEvent(self, event: QMouseEvent):
		super().mouseReleaseEvent(event)
		if event.button() == Qt.MouseButton.LeftButton:
			self.toggle()

	def _start_animation(self, checked):
		self.animation.stop()
		self.animation.setStartValue(self._handle_position)
		self.animation.setEndValue(self._get_target_handle_pos())
		self.animation.start()

	def paintEvent(self, event):
		painter = QPainter(self)
		painter.setRenderHint(QPainter.RenderHint.Antialiasing)

		height = self.height()
		width = self.width()

		track_radius = height / 2
		track_color = self._track_color_on if self.isChecked() else self._track_color_off
		painter.setPen(Qt.PenStyle.NoPen)
		painter.setBrush(QBrush(track_color))
		painter.drawRoundedRect(0, 0, width, height, track_radius, track_radius)

		padding = 2
		handle_radius = (height / 2) - padding
		x = int(self._handle_position)
		y = int(padding)

		painter.setBrush(QBrush(self._handle_shadow_color))
		painter.drawEllipse(x, y + 1, int(handle_radius * 2), int(handle_radius * 2))

		painter.setBrush(QBrush(self._handle_color))
		painter.setPen(QPen(self._track_color_off.darker(110)))
		painter.drawEllipse(x, y, int(handle_radius * 2), int(handle_radius * 2))

		painter.end()

	def _get_target_handle_pos(self):
		padding = 2
		handle_diameter = self.height() - 2 * padding
		return self.width() - handle_diameter - padding if self.isChecked() else padding

	def resizeEvent(self, event):
		super().resizeEvent(event)
		self._handle_position = self._get_target_handle_pos()

class ChatBubble(QFrame):
    def __init__(self, text, speaker):
        super().__init__()
        self.text = text
        self.speaker = speaker
        self.styleBase = ""
        self.margins = (0, 0, 0, 0)
        align = Qt.AlignmentFlag.AlignCenter
        self._suppress_bubble_pop = False

        if self.speaker == "user":
            self.speaker = "User"
            align = Qt.AlignmentFlag.AlignRight
        elif self.speaker == "assistant":
            self.speaker = "Assistant"
            align = Qt.AlignmentFlag.AlignLeft
        elif self.speaker == "system":
            self.speaker = "System"

        layout = QVBoxLayout()
        label = QLabel(self.speaker)
        layout.addWidget(label)
        self.textbox = ChatBubbleText(self.text, align=align)
        layout.addWidget(self.textbox)
        btnS = QHBoxLayout()
        copyBtn = QPushButton("🗎")
        copyBtn.setToolTip("Copy text to clipboard")
        copyBtn.clicked.connect(self.copy_to_clipboard)
        copyBtn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.deleteBtn = QPushButton("🗑")
        self.deleteBtn.setToolTip("Delete this bubble")
        self.deleteBtn.clicked.connect(self.deleteLater)
        self.deleteBtn.setVisible(False)
        self.deleteBtn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.generateBtn = QPushButton("Regenerate response")
        self.generateBtn.setToolTip("Generate an assistant response")
        self.generateBtn.setVisible(False)
        self.generateBtn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        btnS.addStretch()
        if self.speaker == "Assistant":
            self.statsBtn = HoverLabel("📊", "")
            self.statsBtn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            btnS.addWidget(self.statsBtn)
        btnS.addWidget(copyBtn)
        btnS.addWidget(self.deleteBtn)
        btnS.addWidget(self.generateBtn)
        layout.addLayout(btnS)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        if self.speaker == "User":
            self.styleBase = """
QFrame {
    background-color: #d1e7dd;
    border: 1px solid #badbcc;
    border-radius: 8px;
    padding: 8px;
    margin: 8px 16px 8px 144px;
}
QLabel {
    color: #0f5132;
    font-weight: bold;
}
QTextBrowser {
    color: #0f5132;
    font-weight: bold;
}
QPushButton {
    background-color: #badbcc;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    color: #0f5132;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #0f5132;
    color: #ffffff;
}
"""
            self.margins = (8, 16, 8, 144)
        elif self.speaker == "Assistant":
            self.styleBase = """
QFrame {
    background-color: #cff4fc;
    border: 1px solid #b6effb;
    border-radius: 8px;
    padding: 8px;
    margin: 8px 144px 8px 16px;
}
QLabel {
    color: #055160;
    font-weight: bold;
}
QTextBrowser {
    color: #055160;
    font-weight: bold;
}
QPushButton {
    background-color: #b6effb;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    color: #055160;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #055160;
    color: #ffffff;
}
"""
            self.margins = (8, 144, 8, 16)
        elif self.speaker == "System":
            self.styleBase ="""
QFrame {
    background-color: #e2e3e5;
    border: 1px solid #d3d6d8;
    border-radius: 8px;
    padding: 8px;
    margin: 8px 80px 8px 80px;
}
QLabel {
    color: #41464b;
    font-weight: bold;
}
QTextBrowser {
    color: #41464b;
    font-weight: bold;
}
QPushButton {
    background-color: #d3d6d8;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    color: #41464b;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #41464b;
    color: #ffffff;
}
"""
            self.margins = (8, 80, 8, 80)
        self.setStyleSheet(self.styleBase)

    def showEvent(self, e):
        self._applyResponsiveMargins()
        super().showEvent(e)

    def resizeEvent(self, e):
        self._applyResponsiveMargins()
        super().resizeEvent(e)

    def _basisWidth(self):
        w = self.parentWidget()
        while w and not isinstance(w, QScrollArea):
            w = w.parentWidget()
        if isinstance(w, QScrollArea):
            return max(1, w.viewport().width())
        win = self.window()
        return 800
    
    def _applyResponsiveMargins(self):
        if self.styleBase is None:
            return
        base = self._basisWidth()
        t0, r0, b0, l0 = self.margins
        k = base / float(800)
        clamp = lambda v: int(round(max(8, min(160, v))))
        mt, mr, mb, ml = map(clamp, (t0 * k, r0 * k, b0 * k, l0 * k))
        
        override = f"QFrame {{ margin: {mt}px {mr}px {mb}px {ml}px; }}"
        self.setStyleSheet(f"{self.styleBase}\n{override}")

    def copy_to_clipboard(self):
        QApplication.clipboard().setText(self.textbox.toPlainText())

class ChatBubbleText(QTextBrowser):
    def __init__(self, text="", align=Qt.AlignmentFlag.AlignCenter):
        super().__init__()
        self._align = align
        self.document().setDefaultTextOption(QTextOption(self._align))
        self.setReadOnly(True)
        self.setOpenExternalLinks(True)
        self.setUndoRedoEnabled(False)
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet("margin:0;padding:0;border:0;")
        self.document().setDocumentMargin(0)
        self.document().setDefaultStyleSheet("""
            p, pre, ul, ol, h1, h2, h3, h4, h5, h6 { margin-top:0px; margin-bottom:0px; }
            ul, ol { padding-left: 18px; }
            code, pre { font-family: monospace; }
        """)

        if "<" in text and "</" in text:
            self.setHtml(text)
        else:
            self.setPlainText(text)

        self.document().contentsChanged.connect(self._apply_height)
        self.document().documentLayout().documentSizeChanged.connect(lambda _=None: self._apply_height())

        self._apply_height()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._apply_height()

    def setPlainText(self, text: str) -> None:
        super().setPlainText(text)
        self._apply_height()

    def setHtml(self, html: str) -> None:
        super().setHtml(html)
        self._apply_height()

    def _apply_height(self):
        w = max(1, self.viewport().width())
        if self.document().textWidth() != w:
            self.document().setTextWidth(w)

        doc_h = math.ceil(self.document().documentLayout().documentSize().height())
        h = max(1, doc_h + 2 * self.frameWidth())
        self.setFixedHeight(h)
        self.updateGeometry()

class HoverLabel(QPushButton):
    def __init__(self, text, info):
        super().__init__(text)
        self.info = info

    def enterEvent(self, event):
        if self.info == "":
            return
        QToolTip.hideText()
        QToolTip.showText(
            self.mapToGlobal(self.rect().topLeft()),
            self.info,
            self
        )
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setMinimumSize(600, 338)
        self.setWindowTitle("Qully Chat")

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
            {'type': 'text', 'name': 'system_prompt', 'display': 'System prompt', 'default': 'You are a helpful assistant.', 'use_case': [0, 1, 2]}
        ]   # 0: llm settings tab; 1: llm model-specific settings; 2: chat-specific settings
        self.currentAddress = "http://127.0.0.1:5175/v1/chat/completions"
        self.last_stats = ""

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
            if os.path.exists(f"chats/{filename}_settings.json"):
                self.loadLLMSettings(path=f"chats/{filename}_settings.json", type=2, display=0)
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
        self.loadLLMSettings(path=self.models[int(idx)].get("path", ""), type=1, display=0)
        if self.LLMSettings.get('model_settings', False) == True:
            if settings_set == 0:
                settings_set = 1
                settings_build = self.LLMSettings.copy()
            else:
                for key, value in self.LLMSettings.items():
                    if key not in settings_build:
                        settings_build[key] = value
        ### else use profile settings
        self.loadLLMSettings(path=f"settings/{self.profileSelect.currentData()}", type=0, display=0)
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
        self.chatList.itemChanged.connect(self.save_chat_list)
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
        sendBtn.clicked.connect(self.send_prompt)

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

    def create_new_chat(self, title=None):
        ok = True
        if isinstance(title, bool):
            title = None
        if title is None:
            title, ok = QInputDialog.getText(self, "New Chat", "Enter chat title:")
        if ok and title.strip():
            self.save_chat()
            self.chatHistory = [{"role": "system", "content": self.LLMSettings['system_prompt']}]
            chat = QListWidgetItem(title.strip())
            chat.setData(Qt.ItemDataRole.UserRole, f"chat_{self.chatList.count()}.json")
            chat.setFlags(chat.flags() | Qt.ItemFlag.ItemIsEditable)
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
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                    self.chatList.addItem(item)
        except (FileNotFoundError, json.JSONDecodeError):
            self.create_new_chat("Default Chat")

    def load_chat(self):
        chat = self.chatList.currentItem()
        if chat:
            if not os.path.exists("chats"):
                os.makedirs("chats")
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
            if not os.path.exists("chats"):
                os.makedirs("chats")
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
        self.loadLLMSettings(path=f"chats/{filename}", type=2)

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
            self.convert_chat_toLegacy()
            QApplication.processEvents()
            bubble_a = ChatBubble("", "assistant")
            self.chatDisplay.addWidget(bubble_a)
            request = {"messages": self.chatLegacyHistory, "max_tokens": -1, "n_predict": -1, "stream": True}
            self.worker = LLMWorker(request, self.currentAddress)
            self.worker.token_emit.connect(lambda token: bubble_a.textbox.insertPlainText(token))
            self.worker.result_ready.connect(self.handle_reply)
            self.worker.error_emit.connect(lambda e: bubble_a.textbox.insertPlainText(e))
            self.worker.stats_emit.connect(lambda s: self.connect_stats(s))
            self.worker.start()
            self.worker.exec()
        else:
            QTest.mouseClick(self.modelSelect, Qt.MouseButton.LeftButton)
            
    def convert_chat_toLegacy(self):
        self.chatLegacyHistory = [self.chatHistory[0]]
        for message in self.chatHistory:
            if message['role'] == 'system':
                self.chatLegacyHistory[0] = message
            else:
                self.chatLegacyHistory.append(message)

    def handle_reply(self, reply):
        self.chatHistory.append({"role": "assistant", "content": reply.split("</think>")[1], "stats": self.last_stats})
        self.update_chat_display()

    def update_chat_display(self):
        self._suppress_bubble_pop = True
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
            content = md_to_html(content, extensions=["extra", "fenced_code", "sane_lists", "nl2br"])
            if role == 'user':
                bubble = ChatBubble(content, "user")
            elif role == 'assistant':
                bubble = ChatBubble(content, "assistant")
                stats = message.get('stats', {})
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
                bubble.statsBtn.info = stats_html
            elif role == 'system':
                bubble = ChatBubble(content, "system")
            else:
                bubble = ChatBubble(content, role)
            self.chatDisplay.addWidget(bubble)

        if hasattr(self, "chatDisplayWidget"):
            self.chatDisplayWidget.adjustSize()
            self.chatDisplayWidget.update()
        if hasattr(self, "chatDisplayScroll"):
            self.chatDisplayScroll.viewport().update()
        self.save_chat()

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
                last.deleteBtn.setVisible(True)
                if last.speaker == 'User':
                    last.generateBtn.setVisible(True)
                    last.generateBtn.clicked.connect(self.send_prompt)
            elif atype == "rem":
                last.deleteBtn.setVisible(True)
                if last.speaker == 'User':
                    last.generateBtn.setVisible(True)
                    last.generateBtn.clicked.connect(self.send_prompt)
                if not self._suppress_bubble_pop:
                    self.chatHistory.pop()
        except:
            pass

    def connect_stats(self, stats_d):
        self.last_stats = {'input_ms': round(stats_d['timings']['prompt_ms'], 2), 'gen_ms': round(stats_d['timings']['predicted_ms'], 2),
                           'total_ms': round(stats_d['timings']['prompt_ms'] + stats_d['timings']['predicted_ms'], 2),
                           'input_t': stats_d['usage']['prompt_tokens'], 'gen_t': stats_d['usage']['completion_tokens'],
                           'total_t': stats_d['usage']['total_tokens'], 't_s': round(stats_d['timings']['predicted_per_second'], 2)}

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

        if not os.path.exists("models"):
            os.makedirs("models")

        try:
            with open("models/models.json", "r") as f:
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

            self.modelSelect.addItem(model.get("name", "Unknown") + " (" + model.get("weights", "Unknown") + ")", {"row": str(row), "path": model.get("path", "")})

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
        self.modelsTable.setRowCount(0)
        self.modelSelect.clear()
        if not os.path.exists("models"):
            os.makedirs("models")
        if not self.models:
            with open("models/models.json", "w") as f:
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

            self.modelSelect.addItem(model.get("name", "Unknown") + " (" + model.get("weights", "Unknown") + ")", {"row": str(row), "path": model.get("path", "")})
        with open("models/models.json", "w") as f:
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
            if self.llmSettingsList.count() == 0:
                self.loadLLMSettings(type=-1)
                settings = QListWidgetItem(title.strip())
                settings.setData(Qt.ItemDataRole.UserRole, f"settings_llm_default.json")
                settings.setFlags(settings.flags() | Qt.ItemFlag.ItemIsEditable)
                self.llmSettingsList.addItem(settings)
                self.saveLLMSettings(path=f"settings/{settings.data(Qt.ItemDataRole.UserRole)}", type=0)
                self.llmSettingsList.setCurrentItem(settings, QItemSelectionModel.SelectionFlag.ClearAndSelect)
                self.profileSelect.addItem(title, settings.data(Qt.ItemDataRole.UserRole))
                self.profileSelect.setCurrentIndex(0)
            else:
                self.saveLLMSettings(type=0)
                self.loadLLMSettings(type=-1)
                settings = QListWidgetItem(title.strip())
                settings.setData(Qt.ItemDataRole.UserRole, f"settings_llm_{self.llmSettingsList.count() - 1}.json")
                settings.setFlags(settings.flags() | Qt.ItemFlag.ItemIsEditable)
                self.llmSettingsList.addItem(settings)
                self.saveLLMSettings(path=f"settings/{settings.data(Qt.ItemDataRole.UserRole)}", type=0)
                self.llmSettingsList.setCurrentItem(settings, QItemSelectionModel.SelectionFlag.ClearAndSelect)
                self.profileSelect.addItem(title, settings.data(Qt.ItemDataRole.UserRole))
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

    def loadLLMSettings(self, path=None, type=0, display=1):
        if display == 0:
            self.LLMSettingsTable.setRowCount(0)
            self.LLMModelSettingsTable.setRowCount(0)
            self.chatSettingsTable.setRowCount(0)
        if type == 0 and self.llmSettingsList.currentItem() is None:
            return
        try:
            if type == 0 and path is None:
                filename = self.llmSettingsList.currentItem().data(Qt.ItemDataRole.UserRole)
                path = f"settings/{filename}"
            if type == -1:
                path = f"settings/settings_llm_default.json"
            if type == 1:
                path = path[:-5]
                path = f"{path}.json"
            if type == 2:
                if not path.endswith("_settings.json"):
                    path = path[:-5]
                    path = f"{path}_settings.json"
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

        if type == -1 or display == 0:
            return

        target = None
        if type == 0:
            target = self.LLMSettingsTable
        elif type == 1:
            target = self.LLMModelSettingsTable
        elif type == 2:
            target = self.chatSettingsTable
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
                    slider.setWhatsThis(setting['name'])
                    value_layout.addWidget(slider)
                    curr_label = QLabel(self.LLMSettings[setting['name']])
                    slider.valueChanged.connect(lambda curr_value, slider_el = slider, label=curr_label: self.update_slider(slider_el,label, curr_value, path=path, type=type))
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
                    value.currentTextChanged.connect(lambda text, name=setting['name']: self.llm_setting_changed({'value': text, "name": name},native=False, path=path, type=type))
                elif setting['type'] == 'checkbox':
                    value = QCheckBox()
                    value.setChecked(bool(self.LLMSettings[setting['name']]))
                    value.stateChanged.connect(lambda state, name=setting['name']: self.llm_setting_changed({'value': state, "name": name}, native=False, path=path, type=type))
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
                    value.toggled.connect(lambda checked, name=setting['name']: self.llm_setting_changed({'value': checked, "name": name}, native=False, path=path, type=type))

                target.setCellWidget(row, 1, value) if not isinstance(value, QTableWidgetItem) else target.setItem(row, 1, value)
                target.resizeRowToContents(row)
        if type == 1:
            self.model_settings_switcher(self.LLMSettings.get('model_settings', False))
        elif type == 2:
            self.chat_settings_switcher(self.LLMSettings.get('chat_settings', False))

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
        if type == 2:
            if not path.endswith("_settings.json"):
                path = path[:-5]
                path = f"{path}_settings.json"
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
            self.profileSelect.removeItem(self.profileSelect.findData(settings.data(Qt.ItemDataRole.UserRole)))
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