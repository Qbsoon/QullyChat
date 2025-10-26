import math
from PyQt6.QtWidgets import (
	QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QSizePolicy, QFrame,
    QScrollArea, QTextBrowser, QStackedLayout, QTreeWidget, QTreeWidgetItem
)
from PyQt6.QtGui import (
    QTextOption, QFontDatabase, QFont, QBrush, QColor
)
from PyQt6.QtCore import (
    Qt
)
from hover_label import HoverLabel

class ChatBubble(QFrame):
    def __init__(self, text, speaker, llm = None, think = None):
        super().__init__()
        self.text = text
        self.speaker = speaker
        self.speaker_print = ""
        self.styleBase = ""
        self.margins = (0, 0, 0, 0)
        self.think = think
        align = Qt.AlignmentFlag.AlignCenter

        font_id = QFontDatabase.addApplicationFont("FiraCodeNerdFont-Regular.ttf")
        self.font_family = QFontDatabase.applicationFontFamilies(font_id)[0]

        if self.speaker == "user":
            self.speaker = "User"
            self.speaker_print = self.speaker
            align = Qt.AlignmentFlag.AlignRight
        elif self.speaker == "assistant":
            self.speaker = "Assistant"
            self.speaker_print = llm if llm else self.speaker
            align = Qt.AlignmentFlag.AlignLeft
        elif self.speaker == "system":
            self.speaker = "System"
            self.speaker_print = self.speaker

        layout = QStackedLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        mainPage = QWidget()
        mainPage.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        mp_layout = QVBoxLayout()
        self.label = QLabel(self.speaker_print)
        mp_layout.addWidget(self.label)
        if think:
            self.thinkbox = CollapsableThink(think)
            self.thinkbox.setStyleSheet("margin: 0px;")
            mp_layout.addWidget(self.thinkbox)
        self.textbox = ChatBubbleText(self.text, align=align)
        mp_layout.addWidget(self.textbox, 10)

        btnS = QHBoxLayout()
        btnS.addStretch()
        btnS.setSizeConstraint(QHBoxLayout.SizeConstraint.SetMinimumSize)

        if self.speaker == "Assistant":
            self.statsBtn = HoverLabel("📊", "")
            self.statsBtn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            btnS.addWidget(self.statsBtn)

        copyBtn = QPushButton("🗎")
        copyBtn.setToolTip("Copy text to clipboard")
        copyBtn.clicked.connect(self.copy_to_clipboard)
        copyBtn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        btnS.addWidget(copyBtn)

        if self.speaker == "User":
            self.editBtn = QPushButton("🖍️")
            self.editBtn.setToolTip("Edit this prompt")
            self.editBtn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            self.editBtn.clicked.connect(self.edit)
            btnS.addWidget(self.editBtn)
        
        self.branchBtn = QPushButton("\ue0a0")
        self.branchBtn.setFont(QFont(self.font_family))
        self.branchBtn.setToolTip("Branch from this bubble to a new chat")
        self.branchBtn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        btnS.addWidget(self.branchBtn)

        self.deleteBtn = QPushButton("🗑")
        self.deleteBtn.setToolTip("Delete this bubble")
        self.deleteBtn.clicked.connect(self.deleteLater)
        self.deleteBtn.setVisible(False)
        self.deleteBtn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        btnS.addWidget(self.deleteBtn)

        self.deleteDownBtn = QPushButton("🗑⬇")
        self.deleteDownBtn.setToolTip("Delete all below bubbles")
        self.deleteDownBtn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        btnS.addWidget(self.deleteDownBtn)

        self.generateBtn = QPushButton("Regenerate response")
        self.generateBtn.setToolTip("Generate an assistant response")
        self.generateBtn.setVisible(False)
        self.generateBtn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        btnS.addWidget(self.generateBtn)

        mp_layout.addLayout(btnS, 1)
        mainPage.setLayout(mp_layout)
        layout.addWidget(mainPage)

        if self.speaker == "User":
            editPage = QWidget()
            ed_layout = QVBoxLayout()
            self.editbox = QTextEdit()
            self.editbox.setPlainText(self.text)
            self.editbox.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
            self.editbox.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            self.editbox.setMaximumHeight(self.textbox.height()*4)
            self.editbox.setStyleSheet("QTextEdit { color: black; }")
            ed_layout.addWidget(self.editbox)
            ed_BtnS = QHBoxLayout()
            ed_BtnS.addStretch()

            self.saveBtn = QPushButton("💾 && 🗑⬇")
            self.saveBtn.setToolTip("Save changes and delete all bubbles below")
            self.saveBtn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            ed_BtnS.addWidget(self.saveBtn)

            self.editBranchBtn = QPushButton("\ue0a0")
            self.editBranchBtn.setFont(QFont(self.font_family))
            self.editBranchBtn.setToolTip("Branch to a new chat with edits")
            self.editBranchBtn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            ed_BtnS.addWidget(self.editBranchBtn)

            cancelBtn = QPushButton("❌")
            cancelBtn.setToolTip("Cancel editing")
            cancelBtn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            cancelBtn.clicked.connect(self.cancel_edit)
            ed_BtnS.addWidget(cancelBtn)

            ed_layout.addLayout(ed_BtnS)
            editPage.setLayout(ed_layout)
            layout.addWidget(editPage)

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
        if self.think:
            current_state = self.thinkbox.parent_item.isExpanded()
            self.thinkbox.parent_item.setExpanded(False)
            self.thinkbox.parent_item.setExpanded(current_state)

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
    
    def edit(self):
        self.editbox.setPlainText(self.text)
        self.layout().setCurrentIndex(1)

    def cancel_edit(self):
        self.editbox.setPlainText(self.text)
        self.layout().setCurrentIndex(0)

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

class CollapsableThink(QWidget):
    def __init__(self, text):
        super().__init__()
        layout = QVBoxLayout(self)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemExpanded.connect(self.on_expand)
        self.tree.itemCollapsed.connect(self.on_collapse)
        self.tree.setMaximumHeight(40)
        self.tree.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tree.setUniformRowHeights(False)

        self.parent_item = QTreeWidgetItem(self.tree)
        self.parent_item.setText(0, "Thought process")
        self.parent_item.setForeground(0, QBrush(QColor("#055160")))
        f = self.tree.font()
        f.setBold(True)
        self.parent_item.setFont(0, f)
        self.parent_item.setExpanded(False)

        self.child_item = QTreeWidgetItem(self.parent_item)
        self.think_widget = ChatBubbleText(text, align=Qt.AlignmentFlag.AlignLeft)
        self.tree.setItemWidget(self.child_item, 0, self.think_widget)
        self.think_widget.setVisible(False)

        layout.addWidget(self.tree)
        self.parent_item.setExpanded(True)
        self.parent_item.setExpanded(False)

        style = '''
QTreeWidget {
    background-color: #a6c3ca;
}
QFrame {
    background-color: #a6c3ca !important;
}
'''
        self.tree.setStyleSheet(style)

    def on_expand(self, item):
        if item == self.parent_item:
            self.think_widget.setVisible(True)
            self.tree.resizeColumnToContents(0)
            self.tree.viewport().update()
            self.tree.setMinimumHeight(round(self.think_widget.height()) + 40)
            self.tree.setMaximumHeight(6000)
    
    def on_collapse(self, item):
        if item == self.parent_item:
            self.think_widget.setVisible(False)
            self.tree.setMinimumHeight(40)
            self.tree.setMaximumHeight(40)