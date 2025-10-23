from PyQt6.QtWidgets import (
	QPushButton, QToolTip
)
from PyQt6.QtCore import (
    QPoint
)

class HoverLabel(QPushButton):
    def __init__(self, text, info):
        super().__init__(text)
        self.info = info

    def enterEvent(self, event):
        if self.info == "":
            return
        QToolTip.hideText()
        QToolTip.showText(
            self.mapToGlobal(QPoint(0, self.height())),
            self.info,
            self
        )
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)