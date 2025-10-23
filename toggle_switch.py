from PyQt6.QtWidgets import (
	QRadioButton
)
from PyQt6.QtGui import (
    QPainter, QColor, QBrush, QPen, QMouseEvent
)
from PyQt6.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, pyqtProperty
)

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