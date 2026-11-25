# Лабораторная работа №10. Построение кривых Безье и Чайкина.
# Требуются: Python 3, PyQt6, numpy.
import sys
import numpy as np
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QPushButton,
    QMessageBox, QGroupBox
)


def de_casteljau(points, t):
    """Возвращает точку кривой Безье и уровни промежуточных точек."""
    levels = [np.asarray(points, dtype=float)]
    while len(levels[-1]) > 1:
        prev = levels[-1]
        levels.append((1.0 - t) * prev[:-1] + t * prev[1:])
    return levels[-1][0], levels


def bezier_points(points, count=300):
    """Вычисляет точки кривой Безье для t от 0 до 1."""
    ts = np.linspace(0.0, 1.0, count)
    result = [de_casteljau(points, float(t))[0] for t in ts]
    return np.asarray(result)


def chaikin(points, iterations=2):
    """Одно уточнение Чайкина: Q=3/4 Pi+1/4 Pi+1, R=1/4 Pi+3/4 Pi+1."""
    result = np.asarray(points, dtype=float)
    for _ in range(iterations):
        refined = [result[0]]
        for p, q in zip(result[:-1], result[1:]):
            refined.append(0.75 * p + 0.25 * q)
            refined.append(0.25 * p + 0.75 * q)
        refined.append(result[-1])
        result = np.asarray(refined)
    return result


class CurveCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(650, 430)
        self.mode = 0
        self.t = 0.35
        self.iterations = 3
        self.points = [
            np.array([0.12, 0.20]), np.array([0.30, 0.88]),
            np.array([0.70, 0.82]), np.array([0.88, 0.22])
        ]
        self.drag_index = None

    def set_mode(self, mode):
        self.mode = mode
        if mode == 0 and len(self.points) != 3:
            self.points = [np.array([0.12, 0.20]), np.array([0.50, 0.88]), np.array([0.88, 0.20])]
        elif mode == 1 and len(self.points) != 4:
            self.points = [np.array([0.12, 0.20]), np.array([0.30, 0.88]),
                           np.array([0.70, 0.82]), np.array([0.88, 0.22])]
        elif mode == 2 and len(self.points) < 3:
            self.points = [np.array([0.10, 0.25]), np.array([0.30, 0.82]),
                           np.array([0.55, 0.30]), np.array([0.80, 0.78]),
                           np.array([0.92, 0.25])]
        self.update()

    def set_t(self, value):
        self.t = value
        self.update()

    def set_iterations(self, value):
        self.iterations = value
        self.update()

    def reset_points(self):
        if self.mode == 0:
            self.points = [np.array([0.12, 0.20]), np.array([0.50, 0.88]), np.array([0.88, 0.20])]
        elif self.mode == 1:
            self.points = [np.array([0.12, 0.20]), np.array([0.30, 0.88]),
                           np.array([0.70, 0.82]), np.array([0.88, 0.22])]
        else:
            self.points = [np.array([0.10, 0.25]), np.array([0.30, 0.82]),
                           np.array([0.55, 0.30]), np.array([0.80, 0.78]),
                           np.array([0.92, 0.25])]
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            x = np.clip(event.position().x() / self.width(), 0.03, 0.97)
            y = np.clip(1.0 - event.position().y() / self.height(), 0.05, 0.95)
            clicked = np.array([x, y])
            # Если щёлкнули рядом с существующей вершиной, выбираем её для перемещения.
            distances = [np.linalg.norm((p - clicked) * np.array([self.width(), self.height()]))
                         for p in self.points]
            nearest = int(np.argmin(distances)) if distances else -1
            if nearest >= 0 and distances[nearest] < 22:
                self.drag_index = nearest
            elif self.mode == 2:
                self.points.append(clicked)
                self.update()
            else:
                required = 3 if self.mode == 0 else 4
                if len(self.points) < required:
                    self.points.append(clicked)
                    self.update()
                else:
                    QMessageBox.information(
                        self, "Контрольные точки",
                        "Для изменения формы перетащите существующую точку мышью. "
                        "Для нового набора нажмите «Сбросить точки»."
                    )

    def mouseMoveEvent(self, event):
        if self.drag_index is not None and event.buttons() & Qt.MouseButton.LeftButton:
            x = np.clip(event.position().x() / self.width(), 0.03, 0.97)
            y = np.clip(1.0 - event.position().y() / self.height(), 0.05, 0.95)
            self.points[self.drag_index] = np.array([x, y])
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_index = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fbfcff"))
        w, h = self.width(), self.height()

        def xy(p):
            return QPointF(float(p[0] * w), float((1.0 - p[1]) * h))

        # Лёгкая координатная сетка.
        grid_pen = QPen(QColor("#e6eaf0"), 1)
        painter.setPen(grid_pen)
        for k in range(1, 10):
            x = int(w * k / 10)
            y = int(h * k / 10)
            painter.drawLine(x, 0, x, h)
            painter.drawLine(0, y, w, y)

        pts = np.asarray(self.points, dtype=float)
        if len(pts) >= 2:
            painter.setPen(QPen(QColor("#9aa5b1"), 2, Qt.PenStyle.DashLine))
            for a, b in zip(pts[:-1], pts[1:]):
                painter.drawLine(xy(a), xy(b))

        if self.mode in (0, 1) and len(pts) == (3 if self.mode == 0 else 4):
            curve = bezier_points(pts)
            painter.setPen(QPen(QColor("#1769aa"), 3))
            for a, b in zip(curve[:-1], curve[1:]):
                painter.drawLine(xy(a), xy(b))

            _, levels = de_casteljau(pts, self.t)
            colors = ["#e09f3e", "#2a9d8f", "#8e44ad"]
            for level_index, level in enumerate(levels[:-1]):
                painter.setPen(QPen(QColor(colors[min(level_index, 2)]), 2))
                for a, b in zip(level[:-1], level[1:]):
                    painter.drawLine(xy(a), xy(b))
                painter.setBrush(QBrush(QColor(colors[min(level_index, 2)])))
                for p in level:
                    q = xy(p)
                    painter.drawEllipse(q, 4, 4)
            painter.setBrush(QBrush(QColor("#d62828")))
            painter.setPen(QPen(QColor("#d62828"), 1))
            q = xy(levels[-1][0])
            painter.drawEllipse(q, 6, 6)
        elif self.mode == 2 and len(pts) >= 3:
            refined = chaikin(pts, self.iterations)
            painter.setPen(QPen(QColor("#1769aa"), 3))
            for a, b in zip(refined[:-1], refined[1:]):
                painter.drawLine(xy(a), xy(b))

        painter.setBrush(QBrush(QColor("#f4a261")))
        painter.setPen(QPen(QColor("#7f5539"), 1))
        for p in pts:
            painter.drawEllipse(xy(p), 6, 6)

        painter.setPen(QPen(QColor("#243447"), 1))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(12, 20, "Контрольные точки — оранжевые; итоговая кривая — синяя")
        if self.mode in (0, 1) and len(pts) == (3 if self.mode == 0 else 4):
            painter.drawText(12, 38, f"Параметр t = {self.t:.2f}; красная точка — P(t)")
        elif self.mode == 2:
            painter.drawText(12, 38, f"Уточнений Чайкина: {self.iterations}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа №10 — построение кривых")
        self.resize(900, 620)
        root = QWidget()
        layout = QVBoxLayout(root)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Алгоритм:"))
        self.mode = QComboBox()
        self.mode.addItems(["Безье 2-й степени", "Безье 3-й степени", "Чайкин"])
        controls.addWidget(self.mode)

        controls.addWidget(QLabel("Параметр t:"))
        self.t_spin = QDoubleSpinBox()
        self.t_spin.setRange(0.0, 1.0)
        self.t_spin.setSingleStep(0.05)
        self.t_spin.setValue(0.35)
        controls.addWidget(self.t_spin)

        controls.addWidget(QLabel("Итерации:"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 5)
        self.iter_spin.setValue(3)
        controls.addWidget(self.iter_spin)

        self.reset_btn = QPushButton("Сбросить точки")
        controls.addWidget(self.reset_btn)
        layout.addLayout(controls)

        self.canvas = CurveCanvas()
        layout.addWidget(self.canvas, 1)

        note = QLabel("Перетаскивайте контрольные точки мышью, чтобы менять форму кривой. Для Чайкина можно добавлять точки щелчком по свободному месту. Кнопка «Сбросить точки» возвращает пример.")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.mode.currentIndexChanged.connect(self.on_mode)
        self.t_spin.valueChanged.connect(self.canvas.set_t)
        self.iter_spin.valueChanged.connect(self.canvas.set_iterations)
        self.reset_btn.clicked.connect(self.canvas.reset_points)
        self.canvas.set_t(self.t_spin.value())
        self.canvas.set_iterations(self.iter_spin.value())
        self.setCentralWidget(root)

    def on_mode(self, index):
        self.canvas.set_mode(index)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
