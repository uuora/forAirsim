"""PyQt5 workbench for YOLO proposals and VLX-Seek grounding."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QTextEdit, QFileDialog, QGroupBox,
                             QSplitter, QProgressBar, QListWidget, QScrollArea,
                             QFrame, QMessageBox, QTabWidget, QComboBox, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QFontDatabase
from dotenv import load_dotenv

# Load environment
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from vlx_api_client import VLXSeekAPIClient


def configure_qt_fonts():
    """Register Windows fonts when a headless Qt plugin has no font database."""
    database = QFontDatabase()
    families = set(database.families())
    for family, filename in [
            ("Microsoft YaHei UI", "msyh.ttc"),
            ("Segoe UI", "segoeui.ttf"),
            ("Consolas", "consola.ttf")]:
        path = Path("C:/Windows/Fonts") / filename
        if family not in families and path.is_file():
            QFontDatabase.addApplicationFont(str(path))
    app = QApplication.instance()
    if app is not None:
        app.setFont(QFont("Microsoft YaHei UI", 10))


class DetectionThread(QThread):
    """Background thread for YOLO detection."""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, image, profile="balloon"):
        super().__init__()
        self.image = image.copy()
        self.profile = profile

    def run(self):
        try:
            if self.profile == "balloon":
                from t1_qt_evidence import read_json, sha256
                from specialist_balloon_detector import SpecialistBalloonDetector
                config = read_json(ROOT / "configs/t1_balloon_detector_v1p2.json")
                if (sha256(config["weights"]) != config["weights_sha256"] or
                        sha256(config["training_receipt"]) != config["training_receipt_sha256"]):
                    raise ValueError("气球模型或训练回执与配置的校验值不一致。")
                detector = SpecialistBalloonDetector(
                    config["weights"], confidence=config["confidence"],
                    imgsz=config["imgsz"], nms_iou=config["nms_iou"], device="cpu")
            elif self.profile == "general":
                from yolo_detector import YoloTargetDetector
                detector = YoloTargetDetector(ROOT / "models" / "yolo11n.pt",
                                              confidence=0.1, imgsz=1280, device="cpu")
            else:
                raise ValueError("未知检测模型配置。")
            result = detector.detect(self.image)
            boxes = []
            for candidate in result["candidates"]:
                boxes.append({
                    "bbox": candidate["bbox_xyxy_pixels"],
                    "class": candidate["class_name"],
                    "confidence": candidate["confidence"],
                    "source": "气球YOLO" if self.profile == "balloon" else "通用YOLO",
                })
            self.finished.emit(boxes)
        except Exception as e:
            self.error.emit(str(e))


class VLXThread(QThread):
    """Background thread for VLX API call."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, vlx_client, pil_image, boxes, description):
        super().__init__()
        self.vlx_client = vlx_client
        self.pil_image = pil_image
        self.boxes = boxes
        self.description = description

    def run(self):
        try:
            started = time.perf_counter()
            lang = "zh" if any("\u4e00" <= char <= "\u9fff" for char in self.description) else "en"
            result = self.vlx_client.ground(
                image=self.pil_image,
                boxes=self.boxes,
                description=self.description,
                lang=lang,
                temperature=0.0,
                max_new_tokens=128
            )
            result = dict(result, gui_call_wall_s=time.perf_counter()-started,
                          gui_request_description=self.description)
            self.finished.emit(result)
        except Exception:
            self.error.emit("VLX服务未返回有效结果，请检查本地API配置或网络。")


class ImageCanvas(QLabel):
    """Canvas for displaying image with bounding boxes."""
    box_drawn = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setStyleSheet("QLabel { background-color: #e8eef4; border-radius: 8px; }")
        self.setAlignment(Qt.AlignCenter)
        self.setText("加载图像后拖动鼠标添加候选框")

        self.original_image = None
        self.yolo_boxes = []
        self.vlx_box = None
        self.drag_start = None
        self.drag_end = None

    def set_image(self, cv_image):
        """Set the base image."""
        self.original_image = cv_image.copy()
        self.update_display()

    def set_yolo_boxes(self, boxes):
        """Set YOLO detection boxes."""
        self.yolo_boxes = boxes
        self.update_display()

    def set_vlx_box(self, box):
        """Set VLX selected box."""
        self.vlx_box = box
        self.update_display()

    def clear_vlx(self):
        """Clear VLX selection."""
        self.vlx_box = None
        self.update_display()

    def image_point(self, position):
        if self.original_image is None:
            return None
        h, w = self.original_image.shape[:2]
        scale = min(self.width() / w, self.height() / h)
        shown_w, shown_h = w * scale, h * scale
        left = (self.width() - shown_w) / 2
        top = (self.height() - shown_h) / 2
        if not (left <= position.x() <= left + shown_w and
                top <= position.y() <= top + shown_h):
            return None
        return (max(0, min(w - 1, round((position.x() - left) / scale))),
                max(0, min(h - 1, round((position.y() - top) / scale))))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start = self.image_point(event.pos())
            self.drag_end = self.drag_start

    def mouseMoveEvent(self, event):
        if self.drag_start is not None:
            self.drag_end = self.image_point(event.pos()) or self.drag_end
            self.update_display()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self.drag_start is None:
            return
        end = self.image_point(event.pos()) or self.drag_end
        if end is not None:
            box = [min(self.drag_start[0], end[0]), min(self.drag_start[1], end[1]),
                   max(self.drag_start[0], end[0]), max(self.drag_start[1], end[1])]
            if box[2] - box[0] >= 8 and box[3] - box[1] >= 8:
                self.box_drawn.emit(box)
        self.drag_start = self.drag_end = None
        self.update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()

    def update_display(self):
        """Render image with boxes."""
        if self.original_image is None:
            return

        display = self.original_image.copy()
        h, w = display.shape[:2]

        # Draw YOLO boxes (green)
        for i, box in enumerate(self.yolo_boxes):
            x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            confidence = box.get("confidence")
            # OpenCV's built-in font cannot render Chinese; the sidebar keeps
            # the full localized label.
            canvas_class = "manual" if box.get("source") == "手动" else box["class"]
            label = f"{i+1}: {canvas_class}" + (f" ({confidence:.2f})" if confidence is not None else "")
            font_scale = 0.5
            thickness = 1
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                     font_scale, thickness)

            cv2.rectangle(display, (x1, y1 - label_h - 8),
                         (x1 + label_w + 4, y1), (0, 255, 0), -1)
            cv2.putText(display, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        if self.drag_start is not None and self.drag_end is not None:
            cv2.rectangle(display, self.drag_start, self.drag_end, (180, 160, 0), 2)

        # Draw VLX selection (red, thick)
        if self.vlx_box:
            x1 = int(self.vlx_box['xmin'])
            y1 = int(self.vlx_box['ymin'])
            x2 = int(self.vlx_box['xmax'])
            y2 = int(self.vlx_box['ymax'])

            cv2.rectangle(display, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 6)

            label = "VLX SELECTED"
            font_scale = 0.8
            thickness = 2
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                     font_scale, thickness)

            label_y = max(y1 - 15, label_h + 10)
            cv2.rectangle(display, (x1-5, label_y - label_h - 10),
                         (x1 + label_w + 10, label_y + 5), (0, 0, 255), -1)
            cv2.putText(display, label, (x1, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Convert to QPixmap
        rgb_image = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale to fit widget
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)


class VLXSeekGUI(QMainWindow):
    """Main GUI window."""

    def __init__(self, *, auto_load=True, initialize_api=True):
        super().__init__()
        configure_qt_fonts()
        self.setWindowTitle("AirSim · 感知轨迹实验工作台")
        self.resize(1440, 880)
        self.setMinimumSize(1050, 700)

        # Initialize models
        self.vlx = None
        self.current_image = None
        self.yolo_boxes = []
        self.busy = False

        self.setup_ui()
        if initialize_api:
            self.init_models()
        else:
            self.refresh_controls()
            self.statusBar().showMessage("本地界面检查 · 未初始化API")
        if auto_load:
            QTimer.singleShot(0, self.research.load_latest)

    def setup_ui(self):
        """Setup the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f3f6fa; color: #203247;
                font-family: 'Microsoft YaHei UI', 'Segoe UI'; font-size: 13px; }
            QGroupBox { background: white; border: 1px solid #dfe6ee;
                border-radius: 10px; margin-top: 12px; padding: 12px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px;
                padding: 0 5px; color: #193b5a; }
            QPushButton { background: #eaf0f6; border: 1px solid #d9e3ee;
                border-radius: 7px; padding: 8px 12px; color: #1d405d; font-weight: 600; }
            QPushButton:hover { background: #dceaf5; }
            QPushButton:disabled { background: #edf0f4; color: #9aa8b5; }
            QLineEdit, QTextEdit, QListWidget { background: white; border: 1px solid #d6e0ea;
                border-radius: 7px; padding: 8px; selection-background-color: #cde9ed; }
            QProgressBar { border: none; background: #dce8f1; max-height: 5px; }
            QProgressBar::chunk { background: #0c8b80; }
            QTabWidget::pane { border: none; }
            QTabBar::tab { background: #e9eef3; padding: 12px 22px;
                margin-right: 3px; border-bottom: 3px solid transparent; }
            QTabBar::tab:selected { background: #f3f6fa; color: #146b72;
                border-bottom-color: #0c8b80; }
            QFrame#metricCard { background: white; border: 1px solid #dfe6ee;
                border-radius: 8px; padding: 3px; }
            QLabel#researchTitle { font-size: 22px; font-weight: 600; color: #173b4d; }
            QLabel#metricValue { font-size: 24px; font-weight: 600; color: #173b4d; }
            QLabel#muted { color: #617681; }
            QLabel#evidenceBadge { color: #38635f; padding: 7px 12px;
                background: #e3efeb; border-radius: 5px; }
            QLabel#replayImage { background: #dfe7ed; border: 1px solid #d5e0e8;
                border-radius: 6px; color: #526774; }
            QSlider::groove:horizontal { height: 4px; background: #c9d8e1; }
            QSlider::handle:horizontal { background: #188c83; width: 13px;
                margin: -5px 0; border-radius: 6px; }
        """)

        # Main layout
        shell = QVBoxLayout(central)
        shell.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        shell.addWidget(self.tabs)
        image_page = QWidget()
        main_layout = QHBoxLayout(image_page)
        main_layout.setContentsMargins(18, 18, 18, 12)

        # Left panel - Image canvas
        left_panel = QVBoxLayout()

        self.canvas = ImageCanvas()
        self.canvas.box_drawn.connect(self.add_manual_box)
        left_panel.addWidget(self.canvas)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        left_panel.addWidget(self.progress)

        # Right panel - Controls
        right_panel = QVBoxLayout()

        # Title
        title = QLabel("YOLO＋VLX · 单图验证")
        title_font = QFont("Microsoft YaHei UI", 17, QFont.Bold)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(title)

        # Step 1: Load Image
        step1_group = QGroupBox("01  加载图像")
        step1_layout = QVBoxLayout()

        self.load_btn = QPushButton("选择图像…")
        self.load_btn.clicked.connect(self.load_image)
        step1_layout.addWidget(self.load_btn)

        self.image_info = QLabel("尚未加载图像")
        self.image_info.setWordWrap(True)
        self.image_info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.image_info.setStyleSheet("QLabel { color: #888; }")
        step1_layout.addWidget(self.image_info)

        step1_group.setLayout(step1_layout)
        right_panel.addWidget(step1_group)

        # Step 2: YOLO Detection
        step2_group = QGroupBox("02  候选目标")
        step2_layout = QVBoxLayout()

        self.detector_profile = QComboBox()
        self.detector_profile.addItem("气球专用YOLO · 与任务一相同权重", "balloon")
        self.detector_profile.addItem("通用YOLO · 演示模型", "general")
        self.detector_profile.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.detector_profile.setMinimumContentsLength(16)
        step2_layout.addWidget(self.detector_profile)
        self.detect_btn = QPushButton("运行 YOLO 检测")
        self.detect_btn.clicked.connect(self.run_yolo)
        self.detect_btn.setEnabled(False)
        step2_layout.addWidget(self.detect_btn)

        self.yolo_info = QLabel("可拖动鼠标画框；当前云端请求只发送图像与文字，框仅供本地参考")
        self.yolo_info.setWordWrap(True)
        self.yolo_info.setStyleSheet("QLabel { color: #888; }")
        step2_layout.addWidget(self.yolo_info)

        self.request_note = QLabel("云端定位使用图像＋文字；候选框供本地参考。")
        self.request_note.setWordWrap(True)
        self.request_note.setStyleSheet("QLabel { color: #617681; font-size: 12px; }")
        step2_layout.addWidget(self.request_note)

        self.candidate_list = QListWidget()
        self.candidate_list.setMinimumHeight(100)
        step2_layout.addWidget(self.candidate_list)
        box_actions = QHBoxLayout()
        self.remove_btn = QPushButton("移除选中")
        self.remove_btn.clicked.connect(self.remove_box)
        self.clear_boxes_btn = QPushButton("清空候选框")
        self.clear_boxes_btn.clicked.connect(self.clear_boxes)
        box_actions.addWidget(self.remove_btn)
        box_actions.addWidget(self.clear_boxes_btn)
        step2_layout.addLayout(box_actions)
        self.candidate_list.currentRowChanged.connect(self.refresh_controls)

        step2_group.setLayout(step2_layout)
        right_panel.addWidget(step2_group)

        # Step 3: VLX Grounding
        step3_group = QGroupBox("03  语言定位")
        step3_layout = QVBoxLayout()

        desc_label = QLabel("描述要找的目标：")
        step3_layout.addWidget(desc_label)

        self.description_input = QLineEdit()
        self.description_input.setPlaceholderText("例如：带白色横条的红色气球")
        self.description_input.setText("带白色横条的红色气球")
        self.description_input.returnPressed.connect(self.run_vlx)
        self.description_input.textChanged.connect(self.refresh_controls)
        self.description_input.setStyleSheet("QLineEdit { padding: 8px; font-size: 13px; }")
        step3_layout.addWidget(self.description_input)

        # Quick examples
        examples_layout = QHBoxLayout()
        ex1 = QPushButton("指定横条气球")
        ex1.clicked.connect(lambda: self.set_description("带白色横条的红色气球"))
        ex2 = QPushButton("最右侧")
        ex2.clicked.connect(lambda: self.set_description("最右侧的红色气球"))
        examples_layout.addWidget(ex1)
        examples_layout.addWidget(ex2)
        self.example_buttons = [ex1, ex2]
        step3_layout.addLayout(examples_layout)

        self.vlx_btn = QPushButton("运行 VLX-Seek")
        self.vlx_btn.clicked.connect(self.run_vlx)
        self.vlx_btn.setEnabled(False)
        self.vlx_btn.setStyleSheet("QPushButton { padding: 10px; font-size: 14px; "
                                   "background-color: #0066cc; color: white; }")
        step3_layout.addWidget(self.vlx_btn)

        self.vlx_info = QLabel("等待定位")
        self.vlx_info.setWordWrap(True)
        self.vlx_info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.vlx_info.setStyleSheet("QLabel { color: #888; }")
        step3_layout.addWidget(self.vlx_info)

        step3_group.setLayout(step3_layout)
        right_panel.addWidget(step3_group)

        # Results panel
        results_group = QGroupBox("运行记录")
        results_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        self.results_text.setStyleSheet("QTextEdit { font-family: 'Consolas'; font-size: 11px; }")
        results_layout.addWidget(self.results_text)

        self.clear_results_btn = QPushButton("清除记录与结果")
        self.clear_results_btn.clicked.connect(self.clear_results)
        results_layout.addWidget(self.clear_results_btn)

        results_group.setLayout(results_layout)
        right_panel.addWidget(results_group)

        right_panel.addStretch()

        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)

        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        splitter.addWidget(left_widget)
        scroll = self.image_controls_scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(340)
        scroll.setWidget(right_widget)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([900, 420])

        main_layout.addWidget(splitter)
        from t1_qt_dashboard import ExperimentDashboard
        self.research = ExperimentDashboard(ROOT)
        from t1_manual_workbench import ManualFlightWorkbench
        self.manual = ManualFlightWorkbench()
        self.tabs.addTab(self.research, "任务一 · 实验回放")
        self.tabs.addTab(image_page, "YOLO＋VLX · 单图验证")
        self.tabs.addTab(self.manual, "控制闭环 · 人工操作")
        self.tabs.setCurrentWidget(self.manual)

        # Status bar
        self.statusBar().showMessage("Ready")

    def init_models(self):
        """Set up the lightweight API client; YOLO loads on demand."""
        try:
            load_dotenv(ROOT / ".env")
            self.vlx = VLXSeekAPIClient(model="vlx-seek-1.5-3b",
                                        timeout=30, max_retries=1)
            self.statusBar().showMessage("API配置已读取 · 实验记录可直接回放")
        except Exception as e:
            self.vlx = None
            self.statusBar().showMessage("VLX API 未配置；仍可加载图像并整理候选框")
            self.log_result(f"VLX API 未配置：{e}")
        self.refresh_controls()

    def refresh_controls(self, *_):
        has_image = self.current_image is not None
        self.load_btn.setEnabled(not self.busy)
        self.detect_btn.setEnabled(has_image and not self.busy)
        self.detector_profile.setEnabled(not self.busy)
        self.clear_results_btn.setEnabled(not self.busy)
        self.remove_btn.setEnabled(bool(self.yolo_boxes) and
                                   self.candidate_list.currentRow() >= 0 and not self.busy)
        self.clear_boxes_btn.setEnabled(bool(self.yolo_boxes) and not self.busy)
        self.vlx_btn.setEnabled(bool(self.vlx and self.yolo_boxes and
                                     self.description_input.text().strip() and not self.busy))
        self.description_input.setEnabled(not self.busy)
        for button in self.example_buttons:
            button.setEnabled(not self.busy)

    def set_busy(self, busy, message):
        self.busy = busy
        self.progress.setVisible(busy)
        self.statusBar().showMessage(message)
        self.refresh_controls()

    def sync_boxes(self):
        self.candidate_list.blockSignals(True)
        self.candidate_list.clear()
        for index, box in enumerate(self.yolo_boxes, 1):
            score = box.get("confidence")
            suffix = f" · 检测分数 {score:.2f}" if score is not None else ""
            self.candidate_list.addItem(
                f"{index:02d}  {box['class']} · {box.get('source', 'YOLO')}{suffix}")
        if self.yolo_boxes:
            self.candidate_list.setCurrentRow(0)
        self.candidate_list.blockSignals(False)
        self.canvas.set_yolo_boxes(self.yolo_boxes)
        self.canvas.clear_vlx()
        self.vlx_info.setText("候选框已更新，等待重新定位")
        self.refresh_controls()

    def add_manual_box(self, bbox):
        if self.busy:
            return
        self.yolo_boxes.append({"bbox": bbox, "class": "候选目标",
                                "confidence": None, "source": "手动"})
        self.sync_boxes()
        self.candidate_list.setCurrentRow(len(self.yolo_boxes) - 1)
        self.statusBar().showMessage(f"已添加候选框 {len(self.yolo_boxes)}")

    def remove_box(self):
        index = self.candidate_list.currentRow()
        if 0 <= index < len(self.yolo_boxes):
            self.yolo_boxes.pop(index)
            self.sync_boxes()

    def clear_boxes(self):
        self.yolo_boxes = []
        self.sync_boxes()

    def load_image(self):
        """Load image from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", str(ROOT / "logs"),
            "图像 (*.png *.jpg *.jpeg *.bmp)")

        if not file_path:
            return
        self.load_image_file(file_path)

    def load_image_file(self, file_path):
        """Load the same local image path used by the file dialog."""
        if self.busy:
            return False
        try:
            image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Cannot load image")

            self.current_image = image
            self.yolo_boxes = []

            self.canvas.set_image(image)
            self.sync_boxes()

            self.image_info.setText(f"{Path(file_path).name} · "
                                   f"{image.shape[1]}×{image.shape[0]} · 拖动鼠标添加候选框")
            self.image_info.setToolTip(str(Path(file_path).resolve()))

            self.statusBar().showMessage("图像已加载")
            self.log_result(f"已加载：{Path(file_path).name}")
            return True

        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")
            self.log_result(f"Error loading image:\n{e}")
            QMessageBox.warning(self, "图像加载失败", str(e))
            return False

    def run_yolo(self):
        """Run YOLO detection."""
        if self.current_image is None or self.busy:
            return
        if getattr(self, "detection_thread", None) is not None and self.detection_thread.isRunning():
            return

        self.set_busy(True, "正在运行 YOLO…")

        self.detection_thread = DetectionThread(self.current_image,
                                                self.detector_profile.currentData())
        self.detection_thread.finished.connect(self.on_yolo_finished)
        self.detection_thread.error.connect(self.on_yolo_error)
        self.detection_thread.start()

    def on_yolo_finished(self, boxes):
        """Handle YOLO detection completion."""
        manual = [box for box in self.yolo_boxes if box.get("source") == "手动"]
        self.yolo_boxes = manual + boxes
        self.sync_boxes()

        self.yolo_info.setText(f"YOLO 找到 {len(boxes)} 个目标 · 手动框 {len(manual)} 个")
        self.set_busy(False, f"YOLO 找到 {len(boxes)} 个目标")

        log_text = f"YOLO Detection:\n{len(boxes)} objects found\n"
        for i, box in enumerate(boxes):
            log_text += f"  [{i+1}] {box['class']} (conf: {box['confidence']:.2f})\n"
        self.log_result(log_text)

    def on_yolo_error(self, error_msg):
        """Handle YOLO detection error."""
        self.set_busy(False, "YOLO 运行失败")
        self.log_result(f"YOLO error:\n{error_msg}")
        QMessageBox.warning(self, "YOLO 运行失败", error_msg)

    def set_description(self, text):
        """Set description from quick example."""
        if not self.busy:
            self.description_input.setText(text)

    def run_vlx(self):
        """Run VLX grounding."""
        if self.busy or self.vlx is None or not self.yolo_boxes or self.current_image is None:
            return
        if getattr(self, "vlx_thread", None) is not None and self.vlx_thread.isRunning():
            return

        description = self.description_input.text().strip()
        if not description:
            self.statusBar().showMessage("Please enter a description")
            return

        if len(self.yolo_boxes) > 100:
            QMessageBox.warning(self, "候选框过多", "请将本地参考框减少至100个以内再运行。")
            return
        self.canvas.clear_vlx()
        self.vlx_info.setText("正在定位…")
        self.set_busy(True, "正在调用 VLX-Seek…")

        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))
        boxes = [box["bbox"] for box in self.yolo_boxes]

        self.vlx_thread = VLXThread(self.vlx, pil_image, boxes, description)
        self.vlx_thread.finished.connect(self.on_vlx_finished)
        self.vlx_thread.error.connect(self.on_vlx_error)
        self.vlx_thread.start()

    def on_vlx_finished(self, result):
        """Handle VLX completion."""
        bbox_list = result.get('result_bbox_list', [])

        if bbox_list:
            selected = bbox_list[0]
            self.canvas.set_vlx_box(selected)

            label = selected.get('label', 'target')
            elapsed = result.get("gui_call_wall_s")
            suffix = f" · {elapsed:.2f}秒" if isinstance(elapsed, (int, float)) else ""
            self.vlx_info.setText(f"已返回定位框：{label}{suffix}")

            log_text = f"VLX Selection:\n"
            log_text += f"Description: {result.get('gui_request_description', self.description_input.text())}\n"
            log_text += f"Selected: {label}\n"
            log_text += f"Box: [{selected['xmin']:.0f}, {selected['ymin']:.0f}, "
            log_text += f"{selected['xmax']:.0f}, {selected['ymax']:.0f}]\n"
            log_text += f"Answer: {result.get('answer', '')}"
            self.log_result(log_text)

            self.set_busy(False, "定位完成")
        else:
            self.vlx_info.setText("未返回定位框，请检查目标是否可见或调整文字描述")
            self.set_busy(False, "未找到匹配目标")

    def on_vlx_error(self, error_msg):
        """Handle VLX error."""
        self.vlx_info.setText("定位失败，请检查 API 配置或网络")
        self.set_busy(False, "VLX 调用失败")
        self.log_result(f"VLX error:\n{error_msg}")
        QMessageBox.warning(self, "VLX 调用失败", error_msg)

    def log_result(self, text):
        """Log message to results panel."""
        self.results_text.append("=" * 40)
        self.results_text.append(text)
        self.results_text.append("")

    def clear_results(self):
        """Clear results and VLX selection."""
        if self.busy:
            return
        self.results_text.clear()
        self.canvas.clear_vlx()
        self.vlx_info.setText("等待定位")

    def closeEvent(self, event):
        running = any(thread.isRunning() for thread in (
            getattr(self, "detection_thread", None),
            getattr(self, "vlx_thread", None),
        ) if thread is not None)
        if running or self.research.has_running_job() or self.manual.has_running_job():
            QMessageBox.information(self, "任务正在运行", "请等待当前任务完成后关闭窗口。")
            event.ignore()
        else:
            event.accept()


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    window = VLXSeekGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
