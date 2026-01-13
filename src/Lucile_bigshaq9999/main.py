import os
import sys

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

from Lucile_bigshaq9999.BubbleSegmenter import BubbleSegmenter
from Lucile_bigshaq9999.ElanMtJaEnTranslator import ElanMtJaEnTranslator
from Lucile_bigshaq9999.MangaOCRModel import MangaOCRModel
from Lucile_bigshaq9999.MangaTypesetter import MangaTypesetter


class BubbleData:
    def __init__(self, polygon, bbox, raw_mask, ocr_text="", trans_text=""):
        self.polygon = polygon
        self.bbox = bbox
        self.raw_mask = raw_mask
        self.ocr_text = ocr_text
        self.trans_text = trans_text


class ImageProject:
    """Represents one file in the batch queue"""

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.status = "queued"
        self.bubbles = []
        self.thumbnail = None


# --- Workers --- #
class ModelManager(QtCore.QObject):
    """Holds the loaded models and handles loading them in background"""

    models_ready = QtCore.Signal()
    status_update = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.seg_model = None
        self.ocr_model = None
        self.trans_model = None
        self.is_ready = False

    def load_all(self):
        """Run in a thread"""
        try:
            # Segmentation (default: yolov8a)
            self.status_update.emit("Loading YOLO...")
            model_name = "yolov8s"
            path = hf_hub_download(
                repo_id=f"TheBlindMaster/{model_name}-manga-bubble-seg",
                filename="best.pt",
            )
            self.seg_model = BubbleSegmenter(path)

            # OCR
            self.status_update.emit("Loading OCR...")
            self.ocr_model = MangaOCRModel()
            self.ocr_model.load_model()

            # Warm-up dummy
            self.ocr_model.predict(Image.new("RGB", (50, 50)), [[0, 0, 50, 50]])

            # Translate
            self.status_update.emit("Loading translator...")
            self.trans_model = ElanMtJaEnTranslator()
            self.trans_model.load_model(device="auto", elan_model="base")

            self.is_ready = True
            self.status_update.emit("Ready")
            self.models_ready.emit()

        except Exception as e:
            self.status_update.emit(f"Load error: {e}")


class BatchProcessor(QtCore.QObject):
    """
    Watches the queue and processes images using the ModelManager models
    """

    image_processed = QtCore.Signal(str)
    progress = QtCore.Signal(str, str)

    def __init__(self, model_manager):
        super().__init__()
        self.models = model_manager
        self.queue = []
        self.is_running = False

    def add_to_queue(self, project: ImageProject):
        self.queue.append(project)
        self.process_next()

    def process_next(self):
        if self.is_running or not self.queue:
            return

        if not self.models.is_ready:
            return

        self.is_running = True
        project = self.queue.pop(0)
        project.status = "processing"
        self.progress.emit(project.path, "Processing...")

        self._run_inference(project)

    def _run_inference(self, project):
        try:
            img_rgb, _, refined = self.models.seg_model.detect_and_segment(project.path)

            if not refined:
                project.status = "ready (no bubbles)"
            else:
                pil_img = Image.fromarray(img_rgb)
                bboxes = [b["bbox"] for b in refined]
                xyxy = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in bboxes]

                texts = self.models.ocr_model.predict(pil_img, xyxy)

                # Translate
                trans = self.models.trans_model.predict(texts)

                # Store
                project.bubbles = []
                for i, r in enumerate(refined):
                    poly = QtGui.QPolygonF()
                    for pt in r["contour"]:
                        poly.append(QtCore.QPointF(pt[0][0], pt[0][1]))

                    bubble = BubbleData(
                        polygon=poly,
                        bbox=xyxy[i],
                        raw_mask=r["original_mask"],
                        ocr_text=texts[i],
                        trans_text=trans[i],
                    )
                    project.bubbles.append(bubble)

                project.status = "ready"

            self.image_processed.emit(project.path)

        except Exception as e:
            print(f"Error processing {project.name}: {e}")
            project.status = "error"
            self.progress.emit(project.path, "Error")
            self.image_processed.emit(project.path)

        self.is_running = False
        self.process_next()


# --- UI Widgets --- #
class FileQueueList(QtWidgets.QListWidget):
    """Handles drag & drop of files"""

    files_dropped = QtCore.Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DropOnly)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                paths.append(path)

        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()


class ZoomableGraphicsView(QtWidgets.QGraphicsView):
    """View with zoom and pan"""

    def __init__(self):
        super().__init__()
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(50, 50, 50)))

    def wheelEvent(self, event):
        if event.modifiers() & QtCore.Qt.ControlModifier:
            zoom_in = 1.15
            if event.angleDelta().y() > 0:
                self.scale(zoom_in, zoom_in)
            else:
                self.scale(1 / zoom_in, 1 / zoom_in)
            event.accept()
        else:
            super().wheelEvent(event)


# --- panes --- #
class EditorPanel(QtWidgets.QWidget):
    """Right pane: OCR and Translation editing"""

    text_changed = QtCore.Signal(str)
    save_requested = QtCore.Signal()

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(QtWidgets.QLabel("<b>Original text (OCR)</b>"))
        self.ocr_edit = QtWidgets.QTextEdit()
        self.ocr_edit.setMaximumHeight(100)
        layout.addWidget(self.ocr_edit)

        layout.addSpacing(10)

        layout.addWidget(QtWidgets.QLabel("<b>Translation</b>"))
        self.trans_edit = QtWidgets.QTextEdit()
        self.trans_edit.textChanged.connect(self.on_text_changed)
        layout.addWidget(self.trans_edit)

        layout.addStretch()

        self.save_button = QtWidgets.QPushButton("Save / export image")
        self.save_button.clicked.connect(self.save_requested.emit)
        layout.addWidget(self.save_button)

        self.current_bubble = None
        self.is_internal_update = False

    def load_bubble(self, bubble: BubbleData):
        self.is_internal_update = True
        self.current_bubble = bubble
        self.ocr_edit.setText(bubble.ocr_text)
        self.trans_edit.setText(bubble.trans_text)
        self.is_internal_update = False

    def on_text_changed(self):
        if self.current_bubble and not self.is_internal_update:
            new_text = self.trans_edit.toPlainText()
            self.current_bubble.trans_text = new_text
            self.text_changed.emit(new_text)

    def clear_fields(self):
        self.is_internal_update = True
        self.ocr_edit.clear()
        self.trans_edit.clear()
        self.current_bubble = None
        self.is_internal_update = False


class MainWindow(QtWidgets.QMainWindow):
    process_signal = QtCore.Signal(ImageProject)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lucile")
        self.resize(1200, 800)

        self.projects = {}
        self.current_project = None
        self.active_visuals = {}

        # Loading Dialog
        self.loading_modal = QtWidgets.QProgressDialog()
        self.loading_modal.setLabelText("Loading AI models...")
        self.loading_modal.setCancelButton(None)
        self.loading_modal.setRange(0, 0)
        self.loading_modal.setWindowModality(QtCore.Qt.ApplicationModal)
        self.loading_modal.setWindowFlags(
            QtCore.Qt.Dialog | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowTitleHint
        )
        self.loading_modal.show()

        # Model manager
        self.model_thread = QtCore.QThread()
        self.model_manager = ModelManager()
        self.model_manager.moveToThread(self.model_thread)
        self.model_thread.started.connect(self.model_manager.load_all)
        self.model_manager.models_ready.connect(self.loading_modal.accept)
        self.model_thread.start()

        self.proc_thread = QtCore.QThread()
        self.processor = BatchProcessor(self.model_manager)
        self.processor.moveToThread(self.proc_thread)
        self.proc_thread.start()

        # Signal
        self.model_manager.status_update.connect(self.update_global_status)
        self.model_manager.models_ready.connect(self.on_models_ready)

        self.processor.image_processed.connect(self.on_image_processed)
        self.processor.progress.connect(self.update_file_status)

        self.process_signal.connect(self.processor.add_to_queue)

        self.setup_ui()

    def setup_ui(self):
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(self.splitter)

        # left pane
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.file_list = FileQueueList()
        self.file_list.files_dropped.connect(self.add_files)
        self.file_list.itemClicked.connect(self.load_project_into_view)

        left_layout.addWidget(QtWidgets.QLabel("<b>File queue</b>"))
        left_layout.addWidget(self.file_list)

        self.canvas_view = ZoomableGraphicsView()
        self.scene = QtWidgets.QGraphicsScene()
        self.canvas_view.setScene(self.scene)

        self.editor = EditorPanel()
        self.editor.text_changed.connect(self.update_bubble_preview)
        self.editor.save_requested.connect(self.save_current)

        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(self.canvas_view)
        self.splitter.addWidget(self.editor)

        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([200, 750, 300])

        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QtWidgets.QLabel("Initializing...")
        self.status_bar.addWidget(self.status_label)
        self.scene.selectionChanged.connect(self.on_selection_changed)

    def closeEvent(self, event):
        if self.model_thread.isRunning():
            self.model_thread.quit()
            self.model_thread.wait()

        if self.proc_thread.isRunning():
            self.proc_thread.quit()
            self.proc_thread.wait()

    # --- logic --- #
    def update_global_status(self, msg):
        self.status_label.setText(msg)

    def on_models_ready(self):
        self.status_label.setText("Models ready.")
        self.processor.process_next()

    def add_files(self, paths):
        for path in paths:
            if path in self.projects:
                continue

            proj = ImageProject(path)
            self.projects[path] = proj
            self.process_signal.emit(proj)

            item = QtWidgets.QListWidgetItem(proj.name)
            item.setData(QtCore.Qt.UserRole, path)
            item.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))
            self.file_list.addItem(item)

            item.setText(f"{proj.name} (Queued)")

    def update_file_status(self, path, status_msg):
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(QtCore.Qt.UserRole) == path:
                item.setText(f"{os.path.basename(path)} ({status_msg})")
                if status_msg == "Error":
                    item.setForeground(QtGui.QBrush(QtCore.Qt.red))
                break

    def on_image_processed(self, path):
        self.projects[path].status = "ready"
        self.update_file_status(path, "Ready")

        # if currently viewing this project, refresh view
        if self.current_project and self.current_project.path == path:
            self.load_project_into_view(self.file_list.currentItem())

    def load_project_into_view(self, item):
        path = item.data(QtCore.Qt.UserRole)
        project = self.projects[path]
        self.current_project = project
        self.scene.clear()
        self.active_visuals.clear()
        self.editor.clear_fields()

        if not os.path.exists(path):
            return
        pix = QtGui.QPixmap(path)
        self.scene.addPixmap(pix)
        self.scene.setSceneRect(QtCore.QRectF(pix.rect()))
        self.canvas_view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

        if project.status == "ready":
            for bubble in project.bubbles:
                box_item = QtWidgets.QGraphicsPolygonItem(bubble.polygon)
                box_item.setPen(QtGui.QPen(QtCore.Qt.green, 3))
                box_item.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 0, 20)))
                box_item.setFlags(QtWidgets.QGraphicsItem.ItemIsSelectable)

                box_item.setData(0, bubble)
                self.scene.addItem(box_item)

                bubble.graphics_item = box_item

                visuals = self.create_text_preview(bubble)
                self.active_visuals[bubble] = visuals

    def create_text_preview(self, bubble: BubbleData):
        text_content = bubble.trans_text if bubble.trans_text else ""
        t_item = QtWidgets.QGraphicsTextItem(text_content)
        t_item.setDefaultTextColor(QtCore.Qt.black)

        font = QtGui.QFont("Comic Sans MS", 12, QtGui.QFont.Bold)
        t_item.setFont(font)

        x1, y1, x2, y2 = bubble.bbox
        t_item.setPos(x1, y1)

        bg = QtWidgets.QGraphicsRectItem(t_item.boundingRect())
        bg.setBrush(QtCore.Qt.white)
        bg.setPen(QtCore.Qt.NoPen)
        bg.setPos(x1, y1)
        bg.setZValue(5)
        t_item.setZValue(6)

        self.scene.addItem(bg)
        self.scene.addItem(t_item)

        return (bg, t_item)

    def on_selection_changed(self):
        items = self.scene.selectedItems()
        if items:
            bubble = items[0].data(0)
            self.editor.load_bubble(bubble)

    @QtCore.Slot()
    def update_bubble_preview(self, new_text):
        bubble = self.editor.current_bubble

        if bubble in self.active_visuals:
            bg_item: QtWidgets.QGraphicsRectItem
            text_item: QtWidgets.QGraphicsTextItem
            bg_item, text_item = self.active_visuals[bubble]
            text_item.setPlainText(new_text)
            bg_item.setRect(text_item.boundingRect())

    @QtCore.Slot()
    def save_current(self):
        if not self.current_project or self.current_project.status != "ready":
            QtWidgets.QMessageBox.warning(self, "Wait", "Image not ready yet.")
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            ts = MangaTypesetter()
            img_np = np.array(Image.open(self.current_project.path))
            h, w, _ = img_np.shape
            fmt_bubbles = []

            def p2m(poly, w, h):
                m = np.zeros((h, w), dtype=np.uint8)
                pts = [[int(p.x()), int(p.y())] for p in poly]
                if pts:
                    cv2.fillPoly(m, np.array([pts], dtype=np.int32), 255)
                return m

            b: BubbleData
            for b in self.current_project.bubbles:
                mask = p2m(b.polygon, w, h)
                fmt_bubbles.append({
                    "translated_text": b.trans_text,
                    "mask": mask,
                    "original_mask": b.raw_mask,
                })

            final_np = ts.render(img_np, fmt_bubbles)
            final_pil = Image.fromarray(final_np)

            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save image", "", "PNG (*.png)"
            )
            if save_path:
                final_pil.save(save_path)
                QtWidgets.QMessageBox.information(
                    self, "Saved", f"Saved to  {save_path}"
                )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
