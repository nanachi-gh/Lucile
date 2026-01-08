import sys
import os
import numpy as np
import cv2
from PIL import Image
from PySide6 import QtCore, QtWidgets, QtGui

from Lucile_bigshaq9999.MangaTypesetter import MangaTypesetter
from Lucile_bigshaq9999.BubbleSegmenter import BubbleSegmenter


# --- WORKERS ---
class BackgroundLoader(QtCore.QObject):
    """
    Loads OCR and Translator sequentially in the background.
    """

    finished = QtCore.Signal(object, object)  # Emits (ocr_model, translator_model)
    progress = QtCore.Signal(str)

    def run(self):
        ocr_model = None
        translator_model = None

        try:
            self.progress.emit("Pre-loading OCR Model...")
            from Lucile_bigshaq9999.MangaOCRModel import MangaOCRModel

            ocr_model = MangaOCRModel()
            ocr_model.load_model()

            # Warming up with dummy input
            dummy = Image.new("RGB", (50, 50), "white")
            ocr_model.predict(dummy, [[0, 0, 50, 50]])

            self.progress.emit("Pre-loading Translator...")
            from Lucile_bigshaq9999.ElanMtJaEnTranslator import ElanMtJaEnTranslator

            translator_model = ElanMtJaEnTranslator()
            # Default to base for balance, or make configurable
            translator_model.load_model(device="auto", elan_model="bt")
            # TODO make this changeable

            self.progress.emit("Background models ready.")
            self.finished.emit(ocr_model, translator_model)

        except Exception as e:
            print(f"Background load failed: {e}")
            # Even if failed, emit what we have (None) so app doesn't hang
            self.finished.emit(ocr_model, translator_model)


class ZoomableGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        # High quality rendering
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        # Allow panning with left mouse click & drag
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        # Show scrollbars only when zoomed in (needed for standard scrolling)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        # Zooming will center on the mouse cursor
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event):
        """
        Ctrl + Scroll to Zoom
        Standard Scroll to Pan vertically
        """
        # Check if the Control key is currently pressed
        if event.modifiers() & QtCore.Qt.ControlModifier:
            # --- ZOOM LOGIC ---
            zoom_in_factor = 1.15
            zoom_out_factor = 1 / zoom_in_factor

            # angleDelta().y() > 0 means scrolling UP (Zoom In)
            if event.angleDelta().y() > 0:
                self.scale(zoom_in_factor, zoom_in_factor)
            else:
                self.scale(zoom_out_factor, zoom_out_factor)

            # Important: Accept the event so it doesn't propagate to the parent
            # or trigger the default scroll behavior simultaneously.
            event.accept()

        else:
            # --- STANDARD SCROLL LOGIC ---
            # Pass the event to the parent class to handle normal scrolling
            super().wheelEvent(event)


class ProcessingPipeline(QtCore.QObject):
    """
    Runs the actual image processing:
    Seg -> OCR -> Translate -> Typeset
    """

    status_update = QtCore.Signal(int, str)  # progress %, message
    finished = QtCore.Signal(object)  # Final PIL Image
    error = QtCore.Signal(str)

    def __init__(self, image_path, yolo_model_name, ocr_model, translator_model):
        super().__init__()
        self.image_path = image_path
        self.yolo_model_name = yolo_model_name
        self.ocr_model = ocr_model
        self.translator_model = translator_model

    def _polygon_to_mask(self, polygon, width, height):
        # Helper specific to typesetting preparation
        mask = np.zeros((height, width), dtype=np.uint8)
        points = []
        for point in polygon:
            points.append([int(point.x()), int(point.y())])

        if not points:
            return mask
        pts = np.array([points], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
        return mask

    def run(self):
        try:
            # --- STEP 1: SEGMENTATION ---
            self.status_update.emit(10, "Downloading/Loading YOLO...")

            # Lazy import huggingface
            from huggingface_hub import hf_hub_download

            # Download/Locate Model
            file_path = hf_hub_download(
                repo_id=f"TheBlindMaster/{self.yolo_model_name}-manga-bubble-seg",
                filename="best.pt",
            )

            self.status_update.emit(20, "Running Segmentation...")
            segmenter = BubbleSegmenter(file_path)

            # Run Inference
            image_rgb, _, refined_bubbles = segmenter.detect_and_segment(
                self.image_path
            )

            if not refined_bubbles:
                raise ValueError("No bubbles found in image.")

            # --- STEP 2: PREPARE DATA ---
            self.status_update.emit(40, "Preparing regions...")
            pil_image = Image.fromarray(image_rgb)
            bboxes = []
            valid_bubbles_data = []  # Store logic data needed for next steps

            for b in refined_bubbles:
                x, y, w, h = b["bbox"]
                bboxes.append([x, y, x + w, y + h])

                # Convert contour to QPolygonF-like list for typesetter helper
                poly_points = [
                    QtCore.QPointF(pt[0][0], pt[0][1]) for pt in b["contour"]
                ]

                valid_bubbles_data.append({
                    "polygon": poly_points,
                    "original_mask": b["original_mask"],
                })

            # --- STEP 3: OCR ---
            self.status_update.emit(50, "Running OCR...")
            if not self.ocr_model:
                raise RuntimeError("OCR Model not loaded yet.")

            ocr_texts = self.ocr_model.predict(pil_image, bboxes)

            # --- STEP 4: TRANSLATION ---
            self.status_update.emit(70, "Translating Text...")
            if not self.translator_model:
                raise RuntimeError("Translator Model not loaded yet.")

            translated_texts = self.translator_model.predict(ocr_texts)

            # --- STEP 5: TYPESETTING ---
            self.status_update.emit(90, "Typesetting Final Image...")

            height, width, _ = image_rgb.shape
            typesetter_data = []

            for i, data in enumerate(valid_bubbles_data):
                # Rasterize polygon for typesetting
                mask = self._polygon_to_mask(data["polygon"], width, height)

                typesetter_data.append({
                    "translated_text": translated_texts[i],
                    "mask": mask,
                    "original_mask": data["original_mask"],
                })

            ts = MangaTypesetter()
            final_np = ts.render(image_rgb, typesetter_data)
            final_pil = Image.fromarray(final_np)

            self.status_update.emit(100, "Done!")
            self.finished.emit(final_pil)

        except Exception as e:
            self.error.emit(str(e))


# --- PAGES (UI) ---


class SetupPage(QtWidgets.QWidget):
    start_processing = QtCore.Signal(str, str)  # image_path, model_name

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        # Title
        title = QtWidgets.QLabel("Lucile")
        title.setFont(QtGui.QFont("Arial", 20, QtGui.QFont.Bold))
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        # Model Selector
        form_layout = QtWidgets.QFormLayout()
        self.modelCombo = QtWidgets.QComboBox()
        self.modelCombo.addItems(["yolov8s", "yolov8n", "yolov11s", "yolov11n"])
        form_layout.addRow("Segmentation Model:", self.modelCombo)
        layout.addLayout(form_layout)

        # Image Selection
        self.imgBtn = QtWidgets.QPushButton("Select Image")
        self.imgBtn.clicked.connect(self.select_image)
        layout.addWidget(self.imgBtn)

        self.imgLabel = QtWidgets.QLabel("No image selected")
        self.imgLabel.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.imgLabel)

        # Start Button
        self.nextBtn = QtWidgets.QPushButton("Start Processing")
        self.nextBtn.clicked.connect(self.on_next)
        self.nextBtn.setEnabled(False)
        layout.addWidget(self.nextBtn)

        # Status of background loading
        self.statusLabel = QtWidgets.QLabel("Initializing AI models...")
        self.statusLabel.setStyleSheet("color: gray; font-style: italic;")
        self.statusLabel.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.statusLabel)

        self.selected_image_path = None

    def select_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if path:
            self.selected_image_path = path
            self.imgLabel.setText(os.path.basename(path))
            self.nextBtn.setEnabled(True)

    def on_next(self):
        if self.selected_image_path:
            self.start_processing.emit(
                self.selected_image_path, self.modelCombo.currentText()
            )

    def update_bg_status(self, msg):
        self.statusLabel.setText(msg)


class ProgressPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        self.infoLabel = QtWidgets.QLabel("Processing...")
        self.infoLabel.setFont(QtGui.QFont("Arial", 14))
        layout.addWidget(self.infoLabel)

        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.setFixedWidth(400)
        layout.addWidget(self.progressBar)

        self.logLabel = QtWidgets.QLabel("Starting pipeline...")
        layout.addWidget(self.logLabel)

    def update_progress(self, val, msg):
        self.progressBar.setValue(val)
        self.logLabel.setText(msg)


class ResultPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)

        # --- CHANGED LINE BELOW ---
        # Use the custom class instead of the standard QGraphicsView
        self.view = ZoomableGraphicsView()

        self.scene = QtWidgets.QGraphicsScene()
        self.view.setScene(self.scene)

        layout.addWidget(self.view)

        # Bottom buttons
        btnLayout = QtWidgets.QHBoxLayout()
        self.saveBtn = QtWidgets.QPushButton("Save Image")
        self.saveBtn.clicked.connect(self.save_image)
        self.backBtn = QtWidgets.QPushButton("Process Another")

        btnLayout.addWidget(self.backBtn)
        btnLayout.addWidget(self.saveBtn)
        layout.addLayout(btnLayout)

        self.final_pil = None

    def display_image(self, pil_img):
        self.final_pil = pil_img
        self.scene.clear()

        # Convert PIL to QPixmap
        data = pil_img.convert("RGBA").tobytes("raw", "RGBA")
        qim = QtGui.QImage(
            data, pil_img.width, pil_img.height, QtGui.QImage.Format_RGBA8888
        )
        pixmap = QtGui.QPixmap.fromImage(qim)

        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def save_image(self):
        if not self.final_pil:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save", "output.png", "Images (*.png *.jpg)"
        )
        if path:
            self.final_pil.save(path)
            QtWidgets.QMessageBox.information(
                self, "Saved", "Image saved successfully."
            )


# --- MAIN CONTROLLER ---


class MainWizard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lucile")
        self.resize(500, 400)  # Small start window

        # State
        self.ocr_model = None
        self.translator_model = None
        self.are_models_ready = False
        self.pending_pipeline_request = (
            None  # If user clicks next before models are ready
        )

        # Stack Setup
        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self.setupPage = SetupPage()
        self.progressPage = ProgressPage()
        self.resultPage = ResultPage()

        self.stack.addWidget(self.setupPage)
        self.stack.addWidget(self.progressPage)
        self.stack.addWidget(self.resultPage)

        # Signals
        self.setupPage.start_processing.connect(self.initiate_processing)
        self.resultPage.backBtn.clicked.connect(self.reset_ui)

        # Start Background Loading Immediately
        self.start_background_loading()

    def start_background_loading(self):
        self.bg_thread = QtCore.QThread()
        self.bg_worker = BackgroundLoader()
        self.bg_worker.moveToThread(self.bg_thread)

        self.bg_thread.started.connect(self.bg_worker.run)
        self.bg_worker.progress.connect(self.setupPage.update_bg_status)
        self.bg_worker.finished.connect(self.on_models_loaded)

        # Cleanup
        self.bg_worker.finished.connect(self.bg_thread.quit)
        self.bg_worker.finished.connect(self.bg_worker.deleteLater)
        self.bg_thread.finished.connect(self.bg_thread.deleteLater)

        self.bg_thread.start()

    def on_models_loaded(self, ocr, trans):
        self.ocr_model = ocr
        self.translator_model = trans
        self.are_models_ready = True

        # If user already clicked next, execute now
        if self.pending_pipeline_request:
            self.run_pipeline(*self.pending_pipeline_request)
        else:
            self.setupPage.update_bg_status("Models Ready. Select image to begin.")

    def initiate_processing(self, img_path, model_name):
        # Switch to progress immediately
        self.stack.setCurrentWidget(self.progressPage)

        if self.are_models_ready:
            self.run_pipeline(img_path, model_name)
        else:
            self.progressPage.update_progress(
                0, "Waiting for AI models to finish loading..."
            )
            self.pending_pipeline_request = (img_path, model_name)

    def run_pipeline(self, img_path, model_name):
        self.pipe_thread = QtCore.QThread()
        self.pipe_worker = ProcessingPipeline(
            img_path, model_name, self.ocr_model, self.translator_model
        )
        self.pipe_worker.moveToThread(self.pipe_thread)

        self.pipe_thread.started.connect(self.pipe_worker.run)
        self.pipe_worker.status_update.connect(self.progressPage.update_progress)
        self.pipe_worker.finished.connect(self.on_pipeline_finished)
        self.pipe_worker.error.connect(self.on_pipeline_error)

        # Cleanup
        self.pipe_worker.finished.connect(self.pipe_thread.quit)
        self.pipe_worker.finished.connect(self.pipe_worker.deleteLater)
        self.pipe_thread.finished.connect(self.pipe_thread.deleteLater)

        self.pipe_thread.start()

    def on_pipeline_finished(self, final_pil):
        # Resize window for the result
        self.resize(1000, 800)
        self.stack.setCurrentWidget(self.resultPage)
        self.resultPage.display_image(final_pil)

    def on_pipeline_error(self, err):
        QtWidgets.QMessageBox.critical(self, "Error", f"Processing Failed:\n{err}")
        self.stack.setCurrentWidget(self.setupPage)

    def reset_ui(self):
        self.resize(500, 400)
        self.stack.setCurrentWidget(self.setupPage)
        self.setupPage.statusLabel.setText("Models Ready.")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWizard()
    window.show()
    sys.exit(app.exec())
