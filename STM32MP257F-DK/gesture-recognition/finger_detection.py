#!/usr/bin/env python3
import os
import time
import numpy as np
import onnxruntime as ort
import cv2

import gi
gi.require_version("Gst", "1.0")
gi.require_version("Gtk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gst, Gtk, GLib, GdkPixbuf, Gdk

import cairo

# =========================
# CONFIG
# =========================
APP_NAME = "Finger Detection - NPU PerChannel"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "gesture_model_PerChannel_quant.onnx")

INPUT_W = 128
INPUT_H = 128
CAP_W = 640
CAP_H = 480

LABELS = ["0","1","2","3","4","5"]

# =========================
# APP
# =========================
class App(Gtk.Window):
    def __init__(self):
        super().__init__(title=APP_NAME)
        self.set_default_size(CAP_W, CAP_H)

        self.image = Gtk.Image()
        self.add(self.image)
        self.connect("destroy", Gtk.main_quit)

        # =====================
        # ONNX + NPU
        # =====================
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(
            MODEL_PATH,
            sess_options=so,
            providers=["VSINPUExecutionProvider", "CPUExecutionProvider"]
        )

        print("Providers:", self.sess.get_providers())
        self.input_name = self.sess.get_inputs()[0].name

        # FPS
        self.inf_count = 0
        self.disp_count = 0
        self.t0 = time.time()
        self.fps_inf = 0
        self.fps_disp = 0

        self.last_class = 0
        self.last_conf = 0.0

        # =====================
        # GStreamer
        # =====================
        Gst.init(None)

        pipeline = (
            f"libcamerasrc ! "
            f"video/x-raw,format=BGR,width={CAP_W},height={CAP_H} ! "
            "appsink name=sink emit-signals=true max-buffers=1 drop=true"
        )

        self.pipeline = Gst.parse_launch(pipeline)
        self.sink = self.pipeline.get_by_name("sink")
        self.sink.connect("new-sample", self.on_frame)
        self.pipeline.set_state(Gst.State.PLAYING)

        self.show_all()

    def update_fps(self):
        dt = time.time() - self.t0
        if dt >= 1:
            self.fps_inf = self.inf_count / dt
            self.fps_disp = self.disp_count / dt
            self.inf_count = 0
            self.disp_count = 0
            self.t0 = time.time()

    def on_frame(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()

        w = caps.get_structure(0).get_value("width")
        h = caps.get_structure(0).get_value("height")

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK

        frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3))
        buf.unmap(mapinfo)

        rgb = frame[:, :, ::-1]

        # =====================
        # Crop central 60%
        # =====================
        box_size = int(min(h, w) * 0.6)
        x0 = (w - box_size) // 2
        y0 = (h - box_size) // 2
        crop = rgb[y0:y0+box_size, x0:x0+box_size]

        small = cv2.resize(crop, (INPUT_W, INPUT_H))

        # =====================
        # Preprocess
        # =====================
        inp = small.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, :, :, :]

        # =====================
        # Inference
        # =====================
        outputs = self.sess.run(None, {self.input_name: inp})
        preds = outputs[0][0]

        class_id = int(np.argmax(preds))
        confidence = float(np.max(preds))

        self.last_class = class_id
        self.last_conf = confidence

        self.inf_count += 1
        self.disp_count += 1
        self.update_fps()

        GLib.idle_add(self.update_image, rgb, x0, y0, box_size)

        return Gst.FlowReturn.OK

    def update_image(self, frame, x0, y0, size):
        frame = frame.copy()  # FIX read-only

        h, w, _ = frame.shape

        # Draw guide box
        cv2.rectangle(frame, (x0, y0), (x0+size, y0+size), (0,255,0), 2)

        pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            frame.tobytes(),
            GdkPixbuf.Colorspace.RGB,
            False, 8, w, h, w*3
        )

        surface = Gdk.cairo_surface_create_from_pixbuf(pixbuf, 1, None)
        cr = cairo.Context(surface)

        # Overlay background
        cr.set_source_rgba(0, 0, 0, 0.6)
        cr.rectangle(0, 0, 350, 140)
        cr.fill()

        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(22)

        cr.set_source_rgb(0, 1, 0)
        cr.move_to(15, 30)
        cr.show_text(f"INF_FPS: {self.fps_inf:.1f}")

        cr.move_to(15, 60)
        cr.show_text(f"DISP_FPS: {self.fps_disp:.1f}")

        cr.set_source_rgb(1, 1, 0)
        cr.move_to(15, 95)
        cr.show_text(f"Fingers: {LABELS[self.last_class]}")

        cr.move_to(15, 125)
        cr.show_text(f"Conf: {self.last_conf:.2f}")

        cr.stroke()

        self.image.set_from_surface(surface)
        return False


if __name__ == "__main__":
    App()
    Gtk.main()