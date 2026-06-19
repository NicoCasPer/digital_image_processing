"""
Sistema de Deteccion de Baches Viales - YOLOv8n exportado a ExecuTorch (.pte)
Universidad Nacional de Colombia - Procesamiento Digital de Imagenes
Autores: Juan C. Giraldo - Nicolas Castano

El dashboard ejecuta el modelo .pte con el runtime de ExecuTorch.
La cabeza Detect entrega [1, 5, 8400] (cx,cy,w,h + score); el decode
(xywh->xyxy), el NMS y el reescalado letterbox se hacen manualmente.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import gradio as gr
import numpy as np
import torch
from executorch.runtime import Runtime

# ----------------------------------------------------------------------
# 1. CARGA DEL MODELO EXECUTORCH (.pte)
# ----------------------------------------------------------------------
def first_existing_path(candidates: list[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


PTE_CANDIDATES = [
    #Path("best_baches.pte"),
    Path("model.pte"),
    Path("yolov8n_executorch_model/best_baches.pte"),
]
PTE_PATH = first_existing_path(PTE_CANDIDATES)
if PTE_PATH is None:
    raise FileNotFoundError(
        "No se encontro el modelo ExecuTorch. Sube 'best_baches.pte' a la raiz del Space."
    )

IMG_SZ = 640
CLASS_NAMES = ["pothole"]
MODEL_SIZE_MB = PTE_PATH.stat().st_size / 1e6

_runtime = Runtime.get()
_program = _runtime.load_program(str(PTE_PATH))
_method = _program.load_method("forward")
print(f"[INFO] Modelo ExecuTorch cargado desde: {PTE_PATH} ({MODEL_SIZE_MB:.1f} MB)")

# Warmup: la 1a ejecucion incluye inicializacion perezosa del runtime (cold start);
# la lanzamos aqui para que la latencia mostrada al usuario sea la de regimen.
try:
    _method.execute([torch.zeros(1, 3, IMG_SZ, IMG_SZ)])
    print("[INFO] Warmup de inferencia completado.")
except Exception as _e:
    print(f"[WARN] Warmup fallido (no critico): {_e}")


# ----------------------------------------------------------------------
# 2. PRE / POST-PROCESADO
# ----------------------------------------------------------------------
def letterbox(im: np.ndarray, new_shape: int = 640, color=(114, 114, 114)):
    """Redimensiona manteniendo aspect ratio y rellena a new_shape x new_shape."""
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    left, top = (new_shape - nw) // 2, (new_shape - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Non-Maximum Suppression en numpy. boxes en formato xyxy."""
    if boxes.shape[0] == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def run_executorch(img_rgb: np.ndarray, conf_thr: float, iou_thr: float):
    """Ejecuta el .pte y devuelve (boxes_xyxy_originales, scores, latencia_ms)."""
    canvas, r, left, top = letterbox(img_rgb, IMG_SZ)
    x = canvas.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]                 # (1,3,640,640)
    xt = torch.from_numpy(np.ascontiguousarray(x))

    t0 = time.perf_counter()
    out = _method.execute([xt])[0]                       # (1,5,8400)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    pred = out[0].cpu().numpy().transpose(1, 0)          # (8400,5)
    scores = pred[:, 4]
    mask = scores >= conf_thr
    pred, scores = pred[mask], scores[mask]
    if pred.shape[0] == 0:
        return np.zeros((0, 4)), np.zeros((0,)), latency_ms

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    keep = nms_numpy(boxes, scores, iou_thr)
    boxes, scores = boxes[keep], scores[keep]

    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes /= r
    return boxes, scores, latency_ms


# ----------------------------------------------------------------------
# 3. SEVERIDAD, COLORES Y METRICAS
# ----------------------------------------------------------------------
SEVERITY_THRESHOLDS = {
    "Leve": (0.00, 0.03),
    "Moderado": (0.03, 0.10),
    "Severo": (0.10, 0.25),
    "Critico": (0.25, 1.00),
}

# Colores RGB por severidad (verde -> amarillo -> naranja -> rojo)
SEVERITY_COLORS = {
    "Leve":     (34, 197, 94),
    "Moderado": (234, 179, 8),
    "Severo":   (249, 115, 22),
    "Critico":  (239, 68, 68),
    "Sin dano": (148, 163, 184),
}

DEFAULT_METRICS = {
    "map50": 0.7964,
    "map5095": 0.5203,
    "precision": 0.8074,
    "recall": 0.7068,
}

METRICS_PATH = first_existing_path([Path("metrics.json")])
TRAIN_RESULTS_PATH = first_existing_path([Path("runs/detect/train/results.png")])
PR_CURVE_PATH = first_existing_path([Path("runs/detect/train/BoxPR_curve.png"),
                                     Path("runs/detect/train/PR_curve.png")])
F1_CURVE_PATH = first_existing_path([Path("runs/detect/train/BoxF1_curve.png"),
                                     Path("runs/detect/train/F1_curve.png")])


def load_metrics(path: Optional[Path]) -> dict[str, float]:
    if path is None:
        return DEFAULT_METRICS.copy()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: float(data.get(k, DEFAULT_METRICS[k])) for k in DEFAULT_METRICS}
    except Exception:
        return DEFAULT_METRICS.copy()


METRICS = load_metrics(METRICS_PATH)

# ----------------------------------------------------------------------
# 4. ESTILO CSS
# ----------------------------------------------------------------------
CSS = """
:root {
    --accent: #E8593C;
    --bg-card: #FAFAF9;
    --border: rgba(0,0,0,0.10);
    --mono: 'JetBrains Mono', 'Fira Code', monospace;
}
.gradio-container { font-family: 'Inter', system-ui, sans-serif !important; }
.tab-nav button { font-weight: 600 !important; letter-spacing: .02em; }
#header-block {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-radius: 12px; padding: 24px 32px; margin-bottom: 8px;
}
#header-block h1 { color: #F0EEE4; margin: 0 0 4px; font-size: 1.6rem; font-weight: 700; }
#header-block p { color: #9FA6B2; margin: 0; font-size: 0.875rem; }
#header-block .badge {
    display: inline-block; background: var(--accent); color: #fff; border-radius: 6px;
    padding: 2px 10px; font-size: 0.75rem; font-weight: 700; margin-left: 8px; vertical-align: middle;
}
#edge-bar {
    display: flex; gap: 18px; flex-wrap: wrap; margin: 10px 0 4px;
    font-family: var(--mono); font-size: 0.8rem; color: #334155;
}
#edge-bar .chip {
    background: #F1F5F9; border: 1px solid var(--border); border-radius: 8px; padding: 6px 12px;
}
#edge-bar .chip b { color: var(--accent); }
.control-panel { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
.control-panel label { font-size: 0.82rem !important; font-weight: 600; }
.metric-card {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px 20px; text-align: center;
}
.metric-card .metric-value { font-size: 2rem; font-weight: 700; color: var(--accent); display: block; line-height: 1.1; }
.metric-card .metric-label { font-size: 0.78rem; color: #6B7280; margin-top: 4px; display: block; }
.legend { display:flex; gap:14px; flex-wrap:wrap; margin-top:8px; font-size:0.82rem; }
.legend span { display:inline-flex; align-items:center; gap:6px; }
.legend i { width:14px; height:14px; border-radius:3px; display:inline-block; }
"""

EDGE_BAR_HTML = f"""
<div id='edge-bar'>
  <div class='chip'>Motor: <b>ExecuTorch (.pte)</b></div>
  <div class='chip'>Modelo: <b>{MODEL_SIZE_MB:.1f} MB</b></div>
  <div class='chip'>Backbone: <b>YOLOv8n</b></div>
  <div class='chip'>Entrada: <b>640 x 640</b></div>
</div>
"""

LEGEND_HTML = """
<div class='legend'>
  <span><i style='background:#22C55E'></i>Leve</span>
  <span><i style='background:#EAB308'></i>Moderado</span>
  <span><i style='background:#F97316'></i>Severo</span>
  <span><i style='background:#EF4444'></i>Critico</span>
</div>
"""

# ----------------------------------------------------------------------
# 5. INFERENCIA + ANALITICA
# ----------------------------------------------------------------------
def classify_severity(rel_area: float) -> str:
    if rel_area <= 0:
        return "Sin dano"
    for label, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= rel_area < hi:
            return label
    return "Critico"


def predict_potholes(image: np.ndarray, conf_threshold: float = 0.35, iou_threshold: float = 0.45):
    if image is None:
        return np.zeros((480, 640, 3), dtype=np.uint8), {"error": "No se recibio imagen."}
    try:
        H, W = image.shape[:2]
        img_area = float(H * W) if H > 0 and W > 0 else 1.0

        boxes, scores, latency_ms = run_executorch(image, conf_threshold, iou_threshold)

        annotated = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        detections: list[dict[str, Any]] = []
        total_bbox_area = 0.0

        for idx, (box, conf) in enumerate(zip(boxes, scores), start=1):
            x1, y1, x2, y2 = [int(max(0, v)) for v in box]
            bbox_area = float(max(0, x2 - x1) * max(0, y2 - y1))
            total_bbox_area += bbox_area
            rel = bbox_area / img_area
            sev = classify_severity(rel)
            color = SEVERITY_COLORS.get(sev, (232, 89, 60))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{sev} {conf * 100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
            ytop = max(th + 6, y1)
            cv2.rectangle(annotated, (x1, ytop - th - 6), (x1 + tw + 8, ytop), color, -1)
            cv2.putText(annotated, label, (x1 + 4, ytop - 4), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            detections.append({
                "id": idx,
                "confianza": round(float(conf), 3),
                "bbox": [x1, y1, x2, y2],
                "area_px": int(bbox_area),
                "area_relativa": f"{rel * 100:.2f} %",
                "severidad": sev,
            })

        total_rel = total_bbox_area / img_area
        analytics = {
            "resumen": {
                "baches_detectados": len(detections),
                "severidad_global_via": classify_severity(total_rel),
                "area_total_afectada": f"{total_rel * 100:.2f} %",
                "conf_threshold": round(conf_threshold, 2),
                "iou_threshold": round(iou_threshold, 2),
            },
            "rendimiento_edge": {
                "latencia_inferencia_ms": round(latency_ms, 1),
                "fps_estimado": round(1000.0 / latency_ms, 1) if latency_ms > 0 else None,
                "tamano_modelo_mb": round(MODEL_SIZE_MB, 1),
                "motor": "ExecuTorch (.pte, CPU)",
            },
            "detecciones": detections,
        }
        return annotated, analytics
    except Exception as exc:
        return np.zeros((480, 640, 3), dtype=np.uint8), {"error": f"Error durante la inferencia: {exc}"}


def build_metrics_html(m: dict[str, float]) -> str:
    return f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px;">
  <div class="metric-card"><span class="metric-value">{m['map50']:.3f}</span><span class="metric-label">mAP@0.5</span></div>
  <div class="metric-card"><span class="metric-value">{m['map5095']:.3f}</span><span class="metric-label">mAP@0.5:0.95</span></div>
  <div class="metric-card"><span class="metric-value">{m['precision']:.3f}</span><span class="metric-label">Precision</span></div>
  <div class="metric-card"><span class="metric-value">{m['recall']:.3f}</span><span class="metric-label">Recall</span></div>
</div>
"""


METRICS_HTML = build_metrics_html(METRICS)
THEORY_MD = r"""
### YOLOv8 sobre ExecuTorch en el borde

El modelo se entrena en PyTorch y se exporta a **ExecuTorch** (`.pte`), un runtime
ligero para dispositivos edge/movil. La cabeza `Detect` entrega un tensor
$[1, 5, 8400]$ (4 coordenadas + 1 score). El decodificado xywh->xyxy, el
**NMS** y el reescalado *letterbox* se ejecutan en este Space sin depender de Ultralytics.

$$\text{IoU}(A,B) = \frac{|A \cap B|}{|A \cup B|} \qquad P = \frac{TP}{TP+FP} \qquad R = \frac{TP}{TP+FN}$$
"""

# ----------------------------------------------------------------------
# 6. INTERFAZ gr.Blocks
# ----------------------------------------------------------------------
with gr.Blocks(
    title="Detector de Baches Viales - UNAL Manizales",
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
    ),
    css=CSS,
) as demo:

    gr.HTML(
        """
        <div id='header-block'>
          <h1>Sistema de Deteccion de Baches Viales <span class='badge'>YOLOv8n - ExecuTorch</span></h1>
          <p>Universidad Nacional de Colombia - Sede Manizales | PDI |
             Juan C. Giraldo - Nicolas Castano</p>
        </div>
        """
    )
    gr.HTML(EDGE_BAR_HTML)

    with gr.Tab("Inferencia Interactiva"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("### Hiperparametros")
                conf_slider = gr.Slider(0.05, 0.95, value=0.35, step=0.05,
                                        label="Umbral de Confianza (conf)",
                                        elem_classes=["control-panel"])
                iou_slider = gr.Slider(0.10, 0.90, value=0.45, step=0.05,
                                       label="Umbral de IoU - NMS (iou)",
                                       elem_classes=["control-panel"])
                gr.HTML(LEGEND_HTML)
            with gr.Column(scale=3):
                with gr.Row():
                    input_image = gr.Image(type="numpy", label="Imagen de Entrada",
                                           sources=["upload", "webcam"])
                    output_image = gr.Image(type="numpy", label="Baches Detectados", interactive=False)
                run_btn = gr.Button("Detectar Baches", variant="primary", size="lg")

        gr.Markdown("### Analitica y Rendimiento Edge")
        analytics_box = gr.JSON(label="Resumen analitico (JSON)",
                                value={"resumen": "Sube una imagen para comenzar."})

        for trigger in (run_btn.click, conf_slider.change, iou_slider.change):
            trigger(fn=predict_potholes,
                    inputs=[input_image, conf_slider, iou_slider],
                    outputs=[output_image, analytics_box])

    with gr.Tab("Metricas & Teoria"):
        gr.Markdown("## Metricas de Entrenamiento")
        gr.HTML(METRICS_HTML)
        with gr.Row():
            with gr.Column(scale=3):
                if TRAIN_RESULTS_PATH:
                    gr.Image(value=str(TRAIN_RESULTS_PATH), label="Curvas de Entrenamiento", interactive=False)
                if PR_CURVE_PATH:
                    gr.Image(value=str(PR_CURVE_PATH), label="Curva Precision-Recall", interactive=False)
                if F1_CURVE_PATH:
                    gr.Image(value=str(F1_CURVE_PATH), label="Curva F1-Confidence", interactive=False)
            with gr.Column(scale=2):
                gr.Markdown(THEORY_MD)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
