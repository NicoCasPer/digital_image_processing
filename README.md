---
title: Detector de Baches Viales YOLOv8n
emoji: 🚧
colorFrom: yellow
colorTo: blue
sdk: gradio
app_file: app.py
pinned: true
license: agpl-3.0
tags:
  - object-detection
  - yolov8
  - executorch
  - edge-ai
  - computer-vision
---

<!-- El bloque de arriba (entre ---) es la configuración del Space en HuggingFace
     (título, emoji, colores, SDK). NO se muestra como texto; solo configura la tarjeta. -->

# 🚧 Detector de Baches Viales — YOLOv8n + ExecuTorch

> **Universidad Nacional de Colombia — Sede Manizales**
> Procesamiento Digital de Imágenes · Juan C. Giraldo · Nicolás Castaño · 2025

---

## 1. La Base de Datos (Dataset)

El proyecto se entrena sobre el dataset público **[Potholes Detection YOLOv8](https://www.kaggle.com/datasets/anggadwisunarto/potholes-detection-yolov8)** (Kaggle), un conjunto de imágenes reales de vías con baches anotados en formato **YOLO** (un archivo `.txt` por imagen con `clase x_centro y_centro ancho alto`, todo normalizado entre 0 y 1).

### 1.1 Composición

| Partición | Imágenes | Etiquetas | Uso |
|---|---|---|---|
| **Train** | 1 581 | 1 581 | Entrenamiento del modelo |
| **Valid** | 396 | 396 | Validación y métricas |
| **Total** | **1 977** | **1 977** | — |

- **Clases:** 1 → `pothole` (problema de detección de **una sola clase**).
- **Anotaciones de entrenamiento:** **5 942** cajas delimitadoras (bounding boxes).
- **Resolución original de las imágenes:** 200 × 200 px (se reescalan a **640 × 640** con *letterbox* para el entrenamiento e inferencia).

### 1.2 Análisis estadístico

| Métrica | Valor | Interpretación |
|---|---|---|
| Baches por imagen (media) | **3.76** | Escenas densas, varios baches por foto |
| Baches por imagen (máx.) | **79** | Tramos de vía muy deteriorados |
| Imágenes sin baches | **0** | Todas las imágenes contienen al menos un bache |
| Área relativa del bbox (mediana) | **0.9 %** | La mayoría de los baches son **pequeños** |
| Área relativa del bbox (media) | **5.9 %** | Distribución sesgada por unos pocos baches grandes |

> **Reto técnico clave:** como la mediana del área de un bache es apenas **0.9 %** de la imagen, este es un problema de **detección de objetos pequeños**. Esto justifica el uso de YOLOv8n con su cabeza *anchor-free* y la fusión multiescala (FPN + PAN), que conservan la resolución espacial necesaria para localizar objetos diminutos.

*(Ver `bbox_distribution.png` y `labels.jpg` para la distribución visual de tamaños y posiciones.)*

---

## 2. Descripción General del Proyecto

Sistema de **detección automática de baches** en vías, pensado para ejecutarse en el **borde (edge)**: el modelo se entrena en PyTorch con **YOLOv8n** y se exporta a **ExecuTorch** (`.pte`), el runtime ligero de PyTorch para móviles y dispositivos embebidos. Una interfaz web en **Gradio** (desplegada en HuggingFace Spaces) permite probar el modelo de forma interactiva y muestra métricas de rendimiento en tiempo real.

**Objetivos:**
1. Entrenar un detector preciso de baches con un dataset real.
2. Optimizar el modelo para inferencia en CPU/edge mediante ExecuTorch + XNNPACK.
3. Construir un dashboard profesional que clasifique la **severidad** de cada bache y reporte la **latencia** de inferencia.

---

## 3. Arquitectura del Modelo — YOLOv8n

**YOLOv8n** (*nano*) es la variante más ligera de la familia YOLOv8, ideal para edge:

```
📷 Imagen 640×640
       │
       ▼
🧠 Backbone (C2f)        →  extracción jerárquica de características
       │
       ▼
🔀 Neck (FPN + PAN)      →  fusión de características multiescala
       │
       ▼
🎯 Head (Anchor-Free)    →  predice cajas + score por anclaje
       │
       ▼
✂️ NMS                   →  elimina detecciones duplicadas
       │
       ▼
🚧 Baches detectados
```

- **Anchor-free:** predice directamente la caja sin cajas-ancla predefinidas → más simple y rápido.
- **Salida cruda del `.pte`:** tensor `[1, 5, 8400]` = 4 coordenadas (cx, cy, w, h) + 1 score, sobre 8400 anclajes.
- **Post-procesado** (decode xywh→xyxy, NMS, reescalado *letterbox*) implementado manualmente en el dashboard, sin depender de Ultralytics en runtime.

---

## 4. Pipeline: de los Píxeles al Despliegue Edge

```
Dataset (Kaggle)
   │  letterbox 640×640 + data augmentation (mosaic, mixup)
   ▼
Entrenamiento YOLOv8n  (50 épocas · AdamW · transfer learning desde COCO)
   │
   ▼
best.pt (PyTorch)
   │  torch.export + XnnpackPartitioner
   ▼
best_baches.pte (ExecuTorch ~12 MB)
   │
   ▼
Dashboard Gradio (HuggingFace Space)  →  inferencia en CPU
```

---

## 5. Resultados y Métricas

Evaluación sobre el conjunto de validación (396 imágenes):

| mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|:---:|:---:|:---:|:---:|
| **0.799** | **0.521** | **0.812** | **0.715** |

**Rendimiento edge (ExecuTorch + XNNPACK, CPU):**

| Métrica | Valor |
|---|---|
| Latencia de inferencia | **~34 ms** |
| FPS estimado | **~29** |
| Tamaño del modelo | **~12 MB** |
| Aceleración vs. backend portable | **~140×** |

*Artefactos visuales disponibles: `results.png` (curvas de pérdida y mAP), `BoxPR_curve.png`, `BoxF1_curve.png`, `confusion_matrix.png`, `val_batch0_pred.jpg` (predicciones).*

---

## 6. El Dashboard (HuggingFace Space)

- **Pestaña 1 — Inferencia interactiva:** sube una imagen o usa la webcam, ajusta los umbrales de confianza e IoU, y obtén las detecciones.
- **Clasificación de severidad por color:** cada bache se colorea según el área relativa que ocupa:

  | Severidad | Color | Área relativa |
  |---|---|---|
  | Leve | 🟢 Verde | < 3 % |
  | Moderado | 🟡 Amarillo | 3 – 10 % |
  | Severo | 🟠 Naranja | 10 – 25 % |
  | Crítico | 🔴 Rojo | > 25 % |

- **Analítica JSON en vivo:** número de baches, severidad global de la vía, latencia, FPS y tamaño del modelo.
- **Pestaña 2 — Métricas y teoría:** curvas de entrenamiento, P-R, F1 y fundamento matemático.

---

## 7. Cómo Usar

1. Abre el Space y ve a la pestaña **Inferencia Interactiva**.
2. Sube una foto de una vía (o activa la webcam).
3. Ajusta **conf** (confianza) e **IoU** (supresión de duplicados) según necesites.
4. Revisa las cajas coloreadas por severidad y la analítica en el panel JSON.

---

## 8. Stack Técnico y Reproducibilidad

| Componente | Tecnología |
|---|---|
| Entrenamiento | Ultralytics YOLOv8n · PyTorch 2.12 |
| Exportación edge | ExecuTorch 1.3.1 (XNNPACK) |
| Interfaz | Gradio |
| Post-procesado | NumPy + OpenCV (NMS manual) |
| Despliegue | HuggingFace Spaces |

**Notebooks del proyecto:**
- `Proyecto_PDI_entrenamiento.ipynb` — adquisición, EDA, entrenamiento, validación y exportación a `.pte`.
- `Proyecto_PDI_HuggingFace.ipynb` — construcción del `app.py` y despliegue del Space.

---

## 9. Estructura del Repositorio

```
digital_image_processing/
├── dataset/                       # Imágenes y etiquetas (train/valid)
├── runs/detect/.../train/         # Resultados de entrenamiento (curvas, matrices)
├── best_baches.pte                # Modelo ExecuTorch desplegable
├── app.py                         # Dashboard Gradio (inferencia .pte)
├── metrics.json                   # Métricas de validación
├── Proyecto_PDI_entrenamiento.ipynb
└── Proyecto_PDI_HuggingFace.ipynb
```

---

## Dataset

[Potholes Detection YOLOv8 — Kaggle](https://www.kaggle.com/datasets/anggadwisunarto/potholes-detection-yolov8)

---

*Proyecto Final · Procesamiento Digital de Imágenes · UNAL Manizales · 2026*
