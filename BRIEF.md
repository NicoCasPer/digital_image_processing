# BRIEF para NotebookLM — Generación de Diapositivas

## Instrucción para NotebookLM

Usa **este documento (BRIEF.md)** junto con **README.md** y las **imágenes adjuntas** como fuentes
para generar una **presentación de diapositivas profesional** del proyecto.

**Reglas para las diapositivas:**
1. Sigue el orden y los títulos del guion de la sección "Guion de Diapositivas".
2. En cada diapositiva que indique `🖼️ IMAGEN: <archivo>`, **integra esa imagen** como elemento visual principal o de apoyo.
3. Mantén un tono **técnico pero claro**, apto para una sustentación académica universitaria.
4. Usa frases cortas y viñetas; evita párrafos largos en las diapositivas.
5. Idioma: **español**.

---

## ⭐ CONTENIDO OBLIGATORIO (debe aparecer sí o sí en las diapositivas)

La presentación **debe** incluir, de forma clara y dedicada, las siguientes tres secciones:

1. **Contexto del problema** — Un problema de la vida real, por qué es importante resolverlo, y cuál es la **tarea de inferencia de visión por computador** específica (clasificación, detección o segmentación) que le da solución. → *Diapositiva 2.*

2. **Argumentación técnica** — Justificación de: (a) la **arquitectura** escogida, (b) el **modelo** seleccionado, (c) la **función de pérdida (loss)** implementada y (d) las **métricas de rendimiento** utilizadas, explicando **por qué** cada una es la adecuada para la tarea. → *Diapositivas 5 y 6.*

3. **Gráficas de entrenamiento** — Mostrar de forma clara la progresión del modelo a lo largo de las épocas, incluyendo:
   - Gráfica de la **disminución de la función de pérdida**.
   - Gráfica del **progreso de la métrica seleccionada**, tanto para **entrenamiento** como para **validación**. → *Diapositiva 7.*

---

## Resumen del Proyecto (contexto para NotebookLM)

- **Título:** Detección Automática de Baches Viales con YOLOv8n + ExecuTorch.
- **Institución:** Universidad Nacional de Colombia, Sede Manizales — Procesamiento Digital de Imágenes.
- **Autores:** Juan C. Giraldo y Nicolás Castaño (2025).
- **Problema:** detectar baches en vías a partir de imágenes; los baches son objetos pequeños (mediana 0.9 % del área), lo que lo hace un reto de detección de objetos pequeños.
- **Solución:** entrenar YOLOv8n y exportarlo a ExecuTorch (.pte) para inferencia eficiente en el borde, con un dashboard web en Gradio/HuggingFace Spaces.
- **Logro destacado:** la exportación con XNNPACK logró **~34 ms / 29 FPS** (≈140× más rápido que el backend portable), con un modelo de solo **~12 MB**.

**Datos clave (úsalos en las diapositivas):**
- Dataset: 1 581 imágenes de entrenamiento + 396 de validación (1 977 total), 5 942 anotaciones, 1 clase (`pothole`).
- Métricas: mAP@0.5 = **0.799**, mAP@0.5:0.95 = **0.521**, Precision = **0.812**, Recall = **0.715**.
- Función de pérdida YOLOv8: suma ponderada de **CIoU** (localización) + **DFL** (Distribution Focal Loss, refinamiento de cajas) + **BCE** (clasificación/objectness).

---

## Guion de Diapositivas (deck propuesto)

### Diapositiva 1 — Portada
- Título del proyecto, autores, universidad, año.
- 🖼️ IMAGEN: `val_batch0_pred.jpg` (de fondo o miniatura: el modelo detectando baches).

### Diapositiva 2 — Contexto del Problema  ⭐ OBLIGATORIA
- **Problema real:** los baches deterioran las vías, causan accidentes, daños a vehículos y costos de mantenimiento; su detección manual es lenta y subjetiva.
- **Por qué importa:** automatizar la inspección permite priorizar reparaciones, mejorar la seguridad vial y reducir costos.
- **Tarea de visión por computador:** **DETECCIÓN de objetos** (no clasificación ni segmentación), porque se debe **localizar y contar múltiples baches** dentro de una misma imagen, devolviendo una caja (bounding box) por cada uno.
- (Opcional: ícono/foto de una vía con baches.)

### Diapositiva 3 — La Base de Datos (Dataset)
- Fuente: Potholes Detection YOLOv8 (Kaggle).
- 1 581 train / 396 valid · 1 clase (`pothole`) · formato YOLO.
- Imágenes 200×200 reescaladas a 640×640.
- 🖼️ IMAGEN: `labels.jpg` (distribución de las anotaciones y posiciones de los baches).

### Diapositiva 4 — Análisis Exploratorio del Dataset
- 5 942 anotaciones · media 3.76 baches/imagen · máx. 79.
- Reto: baches pequeños (mediana 0.9 % del área de la imagen).
- 🖼️ IMAGEN: `bbox_distribution.png` (histogramas de ancho, alto y área de las cajas).

### Diapositiva 5 — Argumentación Técnica I: Arquitectura y Modelo  ⭐ OBLIGATORIA
- **Tarea → arquitectura:** la detección requiere un detector *one-stage*; se elige **YOLOv8** por su balance precisión/velocidad y su procesamiento en tiempo real.
- **Por qué YOLOv8n (nano):** la variante más ligera, pensada para **edge** (pocos parámetros, ~12 MB), adecuada para desplegar con ExecuTorch.
- **Componentes:** Backbone **C2f** (características) → Neck **FPN + PAN** (fusión multiescala, clave para **objetos pequeños**) → Head **Anchor-Free** → NMS.
- **Transfer learning:** se parte de pesos preentrenados en COCO para acelerar la convergencia.
- 🖼️ IMAGEN: `yolov8n_architecture.png` (diagrama de arquitectura — VER NOTA DE IMÁGENES).

### Diapositiva 6 — Argumentación Técnica II: Loss y Métricas  ⭐ OBLIGATORIA
- **Función de pérdida (loss):** $\mathcal{L} = \lambda_{box}\,\mathcal{L}_{CIoU} + \lambda_{dfl}\,\mathcal{L}_{DFL} + \lambda_{cls}\,\mathcal{L}_{BCE}$
  - **CIoU** (localización): mejor que IoU porque penaliza la distancia entre centros y la diferencia de relación de aspecto → cajas más precisas.
  - **DFL** (Distribution Focal Loss): modela la caja como una distribución → bordes más finos, ideal para objetos pequeños.
  - **BCE** (clasificación/objectness): adecuada para confianza por clase.
- **Métricas y por qué son las adecuadas para detección:**
  - **mAP@0.5** y **mAP@0.5:0.95**: miden localización (vía IoU) + clasificación a múltiples umbrales; estándar en detección.
  - **Precision / Recall**: equilibran falsos positivos vs. baches no detectados; críticas en seguridad vial.
  - *(Se descarta "accuracy" simple porque la detección exige emparejar predicción y verdad por IoU, no solo acertar la clase.)*

### Diapositiva 7 — Gráficas de Entrenamiento  ⭐ OBLIGATORIA
- **Disminución de la función de pérdida:** las curvas de `box_loss`, `cls_loss` y `dfl_loss` descienden de forma sostenida a lo largo de las 50 épocas (tanto en train como en val).
- **Progreso de la métrica (train y val):** las curvas de **Precision, Recall y mAP** crecen y se estabilizan, mostrando convergencia sin sobreajuste severo.
- 🖼️ IMAGEN: `results.png` (este gráfico de Ultralytics contiene **ambas** cosas: las pérdidas de entrenamiento/validación y la progresión de las métricas — señálalo explícitamente).

### Diapositiva 8 — Métricas Finales de Validación
- mAP@0.5 = 0.799 · mAP@0.5:0.95 = 0.521 · Precision = 0.812 · Recall = 0.715.
- 🖼️ IMAGEN: `BoxPR_curve.png` (curva Precision-Recall).

### Diapositiva 9 — Diagnóstico del Modelo
- Matriz de confusión y curva F1-Confidence.
- 🖼️ IMAGEN: `confusion_matrix.png` (y opcionalmente `BoxF1_curve.png` como segunda imagen).

### Diapositiva 10 — Resultados Cualitativos
- Comparación predicción vs. verdad (ground truth).
- 🖼️ IMAGEN: `val_batch0_pred.jpg` (predicciones) — opcional comparar con `val_batch0_labels.jpg`.

### Diapositiva 11 — Optimización Edge con ExecuTorch
- Export torch.export + XnnpackPartitioner → `.pte` de ~12 MB.
- Resultado: ~34 ms / ~29 FPS / ~140× vs. portable.
- (Tabla de números; sin imagen o ícono de chip/edge.)

### Diapositiva 12 — El Dashboard (HuggingFace Space)
- Inferencia interactiva, sliders conf/IoU, severidad por color, analítica JSON en vivo.
- Tabla de severidad: Leve (verde), Moderado (amarillo), Severo (naranja), Crítico (rojo).
- 🖼️ IMAGEN: captura del dashboard (VER NOTA DE IMÁGENES) o reusar `val_batch0_pred.jpg`.

### Diapositiva 13 — Conclusiones y Trabajo Futuro
- YOLOv8n + ExecuTorch = detección precisa y eficiente en el borde.
- Futuro: más datos, cuantización INT8, despliegue en móvil/Android.

---

## NOTA DE IMÁGENES — Qué archivos subir a NotebookLM

Sube estos archivos como **fuentes/imágenes** del notebook (junto a README.md y BRIEF.md):

**Ya disponibles en el repositorio (carpeta `presentacion_imagenes/`):**
| Archivo | Para la diapositiva |
|---|---|
| `labels.jpg` | 3 (dataset) |
| `bbox_distribution.png` | 4 (análisis EDA) |
| `results.png` | 7 (gráficas de entrenamiento: loss + métricas) |
| `BoxPR_curve.png` | 8 (curva P-R) |
| `BoxF1_curve.png` | 9 (F1, opcional) |
| `confusion_matrix.png` | 9 (diagnóstico) |
| `val_batch0_pred.jpg` | 1, 10, 12 (resultados) |
| `val_batch0_labels.jpg` | 10 (ground truth, opcional) |

**Que debes conseguir/crear tú (no están en el repo):**
| Archivo sugerido | Cómo obtenerlo | Para la diapositiva |
|---|---|---|
| `yolov8n_architecture.png` | Diagrama de arquitectura YOLOv8 (documentación de Ultralytics o un diagrama backbone C2f + Neck FPN/PAN + Head) | 5 (arquitectura) |
| Captura del dashboard | Screenshot del Space funcionando con una detección | 12 (dashboard) |

> Consejo: en NotebookLM, sube primero `README.md` y `BRIEF.md` como fuentes de texto, y luego las imágenes. Al pedir la presentación, indica: *"Genera las diapositivas siguiendo el guion del BRIEF.md, respeta el CONTENIDO OBLIGATORIO e integra cada imagen en la diapositiva indicada."*
