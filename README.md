# Driver Fatigue Detection

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-linux%2Famd64-2496ED)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Note:** This is a fork of the original team project developed at Instituto Tecnológico de Jiquilpan.  
> **Original repo:** [AngelCanelaaa/deteccion-fatiga-conductores](https://github.com/AngelCanelaaa/deteccion-fatiga-conductores)

Computer vision and AI system that detects signs of driver fatigue from webcam images. It classifies the driver's state into four categories (eyes open, eyes closed, yawning, not yawning) and generates visual alerts in real time.

It implements a double-prediction strategy: one inference over the eye region and another over the full face, choosing the one with the highest confidence. The MobileNetV2 model with Transfer Learning reached **99.02% test accuracy**.

---

## English

### Table of contents

- [Context](#context)
- [What it solves](#what-it-solves)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Installation and usage](#installation-and-usage)
- [API endpoints](#api-endpoints)
- [Double-prediction strategy](#double-prediction-strategy)
- [Dataset and training](#dataset-and-training)
- [Team roles](#team-roles)
- [Additional documentation](#additional-documentation)
- [License](#license)

### Context

Drowsiness behind the wheel is one of the leading causes of traffic accidents. Detecting early signs of fatigue (closing eyes, yawning) can prevent incidents if the system alerts the driver in time.

This project implements a complete fatigue detection system: the frontend captures webcam frames every 800 ms and sends them to a FastAPI service that applies OpenCV preprocessing (face detection, eye cropping, CLAHE) and classifies the state with a MobileNetV2 model trained with Transfer Learning.

Team project developed for the Computer Vision and Machine Learning course.

### What it solves

| Challenge | Implemented solution |
|---|---|
| Robust detection of fatigue signals | Double-prediction strategy: eye region + full face |
| Gap between training images and real use | Datasets contain cropped eyes; the webcam captures the full face |
| Image preprocessing in production | OpenCV pipeline: Haar Cascade → CLAHE → resize 224×224 → normalization |
| Mac M3 compatibility | Docker with `--platform=linux/amd64` and `tensorflow-cpu` inside the container |
| Real-time visual alerts | Frontend with automatic capture every 800 ms and color-coded feedback |

### Screenshots

**Open-eye detection**

The driver is alert. The double-prediction strategy chooses the cropped eye region because it provides higher confidence.

![Open eye](capturas/01-frontend-ojo-abierto.png)

**Yawn detection**

The system identifies a yawn (98.9% confidence) and shows an attention alert.

![Yawn detected](capturas/02-frontend-bostezo.png)

**Interactive API documentation (Swagger UI)**

![Swagger UI](capturas/03-swagger.png)

### Architecture

```
[ Driver's webcam ]
       |
       | Frames every 800ms (JPEG)
       v
[ HTML5 Frontend ]
       |
       | multipart/form-data
       v
[ Docker Port 8000:8000 ]
       |
       v
[ FastAPI + Uvicorn ]
       |
       | Face detection (Haar Cascade)
       | Eye region crop
       | CLAHE + resize + normalization
       v
[ MobileNetV2 Model (modelo_fatiga.keras) ]
       |
       | 4 probabilities (softmax)
       v
[ JSON Response: class + confidence + alert_level + mode ]
       |
       v
[ Frontend: visual alert + bars + history ]
```

### Tech stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| AI model | TensorFlow / Keras 2.16 |
| Architecture | MobileNetV2 + Transfer Learning |
| Computer vision | OpenCV 4.11 |
| API | FastAPI 0.115 + Uvicorn |
| Container | Docker with `linux/amd64` platform |
| Frontend | HTML5 + JavaScript + Canvas API |

### Project structure

```
deteccion-fatiga-conductores/
├── api/
│   ├── Dockerfile
│   ├── main.py                          # FastAPI application
│   ├── requirements.txt
│   ├── frontend/
│   │   └── index.html                   # Camera interface
│   └── model/
│       ├── modelo_fatiga.keras          # Trained model
│       ├── modelo_fatiga_metadata.json
│       └── cascades/
│           ├── haarcascade_frontalface_default.xml
│           └── haarcascade_eye.xml
├── docs/
│   ├── documentacion-tecnica.pdf
│   └── guia-despliegue.md
├── capturas/
├── LICENSE
└── README.md
```

### Installation and usage

**Requirements**

- Python 3.12 (recommended: install with `pyenv`)
- Docker Desktop (for the container variant)

**Option A — Run locally with Uvicorn**

1. Clone the repository

   ```bash
   git clone https://github.com/KPVE2004/deteccion-fatiga-conductores.git
   cd deteccion-fatiga-conductores
   ```

2. Create and activate a virtual environment

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies

   ```bash
   pip install -r api/requirements.txt
   ```

4. Start the API

   ```bash
   cd api
   uvicorn main:app --reload
   ```

5. Open the frontend

   Open `http://localhost:8000/` in your browser. The camera will ask for permission.

**Option B — Run with Docker**

```bash
cd api
docker buildx build --platform linux/amd64 -t img-fatiga .
docker run -d --name srv-fatiga -p 8000:8000 img-fatiga:latest
```

Verify it's running:

```bash
docker ps
```

Open `http://localhost:8000/` for the interface and `http://localhost:8000/docs` for Swagger.

### API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the HTML frontend |
| GET | `/health` | Model and system status |
| POST | `/predict` | Receives an image, returns class + confidence + alert level |
| GET | `/modelo/info` | Technical information about the model |

Example `/predict` response:

```json
{
  "clase": "ojo_cerrado",
  "probabilidad": 0.9923,
  "nivel_alerta": "PELIGRO",
  "color_alerta": "#F44336",
  "mensaje_alerta": "Ojos cerrados - posible fatiga",
  "todas_las_clases": {
    "ojo_abierto": 0.0041,
    "ojo_cerrado": 0.9923,
    "bostezo": 0.0022,
    "no_bostezo": 0.0014
  },
  "tiempo_inferencia_ms": 43.7,
  "modo_preprocesamiento": "ojos_recortados",
  "cara_detectada": true
}
```

**Alert levels:**

| Class | Level | Color | Action |
|---|---|---|---|
| ojo_abierto | NORMAL | Green | No action |
| no_bostezo | NORMAL | Green | No action |
| bostezo | WARNING | Orange | Suggest a break |
| ojo_cerrado | DANGER | Red | Immediate alarm |

### Double-prediction strategy

The model was trained with three datasets containing cropped eye images. However, the webcam captures the driver's full face. To bridge that gap, a double-prediction strategy was implemented:

1. **Prediction A — Eye region:** the upper half of the detected face is cropped and a single eye is located. Specialized in `ojo_abierto` / `ojo_cerrado`.
2. **Prediction B — Full face:** the entire face is cropped. Specialized in `bostezo` / `no_bostezo`.
3. **Final decision:** the maximum confidence of A on eye classes is compared with the maximum confidence of B on mouth classes. The highest wins.
4. **Fallback:** if no face is detected, prediction is made directly on the full image.

The `modo_preprocesamiento` field in the response indicates which path was taken.

### Dataset and training

**Datasets used** (all public on Kaggle):

| Dataset | Images | Original classes |
|---|---|---|
| MRL Eye Dataset | 4,000 | Open_Eyes, Closed_Eyes |
| Yawn Eye Dataset | 2,900 | yawn, no_yawn, Open, Closed |
| Drowsiness / CEW | 47,992 | open_eye, closed_eye |
| **Total** | **54,892** | **4 unified classes** |

**Preprocessing applied:**

- Curation: MD5 verification, removal of 8 corrupted images (0.02%).
- CLAHE in LAB space: contrast improvement +48% (MRL) and +60.5% (CEW).
- Resize to 224×224 px.
- Normalization to [0, 1].
- Stratified 70/15/15 split (seed 42).

**Training (MobileNetV2 + Transfer Learning):**

- Phase 1 — Feature Extraction (10 epochs, frozen backbone): 97.15% val accuracy.
- Phase 2 — Fine-Tuning (15 epochs, last 30 layers unfrozen, LR=1e-5): 99.08% val accuracy.
- Final test accuracy: **99.02%**.
- Model: 3,049,284 parameters, 29.9 MB.

### Team roles

Project developed by three team members.

**Ángel de Jesús Canela Figueroa**

- Implemented the double-prediction strategy in the backend.
- Developed the preprocessing pipeline (Haar Cascade, CLAHE, smart cropping).
- Configured the Dockerfile for Mac M3 (Apple Silicon) compatibility.
- Prepared the deployment on Google Cloud Run.
- Technical documentation and deployment guide.

**Isamar Chávez Díaz**

- **Training, phase 1:** Transfer Learning with MobileNetV2 and frozen backbone (10 epochs), with 97.15% validation accuracy.
- **Dataset preparation:** integrated three public Kaggle datasets (54,892 images) into 4 unified classes.
- **Data curation and preprocessing:** MD5 verification (8 corrupted images removed), contrast enhancement with CLAHE in LAB space, resize to 224×224, normalization, and stratified 70/15/15 split.
- **Model training** (joint): participated in MobileNetV2 training with Transfer Learning and Fine-Tuning.
- **Web interface** (joint): participated in the frontend development with camera, visual alerts, and probability bars.
- **System testing:** end-to-end validation of the camera, API, and visual alert.

**Katia Paola Victoria Escalera**

- **Training, phase 2:** Fine-Tuning by unfreezing the last 30 layers with a learning rate of 1e-5 (15 epochs), with 99.08% validation accuracy.
- **Model evaluation:** measurement on the test set, with a final accuracy of 99.02%.
- **Web interface:** frontend development in HTML5, JavaScript, and Canvas API, with webcam access and automatic frame capture every 800 ms.
- **Dataset preparation** (joint): integrated three public Kaggle datasets (54,892 images) into 4 unified classes.
- **Data curation and preprocessing** (joint): MD5 verification, CLAHE contrast enhancement, resize to 224×224, normalization, and stratified 70/15/15 split.
- **System testing** (joint): end-to-end validation of the camera, API, and visual alert.

### Additional documentation

- [Technical documentation](docs/documentacion-tecnica.pdf) — full project report.
- [Deployment guide](docs/guia-despliegue.md) — deployment process on Google Cloud Run.

### License

MIT — see [LICENSE](LICENSE).

---

## Español

Sistema de visión por computadora e inteligencia artificial que detecta señales de fatiga en conductores a partir de imágenes de la cámara web. Clasifica el estado del conductor en cuatro categorías (ojo abierto, ojo cerrado, bostezo, no bostezo) y genera alertas visuales en tiempo real.

Implementa una estrategia de doble predicción: una inferencia sobre la región de ojos y otra sobre la cara completa, eligiendo la de mayor confianza. El modelo MobileNetV2 con Transfer Learning alcanzó **99.02% de precisión en test**.

### Tabla de contenidos

- [Contexto](#contexto)
- [Qué resuelve](#qué-resuelve)
- [Capturas](#capturas)
- [Arquitectura](#arquitectura)
- [Stack tecnológico](#stack-tecnológico)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación y uso](#instalación-y-uso)
- [Endpoints de la API](#endpoints-de-la-api)
- [Estrategia de doble predicción](#estrategia-de-doble-predicción)
- [Dataset y entrenamiento](#dataset-y-entrenamiento)
- [Roles del equipo](#roles-del-equipo)
- [Documentación adicional](#documentación-adicional)
- [Licencia](#licencia)

### Contexto

La somnolencia al volante es una de las principales causas de accidentes de tránsito. Detectar señales tempranas de fatiga (ojos cerrándose, bostezos) puede prevenir incidentes si el sistema alerta al conductor a tiempo.

Este proyecto implementa un sistema completo de detección de fatiga: el frontend captura frames de la cámara web cada 800 ms y los envía a una API FastAPI que aplica preprocesamiento con OpenCV (detección de cara, recorte de ojos, CLAHE) y clasifica el estado con un modelo MobileNetV2 entrenado con Transfer Learning.

Proyecto desarrollado en equipo para la materia de Visión por Computadora y Machine Learning.

### Qué resuelve

| Reto | Solución implementada |
|---|---|
| Detección robusta de señales de fatiga | Estrategia de doble predicción: región de ojos + cara completa |
| Diferencia entre imágenes de entrenamiento y uso real | Datasets contienen ojos recortados, la cámara captura cara completa |
| Preprocesamiento de imágenes en producción | Pipeline OpenCV: Haar Cascade → CLAHE → resize 224×224 → normalización |
| Compatibilidad con Mac M3 | Docker con `--platform=linux/amd64` y `tensorflow-cpu` dentro del contenedor |
| Alertas visuales en tiempo real | Frontend con captura automática cada 800 ms y feedback codificado por color |

### Capturas

**Detección de ojo abierto**

El conductor está alerta. La estrategia de doble predicción elige la región de ojos recortada porque ofrece mayor confianza.

![Ojo abierto](capturas/01-frontend-ojo-abierto.png)

**Detección de bostezo**

El sistema identifica un bostezo (98.9% de confianza) y muestra una alerta de atención.

![Bostezo detectado](capturas/02-frontend-bostezo.png)

**Documentación interactiva de la API (Swagger UI)**

![Swagger UI](capturas/03-swagger.png)

### Arquitectura

```
[ Cámara web del conductor ]
       |
       | Frames cada 800 ms (JPEG)
       v
[ Frontend HTML5 ]
       |
       | multipart/form-data
       v
[ Docker Puerto 8000:8000 ]
       |
       v
[ FastAPI + Uvicorn ]
       |
       | Detección de cara (Haar Cascade)
       | Recorte región de ojos
       | CLAHE + resize + normalización
       v
[ Modelo MobileNetV2 (modelo_fatiga.keras) ]
       |
       | 4 probabilidades (softmax)
       v
[ JSON Response: clase + confianza + nivel_alerta + modo ]
       |
       v
[ Frontend: alerta visual + barras + historial ]
```

### Stack tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.12 |
| Modelo IA | TensorFlow / Keras 2.16 |
| Arquitectura | MobileNetV2 + Transfer Learning |
| Visión por computadora | OpenCV 4.11 |
| API | FastAPI 0.115 + Uvicorn |
| Contenedor | Docker con plataforma `linux/amd64` |
| Frontend | HTML5 + JavaScript + Canvas API |

### Estructura del proyecto

```
deteccion-fatiga-conductores/
├── api/
│   ├── Dockerfile
│   ├── main.py                          # API FastAPI
│   ├── requirements.txt
│   ├── frontend/
│   │   └── index.html                   # Interfaz con cámara
│   └── model/
│       ├── modelo_fatiga.keras          # Modelo entrenado
│       ├── modelo_fatiga_metadata.json
│       └── cascades/
│           ├── haarcascade_frontalface_default.xml
│           └── haarcascade_eye.xml
├── docs/
│   ├── documentacion-tecnica.pdf
│   └── guia-despliegue.md
├── capturas/
├── LICENSE
└── README.md
```

### Instalación y uso

**Requisitos**

- Python 3.12 (recomendado: instalar con `pyenv`)
- Docker Desktop (para la variante con contenedor)

**Opción A — Correr local con Uvicorn**

1. Clonar el repositorio

   ```bash
   git clone https://github.com/KPVE2004/deteccion-fatiga-conductores.git
   cd deteccion-fatiga-conductores
   ```

2. Crear y activar entorno virtual

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Instalar dependencias

   ```bash
   pip install -r api/requirements.txt
   ```

4. Levantar la API

   ```bash
   cd api
   uvicorn main:app --reload
   ```

5. Abrir el frontend

   Abre `http://localhost:8000/` en el navegador. La cámara te pedirá permiso.

**Opción B — Correr con Docker**

```bash
cd api
docker buildx build --platform linux/amd64 -t img-fatiga .
docker run -d --name srv-fatiga -p 8000:8000 img-fatiga:latest
```

Verificar que corre:

```bash
docker ps
```

Abrir `http://localhost:8000/` para la interfaz, `http://localhost:8000/docs` para Swagger.

### Endpoints de la API

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/` | Sirve el frontend HTML |
| GET | `/health` | Estado del modelo y sistema |
| POST | `/predict` | Recibe imagen, devuelve clase + confianza + nivel de alerta |
| GET | `/modelo/info` | Información técnica del modelo |

Ejemplo de respuesta de `/predict`:

```json
{
  "clase": "ojo_cerrado",
  "probabilidad": 0.9923,
  "nivel_alerta": "PELIGRO",
  "color_alerta": "#F44336",
  "mensaje_alerta": "Ojos cerrados - posible fatiga",
  "todas_las_clases": {
    "ojo_abierto": 0.0041,
    "ojo_cerrado": 0.9923,
    "bostezo": 0.0022,
    "no_bostezo": 0.0014
  },
  "tiempo_inferencia_ms": 43.7,
  "modo_preprocesamiento": "ojos_recortados",
  "cara_detectada": true
}
```

**Niveles de alerta:**

| Clase | Nivel | Color | Acción |
|---|---|---|---|
| ojo_abierto | NORMAL | Verde | Sin acción |
| no_bostezo | NORMAL | Verde | Sin acción |
| bostezo | ATENCIÓN | Naranja | Sugerir descanso |
| ojo_cerrado | PELIGRO | Rojo | Alarma inmediata |

### Estrategia de doble predicción

El modelo se entrenó con tres datasets que contienen imágenes de ojos recortados. Sin embargo, la cámara web captura el rostro completo del conductor. Para cerrar esa brecha, se implementó una estrategia de doble predicción:

1. **Predicción A — Región de ojos:** se recorta la mitad superior de la cara detectada y se busca un ojo individual. Especializada en `ojo_abierto` / `ojo_cerrado`.
2. **Predicción B — Cara completa:** se recorta toda la cara. Especializada en `bostezo` / `no_bostezo`.
3. **Decisión final:** se compara la confianza máxima de A en clases de ojos contra la confianza máxima de B en clases de boca. Gana la mayor.
4. **Fallback:** si no se detecta cara, se predice directamente con la imagen completa.

El campo `modo_preprocesamiento` en la respuesta indica cuál ruta se tomó.

### Dataset y entrenamiento

**Datasets utilizados** (todos públicos en Kaggle):

| Dataset | Imágenes | Clases originales |
|---|---|---|
| MRL Eye Dataset | 4,000 | Open_Eyes, Closed_Eyes |
| Yawn Eye Dataset | 2,900 | yawn, no_yawn, Open, Closed |
| Drowsiness / CEW | 47,992 | open_eye, closed_eye |
| **Total** | **54,892** | **4 clases unificadas** |

**Preprocesamiento aplicado:**

- Curación: verificación MD5, eliminación de 8 imágenes corruptas (0.02%).
- CLAHE en espacio LAB: mejora de contraste +48% (MRL) y +60.5% (CEW).
- Resize a 224×224 px.
- Normalización a [0, 1].
- División estratificada 70/15/15 (semilla 42).

**Entrenamiento (MobileNetV2 + Transfer Learning):**

- Fase 1 — Feature Extraction (10 epochs, backbone congelado): 97.15% val accuracy.
- Fase 2 — Fine-Tuning (15 epochs, últimas 30 capas descongeladas, LR=1e-5): 99.08% val accuracy.
- Test accuracy final: **99.02%**.
- Modelo: 3,049,284 parámetros, 29.9 MB.

### Roles del equipo

Proyecto desarrollado en equipo por tres integrantes.

**Ángel de Jesús Canela Figueroa**

- Implementación de la estrategia de doble predicción en el backend.
- Desarrollo del pipeline de preprocesamiento (Haar Cascade, CLAHE, recorte inteligente).
- Configuración del Dockerfile para compatibilidad con Mac M3 (Apple Silicon).
- Preparación del despliegue en Google Cloud Run.
- Documentación técnica y guía de despliegue.

**Isamar Chávez Díaz**

- **Entrenamiento, fase 1:** Transfer Learning con MobileNetV2 y backbone congelado (10 epochs), con 97.15 % de precisión en validación.
- **Preparación de los datasets:** integración de tres datasets públicos de Kaggle (54,892 imágenes) en 4 clases unificadas.
- **Curación y preprocesamiento de datos:** verificación MD5 (8 imágenes corruptas eliminadas), mejora de contraste con CLAHE en espacio LAB, redimensionado a 224×224, normalización y división estratificada 70/15/15.
- **Entrenamiento del modelo** (en conjunto): participación en el entrenamiento de MobileNetV2 con Transfer Learning y Fine-Tuning.
- **Interfaz web** (en conjunto): participación en el desarrollo del frontend con cámara, alertas visuales y barras de probabilidad.
- **Pruebas del sistema:** validación de punta a punta de la cámara, la API y la alerta visual.

**Katia Paola Victoria Escalera**

- **Entrenamiento, fase 2:** Fine-Tuning descongelando las últimas 30 capas con learning rate de 1e-5 (15 epochs), con 99.08 % de precisión en validación.
- **Evaluación del modelo:** medición sobre el conjunto de test, con una precisión final de 99.02 %.
- **Interfaz web:** desarrollo del frontend en HTML5, JavaScript y Canvas API, con acceso a la cámara web y captura automática de frames cada 800 ms.
- **Preparación de los datasets** (en conjunto): integración de tres datasets públicos de Kaggle (54,892 imágenes) en 4 clases unificadas.
- **Curación y preprocesamiento de datos** (en conjunto): verificación MD5, mejora de contraste con CLAHE, redimensionado a 224×224, normalización y división estratificada 70/15/15.
- **Pruebas del sistema** (en conjunto): validación de punta a punta de la cámara, la API y la alerta visual.

### Documentación adicional

- [Documentación técnica](docs/documentacion-tecnica.pdf) — informe completo del proyecto.
- [Guía de despliegue](docs/guia-despliegue.md) — proceso de despliegue en Google Cloud Run.

### Licencia

MIT — ver [LICENSE](LICENSE).

---

## Author · Autora

**Katia Paola Victoria Escalera**

[Portfolio](https://katia-portfolio-six.vercel.app/) · [GitHub](https://github.com/KPVE2004) · [LinkedIn](https://www.linkedin.com/in/katia-victoria) · [Email](mailto:kp_victoria08@hotmail.com)

**Team / Equipo:** Ángel de Jesús Canela Figueroa · Isamar Chávez Díaz · Katia Paola Victoria Escalera  
**Institution / Institución:** Instituto Tecnológico de Jiquilpan (TecNM)  
**Course / Materia:** Computer Vision and Machine Learning / Visión por Computadora y Machine Learning
