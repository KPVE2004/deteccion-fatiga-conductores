# Sistema de Detección de Fatiga en Conductores

## Guía de Despliegue en Google Cloud Run

MobileNetV2 + Transfer Learning · Accuracy 99.02% · 4 clases

Proyecto: api2026-497116 · Mayo 2026

---
Esta guía documenta el proceso completo para desplegar una aplicación web fullstack (FastAPI + Frontend HTML) con un modelo de detección de fatiga en conductores usando CNN con MobileNetV2 + Transfer Learning en Google Cloud Run.

Autores: Katia Paola Victoria Escalera, Isamar Chávez Díaz, Angel de Jesús Canela Figueroa

Materia: Visión Por Computadora Y Machine Learning — Mayo 2026

---

## Arquitectura del Proyecto

La aplicación consta de los siguientes archivos y carpetas:

| **Archivo / Carpeta** | **Descripción** |
|----|----|
| Dockerfile | Imagen Docker basada en python:3.12-slim con dependencias del sistema para OpenCV y TensorFlow |
| main.py | API REST con FastAPI: endpoints GET / (frontend), GET /health, POST /predict y GET /modelo/info |
| requirements.txt | Dependencias: FastAPI, Uvicorn, tensorflow-cpu, numpy, pillow, opencv-python-headless, pydantic |
| frontend/index.html | Frontend con cámara en tiempo real, visualización de resultados y historial de predicciones |
| model/modelo_fatiga.keras | Modelo MobileNetV2 con Transfer Learning entrenado para 4 clases de fatiga |
| model/modelo_fatiga_metadata.json | Metadatos del modelo: clases, accuracy y arquitectura |
| model/cascades/ | Archivos Haar Cascade para detección de cara y ojos (se descargan automáticamente si no existen) |

### El modelo clasifica imágenes en 4 clases

- ojo_abierto — conductor alerta (NORMAL)

- ojo_cerrado — posible fatiga (PELIGRO)

- bostezo — tomar descanso (ATENCIÓN)

- no_bostezo — conductor alerta (NORMAL)

### Estrategia de doble predicción

- Predicción A: región de ojos recortada → especializada en ojo_abierto / ojo_cerrado

- Predicción B: cara completa → especializada en bostezo / no_bostezo

- Se elige la predicción con mayor confianza en su clase ganadora

- Si no se detecta cara → predicción directa con imagen completa

---

## Paso 1. Preparar los archivos

Modificar main.py para agregar el endpoint que sirve el frontend. Se agregaron las siguientes líneas:

```
from fastapi.responses import FileResponse

@app.get('/', response_class=FileResponse)

def frontend():

return FileResponse('/app/frontend/index.html', media_type='text/html')
```

También se movió el health check al endpoint /health para no conflictar con el frontend:

```
@app.get('/health')

def health_check():

return { 'mensaje': 'Sistema activo', 'modelo_cargado': modelo is not None, ... }
```

Estructura final de la carpeta api-fatiga:

```
api-fatiga/

├ Dockerfile

├ main.py

├ requirements.txt

├ frontend/

│ └ index.html

└ model/

├ modelo_fatiga.keras

├ modelo_fatiga_metadata.json

└ cascades/

├ haarcascade_frontalface_default.xml

└ haarcascade_eye.xml
```

## Paso 2. Subir archivos a Cloud Shell

En la consola de Google Cloud, abrir Cloud Shell y subir la carpeta api-fatiga usando el ícono de opciones (⋮) → Upload → Seleccionar carpeta.

Verificar que todos los archivos llegaron correctamente:

```
cd ~/api-fatiga

ls -la

ls frontend/

ls model/

ls model/cascades/
```

## Paso 3. Crear repositorio en Artifact Registry

```
gcloud artifacts repositories create repo-fatiga \\

--repository-format=docker \\

--location=northamerica-south1 \\

--description="API Detección de Fatiga en Conductores"
```

Verificar que se creó:

```
gcloud artifacts repositories list --location=northamerica-south1
```

## Paso 4. Build y push con Cloud Build

Se usa gcloud builds submit en lugar de docker push porque Cloud Shell bloquea las conexiones TCP directas a Artifact Registry:

```
gcloud builds submit \\

--tag northamerica-south1-docker.pkg.dev/api2026-497116/repo-fatiga/img-fatiga:v1 \\

.
```

Al finalizar debe mostrar STATUS: SUCCESS con la imagen publicada.

## Paso 5. Desplegar en Cloud Run

```
gcloud run deploy api-fatiga \\

--image northamerica-south1-docker.pkg.dev/api2026-497116/repo-fatiga/img-fatiga:v1 \\

--platform managed \\

--region northamerica-south1 \\

--port 8080 \\

--allow-unauthenticated \\

--memory 2Gi \\

--cpu 2 \\

--timeout 300
```

## Parámetros clave:

- --memory 2Gi: TensorFlow + OpenCV requieren al menos 2 GB de RAM

- --cpu 2: mejora el tiempo de arranque y la inferencia

- --timeout 300: 5 minutos para que el contenedor inicie

- --allow-unauthenticated: acceso público al frontend y al API

## Paso 6. Corregir URL del API en el frontend

El index.html tenía la URL apuntando a localhost. Corregirla con sed directamente en Cloud Shell:

```
sed -i "s\|http://localhost:8000\|https://api-fatiga-551124884034.northamerica-south1.run.app\|g" frontend/index.html
```

Verificar el cambio:

```
grep "const API" frontend/index.html
```

Build y deploy con versión 2:

```
gcloud builds submit \\

--tag northamerica-south1-docker.pkg.dev/api2026-497116/repo-fatiga/img-fatiga:v2 \\

.

gcloud run deploy api-fatiga \\

--image northamerica-south1-docker.pkg.dev/api2026-497116/repo-fatiga/img-fatiga:v2 \\

--platform managed \\

--region northamerica-south1 \\

--port 8080 \\

--allow-unauthenticated \\

--memory 2Gi \\

--cpu 2 \\

--timeout 300
```

## Paso 7. Ajustes al frontend (v3)

Se eliminaron dos elementos del frontend que no eran necesarios en producción:

- Indicador de estado de la API (#api-status) — mostraba 'API no disponible' porque el endpoint GET / ya no retorna JSON sino HTML

- Botón 'Subir imagen' y su input file — la aplicación solo usa cámara en tiempo real

Reemplazar el archivo y hacer build y deploy v3:

```
gcloud builds submit \\

--tag northamerica-south1-docker.pkg.dev/api2026-497116/repo-fatiga/img-fatiga:v3 \\

.

gcloud run deploy api-fatiga \\

--image northamerica-south1-docker.pkg.dev/api2026-497116/repo-fatiga/img-fatiga:v3 \\

--platform managed \\

--region northamerica-south1 \\

--port 8080 \\

--allow-unauthenticated \\

--memory 2Gi \\

--cpu 2 \\

--timeout 300
```

## Paso 8. Verificar el despliegue

Abrir la URL en el navegador:

```
https://api-fatiga-551124884034.northamerica-south1.run.app
```

Verificar el health check:

```
curl https://api-fatiga-551124884034.northamerica-south1.run.app/health
```

Ver info del modelo:

```
curl https://api-fatiga-551124884034.northamerica-south1.run.app/modelo/info
```

---

## Problemas encontrados y soluciones

| **Problema** | **Causa** | **Solución** |
|----|----|----|
| docker push connection refused | Cloud Shell bloquea conexiones TCP directas a Artifact Registry | Usar gcloud builds submit |
| Container failed to start (512 MiB) | TensorFlow + OpenCV superan el límite de memoria por defecto | Aumentar memoria a 2Gi |
| URL del API apunta a localhost | index.html tenía http://localhost:8000 hardcodeado | Corregir con sed o editar manualmente y hacer nuevo build |
| Indicador de API muestra 'no disponible' | GET / retorna HTML en lugar de JSON tras agregar el frontend | Cambiar verificarAPI() a GET /health o eliminar el indicador |

---

## Notas adicionales

- La URL del servicio tiene el formato: https://SERVICIO-PROJECT_NUMBER.REGION.run.app

- Cloud Run escala automáticamente a cero instancias cuando no hay tráfico (cold start ~10-15s con TensorFlow)

- La documentación interactiva del API está disponible en /docs (generada automáticamente por FastAPI)

- Los archivos Haar Cascade se descargan automáticamente desde GitHub al iniciar el contenedor si no existen

- Para actualizar el modelo o el código, repetir pasos 4 y 5 con una nueva versión (v4, v5, etc.)

- El frontend y el backend se sirven desde el mismo contenedor en el puerto 8080

- La cámara web requiere que el navegador tenga permisos de acceso a la cámara
