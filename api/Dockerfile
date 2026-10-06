FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app

# Dependencias del sistema para OpenCV y TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Dentro del contenedor (linux/amd64) se usa tensorflow-cpu.
# En el entorno local Mac M3 (ARM64) se usa tensorflow nativo
# desde requirements.txt.
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.6 \
    tensorflow-cpu==2.16.2 \
    numpy \
    pillow \
    opencv-python-headless \
    python-multipart \
    pydantic==2.7.4

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
