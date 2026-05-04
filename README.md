
## Descripción
Sistema de clasificación automática de frutas mediante inteligencia artificial y visión por computadora. El modelo fue entrenado usando **ResNet18 con transfer learning** y desplegado en una aplicación capaz de realizar **inferencia en tiempo real mediante cámara**.

El proyecto se enfoca en tres especies:
- Tamarillo (tomate de árbol)
- Naranjilla
- Pitahaya

Este trabajo contribuye a la **agricultura de precisión** y a la **automatización de procesos postcosecha**.

---

## Objetivos
- Desarrollar un modelo de clasificación de imágenes robusto.
- Evaluar el rendimiento usando CPU vs GPU.
- Implementar una aplicación funcional en tiempo real.
- Validar el uso de deep learning en entornos agrícolas.

---

## Metodología

### 1. Dataset
- Imágenes etiquetadas por clase
- Preprocesamiento y limpieza de datos

### 2. Modelo
- Arquitectura: **ResNet18**
- Técnica: **Transfer Learning**
- Framework: **PyTorch**

### 3. Entrenamiento
- Entrenamiento supervisado
- Evaluación por épocas
- Comparación CPU vs GPU (AWS)

### 4. Despliegue
- Aplicación en tiempo real con:
  - OpenCV
  - Streamlit (opcional)

---

## Instalación

Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/ClasificadorFrutasAWS.git
cd ClasificadorFrutasAWS