# AlphaDeforest 🛰️🌲

[![Python Tests](https://github.com/liturriago/alphadeforest_project/actions/workflows/python-tests.yml/badge.svg?branch=main)](https://github.com/liturriago/alphadeforest_project/actions/workflows/python-tests.yml)

**AlphaDeforest** es un framework de Deep Learning diseñado para la detección de cambios en la cobertura forestal mediante el análisis de secuencias temporales de tiles satelitales. Este proyecto aplica principios de **ML Engineering y MLOps** para garantizar la reproducibilidad y escalabilidad en la investigación académica.

## 🔬 Descripción del Proyecto
El sistema utiliza una arquitectura híbrida para capturar dependencias espaciales y temporales:
1. **Extracción Espacial:** Un *Convolutional Autoencoder* (CAE) que reduce la dimensionalidad de los tiles preservando características críticas.
2. **Dinámica Temporal:** Una *Memory Network* con mecanismo de atención y celdas LSTM para predecir y detectar anomalías en la secuencia temporal.

Este trabajo se enfoca en la ciencia de datos aplicada a la visión por computador, alineado con las líneas de investigación del Doctorado en Automática.

## 🛠️ Estructura del Repositorio
```text
alphadeforest_project/
├── configs/          # Configuraciones validadas con Pydantic (.yaml)
├── src/              # Código fuente modular
│   └── alphadeforest/
│       ├── data/     # Pipeline de datos (WebDataset)
│       ├── models/   # Arquitecturas (CAE, LSTM, Attention)
│       └── engine/   # Motor de entrenamiento (Trainer)
├── tests/            # Pruebas unitarias con pytest
└── notebooks/        # Experimentos y análisis visual

```

## 🚀 Instalación

Para configurar el entorno de desarrollo y utilizar el paquete de forma local:

```bash
git clone [https://github.com/liturriago/alphadeforest_project.git](https://github.com/liturriago/alphadeforest_project.git)
cd alphadeforest_project
pip install -e .

```

## 📊 Uso

### Entrenamiento vía CLI

Puedes lanzar experimentos utilizando archivos de configuración para asegurar la reproducibilidad:

```bash
python scripts/train.py --config configs/train_config.yaml

```

### Investigación en Notebooks

El paquete está diseñado para ser importado fácilmente en entornos de Jupyter para experimentación rápida:

```python
from alphadeforest.models.alpha_deforest import AlphaDeforest
# Carga de modelos y análisis de resultados...

```

## ✅ Calidad y CI/CD

Este repositorio utiliza **GitHub Actions** para ejecutar pruebas automáticas en cada `push` o `pull_request`, asegurando que las dimensiones de los tensores y la lógica del modelo se mantengan consistentes tras cada cambio.

## 🎓 Créditos

Desarrollado por **Lucas Iturriago**, estudiante de Doctorado en Automática en la **Universidad Nacional de Colombia, sede Manizales**.