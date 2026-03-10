# Sign Language MNIST Classifier 🤟 / Clasificador de Lenguaje de Señas

*Read this in [English](https://www.google.com/search?q=%23english-version) | Leer en [Español*](https://www.google.com/search?q=%23versi%C3%B3n-en-espa%C3%B1ol)

---

## English Version

### 📌 Project Overview

This project focuses on building a machine learning pipeline to classify American Sign Language (ASL) hand gestures using the **Sign Language MNIST** dataset. The goal is to translate 24 static alphabet gestures (excluding 'J' and 'Z') from 28x28 grayscale images into their corresponding text labels.

The project heavily emphasizes efficiency, exploring dimensionality reduction techniques to create a lightweight model suitable for real-time or edge-device deployment without sacrificing predictive power.

### 📊 Dataset Download Instructions

Due to storage limits and best practices, the dataset is not included in this repository. To run this project, you must download the data manually:

1. Visit the [Sign Language MNIST dataset on Kaggle](https://www.google.com/search?q=https://www.kaggle.com/datasets/datamunge/sign-language-mnist).
2. Download the archive and extract it.
3. Place the following files into the `input/` directory of this project:
* `sign_mnist_train.csv`
* `sign_mnist_test.csv`



### 🏗️ Project Structure

```text
.
├── input/               # Raw data goes here (gitignored)
├── models/              # Serialized trained models (.bin)
├── notebooks/           # Jupyter notebooks with detailed analysis
│   ├── 01_EDA.ipynb
│   ├── 02_model-selection.ipynb
│   └── 03_evaluation-and-report.ipynb
├── report/              # Output predictions 
├── src/                 # Python source code for training and prediction
│   ├── config.py
│   ├── create_folds.py  # Stratified K-Fold generator
│   ├── train.py         # Cross-validation training script
│   ├── train_final_model.py
│   └── predict_test.py
├── environment.yml      # Conda environment file
└── requirements.txt     # Pip requirements file

```

### 🧠 Methodology & Results

1. **Exploratory Data Analysis (EDA) & PCA:**
* Images display high spatial redundancy. Backgrounds are mostly static, with the variance concentrated around the fingers.
* Principal Component Analysis (PCA) was used to compress the 784-pixel space. Retaining just **35 components** explained 85% of the variance while perfectly preserving morphological integrity.


2. **Model Selection:**
* Evaluated 5 baseline and non-linear models (Logistic Regression, k-NN, SVM, Random Forest, MLP) using Stratified 5-Fold Cross-Validation.
* **Selected Model:** Multi-Layer Perceptron (MLP) with a hidden layer of `(128,)`. It achieved a perfect 1.0 F1-score during cross-validation while maintaining an extremely small memory footprint (~0.48 MB) and high inference speed (<0.2 ms per sample).


3. **Final Evaluation:**
* Evaluated on an unseen hold-out test set, the final pipeline (Scaler + PCA + MLP) achieved an **Average Macro F1-Score of 0.80**.
* Error analysis revealed that the model struggles primarily with highly similar morphological gestures compressed into the latent space (e.g., K vs U, T vs X).



### 🚀 Setup & Execution

**1. Environment Setup**
You can use `conda` or `pip` to install the dependencies:

```bash
# Using Conda
conda env create -f environment.yml
conda activate signlang

# Using Pip
pip install -r requirements.txt

```

**2. Running the Pipeline**
*Generate stratified folds for validation:*

```bash
python -m src.create_folds

```

*Train the final production model on the entire training set:*

```bash
python -m src.train_final_model --model mlp

```

*Run predictions on the test set and evaluate:*

```bash
python -m src.predict_test --model_path models/final_mlp_n35.bin

```

---

## Versión en Español

### 📌 Resumen del Proyecto

Este proyecto se centra en la construcción de un pipeline de Machine Learning para clasificar gestos del Alfabeto Manual Americano (ASL) utilizando el dataset **Sign Language MNIST**. El objetivo es traducir 24 gestos estáticos del alfabeto (excluyendo la 'J' y la 'Z') a partir de imágenes en escala de grises de 28x28 píxeles.

El proyecto pone un fuerte énfasis en la eficiencia operativa, explorando técnicas de reducción de dimensionalidad para crear un modelo ligero, ideal para implementaciones en tiempo real o en dispositivos de bajos recursos, sin sacrificar la capacidad predictiva.

### 📊 Instrucciones para Descargar los Datos

Debido a límites de almacenamiento y buenas prácticas, el conjunto de datos no está incluido en este repositorio. Para ejecutar este proyecto, debes descargar los datos manualmente:

1. Visita el [dataset Sign Language MNIST en Kaggle](https://www.google.com/search?q=https://www.kaggle.com/datasets/datamunge/sign-language-mnist).
2. Descarga el archivo y extráelo.
3. Ubica los siguientes archivos dentro de la carpeta `input/` de este proyecto:
* `sign_mnist_train.csv`
* `sign_mnist_test.csv`



### 🏗️ Estructura del Proyecto

```text
.
├── input/               # Datos crudos (ignorado en git)
├── models/              # Modelos entrenados serializados (.bin)
├── notebooks/           # Notebooks de Jupyter con análisis detallado
│   ├── 01_EDA.ipynb
│   ├── 02_model-selection.ipynb
│   └── 03_evaluation-and-report.ipynb
├── report/              # Predicciones de salida
├── src/                 # Código fuente en Python
│   ├── config.py
│   ├── create_folds.py  # Generador de particiones Stratified K-Fold
│   ├── train.py         # Script de entrenamiento para validación cruzada
│   ├── train_final_model.py
│   └── predict_test.py
├── environment.yml      # Archivo de entorno Conda
└── requirements.txt     # Archivo de dependencias Pip

```

### 🧠 Metodología y Resultados

1. **Análisis Exploratorio (EDA) y PCA:**
* Las imágenes presentan una alta redundancia espacial. Los fondos son estáticos, concentrándose la varianza en los dedos.
* Se utilizó Análisis de Componentes Principales (PCA) para comprimir el espacio de 784 píxeles. Retener solo **35 componentes** explicó el 85% de la varianza, preservando perfectamente la integridad morfológica.


2. **Selección de Modelo:**
* Se evaluaron 5 arquitecturas (Regresión Logística, k-NN, SVM, Random Forest, MLP) usando Stratified 5-Fold Cross-Validation.
* **Modelo Seleccionado:** Perceptrón Multicapa (MLP) con una capa oculta de `(128,)`. Alcanzó un F1-Score perfecto de 1.0 en validación, manteniendo una huella de memoria extremadamente baja (~0.48 MB) y alta velocidad de inferencia (<0.2 ms por muestra).


3. **Evaluación Final:**
* Al evaluarse en el conjunto de prueba nunca visto (Hold-out), el pipeline final (Scaler + PCA + MLP) logró un **F1-Score Macro Promedio de 0.80**.
* El análisis de errores reveló confusiones en gestos con morfologías muy similares que se solapan en el espacio latente (ej. K vs U, T vs X).



### 🚀 Configuración y Ejecución

**1. Preparar el Entorno**
Puedes usar `conda` o `pip` para instalar las dependencias:

```bash
# Usando Conda
conda env create -f environment.yml
conda activate signlang

# Usando Pip
pip install -r requirements.txt

```

**2. Ejecutar el Pipeline**
*Generar los particionamientos (folds) para la validación:*

```bash
python -m src.create_folds

```

*Entrenar el modelo final de producción usando todos los datos de entrenamiento:*

```bash
python -m src.train_final_model --model mlp

```

*Realizar predicciones sobre el conjunto de prueba y evaluar:*

```bash
python -m src.predict_test --model_path models/final_mlp_n35.bin

```


