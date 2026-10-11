# 🏠 Predicción y Valuación Inmobiliaria Masiva con Big Data & Machine Learning
### Taller Práctico 01 — Metodología CRISP-DM sobre el Mercado Residencial de Inglaterra y Gales (2011–2019)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bernal-XFK/BigData-Housing-Price-Prediction-UK/blob/main/Taller_BigData_Prediccion_Viviendas_UK.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT%20Academic-green.svg)](LICENSE)
[![Institution](https://img.shields.io/badge/Universidad-CESMAG-red.svg)](https://www.unicesmag.edu.co/)

---

## 📌 Resumen Ejecutivo del Proyecto

Este proyecto aborda la modelación predictiva del precio de venta de propiedades residenciales en **Inglaterra y Gales**, a partir de la integración masiva de dos de las bases de datos abiertas más representativas del Reino Unido: el registro catastral de transacciones (**HM Land Registry**) y los certificados de eficiencia energética (**Energy Performance Certificates - EPC**), cubriendo el período comprendido entre **2011 y 2019**.

El desarrollo sigue rigurosamente el estándar metodológico **CRISP-DM** (Cross-Industry Standard Process for Data Mining), cubriendo desde la comprensión del dominio de negocio y la ingeniería de datos en entornos con restricciones severas de memoria, hasta la evaluación comparativa entre modelos lineales regularizados y ensambles modernos de árboles de decisión.

* **Autor:** Jose Alejandro Bernal Figueroa
* **Institución:** Universidad CESMAG
* **Asignatura:** Aprendizaje Automático / Big Data
* **Entorno:** Google Colab / Python

---

## 🔬 Metodología CRISP-DM Implementada

```
  ┌─────────────────────────┐
  │ 1. Negocio              │  Definición de objetivos, métricas clave (MAE < £50k)
  └───────────┬─────────────┘
              ▼
  ┌─────────────────────────┐
  │ 2. Datos y EDA          │  Muestreo skiprows (650k filas), asimetría de precio, nulos
  └───────────┬─────────────┘
              ▼
  ┌─────────────────────────┐
  │ 3. Preparación          │  Log-Transform, One-Hot Encoding geográfico, año continuo
  └───────────┬─────────────┘
              ▼
  ┌─────────────────────────┐
  │ 4. Modelado             │  OLS, Ridge, Lasso, ElasticNet, Random Forest, HistGradientBoosting
  └───────────┬─────────────┘
              ▼
  ┌─────────────────────────┐
  │ 5. Evaluación           │  Test set independiente, métricas RMSE, MAE, MAPE y R²
  └───────────┬─────────────┘
              ▼
  ┌─────────────────────────┐
  │ 6. Reflexión y Negocio  │  Importancia de variables, inflación, límites de los datos
  └─────────────────────────┘
```

---

## 💡 Desafíos de Big Data y Soluciones de Ingeniería

1. **Gestión de Memoria en Google Colab (5.7M de filas):**
   * El archivo fuente supera los 5.7 millones de registros y 105 columnas (>3 GB), desbordando la memoria RAM gratuita de Colab.
   * **Solución:** Implementación de un muestreo pseudoaleatorio eficiente con la técnica `skiprows` en `pd.read_csv()`, seleccionando una muestra representativa de **650,000 registros** sin cargar el archivo completo en memoria y filtrando columnas a una selección parsimoniosa de 25 variables candidatas.

2. **Asimetría Positiva Severa del Precio (Data Skewness):**
   * El precio oscila entre £10,000 y propiedades de lujo de £2,000,000. Entrenar directamente con error cuadrático hacía que los modelos se sesgaran hacia las mansiones atípicas, degradando la predicción de las viviendas comunes.
   * **Solución:** Envoltorio sistemático con `TransformedTargetRegressor` aplicando $\log(1 + y)$ antes del entrenamiento y $\exp(y) - 1$ en la inferencia, normalizando los residuos y estabilizando el cálculo.

3. **La Inflación y el Factor Temporal (`transaction_year`):**
   * El muestreo abarca 8 años de ventas. Asumir linealidad estática ignoraba la inflación inmobiliaria acumulada en Inglaterra entre 2011 y 2019.
   * **Solución:** Extracción del año continuo de transferencia ($año + mes/12$) desde `dateoftransfer`, permitiendo a los modelos ajustar dinámicamente la tendencia macroeconómica y desbloqueando el salto de $R^2$ de ~0.60 a más de **0.78–0.80**.

4. **Alta Cardinalidad Espacial y Calibración para Árboles:**
   * La variable `district` contiene cientos de divisiones territoriales. Para modelos lineales, el *One-Hot Encoding* asignó primas territoriales individuales muy precisas.
   * Para los árboles de decisión (que sufren en espacios dispersos de 500 columnas binarias), se evaluó y demostró la superioridad de **`HistGradientBoostingRegressor`** mediante discretización en histogramas continuos (bins) frente al Random Forest clásico.

---

## 📊 Tabla Comparativa de Modelos Evaluados

Evaluación calculada sobre el **conjunto de prueba (Test set)**, completamente independiente del entrenamiento y ajuste de hiperparámetros:

| Modelo | Familia | Regularización | RMSE (£) | MAE (£) | MAPE (%) | $R^2$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **OLS (Baseline)** | Lineal | Ninguna | £88,195 | £51,671 | 22.1% | 0.8029 |
| **Ridge** | Lineal | L2 ($\alpha=0.1$) | £88,195 | £51,671 | 22.1% | 0.8029 |
| **Lasso** | Lineal | L1 ($\alpha=10^{-5}$) | £88,259 | £51,674 | 22.2% | 0.8026 |
| **ElasticNet** | Lineal | L1 + L2 | £88,201 | £51,673 | 22.1% | 0.8028 |
| **Random Forest** | Ensamble Bagging | 50 árboles ($max\_depth=18$) | £92,410 | £53,820 | 23.4% | 0.7850 |
| **HistGradientBoosting** | Ensamble Boosting | 250 iteraciones + Histograma | **£84,350** | **£50,480** | **20.8%** | **0.8140** |

> **Observación Técnica:** En datasets con cientos de miles de observaciones y variables escaladas, la diferencia entre OLS y modelos regularizados (Ridge/Lasso) es mínima debido a que la gran cantidad de datos estabiliza de por sí la varianza de los coeficientes. El verdadero salto no lineal lo aporta `HistGradientBoosting`, que logra el error absoluto medio más bajo y la mayor varianza explicada.

---

## 🏙️ Variables Más Influyentes y Criterio de Negocio

El análisis de coeficientes e importancia relativa (Gini / reducción de varianza) arroja conclusiones coherentes con el sector inmobiliario británico:

1. **Área Total Construida (`tfarea`):** Explica más del 40% de la importancia relativa en modelos de ensamble. El metro cuadrado es el fundamento cuantitativo de cualquier tasación.
2. **Ubicación Geográfica (`district` / `county`):** Los distritos del Gran Londres (**Kensington and Chelsea**, **City of Westminster**, **Camden**) exhiben los coeficientes positivos más altos del país, multiplicando el precio base frente a regiones del norte de Inglaterra.
3. **Año de Transacción (`transaction_year`):** Refleja la curva de inflación y apreciación sostenida del mercado británico entre 2011 y 2019.
4. **Tipología de Propiedad (`propertytype`):** Primas marcadas para casas independientes (*Detached*) sobre casas adosadas (*Terraced*) y apartamentos (*Flats*), asociadas a la propiedad del terreno y privacidad.
5. **Régimen de Tenencia (`duration`):** Las propiedades de dominio absoluto (*Freehold*) son consistentemente más valoradas que las de arrendamiento a largo plazo (*Leasehold*).

---

## 🧱 El Techo de los Datos (Data Ceiling)

El modelo logró explicar más del **81% de la varianza** con un error absoluto medio cercano a £50,000 en un período de 8 años en todo un país. Acercarse a este límite pone en evidencia que el margen de error residual no se debe a deficiencias del algoritmo, sino a características no observables en el dataset disponible:
* Estado de conservación y remodelaciones interiores (acabados de cocina, baños, antigüedad de instalaciones).
* Micro-ubicación y factores de entorno (cercanía a estaciones de metro/tren, parques, colegios de calificación Ofsted sobresaliente).
* Orientación solar, iluminación natural, vistas y ruido ambiental.

---

## 🚀 Cómo Ejecutar el Proyecto

### Opción 1: Google Colab (Recomendada)
Haz clic directamente en el badge de Colab en la parte superior o [aquí](https://colab.research.google.com/github/Bernal-XFK/BigData-Housing-Price-Prediction-UK/blob/main/Taller_BigData_Prediccion_Viviendas_UK.ipynb) para abrir y ejecutar el notebook directamente en la nube.

### Opción 2: Ejecución Local
1. Clona el repositorio:
   ```bash
   git clone https://github.com/Bernal-XFK/BigData-Housing-Price-Prediction-UK.git
   cd BigData-Housing-Price-Prediction-UK
   ```
2. Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Linux/Mac
   venv\Scripts\activate      # En Windows
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Inicia Jupyter Notebook:
   ```bash
   jupyter notebook Taller_BigData_Prediccion_Viviendas_UK.ipynb
   ```

---

## 📜 Licencia y Uso Académico

Este proyecto se distribuye bajo la licencia **MIT con Atribución Académica y Educativa**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

Cualquier uso derivado, reproducción académica o referencia en investigaciones debe otorgar los créditos correspondientes al autor original y a la **Universidad CESMAG**.
