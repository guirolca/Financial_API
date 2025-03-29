# 📈 API REST - Modelos de Predicción Financiera

Esta API permite realizar predicciones sobre activos financieros y generar recomendaciones de inversión personalizadas, integrando múltiples modelos de machine learning desarrollados previamente.

---

## 🚀 Tecnologías utilizadas

- **Streamlit**: Creación de la interfaz interactiva.
- **Scikit-learn / XGBoost / SVR / Prophet / SARIMAX**: Modelos de predicción.
- **Pandas / NumPy**: Manipulación y análisis de datos.
- **Matplotlib**: Visualización de resultados.

---

## 📁 Estructura del proyecto

streamlit_api/ 
├── README.md # Documentación básica del proyecto 
├── requirements.txt # Librerías necesarias para la ejecución 
└── app/ 
    ├── data_raw/ # Datos sin procesar (originales) 
    ├── data/ # Datos preprocesados o estructurados 
    ├── models/ # Modelos de predicción entrenados (.pkl) 
    ├── recommendation/ # Modelos y archivos para el sistema de recomendación 
    ├── utils/ # Funciones auxiliares (carga de datos, preprocesamiento, etc.) 
    └── app.py # Archivo principal que lanza la aplicación Streamlit

---

## ⚙️ Ejecución local

1. Clona el repositorio y accede al directorio:
git clone https://github.com/tuusuario/streamlit_api.git
cd streamlit_api

2. Instala las dependencias necesarias:
pip install -r requirements.txt

3. Lanza la aplicación:
streamlit run app/app.py

---

## 🧠 Funcionalidades principales
📈 Predicción de activos financieros
Selecciona una clase de activo (acciones, criptomonedas, bonos, etc.) y elige un horizonte temporal. La aplicación usará el modelo correspondiente para predecir su evolución.

💡 Recomendación de inversión
Introduce datos personales (edad, ingresos, tolerancia al riesgo, etc.) y la app asignará un perfil inversor y sugerirá tipos de activos adecuados.

---

## 🧩 Proyecto basado en:
Este proyecto fue inicialmente desarrollado como una API REST en FastAPI y posteriormente migrado a Streamlit para facilitar su despliegue interactivo.

---

📬 Contacto
Proyecto desarrollado como parte del Trabajo de Fin de Máster por Guillermo Roldán Caselles. 
[guillermoroldancaselles@gmail.com]

