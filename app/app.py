import streamlit as st
import pandas as pd
import numpy as np
import os

from utils.model_loader import load_models_from_dropbox, load_models_for_category, load_recommendation_models
from utils.data_loader import load_data_for_category, get_last_available_date
from utils.forecasting_utils import create_lagged_features, iterative_forecast, get_frequency, get_prediction_horizon
from utils.generate_plot import generate_prediction_plot, plot_user_pca
from sklearn.svm import SVR
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
import xgboost as xgb
import matplotlib.pyplot as plt
import io
import plotly.graph_objects as go
import time

RECOMMENDATION_DIR = "app/recommendation"
MODELS_DIR = "app/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ⬇️ Mostrar splash screen si los modelos aún no están descargados
if not any(f.endswith(".pkl") for f in os.listdir(MODELS_DIR)):
    st.set_page_config(page_title="Robo-Advisor", layout="centered")

    st.markdown("""
        <div style="text-align: center; margin-top: 100px;">
            <h1 style="font-size: 3em;">🏦 Robo-Advisor TFM</h1>
            <p style="font-size: 1.3em;">Cargando modelos desde la nube... esto puede tardar unos segundos ⏳</p>
            <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNWxpYTVvaTUydWhnMXVmbG56MHd2bzJqbWU3anFtNW53YWRzaTZmdyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/VbE1xtnPHx6D34GXhv/giphy.gif" width="350"/>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("Inicializando modelos..."):
        load_models_from_dropbox()
        time.sleep(1.5)

    st.rerun()

# ✅ App normal una vez los modelos están listos
st.set_page_config(page_title="Robo-Advisor", layout="wide")
st.title("🏦 Robo-Advisor API")

# ------------------------ MENÚ LATERAL ------------------------

st.sidebar.title("📊 Navegación")
option = st.sidebar.radio(
    "Selecciona una opción",
    ["📄 Inicio", "💡 Recomendación de Inversión", "📈 Predicción de Activos"]
)

# ------------------------ INICIO ------------------------

if option == "📄 Inicio":
    st.header("📊 Robo-Advisor Financiero - TFM")
    st.markdown("""
    Bienvenido a la aplicación interactiva del **Trabajo de Fin de Máster** de *Guillermo Roldán Caselles*.

    Esta herramienta permite:
    
    - 📈 **Predecir la evolución de activos financieros** usando modelos de Machine Learning y series temporales.
    - 💡 **Recomendar un perfil de inversión** en base a tus características personales y económicas.

    ---
    ### 🧠 Tecnologías utilizadas
    - Streamlit para la interfaz interactiva.
    - Scikit-learn, XGBoost, SVR, Prophet, SARIMAX para predicción.
    - KMeans para clasificación de perfiles inversores.
    - Pandas y Numpy para manipulación de datos.

    ---
    ### 🗂 Estructura de la aplicación
    - **Predicción de activos**: selecciona una clase de activo y obtén su predicción futura.
    - **Recomendación de inversión**: responde unas preguntas y descubre tu perfil inversor.
    
    ---
    ### 📬 Contacto
    Desarrollado por **Guillermo Roldán Caselles**  
    *Trabajo de Fin de Máster - 2025*

    """)
# ------------------------ RECOMENDACIÓN DE INVERSIÓN ------------------------

elif option == "💡 Recomendación de Inversión":
    st.header("🧠 Recomendación de perfil inversor")
    st.markdown("Completa la información para recibir una sugerencia personalizada de perfil y activos recomendados.")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("🎂 Edad", min_value=18, max_value=100, value=30)
        education = st.selectbox(
            "🎓 Nivel educativo",
            [1, 2, 3],
            format_func=lambda x: {1: "Secundaria", 2: "Universidad", 3: "Postgrado"}[x]
        )

    with col2:
        income = st.number_input("💼 Ingresos anuales ($)", min_value=0.0, value=50000.0)
        wealth = st.number_input("🏦 Patrimonio neto ($)", min_value=0.0, value=100000.0)

    with col3:
        debt = st.number_input("💳 Deuda ($)", min_value=0.0, value=20000.0)
        risk_tolerance = st.slider("📉 Tolerancia al riesgo", 1, 10, 5)
        investment_horizon = st.slider("⏳ Horizonte de inversión (años)", 1, 30, 10)

    st.divider()

    if st.button("📌 Obtener recomendación"):
        recommendation_models = load_recommendation_models()
        kmeans = recommendation_models["kmeans"]
        scaler = recommendation_models["scaler"]

        input_data = pd.DataFrame([{
            "age": age,
            "income": income,
            "education": education,
            "wealth": wealth,
            "debt": debt,
            "risk_tolerance": risk_tolerance,
            "investment_horizon": investment_horizon
        }])

        scaled_input = scaler.transform(input_data)
        cluster = kmeans.predict(scaled_input)[0]

        CLUSTER_PROFILES = {
            0: {"perfil": "Conservador 🟢", "activos": ["Bonds", "Forex"]},
            1: {"perfil": "Moderado 🔵", "activos": ["Banco Mundial", "ETFs", "Macro"]},
            2: {"perfil": "Agresivo 🔴", "activos": ["Crypto", "Stocks", "Commodities"]}
        }

        perfil = CLUSTER_PROFILES[cluster]["perfil"]
        activos = CLUSTER_PROFILES[cluster]["activos"]

        st.success(f"🎯 Perfil asignado: **{perfil}**")

        # Tabs: Perfil + Activos
        tabs = st.tabs(["🧠 Perfil y Visualización", "📌 Activos Recomendados"])

        # Tab 1: Perfil + gráfico PCA
        with tabs[0]:
            st.markdown(f"### 🎯 Perfil asignado: {perfil}")
        
            # Cargar tabla de centroides desnormalizados
            centroides_real = pd.read_csv(os.path.join(RECOMMENDATION_DIR, "cluster_centers_real.csv"))
        
            # Extraer fila correspondiente al perfil del usuario
            perfil_simple = perfil.split()[0]  # Conservador, Moderado, Agresivo
            fila = centroides_real[centroides_real["profile"].str.startswith(perfil_simple)]
        
            if not fila.empty:
                edad = int(round(fila["age"].values[0]))
                ingresos = int(round(fila["income"].values[0]))
                patrimonio = int(round(fila["wealth"].values[0]))
                deuda = int(round(fila["debt"].values[0]))
                riesgo = int(round(fila["risk_tolerance"].values[0]))
                horizonte = int(round(fila["investment_horizon"].values[0]))
        
                st.markdown(
                    f"**Descripción del perfil:** Este perfil se caracteriza por una edad media de **{edad:.0f} años**, "
                    f"ingresos anuales cercanos a **\${ingresos:,.0f}**, "
                    f"un patrimonio neto promedio de **\${patrimonio:,.0f}**, y una deuda media de **\${deuda:,.0f}**. "
                    f"Muestra una tolerancia al riesgo de **{riesgo}/10** y un horizonte típico de inversión de **{horizonte} años**."
                )
        
                # 📊 Primera fila
                col1, col2, col3 = st.columns(3)
                col1.metric("Edad media", f"{edad:.0f} años")
                col2.metric("Ingresos promedio", f"${ingresos:,.0f}")
                col3.metric("Patrimonio promedio", f"${patrimonio:,.0f}")
                
                # 📊 Segunda fila
                col4, col5, col6 = st.columns(3)
                col4.metric("Deuda promedio", f"${deuda:,.0f}")
                col5.metric("Tolerancia al riesgo", f"{riesgo}/10")
                col6.metric("Horizonte (años)", f"{horizonte}")
            else:
                st.warning("No se pudo encontrar información del perfil.")
        
            st.markdown("---")
            st.markdown("### 🔍 Tu posición entre los inversores")
        
            # Gráfico de posición en el PCA
            fig = plot_user_pca(user_scaled=scaled_input, kmeans=kmeans)
            st.pyplot(fig)

        # Tab 2: Activos detallados
        with tabs[1]:
            st.markdown("### 📌 Activos recomendados:")
        
            ASSET_DETAILS = {
                "Bonds": {
                     "definicion": (
                        "Instrumentos de deuda emitidos por gobiernos o corporaciones "
                        "que generan ingresos fijos mediante el pago de intereses."
                    ),
                    "activos": {
                        "DGS10": "Bono del Tesoro a 10 años (USA)",
                        "DGS2": "Bono del Tesoro a 2 años (USA)",
                        "DGS30": "Bono del Tesoro a 30 años (USA)",
                        "GS10": "Rendimiento bonos a 10 años (FRED)",
                        "GS2": "Rendimiento bonos a 2 años (FRED)"
                    }
                },
                "Forex": {
                     "definicion": (
                        "Mercado global donde se compran y venden divisas. Opera de forma descentralizada "
                        "y continua, permitiendo el intercambio de monedas a nivel internacional."
                    ),
                    "activos": {
                        "EURUSD=X": "Euro / Dólar estadounidense",
                        "GBPUSD=X": "Libra / Dólar",
                        "USDJPY=X": "Dólar / Yen japonés",
                        "AUDUSD=X": "Dólar australiano / Dólar",
                        "USDCAD=X": "Dólar / Dólar canadiense"
                    }
                },
                "Crypto": {
                     "definicion": (
                        "activos digitales descentralizados que utilizan tecnología blockchain para "
                        "garantizar la seguridad y transparencia de las transacciones."
                    ),
                    "activos": {
                        "BTC-USD": "Bitcoin",
                        "ETH-USD": "Ethereum",
                        "BNB-USD": "Binance Coin",
                        "XRP-USD": "Ripple",
                        "ADA-USD": "Cardano"
                    }
                },
                "Stocks": {
                     "definicion": (
                        "Activos que representan la propiedad de una parte de una empresa "
                        "y otorgan derechos económicos y políticos a sus poseedores."
                    ),
                    "activos": {
                        "AAPL": "Apple",
                        "MSFT": "Microsoft",
                        "GOOGL": "Alphabet (Google)",
                        "AMZN": "Amazon",
                        "TSLA": "Tesla"
                    }
                },
                "ETFs": {
                     "definicion": (
                        "Fondos de inversión que replican el comportamiento de un índice "
                        "o sector y se negocian en bolsa como una acción."
                    ),
                    "activos": {
                        "SPY": "ETF S&P 500",
                        "IVV": "ETF S&P 500 (iShares)",
                        "VTI": "ETF Total Stock Market",
                        "QQQ": "ETF Nasdaq 100",
                        "DIA": "ETF Dow Jones"
                    }
                },
                "Commodities": {
                     "definicion": (
                        "Activos físicos cuyo valor se basa en la oferta y demanda global. "
                    ),
                    "activos": {
                        "GC=F": "Oro (Gold Futures)",
                        "CL=F": "Petróleo (Crude Oil)",
                        "SI=F": "Plata (Silver Futures)",
                        "NG=F": "Gas Natural (Natural Gas)",
                        "HG=F": "Cobre (Copper Futures)"
                    }
                },
                "Macro": {
                    "definicion": (
                        "Indicadores macroeconómicos que reflejan el estado de salud de una economía "
                        "y afectan directamente al comportamiento de los mercados financieros."
                    ),
                    "activos": {
                        "GDP": "Producto Interno Bruto (PIB)",
                        "CPIAUCSL": "Índice de Precios al Consumidor (CPI USA)",
                        "UNRATE": "Tasa de desempleo (USA)",
                        "FEDFUNDS": "Tipo de interés de la Reserva Federal"
                    }
                },
                "Banco Mundial": {
                    "definicion": (
                        "Variables clave del desarrollo económico global. "
                    ),
                    "activos": {
                        "NY.GDP.MKTP.CD": "PIB global (USD actuales)",
                        "FP.CPI.TOTL.ZG": "Inflación (% anual)",
                        "NE.EXP.GNFS.ZS": "Exportaciones (% PIB)",
                        "NE.IMP.GNFS.ZS": "Importaciones (% PIB)",
                        "SL.UEM.TOTL.ZS": "Desempleo total (% población activa)"
                    }
                }
            }
        
            for tipo in activos:
                detalles = ASSET_DETAILS.get(tipo)
        
                with st.container():
                    st.subheader(f" {tipo}")
        
                    if detalles:
                        st.markdown(f"**📄 Tipo de activo:** {detalles.get('definicion', 'No disponible.')}")
        
                        activos_dict = detalles.get("activos", {})
                        if activos_dict:
                            with st.expander("🔍 Ver activos disponibles"):
                                df_activos = pd.DataFrame.from_dict(
                                    activos_dict,
                                    orient="index",
                                    columns=["Descripción"]
                                ).rename_axis("Ticker").reset_index()
                                st.dataframe(df_activos, use_container_width=True)
                        else:
                            st.warning("⚠️ Este tipo de activo no tiene activos detallados definidos.")
                    else:
                        st.info("ℹ️ No se encontró información detallada para este tipo de activo.")
        
                st.divider()

# ------------------------ PREDICCIÓN DE ACTIVOS ------------------------

elif option == "📈 Predicción de Activos":
    st.header("📈 Predicción de evolución de activos financieros")
    st.markdown("Selecciona una clase de activo y el horizonte temporal para generar una predicción basada en modelos financieros.")

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        asset_class = st.selectbox(
            "📂 Clase de activo",
            ['stocks', 'etfs', 'crypto', 'forex', 'bonds', 'commodities', 'macro', 'banco_mundial']
        )

    with col2:
        horizon_type = st.radio(
            "🕒 Horizonte de predicción",
            ['short', 'long'],
            horizontal=True
        )

    st.divider()

    if st.button("🔮 Generar predicción"):
        with st.spinner("📡 Cargando modelos y datos..."):
            models = load_models_for_category(asset_class)
            data = load_data_for_category(asset_class)
            last_dates = get_last_available_date(asset_class)
            freq = get_frequency(asset_class)
            horizon = get_prediction_horizon(asset_class, horizon_type)

            predictions = {}

            for asset_name, model_info in models.items():
                if asset_name not in data:
                    continue

                df = data[asset_name]
                model = model_info["model"]
                scaler_y = model_info["scaler_y"]

                if isinstance(model, xgb.XGBRegressor):
                    future_dates = pd.date_range(start=df.index[-1], periods=horizon + 1, freq=freq)[1:]
                    future_df = pd.DataFrame(index=future_dates)
                    full_series = pd.concat([df, future_df])
                    full_series[df.columns[0]].name = "target"
                    lagged_df = create_lagged_features(full_series[df.columns[0]])
                    X_test = lagged_df.drop(columns=["target"], errors="ignore").loc[future_dates]
                    prediction = iterative_forecast(model, X_test, df.columns[0])

                elif isinstance(model, SVR):
                    future_dates = pd.date_range(start=df.index[-1], periods=horizon + 1, freq=freq)[1:]
                    full_series = pd.concat([df, pd.DataFrame(index=future_dates)])
                    if scaler_y is not None:
                        scaled_series = pd.Series(
                            scaler_y.transform(full_series[[df.columns[0]]]).flatten(),
                            index=full_series.index,
                            name=df.columns[0]
                        )
                    else:
                        scaled_series = full_series[df.columns[0]]

                    lagged_df = create_lagged_features(scaled_series)
                    X_test = lagged_df.drop(columns=[df.columns[0]], errors="ignore").loc[future_dates]
                    X_test = X_test.loc[:, X_test.columns.intersection(model.feature_names_in_)]
                    prediction_scaled = iterative_forecast(model, X_test, df.columns[0])

                    prediction = scaler_y.inverse_transform(np.array(prediction_scaled).reshape(-1, 1)).flatten().tolist() if scaler_y else prediction_scaled.tolist()

                elif isinstance(model, Prophet):
                    future_df = pd.DataFrame({"ds": pd.date_range(start=df.index[-1], periods=horizon, freq=freq)})
                    forecast = model.predict(future_df)
                    prediction = forecast["yhat"].tolist()

                elif isinstance(model, SARIMAXResultsWrapper):
                    prediction = model.forecast(steps=horizon).tolist()

                else:
                    prediction = None

                predictions[asset_name] = prediction

            # ✅ Guardar resultados en sesión
            st.session_state.predictions = predictions
            st.session_state.data = data
            st.session_state.freq = freq
            st.success("✅ Predicción generada correctamente")

    # ✅ Mostrar resultados si ya hay predicciones en sesión
    if "predictions" in st.session_state:
        predictions = st.session_state.predictions
        data = st.session_state.data
        freq = st.session_state.freq

        tabs = st.tabs(["📊 Gráfico", "📋 Tabla", "⬇️ Exportar CSV"])

        with tabs[0]:
            st.markdown("### 📈 Gráficas individuales por activo")
        
            for asset in predictions:
                st.markdown(f"#### 🪙 {asset}")
        
                # Slider adaptado a la frecuencia
                if freq.startswith("B") or freq.startswith("D"):
                    unidad = "D"
                    min_años = 1
                    max_años = 5
                    default_años = 1
                    step = 1
                    año_a_periodos = 252
                elif freq.startswith("M"):
                    unidad = "M"
                    min_años = 1
                    max_años = 30
                    default_años = 5
                    step = 1
                    año_a_periodos = 12
                elif freq.startswith("Y") or freq.startswith("A"):
                    unidad = "Y"
                    min_años = 5
                    max_años = 50
                    default_años = 20
                    step = 5
                    año_a_periodos = 1
                else:
                    unidad = "D"
                    max_años = 5
                    step = 1
                    año_a_periodos = 252
        
                num_años = st.slider(
                    f"Rango histórico para {asset} (años)",
                    min_value=min_años,
                    max_value=max_años,
                    step=step,
                    value=default_años,  # valor inicial por defecto
                    key=f"slider_{asset}"
                )
        
                max_periods = num_años * año_a_periodos
        
                fig = generate_prediction_plot(
                    {asset: predictions[asset]},
                    {asset: data[asset]},
                    freq,
                    max_periods=max_periods,
                    unidad=unidad
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Tabla
        with tabs[1]:
            for asset, values in predictions.items():
                dates = pd.date_range(start=data[asset].index[-1], periods=len(values)+1, freq=freq)[1:]
                df_pred = pd.DataFrame({
                    "Fecha": dates,
                    "Predicción": values
                })
                st.subheader(f"📌 {asset}")
                st.dataframe(df_pred)
        
        # Exportar CSV
        with tabs[2]:
            csv_buffer = io.StringIO()
            full_df = []
        
            for asset, values in predictions.items():
                dates = pd.date_range(start=data[asset].index[-1], periods=len(values)+1, freq=freq)[1:]
                df = pd.DataFrame({
                    "Activo": asset,
                    "Fecha": dates,
                    "Prediccion": values
                })
                full_df.append(df)
        
            export_df = pd.concat(full_df)
            export_df.to_csv(csv_buffer, index=False)
        
            st.download_button(
                label="📥 Descargar predicciones como CSV",
                data=csv_buffer.getvalue(),
                file_name="predicciones_financieras.csv",
                mime="text/csv"
            )

