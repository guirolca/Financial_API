import pandas as pd
import numpy as np
import logging

# Configurar el logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def create_lagged_features(series):
    """
    Crea características de rezagos en la serie de tiempo, priorizando tendencias a corto plazo,
    pero incluyendo ciclos a mediano y largo plazo.

    - series: Serie temporal original (pandas Series con índice de fechas).

    Retorna un DataFrame con los rezagos calculados.
    """
        
    logger.info(f"🔄 Creando características de rezago para {series.name}...")

    df = pd.DataFrame(series)
    target_col = series.name
    max_lag = len(df)  # No exceder la cantidad de datos disponibles
    one_year_lag = min(252, max_lag)  # Máximo 1 año de rezagos

    # **Short-Term Lags (1-30 días)**
    for lag in range(1, min(31, max_lag)):
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    # **Medium-Term Lags (Cada 5 días hasta 1 año)**
    for lag in range(35, one_year_lag, 5):
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    # **Long-Term Lags (1-4 años si hay suficientes datos)**
    for lag in [252, 504, 756, 1008]:  # 1, 2, 3, 4 años
        if lag < max_lag:
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    logger.info(f"✅ Rezagos creados con éxito para {series.name}")
    
    return df


def iterative_forecast(model, X_test, target_name):
    """
    Realiza predicción iterativa para modelos como XGBoost, RF o LSTM,
    asegurando que los valores predichos sean usados en futuras iteraciones.

    - model: Modelo entrenado (XGBoost, Random Forest, LSTM, etc.).
    - X_test: DataFrame con las características (lags) de los datos futuros.
    - target_name: Nombre de la variable objetivo.

    Retorna un array con las predicciones para el horizonte de predicción.
    """
    logger.info(f"🚀 Iniciando predicción iterativa para {target_name}...")

    X_test_copy = X_test.copy()
    preds = []

    # Extraer solo los rezagos dinámicos relacionados con la variable objetivo
    dynamic_lags = sorted([
        int(col.split("_")[-1]) for col in X_test_copy.columns if f"{target_name}_lag_" in col
    ])
    
    # Diccionario para almacenar valores futuros que se insertarán en los rezagos
    future_updates = {lag: {} for lag in dynamic_lags}

    for i in range(len(X_test_copy)):
        # Predecir el siguiente valor
        pred = model.predict(X_test_copy.iloc[[i]])[0]
        preds.append(pred)

        # Guardar predicción para actualizar futuros rezagos
        for lag in dynamic_lags:
            future_index = i + lag - 1
            if future_index < len(X_test_copy):
                future_updates[lag][future_index] = pred

        # Aplicar las actualizaciones cuando corresponda
        for lag, updates in future_updates.items():
            if i in updates:
                update_index = i + 1
                lag_col = f"{target_name}_lag_{lag}"
                
                # Asegurar que el índice y la columna existan
                if update_index < len(X_test_copy) and lag_col in X_test_copy.columns:
                    X_test_copy.iloc[update_index, X_test_copy.columns.get_loc(lag_col)] = updates[i]

    logger.info(f"✅ Predicción iterativa completada para {target_name}.")
    
    return np.array(preds)

def get_frequency(asset_class):
    """
    Devuelve la frecuencia de predicción basada en el tipo de activo.
    """
    if asset_class in ["stocks", "forex", "etfs", "crypto", "commodities"]:  
        return 'B'  # Días hábiles (~21 por mes)
    elif asset_class in ["macro", "bonds"]:  
        return 'M'  # Mensual
    elif asset_class == "banco_mundial":  
        return 'A'  # Anual
    return 'B'  # Por defecto, días hábiles


def get_prediction_horizon(asset_class, horizon_type="short"):
    """
    Devuelve el número de períodos para la predicción según el horizonte de tiempo deseado.
    
    - horizon_type: "short" (corto plazo) o "long" (largo plazo)
    """
    if horizon_type not in ["short", "long"]:
        raise ValueError("horizon_type debe ser 'short' o 'long'.")

    if asset_class in ["stocks", "forex", "etfs", "crypto", "commodities"]:
        return 63 if horizon_type == "short" else 252  # 3 meses o 1 año en días hábiles
    elif asset_class in ["macro", "bonds"]:
        return 24 if horizon_type == "short" else 120  # 2 año o 10 años
    elif asset_class == "banco_mundial":
        return 5 if horizon_type == "short" else 20  # 5 años o 20 años

    return 126 if horizon_type == "short" else 252  # Default medio año o 1 año
