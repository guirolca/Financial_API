import os
import joblib
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # Se importan ambos para evitar errores
import requests
import zipfile

MODELS_DIR = "app/models"
RECOMMENDATION_DIR = "app/recommendation"

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models_from_dropbox():
    """
    Descarga un zip de modelos desde Dropbox y lo guarda en app/models si no existen modelos .pkl.
    """
    dropbox_url = "https://www.dropbox.com/scl/fi/lznyzcxtgfp3zfhuein1p/modelos.zip?rlkey=51ed9ba5wig1io5vhbp6qksfn&st=zzqcj97q&dl=1" 
    zip_path = os.path.join(MODELS_DIR, "modelos.zip")

    # Verifica si ya hay modelos
    existing_pkl_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    if existing_pkl_files:
        print(f"✅ Modelos ya presentes ({len(existing_pkl_files)} archivos).")
        return

    try:
        print("⬇️ Descargando ZIP de modelos desde Dropbox...")
        response = requests.get(dropbox_url)
        if response.status_code == 200:
            with open(zip_path, "wb") as f:
                f.write(response.content)

            print("📦 Extrayendo modelos...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(MODELS_DIR)

            os.remove(zip_path)
            print("✅ Modelos descargados y listos en `app/models/`")
        else:
            print(f"⚠️ Error en la descarga: {response.status_code}")

    except Exception as e:
        print(f"❌ Error durante la descarga o extracción: {e}")


def load_models_for_category(asset_class):
    """
    Carga todos los modelos para una categoría de activos (asset_class).
    Devuelve un diccionario donde cada clave es el activo y su valor es un diccionario con el modelo y el escalador (si aplica).
    """
    models = {}

    for filename in os.listdir(MODELS_DIR):
        if filename.startswith(f"{asset_class}_") and filename.endswith(".pkl"):
            # ✅ Extraer el nombre del asset eliminando el prefijo del asset_class y el sufijo del modelo
            asset_name = filename.replace(f"{asset_class}_", "").replace("_svr.pkl", "").replace("_xgboost.pkl", "").replace("_sarimax.pkl", "").replace("_prophet.pkl", "")
            
            # ✅ Restauramos los puntos solo si es `banco_mundial`
            if asset_class == "banco_mundial":
                asset_name = asset_name.replace("_", ".")

            # ✅ Construimos la ruta completa
            model_path = os.path.join(MODELS_DIR, filename)

            try:
                logger.info(f"📂 Cargando modelo: {filename}")
                model_data = joblib.load(model_path)

                # 🚨 Verificar el tipo de modelo cargado
                if isinstance(model_data, dict) and "model" in model_data:
                    # ✅ Caso: Modelo con escalador (SVR y similares)
                    model = model_data["model"]
                    scaler_y = model_data.get("scaler_y", None)  # Puede no existir en algunos casos
                    logger.info(f"✅ `{asset_name}` tiene `scaler_y` ({type(scaler_y).__name__})")
                else:
                    # ✅ Caso: Modelos sin escalador (XGBoost, Prophet, SARIMAX)
                    model = model_data
                    scaler_y = None
                    logger.info(f"ℹ️ `{asset_name}` es un modelo sin `scaler_y` ({type(model).__name__})")

                # Guardamos en el diccionario final con el asset_name corregido
                models[asset_name] = {"model": model, "scaler_y": scaler_y}

            except Exception as e:
                logger.error(f"⚠️ Error al cargar {filename}: {e}")

    return models

def load_recommendation_models():
    """
    Carga el modelo K-Means y el scaler para la recomendación de perfiles de inversión.
    Retorna un diccionario con ambos objetos.
    """
    kmeans_model_path = os.path.join(RECOMMENDATION_DIR, "kmeans_investment_model.pkl")
    scaler_path = os.path.join(RECOMMENDATION_DIR, "scaler_investment.pkl")

    # 🚨 Verificar si los archivos existen antes de cargarlos
    if not os.path.exists(kmeans_model_path):
        raise RuntimeError("🚨 Modelo K-Means no encontrado en recommendation_dir. Asegúrate de entrenarlo.")
    
    if not os.path.exists(scaler_path):
        raise RuntimeError("🚨 Scaler de recomendación no encontrado en recommendation_dir. Asegúrate de entrenarlo.")

    # ✅ Cargar los modelos
    kmeans = joblib.load(kmeans_model_path)
    scaler = joblib.load(scaler_path)

    return {"kmeans": kmeans, "scaler": scaler}

