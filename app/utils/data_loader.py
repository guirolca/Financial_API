import os
import pandas as pd

DATA_DIR = "app/data"

def load_data_for_category(asset_class):
    """
    Carga los datos históricos de una categoría de activos desde archivos CSV.
    Retorna un diccionario con los DataFrames de cada asset.
    """
    data = {}
    category_path = os.path.join(DATA_DIR, f"{asset_class}.csv")
    
    if not os.path.exists(category_path):
        print(f"⚠️ No se encontraron datos para la categoría {asset_class}")
        return None

    try:
        df = pd.read_csv(category_path, index_col=0, parse_dates=True)
        for asset in df.columns:
            data[asset] = df[[asset]]  # Convertir en DataFrame manteniendo solo la serie del activo
        print(f"✅ Datos cargados para {asset_class}: {list(data.keys())}")
    except Exception as e:
        print(f"⚠️ Error al cargar datos para {asset_class}: {e}")
    
    return data

def get_last_available_date(asset_class):
    """
    Devuelve la última fecha disponible en los datos históricos de un asset class.
    """
    data = load_data_for_category(asset_class)
    if not data:
        return None

    last_dates = {asset: df.index[-1] for asset, df in data.items() if not df.empty}
    return last_dates  # Devuelve un diccionario con la última fecha de cada asset
