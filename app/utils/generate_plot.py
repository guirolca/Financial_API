import plotly.graph_objects as go
import pandas as pd
import logging
import matplotlib.pyplot as plt
import joblib
import os

logger = logging.getLogger(__name__)

RECOMMENDATION_DIR = "app/recommendation"

def generate_prediction_plot(predictions, historical_data, freq, max_periods=None, unidad="D"):
    """
    Gráfico interactivo con históricos y predicciones conectadas.
    """
    fig = go.Figure()

    for asset, pred_values in predictions.items():
        if asset not in historical_data:
            continue

        df = historical_data[asset]
        series = df[df.columns[0]]

        # 🔧 Reindexar histórico según la frecuencia de predicción
        full_range = pd.date_range(start=series.index.min(), end=series.index.max(), freq=freq)
        series = series.reindex(full_range, method='ffill')

        # Recortar histórico por rango
        if max_periods is not None:
            series = series[-max_periods:]

        # Último valor histórico
        last_value = series.values[-1]

        # Insertar el último valor histórico al principio de la predicción
        y_pred = [last_value] + list(pred_values)

        # Última fecha histórica
        last_date = series.index[-1]
        
        # Obtener fecha siguiente exacta según la frecuencia
        if unidad == "M":
            start_date = last_date + pd.offsets.MonthEnd(1)
        elif unidad == "Y":
            start_date = last_date + pd.offsets.YearEnd(1)
        else:
            start_date = last_date  # para diaria o business day
        
        # Insertamos también el último valor del histórico al principio
        y_pred = [series.values[-1]] + list(pred_values)
        pred_index = pd.date_range(start=start_date, periods=len(y_pred)-1, freq=freq)
        pred_index = [last_date] + list(pred_index)  # conectamos con el histórico

        # Histórico
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines',
            name=f"{asset} - Histórico"
        ))

        # Predicción conectada
        fig.add_trace(go.Scatter(
            x=pred_index,
            y=y_pred,
            mode='lines',
            name=f"{asset} - Predicción"
        ))

    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Valor",
        hovermode="x unified",
        template="plotly_white",
        legend_title="Series"
    )

    return fig

def plot_user_pca(user_scaled, kmeans):
    """
    Dibuja al usuario proyectado en el PCA 2D junto al resto de inversores y los centroides.

    Args:
        user_scaled: array 2D del usuario transformado por StandardScaler (shape: 1 x n_features)
        kmeans: modelo KMeans ya cargado

    Returns:
        fig: figura de matplotlib lista para usar con st.pyplot(fig)
    """
    pca_path = os.path.join(RECOMMENDATION_DIR, "pca.pkl")
    pca_users_path = os.path.join(RECOMMENDATION_DIR, "pca_users.csv")

    # Cargar PCA y usuarios
    pca = joblib.load(pca_path)
    df_pca_users = pd.read_csv(pca_users_path)

    # Reducir los nombres de perfil (sin emojis ni descripciones largas)
    df_pca_users["profile_simple"] = df_pca_users["profile"].str.extract(r"(Conservador|Moderado|Agresivo)")

    # Proyectar al usuario en el PCA 2D
    user_2d = pca.transform(user_scaled)[0]
    centers_2d = pca.transform(kmeans.cluster_centers_)

    # Colores por perfil simple
    color_map = {
        "Conservador": "green",
        "Moderado": "blue",
        "Agresivo": "red"
    }

    # Crear figura
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Posición del usuario en el mapa PCA de inversores", fontsize=15)

    # Puntos de inversores
    for perfil, color in color_map.items():
        subset = df_pca_users[df_pca_users["profile_simple"] == perfil]
        ax.scatter(
            subset["PCA_1"], subset["PCA_2"],
            label=f"{perfil}", color=color, alpha=0.4, s=35
        )

    # Centroides
    for i, center in enumerate(centers_2d):
        perfil = list(color_map.keys())[i]
        ax.scatter(
            center[0], center[1],
            color=color_map[perfil],
            edgecolors='black',
            s=220, marker='X', label=f"Centroide {perfil}"
        )

    # Usuario actual
    ax.scatter(
        user_2d[0], user_2d[1],
        color='black', marker='*', s=300, label="Tú"
    )

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()

    return fig
    
