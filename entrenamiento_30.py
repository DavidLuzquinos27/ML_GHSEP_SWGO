# === 1. Librerías ===
import pandas as pd
import numpy as np
import seaborn as sns
from xgboost.callback import EarlyStopping as XGBEarlyStopping 
from sklearn.model_selection import train_test_split, RandomizedSearchCV 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import ( classification_report, confusion_matrix, roc_curve, auc, RocCurveDisplay ) 
from sklearn.ensemble import VotingClassifier 
from lightgbm import early_stopping,log_evaluation 
from lightgbm import LGBMClassifier 
from xgboost import XGBClassifier 
from catboost import CatBoostClassifier 
from xgboost.callback import EarlyStopping 
import warnings 
import json 
import matplotlib.pyplot as plt 
from itertools import product 
import joblib 
from sklearn.metrics import f1_score, precision_score, recall_score 
import logging 
import os 
os.environ["OMP_NUM_THREADS"] = "30" 
os.environ["OPENBLAS_NUM_THREADS"] = "30" 
os.environ["MKL_NUM_THREADS"] = "30" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "30" 
os.environ["NUMEXPR_NUM_THREADS"] = "30" 
training_times = [] 
import logging 
from logging.handlers import RotatingFileHandler 
from pathlib import Path 
import time # <-- añadir 
# Logger principal del pipeline (mensajes generales de orquestación) 
muestra=30 
# === Carpeta y archivo de log === 
LOG_DIR = Path("logs") 
LOG_DIR.mkdir(exist_ok=True) 
LOG_FILE = LOG_DIR / f"pipeline_swgo_{muestra}.log" 
# === Configuración del logger raíz === 
def crear_logger(nombre, archivo): 
    logger_model = logging.getLogger(nombre) 
    logger_model.setLevel(logging.INFO) 
    handler_file = RotatingFileHandler(LOG_DIR / archivo, maxBytes=5_000_000, backupCount=3) 
    handler_console = logging.StreamHandler() 
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S') 
    handler_file.setFormatter(formatter) 
    handler_console.setFormatter(formatter) 
    logger_model.addHandler(handler_file) 
    logger_model.addHandler(handler_console) 
    logger_model.propagate = False # evitar duplicados 
    return logger_model 
logger_lgbm = crear_logger("LightGBM", f"pipeline_lightgbm_{muestra}.log") 
logger_xgb = crear_logger("XGBoost", f"pipeline_xgboost_{muestra}.log") 
logger_cat = crear_logger("CatBoost", f"pipeline_catboost_{muestra}.log") 
logger_voting = crear_logger("Voting", f"pipeline_voting_{muestra}.log") 
logger = crear_logger("MAIN", f"pipeline_main_{muestra}.log") 
warnings.filterwarnings('ignore') 
def obtener_tpr_objetivo(
    energia_bin_center,
    zona,
    df_paper=None
):

    # Zonas 4 y 5 → TPR fijo
    if zona in [4, 5]:
        return 0.8

    # Zonas 1–3 → usar paper
    if df_paper is None:
        return None

    fila = df_paper[
        np.isclose(
            df_paper["bin_center"],
            energia_bin_center,
            atol=1e-6
        )
    ]

    if fila.empty:
        return None

    return float(fila["TPR"].iloc[0])

def cargar_curva_fpr(zona):

    # zonas 4 y 5 siguen usando TPR fijo
    if zona in [4, 5]:
        return None

    archivo = f"zona{zona}FPR.txt"

    if not os.path.exists(archivo):
        raise FileNotFoundError(f"No existe {archivo}")

    df = pd.read_csv(
        archivo,
        sep=r"\s+",
        engine="python"
    )

    return df.sort_values("bin_center")
def obtener_fpr_objetivo(
    energia_bin_center,
    zona,
    df_fpr=None
):

    if zona in [4, 5]:
        return None

    if df_fpr is None:
        return None

    fila = df_fpr[
        np.isclose(
            df_fpr["bin_center"],
            energia_bin_center,
            atol=1e-6
        )
    ]

    if fila.empty:
        return None

    return float(fila["FPR"].iloc[0])
def tpr_interpolado_en_fpr(
    y_true,
    y_proba,
    fpr_objetivo=0.01
):

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_proba,
        pos_label=1
    )

    df_roc = pd.DataFrame({
        "FPR": fpr,
        "TPR": tpr
    })

    # Para FPR repetidos → mayor TPR
    df_roc = (
        df_roc
        .groupby("FPR", as_index=False)["TPR"]
        .max()
        .sort_values("FPR")
    )

    if (
        fpr_objetivo < df_roc["FPR"].min()
        or
        fpr_objetivo > df_roc["FPR"].max()
    ):
        return None

    tpr_interp = np.interp(
        fpr_objetivo,
        df_roc["FPR"].values,
        df_roc["TPR"].values
    )

    q_interp = (
        tpr_interp /
        (np.sqrt(fpr_objetivo) + 1e-10)
    )

    return {
        "threshold": None,
        "TPR": float(tpr_interp),
        "FPR": float(fpr_objetivo),
        "Q": float(q_interp),
        "modo": "interpolado_fpr"
    }

def graficar_tpr_vs_energia(bin_centers, tpr_list, nombre_modelo, carpeta, caso, version): 
    plt.figure(figsize=(8, 5)) 
    plt.plot(bin_centers, tpr_list, marker='o', color='green', label='TPR (gamma)') 
    plt.xlabel("rec.LHLatDistFitEnergy (log10(E/GeV))") 
    plt.ylabel("TPR") 
    plt.title(f"TPR vs Energía reconstruida - {nombre_modelo}") 
    plt.grid(True) 
    plt.legend() 
    plt.tight_layout() 
    plt.savefig(f"{carpeta}/TPR_vs_Energia_{nombre_modelo}_{caso}_{version}.png", dpi=600) 
    plt.close() 
    
def graficar_fpr_vs_energia(bin_centers, fpr_list, nombre_modelo, carpeta, caso, version): 
    plt.figure(figsize=(8, 5)) 
    plt.plot(bin_centers, fpr_list, marker='s', color='red', label='FPR (proton)') 
    plt.xlabel("rec.LHLatDistFitEnergy (log10(E/GeV))") 
    plt.ylabel("FPR") 
    plt.title(f"FPR vs Energía reconstruida - {nombre_modelo}") 
    plt.grid(True) 
    plt.legend() 
    plt.tight_layout() 
    plt.savefig(f"{carpeta}/FPR_vs_Energia_{nombre_modelo}_{caso}_{version}.png", dpi=600) 
    plt.close() 
def graficar_q_vs_energia(bin_centers, q_list, nombre_modelo, carpeta, caso, version): 
    plt.figure(figsize=(8, 5)) 
    plt.plot(bin_centers, q_list, marker='d', color='blue', label='Q-factor') 
    plt.xlabel("rec.LHLatDistFitEnergy (log10(E/GeV))") 
    plt.ylabel("Q = TPR / √FPR") 
    plt.title(f"Q vs Energía reconstruida - {nombre_modelo}") 
    plt.grid(True)
    plt.legend() 
    plt.tight_layout() 
    plt.savefig(f"{carpeta}/Q_vs_Energia_{nombre_modelo}_{caso}_{version}.png", dpi=600) 
    plt.close() 
def graficar_metricas_vs_energia_fpr(
    model,
    X_test,
    y_test,
    energy_column,
    df_test,
    df_fpr,
    zona,
    n_bins=7,
    carpeta=".",
    nombre_modelo="modelo",
    caso="CASO",
    version="v1"
):

    energies = df_test[energy_column].values
    y_proba = model.predict_proba(X_test)[:, 1]

    bins = np.arange(2.0, 5.5 + 0.5, 0.5)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    tpr_list, fpr_list, q_list = [], [], []

    for i in range(n_bins):

        bin_center = bin_centers[i]

        FPR_OBJETIVO = obtener_fpr_objetivo(
            bin_center,
            zona,
            df_fpr
        )

        if FPR_OBJETIVO is None:
            tpr_list.append(np.nan)
            fpr_list.append(np.nan)
            q_list.append(np.nan)
            continue

        mask = (
            (energies >= bins[i]) &
            (energies < bins[i + 1])
        )

        if np.sum(mask) == 0:
            tpr_list.append(np.nan)
            fpr_list.append(np.nan)
            q_list.append(np.nan)
            continue

        y_true_bin = y_test[mask]
        y_proba_bin = y_proba[mask]

        resultado = tpr_interpolado_en_fpr(
            y_true_bin,
            y_proba_bin,
            fpr_objetivo=FPR_OBJETIVO
        )

        if resultado:
            tpr_list.append(resultado["TPR"])
            fpr_list.append(FPR_OBJETIVO)
            q_list.append(resultado["Q"])
        else:
            tpr_list.append(np.nan)
            fpr_list.append(np.nan)
            q_list.append(np.nan)

    # prefijo nuevo
    sufijo = "_FPR_OBJ"

    graficar_tpr_vs_energia(
        bin_centers,
        tpr_list,
        nombre_modelo + sufijo,
        carpeta,
        caso,
        version
    )

    graficar_fpr_vs_energia(
        bin_centers,
        fpr_list,
        nombre_modelo + sufijo,
        carpeta,
        caso,
        version
    )

    graficar_q_vs_energia(
        bin_centers,
        q_list,
        nombre_modelo + sufijo,
        carpeta,
        caso,
        version
    )
def cargar_curva_paper(zona):

    # Zonas 4 y 5 usan TPR fijo
    if zona in [4, 5]:
        return None

    archivo = f"zona_{zona}_paper.txt"

    if not os.path.exists(archivo):
        raise FileNotFoundError(f"No existe {archivo}")

    df = pd.read_csv(
        archivo,
        sep=r"\s+",
        engine="python"
    )

    return df.sort_values("bin_center")

def fpr_interpolado_en_tpr(y_true, y_proba, tpr_objetivo=0.70):
    """
    Devuelve FPR interpolado exactamente en TPR = tpr_objetivo.

    Útil para comparar contra curvas del paper.
    No devuelve una matriz de confusión real, porque el punto puede ser interpolado.
    """

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)

    fpr, tpr, thresholds = roc_curve(y_true, y_proba, pos_label=1)

    df_roc = pd.DataFrame({
        "TPR": tpr,
        "FPR": fpr
    })

    # Para TPR repetidos, quedarse con el menor FPR
    df_roc = (
        df_roc
        .groupby("TPR", as_index=False)["FPR"]
        .min()
        .sort_values("TPR")
    )

    if tpr_objetivo < df_roc["TPR"].min() or tpr_objetivo > df_roc["TPR"].max():
        return None

    fpr_interp = np.interp(
        tpr_objetivo,
        df_roc["TPR"].values,
        df_roc["FPR"].values
    )

    q_interp = tpr_objetivo / (np.sqrt(fpr_interp) + 1e-10)

    return {
        "threshold": None,
        "TPR": float(tpr_objetivo),
        "FPR": float(fpr_interp),
        "Q": float(q_interp),
        "modo": "interpolado"
    }


def encontrar_umbral_min_fpr(y_true, y_proba, tpr_objetivo=0.80, tolerancia_superior=0.02,n_points=1000, thr_min=0.001, thr_max=1.0): 
    thresholds = np.linspace(thr_min, thr_max, n_points) 
    # Expandimos el eje de y_proba para broadcasting 
    y_proba_exp = y_proba[:, np.newaxis] # shape: (N, 1) 
    thresholds_exp = thresholds[np.newaxis, :] # shape: (1, T) 
    # Predicciones binarias para todos los thresholds a la vez 
    y_pred_all = (y_proba_exp >= thresholds_exp).astype(int) # shape: (N, T) 
    # Repetimos y_true para cada columna (threshold) 
    y_true_rep = np.tile(y_true[:, np.newaxis], (1, n_points)) # shape: (N, T) 
    # TP, FN, FP, TN para cada threshold 
    TP = np.sum((y_true_rep == 1) & (y_pred_all == 1), axis=0) 
    FN = np.sum((y_true_rep == 1) & (y_pred_all == 0), axis=0) 
    FP = np.sum((y_true_rep == 0) & (y_pred_all == 1), axis=0) 
    TN = np.sum((y_true_rep == 0) & (y_pred_all == 0), axis=0) 
    TPR = TP / (TP + FN + 1e-10) 
    FPR = FP / (FP + TN + 1e-10) 
    Q = TPR / (np.sqrt(FPR) + 1e-10) 
    
    # Filtro de thresholds que cumplen con TPR mínimo 
    valid = (TPR >= tpr_objetivo- tolerancia_superior) & (TPR <= tpr_objetivo + tolerancia_superior)
    if not np.any(valid):
        valid = TPR >= tpr_objetivo

        if not np.any(valid):
            return None

        valid_indices = np.where(valid)[0]

        best_idx = valid_indices[
            np.lexsort((
                FPR[valid],
                np.abs(TPR[valid] - tpr_objetivo)
            ))
        ][0]

    else:
        valid_indices = np.where(valid)[0]

        # Dentro de la banda, minimizar FPR
        best_idx = valid_indices[np.argmin(FPR[valid])]
    return { 'threshold': thresholds[best_idx], 'TPR': float(TPR[best_idx]), 'FPR': float(FPR[best_idx]), 'Q': float(Q[best_idx]) } 

def obtener_zona(radio_nombre):

    if radio_nombre == "r_0_156m":
        return 1

    elif radio_nombre== "r_156_400m":
        return 2
    elif radio_nombre == "r_400_560m":
        return 3
    elif radio_nombre == "r_560_570m":
        return 4
    elif radio_nombre == "r_570_580m":
        return 5
    else:
        raise ValueError(f"Zona no definida para {radio_nombre}")
def encontrar_umbral_max_q(y_true, y_proba, tpr_min=0.5, n_points=400, thr_min=0.001, thr_max=1.0): 
    thresholds = np.linspace(thr_min, thr_max, n_points) 
    y_proba_exp = y_proba[:, np.newaxis] 
    thresholds_exp = thresholds[np.newaxis, :] 
    y_pred_all = (y_proba_exp >= thresholds_exp).astype(int) 
    y_true_rep = np.tile(y_true[:, np.newaxis], (1, n_points)) 
    TP = np.sum((y_true_rep == 1) & (y_pred_all == 1), axis=0) 
    FN = np.sum((y_true_rep == 1) & (y_pred_all == 0), axis=0) 
    FP = np.sum((y_true_rep == 0) & (y_pred_all == 1), axis=0) 
    TN = np.sum((y_true_rep == 0) & (y_pred_all == 0), axis=0) 
    TPR = TP / (TP + FN + 1e-10) 
    FPR = FP / (FP + TN + 1e-10) 
    Q = TPR / (np.sqrt(FPR) + 1e-10) 
    
    # Filtrar por condición TPR >= tpr_min 
    valid = TPR >= tpr_min 
    if not np.any(valid): 
        return None 
    # Elegir el umbral que maximiza Q 
    best_idx = np.argmax(Q * valid) # Q=0 para no válidos 
    return { 'threshold': thresholds[best_idx], 'TPR': float(TPR[best_idx]), 'FPR': float(FPR[best_idx]), 'Q': float(Q[best_idx]) } 


def graficar_metricas_vs_energia(model, X_test, y_test, energy_column, df_test,df_paper,zona,n_bins=7, carpeta=".", nombre_modelo="modelo", caso="CASO", version="v1",global_range=False): 
    energies = df_test[energy_column].values 
    y_proba = model.predict_proba(X_test)[:, 1] 
    bins = np.arange(2.0, 5.5 + 0.5, 0.5) 
    bin_centers = (bins[:-1] + bins[1:]) / 2 
    tpr_list, fpr_list, q_list = [], [], [] 
    
    for i in range(n_bins): 
        bin_center = bin_centers[i]
        TPR_OBJETIVO = obtener_tpr_objetivo(
            bin_center,
            zona,
            df_paper
        )
        if TPR_OBJETIVO is None:
            tpr_list.append(np.nan)
            fpr_list.append(np.nan)
            q_list.append(np.nan)
            continue
        mask = (energies >= bins[i]) & (energies < bins[i + 1]) 
        if np.sum(mask) == 0: 
            tpr_list.append(np.nan) 
            fpr_list.append(np.nan) 
            q_list.append(np.nan) 
            continue 
        y_true_bin = y_test[mask] 
        y_proba_bin = y_proba[mask] 
        # --- FPR @ TPR≥0.8 (igual que figura del paper) 
        res_thr08 =  fpr_interpolado_en_tpr(y_true_bin,y_proba_bin, tpr_objetivo=TPR_OBJETIVO)
        if res_thr08: 
            tpr_list.append(TPR_OBJETIVO) 
            fpr_list.append(res_thr08["FPR"]) 
        else: 
            tpr_list.append(np.nan) 
            fpr_list.append(np.nan) 
        
        # --- Q máximo con TPR≥0.8 (figura Q-factor del paper) 
        
        if res_thr08:
            q_list.append(res_thr08["Q"])
        else:
            q_list.append(np.nan)
        # Gráficos 
    graficar_tpr_vs_energia(bin_centers, tpr_list, nombre_modelo, carpeta, caso, version) 
    graficar_fpr_vs_energia(bin_centers, fpr_list, nombre_modelo, carpeta, caso, version) 
    graficar_q_vs_energia(bin_centers, q_list, nombre_modelo, carpeta, caso, version) 

################

def guardar_tiempos_csv(training_times, out_csv="output/training_times_summary.csv"):
    df_times = pd.DataFrame(training_times)
    if not df_times.empty:
        df_times.to_csv(out_csv, index=False)
    return df_times


def graficar_tiempos_por_energia(df_times, out_dir="graficos_tiempos"):
    os.makedirs(out_dir, exist_ok=True)

    if df_times.empty:
        return

    energias = df_times["energia_nombre"].dropna().unique()

    for energia in energias:
        df_e = df_times[df_times["energia_nombre"] == energia].copy()
        if df_e.empty:
            continue

        pivot = df_e.pivot_table(
            index="radio_nombre",
            columns="modelo",
            values="tiempo_seg",
            aggfunc="mean"
        )

        plt.figure(figsize=(10, 6))
        for modelo in pivot.columns:
            plt.plot(pivot.index, pivot[modelo], marker='o', label=modelo)

        plt.xlabel("Rango de distancia")
        plt.ylabel("Tiempo de entrenamiento (s)")
        plt.title(f"Tiempo de entrenamiento por modelo\nEnergía: {energia}")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/tiempo_modelos_por_radio__{energia}.png", dpi=600)
        plt.close()


def graficar_tiempos_por_radio(df_times, out_dir="graficos_tiempos"):
    os.makedirs(out_dir, exist_ok=True)

    if df_times.empty:
        return

    radios = df_times["radio_nombre"].dropna().unique()

    for radio in radios:
        df_r = df_times[df_times["radio_nombre"] == radio].copy()
        if df_r.empty:
            continue

        pivot = df_r.pivot_table(
            index="energia_nombre",
            columns="modelo",
            values="tiempo_seg",
            aggfunc="mean"
        )

        plt.figure(figsize=(11, 6))
        for modelo in pivot.columns:
            plt.plot(pivot.index, pivot[modelo], marker='o', label=modelo)

        plt.xlabel("Rango de energía")
        plt.ylabel("Tiempo de entrenamiento (s)")
        plt.title(f"Tiempo de entrenamiento por modelo\nRadio: {radio}")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/tiempo_modelos_por_energia__{radio}.png", dpi=600)
        plt.close()

def graficar_tiempos_combinacion(df_times, out_dir="graficos_tiempos"):
    os.makedirs(out_dir, exist_ok=True)

    if df_times.empty:
        return

    df_times = df_times.copy()
    df_times["combo"] = df_times["energia_nombre"] + "\n" + df_times["radio_nombre"]

    feature_sets = df_times["feature_set_num"].dropna().unique()

    for fs in feature_sets:
        df_fs = df_times[df_times["feature_set_num"] == fs].copy()
        if df_fs.empty:
            continue

        pivot = df_fs.pivot_table(
            index="combo",
            columns="modelo",
            values="tiempo_seg",
            aggfunc="mean"
        )

        plt.figure(figsize=(18, 7))
        for modelo in pivot.columns:
            plt.plot(pivot.index, pivot[modelo], marker='o', label=modelo)

        plt.xlabel("Combinación energía / radio")
        plt.ylabel("Tiempo de entrenamiento (s)")
        plt.title(f"Tiempos de entrenamiento por combinación - Feature set {fs}")
        plt.xticks(rotation=90)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/tiempos_combinacion_feature_set_{fs}.png", dpi=600)
        plt.close()

################
def score_q(model, X_val, y_val, tpr_target=0.80, n_points=400): 
    """ Calcula el Q-factor óptimo para un modelo entrenado, usando evaluación vectorizada.
      Retorna 0 si no alcanza el TPR objetivo. """ 
    if not hasattr(model, "predict_proba"):
        return 0 # El modelo no permite obtener probabilidades 
    
    y_proba = model.predict_proba(X_val)[:, 1] 
    res = encontrar_umbral_min_fpr( 
        y_val, y_proba, tpr_objetivo=tpr_target, n_points=n_points 
        ) 
    return res["Q"] if res else 0 
# === guardado del modelo con ambos thresholds ========================= 
# Para el WhitePaper, conviene que el modelo seleccione hiperparámetros 
# usando menor FPR con TPR ≥ 0.8, no Q. Cambia esa función por esta:
def score_fpr_at_tpr80(model, X_val, y_val, tpr_target, n_points=400):
    """
    Score para selección de hiperparámetros:
    busca el menor FPR posible manteniendo TPR >= 0.80.

    Retorna un score mayor cuando el FPR es menor.
    """
    if not hasattr(model, "predict_proba"):
        return -np.inf

    y_proba = model.predict_proba(X_val)[:, 1]

    res = encontrar_umbral_min_fpr(
        y_val,
        y_proba,
        tpr_objetivo=tpr_target,
        n_points=n_points
    )

    if res is None:
        return -np.inf

    return -res["FPR"]

def save_bundle(model, res_08, res_05, res_qopt, path): 
    bundle = { 
        "model": model, 
        "thr08": res_08["threshold"], 
        "TPR08": res_08["TPR"], 
        "FPR08": res_08["FPR"], 
        "Q08": res_08["Q"], 
        "thr05": None if res_05 is None else res_05["threshold"], 
        "TPR05": None if res_05 is None else res_05["TPR"], 
        "FPR05": None if res_05 is None else res_05["FPR"], 
        "Q05": None if res_05 is None else res_05["Q"], 
        "thrQopt": None if res_qopt is None else res_qopt["threshold"], 
        "TPRQopt": None if res_qopt is None else res_qopt["TPR"], 
        "FPRQopt": None if res_qopt is None else res_qopt["FPR"], 
        "QQopt": None if res_qopt is None else res_qopt["Q"], 
        } 
    joblib.dump(bundle, path) 


def manual_grid_search_xgb(X_train, y_train, X_val, y_val, param_grid,logger_model,tpr_objetivo ,max_rounds=100, patience=20): 
    import time 
    start_time = time.time() 
    logger_model.info("🟢 Inicio de entrenamiento XGBoost") 
    best_f1_global = -1 
    best_model_global = None 
    best_params_global = None 
    
    keys, values = zip(*param_grid.items()) 
    
    for combination in product(*values): 
        params = dict(zip(keys, combination)) 
        params.pop("n_estimators", None) 
        best_f1 = -1 
        no_improve_count = 0 
        model = None 
        
        for round_num in range(1, max_rounds + 1): 
            model = XGBClassifier( objective='binary:logistic',
                                   random_state=42, 
                                   use_label_encoder=False,
                                     n_estimators=round_num, # ir incrementando rondas 
                                     n_jobs=20, **params ) 
            
            model.fit(X_train, y_train, verbose=False) 
            #q_this = score_q(model, X_val, y_val, tpr_target=0.80) 
            score_this = score_fpr_at_tpr80(model, X_val, y_val, tpr_target=tpr_objetivo) 
            
            if score_this > best_f1: # mejor llamado best_q 
                best_f1 = score_this 
                best_model = model 
                no_improve_count = 0 
            else: no_improve_count += 1 
            
            
            if no_improve_count >= patience: 
                print(f"Early stopping for params {params} at round {round_num}") 
                break 
            
        if best_f1 > best_f1_global: 
            best_f1_global = best_f1 
            best_model_global = best_model 
            best_params_global = params 
    
    elapsed = time.time() - start_time 
    logger_model.info(f"🔴 Fin de entrenamiento XGBoost (Duración: {elapsed:.2f} s)") 
    logger_model.info(
    f"Mejores parámetros: {best_params_global} | Mejor FPR@TPR>=0.80 en validación: {-best_f1_global:.5f}"
    )

    print(f"\nMejores parámetros para XGBoost: {best_params_global}")
    print(f"Mejor FPR@TPR>={tpr_objetivo:.2f}en validación: {-best_f1_global:.5f}")
    return best_model_global, elapsed
def calcular_auc(model, X, y):
        """
        Calcula AUC usando ROC sobre un dataset dado.
        Retorna np.nan si el modelo no soporta predict_proba.
        """
        if not hasattr(model, "predict_proba"):
            return np.nan

        y_proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        return auc(fpr, tpr)
    
def evaluar_auc_splits(model, X_train, y_train, X_val, y_val, X_test, y_test):
    auc_train = calcular_auc(model, X_train, y_train)
    auc_val   = calcular_auc(model, X_val, y_val)
    auc_test  = calcular_auc(model, X_test, y_test)

    delta_train_test = auc_train - auc_test

    return {
        "AUC_train": auc_train,
        "AUC_val": auc_val,
        "AUC_test": auc_test,
        "Delta_train_test": delta_train_test
    }


def diagnosticar_overfitting(auc_dict, tol=0.02):
    """
    Determina si hay overfitting según diferencia AUC train-test.
    """
    delta = auc_dict["Delta_train_test"]

    if delta <= tol:
        return "NO overfitting"
    elif delta <= 0.05:
        return "Overfitting leve"
    else:
        return "Overfitting claro"

def plot_roc_train_val_test(
        model,
        X_train, y_train,
        X_val, y_val,
        X_test=None, y_test=None,
        nombre_modelo="modelo",
        carpeta=".",
        caso="CASO",
        version="v1"
    ):
        """
        Grafica curvas ROC para train, validation y opcionalmente test.
        """

        plt.figure(figsize=(8, 7))

        # ===== TRAIN =====
        y_proba_train = model.predict_proba(X_train)[:, 1]
        fpr_tr, tpr_tr, _ = roc_curve(y_train, y_proba_train)
        auc_tr = auc(fpr_tr, tpr_tr)
        plt.plot(
            fpr_tr, tpr_tr,
            label=f"Train (AUC={auc_tr:.3f})",
            linewidth=2
        )

        # ===== VALIDATION =====
        y_proba_val = model.predict_proba(X_val)[:, 1]
        fpr_va, tpr_va, _ = roc_curve(y_val, y_proba_val)
        auc_va = auc(fpr_va, tpr_va)
        plt.plot(
            fpr_va, tpr_va,
            label=f"Validation (AUC={auc_va:.3f})",
            linewidth=2,
            linestyle="--"
        )

        # ===== TEST (opcional) =====
        if X_test is not None and y_test is not None:
            y_proba_test = model.predict_proba(X_test)[:, 1]
            fpr_te, tpr_te, _ = roc_curve(y_test, y_proba_test)
            auc_te = auc(fpr_te, tpr_te)
            plt.plot(
                fpr_te, tpr_te,
                label=f"Test (AUC={auc_te:.3f})",
                linewidth=2,
                linestyle=":"
            )

        # ===== Línea aleatoria =====
        plt.plot([0, 1], [0, 1], 'k--', lw=1)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Train / Validation / Test\n{nombre_modelo} – {caso}")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()

        # ===== Guardar =====
        plt.savefig(
            f"{carpeta}/ROC_Train_Val_{nombre_modelo}_{caso}_{version}.png",
            dpi=600
        )
        plt.close()
def check_probability_calibration(models, names,X_test,y_test): 
        for model, name in zip(models, names): 
            y_proba = model.predict_proba(X_test)[:, 1] 
            mean_proba = np.mean(y_proba) 
            mean_true = np.mean(y_test) 
            print(f"{name}:") 
            print(f" Promedio de y_proba: {mean_proba:.4f}") 
            print(f" Promedio real (y_test): {mean_true:.4f}") 
            print(f" Diferencia absoluta: {abs(mean_proba - mean_true):.4f}\n") 
def plot_feature_importances(model, name, feature_names,caso): 
        if hasattr(model, 'feature_importances_'): 
            importances = model.feature_importances_ 
            indices = np.argsort(importances)[::-1] 
            plt.figure(figsize=(10,6)) 
            plt.title(f'Importancia de variables-{caso}') 
            sns.barplot(x=importances[indices], y=np.array(feature_names)[indices]) 
            plt.tight_layout() 
            plt.savefig(f'{carpeta}/feature_importance_V2_{name}_{caso}_{version}.png', dpi=600) 
            plt.close() 
def plot_feature_importance_gini(model, feature_names,name ,caso,top_n=20): 
        # Obtener importancias 
        importances = model.feature_importances_ 
        df_importance = pd.DataFrame({ 
            'Feature': feature_names, 
            'GiniImportance': importances 
            }).sort_values(by='GiniImportance', ascending=False).head(top_n) 
        # Gráfico de barras 
        plt.figure(figsize=(10, 6)) 
        plt.barh(df_importance['Feature'][::-1], 
                 df_importance['GiniImportance'][::-1], 
                 color='skyblue') 
        plt.xlabel('Importancia tipo Gini') 
        plt.title(f'Top {top_n} características más importantes según Gini-{caso}') 
        plt.grid(axis='x') 
        plt.tight_layout() 
        plt.savefig(f'{carpeta}/feature_importance_gini_V2_{name}_{caso}_{version}.png', dpi=600) 
        plt.close() 
def registrar_tiempo_entrenamiento(
    modelo, elapsed, caso, version, carpeta,
    energia_nombre=None, radio_nombre=None,
    emin=None, emax=None, rmin=None, rmax=None,
    feature_set_num=None, feature_set_nombre=None
):
    training_times.append({
        "modelo": modelo,
        "tiempo_seg": float(elapsed),
        "caso": caso,
        "version": version,
        "carpeta": carpeta,
        "energia_nombre": energia_nombre,
        "radio_nombre": radio_nombre,
        "emin": emin,
        "emax": emax,
        "rmin": rmin,
        "rmax": rmax,
        "feature_set_num": feature_set_num,
        "feature_set_nombre": feature_set_nombre
    })

def ejecutar_pipeline(caso, features ,version,carpeta,emin,emax,rmin=None,rmax=None,energia_nombre=None,radio_nombre=None,
    feature_set_num=None, feature_set_nombre=None): 

    zona = obtener_zona(radio_nombre)
    df_paper = cargar_curva_paper(zona)
    bin_center_pipeline = (emin + emax) / 2

    TPR_OBJETIVO = obtener_tpr_objetivo(
        bin_center_pipeline,
        zona,
        df_paper
    )
    if TPR_OBJETIVO is None:
        logger.warning(
             f"No existe TPR para bin_center={bin_center_pipeline}"
        )
        return
    print(f"Zona detectada: {zona}")
    if df_paper is not None:
      print(df_paper)


    print(f"Zona detectada: {zona}")
    print(f"TPR objetivo leído desde TXT: {TPR_OBJETIVO}")
    print(f"\n--- Ejecutando pipeline para: {caso} | Filtro: {version} ---\n") 
    # === 1. Cargar y filtrar datos === 
    data = pd.read_csv('/home/swgo_2024/RESULTADOS/ML_aerie_gamma_proton_v7_new_variables.csv', sep=",") 
    data = data[data["ID"].astype(str).str.startswith("0_")] #filtro de unicamente tener el set {0} del conjunto {0,1,2,3,4}
    mask=(data['rec.LHLatDistFitEnergy'] > emin) & (data['rec.LHLatDistFitEnergy'] <= emax) 
    #data=data[mask] 
    data = data[mask] 
    if data.empty: 
        logging.warning(f"[{caso}-{version}] Dataset vacío después del filtro de energía.") 
        return 
    
    # Aplicar filtro de distancia si rmin y rmax están definidos 
    
    if rmin is not None and rmax is not None: 
        if 'rec.coreX' not in data.columns or 'rec.coreY' not in data.columns: 
            logging.warning(f"[{caso}-{version}] Faltan columnas 'rec.coreX' o 'rec.coreY'.") 
            return 
        
        r = np.sqrt(data['rec.coreX']**2 + data['rec.coreY']**2) 
        mask = (r >= rmin) & (r < rmax) 
        data = data[mask] 
        
    X=data[features]
    y= 1 - data['Type_Particle'] #estp cambia a gamma a 1 y hadron a 0 
    #añadir un conjunto de validacion (ok) 
    # === 4. División entrenamiento/prueba === 
    X_train_val, X_test, y_train_val, y_test = train_test_split( X, y, test_size=0.2, stratify=y, random_state=42 ) 
    # Guardar índices del test antes de escalar 
    test_indices = X_test.index 
    
    # Luego dividimos entrenamiento+validación en entrenamiento y validación 
    X_train, X_val, y_train, y_val = train_test_split( X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42 ) 
    df_test = data.loc[test_indices].copy() 
    
    # === 5. Escalar === 
    scaler = StandardScaler() 
    X_train = scaler.fit_transform(X_train) 
    X_test = scaler.transform(X_test) 
    X_val= scaler.transform(X_val) 
    # Convertir X_train escalado a DataFrame con nombres de columnas 
    # Volvemos a DataFrame para poder aplicar la función variable por variable 
    
    X_train_df = pd.DataFrame(X_train, columns=features) 
    y_train_series = pd.Series(y_train).reset_index(drop=True) 
    X_train_discretized = X_train_df.copy() 
    for col in X_train_df.columns: 
        X_train_discretized[col] = pd.qcut(X_train_df[col], q=10, duplicates='drop') 
        # 10 quantiles 
        # Ahora la función corregida 
    
    
    def gini_index(feature, target): 
        df = pd.DataFrame({'feature': feature, 'target': target}) 
        gini_total = 0 
        values = df['feature'].unique() 
        
        for v in values: 
            subset = df[df['feature'] == v] 
            size = len(subset) 
        
            if size == 0: 
                continue 
            p = subset['target'].value_counts(normalize=True) 
            gini = 1 - np.sum(p**2) 
            gini_total += (size / len(df)) * gini 
        return gini_total 
    
    # Calcular Gini para cada variable 
    """gini_scores = { col: gini_index(X_train_discretized[col],
      y_train_series) for col in X_train_discretized.columns } 
      gini_df = pd.DataFrame.from_dict(gini_scores, 
      orient='index', columns=['GiniIndex']) 
      gini_df = gini_df.sort_values(by='GiniIndex', 
      ascending=True) 
      print(gini_df) 
    """ 
    # === 6. Modelos y parámetros para RandomizedSearch === 
    lgbm = LGBMClassifier(objective='binary',random_state=42,force_col_wise=True,n_jobs=20)#, metric='logloss') 
    lgbm_params = { 
        'n_estimators': [500, 800, 1200], 
        'learning_rate': [0.01, 0.05, 0.1], 
        'max_depth': [3, 6, 8, 10], 
        'num_leaves': [20, 31, 50] } 
    # 
    #xgb = XGBClassifier(objective='binary:logistic', random_state=42, eval_metric='logloss') 
    xgb_params = { 
        'n_estimators': [100, 300, 500], 
        'learning_rate': [0.01, 0.05, 0.1], 
        'max_depth': [3, 6, 8, 10], 
        'subsample': [0.7, 0.8, 1.0] 
    } 
    
    cat = CatBoostClassifier(loss_function='Logloss', random_state=42, verbose=0,thread_count=20) 
    cat_params = { 
        'iterations': [100, 300, 500], 
        'learning_rate': [0.01, 0.05, 0.1], 
        'depth': [3, 6, 8, 10] 
    } 
    #añadir el early stopping round 
    # === 7. Función para búsqueda con early stopping === 
    # 
    from copy import deepcopy 
    def grid_search_with_early_stopping(model, params, name,logger_model,tpr_objetivo): 
        import time 
        start_time = time.time() 
        logger_model.info(f"🟢 Inicio de entrenamiento {name}") 
        best_score = -1 
        best_model = None 
        best_params = None 
        from itertools import product 
        
        # Guardar la clase del modelo original 
        model_class = type(model) 
        base_params = model.get_params() 
        keys = list(params.keys()) 
        values = list(params.values()) 
        
        for combination in product(*values): 
            param_dict = dict(zip(keys, combination)) 
            # Crear nueva instancia del modelo con los parámetros combinados 
            new_params = {**base_params, **param_dict} 
            current_model = model_class(**new_params) 
            

            if name == "LightGBM": 
                current_model.fit(X_train, y_train,
                                   eval_set=[(X_val, y_val)], 
                                   eval_metric='logloss', 
                                   callbacks=[early_stopping(stopping_rounds=20),log_evaluation(0)], 
                                   ) 
            elif name == "CatBoost": 
                current_model.fit(X_train, y_train,
                                   eval_set=(X_val, y_val), 
                                   early_stopping_rounds=20, 
                                   verbose=0) 
                
            
            score = score_fpr_at_tpr80(current_model, X_val, y_val, tpr_target=tpr_objetivo) 
            if score > best_score: 
                best_score = score 
                best_model = current_model 
                best_params = param_dict 
        
        elapsed = time.time() - start_time 
        
        logger_model.info(f"🔴 Fin de entrenamiento {name} (Duración: {elapsed:.2f} s)") 
        logger_model.info(f"Mejores parámetros: {best_params} | Mejor FPR@TPR>=0.80 en validación: {-best_score:.5f}") 
        
        
        print(f"\nMejores parámetros para {name}: {best_params}") 
        print(f"Mejor FPR@TPR>=0.80 en validación: {-best_score:.5f}")
        return best_model, elapsed
    
    
    # Buscar mejores modelos 
    print("\n----- Buscando hiperparámetros (Grid Search + Early Stopping) -----\n") 
    
    
    best_lgbm,elapsed_lgbm = grid_search_with_early_stopping(lgbm, lgbm_params, "LightGBM",logger_lgbm,TPR_OBJETIVO) 
    registrar_tiempo_entrenamiento(
    modelo="LightGBM",
    elapsed=elapsed_lgbm,
    caso=caso,
    version=version,
    carpeta=carpeta,
    energia_nombre=energia_nombre,
    radio_nombre=radio_nombre,
    emin=emin, emax=emax,
    rmin=rmin, rmax=rmax,
    feature_set_num=feature_set_num,
    feature_set_nombre=feature_set_nombre
    )
    best_xgb,elapsed_xgb  = manual_grid_search_xgb(X_train, y_train, X_val, y_val, xgb_params,logger_xgb,TPR_OBJETIVO) 
    registrar_tiempo_entrenamiento(
    modelo="XGBoost",
    elapsed=elapsed_xgb,
    caso=caso,
    version=version,
    carpeta=carpeta,
    energia_nombre=energia_nombre,
    radio_nombre=radio_nombre,
    emin=emin, emax=emax,
    rmin=rmin, rmax=rmax,
    feature_set_num=feature_set_num,
    feature_set_nombre=feature_set_nombre
    )
    best_cat,elapsed_cat  = grid_search_with_early_stopping(cat, cat_params, "CatBoost",logger_cat,TPR_OBJETIVO) 
    registrar_tiempo_entrenamiento(
    modelo="CatBoost",
    elapsed=elapsed_cat,
    caso=caso,
    version=version,
    carpeta=carpeta,
    energia_nombre=energia_nombre,
    radio_nombre=radio_nombre,
    emin=emin, emax=emax,
    rmin=rmin, rmax=rmax,
    feature_set_num=feature_set_num,
    feature_set_nombre=feature_set_nombre
    )
    # 

    
    
    
    def evaluate_model(model, name, X_test, y_test, df_test, carpeta, caso, version,tpr_objetivo,df_paper): 
        resultado = None 
        preds = model.predict(X_test) 
        report = classification_report(y_test, preds, digits=5) 
        matrix = confusion_matrix(y_test, preds, labels=[0, 1]) 
        output = f"\n===== {name} =====\n" 
        output += report + "\n" 
        output += "Matriz de confusión:\n" + str(matrix) + "\n" 
        
        
        # === Análisis adicional con predict_proba === 
        # === Evaluación TPR ≥ 0.8 === 
        if hasattr(model, "predict_proba"): 
            y_proba = model.predict_proba(X_test)[:, 1] 
            # clase 1 = gamma 
            
            resultado_interp = fpr_interpolado_en_tpr(y_test,y_proba,tpr_objetivo=tpr_objetivo)
            if resultado_interp:
                output += "\n--- FPR interpolado a TPR = 0.8 para comparación con whitepaper ---\n"
                output += "Threshold: interpolado; no aplica matriz de confusión real\n"
                output += f"TPR (ε_γ): {resultado_interp['TPR']:.3f}\n"
                output += f"FPR (ε_p): {resultado_interp['FPR']:.6f}\n"
                output += f"Q-factor: {resultado_interp['Q']:.3f}\n"
                output += f"Modo: {resultado_interp['modo']}\n"
            else:
                output += "\n--- No se pudo interpolar FPR en TPR = 0.8 ---\n"
            df_fpr = cargar_curva_fpr(zona)

            if df_fpr is not None:

                FPR_OBJETIVO = obtener_fpr_objetivo(
                    bin_center_pipeline,
                    zona,
                    df_fpr
                )

                resultado_fpr = tpr_interpolado_en_fpr(
                    y_test,
                    y_proba,
                    fpr_objetivo=FPR_OBJETIVO
                )

                if resultado_fpr:

                    output += "\n--- TPR interpolado a FPR objetivo ---\n"

                    output += (
                        f"FPR objetivo: "
                        f"{resultado_fpr['FPR']:.6f}\n"
                    )

                    output += (
                        f"TPR interpolado: "
                        f"{resultado_fpr['TPR']:.6f}\n"
                    )

                    output += (
                        f"Q-factor: "
                        f"{resultado_fpr['Q']:.6f}\n"
                    )

            resultado = encontrar_umbral_min_fpr(y_test, y_proba, tpr_objetivo=tpr_objetivo) 


            if resultado: 
                thr = resultado["threshold"] 
                y_pred_thr = (y_proba >= thr).astype(int) 
                matrix_thr = confusion_matrix(y_test, y_pred_thr, labels=[0, 1]) 
                report_thr = classification_report(y_test, y_pred_thr, digits=5) 
                output += "\n--- Umbral optimizado para TPR ≥ 0.8 ---\n" 
                output += f"Threshold: {resultado['threshold']:.3f}\n" 
                output += f"TPR (ε_γ): {resultado['TPR']:.3f}\n" 
                output += f"FPR (ε_p): {resultado['FPR']:.3f}\n" 
                output += f"Q-factor: {resultado['Q']:.3f}\n" 
                output += "\nMatriz de confusión (thr óptimo):\n" + str(matrix_thr) + "\n" 
                output += report_thr + "\n" 
                
            else: 
                output += "\n--- No se encontró umbral con TPR ≥ 0.8 ---\n" 
            # === Evaluación adicional para TPR ≥ 0.5 (comparación con Glombitza fig. 9) === 
            
            resultado_05 = encontrar_umbral_min_fpr(y_test, y_proba, tpr_objetivo=0.5) 
            if resultado_05: 
                output += "\n--- Umbral alternativo para TPR ≥ 0.5 (comparación paper) ---\n" 
                output += f"Threshold (TPR≥0.5): {resultado_05['threshold']:.3f}\n" 
                output += f"TPR (ε_γ): {resultado_05['TPR']:.3f}\n" 
                output += f"FPR (ε_p): {resultado_05['FPR']:.3f}\n" 
                output += f"Q-factor: {resultado_05['Q']:.3f}\n" 
            
            # === Q-factor optimizado con TPR >= 0.5 === 
            resultado_qopt = encontrar_umbral_max_q(y_test, y_proba, tpr_min=0.5) 
            
            if resultado_qopt: 
                thr_qopt = resultado_qopt['threshold'] 
                y_pred_qopt = (y_proba >= thr_qopt).astype(int) 
                # Matriz y reporte 
                matrix_qopt = confusion_matrix(y_test, y_pred_qopt, labels=[0, 1]) 
                report_qopt = classification_report(y_test, y_pred_qopt, digits=5) 
                output += "\n--- Umbral que maximiza Q con TPR ≥ 0.5 ---\n" 
                output += f"Threshold: {resultado_qopt['threshold']:.3f}\n" 
                output += f"TPR (ε_γ): {resultado_qopt['TPR']:.3f}\n" 
                output += f"FPR (ε_p): {resultado_qopt['FPR']:.3f}\n" 
                output += f"Q-factor: {resultado_qopt['Q']:.3f}\n" 
                output += "\nMatriz de confusión (Q-opt):\n" + str(matrix_qopt) + "\n" 
                output += report_qopt + "\n" 
            if resultado:
            # Guardar matriz óptima como heatmap 
                plt.figure(figsize=(5, 4)) 
                sns.heatmap(matrix_thr, annot=True, 
                            fmt='d', 
                            cmap="Blues", 
                            xticklabels=["Hadron (0)", "Gamma (1)"], 
                            yticklabels=["Hadron (0)", "Gamma (1)"]) 
                plt.xlabel('Predicción') 
                plt.ylabel('Verdadero') 
                plt.title(f'Matriz óptima - {name} (thr={thr:.3f})') 
                plt.tight_layout() 
                plt.savefig(f'{carpeta}/confusion_matrix_opt_{name}_{caso}_{version}.png', dpi=600) 
                plt.close() 
            # === Generar gráfico TPR/FPR vs energía === 
            graficar_metricas_vs_energia( 
                model=model, 
                X_test=X_test, 
                y_test=y_test, 
                energy_column="rec.LHLatDistFitEnergy", 
                df_test=df_test, # Asegúrate que esté disponible y alineado 
                df_paper=df_paper,
                zona=zona,
                n_bins=7, 
                #threshold=resultado['threshold'], 
                # usar el umbral óptimo 
                carpeta=carpeta, 
                nombre_modelo=name, 
                caso=caso, 
                version=version ) 
            df_fpr = cargar_curva_fpr(zona)

            if df_fpr is not None:

                graficar_metricas_vs_energia_fpr(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    energy_column="rec.LHLatDistFitEnergy",
                    df_test=df_test,
                    df_fpr=df_fpr,
                    zona=zona,
                    n_bins=7,
                    carpeta=carpeta,
                    nombre_modelo=name,
                    caso=caso,
                    version=version
                )
        else: 
            output += "\n--- Modelo no soporta predict_proba ---\n" 
        
        # Imprimir en consola y guardar en archivo 
        print(output) 
        with open(f"{carpeta}/evaluacion_matriz_{name}_{caso}_{version}.txt", 'w') as f: 
            f.write(output) 
        # ---- guardar modelo + thresholds ------------------------------ 
        
        if resultado: # solo si existe TPR≥0.8 
            # nombre base del archivo 
            
            pkl_name = { 
                'LightGBM': 'best_lgbm_with_thr.pkl', 
                'XGBoost': 'best_xgb_with_thr.pkl', 
                'CatBoost': 'best_cat_with_thr.pkl', 
                'VotingClassifier (Soft Voting)': 'voting_classifier_with_thr.pkl' 
                }[name] 
            save_bundle(model, resultado, resultado_05,resultado_qopt, f"{carpeta}/{pkl_name}") 
            
        # --------------------------------------------------------------- 
        return resultado 
    def plot_confusion_matrices(models, model_names, X_test, y_test, labels=None): 
        for model, name in zip(models, model_names): 
            y_pred = model.predict(X_test) 
            cm = confusion_matrix(y_test, y_pred, labels=labels) 
            f1 = f1_score(y_test, y_pred) 
            precision = precision_score(y_test, y_pred) 
            recall = recall_score(y_test, y_pred) 
            plt.figure(figsize=(6, 5)) 
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                        xticklabels=labels if labels else 'auto', 
                        yticklabels=labels if labels else 'auto') 
            plt.xlabel('Predicción') 
            plt.ylabel('Verdadero') 
            plt.title(f'Matriz de Confusión - {name}-{caso}\n' 
                      f'F1: {f1:.3f} | Precisión: {precision:.3f} | Recall: {recall:.3f}') 
            plt.tight_layout() 
            filename = f'{carpeta}/confusion_matrix_{name}_{caso}_{version}.png' 
            plt.savefig(filename, dpi=600) 
            plt.close() 
    # === 9. Votación simple === 
    voting = VotingClassifier( 
        estimators=[ 
            ('lgbm', best_lgbm), 
            ('xgb', best_xgb), 
            ('cat', best_cat) 
            ], 
            voting='soft', 
            n_jobs=20 
            ) 
    start_voting = time.time() 
    logger_voting.info("🟢 Inicio de entrenamiento VotingClassifier") 
    voting.fit(X_train, y_train) 
    elapsed_voting = time.time() - start_voting 
    
    registrar_tiempo_entrenamiento(
    modelo="VotingClassifier",
    elapsed=elapsed_voting,
    caso=caso,
    version=version,
    carpeta=carpeta,
    energia_nombre=energia_nombre,
    radio_nombre=radio_nombre,
    emin=emin, emax=emax,
    rmin=rmin, rmax=rmax,
    feature_set_num=feature_set_num,
    feature_set_nombre=feature_set_nombre
    )
    logger_voting.info(f"🔴 Fin de entrenamiento VotingClassifier (Duración: {elapsed_voting:.2f} s)") 
    # Evaluación individual con thresholds (TXT y métricas) 
    # === Diagnóstico de overfitting por AUC ===

    for model, name in [
        (best_lgbm, "LightGBM"),
        (best_xgb, "XGBoost"),
        (best_cat, "CatBoost"),
        (voting, "VotingClassifier")
    ]:
        auc_info = evaluar_auc_splits(
            model,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test
        )

        estado = diagnosticar_overfitting(auc_info)

        with open(f"{carpeta}/diagnostico_auc_{name}_{caso}_{version}.txt", "w") as f:
            f.write(f"AUC train: {auc_info['AUC_train']:.5f}\n")
            f.write(f"AUC val  : {auc_info['AUC_val']:.5f}\n")
            f.write(f"AUC test : {auc_info['AUC_test']:.5f}\n")
            f.write(f"Delta train-test: {auc_info['Delta_train_test']:.5f}\n")
            f.write(f"Diagnóstico: {estado}\n")


    evaluate_model(best_lgbm, "LightGBM", X_test, y_test, df_test, carpeta, caso, version,TPR_OBJETIVO,df_paper) 
    evaluate_model(best_xgb, "XGBoost", X_test, y_test, df_test, carpeta, caso, version,TPR_OBJETIVO,df_paper) 
    evaluate_model(best_cat, "CatBoost", X_test, y_test, df_test, carpeta, caso, version,TPR_OBJETIVO,df_paper) 
    res_voting = evaluate_model(voting, "VotingClassifier (Soft Voting)", X_test, y_test, df_test, carpeta, caso, version,TPR_OBJETIVO,df_paper) 
    
    if res_voting: 
        bundle = { 
            "model": voting, 
            "threshold": res_voting["threshold"], 
            "TPR": res_voting["TPR"], 
            "FPR": res_voting["FPR"], 
            "Q": res_voting["Q"] 
        } 
        joblib.dump(bundle, f"{carpeta}/voting_classifier_with_thr.pkl") 
        
    plot_confusion_matrices( 
        models=[best_lgbm, best_xgb, best_cat,voting], 
        model_names=['LightGBM', 'XGBoost', 'CatBoost','VotingClassifier'], 
        X_test=X_test, 
        y_test=y_test, 
        labels=[0, 1], 
    ) 
    
    #analizar el y_proba 
    #el prom(y_proba) debe ser similar al promedio del (y_test) 
    #usar el gini para los tres para que esté standarizado (ok) 
    # === 10. Curvas ROC y Gini === 
    
   

    

    # Ejecutar la función 
    plot_roc_train_val_test(
        model=best_lgbm,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,  
        nombre_modelo="LightGBM",
        carpeta=carpeta,
        caso=caso,
        version=version
        )
    plot_roc_train_val_test(
        model=best_xgb,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,  
        nombre_modelo="XGBoost",
        carpeta=carpeta,
        caso=caso,
        version=version
    )
    plot_roc_train_val_test(
        model=best_cat,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        nombre_modelo="CatBoost",
        carpeta=carpeta,
        caso=caso,
        version=version
    )
    plot_roc_train_val_test(
        model=voting,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        nombre_modelo="VotingClassifier",
        carpeta=carpeta,
        caso=caso,
        version=version
    )



    
    
    # Ejecutar la función 
    
    check_probability_calibration(
        [best_lgbm, best_xgb, best_cat],
        ['LightGBM', 'XGBoost', 'CatBoost'],
        X_test, y_test
    )

    #lgbm: gain 
    #xgboost: gain 
    #catboost: gini 
    #usar el gini para los tres para que esté standarizado, rango estimado(buscar) 
    # === 11. Importancia de características === 
    
    
    
    
    print("\n--- Importancia de características ---\n") 
    plot_feature_importances(best_lgbm, 'LightGBM', features,caso) 
    plot_feature_importances(best_xgb, 'XGBoost', features,caso) 
    plot_feature_importances(best_cat, 'CatBoost', features,caso) 
    
    # === 12. Calculo de Gini === 
    
    
    plot_feature_importance_gini(best_lgbm, features,"lgbm",caso) 
    plot_feature_importance_gini(best_xgb,features,"xgb",caso) 
    plot_feature_importance_gini(best_cat,features,"cat",caso) 
    
    # Guardar los modelos 
    # joblib.dump(best_lgbm, f'{carpeta}/best_lgbm_model_V2.pkl') 
    # joblib.dump(best_xgb, f'{carpeta}/best_xgb_model_V2.pkl') 
    # joblib.dump(best_cat, f'{carpeta}/best_cat_model_V2.pkl') 
    # joblib.dump(voting, f'{carpeta}/voting_classifier_V2.pkl') 
    
    print("Modelos guardados exitosamente.") 
with open("feature_sets_rec.json", "r") as f: 
    feature_sets = json.load(f) 
rangos_energia = [ 
        (f"version_1_5TeV_{muestra}", 2.0, 2.5), 
        (f"version_5_10TeV_{muestra}", 2.5, 3.0), 
        (f"version_10-15TeV_{muestra}", 3.0, 3.5), 
        (f"version_15_20TeV_{muestra}", 3.5, 4.0), 
        (f"version_20_25TeV_{muestra}", 4.0, 4.5), 
        (f"version_25_30TeV_{muestra}", 4.5, 5.0), 
        (f"version_30_35TeV_{muestra}", 5.0, 5.5) 
        # (f"version_60_65TeV_{muestra}", (60000), (65000)), 
        # (f"version_65-70TeV_{muestra}", (65000), (70000)), 
        # (f"version_70_75TeV_{muestra}", (70000), (75000)), 
        # (f"version_75_80TeV_{muestra}", (75000), (80000)), 
        # (f"version_80_85TeV_{muestra}", (80000), (85000)), 
        # (f"version_85_90TeV_{muestra}", (85000), (90000)) 
        ] 
rango_radio =[ 
        ("r_0_156m",0,156), 
        ("r_156_400m",156,400), 
        ("r_400_560m",400,560), 
        ("r_560_570m",560,570), 
        ("r_570_580m",570,580) 
        ] 
for num, config in feature_sets.items(): 
    nombre = config["nombre"] 
    features = config["features"] 
    logger.info(f"▶️ Iniciando feature set {num} – {nombre}") 
    for energia_nombre,emin,emax in rangos_energia: 
        for radio_nombre,rmin, rmax in rango_radio: 
            version =f"{energia_nombre}_{radio_nombre}" 
            carpeta = f"{num}_{muestra}/{version}" 
            os.makedirs(carpeta, exist_ok=True) 
            logger.info(f" ↳ Empieza versión {version}") 
            try: 
                ejecutar_pipeline( 
                    nombre, 
                    features, 
                    version, 
                    carpeta, 
                    emin=emin, 
                    emax=emax, 
                    rmin=rmin, 
                    rmax=rmax,
                    energia_nombre=energia_nombre,
                    radio_nombre=radio_nombre,
                    feature_set_num=num,
                    feature_set_nombre=nombre ) 
                logger.info(f" Versión {carpeta} COMPLETADA") 
            except Exception: 
                logger.exception(f" Versión {carpeta} FALLÓ (ver traza)")
    
df_times = guardar_tiempos_csv(training_times, out_csv="output/training_times_summary.csv")
graficar_tiempos_por_energia(df_times, out_dir="output")
graficar_tiempos_por_radio(df_times, out_dir="output")
graficar_tiempos_combinacion(df_times, out_dir="output")
print("Gráficos de tiempos generados correctamente.")
