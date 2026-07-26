import os
import time
import resource
from pathlib import Path
from functools import wraps
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import uproot
from tqdm import tqdm
import sys
import logging

# ---- CONFIGURACIÓN DE LOG ----
LOG_PATH = "/home/swgo_2024/RESULTADOS/missing_files.log"
logging.basicConfig(
    filename=LOG_PATH,
    filemode="a",                    # añadir al final
    level=logging.WARNING,           # solo avisos/errores
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.warning("✦ TEST DE LOG ✦")
# ==================== MEMORIA ========================

def limitar_memoria(max_mem_gb=240):
    """Limita el uso de memoria del proceso (Linux/macOS)."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (int(max_mem_gb * 1024**3), hard))

# ==================== DECORADORES ====================

def timed(func):
    """Decorator para medir el tiempo de ejecución de las funciones."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        print(f"[{func.__name__}] tiempo: {time.time() - t0:.2f}s")
        return res
    return wrapper

# ==================== CÁLCULO DE FEATURES ====================
#EDITADO 23/01/2025
@timed
def extract_basic_features(df_rec):
    """
    Calcula R, hits<=40, energía máxima fuera de 40, PICness y Rmax_aerie
    en un solo groupby-agg.
    """
    df = df_rec.copy()
    df['R'] = np.sqrt(df['event.hit.xPMT']**2 + df['event.hit.yPMT']**2) * 0.01
    
    df['R_shower'] = np.sqrt(
        (df['event.hit.xPMT']* 0.01 - df['rec.coreX'])**2 +
        (df['event.hit.yPMT']* 0.01 - df['rec.coreY'])**2
    ) 
    def agg_funcs(g):
        
        # Compactness
        # Obtener coordenadas del core del evento
        x_core = g['rec.coreX'].iat[0]
        y_core = g['rec.coreY'].iat[0]

        # Calcular distancia desde el core (R_shower)
        g['R_shower'] = np.sqrt(
            (g['event.hit.xPMT']*0.01 - x_core)**2 +
            (g['event.hit.yPMT']*0.01 - y_core)**2
        )  # convertir a metros
        
        g['outside40'] = g['R_shower'] > 40
        # g['outer_charge'] = g['event.hit.charge'].where(~g['hit40'], 0.0)
        # Compactness
        n_hit = len(g)
        cxpe_40 = g.loc[g['outside40'], 'event.hit.charge'].max()
        compact = np.nan if pd.isna(cxpe_40) or cxpe_40 <= 0 else n_hit / cxpe_40


        # Rmax_aerie

        r_vals = np.sqrt(
            (g['event.hit.xPMT']*0.01 - x_core)**2 +
            (g['event.hit.yPMT']*0.01 - y_core)**2
        )
        rmax = r_vals.max() if len(r_vals) > 0 else np.nan

        return pd.Series({
            'Compactness': compact,
            'Rmax_aerie': rmax
        })

    result = df.groupby('id_correlativo', group_keys=False).apply(agg_funcs).reset_index()
    return result


# ==================== FUNCIONES AUXILIARES ====================



def load_root(path_reco):
    print(f"→ Leyendo ROOT: {path_reco}")
    up = uproot.open(path_reco)
    df_rc = up["XCDF;1"].arrays(library="pd")
    # print(f"  columnas Reco: {df_rc.columns.tolist()}")
    return df_rc
# ================= LIMPIEZA RECO (nivel hit) =================

import ast

def parse_numeric_array(x):
    if isinstance(x, np.ndarray):
        return x.astype(float)
    if isinstance(x, list):
        return np.array(x, dtype=float)
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("[") and x.endswith("]"):
            vals = np.fromstring(x.strip("[]").replace(",", " "), sep=' ')
            return vals.astype(float)
    return np.array([], dtype=float)

# ==================== PROCESAMIENTO POR BLOQUE ====================

@timed
def procesar_un_indice(suffix: str, particula: str, tipo: int, carpeta_salida: Path):
    print(f"\n=== Procesando suffix: {suffix} ===")
    
    df_reco_list =  []

    for bloque in range(5):
        base = f"DAT{suffix}_D8_{particula}_{bloque}_50000"
        #pe_p = f"/home/swgo_2024/DATOS/M7_production_D8_{particula}_photoelectrons/hawcsim-{base}_photoelectrons.parquet"
        #ev_p = f"" #f"/home/swgo_2024/DATOS/M7_production_D8_{particula}_events/hawcsim-{base}_events.parquet"
        #wh_p = f""#f"/home/swgo_2024/DATOS/M7_production_D8_{particula}_waterhits/hawcsim-{base}_waterhits.parquet"
        rc_p = f"/home/swgo_2024/DATOS/M7_reco_D8_{particula}_V2_new_variables/reco-{base}.pkl"

        missing = [os.path.basename(p) for p in [rc_p] if not os.path.exists(p)]
    #     paths = {
    #     "photoelectrons": pe_p,
    #     "reco": rc_p,
    #    }

      

        if missing:
            msg = (f"Bloque incompleto -> particula='{particula}', suffix='{suffix}', "
                f"bloque={bloque}, faltan: {', '.join(missing)}")
            print("   > " + msg)
            logging.warning(msg)   # <- se guarda en el log
            continue
 

        try:
            rc = pd.read_pickle(rc_p)
            rc["bloque_origen"] = bloque #0,1,2,3,4
            rc["suffix_origen"] = suffix #000001 hasta 004000
            # Columnas de hits a limpiar
            hit_list_cols = [
                "event.hit.xPMT",
                "event.hit.yPMT",
                "event.hit.charge",
                "event.hit.time"
            ]

            for c in hit_list_cols:
                if c in rc.columns:
                    rc[c] = rc[c].apply(parse_numeric_array)

            # Eliminar eventos con arrays inválidos
            rc = rc[
                rc["event.hit.xPMT"].apply(lambda x: isinstance(x, np.ndarray)) &
                rc["event.hit.yPMT"].apply(lambda x: isinstance(x, np.ndarray)) &
                rc["event.hit.charge"].apply(lambda x: isinstance(x, np.ndarray)) &
                rc["event.hit.time"].apply(lambda x: isinstance(x, np.ndarray))
            ].copy()

            # Longitudes consistentes entre todas las columnas de hits
            rc = rc[
                rc.apply(
                    lambda row: len(row["event.hit.xPMT"]) ==
                                len(row["event.hit.yPMT"]) ==
                                len(row["event.hit.charge"]) ==
                                len(row["event.hit.time"]),
                    axis=1
                )
            ].copy()

        except Exception as e:
            print(f"   [ERROR ROOT] {e}")
            continue

        df_reco_list.append(rc)

    if not (df_reco_list):
        print("   ¡No hay datos completos para este suffix!")
        return

    # Concatenar

    df_reco   = pd.concat(df_reco_list, ignore_index=True)
    

    df_reco['n_hits'] = df_reco['event.hit.charge'].apply(len)

    df_reco = df_reco[df_reco['n_hits'] > 65]
    
    # Corte geométrico
    mask= np.sqrt(df_reco['rec.coreX']**2 + df_reco['rec.coreY']**2) <=580
    df_reco = df_reco[mask]

    #TO DO
    #AÑADIR UN ID A LA TABLA DF_RECO PREVIO AL SPLAT
    df_reco = df_reco.reset_index(drop=True)
    df_reco["id_correlativo"] = np.arange(len(df_reco))
    # ID final tipo #_000000
    df_reco["suffix_origen"] = df_reco["suffix_origen"].astype(str).str.zfill(6)
    df_reco["bloque_origen"] = df_reco["bloque_origen"].astype(int).astype(str)

    df_reco["id_general"] = (
        df_reco["bloque_origen"] + "_" + df_reco["suffix_origen"] 
    )
    df_reco["ID"] = (
        df_reco["bloque_origen"] + "_" + df_reco["suffix_origen"] +"_"+df_reco["id_correlativo"].astype(str)
    )


    # Explode sincronizado es para aplanar vectores
    df_reco = df_reco.explode(
        ['event.hit.xPMT', 'event.hit.yPMT','event.hit.charge','event.hit.time'],
        ignore_index=True
    )

    # Tipos numéricos
    df_reco['event.hit.xPMT'] = df_reco['event.hit.xPMT'].astype(float)
    df_reco['event.hit.yPMT'] = df_reco['event.hit.yPMT'].astype(float)
    df_reco['event.hit.charge']   = df_reco['event.hit.charge'].astype(float)
    df_reco['event.hit.time']   = df_reco['event.hit.time'].astype(float)

    # Normalizar Time por evento
    df_reco['event.hit.time'] = (
    df_reco['event.hit.time']
    - df_reco.groupby('id_correlativo')['event.hit.time'].transform('min')
    )
    quantiles = list(range(5, 100, 10)) 
    q_cols = [f'q_{q}' for q in quantiles]
    def calc_time_quantiles(g):
        vals = g['event.hit.time'].to_numpy(dtype=float)
        out = {}

        for q in quantiles:
            out[f'q_{q}'] = np.quantile(vals, q / 100.0)

        return pd.Series(out)

    # Una fila por id_correlativo
    df_q = (
        df_reco
        .groupby('id_correlativo', group_keys=False)
        .apply(calc_time_quantiles)
        .reset_index()
    )

    # Añadir cuantiles a df_reco
    df_reco = df_reco.merge(df_q, on='id_correlativo', how='left')

    
    # print("Coordenadas de core desde reco")
    # Coordenadas de core desde reco
    # ================= CORE (nivel evento) =================
    


    # ================= columnas =================
    cols = ['ID','id_general', 'id_correlativo','event.hit.charge'
        ,'event.hit.gridId',
        'event.hit.time',
        'event.hit.xPMT',
        'event.hit.yPMT',
        'event.hit.zPMT'
        ,'lcm.LCm'
        ,'lcm.Chi2'
        ,'lcm.Ndof'
        
        ,'lcm.StatusSim'
        ,'lcm.LCmSim'
        ,'lcm.Chi2Sim'
        ,'lcm.NdofSim'
        ,'lcm.Status'


        ,'ptail.status'
        ,'ptail.statusSim'
        ,'ptail.station_Id'
        ,'ptail.station_hasElectrons'
        ,'ptail.station_hasGammas'
        ,'ptail.station_hasMuons'
        ,'ptail.station_hasOther'
        ,'ptail.ptail'
        ,'ptail.ptailSim'
        ,'ptail.station_distAxis'
        ,'ptail.station_distAxisSim'
        ,'ptail.station_ptail'
        ,'ptail.station_ptailSim'
        ,'ptail.station_signal'
        ,'ptail.station_x'
        ,'ptail.station_y'
        ,'ptail.nStations'
        ,'mc.corsikaParticleId'
        ,'mc.logEnergy'
        ,'mc.coreX'
        ,'mc.coreY'
        ,'mc.azimuthAngle'
        ,'mc.zenithAngle'
        ,'planeFit2.phi'
        ,'planeFit2.status'
        ,'planeFit2.theta'
        ,'rec.PINC'
        ,'rec.LHLatDistFitEnergy'
        ,'rec.coreFitStatus'
        ,'rec.coreX' 
        ,'rec.coreY'
        ,'smc.MuChId'
        ,'smc.MuChPE'
        ,'SimEvent.petrace.gridId'
        ,'SimEvent.petrace.npe'
        ,'SimEvent.petrace.time'
        ,'SimEvent.nPE'
        ,'distanceFromRecCore'
        ,'tankR'
        ,'recLogEnergy'
        ,'recZenithAngle'
        ,'distanceFromTrueCore'        
        ,'event.hit.pmu'
        ,'event.log10pmu0'
        ,'event.nMuHE'
        ,'event.nMuHC'
        ,'event.Pgh_40_1.0'
        ,'event.Pgh_40_2.5'
        ,'event.Pgh_40_5.0'
        ,'event.Pgh_40_10.0'

        
           


    ] +q_cols


    # Features avanzadas
    try:
        feats = extract_basic_features(df_reco)
    except Exception as e:
        print(f"[ERROR extract_basic_features] Falló para particula='{particula}', suffix='{suffix}'")
        print(f"Motivo: {type(e).__name__}: {e}")
        return



    # print("Merge final y filtrado de energía y ángulos")
    # Merge final y filtrado de energía y ángulos
    
    try:
        df_final = (
            df_reco[cols] #df de pyswgo 
            .drop_duplicates('id_correlativo')
            .merge(feats, on='id_correlativo',how='left')       
            .query('1 <= `rec.LHLatDistFitEnergy` <= 1e2') 
            .sort_values('id_correlativo')
            
            .assign(Type_Particle=tipo)
        )
        df_final = df_final[df_final['planeFit2.theta'] <= 0.523599]
    except Exception as e:
        print(f"[ERROR merge_final] Falló para particula='{particula}', suffix='{suffix}'")
        print("feats",feats)
        print("df_reco",feats.keys())
        #print("df_wh",df_wh)
        print(f"Motivo: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    out = carpeta_salida / f"processed_{particula}_{suffix}.parquet"
    df_final.to_parquet(out)
    print(f"   → Guardado: {out} (shape: {df_final.shape})")

# ==================== PROCESAMIENTO EN PARALELO ====================

@timed
def procesar_archivos_parallel(particula: str, indices: range, tipo: int, carpeta_salida: Path):
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    tiempos = []

    def wrapper(suffix: str):
        t0 = time.time()
        procesar_un_indice(suffix, particula, tipo, carpeta_salida)
        dur = time.time() - t0
        tiempos.append(dur)
        print(f"[{suffix}] {dur:.2f}s  | avg: {np.mean(tiempos):.2f}s  total: {np.sum(tiempos):.2f}s")

    Parallel(n_jobs=30)(
        delayed(wrapper)(str(i).zfill(6)) for i in tqdm(indices, desc=f"Paralelo {particula}")
    )

# ==================== CONCATENACIÓN FINAL ====================

@timed
def concatenar_resultados(ruta: Path, nombre_df: str):
    archivos = sorted(ruta.glob("processed_*.parquet"))
    print(f"Concatenando {len(archivos)} archivos en {ruta.name}...")
    # Filtra archivos no vacíos
    archivos_validos = [f for f in archivos if os.path.getsize(f) > 0]
    archivos_validos = archivos_validos[9000:]
    if not archivos_validos:
        print("⚠️ No hay archivos válidos para concatenar.")
        return None

    df = pd.concat((pd.read_parquet(f) for f in archivos_validos), ignore_index=True)
    destino = ruta.parent / f"{nombre_df}.parquet"
    df.to_parquet(destino)
    print(f"→ Guardado final: {destino} (shape: {df.shape})")
    return df

# ==================== BLOQUE PRINCIPAL ====================

if __name__ == "__main__":
    limitar_memoria(240)
    print("=== INICIO DEL PROCESAMIENTO ===")
    inicio_total = time.time()

    # gamma
    procesar_archivos_parallel(
        particula="proton",
        indices=range(10000, 10001),
        tipo=1, #proton:1 gamma:0, en el codigo de ML se intercambian 
        carpeta_salida=Path("/home/swgo_2024/RESULTADOS/proton")
      )
    concatenar_resultados(
          Path("/home/swgo_2024/RESULTADOS/proton"),
          "ML_proton_BBDD_new_variables_v1_p9"
      )


    # Proton (descomenta si lo necesitas)
    #procesar_archivos_parallel(
    #   particula="proton",
    #   indices=range(1, 10001),
    #   tipo=1,
    #   carpeta_salida=Path("/home/swgo_2024/RESULTADOS/proton")
    #)
    #concatenar_resultados(
    #   Path("/home/swgo_2024/RESULTADOS/proton"),
    #   "ML_proton_BBDD_v2"
    #)

    fin_total = time.time()
    print(f"\n=== PROCESAMIENTO COMPLETADO en {fin_total - inicio_total:.2f}s ===")

