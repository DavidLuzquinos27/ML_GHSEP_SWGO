# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 19:53:51 2025

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# version= '4'
# tipo_curva = 'TPR_OBJ'
# version= '5'
# tipo_curva = 'FPR_OBJ'
# file_path = rf'/home/acolan/swgo_2024/Simulaciones_2025/NewDataFrame_ML/ML_NUCLEOS/resultados_completos_26-05-26_{version}.xlsx'
zona_1 = 'zona_1_paper.txt'
zona_2 = 'zona_2_paper.txt'
zona_3 = 'zona_3_paper.txt'

zona_1_FPR = 'zona1FPR.txt'
zona_2_FPR = 'zona2FPR.txt'
zona_3_FPR = 'zona3FPR.txt'



palette = ["#d62728", "#9467bd", "#8c564b", "#17becf"]

# Estilos por zona
estilos_zona = {
    1: ('-.', '^'),  # diamante
    2: ('-.', 'v'),   # diamante
    3: ('-.', 'D'),  # diamante
}

colores_zona = {
    1: "#d62728",  # Rojo
    2: "#182398",  # Azul
    3: "#34df15",  # Verde
}

colors = {
    "CatBoost": palette[0],
    "LightGBM": palette[1],
    "XGBoost": palette[2],
    "VotingClassifier (Soft Voting)": palette[3],
}
# Estilos por modelo
estilos_modelo = {
    "CatBoost": ('-.', 'D'),
    "LightGBM": ('-.', 'D'),
    "XGBoost": ('-.', 'D'),
    "VotingClassifier (Soft Voting)": ('-.', 'D'),
}

# Colores por modelo
colores_modelo = {
    "CatBoost": "#d62728",
    "LightGBM": "#182398",
    "XGBoost": "#34df15",
    "VotingClassifier (Soft Voting)": "#e36911",
}
# === 🔸 Lectura de archivos externos de referencia (paper)
# Intentará separador automático (coma o tab)
zona_refs = {}
for i, path in enumerate([zona_1, zona_2, zona_3], start=1):
    try:
        df_ref = pd.read_csv(path, sep=None, engine='python')  # detecta separador automáticamente
        if {'bin_center', 'TPR'}.issubset(df_ref.columns):
            zona_refs[i] = df_ref
        else:
            print(f"⚠️ {path} no tiene columnas esperadas ('bin_center', 'TPR').")
    except Exception as e:
        print(f"❌ Error leyendo {path}: {e}")
        
        
# === 🔸 Lectura de archivos externos de referencia (paper)
# Intentará separador automático (coma o tab)
zona_refs_FPR = {}
for i, path in enumerate([zona_1_FPR, zona_2_FPR, zona_3_FPR], start=1):
    try:
        df_ref_FPR = pd.read_csv(path, sep=None, engine='python')  # detecta separador automáticamente
        if {'bin_center', 'FPR'}.issubset(df_ref_FPR.columns):
            zona_refs_FPR[i] = df_ref_FPR
        else:
            print(f"⚠️ {path} no tiene columnas esperadas ('bin_center', 'TPR').")
    except Exception as e:
        print(f"❌ Error leyendo {path}: {e}")        
        
        

# xls = pd.ExcelFile(file_path)
map_energy = { 5: 2.25,
              10: 2.75,
              15: 3.25,
              20: 3.75,
              25: 4.25,
              30: 4.75,
              35: 5.25 } 
# ============================================================
# GRAFICADO
# Si version == '4', el Excel ya trae TPR y FPR interpolados.
# Si version != '4', se calcula TPR/FPR desde la matriz.
# ============================================================

for version, tipo_curva in [
    ("4", "TPR_OBJ"),
    ("5", "FPR_OBJ")
]:

    file_path = (
        rf'/home/acolan/swgo_2024/Simulaciones_2025/'
        rf'NewDataFrame_ML/ML_NUCLEOS/'
        rf'resultados_completos_26-05-26_{version}.xlsx'
    )

    xls = pd.ExcelFile(file_path)

    for sheet_name in xls.sheet_names:

        df = pd.read_excel(
            xls,
            sheet_name=sheet_name
        )



        # ========================================================
        # CASO NUEVO: VERSION 4 = TPR/FPR INTERPOLADOS
        # ========================================================
        if version in ["4", "5"]:

            required = {'Model', 'Energy', 'Zone', 'TPR', 'FPR'}
            if not required.issubset(df.columns):
                print(f"[{sheet_name}] Faltan columnas requeridas para interpolado: {required}")
                print("Columnas encontradas:", df.columns.tolist())
                continue

            df_interp = df.copy()

            # Convertir Energy = 5,10,15,... a centro log10(E/GeV)
            df_interp["Energy_plot"] = df_interp["Energy"].map(map_energy)

            # Por seguridad, si algún valor no está en map_energy, usar Energy original
            df_interp["Energy_plot"] = df_interp["Energy_plot"].fillna(df_interp["Energy"])

            # Filtrar zonas 1,2,3 para el gráfico model-by-model
            df_plot_123 = df_interp[df_interp["Zone"].isin([1, 2, 3])].copy()

            # ====================================================
            # 1) GRAFICAR MODEL-BY-MODEL CON ZONAS 1-3
            # ====================================================
            for model_name in df_plot_123["Model"].unique():

                fig, axes = plt.subplots(
                    2, 1,
                    figsize=(12, 14),
                    sharex=True,
                    gridspec_kw={'height_ratios': [1, 1.5]}
                )

                ax1, ax2 = axes

                df_m = df_plot_123[df_plot_123["Model"] == model_name].copy()

                # -------- TPR --------
                for z, g in df_m.groupby("Zone"):
                    g = g.dropna(subset=["Energy_plot", "TPR"]).sort_values("Energy_plot")

                    x = g["Energy_plot"].tolist()
                    y = g["TPR"].tolist()

                    estilo, mark = estilos_zona.get(z, ('-', 'o'))

                    ax1.plot(
                        x, y,
                        estilo,
                        marker=mark,
                        color=colores_zona[z],
                        lw=2,
                        label=f"{model_name} - Zone {z}"
                    )

                # Paper TPR
                if tipo_curva == "TPR_OBJ":

                    for z, paper in zona_refs.items():

                        ax1.plot(
                            paper["bin_center"],
                            paper["TPR"],
                            color=colores_zona[z],
                            lw=2,
                            alpha=0.7,
                            label=f"Paper Zone {z}"
                        )

                else:

                    for z, paper in zona_refs_FPR.items():

                        ax1.plot(
                            paper["bin_center"],
                            paper["FPR"],
                            color=colores_zona[z],
                            lw=2,
                            alpha=0.7,
                            label=f"Paper Zone {z}"
                        )

                ax1.set_ylim(0.6, 1.0)
                ax1.grid(True, ls=":")
                ax1.tick_params(axis="both", labelsize=20)

                # -------- FPR --------
                for z, g in df_m.groupby("Zone"):
                    g = g.dropna(subset=["Energy_plot", "FPR"]).sort_values("Energy_plot")
                    g = g[g["FPR"] > 0]

                    x = g["Energy_plot"].tolist()
                    y = g["FPR"].tolist()

                    estilo, mark = estilos_zona.get(z, ('-', 'o'))

                    ax2.plot(
                        x, y,
                        estilo,
                        marker=mark,
                        color=colores_zona[z],
                        lw=2,
                        label=f"{model_name} - Zone {z}"
                    )

                # Paper FPR
                if tipo_curva == "TPR_OBJ":

                    for z, paper in zona_refs_FPR.items():

                        ax2.plot(
                            paper["bin_center"],
                            paper["FPR"],
                            color=colores_zona[z],
                            lw=2,
                            alpha=0.7,
                            label=f"Paper Zone {z}"
                        )

                else:

                    for z, paper in zona_refs.items():

                        ax2.plot(
                            paper["bin_center"],
                            paper["TPR"],
                            color=colores_zona[z],
                            lw=2,
                            alpha=0.7,
                            label=f"Paper Zone {z}"
                        )

                ax2.set_yscale("log")
                ax2.set_ylim(3e-5, 1)
                ax2.grid(True, ls=":")
                ax2.tick_params(axis="both", labelsize=20)
                ax2.legend(loc="upper right", fontsize=14)

                plt.tight_layout()

                out = (
                    f"output/"
                    f"{tipo_curva}_TPR_FPR_"
                    f"{model_name}_"
                    f"Hoja_{sheet_name}_{version}.jpg"
                )
                plt.savefig(out, dpi=600)
                plt.close()

                print(f"✅ Guardado: {out}")

            # ====================================================
            # 2) GUARDAR EXCEL COMPLETO POR ZONAS 1-5
            # ====================================================
            writer = pd.ExcelWriter(
                f"output/TPR_FPR_Zonas_{sheet_name}_{version}.xlsx",
                engine="xlsxwriter"
            )

            for zona in range(1, 6):
                df_out = df_interp[df_interp["Zone"] == zona][
                    ["Model", "Energy", "Energy_plot", "Zone", "TPR", "FPR", "Q", "modo"]
                ].copy()

                df_out.to_excel(writer, sheet_name=f"Zona_{zona}", index=False)

            writer.close()

            print(f"📁 Archivo guardado: output/TPR_FPR_Zonas_{sheet_name}_{version}.xlsx")

            # ====================================================
            # 3) GRAFICAR TODOS LOS MODELOS POR ZONA
            # ====================================================
            for zona in range(1, 6):

                df_z = df_interp[df_interp["Zone"] == zona].copy()

                if df_z.empty:
                    continue

                fig, axes = plt.subplots(
                    2, 1,
                    figsize=(12, 14),
                    sharex=True,
                    gridspec_kw={'height_ratios': [1, 1.5]}
                )

                ax1, ax2 = axes

                # -------- TPR --------
                for model_name in df_z["Model"].unique():
                    d = df_z[df_z["Model"] == model_name].copy()
                    d = d.dropna(subset=["Energy_plot", "TPR"]).sort_values("Energy_plot")

                    x = d["Energy_plot"].tolist()
                    y = d["TPR"].tolist()

                    estilo, mark = estilos_modelo.get(model_name, ('-', 'o'))

                    ax1.plot(
                        x, y,
                        estilo,
                        marker=mark,
                        lw=2,
                        color=colores_modelo.get(model_name, "gray"),
                        label=model_name
                    )

                if tipo_curva == "TPR_OBJ":

                    if zona in zona_refs:

                        ref = zona_refs[zona]

                        ax1.plot(
                            ref["bin_center"],
                            ref["TPR"],
                            color="black",
                            lw=2,
                            label=f"Paper Zone {zona}"
                        )

                else:

                    if zona in zona_refs_FPR:

                        ref = zona_refs_FPR[zona]

                        ax1.plot(
                            ref["bin_center"],
                            ref["FPR"],
                            color="black",
                            lw=2,
                            label=f"Paper Zone {zona}"
                        )

                ax1.set_ylim(0.6, 1.0)
                ax1.grid(True, ls=":")
                ax1.tick_params(axis="both", labelsize=20)

                # -------- FPR --------
                for model_name in df_z["Model"].unique():
                    d = df_z[df_z["Model"] == model_name].copy()
                    d = d.dropna(subset=["Energy_plot", "FPR"]).sort_values("Energy_plot")
                    d = d[d["FPR"] > 0]

                    x = d["Energy_plot"].tolist()
                    y = d["FPR"].tolist()

                    estilo, mark = estilos_modelo.get(model_name, ('-', 'o'))

                    ax2.plot(
                        x, y,
                        estilo,
                        marker=mark,
                        lw=2,
                        color=colores_modelo.get(model_name, "gray"),
                        label=model_name
                    )

                if tipo_curva == "TPR_OBJ":

                    if zona in zona_refs_FPR:

                        ref = zona_refs_FPR[zona]

                        ax2.plot(
                            ref["bin_center"],
                            ref["FPR"],
                            lw=2,
                            color="black",
                            label=f"Paper Zone {zona}"
                        )

                else:

                    if zona in zona_refs:

                        ref = zona_refs[zona]

                        ax2.plot(
                            ref["bin_center"],
                            ref["TPR"],
                            lw=2,
                            color="black",
                            label=f"Paper Zone {zona}"
                        )

                ax2.set_yscale("log")
                ax2.set_ylim(3e-5, 1)
                ax2.grid(True, ls=":")
                ax2.legend(loc="upper right", fontsize=18)
                ax2.tick_params(axis="both", labelsize=20)

                plt.tight_layout()

                out = (
                    f"output/"
                    f"{tipo_curva}_TPR_FPR_"
                    f"Zona{zona}_"
                    f"Hoja_{sheet_name}_{version}.jpg"
                )
                plt.savefig(out, dpi=600)
                plt.close()

                print(f"✅ Guardado: {out}")

        # ========================================================
        # CASO ANTIGUO: VERSIONES 1, 2, 3
        # ========================================================
        else:

            required = {'Model', 'Particle', 'Energy', 'Zone', '##########', 'FALSO'}
            if not required.issubset(df.columns):
                print(f"[{sheet_name}] Faltan columnas requeridas.")
                continue

            # === 1) Calcular TPR para Particle = 1
            df_positivo = df[df['Particle'] == 1].copy()
            denom_positivo = (df_positivo['##########'] + df_positivo['FALSO']).replace(0, np.nan)
            df_positivo['TPR'] = df_positivo['##########'] / denom_positivo

            # === 2) Calcular FPR usando Particle = 0
            df_neg = df[df['Particle'] == 0].copy()
            base_neg = df_neg[['Model', 'Zone', 'Energy', '##########', 'FALSO']].rename(
                columns={'##########': 'TN', 'FALSO': 'FP'}
            )

            df_fpr = df_positivo.merge(base_neg, on=['Model', 'Zone', 'Energy'], how='left')
            denom_fpr = (df_fpr['FP'] + df_fpr['TN']).replace(0, np.nan)
            df_fpr['FPR'] = df_fpr['FP'] / denom_fpr

            df0_all = df_positivo.copy()
            df1m_all = df_fpr.copy()

            df_pos_plot = df_positivo[df_positivo['Zone'].isin([1, 2, 3])]
            df_fpr_plot = df_fpr[df_fpr['Zone'].isin([1, 2, 3])]

            # Aquí podrías conservar tu lógica antigua si aún necesitas
            # graficar versiones 1, 2 y 3.
            print(f"[{sheet_name}] Versión {version}: usa la lógica antigua de matrices.")
