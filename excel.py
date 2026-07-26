import os
import numpy as np
import pandas as pd
import re
def extraer_metricas_y_matrices_confusion(ruta_archivo):
    with open(ruta_archivo, 'r') as archivo:
        lineas = archivo.readlines()
 
    bloques_metricas = []
    matrices_confusion = []
    #print(lineas)
    i = 0
    while i < len(lineas):
        
        linea = lineas[i]
        # Detecta tabla de métricas por columna 'precision' y 'recall'
        if "precision" in linea and "recall" in linea and "f1-score" in linea:
            columnas = linea.split()
            clase_0 = []
            clase_1 = []
            # Leer siguiente líneas donde empiezan los valores (líneas que empiezan con "0" y "1")
            for j in range(i+1, i+10):
                if lineas[j].strip().startswith("0"):
                    clase_0 = lineas[j].strip().split()[1:]
                elif lineas[j].strip().startswith("1"):
                    clase_1 = lineas[j].strip().split()[1:]
                elif "accuracy" in lineas[j]:
                    break
            bloques_metricas.append({
                "columnas": columnas,
                "clase_0": clase_0,
                "clase_1": clase_1
            })
        # Detectar Matriz de Confusión
        if "Matriz de confusi" in linea:
            matriz = []
            for j in range(i+1, i+5):
                if "[" in lineas[j]:
                    fila = [int(x) for x in lineas[j].replace("[", "").replace("]", "").split()]
                    matriz.append(fila)
                elif len(matriz) == 2:
                    break
            matrices_confusion.append(matriz)
            #print(matriz)
        i += 1

    return bloques_metricas, matrices_confusion
def extraer_metricas_interpoladas_tpr08(ruta_archivo):
    """
    Extrae del .txt el bloque:
    --- FPR interpolado a TPR = 0.8 para comparación con whitepaper ---

    Este bloque NO tiene matriz de confusión real.
    Devuelve TPR, FPR, Q y modo.
    """

    with open(ruta_archivo, "r") as archivo:
        lineas = archivo.readlines()

    dentro_bloque = False
    resultado = {
        "TPR": np.nan,
        "FPR": np.nan,
        "Q": np.nan,
        "modo": "interpolado_TPR_0.8"
    }

    for linea in lineas:
        linea_limpia = linea.strip()

        if "FPR interpolado a TPR = 0.8" in linea_limpia:
            dentro_bloque = True
            continue

        if dentro_bloque:
            # Si empieza otro bloque, dejamos de leer
            if linea_limpia.startswith("---") and "FPR interpolado" not in linea_limpia:
                break

            if linea_limpia.startswith("TPR"):
                resultado["TPR"] = float(linea_limpia.split(":")[1].strip())

            elif linea_limpia.startswith("FPR"):
                resultado["FPR"] = float(linea_limpia.split(":")[1].strip())

            elif linea_limpia.startswith("Q-factor"):
                resultado["Q"] = float(linea_limpia.split(":")[1].strip())

            elif linea_limpia.startswith("Modo"):
                resultado["modo"] = linea_limpia.split(":")[1].strip()

    if np.isnan(resultado["TPR"]) or np.isnan(resultado["FPR"]):
        return None

    return resultado
def extraer_metricas_interpoladas_fpr_obj(ruta_archivo):

    with open(ruta_archivo, "r") as archivo:
        lineas = archivo.readlines()

    dentro_bloque = False

    resultado = {
        "TPR": np.nan,
        "FPR": np.nan,
        "Q": np.nan,
        "modo": "interpolado_FPR_OBJ"
    }

    for linea in lineas:

        linea_limpia = linea.strip()

        if "TPR interpolado a FPR objetivo" in linea_limpia:
            dentro_bloque = True
            continue

        if dentro_bloque:

            if linea_limpia.startswith("---"):
                break

            if linea_limpia.startswith("FPR objetivo"):
                resultado["FPR"] = float(
                    linea_limpia.split(":")[1].strip()
                )

            elif linea_limpia.startswith("TPR interpolado"):
                resultado["TPR"] = float(
                    linea_limpia.split(":")[1].strip()
                )

            elif linea_limpia.startswith("Q-factor"):
                resultado["Q"] = float(
                    linea_limpia.split(":")[1].strip()
                )

    if np.isnan(resultado["TPR"]):
        return None

    return resultado
#%%
filas_totales_1 = [[] for _ in range(3)]
filas_totales_2 = [[] for _ in range(3)]
filas_totales_3 = [[] for _ in range(3)]
filas_interpoladas = [[] for _ in range(3)]
filas_interpoladas_fpr = [[] for _ in range(3)]
energias = [(1, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35)]
radios = [(0, 156), (156, 400), (400, 560), (560, 570), (570, 580)]
zonas = {(0, 156): 1, (156, 400): 2, (400, 560): 3, (560, 570): 4, (570, 580): 5}

Method_ = ['CatBoost', 'LightGBM', 'VotingClassifier (Soft Voting)', 'XGBoost']


combinaciones = [(e,r) for e in energias for r in radios]


carpetas = []

# ex_ = ['Ninguna','M', 'MP', 'MC', 'MR','MCR', 'MRP', 'MCP', 'Q', 'QM', 'QMP','QMC', 'QMR','MCRP', 
    #  'Ninguna','M', 'MP', 'MC', 'MR','MCR', 'MRP', 'MCP', 'Q', 'QM', 'QMP','QMC', 'QMR','MCRP']



# carpetas.append(18)
# carpetas.append(19)

# for k in range(20, 33):
#     carpetas.append(k)



# ex_ = ['Ninguna','M', 'MP', 'MC', 'MR','MCR', 'MRP', 'MCP', 'Q', 'QM', 'QMP','QMC', 'QMR','MCRP', 
#      'Ninguna','M', 'MP', 'MC', 'MR','MCR', 'MRP', 'MCP', 'Q', 'QM', 'QMP','QMC', 'QMR','MCRP', ]
ex_ = ['Ninguna','Ptail']
for k in range(1, len(ex_)+1):
    carpetas.append(k)

print(carpetas)

# ex = ['Ninguna', 'Q', 'Ninguna']
k = 0
for fh in carpetas:
    ruta_base = fr"/home/acolan/swgo_2024/Simulaciones_2025/NewDataFrame_ML/ML_NUCLEOS/{fh}_30" 
    ex = ex_[k]
    
    if fh <= 2:
        for energia, radio in combinaciones:
            # Excepción para carpeta (5, 10)
            if energia == (10, 15):
                carpeta = f"version_10-15TeV_30_r_{radio[0]}_{radio[1]}m"
            else:
              carpeta = f"version_{energia[0]}_{energia[1]}TeV_30_r_{radio[0]}_{radio[1]}m"

            ruta_carpeta = os.path.join(ruta_base, carpeta)
            print(f"\n📁 Carpeta: {ruta_carpeta}")

            for me_ in Method_:
                #print(ex)
                if energia == (10, 15):
                    archivo = f"evaluacion_matriz_{me_}_TodasExcepto{ex}_version_{energia[0]}-{energia[1]}TeV_30_r_{radio[0]}_{radio[1]}m.txt"
                else:

                    archivo = f"evaluacion_matriz_{me_}_TodasExcepto{ex}_version_{energia[0]}_{energia[1]}TeV_30_r_{radio[0]}_{radio[1]}m.txt"
                ruta_archivo = os.path.join(ruta_carpeta, archivo)

                if os.path.exists(ruta_archivo):
                    print(f" Existe: {ruta_archivo}")
                    metricas, matrices = extraer_metricas_y_matrices_confusion(ruta_archivo)
                    interp = extraer_metricas_interpoladas_tpr08(ruta_archivo)
                    interp_fpr = extraer_metricas_interpoladas_fpr_obj(
                        ruta_archivo
                    )
                    if interp is not None:
                        fila_interp = {
                            "Model": me_,
                            "Energy": energia[1],
                            "Zone": zonas[radio],
                            "TPR": interp["TPR"],
                            "FPR": interp["FPR"],
                            "Q": interp["Q"],
                            "modo": interp["modo"]
                        }

                        filas_interpoladas[k].append(fila_interp)
                    if interp_fpr is not None:

                        fila_interp_fpr = {
                            "Model": me_,
                            "Energy": energia[1],
                            "Zone": zonas[radio],
                            "TPR": interp_fpr["TPR"],
                            "FPR": interp_fpr["FPR"],
                            "Q": interp_fpr["Q"],
                            "modo": interp_fpr["modo"]
                        }

                        filas_interpoladas_fpr[k].append(
                            fila_interp_fpr
                        )


                    for idx, bloque in enumerate(metricas):
                        print(f"=== Bloque de Métricas {idx+1} ===")
                        print("Columnas:", bloque['columnas'])
                        print("Clase 0:", bloque['clase_0'])
                        print("Clase 1:", bloque['clase_1'])
                        print()

                    for idx, matriz in enumerate(matrices):
                        print(f"=== Matriz de Confusión {idx+1} ===")
                        print(matriz)
                        print()
                        
                    for idx, (bloque, matriz) in enumerate(zip(metricas, matrices)):
                        clase_0 = bloque['clase_0']
                        clase_1 = bloque['clase_1']
                        for clase, valores in zip([0, 1], [clase_0, clase_1]):
                            if clase == 0:
                                valor_true  = matriz[0][0]
                                valor_false = matriz[0][1]
                            else:
                                valor_true  = matriz[1][1]
                                valor_false = matriz[1][0]
                            
                            fila = {
                                "Model": me_,
                                "Particle": clase,
                                "Energy": energia[1],
                                "Zone": zonas[radio],
                                "##########": valor_true,
                                "FALSO": valor_false,
                                "precision": float(valores[0]),
                                "recall": float(valores[1]),
                                "f1-score": float(valores[2]),
                            }

                            # Agregamos la fila al bloque correspondiente
                            if idx == 0:
                                filas_totales_1[k].append(fila)
                            elif idx == 1:
                                filas_totales_2[k].append(fila)
                            elif idx == 2:
                                filas_totales_3[k].append(fila)
                else:
                    print(f"  ❌ No existe: {ruta_archivo}")
    elif fh > 2:
        for energia, radio in combinaciones:
            # Excepción para carpeta (5, 10)
            if energia == (10, 15):
                carpeta = f"version_10-15TeV_30_r_{radio[0]}_{radio[1]}m"
            else:
                carpeta = f"version_{energia[0]}_{energia[1]}TeV_30_r_{radio[0]}_{radio[1]}m"

            ruta_carpeta = os.path.join(ruta_base, carpeta)
            print(f"\n📁 Carpeta: {ruta_carpeta}")

            for me_ in Method_:
                if energia == (10, 15):
                    archivo = f"evaluacion_matriz_{me_}_V2_TodasExcepto{ex}_version_{energia[0]}-{energia[1]}TeV_30_r_{radio[0]}_{radio[1]}m.txt"
                else:

                    archivo = f"evaluacion_matriz_{me_}_V2_TodasExcepto{ex}_version_{energia[0]}_{energia[1]}TeV_30_r_{radio[0]}_{radio[1]}m.txt"
                ruta_archivo = os.path.join(ruta_carpeta, archivo)

                if os.path.exists(ruta_archivo):
                    print(f" Existe: {ruta_archivo}")
                    metricas, matrices = extraer_metricas_y_matrices_confusion(ruta_archivo)
                    interp = extraer_metricas_interpoladas_tpr08(ruta_archivo)

                    if interp is not None:
                        fila_interp = {
                            "Model": me_,
                            "Energy": energia[1],
                            "Zone": zonas[radio],
                            "TPR": interp["TPR"],
                            "FPR": interp["FPR"],
                            "Q": interp["Q"],
                            "modo": interp["modo"]
                        }

                        filas_interpoladas[k].append(fila_interp)


                    for idx, (bloque, matriz) in enumerate(zip(metricas, matrices)):
                        clase_0 = bloque['clase_0']
                        clase_1 = bloque['clase_1']
                        for clase, valores in zip([0, 1], [clase_0, clase_1]):
                            if clase == 0:
                                valor_true  = matriz[0][0]
                                valor_false = matriz[0][1]
                            else:
                                valor_true  = matriz[1][1]
                                valor_false = matriz[1][0]
                            
                            fila = {
                                "Model": me_,
                                "Particle": clase,
                                "Energy": energia[1],
                                "Zone": zonas[radio],
                                "##########": valor_true,
                                "FALSO": valor_false,
                                "precision": float(valores[0]),
                                "recall": float(valores[1]),
                                "f1-score": float(valores[2]),
                            }
                            #print(zonas[radio])

                            # Agregamos la fila al bloque correspondiente
                            if idx == 0:
                                filas_totales_1[k].append(fila)
                            elif idx == 1:
                                filas_totales_2[k].append(fila)
                            elif idx == 2:
                                filas_totales_3[k].append(fila)
                else:
                    print(f"  ❌ No existe: {ruta_archivo}")
    k = k+1

# Crear DataFrames finales por bloque
dfs_1 = {}
dfs_2 = {}
dfs_3 = {}
dfs_interp = {}
dfs_interp_fpr = {}

for idx, fh in enumerate(carpetas):
    print(f"Carpeta {fh}")

    df1 = pd.DataFrame(filas_totales_1[idx])
    df2 = pd.DataFrame(filas_totales_2[idx])
    df3 = pd.DataFrame(filas_totales_3[idx])
    df_interp = pd.DataFrame(filas_interpoladas[idx])
    df_interp_fpr = pd.DataFrame(filas_interpoladas_fpr[idx])
    if not df1.empty:
        df1.set_index("Model", inplace=True)
        dfs_1[f"{fh}"] = df1
    if not df2.empty:
        df2.set_index("Model", inplace=True)
        dfs_2[f"{fh}"] = df2
    if not df3.empty:
        df3.set_index("Model", inplace=True)
        dfs_3[f"{fh}"] = df3
    if not df_interp.empty:
        df_interp.set_index("Model", inplace=True)
        dfs_interp[f"{fh}"] = df_interp
    

    if not df_interp_fpr.empty:

        df_interp_fpr.set_index(
            "Model",
            inplace=True
        )

        dfs_interp_fpr[f"{fh}"] = df_interp_fpr
        
#%%

# Diccionario que asocia el nombre del archivo Excel con el DataFrame correspondiente
dfs_dict = {
    'resultados_completos_26-05-26_1.xlsx': dfs_1,
    'resultados_completos_26-05-26_2.xlsx': dfs_2,
    'resultados_completos_26-05-26_3.xlsx': dfs_3,
    'resultados_completos_26-05-26_4.xlsx': dfs_interp,
    'resultados_completos_26-05-26_5.xlsx': dfs_interp_fpr
}

# Loop para crear los 3 Excel aplicando formato
for archivo_excel, dfs in dfs_dict.items():
    with pd.ExcelWriter(archivo_excel, engine="xlsxwriter") as writer:
        for nombre, df in dfs.items():
            df.to_excel(writer, sheet_name=nombre, startrow=1, header=False)

            workbook = writer.book
            worksheet = writer.sheets[nombre]

            # Formato general para datos
            formato_datos = workbook.add_format({
                'font_name': 'Calibri',
                'font_size': 11,
                'valign': 'top',
                'text_wrap': True,
                'border': 0
            })

            # Formato para encabezados (columnas)
            formato_encabezado = workbook.add_format({
                'font_name': 'Aptos Narrow (Cuerpo)',
                'font_size': 11,
                'bold': True,
                'align': 'center',
                'valign': 'vcenter',
                'text_wrap': True,
                'border': 0
            })

            # Escribir encabezados manualmente con formato
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num + 1, value, formato_encabezado)

            # Escribir índice manualmente con formato de encabezado
            worksheet.write(0, 0, df.index.name if df.index.name else "Index", formato_encabezado)

            # Aplicar formato de datos a toda la tabla
            for row_num, (index, row) in enumerate(df.iterrows(), start=1):
                worksheet.write(row_num, 0, index, formato_datos)  # índice
                for col_num, cell_value in enumerate(row):
                    worksheet.write(row_num, col_num + 1, cell_value, formato_datos)

            # Ajuste de anchos
            worksheet.set_column(0, 0, 15, formato_datos)  # índice
            for i, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i + 1, i + 1, max_len, formato_datos)

print("Archivos Excel generados correctamente con formato personalizado.")
