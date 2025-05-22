import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import holidays
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import calendar
import sys
import psycopg2
from psycopg2 import sql
import psycopg2.extras

warnings.filterwarnings("ignore", message="X has feature names but DecisionTreeRegressor was fitted without feature names")

# Crear objeto de feriados para Argentina
feriados_ar = holidays.country_holidays('AR')

def es_feriado(fecha):
    # Devuelve True si la fecha es feriado en Argentina
    return fecha in feriados_ar

def agregar_features(df):
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['DiaSemana'] = df['Fecha'].dt.dayofweek  # 0=Lunes, 6=Domingo
    df['EsFinDeSemana'] = df['DiaSemana'].apply(lambda x: 1 if x >= 5 else 0)
    df['DiaMes'] = df['Fecha'].dt.day
    df['Mes'] = df['Fecha'].dt.month
    df['Ano'] = df['Fecha'].dt.year
    df['Feriado'] = df['Fecha'].apply(es_feriado)
    df = df.sort_values('Fecha')

    # Promedios móviles para las variables objetivo, solo si existen las columnas
    for col in ['total', 'entregados', 'cancelados']:
        if col in df.columns:
            df[f'{col}_MA_3'] = df[col].rolling(window=3, min_periods=1).mean()
            df[f'{col}_MA_7'] = df[col].rolling(window=7, min_periods=1).mean()

    return df

def preparar_datos(df, target):
    features = ['DiaSemana', 'EsFinDeSemana', 'DiaMes', 'Mes', 'Ano', 'Feriado',
                f'{target}_MA_3', f'{target}_MA_7']
    X = df[features]
    y = df[target].fillna(0)
    return X, y

def entrenar_modelo(X, y):
    import numpy as np
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Calcular intervalo de confianza al 95% usando predicciones de cada árbol
    preds_trees = np.array([tree.predict(X_test) for tree in modelo.estimators_])
    lower_bound = np.percentile(preds_trees, 2.5, axis=0)
    upper_bound = np.percentile(preds_trees, 97.5, axis=0)

    return modelo, y_pred, lower_bound, upper_bound, y_test

def predecir(modelo, X):
    preds = modelo.predict(X)
    return np.round(preds).astype(int)

def guardar_en_postgresql(df):
    # Datos de conexión
    conn_params = {
        'host': '173.230.135.41',
        'port': 5432,
        'user': 'deliverar_user',
        'password': 'unixunix',
        'dbname': 'deliverar'
    }
    try:
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        # Borrar datos existentes
        cursor.execute(sql.SQL("DELETE FROM fact_pedidos_prediccion_diaria"))
        conn.commit()

        # Insertar datos
        # Preparar columnas y valores
        cols = list(df.columns)
        values = [tuple(x) for x in df.to_numpy()]

        # Crear query parametrizada
        insert_query = sql.SQL("INSERT INTO fact_pedidos_prediccion_diaria ({}) VALUES %s").format(
            sql.SQL(', ').join(map(sql.Identifier, cols))
        )

        # Usar mogrify para insertar múltiples filas
        psycopg2.extras.execute_values(cursor, insert_query.as_string(conn), values)
        conn.commit()
        cursor.close()
        conn.close()
        print("Datos guardados en PostgreSQL correctamente.")
    except Exception as e:
        print(f"Error al guardar en PostgreSQL: {e}")

def main():
    # Conexión a la base de datos PostgreSQL
    conn_params = {
        'host': '173.230.135.41',
        'port': 5432,
        'user': 'deliverar_user',
        'password': 'unixunix',
        'dbname': 'deliverar'
    }
    import psycopg2
    import pandas as pd

    query = """
    SELECT
        fecha_pedido AS Fecha,
        estado AS Estado,
        COUNT(*) AS CantidadPedidos
    FROM fact_pedidos
    WHERE LOWER(estado) IN ('entregado', 'cancelado')
    GROUP BY fecha_pedido, estado
    ORDER BY fecha_pedido
    """

    try:
        with psycopg2.connect(**conn_params) as conn:
            df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error al leer datos de PostgreSQL: {e}")
        return

    # Mostrar columnas para verificar nombres reales
    print("Columnas en el DataFrame:", df.columns.tolist())

    # Mostrar valores únicos en la columna estado
    print("Valores únicos en 'estado':", df['estado'].unique())

    # Calcular total de pedidos por fecha
    df_total = df.groupby('fecha')['cantidadpedidos'].sum().reset_index().rename(columns={'cantidadpedidos': 'total'})

    # Pivotar para entregados y cancelados
    df_pivot = df.pivot_table(index='fecha', columns='estado', values='cantidadpedidos', aggfunc='sum').reset_index()

    # Renombrar columnas para entregados y cancelados
    df_pivot.columns.name = None
    df_pivot = df_pivot.rename(columns={
        'ENTREGADO': 'entregados',
        'CANCELADO': 'cancelados'
    })

    # Unir total con entregados y cancelados
    df_final = pd.merge(df_total, df_pivot, on='fecha', how='left')

    # Cambiar nombre de columna 'fecha' a 'Fecha' para compatibilidad con agregar_features
    df_final = df_final.rename(columns={'fecha': 'Fecha'})

    # Validar que las columnas necesarias existan
    columnas_necesarias = ['Fecha', 'total', 'entregados', 'cancelados']
    for col in columnas_necesarias:
        if col not in df_final.columns:
            raise ValueError(f'La columna "{col}" no está en el DataFrame final.')

    # Agregar features
    df_final = agregar_features(df_final)

    # Entrenar modelos
    modelos = {}
    predicciones = pd.DataFrame()
    predicciones['Fecha'] = df_pivot['fecha']
    predicciones['FechaGeneracion'] = datetime.now()

    resultados = {}

    for target in ['total', 'entregados', 'cancelados']:
        X, y = preparar_datos(df_final, target)
        modelo, y_pred, lower_bound, upper_bound, y_test = entrenar_modelo(X, y)
        modelos[target] = modelo
        predicciones[f'{target.capitalize()}Predicho'] = predecir(modelo, X)
        resultados[target] = (y_pred, lower_bound, upper_bound, y_test)

    # Generar fechas futuras para los próximos 6 meses
    ultima_fecha = df_final['Fecha'].max()
    fin_periodo = ultima_fecha + pd.DateOffset(months=6)
    fechas_futuras = pd.date_range(start=ultima_fecha + pd.Timedelta(days=1), end=fin_periodo, freq='D')
    df_futuro = pd.DataFrame({'Fecha': fechas_futuras})
    df_futuro = agregar_features(df_futuro)

    # Añadir columnas faltantes con valores cero para evitar errores
    for col in ['total', 'entregados', 'cancelados']:
        if col not in df_futuro.columns:
            df_futuro[col] = 0
        if f'{col}_MA_3' not in df_futuro.columns:
            df_futuro[f'{col}_MA_3'] = 0
        if f'{col}_MA_7' not in df_futuro.columns:
            df_futuro[f'{col}_MA_7'] = 0

    # Preparar dataframe para predicciones futuras
    predicciones_futuras = pd.DataFrame()
    predicciones_futuras['Fecha'] = fechas_futuras
    predicciones_futuras['FechaGeneracion'] = datetime.now()

    # Predecir para fechas futuras
    for target in ['total', 'entregados', 'cancelados']:
        X_futuro, _ = preparar_datos(df_futuro, target)
        predicciones_futuras[f'{target.capitalize()}Predicho'] = predecir(modelos[target], X_futuro)

    # Concatenar predicciones actuales y futuras
    predicciones = pd.concat([predicciones, predicciones_futuras], ignore_index=True)

    # Seleccionar columnas para insertar en PostgreSQL
    predicciones = predicciones[['Fecha', 'TotalPredicho', 'EntregadosPredicho', 'CanceladosPredicho', 'FechaGeneracion']]

    # Renombrar columnas para que coincidan con la tabla PostgreSQL
    predicciones = predicciones.rename(columns={
        'Fecha': 'fecha',
        'TotalPredicho': 'total_predicho',
        'EntregadosPredicho': 'entregados_predicho',
        'CanceladosPredicho': 'cancelados_predicho',
        'FechaGeneracion': 'fecha_generacion'
    })

    # Guardar en PostgreSQL
    guardar_en_postgresql(predicciones)

    # Mostrar intervalos de confianza y primeros 5 valores al final
    for target in ['total', 'entregados', 'cancelados']:
        y_pred, lower_bound, upper_bound, y_test = resultados[target]
        print(f'Intervalo de confianza 95% para {target} (primeros 5 ejemplos):')
        for i in range(min(5, len(y_test))):
            print(f'  Predicción: {y_pred[i]:.2f}, Intervalo: [{lower_bound[i]:.2f}, {upper_bound[i]:.2f}], Valor real: {y_test.iloc[i]}')
        print()  # Línea en blanco para separar salidas


if __name__ == '__main__':
    main()
