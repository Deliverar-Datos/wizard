import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import holidays
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", message="X has feature names but DecisionTreeRegressor was fitted without feature names")

# Crear objeto de feriados para Argentina
feriados_ar = holidays.country_holidays('AR')

def es_feriado(fecha):
    # Devuelve True si la fecha es feriado en Argentina
    return fecha in feriados_ar

def agregar_features(df):
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Ano'] = df['Fecha'].dt.year
    df['SemanaAno'] = df['Fecha'].dt.isocalendar().week
    df['DiaSemana'] = df['Fecha'].dt.dayofweek  # 0=Lunes, 6=Domingo
    df['EsFinDeSemana'] = df['DiaSemana'].apply(lambda x: 1 if x >= 5 else 0)
    df['Feriado'] = df['Fecha'].apply(es_feriado)
    df = df.sort_values('Fecha')

    # Promedios móviles para las variables objetivo, solo si existen las columnas
    for col in ['total', 'entregados', 'cancelados']:
        if col in df.columns:
            df[f'{col}_MA_3'] = df[col].rolling(window=3, min_periods=1).mean()
            df[f'{col}_MA_7'] = df[col].rolling(window=7, min_periods=1).mean()

    return df

def preparar_datos(df, target):
    features = ['Ano', 'SemanaAno', 'DiaSemana', 'EsFinDeSemana', 'Feriado',
                f'{target}_MA_3', f'{target}_MA_7']
    X = df[features]
    y = df[target]
    return X, y

def entrenar_modelo(X, y):
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

def main():
    # Cargar datos
    df = pd.read_excel('Resumen_Pedidos_Estado.xlsx')

    # Calcular total de pedidos por semana (lunes a domingo)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df.set_index('Fecha', inplace=True)

    df_total = df.groupby('Estado')['CantidadPedidos'].resample('W-MON').sum().unstack(level=0).fillna(0)
    df_total.reset_index(inplace=True)

    # Renombrar columnas para entregados y cancelados
    df_total = df_total.rename(columns={
        'Entregado': 'entregados',
        'Cancelado': 'cancelados'
    })

    # Calcular total sumando entregados y cancelados
    df_total['total'] = df_total['entregados'] + df_total['cancelados']

    # Validar que las columnas necesarias existan
    columnas_necesarias = ['Fecha', 'total', 'entregados', 'cancelados']
    for col in columnas_necesarias:
        if col not in df_total.columns:
            raise ValueError(f'La columna "{col}" no está en el DataFrame final.')

    # Agregar features
    df_final = agregar_features(df_total)

    # Entrenar modelos
    modelos = {}
    predicciones = pd.DataFrame()
    predicciones['Fecha'] = df_final['Fecha']
    predicciones['FechaGeneracion'] = datetime.now()

    resultados = {}

    for target in ['total', 'entregados', 'cancelados']:
        X, y = preparar_datos(df_final, target)
        modelo, y_pred, lower_bound, upper_bound, y_test = entrenar_modelo(X, y)
        modelos[target] = modelo
        predicciones[f'{target.capitalize()}Predicho'] = predecir(modelo, X)
        resultados[target] = (y_pred, lower_bound, upper_bound, y_test)

    # Generar fechas futuras para las próximas 26 semanas (aprox. 6 meses)
    ultima_fecha = df_final['Fecha'].max()
    fin_periodo = ultima_fecha + pd.DateOffset(weeks=26)
    fechas_futuras = pd.date_range(start=ultima_fecha + pd.Timedelta(days=7), end=fin_periodo, freq='W-MON')
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

    # Guardar predicciones en Excel
    predicciones = predicciones[['Fecha', 'TotalPredicho', 'EntregadosPredicho', 'CanceladosPredicho', 'FechaGeneracion']]
    try:
        predicciones.to_excel('PrediccionesPedidosSemanal.xlsx', index=False)
        print('Archivo PrediccionesPedidosSemanal.xlsx generado con éxito.')
    except PermissionError:
        print('Error: No se pudo guardar el archivo PrediccionesPedidosSemanal.xlsx. Por favor, cierre el archivo si está abierto y vuelva a intentarlo.')

    # Mostrar intervalos de confianza y primeros 5 valores al final
    for target in ['total', 'entregados', 'cancelados']:
        y_pred, lower_bound, upper_bound, y_test = resultados[target]
        print(f'Intervalo de confianza 95% para {target} (primeros 5 ejemplos):')
        for i in range(min(5, len(y_test))):
            print(f'  Predicción: {y_pred[i]:.2f}, Intervalo: [{lower_bound[i]:.2f}, {upper_bound[i]:.2f}], Valor real: {y_test.iloc[i]}')
        print()  # Línea en blanco para separar salidas


if __name__ == '__main__':
    main()
