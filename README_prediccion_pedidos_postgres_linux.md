# Ejecutar prediccion_pedidos_postgres.py en Linux

Este documento explica cómo ejecutar el script `prediccion_pedidos_postgres.py` en un entorno Linux.

## Requisitos previos

- Tener instalado Python 3 (preferentemente Python 3.8 o superior).
- Tener instalado `pip` para gestionar paquetes de Python.
- Acceso a la base de datos PostgreSQL con las credenciales adecuadas (host, puerto, usuario, contraseña, base de datos).
- El archivo `prediccion_pedidos_postgres.py` debe estar en el directorio de trabajo.
- El script requiere las siguientes librerías de Python:
  - pandas
  - numpy
  - scikit-learn
  - holidays
  - psycopg2-binary

## Pasos para ejecutar

1. **Actualizar pip e instalar dependencias**

   Abrir una terminal y ejecutar:

   ```bash
   python3 -m pip install --upgrade pip
   pip3 install -r requirements_postgres.txt
   ```

   Si no tienes el archivo `requirements_postgres.txt`, puedes instalar las librerías manualmente:

   ```bash
   pip3 install pandas numpy scikit-learn holidays psycopg2-binary
   ```

2. **Configurar acceso a la base de datos**

   El script `prediccion_pedidos_postgres.py` tiene configurados los parámetros de conexión a la base de datos en el código:

   ```python
   conn_params = {
       'host': '173.230.135.41',
       'port': 5432,
       'user': 'deliverar_user',
       'password': 'unixunix',
       'dbname': 'deliverar'
   }
   ```

   Si necesitas cambiar estos datos, edita el script antes de ejecutarlo.

3. **Ejecutar el script**

   En la terminal, estando en el directorio donde está el script, ejecutar:

   ```bash
   python3 prediccion_pedidos_postgres.py
   ```

4. **Verificar salida**

   El script imprimirá información sobre las columnas y valores únicos, además de intervalos de confianza para las predicciones.

   También guardará los resultados en la tabla `fact_pedidos_prediccion_diaria` de la base de datos PostgreSQL configurada.

## Notas adicionales

- El script puede mostrar advertencias relacionadas con sklearn sobre nombres de características, que no afectan la ejecución.
- Asegúrate de tener acceso a la base de datos y permisos para borrar e insertar datos en la tabla mencionada.
- Para ejecutar en un entorno virtual, crea y activa un entorno virtual antes de instalar dependencias y ejecutar el script.

---

Este README proporciona los pasos básicos para ejecutar el script en Linux. Si necesitas ayuda adicional, no dudes en preguntar.
