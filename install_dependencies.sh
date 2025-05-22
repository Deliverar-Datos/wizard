#!/bin/bash

# Script para instalar dependencias Python necesarias en Linux

# Actualizar pip
python3 -m pip install --upgrade pip

# Instalar paquetes necesarios
pip3 install pandas numpy holidays scikit-learn psycopg2-binary
