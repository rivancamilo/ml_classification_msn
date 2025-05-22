#!/bin/bash

# entrypoint.sh
# Descripción: 
#   Shell, encargada de validar la creación de las carpetas más importantes del proyecto, 
#   y de iniciar los servicios de MLflow y Prefect
# Autor: Ivan Camilo Rosales R.
# Fecha: 2025-05-21
# Version: 1.0

# Instalar herramientas necesarias si no están disponibles
echo "Verificando e instalando herramientas necesarias..."
apt-get update -qq

# Instalar herramientas de extracción
if ! command -v 7z >/dev/null 2>&1; then
    echo "Instalando 7zip..."
    apt-get install -y -qq p7zip-full
fi

if ! command -v unrar >/dev/null 2>&1; then
    echo "Instalando unrar..."
    apt-get install -y -qq unrar || echo "unrar no disponible"
fi

if ! command -v binwalk >/dev/null 2>&1; then
    echo "Instalando binwalk..."
    apt-get install -y -qq binwalk || echo "binwalk no disponible"
fi

#-----  Validamos que todos los directorios más importantes estén creados
mkdir -p /app/mlruns /app/data/input /app/data/output /app/data/modelos
chmod -R 777 /app/mlruns /app/data

export MLFLOW_TRACKING_URI=http://0.0.0.0:5000


DIRECTORIO="/app/data/input"
NOMBRE_BASE="twcs"
DIRECTORIO_DESCOMPRIMIDO="$DIRECTORIO/descomprimido"


#-----  Función para validar que el archivo zip y las partes existan
verificar_archivos() {
    local encontrado=false
    
    #-----  Validamos que el archivo principal exista
    if [[ -f "$DIRECTORIO/$NOMBRE_BASE.zip" ]]; then
        echo "Encontramos el archivo principal: $NOMBRE_BASE.zip"
        encontrado=true
    fi
    
    #-----  Validamos las partas del archivo (.z01, .z02, etc.)
    local partes=$(find "$DIRECTORIO" -name "$NOMBRE_BASE.z*" -type f | wc -l)
    if [[ $partes -gt 0 ]]; then
        echo "Encontradas $partes partes del archivo dividido"
        encontrado=true
    fi
    
    return $($encontrado && echo 0 || echo 1)
}

# Función para mover archivos CSV al directorio principal
mover_archivos_csv() {
    echo "Buscando y moviendo archivos CSV a $DIRECTORIO..."
    
    # Buscar archivos CSV en el directorio descomprimido y subdirectorios
    local archivos_csv=$(find "$DIRECTORIO_DESCOMPRIMIDO" -name "*.csv" -type f 2>/dev/null)
    
    if [[ -n "$archivos_csv" ]]; then
        echo "Archivos CSV encontrados:"
        echo "$archivos_csv"
        
        # Mover cada archivo CSV al directorio principal
        echo "$archivos_csv" | while read -r archivo; do
            if [[ -f "$archivo" ]]; then
                local nombre_archivo=$(basename "$archivo")
                echo "Moviendo: $nombre_archivo"
                mv "$archivo" "$DIRECTORIO/"
                echo "✓ $nombre_archivo movido a $DIRECTORIO/"
            fi
        done
        
        # Verificar que los archivos se movieron correctamente
        local csv_movidos=$(find "$DIRECTORIO" -maxdepth 1 -name "*.csv" -type f | wc -l)
        echo "✓ Total de archivos CSV en $DIRECTORIO: $csv_movidos"
        
        # Mostrar los archivos CSV en el directorio final
        echo "Archivos CSV disponibles en $DIRECTORIO:"
        ls -la "$DIRECTORIO"/*.csv 2>/dev/null || echo "No se encontraron archivos CSV"
        
        return 0
    else
        echo "⚠ No se encontraron archivos CSV en el directorio descomprimido"
        
        # Buscar otros tipos de archivos que podrían ser relevantes
        echo "Buscando otros archivos en el directorio descomprimido..."
        find "$DIRECTORIO_DESCOMPRIMIDO" -type f -exec ls -la {} \; 2>/dev/null
        
        return 1
    fi
}

# Función para limpiar directorio temporal
limpiar_directorio_temporal() {
    if [[ -d "$DIRECTORIO_DESCOMPRIMIDO" ]]; then
        echo "Limpiando directorio temporal: $DIRECTORIO_DESCOMPRIMIDO"
        rm -rf "$DIRECTORIO_DESCOMPRIMIDO"
        echo "✓ Directorio temporal eliminado"
    fi
}

#-----  Función para descomprimir archivos divididos
descomprimir_archivo() {
    
    # Crear directorio de destino si no existe
    mkdir -p "$DIRECTORIO_DESCOMPRIMIDO"
    
    # Verificar si tenemos archivos multi-parte
    local partes=$(find "$DIRECTORIO" -name "$NOMBRE_BASE.z*" -type f | sort)
    
    if [[ -n "$partes" && -f "$DIRECTORIO/$NOMBRE_BASE.zip" ]]; then
        echo "Detectados archivos multi-parte, usando estrategias avanzadas..."
        
        cd "$DIRECTORIO"
        
        # Método 1: Usar 7z directamente sin recombinar
        echo "Método 1: Extracción directa con 7z (sin recombinar)..."
        if 7z x "$NOMBRE_BASE.zip" -o"$DIRECTORIO_DESCOMPRIMIDO" -y 2>/dev/null; then
            echo "✓ Descompresión completada con 7z"
            cd - > /dev/null
            ls -la "$DIRECTORIO_DESCOMPRIMIDO"
            
            # Mover archivos CSV al directorio principal
            if mover_archivos_csv; then
                limpiar_directorio_temporal
                return 0
            fi
        fi
        
        cd - > /dev/null
        
        # Verificar si obtuvimos algún resultado y mover CSV
        if [[ $(ls -A "$DIRECTORIO_DESCOMPRIMIDO" 2>/dev/null | wc -l) -gt 0 ]]; then
            echo "✓ Se extrajeron algunos archivos exitosamente"
            ls -la "$DIRECTORIO_DESCOMPRIMIDO"
            
            # Mover archivos CSV al directorio principal
            if mover_archivos_csv; then
                limpiar_directorio_temporal
                return 0
            fi
        fi
        
        echo "✗ Todos los métodos de descompresión fallaron"
        echo "Información de debug:"
        ls -la "$DIRECTORIO/"
        file "$DIRECTORIO/$NOMBRE_BASE".*
        return 1
        
    else
        # Archivo ZIP simple
        echo "Descomprimiendo archivo ZIP simple..."
        if unzip -o "$DIRECTORIO/$NOMBRE_BASE.zip" -d "$DIRECTORIO_DESCOMPRIMIDO"; then
            echo "✓ Descompresión completada exitosamente"
            ls -la "$DIRECTORIO_DESCOMPRIMIDO"
            
            # Mover archivos CSV al directorio principal
            if mover_archivos_csv; then
                limpiar_directorio_temporal
                return 0
            else
                echo "✗ No se pudieron mover los archivos CSV"
                return 1
            fi
        else
            echo "✗ Error durante la descompresión"
            return 1
        fi
    fi
}

# Función para limpiar archivos temporales (opcional)
limpiar_archivos_comprimidos() {
    read -p "¿Desea eliminar los archivos comprimidos originales? (y/N): " respuesta
    if [[ $respuesta =~ ^[Yy]$ ]]; then
        rm -f "$DIRECTORIO/$NOMBRE_BASE".z* "$DIRECTORIO/$NOMBRE_BASE.zip"
        echo "✓ Archivos comprimidos eliminados"
    fi
}

# Proceso principal de descompresión
echo "**********************************************************************"
echo "Verificando archivos comprimidos..."

if verificar_archivos; then
    echo "Empezamos a descomprimir el archivo..."
    
    if descomprimir_archivo; then
        echo "✓ Proceso de descompresión completado con éxito"
        echo "✓ Archivos CSV disponibles en: $DIRECTORIO"
        
        # Mostrar archivos CSV finales
        echo "Listado final de archivos CSV:"
        ls -la "$DIRECTORIO"/*.csv 2>/dev/null || echo "No se encontraron archivos CSV en el directorio final"
        
        # Descomentar la siguiente línea si quieres limpiar archivos originales
        # limpiar_archivos_comprimidos
    else
        echo "✗ Error en el proceso de descompresión"
        exit 1
    fi
else
    echo "✗ No se encontraron archivos comprimidos necesarios en $DIRECTORIO"
    echo "Buscando archivos disponibles:"
    ls -la "$DIRECTORIO"
    exit 1
fi

echo "**********************************************************************"




#-----  Ejecutamos el comando base para iniciar el servicio de MLFlow
echo "**********************************************************************"
echo "**********************************************************************"
echo "**********************************************************************"
echo "Iniciando MLflow UI..."
mlflow server \
    --backend-store-uri sqlite:////app/mlruns/mlflow.db \
    --default-artifact-root /app/mlruns \
    --host 0.0.0.0 &

#-----  Ejecutamos el comando base para iniciar el servicio de Perfect
echo "**********************************************************************"
echo "**********************************************************************"
echo "**********************************************************************"
echo "Iniciando Prefect Server..."
prefect server start --host 0.0.0.0 &

#-----  Tiempo de espera para que los servicios de MlFlow y Perfect estén disponibles
echo "Esperando que los servidores estén listos..."
sleep 25


#-----  Ejecutamos el comando el archivo inicial de Python
echo "Iniciando ejecución de main.py..."
python main.py

#-----  Comando para mantener el contenedor en ejecución
echo "Servicios iniciados correctamente. Manteniendo el contenedor en ejecución..."
tail -f /dev/null