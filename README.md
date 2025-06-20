# grain_damage_dynamics

Este proyecto analiza la degradación de la calidad del grano en función de los meses de almacenamiento utilizando regresión cuantílica y modelos de regresión lineal y cuadrática.

## Cómo ejecutar la aplicación

Siga estos pasos para ejecutar la aplicación en su máquina local.

### Prerrequisitos

Asegúrese de tener Python 3.8 o superior instalado en su sistema.

### 1. Crear y activar un entorno virtual

Es una buena práctica crear un entorno virtual para aislar las dependencias del proyecto.

**En macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Instalar las dependencias

Instale todas las librerías necesarias utilizando el archivo `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 3. Ejecutar la aplicación

Una vez que las dependencias estén instaladas, puede iniciar la aplicación Streamlit.

```bash
streamlit run app.py
```

Después de ejecutar este comando, la aplicación se abrirá automáticamente en su navegador web.