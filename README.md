# App de limpieza básica de CSV con Streamlit

Esta app permite:

1. Subir un archivo CSV.
2. Detectar automáticamente el separador (coma o punto y coma).
3. Mostrar vista previa de las primeras 20 filas.
4. Mostrar número de filas y columnas.
5. Aplicar una transformación base:
   - Eliminar columnas totalmente vacías.
   - Quitar espacios en nombres de columnas.
   - Convertir nombres de columnas a minúsculas.
6. Descargar el CSV transformado.

---

## 1) Requisitos

- Python 3.9 o superior (recomendado).

---

## 2) Instalación paso a paso (local)

### Paso 1: clonar o descargar el proyecto
Si ya tienes esta carpeta, omite este paso.

### Paso 2: crear entorno virtual (recomendado)

En macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Paso 3: instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: ejecutar la app

```bash
streamlit run app.py
```

Se abrirá en el navegador, normalmente en `http://localhost:8501`.

---

## 3) Uso de la app (muy simple)

1. Haz clic en **“Sube tu archivo CSV”**.
2. Selecciona tu archivo.
3. La app mostrará:
   - Encoding y separador detectado.
   - Filas y columnas.
   - Vista previa original.
   - Tabla transformada.
4. Pulsa **“Descargar CSV transformado”** para bajar el resultado.

---

## 4) Errores comunes que la app ya controla

- **CSV vacío**.
- **Problemas de encoding** (intenta `utf-8-sig`, `utf-8`, `cp1252`, `latin-1`).
- **Separador inconsistente o incorrecto**.

La app muestra mensajes claros para que sepas cómo corregirlo.

---

## 5) Despliegue en Streamlit Community Cloud

1. Sube estos archivos a un repositorio GitHub:
   - `app.py`
   - `requirements.txt`
   - `README.md`
2. Entra a [https://share.streamlit.io](https://share.streamlit.io).
3. Conecta tu cuenta de GitHub y elige el repositorio.
4. En **Main file path**, indica `app.py`.
5. Pulsa **Deploy**.

Listo: tu app quedará publicada.
