import csv
from io import BytesIO, StringIO

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Plantilla de limpieza CSV", page_icon="🧹", layout="wide")

# Título y breve explicación para usuarios sin perfil técnico.
st.title("🧹 Limpieza básica de CSV")
st.write(
    "Sube un archivo CSV y la app aplicará una transformación base: "
    "eliminar columnas vacías y normalizar nombres de columnas."
)


def detect_encoding(file_bytes: bytes) -> str:
    """Prueba varios encodings comunes y devuelve el primero que funcione."""
    candidate_encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

    for encoding in candidate_encodings:
        try:
            file_bytes.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue

    # Si ninguno encaja, devolvemos utf-8 para lanzar un error controlado más abajo.
    return "utf-8"


def detect_delimiter(sample_text: str) -> str:
    """Detecta separador con csv.Sniffer y fallback a conteo de caracteres."""
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",;")
        if dialect.delimiter in [",", ";"]:
            return dialect.delimiter
    except csv.Error:
        pass

    # Fallback simple: elegir el delimitador más frecuente en el muestreo.
    comma_count = sample_text.count(",")
    semicolon_count = sample_text.count(";")

    if semicolon_count > comma_count:
        return ";"
    return ","


def load_csv(uploaded_file) -> tuple[pd.DataFrame, str, str]:
    """Lee CSV con detección de encoding y separador."""
    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise ValueError("El archivo está vacío. Sube un CSV con contenido.")

    encoding = detect_encoding(file_bytes)

    try:
        decoded_text = file_bytes.decode(encoding)
    except UnicodeDecodeError as exc:
        raise ValueError(
            "No se pudo leer el archivo por encoding no soportado. "
            "Prueba guardar el CSV como UTF-8."
        ) from exc

    sample = decoded_text[:5000]
    delimiter = detect_delimiter(sample)

    try:
        df = pd.read_csv(StringIO(decoded_text), sep=delimiter)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("El CSV no tiene datos o cabeceras válidas.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            "No se pudo interpretar el CSV. Puede haber un separador inconsistente. "
            "Prueba revisando si usa coma (,) o punto y coma (;)."
        ) from exc

    if df.empty and len(df.columns) == 0:
        raise ValueError("El CSV no contiene columnas ni filas.")

    return df, delimiter, encoding


def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Transformaciones base solicitadas."""
    transformed = df.copy()

    # 1) Limpiar espacios en nombres de columnas.
    transformed.columns = transformed.columns.map(lambda col: str(col).strip())

    # 2) Normalizar nombres de columnas a minúsculas.
    transformed.columns = transformed.columns.map(lambda col: col.lower())

    # 3) Eliminar columnas completamente vacías.
    transformed = transformed.dropna(axis=1, how="all")

    return transformed


uploaded_file = st.file_uploader("1) Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Leemos el CSV de forma segura.
        original_df, delimiter_used, encoding_used = load_csv(uploaded_file)

        st.success(
            f"Archivo leído correctamente (encoding: {encoding_used}, separador detectado: '{delimiter_used}')."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Filas", f"{original_df.shape[0]:,}")
        with col2:
            st.metric("Columnas", f"{original_df.shape[1]:,}")

        st.subheader("2) Vista previa original (20 filas)")
        st.dataframe(original_df.head(20), use_container_width=True)

        # Aplicamos plantilla de transformación.
        final_df = transform_dataframe(original_df)

        st.subheader("3) Resultado transformado")
        st.dataframe(final_df, use_container_width=True)

        csv_output = final_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="4) Descargar CSV transformado",
            data=BytesIO(csv_output),
            file_name="resultado_transformado.csv",
            mime="text/csv",
        )

    except ValueError as error_message:
        st.error(f"❌ {error_message}")
    except Exception as unexpected_error:
        st.error(
            "❌ Ocurrió un error inesperado al procesar el archivo. "
            "Revisa que sea un CSV válido e inténtalo de nuevo."
        )
        st.exception(unexpected_error)
else:
    st.info("Empieza subiendo un archivo CSV para ver la vista previa y aplicar la transformación.")
