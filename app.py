import csv
import re
from io import BytesIO, StringIO

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Despiece", page_icon="📋", layout="wide")

PREVIEW_ROWS = 10
PREVIEW_HEIGHT = 390

# Título y breve explicación para usuarios sin perfil técnico.
st.title("📋 Despiece")
st.markdown(
    'Sube tu informe de SKP en formato CSV y la app generará el despiece para Preproducción. '
    'Recuerda que el nombre debe estar en formato *XX-00000 Nombre de cliente*.'
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


def get_project_id_from_filename(filename: str) -> str:
    """Extrae el ID Proyecto en formato LL-NNNNN de la primera parte del nombre."""
    clean_name = (filename or "").strip()
    # Ignoramos extensión y nos quedamos con la primera parte del nombre.
    stem = clean_name.rsplit(".", 1)[0]
    first_part = re.split(r"[\s_]+", stem, maxsplit=1)[0]

    match = re.fullmatch(r"([A-Za-z]{2})-(\d{5})", first_part)
    if not match:
        return ""

    letters, numbers = match.groups()
    return f"{letters.upper()}-{numbers}"


def find_column_name(columns: pd.Index, target_name: str) -> str | None:
    """Busca una columna ignorando mayúsculas/minúsculas y espacios extremos."""
    normalized_target = target_name.strip().lower()
    for col in columns:
        if str(col).strip().lower() == normalized_target:
            return str(col)
    return None


def parse_numeric_dimension(value: object) -> float | None:
    """Convierte una medida textual a número para evaluaciones condicionales."""
    if pd.isna(value):
        return None

    text_value = str(value).strip().lower()
    text_value = re.sub(r"\s*mm$", "", text_value, flags=re.IGNORECASE).strip()
    text_value = text_value.replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text_value)
    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def normalize_dimension_text(series: pd.Series) -> pd.Series:
    """Limpia texto dimensional retirando sufijo mm y espacios."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s*mm$", "", regex=True, case=False)
    )


def apply_typology_dimension_rules(transformed: pd.DataFrame) -> pd.DataFrame:
    """Aplica reglas de reordenación Lenx/LenY/LenZ según Tipología."""
    lenx_column = find_column_name(transformed.columns, "Lenx")
    leny_column = find_column_name(transformed.columns, "LenY")
    lenz_column = find_column_name(transformed.columns, "LenZ")
    tipologia_column = find_column_name(transformed.columns, "Tipología") or find_column_name(
        transformed.columns, "Tipologia"
    )

    required_columns = [lenx_column, leny_column, lenz_column, tipologia_column]
    if any(col is None for col in required_columns):
        return transformed

    reordered = transformed.copy()
    d1 = reordered[lenx_column].copy()
    d2 = reordered[leny_column].copy()
    d3 = reordered[lenz_column].copy()
    tipologia = reordered[tipologia_column].astype("string").str.strip().str.upper().fillna("")

    mask_r = tipologia.str.startswith("R")
    mask_e = tipologia.str.startswith("E")
    mask_b = tipologia.str.startswith("B")

    d1_result = d1.copy()
    d2_result = d2.copy()
    d3_result = d3.copy()

    # R: intercambio Dim2 <-> Dim3
    d2_result.loc[mask_r] = d3.loc[mask_r]
    d3_result.loc[mask_r] = d2.loc[mask_r]

    # E: rotación (Dim1, Dim2, Dim3) = (Dim3, Dim1, Dim2)
    d1_result.loc[mask_e] = d3.loc[mask_e]
    d2_result.loc[mask_e] = d1.loc[mask_e]
    d3_result.loc[mask_e] = d2.loc[mask_e]

    # L + Dim2 ~ 18/19/20 (redondeado): intercambio Dim1 <-> Dim2
    rounded_d2 = d2.apply(parse_numeric_dimension).apply(lambda number: round(number) if number is not None else None)
    mask_l = tipologia.str.startswith("L") & rounded_d2.isin([18, 19, 20])
    d1_result.loc[mask_l] = d2.loc[mask_l]
    d2_result.loc[mask_l] = d1.loc[mask_l]

    # B: (Dim2, Dim3) = (Dim1, Dim2)
    d2_result.loc[mask_b] = d1.loc[mask_b]
    d3_result.loc[mask_b] = d2.loc[mask_b]

    # Tras la permuta: Dim1 se elimina y quedan Dim2/Dim3 como LenY/LenZ.
    reordered[leny_column] = normalize_dimension_text(d2_result)
    reordered[lenz_column] = normalize_dimension_text(d3_result)
    reordered = reordered.drop(columns=[lenx_column])

    return reordered


def transform_dataframe(df: pd.DataFrame, project_id: str) -> pd.DataFrame:
    """Transformación de plantilla según requisitos del cliente."""
    transformed = df.copy()

    # Normalización base de texto: trim en columnas string/object.
    text_columns = transformed.select_dtypes(include=["object", "string"]).columns
    for col in text_columns:
        transformed[col] = transformed[col].astype("string").str.strip()

    sku_column = find_column_name(transformed.columns, "SKU")
    if sku_column is None:
        raise ValueError("No se encontró la columna 'SKU' en el CSV.")

    # 1) Eliminar filas sin valor en SKU (vacío, espacios o caracteres invisibles).
    sku_values = (
        transformed[sku_column]
        .astype("string")
        .str.replace(r"[\u200B-\u200D\uFEFF]", "", regex=True)
        .str.strip()
    )
    transformed = transformed[sku_values.notna() & (sku_values != "")].copy()

    # 2) Eliminar filas con Hidden = 1 (admite 1, "1", " 1 ").
    hidden_column = find_column_name(transformed.columns, "Hidden")
    if hidden_column is not None:
        hidden_values = transformed[hidden_column].astype("string").str.strip()
        transformed = transformed[~hidden_values.str.fullmatch(r"1(?:\.0+)?", na=False)].copy()

    # 3) Reglas de reordenación Lenx/LenY/LenZ por tipología.
    transformed = apply_typology_dimension_rules(transformed)

    # 4) Eliminar sufijo "mm" en columnas dimensionales de texto.
    dimensional_keywords = {
        "alto",
        "ancho",
        "fondo",
        "largo",
        "profundidad",
        "espesor",
        "diametro",
        "diámetro",
        "dimension",
        "dimensión",
        "medida",
    }
    for col in transformed.columns:
        normalized_col = str(col).strip().lower()
        if any(keyword in normalized_col for keyword in dimensional_keywords):
            transformed[col] = (
                transformed[col]
                .astype("string")
                .str.strip()
                .str.replace(r"\s*mm$", "", regex=True, case=False)
            )

    # 5) Eliminar la columna de Tirador(0=sin tirador), si existe.
    tirador_column = find_column_name(transformed.columns, "Tirador(0=sin tirador)")
    if tirador_column is not None:
        transformed = transformed.drop(columns=[tirador_column])

    # 6) Insertar la columna ID Proyecto en primera posición.
    transformed.insert(0, "ID Proyecto", project_id)

    return transformed.reset_index(drop=True)


def validate_project_id(project_id: str) -> None:
    """Valida que el identificador de proyecto tenga formato LL-NNNNN."""
    if not re.fullmatch(r"[A-Z]{2}-\d{5}", project_id):
        raise ValueError(
            "No se pudo obtener un ID Proyecto válido del nombre del CSV. "
            "El nombre debe empezar por 2 letras, un guion y 5 números (ejemplo: AB-12345)."
        )


uploaded_file = st.file_uploader("1) Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Leemos el CSV de forma segura.
        original_df, delimiter_used, encoding_used = load_csv(uploaded_file)

        st.success(
            f"Archivo leído correctamente (encoding: {encoding_used}, separador detectado: '{delimiter_used}')."
        )

        st.metric("Piezas", original_df.shape[0])

        st.subheader(f"2) Vista previa original ({PREVIEW_ROWS} piezas visibles)")
        st.dataframe(
            original_df,
            use_container_width=True,
            height=PREVIEW_HEIGHT,
        )

        project_id = get_project_id_from_filename(uploaded_file.name)
        validate_project_id(project_id)

        # Aplicamos plantilla de transformación.
        final_df = transform_dataframe(original_df, project_id)

        st.subheader(f"3) Resultado transformado ({PREVIEW_ROWS} piezas visibles)")
        st.dataframe(
            final_df,
            use_container_width=True,
            height=PREVIEW_HEIGHT,
        )

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
