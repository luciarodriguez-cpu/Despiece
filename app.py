import csv
import re
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Despiece", page_icon="📋", layout="wide")

PREVIEW_ROWS = 10
PREVIEW_HEIGHT = 390

EXPECTED_COLUMNS = {
    "dimensiones_estandar.csv": ["Tipologia", "Ancho", "Largo"],
    "materiales.csv": ["Material", "Core", "Gama"],
    "aperturas.csv": ["clave_apertura", "valor_apertura"],
    "tiradores.csv": ["clave_tirador", "valor_tirador"],
}


# Título y breve explicación para usuarios sin perfil técnico.
st.title("📋 Despiece")
st.markdown(
    'Sube tu informe de SKP en formato CSV y la app generará el despiece para Preproducción. '
    'Recuerda que el nombre debe estar en formato *XX-00000 Nombre de cliente*.'
)


def _read_csv_with_fallback(path: Path, keep_empty_strings: bool = False) -> pd.DataFrame:
    """Lee CSV de base de datos probando separador coma y punto y coma."""
    read_kwargs = {
        "dtype": str,
        "keep_default_na": not keep_empty_strings,
    }

    errors: list[str] = []
    for sep in (",", ";"):
        try:
            return pd.read_csv(path, sep=sep, **read_kwargs)
        except Exception as exc:
            errors.append(f"sep='{sep}': {exc}")

    raise ValueError(
        f"No se pudo leer el CSV '{path.name}' con separador ',' ni ';'. "
        f"Detalle: {' | '.join(errors)}"
    )


def _validate_columns(df: pd.DataFrame, expected: list[str], file_label: str) -> None:
    missing_columns = [col for col in expected if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"El archivo '{file_label}' no tiene las columnas esperadas. "
            f"Faltan: {missing_columns}. Detectadas: {list(df.columns)}"
        )


def load_database() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, str]]:
    """Carga y valida CSV de base de datos para usar en transformaciones internas."""
    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "data"

    required_paths = {name: data_dir / name for name in EXPECTED_COLUMNS}
    missing_files = [name for name, path in required_paths.items() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Faltan archivos en la carpeta data/: {missing_files}. "
            f"Ruta evaluada: {data_dir}"
        )

    df_dimensiones = _read_csv_with_fallback(required_paths["dimensiones_estandar.csv"])
    _validate_columns(df_dimensiones, EXPECTED_COLUMNS["dimensiones_estandar.csv"], "dimensiones_estandar.csv")
    df_dimensiones = df_dimensiones[EXPECTED_COLUMNS["dimensiones_estandar.csv"]].copy()

    df_materiales = _read_csv_with_fallback(required_paths["materiales.csv"])
    _validate_columns(df_materiales, EXPECTED_COLUMNS["materiales.csv"], "materiales.csv")
    df_materiales = df_materiales[EXPECTED_COLUMNS["materiales.csv"]].copy()

    df_aperturas = _read_csv_with_fallback(required_paths["aperturas.csv"])
    _validate_columns(df_aperturas, EXPECTED_COLUMNS["aperturas.csv"], "aperturas.csv")
    map_aperturas = dict(
        zip(df_aperturas["clave_apertura"].astype(str), df_aperturas["valor_apertura"].astype(str))
    )

    df_tiradores = _read_csv_with_fallback(required_paths["tiradores.csv"], keep_empty_strings=True)
    _validate_columns(df_tiradores, EXPECTED_COLUMNS["tiradores.csv"], "tiradores.csv")
    map_tiradores = dict(
        zip(df_tiradores["clave_tirador"].astype(str), df_tiradores["valor_tirador"].astype(str))
    )

    return df_dimensiones, df_materiales, map_aperturas, map_tiradores


@st.cache_data(show_spinner=False)
def get_database() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, str]]:
    """Cachea la base de datos para no recargar archivos en cada interacción."""
    return load_database()


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


def transform_material_to_core_gama_acabado(
    df_user: pd.DataFrame,
    df_materiales: pd.DataFrame,
    source_col: str = "Material",
) -> pd.DataFrame:
    """Transforma la columna Material en Core/Gama/Acabado usando base de materiales."""
    if source_col not in df_user.columns:
        raise ValueError(f"No se encontró la columna '{source_col}' en df_user.")

    required_columns = ["Material", "Core", "Gama"]
    missing_material_columns = [col for col in required_columns if col not in df_materiales.columns]
    if missing_material_columns:
        raise ValueError(
            "df_materiales no tiene las columnas esperadas. "
            f"Faltan: {missing_material_columns}."
        )

    user_values = df_user[source_col].fillna("").astype(str)
    normalized = (
        user_values
        .str.replace(r"\bwood\s*(horizontal|vertical)\b", "WOOD", regex=True, case=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    words = normalized.str.split()
    first_word = words.str[0].fillna("").str.upper()
    last_word = words.str[-1].fillna("").str.upper()
    acabado = words.apply(lambda parts: " ".join(parts[:-1]) if len(parts) >= 2 else "")

    material_lookup = (
        df_materiales[["Material", "Core", "Gama"]]
        .copy()
        .fillna("")
    )
    material_lookup["Material"] = material_lookup["Material"].astype(str).str.upper().str.strip()

    core_map = dict(zip(material_lookup["Material"], material_lookup["Core"].astype(str)))
    gama_map = dict(zip(material_lookup["Material"], material_lookup["Gama"].astype(str)))

    core_from_last = last_word.map(core_map)
    gama_from_last = last_word.map(gama_map)
    core_from_first = first_word.map(core_map)
    gama_from_first = first_word.map(gama_map)

    core = core_from_last.where(core_from_last.notna(), core_from_first).fillna("")
    gama = gama_from_last.where(gama_from_last.notna(), gama_from_first).fillna("")

    transformed = df_user.copy()
    source_index = transformed.columns.get_loc(source_col)

    # Evita conflictos si el CSV de usuario ya trae estas columnas.
    target_columns = ["Core", "Gama", "Acabado"]
    existing_target_columns = [
        c for c in target_columns if c in transformed.columns
    ]
    if existing_target_columns:
        transformed = transformed.drop(columns=existing_target_columns)

    transformed = transformed.drop(columns=[source_col])
    transformed.insert(source_index, "Core", core)
    transformed.insert(source_index + 1, "Gama", gama)
    transformed.insert(source_index + 2, "Acabado", acabado)
    return transformed


def transform_apertura(
    df_user: pd.DataFrame,
    map_aperturas: dict,
    col: str = "Apertura",
) -> pd.DataFrame:
    """Transforma la columna Apertura usando equivalencias de la base de datos.

    Ejemplo mínimo de uso:
        df_resultado = transform_apertura(df_user, map_aperturas)
    """
    if col not in df_user.columns:
        raise ValueError(f"No se encontró la columna '{col}' en df_user.")

    # Normalizamos claves de equivalencia para que 1, "1" y "1.0" se traten como el mismo código.
    normalized_map_aperturas = {}
    for key, value in map_aperturas.items():
        if pd.isna(key):
            continue
        normalized_key = str(key).strip()
        normalized_key = re.sub(r"^(-?\d+)\.0+$", r"\1", normalized_key)
        normalized_map_aperturas[normalized_key] = value

    # NaN/None se tratan como "", luego se convierte a texto y se hace strip.
    normalized_values = df_user[col].fillna("").astype(str).str.strip()
    normalized_values = normalized_values.str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)

    # Si hay equivalencia, usamos el valor mapeado; si no, conservamos el valor original normalizado.
    mapped_values = normalized_values.map(normalized_map_aperturas)
    df_user[col] = mapped_values.where(mapped_values.notna(), normalized_values)

    return df_user


def transform_tiradores(df_user: pd.DataFrame, map_tiradores: dict) -> pd.DataFrame:
    """Transforma la columna de tirador y garantiza salida en columna 'Tirador'."""
    transformed = df_user.copy()

    source_col = find_column_name(transformed.columns, "Tirador")
    if source_col is None:
        source_col = find_column_name(transformed.columns, "Tirador(0=sin tirador)")
    if source_col is None:
        return transformed

    normalized_map_tiradores: dict[str, str] = {}
    for key, value in map_tiradores.items():
        if pd.isna(key):
            continue
        normalized_key = str(key).strip()
        if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized_key):
            try:
                float_value = float(normalized_key)
                if float_value.is_integer():
                    normalized_key = str(int(float_value))
            except ValueError:
                pass
        normalized_map_tiradores[normalized_key] = "" if pd.isna(value) else str(value)

    original_values = transformed[source_col]
    normalized_values = original_values.fillna("").astype(str).str.strip()

    normalized_keys = normalized_values.copy()
    numeric_mask = normalized_keys.str.fullmatch(r"-?\d+(?:\.\d+)?", na=False)
    numeric_series = pd.to_numeric(normalized_keys[numeric_mask], errors="coerce")
    integer_mask = numeric_series.notna() & (numeric_series % 1 == 0)
    normalized_keys.loc[numeric_series[integer_mask].index] = numeric_series[integer_mask].astype(int).astype(str)

    mapped_values = normalized_keys.map(normalized_map_tiradores)
    transformed_values = mapped_values.where(mapped_values.notna(), normalized_values)

    target_index = transformed.columns.get_loc(source_col)
    if source_col != "Tirador" and "Tirador" in transformed.columns:
        transformed = transformed.drop(columns=["Tirador"])
        target_index = transformed.columns.get_loc(source_col)

    transformed[source_col] = transformed_values
    if source_col != "Tirador":
        transformed = transformed.rename(columns={source_col: "Tirador"})
        if transformed.columns.get_loc("Tirador") != target_index:
            tirador_series = transformed.pop("Tirador")
            transformed.insert(target_index, "Tirador", tirador_series)

    return transformed


def transform_trasera_tirador(
    df: pd.DataFrame,
    source_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Genera/sobrescribe 'Trasera Tirador' usando campos de origen del CSV."""
    source = df if source_df is None else source_df

    required_columns = ["Tirador", "Acabado"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    source_required_columns = ["TraseraTirador", "Colortirador"]
    missing_source_columns = [
        column for column in source_required_columns if column not in source.columns
    ]
    missing_columns.extend(missing_source_columns)
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(f"Faltan columnas requeridas para 'Trasera Tirador': {missing_list}.")

    transformed = df.copy()
    source_trasera_tirador = source["TraseraTirador"].reindex(transformed.index)
    source_color_tirador = source["Colortirador"].reindex(transformed.index)

    def _normalize_text(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    def _from_trasera_tirador(value: object) -> str:
        normalized = _normalize_text(value)
        if not normalized:
            return ""

        words = normalized.split()
        words = words[:-1] if words else []
        filtered_words = [word for word in words if word.upper() != "WOOD"]
        return " ".join(filtered_words).upper()

    def _compute_row(row: pd.Series) -> str:
        tirador_value = _normalize_text(row["Tirador"])
        if tirador_value == "":
            return ""

        if tirador_value in {"Pill", "Round", "Square"}:
            return _from_trasera_tirador(source_trasera_tirador.loc[row.name])

        if tirador_value == "U-Shape)":
            return _normalize_text(row["Acabado"]).upper()

        return _normalize_text(source_color_tirador.loc[row.name]).upper()

    trasera_tirador = transformed.apply(_compute_row, axis=1)

    if "Trasera Tirador" in transformed.columns:
        transformed = transformed.drop(columns=["Trasera Tirador"])
    transformed.insert(len(transformed.columns), "Trasera Tirador", trasera_tirador)

    return transformed


def transform_dataframe(
    df: pd.DataFrame,
    project_id: str,
    df_materiales: pd.DataFrame,
    map_tiradores: dict,
) -> pd.DataFrame:
    """Transformación de plantilla según requisitos del cliente."""
    transformed = df.copy()

    # Normalización base de texto: trim en columnas string/object.
    text_columns = transformed.select_dtypes(include=["object", "string"]).columns
    for column_name in text_columns:
        transformed[column_name] = transformed[column_name].astype("string").str.strip()

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

    source_for_trasera_tirador = transformed.copy()

    # 3) Reglas de reordenación Lenx/LenY/LenZ por tipología.
    transformed = apply_typology_dimension_rules(transformed)

    # 4) Transformar Material en Core/Gama/Acabado con la BBDD de materiales.
    material_column = find_column_name(transformed.columns, "Material")
    if material_column is None:
        raise ValueError("No se encontró la columna 'Material' en el CSV.")
    transformed = transform_material_to_core_gama_acabado(
        transformed,
        df_materiales,
        source_col=material_column,
    )

    # 5) Transformar Tirador/Tirador(0=sin tirador) con equivalencias y salida final en "Tirador".
    transformed = transform_tiradores(transformed, map_tiradores)

    # 6) Eliminar sufijo "mm" en columnas dimensionales de texto.
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

    # 7) Insertar la columna ID Proyecto en primera posición.
    transformed.insert(0, "ID Proyecto", project_id)

    # 8) Ajustes de columnas para salida final.
    obs_column = find_column_name(transformed.columns, "Obs")
    if obs_column is not None and obs_column != "Observaciones":
        transformed = transformed.rename(columns={obs_column: "Observaciones"})

    transformed = transform_trasera_tirador(
        transformed,
        source_df=source_for_trasera_tirador,
    )

    removable_columns = [
        col_name
        for col_name in ["Tirador(0=sin tirador)", "Hidden", "TraseraTirador", "Colortirador"]
        if find_column_name(transformed.columns, col_name) is not None
    ]
    if removable_columns:
        transformed = transformed.drop(columns=removable_columns)

    transformed = transform_trasera_tirador(transformed)

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
        # Carga interna de base de datos (sin mostrarla en la UI).
        df_dimensiones, df_materiales, map_aperturas, map_tiradores = get_database()

        _ = (df_dimensiones, map_aperturas, map_tiradores)

        # Leemos el CSV de forma segura.
        original_df, delimiter_used, encoding_used = load_csv(uploaded_file)

        st.success(
            f"Archivo leído correctamente (encoding: {encoding_used}, separador detectado: '{delimiter_used}')."
        )

        st.subheader(f"2) Vista previa original ({PREVIEW_ROWS} piezas visibles)")
        st.dataframe(
            original_df,
            width="stretch",
            height=PREVIEW_HEIGHT,
        )

        project_id = get_project_id_from_filename(uploaded_file.name)
        validate_project_id(project_id)

        # Aplicamos plantilla de transformación.
        final_df = transform_dataframe(original_df, project_id, df_materiales, map_tiradores)
        final_df = transform_apertura(final_df, map_aperturas, col="Apertura")

        st.markdown(
            f"<p style='font-size:2.25rem;font-weight:600;margin:0;'>{final_df.shape[0]} piezas</p>",
            unsafe_allow_html=True,
        )

        st.subheader(f"3) Resultado transformado ({PREVIEW_ROWS} piezas visibles)")

        editable_column = find_column_name(final_df.columns, "Observaciones")
        disabled_columns = [col for col in final_df.columns if col != editable_column]

        if editable_column is not None:
            final_df = st.data_editor(
                final_df,
                width="stretch",
                height=PREVIEW_HEIGHT,
                num_rows="fixed",
                disabled=disabled_columns,
            )
        else:
            st.dataframe(
                final_df,
                width="stretch",
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
