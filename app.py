import csv
import hashlib
import html
import math
import re
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import streamlit as st
from mueble_abierto_svg import generar_svg_mueble_abierto


st.set_page_config(page_title="Despiece", page_icon="📋", layout="wide")

PREVIEW_ROWS = 10
PREVIEW_HEIGHT = 390
RESULT_COLUMN_ORDER = [
    "ID Proyecto",
    "SKU",
    "Name",
    "Tipología",
    "LenY",
    "LenZ",
    "Core",
    "Gama",
    "Acabado",
    "MECANIZADO",
    "Tirador",
    "PosicionTirador",
    "Apertura",
    "Trasera Tirador",
    "Observaciones",
    "Orden CSV",
]

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

st.markdown(
    """
    <style>
      /* Refuerza las líneas de la tabla para mejorar visibilidad. */
      .stDataFrame table th,
      .stDataFrame table td {
        border: 1px solid rgba(49, 51, 63, 0.35) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
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


def get_project_subtitle_from_filename(filename: str) -> str:
    """Extrae el texto posterior al prefijo LL-NNNNN para usarlo como subtítulo."""
    clean_name = (filename or "").strip()
    stem = clean_name.rsplit(".", 1)[0].strip()

    match = re.match(r"^[A-Za-z]{2}-\d{5}(.*)$", stem)
    if not match:
        return stem

    subtitle = match.group(1).strip()
    subtitle = subtitle.lstrip("-_ ").strip()
    return subtitle or stem


def get_source_letter_prefix(source_index: int) -> str:
    """Devuelve prefijo alfabético por índice de archivo: 1->A-, 2->B-, ..., 27->AA-."""
    if source_index < 1:
        return ""

    letters: list[str] = []
    value = source_index
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return f"{''.join(reversed(letters))}-"


def add_source_metadata_columns(
    df: pd.DataFrame,
    source_index: int,
    order_column_name: str = "Orden CSV",
    apply_name_prefix: bool = True,
) -> pd.DataFrame:
    """Añade metadatos de origen: prefijo opcional en Name y orden CSV al final."""
    with_source = df.copy()

    name_column = find_column_name(with_source.columns, "Name")
    if apply_name_prefix and name_column is not None:
        name_values = with_source[name_column].fillna("").astype("string").str.strip()
        source_prefix = get_source_letter_prefix(source_index)
        with_source[name_column] = source_prefix + name_values

    if order_column_name in with_source.columns:
        with_source = with_source.drop(columns=[order_column_name])

    with_source[order_column_name] = str(source_index)
    return with_source


def add_section_title_rows(dataframes: list[pd.DataFrame], subtitles: list[str]) -> pd.DataFrame:
    """Inserta una fila de título por cada CSV de origen en el resultado combinado."""
    if not dataframes:
        return pd.DataFrame()

    combined_parts: list[pd.DataFrame] = []
    first_column = str(dataframes[0].columns[0])

    for index, df in enumerate(dataframes):
        subtitle = subtitles[index] if index < len(subtitles) else f"Bloque {index + 1}"

        title_row = pd.DataFrame([{column_name: "" for column_name in df.columns}])
        title_row.at[0, first_column] = subtitle
        combined_parts.append(title_row)
        combined_parts.append(df)

    return pd.concat(combined_parts, ignore_index=True)


def render_sectioned_result_table(dataframes: list[pd.DataFrame], subtitles: list[str]) -> str:
    """Renderiza la tabla combinada con fila de subtítulo usando merge visual (colspan)."""
    if not dataframes:
        return ""

    columns = [str(col) for col in dataframes[0].columns]
    header_html = "".join(f"<th>{html.escape(col)}</th>" for col in columns)

    body_rows: list[str] = []
    for index, df in enumerate(dataframes):
        subtitle = subtitles[index] if index < len(subtitles) else f"Bloque {index + 1}"
        body_rows.append(
            f"<tr class='section-row'><td colspan='{len(columns)}'>{html.escape(subtitle)}</td></tr>"
        )

        for _, row in df.iterrows():
            cells: list[str] = []
            for column_name in df.columns:
                value = row[column_name]
                text = "" if pd.isna(value) else str(value)
                cells.append(f"<td>{html.escape(text)}</td>")
            body_rows.append(f"<tr>{''.join(cells)}</tr>")

    table_html = f"""
    <style>
      .sectioned-table-wrap {{
        max-height: {PREVIEW_HEIGHT}px;
        overflow: auto;
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 0.5rem;
      }}
      table.sectioned-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
      }}
      table.sectioned-table thead th {{
        position: sticky;
        top: 0;
        background: white;
        z-index: 1;
      }}
      table.sectioned-table th,
      table.sectioned-table td {{
        border: 1px solid rgba(49, 51, 63, 0.12);
        padding: 0.35rem 0.5rem;
        text-align: left;
        white-space: nowrap;
      }}
      table.sectioned-table tr.section-row td {{
        background: #f2f2f2;
        font-weight: 700;
        font-size: 1rem;
        text-align: left;
      }}
    </style>
    <div class="sectioned-table-wrap">
      <table class="sectioned-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    """
    return table_html


def find_column_name(columns: pd.Index, target_name: str) -> str | None:
    """Busca una columna ignorando mayúsculas/minúsculas y espacios extremos."""
    normalized_target = target_name.strip().lower()
    for col in columns:
        if str(col).strip().lower() == normalized_target:
            return str(col)
    return None


def detect_u_shape(final_df: pd.DataFrame) -> bool:
    """Detecta U-Shape en Tirador ignorando mayúsculas/minúsculas y espacios."""
    tirador_column = find_column_name(final_df.columns, "Tirador")
    if tirador_column is None:
        return False

    normalized_tirador = (
        final_df[tirador_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "", regex=True)
    )
    return normalized_tirador.eq("u-shape").any()


def get_tl_candidates_mask(final_df: pd.DataFrame) -> pd.Series:
    """Devuelve máscara booleana de candidatas con Tipología/Tipologia que empieza por T o L."""
    tipologia_column = find_column_name(final_df.columns, "Tipología") or find_column_name(final_df.columns, "Tipologia")
    if tipologia_column is None:
        return pd.Series(False, index=final_df.index)

    tip_series = final_df[tipologia_column].fillna("").astype(str).str.strip().str.upper()
    return tip_series.str.startswith(("T", "L"))


def build_u22_row_keys(final_df: pd.DataFrame) -> pd.Series:
    """Construye una clave estable por fila para selección manual de 22mm."""
    name_column = find_column_name(final_df.columns, "Name")
    order_column = find_column_name(final_df.columns, "Orden CSV")

    name_values = final_df[name_column] if name_column is not None else pd.Series("", index=final_df.index)
    order_values = final_df[order_column] if order_column is not None else pd.Series("", index=final_df.index)

    return pd.Series(
        [
            f"{row_index}|{'' if pd.isna(name_value) else str(name_value).strip()}|"
            f"{'' if pd.isna(order_value) else str(order_value).strip()}"
            for row_index, name_value, order_value in zip(final_df.index, name_values, order_values)
        ],
        index=final_df.index,
    )


def append_22mm_to_observaciones(df: pd.DataFrame, mask_22mm: pd.Series) -> pd.DataFrame:
    """Añade 22mm en Observaciones sin duplicados y con separador ' | '."""
    updated_df = df.copy()
    if "Observaciones" not in updated_df.columns:
        updated_df["Observaciones"] = ""

    mask_aligned = mask_22mm.reindex(updated_df.index, fill_value=False).astype(bool)
    if not mask_aligned.any():
        return updated_df

    current_obs = updated_df.loc[mask_aligned, "Observaciones"].fillna("").astype(str).str.strip()
    has_22mm = current_obs.str.contains(r"\b22mm\b", case=False, regex=True)

    to_append_mask = mask_aligned.copy()
    to_append_mask.loc[mask_aligned] = ~has_22mm.values
    if not to_append_mask.any():
        return updated_df

    obs_to_update = updated_df.loc[to_append_mask, "Observaciones"].fillna("").astype(str).str.strip()
    updated_df.loc[to_append_mask, "Observaciones"] = obs_to_update.where(obs_to_update == "", obs_to_update + " | ") + "22mm"
    return updated_df


def natural_sort_key(value: object) -> tuple[object, ...]:
    """Genera clave alfanumérica natural: M2 < M10, C2 < C10."""
    text = "" if pd.isna(value) else str(value).strip()
    tokens = re.findall(r"\d+|\D+", text.upper())
    key: list[object] = []
    for token in tokens:
        if token.isdigit():
            key.append(int(token))
        else:
            key.append(token)
    return tuple(key)


def get_core_name(name_value: object) -> str:
    """Elimina prefijo de origen de una letra y guion (A-, B-, ...), si existe."""
    text = "" if pd.isna(name_value) else str(name_value).strip()
    return re.sub(r"^[A-Za-z]-", "", text, count=1)


def _sort_result_rows_single_block(df_block: pd.DataFrame) -> pd.DataFrame:
    """Aplica reglas de ordenación de piezas en un único bloque de resultados."""
    if df_block.empty:
        return df_block.copy()

    tipologia_column = find_column_name(df_block.columns, "Tipologia") or find_column_name(df_block.columns, "Tipología")
    name_column = find_column_name(df_block.columns, "Name")
    gama_column = find_column_name(df_block.columns, "Gama")
    acabado_column = find_column_name(df_block.columns, "Acabado")

    if name_column is None:
        return df_block.copy().reset_index(drop=True)

    work_df = df_block.copy()
    work_df["__orig_pos"] = range(len(work_df))
    work_df["__core_name"] = work_df[name_column].map(get_core_name)
    work_df["__natural"] = work_df["__core_name"].map(natural_sort_key)

    if tipologia_column is None:
        ordered = work_df.sort_values(by=["__natural", "__orig_pos"], kind="stable")
        return ordered[df_block.columns].reset_index(drop=True)

    tip_series = work_df[tipologia_column].fillna("").astype(str).str.strip().str.upper().str[:1]
    work_df["__tip"] = tip_series

    m_match = work_df["__core_name"].str.extract(r"^M(\d+)-", expand=False)
    work_df["__grupo_m"] = pd.to_numeric(m_match, errors="coerce")

    if gama_column is not None and acabado_column is not None:
        work_df["__pair"] = list(
            zip(
                work_df[gama_column].fillna("").astype(str).str.strip().str.upper(),
                work_df[acabado_column].fillna("").astype(str).str.strip().str.upper(),
            )
        )
    else:
        work_df["__pair"] = [("", "") for _ in range(len(work_df))]

    is_r = work_df["__tip"] == "R"
    is_e = work_df["__tip"] == "E"
    is_b = work_df["__tip"] == "B"
    in_m_group = work_df["__grupo_m"].notna()
    is_lt = work_df["__tip"].isin(["L", "T"])

    rule1_mask = (~is_r) & (~is_e) & in_m_group
    rule2_mask = (~is_r) & (~is_e) & (~in_m_group) & is_lt
    b_non_m_mask = is_b & (~in_m_group)
    pre_r_other_mask = (~is_r) & (~is_e) & (~rule1_mask) & (~rule2_mask) & (~b_non_m_mask)

    rule1_df = work_df[rule1_mask].copy()
    if not rule1_df.empty:
        rule1_df["__rule1_sub"] = 2
        rule1_df.loc[rule1_df["__tip"].isin(["C", "P", "B"]), "__rule1_sub"] = 0
        rule1_df.loc[rule1_df["__tip"].isin(["L", "T"]), "__rule1_sub"] = 1
        rule1_df = rule1_df.sort_values(
            by=["__grupo_m", "__rule1_sub", "__natural", "__orig_pos"],
            kind="stable",
        )

    rule2_df = work_df[rule2_mask].sort_values(by=["__natural", "__orig_pos"], kind="stable")
    pre_r_other_df = work_df[pre_r_other_mask].sort_values(by=["__natural", "__orig_pos"], kind="stable")

    non_r_df = pd.concat([rule1_df, rule2_df, pre_r_other_df], axis=0)
    ordered_indexes: list[int] = []

    if not non_r_df.empty:
        ordered_indexes.extend(non_r_df.index.tolist())

    r_df = work_df[is_r].sort_values(by=["__natural", "__orig_pos"], kind="stable")
    if not r_df.empty:
        if gama_column is not None and acabado_column is not None and not non_r_df.empty:
            ordered_indexes = []
            r_groups: dict[tuple[str, str], list[int]] = {
                pair: group.index.tolist()
                for pair, group in r_df.groupby("__pair", sort=False)
            }

            positions_by_ga: dict[tuple[str, str], list[int]] = {}
            for position, pair in enumerate(non_r_df["__pair"].tolist()):
                positions_by_ga.setdefault(pair, []).append(position)

            valid_block_end_index: dict[tuple[str, str], int] = {}
            for pair, positions in positions_by_ga.items():
                if not positions:
                    continue
                if positions[-1] - positions[0] + 1 == len(positions):
                    valid_block_end_index[pair] = positions[-1]

            non_r_ordered_indices = non_r_df.index.tolist()
            inserted_pairs: set[tuple[str, str]] = set()
            unmatched_r_rows: list[int] = []

            for position, non_r_index in enumerate(non_r_ordered_indices):
                ordered_indexes.append(non_r_index)
                for pair, r_indexes in r_groups.items():
                    if pair in inserted_pairs:
                        continue
                    if valid_block_end_index.get(pair) == position:
                        ordered_indexes.extend(r_indexes)
                        inserted_pairs.add(pair)

            for pair, r_indexes in r_groups.items():
                if pair in inserted_pairs:
                    continue
                unmatched_r_rows.extend(r_indexes)

            ordered_indexes.extend(unmatched_r_rows)
        else:
            ordered_indexes.extend(r_df.index.tolist())

    e_df = work_df[is_e].sort_values(by=["__natural", "__orig_pos"], kind="stable")
    ordered_indexes.extend(e_df.index.tolist())

    b_non_m_df = work_df[b_non_m_mask].sort_values(by=["__natural", "__orig_pos"], kind="stable")
    ordered_indexes.extend(b_non_m_df.index.tolist())

    ordered_seen = set(ordered_indexes)
    leftovers = [idx for idx in work_df.index if idx not in ordered_seen]
    ordered_indexes.extend(leftovers)

    sorted_block = work_df.loc[ordered_indexes, df_block.columns]
    return sorted_block.reset_index(drop=True)


def sort_result_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena filas finales por reglas de tipología y bloques de Orden CSV."""
    if df.empty:
        return df.copy()

    order_column = find_column_name(df.columns, "Orden CSV")
    if order_column is None:
        return _sort_result_rows_single_block(df).reset_index(drop=True)

    order_values = df[order_column].fillna("").astype(str).str.strip()
    unique_order_values = [value for value in pd.unique(order_values) if value != ""]

    numeric_values: list[tuple[int, str]] = []
    all_numeric = True
    for value in unique_order_values:
        if re.fullmatch(r"\d+", value):
            numeric_values.append((int(value), value))
        else:
            all_numeric = False
            break

    if all_numeric:
        ordered_blocks = [value for _, value in sorted(numeric_values, key=lambda item: item[0])]
    else:
        ordered_blocks = unique_order_values

    sorted_parts: list[pd.DataFrame] = []
    for block_value in ordered_blocks:
        block_df = df[order_values == block_value].copy()
        sorted_parts.append(_sort_result_rows_single_block(block_df))

    empty_block_df = df[order_values == ""].copy()
    if not empty_block_df.empty:
        sorted_parts.append(_sort_result_rows_single_block(empty_block_df))

    if not sorted_parts:
        return df.copy().reset_index(drop=True)

    return pd.concat(sorted_parts, ignore_index=True)


def reorder_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reordena columnas de salida priorizando el orden funcional solicitado."""
    reordered = df.copy()

    tipologia_column = find_column_name(reordered.columns, "Tipologia")
    if tipologia_column is not None and tipologia_column != "Tipología":
        reordered = reordered.rename(columns={tipologia_column: "Tipología"})

    available_order = [col for col in RESULT_COLUMN_ORDER if col in reordered.columns]
    remaining_columns = [col for col in reordered.columns if col not in available_order]
    return reordered[available_order + remaining_columns]


def render_observaciones_editor(df: pd.DataFrame, editor_key: str) -> pd.DataFrame:
    """Muestra el resultado permitiendo editar únicamente la columna Observaciones."""
    if "Observaciones" not in df.columns:
        st.dataframe(
            df,
            width="stretch",
            hide_index=False,
            column_config={
                "_index": st.column_config.NumberColumn(
                    "Línea",
                    help="Número de línea de referencia en la tabla.",
                    format="%d",
                )
            },
        )
        return df

    disabled_columns = [col for col in df.columns if col != "Observaciones"]
    return st.data_editor(
        df,
        width="stretch",
        hide_index=False,
        num_rows="fixed",
        disabled=disabled_columns,
        column_config={
            "_index": st.column_config.NumberColumn(
                "Línea",
                help="Número de línea de referencia en la tabla.",
                format="%d",
            ),
            "Observaciones": st.column_config.TextColumn(
                "Observaciones",
                help="Puedes editar esta columna antes de descargar el CSV.",
            )
        },
        key=editor_key,
    )


def render_sectioned_observaciones_editor(
    final_df: pd.DataFrame,
    subtitles: list[str],
    key_prefix: str,
) -> pd.DataFrame:
    """Renderiza editores por bloque con subtítulo en negrita ocupando una sola línea visual."""
    if final_df.empty:
        return final_df.copy()

    order_column = find_column_name(final_df.columns, "Orden CSV")
    if order_column is None:
        return render_observaciones_editor(final_df, f"{key_prefix}_single")

    order_series = final_df[order_column].fillna("").astype(str).str.strip()
    edited_sections: list[pd.DataFrame] = []

    for order_value in pd.unique(order_series):
        if order_value == "":
            continue

        section_df = final_df[order_series == order_value].copy()

        subtitle = f"Bloque {order_value}"
        if order_value.isdigit():
            subtitle_index = int(order_value) - 1
            if 0 <= subtitle_index < len(subtitles):
                subtitle = subtitles[subtitle_index]

        st.markdown(
            f"<div style='font-weight:800; background:#f2f2f2; border:1px solid rgba(49,51,63,0.2);"
            f" border-bottom:none; border-radius:8px 8px 0 0; padding:0.5rem 0.75rem; margin-top:0.6rem;'>{html.escape(subtitle)}</div>",
            unsafe_allow_html=True,
        )

        edited_section_df = render_observaciones_editor(
            section_df,
            f"{key_prefix}_{order_value}",
        )
        edited_sections.append(edited_section_df)

    if not edited_sections:
        return final_df.copy()

    return pd.concat(edited_sections, axis=0).sort_index(kind="stable")


def build_sectioned_editor_dataframe(final_df: pd.DataFrame, subtitles: list[str]) -> pd.DataFrame:
    """Construye una vista con filas de subtítulo para edición de Observaciones."""
    if final_df.empty:
        return final_df.copy()

    order_column = find_column_name(final_df.columns, "Orden CSV")
    if order_column is None:
        return final_df.copy()

    sectioned_parts: list[pd.DataFrame] = []
    order_series = final_df[order_column].fillna("").astype(str).str.strip()

    for order_value in pd.unique(order_series):
        if order_value == "":
            continue

        section_df = final_df[order_series == order_value].copy()
        title_row = pd.DataFrame([{column_name: "" for column_name in final_df.columns}])

        subtitle = f"Bloque {order_value}"
        if order_value.isdigit():
            subtitle_index = int(order_value) - 1
            if 0 <= subtitle_index < len(subtitles):
                subtitle = subtitles[subtitle_index]

        first_column = str(final_df.columns[0])
        title_row.at[0, first_column] = subtitle

        sectioned_parts.append(title_row)
        sectioned_parts.append(section_df)

    if not sectioned_parts:
        return final_df.copy()

    return pd.concat(sectioned_parts, ignore_index=True)


def sync_observaciones_from_sectioned_editor(final_df: pd.DataFrame, edited_sectioned_df: pd.DataFrame) -> pd.DataFrame:
    """Sincroniza Observaciones editadas en vista con subtítulos hacia el DataFrame real."""
    if "Observaciones" not in final_df.columns or "Observaciones" not in edited_sectioned_df.columns:
        return final_df

    order_column = find_column_name(edited_sectioned_df.columns, "Orden CSV")
    if order_column is None:
        return final_df

    real_row_mask = edited_sectioned_df[order_column].fillna("").astype(str).str.strip() != ""
    edited_observaciones = edited_sectioned_df.loc[real_row_mask, "Observaciones"].reset_index(drop=True)

    if edited_observaciones.shape[0] != final_df.shape[0]:
        return final_df

    synced_df = final_df.copy()
    synced_df["Observaciones"] = edited_observaciones.values
    return synced_df


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


def _project_point(x: float, y: float, dx: float, dy: float, to_back: bool = False) -> tuple[float, float]:
    """Proyecta un punto al plano trasero usando un desplazamiento de perspectiva."""
    if not to_back:
        return x, y
    return x + dx, y + dy


def generate_open_cabinet_svg(
    ancho_mm: int,
    alto_mm: int,
    fondo_mm: int,
    num_baldas: int,
    colgado: bool,
    zocalo_mm: int,
) -> str:
    """Genera un SVG paramétrico de mueble abierto respetando reglas geométricas visibles."""
    width_px, height_px, margin = 320, 250, 18

    draw_w = width_px - (2 * margin) - 55
    draw_h = height_px - (2 * margin) - 16
    sx = draw_w / max(float(ancho_mm), 1.0)
    sy = draw_h / max(float(alto_mm), 1.0)
    scale = min(sx, sy)

    w = ancho_mm * scale
    h = alto_mm * scale
    d = max(1.0, fondo_mm * scale)
    t = max(3.0, 19.0 * scale)
    depth_dx = max(20.0, min(52.0, d * 0.27))
    depth_dy = -depth_dx * 0.38

    ox = margin + 12
    oy = margin + h + 4

    zocalo_s = max(0.0, zocalo_mm * scale)
    base_bottom = h - zocalo_s if zocalo_mm > 0 else h
    base_top = base_bottom - t
    inner_top = t
    inner_bottom = base_top

    def p(x: float, y: float, z: float = 0.0) -> tuple[float, float]:
        return ox + x + ((z / max(d, 1e-6)) * depth_dx), oy - y + ((z / max(d, 1e-6)) * depth_dy)

    def line3(a: tuple[float, float, float], b: tuple[float, float, float]) -> str:
        x1, y1 = p(*a)
        x2, y2 = p(*b)
        return f"<line x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}' />"

    parts: list[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width_px}' height='{height_px}' viewBox='0 0 {width_px} {height_px}'>",
        "<rect width='100%' height='100%' fill='white' />",
        "<g stroke='#2f2f2f' stroke-width='1.55' fill='none' stroke-linecap='round' stroke-linejoin='round'>",
    ]

    side_bottom = h if zocalo_mm > 0 else base_bottom

    # Caras visibles exteriores: frontal + superior + lateral derecho.
    parts.extend(
        [
            line3((0, 0, 0), (w, 0, 0)),
            line3((0, 0, 0), (0, side_bottom, 0)),
            line3((w, 0, 0), (w, side_bottom, 0)),
            line3((0, 0, 0), (0, 0, d)),
            line3((w, 0, 0), (w, 0, d)),
            line3((0, 0, d), (w, 0, d)),
            line3((w, 0, d), (w, side_bottom, d)),
            line3((w, side_bottom, 0), (w, side_bottom, d)),
        ]
    )

    # Tapa y base: cara superior + arista frontal inferior visible del espesor.
    parts.extend(
        [
            line3((0, t, 0), (w, t, 0)),
            line3((w, t, 0), (w, t, d)),
            line3((0, base_top, 0), (w, base_top, 0)),
            line3((w, base_top, 0), (w, base_top, d)),
            line3((0, base_bottom, 0), (w, base_bottom, 0)),
        ]
    )

    # Baldas equidistantes entre cara inferior de tapa y cara superior de base.
    shelf_tops: list[float] = []
    if num_baldas > 0 and inner_bottom > inner_top:
        step = (inner_bottom - inner_top) / (num_baldas + 1)
        for i in range(num_baldas):
            center_y = inner_top + ((i + 1) * step)
            shelf_tops.append(center_y - (t / 2))

    shelf_x0 = t
    shelf_x1 = w - t
    for shelf_top in shelf_tops:
        shelf_bottom = shelf_top + t
        parts.extend(
            [
                line3((shelf_x0, shelf_top, 0), (shelf_x1, shelf_top, 0)),
                line3((shelf_x1, shelf_top, 0), (shelf_x1, shelf_top, d)),
                line3((shelf_x0, shelf_bottom, 0), (shelf_x1, shelf_bottom, 0)),
                line3((shelf_x1, shelf_bottom, 0), (shelf_x1, shelf_bottom, d)),
            ]
        )

    # Arista trasera derecha del lateral izquierdo en tramos visibles.
    rear_x = p(0, 0, d)[0]
    y_cursor = p(0, t + 0.6, d)[1]
    y_end = p(0, base_top, d)[1]
    for shelf_top in shelf_tops:
        shelf_back_top = p(0, shelf_top, d)[1]
        shelf_front_bottom = p(0, shelf_top + t, 0)[1]
        if shelf_back_top > y_cursor:
            parts.append(f"<line x1='{rear_x:.2f}' y1='{y_cursor:.2f}' x2='{rear_x:.2f}' y2='{shelf_back_top:.2f}' />")
        y_cursor = shelf_front_bottom
    if y_end > y_cursor:
        parts.append(f"<line x1='{rear_x:.2f}' y1='{y_cursor:.2f}' x2='{rear_x:.2f}' y2='{y_end:.2f}' />")

    # Trasera real en límites internos exactos (sin invadir piezas).
    back_left = t
    back_right = w - t
    parts.extend(
        [
            line3((back_left, inner_bottom, d), (back_right, inner_bottom, d)),
            line3((back_left, inner_top, d), (back_left, inner_bottom, d)),
            line3((back_right, inner_top, d), (back_right, inner_bottom, d)),
        ]
    )

    # Zócalo opcional retranqueado 100 mm hacia atrás en la misma línea de perspectiva.
    if zocalo_mm > 0:
        z_h = max(0.0, (zocalo_mm - 5) * scale)
        z_top = base_bottom - (5 * scale)
        z_bottom = min(h, z_top + z_h)
        z_depth = min(d, 100 * scale)
        z_left = t
        z_right = w - t
        parts.extend(
            [
                line3((z_left, z_top, z_depth), (z_left, z_bottom, z_depth)),
                line3((z_left, z_bottom, z_depth), (z_right, z_bottom, z_depth)),
                line3((z_right, z_top, z_depth), (z_right, z_bottom, z_depth)),
            ]
        )

    # Agujeros de colgar (Ø17) sobre trasera real; solo visibles.
    if colgado:
        hole_r = max(1.5, (17 * scale) / 2)
        hole_y = inner_top + (75 * scale)
        hole_left_x = back_left + (16.5 * scale)
        hole_right_x = back_right - (16.5 * scale)

        hx_l, hy_l = p(hole_left_x, hole_y, d)
        parts.append(
            f"<ellipse cx='{hx_l:.2f}' cy='{hy_l:.2f}' rx='{hole_r * 0.95:.2f}' ry='{hole_r * 0.75:.2f}' />"
        )

        hx_r, hy_r = p(hole_right_x, hole_y, d)
        if hx_r < p(w, hole_y, 0)[0] - 0.2:
            parts.append(
                f"<ellipse cx='{hx_r:.2f}' cy='{hy_r:.2f}' rx='{hole_r * 0.95:.2f}' ry='{hole_r * 0.75:.2f}' />"
            )

    parts.append("</g></svg>")
    return "".join(parts)


def _default_open_cabinet() -> dict[str, object]:
    """Devuelve la configuración por defecto de un mueble abierto."""
    return {
        "ancho_mm": 600,
        "alto_mm": 800,
        "fondo_mm": 396,
        "num_baldas": 1,
        "colgado": False,
        "zocalo_mm": 0,
        "aceptado": False,
        "svg": "",
    }


def _ensure_open_cabinets_state() -> None:
    """Inicializa el estado base de la sección de muebles abiertos."""
    if "open_cabinets_visible" not in st.session_state:
        st.session_state["open_cabinets_visible"] = False
    if "muebles_abiertos" not in st.session_state:
        st.session_state["muebles_abiertos"] = []


def _sync_open_cabinets_count(target_count: int) -> None:
    """Ajusta la lista de muebles abiertos al número solicitado."""
    cabinets = st.session_state["muebles_abiertos"]
    while len(cabinets) < target_count:
        cabinets.append(_default_open_cabinet())
    if len(cabinets) > target_count:
        del cabinets[target_count:]


def _build_open_cabinet_description(cabinet: dict[str, object]) -> str:
    """Construye descripción textual del mueble abierto con pluralización."""
    num_baldas = int(cabinet["num_baldas"])
    baldas_text = "1 balda" if num_baldas == 1 else f"{num_baldas} baldas"
    colgado_text = "colgado" if bool(cabinet["colgado"]) else "no colgado"
    return (
        f"{cabinet['ancho_mm']} x {cabinet['alto_mm']} x {cabinet['fondo_mm']} mm · "
        f"{baldas_text} · {colgado_text} · rodapié {cabinet['zocalo_mm']} mm"
    )


def _render_open_cabinet_card(index: int) -> None:
    """Renderiza una tarjeta individual de mueble abierto con modo editar/aceptado."""
    cabinet = st.session_state["muebles_abiertos"][index]

    card_key = f"mueble_card_{index}"
    with st.container(key=card_key, border=True):
        st.markdown(
            """
            <style>
              [class*="st-key-mueble_card_"] {
                background-color: #fbfcff;
                border-radius: 12px;
                padding: 0.35rem;
                margin: 0 auto 0.5rem auto;
                width: 270px;
                max-width: 270px;
                min-height: 430px;
              }
              [class*="st-key-mueble_card_"] .open-cabinet-preview {
                width: 200px;
                height: 160px;
                margin: 8px auto;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
                border: 1px solid #e7ebf2;
                border-radius: 8px;
                background: #ffffff;
              }
              [class*="st-key-mueble_card_"] .open-cabinet-preview svg {
                width: 100%;
                height: 100%;
                object-fit: contain;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"**Mueble abierto {index + 1}**")

        if not cabinet.get("aceptado", False):
            ancho_mm = int(
                st.number_input(
                    "ancho en mm",
                    min_value=0,
                    max_value=5000,
                    step=1,
                    value=int(cabinet.get("ancho_mm", 600)),
                    key=f"mueble_abierto_ancho_{index}",
                )
            )
            alto_mm = int(
                st.number_input(
                    "alto en mm",
                    min_value=0,
                    max_value=5000,
                    step=1,
                    value=int(cabinet.get("alto_mm", 800)),
                    key=f"mueble_abierto_alto_{index}",
                )
            )
            fondo_mm = int(
                st.number_input(
                    "fondo en mm",
                    min_value=0,
                    max_value=5000,
                    step=1,
                    value=int(cabinet.get("fondo_mm", 396)),
                    key=f"mueble_abierto_fondo_{index}",
                )
            )
            num_baldas = int(
                st.number_input(
                    "número de baldas",
                    min_value=0,
                    max_value=20,
                    step=1,
                    value=int(cabinet.get("num_baldas", 1)),
                    key=f"mueble_abierto_baldas_{index}",
                )
            )
            zocalo_mm = int(
                st.number_input(
                    "altura de rodapié en mm",
                    min_value=0,
                    max_value=500,
                    step=1,
                    value=int(cabinet.get("zocalo_mm", 0)),
                    key=f"mueble_abierto_zocalo_{index}",
                )
            )
            colgado = st.checkbox(
                "Lleva herrajes de colgar",
                value=bool(cabinet.get("colgado", False)),
                key=f"mueble_abierto_colgado_{index}",
            )

            if st.button("Aceptar", key=f"mueble_abierto_aceptar_{index}", use_container_width=True):
                svg = generar_svg_mueble_abierto(
                    ancho_mm=ancho_mm,
                    alto_mm=alto_mm,
                    fondo_mm=fondo_mm,
                    num_baldas=num_baldas,
                    colgado=colgado,
                    zocalo_mm=zocalo_mm,
                )
                st.session_state["muebles_abiertos"][index] = {
                    "ancho_mm": ancho_mm,
                    "alto_mm": alto_mm,
                    "fondo_mm": fondo_mm,
                    "num_baldas": num_baldas,
                    "colgado": colgado,
                    "zocalo_mm": zocalo_mm,
                    "aceptado": True,
                    "svg": svg,
                }
                st.rerun()
        else:
            svg = str(cabinet.get("svg", ""))
            st.markdown(
                f"<div class='open-cabinet-preview'>{svg}</div>",
                unsafe_allow_html=True,
            )
            st.caption(_build_open_cabinet_description(cabinet))

            if st.button("Editar", key=f"mueble_abierto_editar_{index}", use_container_width=True):
                st.session_state["muebles_abiertos"][index]["aceptado"] = False
                st.session_state["muebles_abiertos"][index]["svg"] = ""
                st.rerun()


def render_open_cabinet_generator_section() -> None:
    """Renderiza sección de muebles abiertos con tarjetas independientes."""
    _ensure_open_cabinets_state()

    control_col_button, control_col_select = st.columns([3.4, 1], vertical_alignment="bottom")
    with control_col_button:
        if st.button("Añadir muebles abiertos"):
            st.session_state["open_cabinets_visible"] = True

    if not st.session_state["open_cabinets_visible"]:
        return

    current_count = len(st.session_state["muebles_abiertos"])
    opciones_cantidad = list(range(0, 11))
    default_index = current_count if current_count in opciones_cantidad else min(current_count, 10)

    with control_col_select:
        cantidad_muebles_abiertos = int(
            st.selectbox(
                "Cantidad",
                options=opciones_cantidad,
                index=default_index,
                key="cantidad_muebles_abiertos",
                label_visibility="collapsed",
            )
        )

    _sync_open_cabinets_count(cantidad_muebles_abiertos)

    cards_per_row = 4
    for row_start in range(0, cantidad_muebles_abiertos, cards_per_row):
        row_indexes = range(row_start, min(row_start + cards_per_row, cantidad_muebles_abiertos))
        row_columns = st.columns(cards_per_row, gap="small")
        for col_position, card_index in enumerate(row_indexes):
            with row_columns[col_position]:
                _render_open_cabinet_card(card_index)


TIP_RULES = {
    "B": re.compile(r"^B\d+$"),
    "E": re.compile(r"^(?:E\d+|FE\d+|M\d+-(?:E|FE)\d+|P\d+-(?:E|FE)\d+)$"),
    "L": re.compile(r"^(?:PL\d+|M\d+-PL\d+|P\d+-PL\d+|H\d+-PL\d+)$"),
    "P": re.compile(r"^(?:P\d+|M\d+-P\d+)$"),
    "C": re.compile(r"^(?:C\d+|M\d+-C\d+)$"),
    "X": re.compile(r"^(?:P\d+|P\d+-P\d+|AP\d+-P\d+)$"),
    "R": re.compile(r"^R\d+$"),
    "TALL": re.compile(r"^(?:T\d+|M\d+-T\d+|P\d+-T\d+|H\d+-T\d+|AP\d+-T\d+)$"),
    "CT": re.compile(r"^CT\d+$"),
    "MCT": re.compile(r"^M\d+-CT\d+$"),
    "PCT": re.compile(r"^P\d+-CT\d+$"),
}


def parse_dimension_to_float(value: object) -> float | None:
    """Convierte dimensiones textuales a float eliminando mm, comas y espacios."""
    if pd.isna(value):
        return None

    clean_text = str(value).strip().lower()
    if clean_text == "":
        return None

    clean_text = clean_text.replace("mm", "")
    clean_text = clean_text.replace(" ", "")
    clean_text = clean_text.replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", clean_text)
    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def get_prefix_from_name(name: object) -> str:
    """Obtiene prefijo de Name con patrón LETRA-; fallback a A."""
    if pd.isna(name):
        return "A"

    match = re.match(r"^\s*([A-Za-z])-", str(name))
    if match:
        return match.group(1).upper()

    return "A"


def _prefix_to_source_index(prefix: str) -> int | None:
    """Convierte prefijo alfabético (A, B, ..., AA) a índice 1-based."""
    clean_prefix = (prefix or "").strip().upper()
    if clean_prefix == "" or not clean_prefix.isalpha():
        return None

    value = 0
    for character in clean_prefix:
        value = value * 26 + (ord(character) - ord("A") + 1)
    return value


def recalculate_r_bars(final_df: pd.DataFrame) -> pd.DataFrame:
    """Recalcula barras de tipología R/RV agrupando tramos por barra equivalente."""
    if final_df.empty:
        return final_df.copy()

    tipologia_column = find_column_name(final_df.columns, "Tipologia") or find_column_name(final_df.columns, "Tipología")
    name_column = find_column_name(final_df.columns, "Name")
    ancho_column = find_column_name(final_df.columns, "Ancho") or find_column_name(final_df.columns, "LenY")
    tramo_column = find_column_name(final_df.columns, "Largo") or find_column_name(final_df.columns, "LenZ")
    core_column = find_column_name(final_df.columns, "Core")
    gama_column = find_column_name(final_df.columns, "Gama")
    acabado_column = find_column_name(final_df.columns, "Acabado")
    orden_column = find_column_name(final_df.columns, "Orden CSV")
    q_column = find_column_name(final_df.columns, "Q")
    id_proyecto_column = find_column_name(final_df.columns, "ID Proyecto")
    sku_column = find_column_name(final_df.columns, "SKU")

    if tipologia_column is None or ancho_column is None or tramo_column is None:
        return final_df.copy()

    def _normalize_text(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip().upper()

    def _prefix_from_row(row: pd.Series) -> str:
        prefix = get_prefix_from_name(row.get(name_column) if name_column is not None else "")
        if prefix != "A":
            return prefix

        for source_column in [orden_column, q_column]:
            if source_column is None:
                continue

            source_value = row.get(source_column)
            if pd.isna(source_value):
                continue

            source_text = str(source_value).strip()
            if source_text == "":
                continue

            if source_text.isdigit() and int(source_text) > 0:
                return get_source_letter_prefix(int(source_text)).rstrip("-") or "A"

            source_match = re.match(r"^\s*([A-Za-z])", source_text)
            if source_match:
                return source_match.group(1).upper()

        return "A"

    tip_series = final_df[tipologia_column].fillna("").astype(str).str.strip().str.upper()
    rr_mask = tip_series.isin(["R", "RV"])
    if not rr_mask.any():
        return final_df.copy()

    grouped_lengths: dict[tuple[str, float, str, str, str], float] = {}
    grouped_sources: dict[tuple[str, float, str, str, str], dict[str, str]] = {}
    valid_rr_indexes: list[int] = []

    should_use_prefix = False
    if orden_column is not None:
        order_values = final_df[orden_column].fillna("").astype(str).str.strip()
        should_use_prefix = order_values[order_values != ""].nunique() > 1
    elif name_column is not None:
        name_values = final_df[name_column].fillna("").astype(str)
        should_use_prefix = name_values.str.match(r"^\s*[A-Za-z]-").any()

    next_r_index_by_prefix: dict[str, int] = {}
    if name_column is not None:
        preserved_tip_mask = ~rr_mask
        if preserved_tip_mask.any():
            for _, preserved_row in final_df.loc[preserved_tip_mask].iterrows():
                raw_name = preserved_row.get(name_column)
                if pd.isna(raw_name):
                    continue

                name_text = str(raw_name).strip().upper()
                if should_use_prefix:
                    match = re.match(r"^\s*([A-Za-z])\s*-\s*R(\d+)\s*$", name_text)
                    if not match:
                        continue
                    prefix_name = match.group(1)
                    index_value = int(match.group(2))
                else:
                    match = re.match(r"^\s*R(\d+)\s*$", name_text)
                    if not match:
                        continue
                    prefix_name = ""
                    index_value = int(match.group(1))

                next_r_index_by_prefix[prefix_name] = max(next_r_index_by_prefix.get(prefix_name, 0), index_value)

    for row_index in final_df.index[rr_mask]:
        row = final_df.loc[row_index]
        ancho_value = parse_dimension_to_float(row.get(ancho_column))
        tramo_value = parse_dimension_to_float(row.get(tramo_column))
        if ancho_value is None or tramo_value is None:
            continue

        group_key = (
            _prefix_from_row(row),
            float(ancho_value),
            _normalize_text(row.get(core_column)) if core_column is not None else "",
            _normalize_text(row.get(gama_column)) if gama_column is not None else "",
            _normalize_text(row.get(acabado_column)) if acabado_column is not None else "",
        )
        grouped_lengths[group_key] = grouped_lengths.get(group_key, 0.0) + float(tramo_value)

        if group_key not in grouped_sources:
            source_meta: dict[str, str] = {}
            if orden_column is not None:
                raw_order = row.get(orden_column)
                source_meta["orden"] = "" if pd.isna(raw_order) else str(raw_order).strip()
            if q_column is not None:
                raw_q = row.get(q_column)
                source_meta["q"] = "" if pd.isna(raw_q) else str(raw_q).strip()
            if id_proyecto_column is not None:
                raw_project_id = row.get(id_proyecto_column)
                source_meta["id_proyecto"] = "" if pd.isna(raw_project_id) else str(raw_project_id).strip()
            grouped_sources[group_key] = source_meta

        valid_rr_indexes.append(row_index)

    if not valid_rr_indexes:
        return final_df.copy()

    preserved_df = final_df.drop(index=valid_rr_indexes).copy()

    new_rows: list[dict[str, object]] = []
    for (prefix, ancho, core, gama, acabado), total_length in grouped_lengths.items():
        n_barras = int(math.ceil(total_length / 2300))
        for _ in range(1, n_barras + 1):
            new_row = {column_name: "" for column_name in final_df.columns}
            new_row[tipologia_column] = "R"
            if name_column is not None:
                counter_key = prefix if should_use_prefix else ""
                next_index = next_r_index_by_prefix.get(counter_key, 0) + 1
                next_r_index_by_prefix[counter_key] = next_index
                new_row[name_column] = f"{prefix}-R{next_index}" if should_use_prefix else f"R{next_index}"
            if sku_column is not None:
                new_row[sku_column] = "R"
            new_row[ancho_column] = str(int(round(ancho)))
            new_row[tramo_column] = "2400"
            if core_column is not None:
                new_row[core_column] = core
            if gama_column is not None:
                new_row[gama_column] = gama
            if acabado_column is not None:
                new_row[acabado_column] = acabado

            source_meta = grouped_sources.get((prefix, ancho, core, gama, acabado), {})
            if orden_column is not None:
                order_value = source_meta.get("orden", "")
                if order_value == "":
                    source_index = _prefix_to_source_index(prefix)
                    order_value = "" if source_index is None else str(source_index)
                new_row[orden_column] = order_value

            if q_column is not None and source_meta.get("q", "") != "":
                new_row[q_column] = source_meta["q"]
            if id_proyecto_column is not None and source_meta.get("id_proyecto", "") != "":
                new_row[id_proyecto_column] = source_meta["id_proyecto"]

            new_rows.append(new_row)

    if not new_rows:
        return preserved_df.reset_index(drop=True)

    recalculated_df = pd.DataFrame(new_rows, columns=final_df.columns)
    return pd.concat([preserved_df, recalculated_df], ignore_index=True)


def apply_lac_cor_mecanizado(final_df: pd.DataFrame) -> pd.DataFrame:
    """Asigna "cor." en MECANIZADO para gama LAC con dimensión pequeña y mecanizado vacío."""
    required_columns = ["Gama", "MECANIZADO", "LenY", "LenZ"]
    if any(column_name not in final_df.columns for column_name in required_columns):
        return final_df

    updated_df = final_df.copy()
    gama_norm = updated_df["Gama"].fillna("").astype(str).str.strip().str.upper()
    mecanizado_norm = updated_df["MECANIZADO"].fillna("").astype(str).str.strip()

    leny_num = updated_df["LenY"].apply(parse_dimension_to_float)
    lenz_num = updated_df["LenZ"].apply(parse_dimension_to_float)
    small_dim = (
        (leny_num.notna() & (leny_num < 100))
        | (lenz_num.notna() & (lenz_num < 100))
    )

    apply_mask = (gama_norm == "LAC") & (mecanizado_norm == "") & small_dim
    updated_df.loc[apply_mask, "MECANIZADO"] = "cor."
    return updated_df


def fit_dimensions_to_standards(
    final_df: pd.DataFrame,
    df_dimensiones: pd.DataFrame,
    tolerance: float = 1.75,
) -> pd.DataFrame:
    """Ajusta Ancho/Largo de piezas E/T/L/B a estándares definidos en df_dimensiones."""

    def _parse_float(value: object) -> float | None:
        if pd.isna(value):
            return None

        clean_text = str(value).strip().lower()
        if clean_text == "":
            return None

        clean_text = clean_text.replace("mm", "").replace(",", ".")
        match = re.search(r"-?\d+(?:\.\d+)?", clean_text)
        if not match:
            return None

        try:
            return float(match.group(0))
        except ValueError:
            return None

    def _pick_by_distance(candidates: list[tuple[float, float]], width: float, length: float) -> tuple[float, float] | None:
        if not candidates:
            return None
        return min(candidates, key=lambda item: abs(item[0] - width) + abs(item[1] - length))

    def _pick_by_coverage(candidates: list[tuple[float, float]], width: float, length: float) -> tuple[float, float] | None:
        if not candidates:
            return None
        return min(candidates, key=lambda item: (item[0] - width) + (item[1] - length))

    def _format_int(value: float) -> str:
        return str(int(round(value)))

    tipologia_column = find_column_name(final_df.columns, "Tipologia") or find_column_name(final_df.columns, "Tipología")
    width_column = next(
        (col for col in ["Ancho", "LenY", "Width"] if find_column_name(final_df.columns, col) is not None),
        None,
    )
    length_column = next(
        (
            col
            for col in ["Largo", "Alto", "LenZ", "Height"]
            if find_column_name(final_df.columns, col) is not None
        ),
        None,
    )

    if tipologia_column is None or width_column is None or length_column is None:
        return final_df

    resolved_width_column = find_column_name(final_df.columns, width_column)
    resolved_length_column = find_column_name(final_df.columns, length_column)
    if resolved_width_column is None or resolved_length_column is None:
        return final_df

    standards = df_dimensiones.copy()
    if standards.empty:
        return final_df

    standards_tip = standards["Tipologia"].fillna("").astype(str).str.strip().str.upper().str[:1]
    standards_width = standards["Ancho"].apply(_parse_float)
    standards_length = standards["Largo"].apply(_parse_float)
    valid_mask = standards_tip.isin(["E", "T", "L", "B"]) & standards_width.notna() & standards_length.notna()

    grouped_standards: dict[str, list[tuple[float, float]]] = {}
    for std_tip, std_width, std_length in zip(
        standards_tip[valid_mask],
        standards_width[valid_mask],
        standards_length[valid_mask],
    ):
        grouped_standards.setdefault(std_tip, []).append((float(std_width), float(std_length)))

    adjusted = final_df.copy()

    for row_index, row in adjusted.iterrows():
        raw_tip = "" if pd.isna(row.get(tipologia_column)) else str(row.get(tipologia_column)).strip().upper()
        tip = raw_tip[:1]
        if tip not in {"E", "T", "L", "B"}:
            continue

        width = _parse_float(row.get(resolved_width_column))
        length = _parse_float(row.get(resolved_length_column))
        if width is None or length is None:
            continue

        standards_for_tip = grouped_standards.get(tip, [])
        if not standards_for_tip:
            adjusted.at[row_index, resolved_width_column] = _format_int(math.ceil(width))
            adjusted.at[row_index, resolved_length_column] = _format_int(math.ceil(length))
            continue

        block_rotation = False
        gama_column = find_column_name(adjusted.columns, "Gama")
        acabado_column = find_column_name(adjusted.columns, "Acabado")
        if gama_column is not None:
            gama_value = "" if pd.isna(row.get(gama_column)) else str(row.get(gama_column)).strip().upper()
            if gama_value == "WOO":
                block_rotation = True
        if acabado_column is not None:
            acabado_value = "" if pd.isna(row.get(acabado_column)) else str(row.get(acabado_column)).strip().upper()
            if acabado_value == "METAL":
                block_rotation = True

        chosen_standard: tuple[float, float] | None = None

        if tip == "T":
            valid_candidates = [(sw, sl) for sw, sl in standards_for_tip if width < (sw - 15)]
            near_candidates = [(sw, sl) for sw, sl in valid_candidates if abs(sl - length) <= tolerance]
            chosen_standard = _pick_by_distance(near_candidates, width, length)

            if chosen_standard is None:
                cover_candidates = [(sw, sl) for sw, sl in valid_candidates if sl >= length]
                if cover_candidates:
                    chosen_standard = min(cover_candidates, key=lambda item: item[1] - length)
        else:
            near_candidates = [
                (sw, sl)
                for sw, sl in standards_for_tip
                if abs(sw - width) <= tolerance and abs(sl - length) <= tolerance
            ]
            chosen_standard = _pick_by_distance(near_candidates, width, length)

            if chosen_standard is None:
                cover_candidates = [(sw, sl) for sw, sl in standards_for_tip if sw >= width and sl >= length]
                chosen_standard = _pick_by_coverage(cover_candidates, width, length)

            if chosen_standard is None and not block_rotation:
                rotated_width, rotated_length = length, width
                near_candidates_rotated = [
                    (sw, sl)
                    for sw, sl in standards_for_tip
                    if abs(sw - rotated_width) <= tolerance and abs(sl - rotated_length) <= tolerance
                ]
                chosen_standard = _pick_by_distance(near_candidates_rotated, rotated_width, rotated_length)

                if chosen_standard is None:
                    cover_candidates_rotated = [
                        (sw, sl) for sw, sl in standards_for_tip if sw >= rotated_width and sl >= rotated_length
                    ]
                    chosen_standard = _pick_by_coverage(cover_candidates_rotated, rotated_width, rotated_length)

        if chosen_standard is None:
            adjusted.at[row_index, resolved_width_column] = _format_int(math.ceil(width))
            adjusted.at[row_index, resolved_length_column] = _format_int(math.ceil(length))
        else:
            adjusted.at[row_index, resolved_width_column] = _format_int(chosen_standard[0])
            adjusted.at[row_index, resolved_length_column] = _format_int(chosen_standard[1])

    return adjusted


def validate_name_by_tipologia(
    tipologia: object,
    name: object,
    ancho: float | None,
    alto: float | None,
) -> tuple[bool, str]:
    """Valida Name según tipología y excepciones de dimensiones."""
    tipologia_str = str(tipologia).strip().upper()
    tipologia_key = tipologia_str[:1] if tipologia_str else ""

    name_str = "" if pd.isna(name) else str(name).strip()
    if name_str == "":
        return False, "Name vacío"

    core = re.sub(r"^[A-Za-z]-", "", name_str)

    if tipologia_key == "B":
        return (True, "") if TIP_RULES["B"].fullmatch(core) else (False, "Formato inválido para tipología B")
    if tipologia_key == "E":
        return (True, "") if TIP_RULES["E"].fullmatch(core) else (False, "Formato inválido para tipología E")
    if tipologia_key == "L":
        return (True, "") if TIP_RULES["L"].fullmatch(core) else (False, "Formato inválido para tipología L")
    if tipologia_key == "P":
        return (True, "") if TIP_RULES["P"].fullmatch(core) else (False, "Formato inválido para tipología P")
    if tipologia_key == "C":
        is_c_valid = TIP_RULES["C"].fullmatch(core) is not None
        is_width_exception = ancho is not None and round(ancho) == 596 and TIP_RULES["P"].fullmatch(core) is not None
        return (True, "") if is_c_valid or is_width_exception else (False, "Formato inválido para tipología C")
    if tipologia_key == "X":
        return (True, "") if TIP_RULES["X"].fullmatch(core) else (False, "Formato inválido para tipología X")
    if tipologia_key == "R":
        return (True, "") if TIP_RULES["R"].fullmatch(core) else (False, "Formato inválido para tipología R")
    if tipologia_key == "T":
        is_t_valid = TIP_RULES["TALL"].fullmatch(core) is not None
        is_ct_valid = any(TIP_RULES[key].fullmatch(core) is not None for key in ["CT", "MCT", "PCT"])
        rounded_pair = (
            round(ancho) if ancho is not None else None,
            round(alto) if alto is not None else None,
        )
        t_exception_pairs = {(598, 50), (598, 38), (50, 598), (38, 598)}
        is_t_exception = rounded_pair in t_exception_pairs and TIP_RULES["C"].fullmatch(core) is not None
        return (True, "") if is_t_valid or is_ct_valid or is_t_exception else (False, "Formato inválido para tipología T")

    return False, "Tipología no reconocida"


def detect_name_issues(final_df: pd.DataFrame) -> pd.DataFrame:
    """Detecta Name vacío, formato inválido y duplicados en filas de piezas."""
    name_column = find_column_name(final_df.columns, "Name")
    tipologia_column = find_column_name(final_df.columns, "Tipologia") or find_column_name(final_df.columns, "Tipología")
    ancho_column = next(
        (find_column_name(final_df.columns, column) for column in ["Ancho", "LenY", "Width"] if find_column_name(final_df.columns, column) is not None),
        None,
    )
    alto_column = next(
        (find_column_name(final_df.columns, column) for column in ["Alto", "LenZ", "Height"] if find_column_name(final_df.columns, column) is not None),
        None,
    )

    if name_column is None or tipologia_column is None:
        return pd.DataFrame(columns=["fila", "Tipologia", "Name actual", "Motivo del error", "Nuevo Name"])

    issues: list[dict[str, object]] = []
    duplicate_candidates: list[tuple[int, str]] = []
    issue_index: dict[int, int] = {}

    for row_index, row in final_df.iterrows():
        current_name = "" if pd.isna(row.get(name_column)) else str(row.get(name_column)).strip()
        tipologia_value = "" if pd.isna(row.get(tipologia_column)) else str(row.get(tipologia_column)).strip()
        ancho_value = parse_dimension_to_float(row.get(ancho_column)) if ancho_column is not None else None
        alto_value = parse_dimension_to_float(row.get(alto_column)) if alto_column is not None else None

        is_title_row = (
            current_name == ""
            and tipologia_value == ""
            and (ancho_value is None)
            and (alto_value is None)
        )
        if is_title_row:
            continue

        ok, reason = validate_name_by_tipologia(tipologia_value, current_name, ancho_value, alto_value)
        if not ok:
            issue_index[row_index] = len(issues)
            issues.append(
                {
                    "fila": int(row_index),
                    "Tipologia": tipologia_value,
                    "Name actual": current_name,
                    "Motivo del error": reason,
                    "Nuevo Name": current_name,
                }
            )

        normalized_name = current_name.strip().upper()
        if normalized_name:
            duplicate_candidates.append((row_index, normalized_name))

    duplicates: dict[str, list[int]] = {}
    for row_index, normalized_name in duplicate_candidates:
        duplicates.setdefault(normalized_name, []).append(row_index)

    for duplicated_name, rows in duplicates.items():
        if len(rows) <= 1:
            continue
        for row_index in rows:
            other_rows = [str(other) for other in rows if other != row_index]
            duplicate_reason = f"Duplicado con fila(s): {', '.join(other_rows)} (Name: {duplicated_name})"
            if row_index in issue_index:
                issues[issue_index[row_index]]["Motivo del error"] = (
                    f"{issues[issue_index[row_index]]['Motivo del error']} | {duplicate_reason}"
                )
            else:
                source_row = final_df.loc[row_index]
                issues.append(
                    {
                        "fila": int(row_index),
                        "Tipologia": "" if pd.isna(source_row.get(tipologia_column)) else str(source_row.get(tipologia_column)).strip(),
                        "Name actual": "" if pd.isna(source_row.get(name_column)) else str(source_row.get(name_column)).strip(),
                        "Motivo del error": duplicate_reason,
                        "Nuevo Name": "" if pd.isna(source_row.get(name_column)) else str(source_row.get(name_column)).strip(),
                    }
                )

    if not issues:
        return pd.DataFrame(columns=["fila", "Tipologia", "Name actual", "Motivo del error", "Nuevo Name"])

    issues_df = pd.DataFrame(issues)
    return issues_df.sort_values(by="fila", kind="stable").reset_index(drop=True)


def apply_name_fixes(final_df: pd.DataFrame, name_fixes: dict[int, str]) -> pd.DataFrame:
    """Aplica correcciones de Name por índice de fila."""
    if not name_fixes:
        return final_df

    name_column = find_column_name(final_df.columns, "Name")
    if name_column is None:
        return final_df

    updated_df = final_df.copy()
    for row_index, new_name in name_fixes.items():
        if row_index in updated_df.index:
            updated_df.at[row_index, name_column] = "" if pd.isna(new_name) else str(new_name).strip()

    return updated_df


def detect_skirting_shortage_by_source(final_df: pd.DataFrame) -> list[dict[str, float | str]]:
    """Detecta por CSV si LenZ de tipología R es menor que LenY único de M<n> en tipología P."""
    if final_df.empty:
        return []

    order_column = find_column_name(final_df.columns, "Orden CSV")
    tipologia_column = find_column_name(final_df.columns, "Tipologia") or find_column_name(final_df.columns, "Tipología")
    name_column = find_column_name(final_df.columns, "Name")
    leny_column = find_column_name(final_df.columns, "LenY")
    lenz_column = find_column_name(final_df.columns, "LenZ")

    required_columns = [order_column, tipologia_column, name_column, leny_column, lenz_column]
    if any(column_name is None for column_name in required_columns):
        return []

    alerts: list[dict[str, float | str]] = []
    order_series = final_df[order_column].fillna("").astype(str).str.strip()

    for order_value in pd.unique(order_series):
        if order_value == "":
            continue

        source_df = final_df[order_series == order_value].copy()
        tip_series = source_df[tipologia_column].fillna("").astype(str).str.strip().str.upper()

        p_df = source_df[tip_series.str.startswith("P")]
        unique_m_leny: dict[str, float] = {}

        for _, row in p_df.iterrows():
            name_text = "" if pd.isna(row.get(name_column)) else str(row.get(name_column)).upper()
            matches = re.findall(r"\bM(\d+)\b", name_text)
            if not matches:
                continue

            leny_value = parse_dimension_to_float(row.get(leny_column))
            if leny_value is None:
                continue

            for m_number in matches:
                key = f"M{m_number}"
                if key not in unique_m_leny:
                    unique_m_leny[key] = leny_value

        leny_sum = sum(unique_m_leny.values())

        r_df = source_df[tip_series.str.startswith("R")]
        lenz_sum = r_df[lenz_column].apply(parse_dimension_to_float).dropna().sum()

        if lenz_sum < leny_sum:
            alerts.append(
                {
                    "order_value": order_value,
                    "lenz_sum": float(lenz_sum),
                    "leny_sum": float(leny_sum),
                }
            )

    return alerts


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


def clear_posicion_tirador_when_tirador_empty(transformed: pd.DataFrame) -> pd.DataFrame:
    """Vacía PosicionTirador cuando Tirador no tiene valor."""
    tirador_column = find_column_name(transformed.columns, "Tirador")
    posicion_column = find_column_name(transformed.columns, "PosicionTirador")

    if tirador_column is None or posicion_column is None:
        return transformed

    result = transformed.copy()
    tirador_empty = result[tirador_column].fillna("").astype(str).str.strip() == ""
    result.loc[tirador_empty, posicion_column] = ""
    return result


def normalize_posicion_tirador_to_integer(transformed: pd.DataFrame) -> pd.DataFrame:
    """Normaliza PosicionTirador para que solo contenga enteros sin decimales."""
    posicion_column = find_column_name(transformed.columns, "PosicionTirador")
    if posicion_column is None:
        return transformed

    result = transformed.copy()
    raw_values = result[posicion_column].fillna("").astype(str).str.strip()
    non_empty_mask = raw_values != ""
    if not non_empty_mask.any():
        return result

    numeric_values = pd.to_numeric(raw_values[non_empty_mask], errors="coerce")
    invalid_numeric_mask = numeric_values.isna()
    if invalid_numeric_mask.any():
        invalid_examples = raw_values[non_empty_mask][invalid_numeric_mask].drop_duplicates().head(10).tolist()
        raise ValueError(
            "La columna 'PosicionTirador' solo admite números enteros sin decimales. "
            f"Valores no numéricos detectados: {invalid_examples}."
        )

    non_integer_mask = (numeric_values % 1) != 0
    if non_integer_mask.any():
        invalid_examples = raw_values[non_empty_mask][non_integer_mask].drop_duplicates().head(10).tolist()
        raise ValueError(
            "La columna 'PosicionTirador' solo admite números enteros sin decimales. "
            f"Valores con decimales detectados: {invalid_examples}."
        )

    result.loc[non_empty_mask, posicion_column] = numeric_values.astype(int).astype(str)
    return result


def add_trasera_tirador(transformed: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    """Añade/actualiza la columna final 'Trasera Tirador' con reglas de negocio."""
    if not transformed.index.equals(source_df.index):
        raise ValueError(
            "Índices desalineados entre transformed y source_df. "
            "Asegura que source_df esté filtrado con transformed.index."
        )

    if "Tirador" not in transformed.columns:
        raise ValueError("No se encontró la columna 'Tirador' en transformed.")
    if "Acabado" not in transformed.columns:
        raise ValueError("No se encontró la columna 'Acabado' en transformed.")

    required_source_columns = ["TraseraTirador", "Colortirador"]
    missing_source_columns = [col for col in required_source_columns if col not in source_df.columns]
    if missing_source_columns:
        raise ValueError(
            "source_df no tiene las columnas esperadas para Trasera Tirador. "
            f"Faltan: {missing_source_columns}."
        )

    tirador_clean = transformed["Tirador"].fillna("").astype(str).str.strip()
    tirador_key = tirador_clean.str.lower()
    tirador_empty = tirador_clean == ""

    trasera_base = source_df["TraseraTirador"].fillna("").astype(str).str.strip()
    trasera_processed = (
        trasera_base
        .str.split()
        .str[:-1]
        .str.join(" ")
        .str.replace(r"\bWOOD\b", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.upper()
    )

    acabado_upper = transformed["Acabado"].fillna("").astype(str).str.strip().str.upper()
    colortirador_upper = source_df["Colortirador"].fillna("").astype(str).str.strip().str.upper()

    mask_u_shape = tirador_key == "u-shape"
    mask_shapes = tirador_key.isin(["pill", "round", "square"])
    mask_other = (~tirador_empty) & (~mask_shapes) & (~mask_u_shape)

    missing_shapes = mask_shapes & (trasera_base == "")
    missing_u_shape = mask_u_shape & (acabado_upper == "")
    missing_other = mask_other & (colortirador_upper == "")

    def _raise_missing(mask: pd.Series, missing_field: str) -> None:
        if not mask.any():
            return
        tirador_examples = tirador_clean[mask].drop_duplicates().head(10).tolist()
        raise ValueError(
            "Trasera Tirador no puede quedar vacía para Tirador no vacío. "
            f"Filas afectadas: {int(mask.sum())}. "
            f"Ejemplos Tirador: {tirador_examples}. "
            f"Campo faltaba: {missing_field}."
        )

    _raise_missing(missing_shapes, "TraseraTirador")
    _raise_missing(missing_u_shape, "Acabado")
    _raise_missing(missing_other, "Colortirador")

    trasera_tirador = pd.Series("", index=transformed.index, dtype="string")
    trasera_tirador = trasera_tirador.mask(mask_shapes, trasera_processed)
    trasera_tirador = trasera_tirador.mask(mask_u_shape, acabado_upper)
    trasera_tirador = trasera_tirador.mask(mask_other, colortirador_upper)

    invalid_global = (~tirador_empty) & (trasera_tirador.fillna("").astype(str).str.strip() == "")
    if invalid_global.any():
        tirador_examples = tirador_clean[invalid_global].drop_duplicates().head(10).tolist()
        raise ValueError(
            "Trasera Tirador no puede quedar vacía para Tirador no vacío. "
            f"Filas afectadas: {int(invalid_global.sum())}. "
            f"Ejemplos Tirador: {tirador_examples}. "
            "Campo faltaba: indeterminado."
        )

    result = transformed.copy()
    insert_pos = len(result.columns)
    if "TraseraTirador" in source_df.columns:
        source_insert_pos = int(source_df.columns.get_loc("TraseraTirador"))
        insert_pos = min(source_insert_pos, len(result.columns))

    if "Trasera Tirador" in result.columns:
        result = result.drop(columns=["Trasera Tirador"])

    result.insert(insert_pos, "Trasera Tirador", trasera_tirador)

    drop_columns = [
        col_name
        for col_name in ["TraseraTirador", "Colortirador", "ColorTirador"]
        if col_name in result.columns
    ]
    if drop_columns:
        result = result.drop(columns=drop_columns)

    return result


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
    source_for_trasera = transformed.copy()

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
    transformed = clear_posicion_tirador_when_tirador_empty(transformed)
    transformed = normalize_posicion_tirador_to_integer(transformed)
    transformed = add_trasera_tirador(transformed, source_for_trasera)
    forbidden_columns = [
        col_name
        for col_name in ["TraseraTirador", "Colortirador", "ColorTirador"]
        if col_name in transformed.columns
    ]
    if forbidden_columns:
        transformed = transformed.drop(columns=forbidden_columns)

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

    observaciones_column = find_column_name(transformed.columns, "Observaciones")
    if observaciones_column is None:
        transformed["Observaciones"] = ""

    removable_columns = [
        col_name
        for col_name in ["Tirador(0=sin tirador)", "Hidden"]
        if find_column_name(transformed.columns, col_name) is not None
    ]
    if removable_columns:
        transformed = transformed.drop(columns=removable_columns)

    transformed = reorder_result_columns(transformed)
    return transformed.reset_index(drop=True)


def validate_project_id(project_id: str) -> None:
    """Valida que el identificador de proyecto tenga formato LL-NNNNN."""
    if not re.fullmatch(r"[A-Z]{2}-\d{5}", project_id):
        raise ValueError(
            "No se pudo obtener un ID Proyecto válido del nombre del CSV. "
            "El nombre debe empezar por 2 letras, un guion y 5 números (ejemplo: AB-12345)."
        )


def get_project_id_validation_detail(filename: str) -> str:
    """Devuelve una explicación breve del fallo de formato en el nombre del archivo."""
    clean_name = (filename or "").strip()
    stem = clean_name.rsplit(".", 1)[0]
    first_part = re.split(r"[\s_]+", stem, maxsplit=1)[0]

    if not first_part:
        return "No se encontró ningún texto antes de la extensión del archivo."

    if "-" not in first_part:
        return f"'{first_part}' no contiene el guion obligatorio entre letras y números."

    parts = first_part.split("-", 1)
    if len(parts) != 2:
        return f"'{first_part}' no respeta la estructura esperada LL-NNNNN."

    letters_part, numbers_part = parts

    if len(letters_part) != 2 or not letters_part.isalpha():
        return (
            f"'{letters_part}' no es válido: la parte inicial debe tener exactamente 2 letras."
        )

    if len(numbers_part) != 5 or not numbers_part.isdigit():
        return (
            f"'{numbers_part}' no es válido: la parte final debe tener exactamente 5 números."
        )

    return f"'{first_part}' no coincide con el patrón esperado LL-NNNNN."


uploaded_files = st.file_uploader(
    "Sube uno o varios archivos CSV",
    type=["csv"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not uploaded_files:
    st.session_state.pop("uploaded_signature", None)
    st.session_state.pop("name_fixes", None)
    st.session_state.pop("name_review_editor", None)
    st.session_state.pop("name_review_round", None)
    st.session_state.pop("final_df_base", None)
    st.session_state.pop("final_df_confirmed", None)
    st.session_state.pop("final_df_candidate", None)
    st.session_state.pop("corrections_applied", None)
    st.session_state.pop("post_issues_df", None)
    st.session_state.pop("u22_mode", None)
    st.session_state.pop("u22_selected_keys", None)
    for state_key in list(st.session_state.keys()):
        if str(state_key).startswith("name_review_editor_"):
            st.session_state.pop(state_key, None)
    st.info("Empieza subiendo uno o varios archivos CSV para ver la vista previa y aplicar la transformación.")
else:
    file_signature = tuple(
        sorted(
            (
                uploaded_file.name,
                hashlib.sha256(uploaded_file.getvalue()).hexdigest(),
            )
            for uploaded_file in uploaded_files
        )
    )

    previous_signature = st.session_state.get("uploaded_signature")
    if previous_signature != file_signature:
        st.session_state["uploaded_signature"] = file_signature
        st.session_state["name_fixes"] = {}
        st.session_state.pop("name_review_editor", None)
        st.session_state.pop("name_review_round", None)
        st.session_state.pop("final_df_base", None)
        st.session_state.pop("final_df_confirmed", None)
        st.session_state.pop("final_df_candidate", None)
        st.session_state.pop("corrections_applied", None)
        st.session_state.pop("u22_mode", None)
        st.session_state.pop("u22_selected_keys", None)
        st.session_state.pop("post_issues_df", None)
        for state_key in list(st.session_state.keys()):
            if str(state_key).startswith("name_review_editor_"):
                st.session_state.pop(state_key, None)

    try:
        # Carga interna de base de datos (sin mostrarla en la UI).
        df_dimensiones, df_materiales, map_aperturas, map_tiradores = get_database()

        _ = (df_dimensiones, map_aperturas, map_tiradores)

        original_dfs: list[pd.DataFrame] = []
        seen_names: set[str] = set()

        if len(uploaded_files) > 1:
            project_ids: list[str] = []
            for uploaded_file in uploaded_files:
                project_id = get_project_id_from_filename(uploaded_file.name)
                try:
                    validate_project_id(project_id)
                except ValueError as exc:
                    detail = get_project_id_validation_detail(uploaded_file.name)
                    raise ValueError(
                        f"Error en '{uploaded_file.name}': {exc} Detalle del nombre recibido: {detail}"
                    ) from exc
                project_ids.append(project_id)

            if len(set(project_ids)) > 1:
                raise ValueError("Ambos informes deben pertenecer al mismo proyecto y empezar por el mismo ID")

        transformed_dfs: list[pd.DataFrame] = []
        source_subtitles: list[str] = []
        for file_position, uploaded_file in enumerate(uploaded_files, start=1):
            file_name = uploaded_file.name or ""

            # Leemos cada CSV de forma segura.
            original_df, delimiter_used, encoding_used = load_csv(uploaded_file)
            original_dfs.append(original_df)

            if file_name in seen_names:
                raise ValueError(
                    "No se permite subir varios CSV con el mismo nombre exacto. "
                    f"Nombre repetido: '{file_name}'."
                )

            seen_names.add(file_name)

            project_id = get_project_id_from_filename(uploaded_file.name)
            try:
                validate_project_id(project_id)
            except ValueError as exc:
                detail = get_project_id_validation_detail(uploaded_file.name)
                raise ValueError(
                    f"Error en '{uploaded_file.name}': {exc} Detalle del nombre recibido: {detail}"
                ) from exc

            # Aplicamos plantilla de transformación por archivo y lo acumulamos.
            transformed_df = transform_dataframe(original_df, project_id, df_materiales, map_tiradores)
            transformed_df = transform_apertura(transformed_df, map_aperturas, col="Apertura")
            transformed_df = add_source_metadata_columns(
                transformed_df,
                file_position,
                apply_name_prefix=len(uploaded_files) > 1,
            )
            transformed_dfs.append(transformed_df)
            source_subtitles.append(get_project_subtitle_from_filename(uploaded_file.name))

        final_df = pd.concat(transformed_dfs, ignore_index=True)
        final_df = fit_dimensions_to_standards(final_df, df_dimensiones, tolerance=1.75)
        final_df = recalculate_r_bars(final_df)
        final_df = apply_lac_cor_mecanizado(final_df)
        final_df = reorder_result_columns(final_df)
        final_df = sort_result_rows(final_df)

        st.markdown(
            "<h3 style='font-weight:800; margin: 0;'>Despiece para Pre-producción (Observaciones editables)</h3>",
            unsafe_allow_html=True,
        )
        st.caption("La tabla incluye subtítulos por archivo con línea de sección y solo permite editar Observaciones.")

        if "final_df_base" not in st.session_state:
            st.session_state["final_df_base"] = final_df.copy()
            st.session_state["final_df_confirmed"] = final_df.copy()
            st.session_state.pop("final_df_candidate", None)
            st.session_state["name_review_round"] = 0
            st.session_state["name_fixes"] = {}

        if "name_review_round" not in st.session_state:
            st.session_state["name_review_round"] = 0
        if "name_fixes" not in st.session_state:
            st.session_state["name_fixes"] = {}

        confirmed_df = st.session_state.get("final_df_confirmed", final_df).copy()

        has_u_shape = detect_u_shape(confirmed_df)
        tl_candidates_mask = get_tl_candidates_mask(confirmed_df)
        row_keys = build_u22_row_keys(confirmed_df)

        if has_u_shape:
            if "u22_mode" not in st.session_state:
                st.session_state["u22_mode"] = "pending"
            if "u22_selected_keys" not in st.session_state:
                st.session_state["u22_selected_keys"] = []

            st.markdown("### Marcado rápido 22mm (U-Shape detectado)")
            quick_col_all, quick_col_none, quick_col_manual = st.columns(3)

            if quick_col_all.button("Marcar TODO T/L como 22mm", use_container_width=True):
                st.session_state["u22_mode"] = "all"
                st.session_state["u22_selected_keys"] = []
                st.rerun()

            if quick_col_none.button("Marcar NINGUNO como 22mm", use_container_width=True):
                st.session_state["u22_mode"] = "none"
                st.session_state["u22_selected_keys"] = []
                st.rerun()

            if quick_col_manual.button("Revisar manual", use_container_width=True):
                st.session_state["u22_mode"] = "manual"

            if st.session_state.get("u22_mode") == "manual":
                candidate_rows = confirmed_df.loc[tl_candidates_mask].copy()
                candidate_rows["__u22_key"] = row_keys.loc[tl_candidates_mask].values

                previous_selected = set(st.session_state.get("u22_selected_keys", []))
                candidate_rows["22mm"] = candidate_rows["__u22_key"].isin(previous_selected)

                name_column = find_column_name(candidate_rows.columns, "Name")
                tipologia_column = find_column_name(candidate_rows.columns, "Tipología") or find_column_name(
                    candidate_rows.columns, "Tipologia"
                )

                visible_columns = [col for col in [name_column, tipologia_column, "Orden CSV", "Observaciones"] if col in candidate_rows.columns]
                manual_editor_columns = ["22mm"] + visible_columns

                st.caption("Selecciona qué piezas T/L deben marcarse como 22mm y confirma.")
                manual_edited = st.data_editor(
                    candidate_rows[manual_editor_columns + ["__u22_key"]],
                    width="stretch",
                    hide_index=False,
                    num_rows="fixed",
                    disabled=[col for col in manual_editor_columns if col != "22mm"] + ["__u22_key"],
                    column_config={
                        "22mm": st.column_config.CheckboxColumn(
                            "22mm",
                            help="Marca las filas que deben añadir 22mm en Observaciones.",
                        ),
                    },
                    key="u22_manual_editor",
                )

                if st.button("Confirmar 22mm", type="primary"):
                    selected_keys = manual_edited.loc[manual_edited["22mm"], "__u22_key"].astype(str).tolist()
                    st.session_state["u22_selected_keys"] = selected_keys
                    st.session_state["u22_mode"] = "manual_confirmed"
                    st.rerun()

        u22_mode = st.session_state.get("u22_mode", "none") if has_u_shape else "none"
        mask_22mm = pd.Series(False, index=confirmed_df.index)

        if has_u_shape and u22_mode == "all":
            mask_22mm = tl_candidates_mask
        elif has_u_shape and u22_mode == "manual_confirmed":
            selected_keys = set(st.session_state.get("u22_selected_keys", []))
            mask_22mm = row_keys.isin(selected_keys) & tl_candidates_mask

        final_with_22mm = confirmed_df.copy()
        if has_u_shape and u22_mode not in {"none", "pending", "manual"}:
            final_with_22mm = append_22mm_to_observaciones(final_with_22mm, mask_22mm)

        display_df = render_sectioned_observaciones_editor(
            final_with_22mm,
            source_subtitles,
            "resultado_observaciones_editor",
        )
        st.session_state["final_df_confirmed"] = display_df.copy()
        confirmed_df = st.session_state["final_df_confirmed"]

        candidate_df = st.session_state.get("final_df_candidate")
        review_df = candidate_df if candidate_df is not None else confirmed_df
        issues_df = detect_name_issues(review_df)
        confirmed_issues_df = detect_name_issues(confirmed_df)

        can_download = (not has_u_shape) or (u22_mode not in {"pending", "manual"})

        if confirmed_issues_df.empty:
            st.success("Names OK. Puedes descargar el CSV.")
            csv_output = confirmed_df.to_csv(index=False).encode("utf-8-sig")
            skirting_alerts_data = detect_skirting_shortage_by_source(confirmed_df)
            for alert in skirting_alerts_data:
                order = str(alert.get("order_value", "")).strip()
                lenz = float(alert.get("lenz_sum", 0.0))
                leny = float(alert.get("leny_sum", 0.0))
                subtitulo = f"Bloque {order}"
                if order.isdigit():
                    idx = int(order) - 1
                    if 0 <= idx < len(source_subtitles):
                        subtitulo = source_subtitles[idx]

                lenz_mm = int(round(lenz))
                leny_mm = int(round(leny))
                st.warning(
                    "Cuidado: Es posible que haya menos rodapiés de los necesarios en el despiece "
                    f"(REFERENCIA: {subtitulo} | Rodapiés: {lenz_mm} mm | Longitud de módulos: {leny_mm} mm)."
                )
            render_open_cabinet_generator_section()
            if can_download:
                st.download_button(
                    label="Descargar despiece",
                    data=BytesIO(csv_output),
                    file_name="resultado_transformado.csv",
                    mime="text/csv",
                )
            else:
                st.warning("Debes elegir cómo aplicar 22mm para piezas T/L antes de poder descargar.")
            st.session_state.pop("final_df_candidate", None)
            st.session_state.pop("post_issues_df", None)
        else:
            st.error(
                f"Hay {issues_df.shape[0]} piezas con Name incorrecto o duplicado. Corrige antes de descargar."
            )

            current_fixes = st.session_state.get("name_fixes", {})
            issues_to_edit = issues_df.copy()
            is_multi_csv = len(uploaded_files) > 1

            def split_name_prefix(raw_name: str) -> tuple[str, str]:
                name_value = "" if pd.isna(raw_name) else str(raw_name).strip()
                match = re.match(r"^([A-Za-z]-)(.*)$", name_value)
                if not match:
                    return "", name_value
                return match.group(1), match.group(2)

            for edit_index, issue_row in issues_to_edit.iterrows():
                row_index = int(issue_row["fila"])
                if row_index in current_fixes:
                    issues_to_edit.at[edit_index, "Nuevo Name"] = current_fixes[row_index]

            if is_multi_csv:
                issues_to_edit["Prefijo"] = issues_to_edit["Name actual"].apply(
                    lambda value: split_name_prefix(value)[0]
                )
                issues_to_edit["Nuevo Name"] = issues_to_edit["Nuevo Name"].apply(
                    lambda value: split_name_prefix(value)[1]
                )

            current_round = int(st.session_state.get("name_review_round", 0))
            editor_key = f"name_review_editor_{current_round}"
            if is_multi_csv:
                st.caption("No añadas el prefijo en la corrección del nombre de la pieza. Se añade automáticamente")
            edited_issues = st.data_editor(
                issues_to_edit,
                width="stretch",
                hide_index=True,
                num_rows="fixed",
                disabled=["fila", "Tipologia", "Name actual", "Motivo del error", "Prefijo"]
                if is_multi_csv
                else ["fila", "Tipologia", "Name actual", "Motivo del error"],
                column_config={
                    "Prefijo": st.column_config.TextColumn(
                        "Prefijo",
                        help="Prefijo fijo según el CSV de origen.",
                    ),
                    "Nuevo Name": st.column_config.TextColumn(
                        "Nuevo Name",
                        help="Edita solo este campo para proponer correcciones.",
                    )
                },
                key=editor_key,
            )

            if st.button("Aplicar correcciones", type="primary"):
                updated_fixes: dict[int, str] = {}
                for _, row in edited_issues.iterrows():
                    row_index = int(row["fila"])
                    core_value = "" if pd.isna(row["Nuevo Name"]) else str(row["Nuevo Name"]).strip()
                    if is_multi_csv:
                        prefix_value = ""
                        if "Prefijo" in row:
                            prefix_value = "" if pd.isna(row["Prefijo"]) else str(row["Prefijo"]).strip()
                        updated_fixes[row_index] = f"{prefix_value}{core_value}" if prefix_value else core_value
                    else:
                        updated_fixes[row_index] = core_value

                candidate_after_apply = sort_result_rows(apply_name_fixes(confirmed_df, updated_fixes))
                post_issues = detect_name_issues(candidate_after_apply)

                invalid_rows = set()
                if not post_issues.empty:
                    invalid_rows = set(post_issues["fila"].astype(int).tolist())

                valid_fixes = {row: val for row, val in updated_fixes.items() if row not in invalid_rows}
                pending_fixes = {row: val for row, val in updated_fixes.items() if row in invalid_rows}

                confirmed_updated = confirmed_df
                if valid_fixes:
                    confirmed_updated = sort_result_rows(apply_name_fixes(confirmed_df, valid_fixes))
                st.session_state["final_df_confirmed"] = confirmed_updated

                st.session_state["name_fixes"] = pending_fixes
                st.session_state["name_review_round"] = current_round + 1

                if not pending_fixes:
                    st.session_state.pop("final_df_candidate", None)
                    st.session_state.pop("post_issues_df", None)
                else:
                    candidate_pending = sort_result_rows(apply_name_fixes(confirmed_updated, pending_fixes))
                    st.session_state["final_df_candidate"] = candidate_pending
                    st.session_state["post_issues_df"] = detect_name_issues(candidate_pending)

                st.session_state.pop(editor_key, None)
                st.rerun()

    except ValueError as error_message:
        st.error(f"❌ {error_message}")
    except Exception as unexpected_error:
        st.error(
            "❌ Ocurrió un error inesperado al procesar el archivo. "
            "Revisa que sea un CSV válido e inténtalo de nuevo."
        )
        st.exception(unexpected_error)
