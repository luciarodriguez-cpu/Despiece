# Línea 1
from __future__ import annotations

# Línea 3
from dataclasses import dataclass
import re
from uuid import uuid4


# Línea 6
@dataclass
class MuebleAbiertoInput:
    ancho_mm: float
    alto_mm: float
    fondo_mm: float
    num_baldas: int
    colgado: bool
    zocalo_mm: float
    espesor_mm: float = 19.0
    color_hex: str = "#FFFFFF"


# Línea 16
def generar_svg_mueble_abierto(
    ancho_mm: float,
    alto_mm: float,
    fondo_mm: float,
    num_baldas: int,
    colgado: bool,
    zocalo_mm: float,
    espesor_mm: float = 19.0,
    color_hex: str = "#FFFFFF",
) -> str:
    """
    Genera el SVG paramétrico de un mueble abierto siguiendo las reglas
    geométricas acordadas en la conversación.

    Reglas implementadas:
    - vista ligeramente frontal
    - solo líneas visibles
    - baldas equidistantes entre tapa y base
    - base apoyada en suelo si zócalo=0
    - si zócalo>0, base elevada y aparece pieza de zócalo
    - agujeros de colgar referidos a la pieza trasera real
    - arista trasera derecha del lateral izquierdo cortada correctamente
    """

    # Línea 35
    _validar_inputs(
        ancho_mm=ancho_mm,
        alto_mm=alto_mm,
        fondo_mm=fondo_mm,
        num_baldas=num_baldas,
        zocalo_mm=zocalo_mm,
        espesor_mm=espesor_mm,
    )

    color_relleno = _normalizar_hex(color_hex)
    color_linea = _color_contraste(color_relleno)
    uid = uuid4().hex[:8]
    clase_relleno = f"f_{uid}"
    clase_linea = f"s_{uid}"
    clase_agujero = f"h_{uid}"

    # =========================================================
    # Línea 45
    # PARÁMETROS DE PROYECCIÓN
    # =========================================================

    # Línea 49
    x0 = 170.0
    y0 = 110.0

    # Línea 52
    px_por_mm_x = 0.50
    px_por_mm_y = 0.525

    # Línea 55
    # Vista algo más frontal que isométrica
    fondo_dx_por_mm = 0.152
    fondo_dy_por_mm = 0.061

    # Línea 59
    ancho_px = ancho_mm * px_por_mm_x
    alto_px = alto_mm * px_por_mm_y
    dx_fondo = fondo_mm * fondo_dx_por_mm
    dy_fondo = fondo_mm * fondo_dy_por_mm

    # Línea 64
    espesor_px_y = max(10.0, espesor_mm * px_por_mm_y)
    espesor_px_x = max(10.0, espesor_mm * px_por_mm_x)

    # Línea 67
    # Retranqueo zócalo = 100 mm siguiendo la línea de perspectiva
    zocalo_retranqueo_mm = 100.0
    zocalo_dx = zocalo_retranqueo_mm * fondo_dx_por_mm
    zocalo_dy = zocalo_retranqueo_mm * fondo_dy_por_mm

    # =========================================================
    # Línea 74
    # PUNTOS EXTERIORES
    # =========================================================

    # Línea 78
    x_front_left = x0
    y_front_top = y0

    # Línea 81
    x_front_right = x_front_left + ancho_px
    y_front_right = y_front_top

    # Línea 84
    x_back_left = x_front_left + dx_fondo
    y_back_left = y_front_top - dy_fondo

    # Línea 87
    x_back_right = x_front_right + dx_fondo
    y_back_right = y_front_top - dy_fondo

    # Línea 90
    y_suelo = y_front_top + alto_px

    # =========================================================
    # Línea 94
    # CARAS INTERIORES
    # =========================================================

    # Línea 98
    x_inner_left_front = x_front_left + espesor_px_x
    x_inner_right_front = x_front_right - espesor_px_x
    x_inner_back_left = x_back_left + espesor_px_x

    # Línea 102
    # Lateral derecho visto al fondo
    x_right_side_outer_back = x_back_right

    # =========================================================
    # Línea 106
    # TAPA
    # =========================================================

    # Línea 110
    y_tapa_top_front = y_front_top
    y_tapa_bottom_front = y_tapa_top_front + espesor_px_y

    # Línea 113
    y_tapa_top_back = y_back_left
    y_tapa_bottom_back = y_tapa_top_back + espesor_px_y

    # =========================================================
    # Línea 117
    # BASE
    # =========================================================

    # Línea 121
    subida_base_px = zocalo_mm * px_por_mm_y if zocalo_mm > 0 else 0.0

    # Línea 124
    y_base_bottom_front = y_suelo - subida_base_px
    y_base_top_front = y_base_bottom_front - espesor_px_y

    # Línea 127
    y_base_bottom_back = y_base_bottom_front - dy_fondo
    y_base_top_back = y_base_top_front - dy_fondo

    # =========================================================
    # Línea 131
    # BALDAS EQUIDISTANTES ENTRE TAPA Y BASE
    # =========================================================

    # Línea 135
    hueco_libre_front = y_base_top_front - y_tapa_bottom_front
    separacion_front = hueco_libre_front / (num_baldas + 1) if num_baldas > 0 else 0.0

    # Línea 138
    baldas: list[dict[str, float]] = []

    # Línea 140
    for i in range(num_baldas):
        y_sup_front = y_tapa_bottom_front + separacion_front * (i + 1)
        y_inf_front = y_sup_front + espesor_px_y

        y_sup_back = y_sup_front - dy_fondo
        y_inf_back = y_inf_front - dy_fondo

        baldas.append(
            {
                "y_sup_front": y_sup_front,
                "y_inf_front": y_inf_front,
                "y_sup_back": y_sup_back,
                "y_inf_back": y_inf_back,
            }
        )

    # =========================================================
    # Línea 155
    # PIEZA TRASERA REAL
    # =========================================================

    # Línea 159
    x_trasera_left = x_inner_back_left
    x_trasera_right = x_back_right - espesor_px_x
    y_trasera_top = y_tapa_bottom_back
    y_trasera_bottom = y_base_top_back

    # =========================================================
    # Línea 165
    # ZÓCALO
    # =========================================================

    # Línea 169
    hay_zocalo = zocalo_mm > 0

    # Línea 171
    zocalo = None
    if hay_zocalo:
        alto_zocalo_mm = max(0.0, zocalo_mm - 5.0)
        alto_zocalo_px = alto_zocalo_mm * px_por_mm_y

        # Línea 176
        # Arranca desde la esquina inferior derecha del lateral izquierdo
        # y se retranquea 100 mm hacia atrás siguiendo la línea de perspectiva.
        x_zoc_left = x_inner_left_front + zocalo_dx
        y_zoc_left_bottom = y_suelo - zocalo_dy

        # Línea 181
        # Su altura es zócalo_mm - 5
        y_zoc_left_top = y_zoc_left_bottom - alto_zocalo_px

        # Línea 184
        # Ancho: desde arista derecha frontal lateral izquierdo
        # hasta arista izquierda frontal lateral derecho
        # pero con el mismo retranqueo visual
        x_zoc_right = x_inner_right_front + zocalo_dx
        y_zoc_right_bottom = y_suelo - zocalo_dy

        # Línea 189
        # La arista derecha del zócalo debe morir en la arista izquierda del lateral derecho
        # visualmente, por eso su parte superior se corta en esa zona.
        y_zoc_right_top = y_base_bottom_front - espesor_px_y + 5.0

        # Línea 193
        # La arista izquierda del zócalo debe morir en la arista frontal inferior de la base
        y_zoc_left_top_visible = y_base_bottom_front

        zocalo = {
            "x_left": x_zoc_left,
            "y_left_bottom": y_zoc_left_bottom,
            "y_left_top_visible": y_zoc_left_top_visible,
            "x_right": x_inner_right_front,
            "y_right_bottom": y_suelo,
            "y_right_top_visible": y_zoc_right_top,
            "y_top_real_left": y_zoc_left_top,
        }

    # =========================================================
    # Línea 207
    # SALIDA SVG
    # =========================================================

    # Línea 211
    rellenos: list[str] = []
    lineas: list[str] = []
    agujeros: list[str] = []

    # Línea 213
    def add_line(x1: float, y1: float, x2: float, y2: float, clase: str | None = None) -> None:
        if clase is None:
            clase = clase_linea
        lineas.append(
            f'<line class="{clase}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>'
        )

    # Línea 218
    def add_polygon(puntos: list[tuple[float, float]], clase: str | None = None) -> None:
        if clase is None:
            clase = clase_relleno
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in puntos)
        rellenos.append(f'<polygon class="{clase}" points="{p}"/>')

    # Línea 221
    def add_ellipse(cx: float, cy: float, rx: float, ry: float, clase: str | None = None) -> None:
        if clase is None:
            clase = clase_linea
        agujeros.append(
            f'<ellipse class="{clase}" cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}"/>'
        )

    # =========================================================
    # RELLENOS
    # =========================================================

    # Tapa (cara superior)
    add_polygon(
        [
            (x_front_left, y_tapa_top_front),
            (x_front_right, y_tapa_top_front),
            (x_back_right, y_tapa_top_back),
            (x_back_left, y_tapa_top_back),
        ]
    )

    # Tapa (caras de espesor visibles)
    add_polygon(
        [
            (x_front_left, y_tapa_top_front),
            (x_inner_left_front, y_tapa_top_front),
            (x_inner_left_front, y_tapa_bottom_front),
            (x_front_left, y_tapa_bottom_front),
        ]
    )
    add_polygon(
        [
            (x_inner_left_front, y_tapa_top_front),
            (x_back_left, y_tapa_top_back),
            (x_inner_back_left, y_tapa_top_back),
            (x_inner_left_front, y_tapa_bottom_front),
        ]
    )
    add_polygon(
        [
            (x_inner_right_front, y_tapa_top_front),
            (x_front_right, y_tapa_top_front),
            (x_front_right, y_tapa_bottom_front),
            (x_inner_right_front, y_tapa_bottom_front),
        ]
    )
    add_polygon(
        [
            (x_inner_right_front, y_tapa_top_front),
            (x_front_right, y_tapa_top_front),
            (x_back_right, y_tapa_top_back),
            (x_back_right - espesor_px_x, y_tapa_top_back),
        ]
    )

    # Laterales frontales
    add_polygon(
        [
            (x_front_left, y_tapa_top_front),
            (x_inner_left_front, y_tapa_top_front),
            (x_inner_left_front, y_suelo),
            (x_front_left, y_suelo),
        ]
    )
    add_polygon(
        [
            (x_inner_right_front, y_tapa_top_front),
            (x_front_right, y_tapa_top_front),
            (x_front_right, y_suelo),
            (x_inner_right_front, y_suelo),
        ]
    )

    # Lateral derecho exterior (cara lateral en perspectiva)
    add_polygon(
        [
            (x_front_right, y_tapa_top_front),
            (x_back_right, y_tapa_top_back),
            (x_right_side_outer_back, y_suelo - dy_fondo),
            (x_front_right, y_suelo),
        ]
    )

    # Trasera interior visible
    add_polygon(
        [
            (x_trasera_left, y_trasera_top),
            (x_trasera_right, y_trasera_top),
            (x_trasera_right, y_trasera_bottom),
            (x_trasera_left, y_trasera_bottom),
        ]
    )

    # Lateral izquierdo de la hornacina (cara interior)
    add_polygon(
        [
            (x_inner_left_front, y_tapa_bottom_front),
            (x_inner_back_left, y_tapa_bottom_back),
            (x_inner_back_left, y_base_top_back),
            (x_inner_left_front, y_base_top_front),
        ]
    )

    # Base (cara superior y frontal)
    add_polygon(
        [
            (x_inner_left_front, y_base_top_front),
            (x_inner_right_front, y_base_top_front),
            (x_inner_right_front, y_base_top_back),
            (x_inner_back_left, y_base_top_back),
        ]
    )
    add_polygon(
        [
            (x_inner_left_front, y_base_top_front),
            (x_inner_right_front, y_base_top_front),
            (x_inner_right_front, y_base_bottom_front),
            (x_inner_left_front, y_base_bottom_front),
        ]
    )

    # Baldas
    for balda in baldas:
        add_polygon(
            [
                (x_inner_left_front, balda["y_sup_front"]),
                (x_inner_right_front, balda["y_sup_front"]),
                (x_inner_right_front, balda["y_sup_back"]),
                (x_inner_back_left, balda["y_sup_back"]),
            ]
        )
        add_polygon(
            [
                (x_inner_left_front, balda["y_sup_front"]),
                (x_inner_right_front, balda["y_sup_front"]),
                (x_inner_right_front, balda["y_inf_front"]),
                (x_inner_left_front, balda["y_inf_front"]),
            ]
        )

    # Zócalo
    if zocalo is not None:
        add_polygon(
            [
                (zocalo["x_left"], zocalo["y_left_top_visible"]),
                (zocalo["x_left"], zocalo["y_left_bottom"]),
                (zocalo["x_right"], zocalo["y_right_bottom"]),
                (zocalo["x_right"], zocalo["y_right_top_visible"]),
            ]
        )

    # =========================================================
    # Línea 224
    # TAPA
    # =========================================================

    # Línea 228
    add_line(x_front_left, y_tapa_top_front, x_front_right, y_tapa_top_front)
    add_line(x_front_right, y_tapa_top_front, x_back_right, y_tapa_top_back)
    add_line(x_front_left, y_tapa_top_front, x_back_left, y_tapa_top_back)
    add_line(x_back_left, y_tapa_top_back, x_back_right, y_tapa_top_back)

    # Línea 234
    add_line(x_inner_left_front, y_tapa_bottom_front, x_inner_right_front, y_tapa_bottom_front)
    add_line(x_inner_left_front, y_tapa_top_front, x_inner_left_front, y_tapa_bottom_front)
    add_line(x_inner_right_front, y_tapa_top_front, x_inner_right_front, y_tapa_bottom_front)

    # =========================================================
    # Línea 240
    # LATERAL IZQUIERDO
    # =========================================================

    # Línea 244
    add_line(x_front_left, y_tapa_top_front, x_front_left, y_suelo)
    add_line(x_inner_left_front, y_tapa_top_front, x_inner_left_front, y_suelo)
    add_line(x_front_left, y_suelo, x_inner_left_front, y_suelo)

    # Línea 248
    add_line(x_front_left, y_tapa_top_front, x_inner_left_front, y_tapa_top_front)
    add_line(x_inner_left_front, y_tapa_top_front, x_inner_back_left, y_tapa_top_back)
    add_line(x_back_left, y_tapa_top_back, x_inner_back_left, y_tapa_top_back)

    # =========================================================
    # Línea 254
    # ARISTA TRASERA DERECHA DEL LATERAL IZQUIERDO
    # =========================================================

    # Línea 258
    # Tramo superior: justo por debajo de la tapa
    inicio = y_tapa_bottom_front + 1.0

    # Línea 261
    if num_baldas > 0:
        fin = baldas[0]["y_sup_back"]
        if fin > inicio:
            add_line(x_inner_back_left, inicio, x_inner_back_left, fin)

        # Línea 266
        for i in range(num_baldas - 1):
            inicio_i = baldas[i]["y_inf_front"] + 1.0
            fin_i = baldas[i + 1]["y_sup_back"]
            if fin_i > inicio_i:
                add_line(x_inner_back_left, inicio_i, x_inner_back_left, fin_i)

        # Línea 272
        inicio_last = baldas[-1]["y_inf_front"] + 1.0
        fin_last = y_base_top_back
        if fin_last > inicio_last:
            add_line(x_inner_back_left, inicio_last, x_inner_back_left, fin_last)

    # Línea 278
    if num_baldas == 0:
        fin = y_base_top_back
        if fin > inicio:
            add_line(x_inner_back_left, inicio, x_inner_back_left, fin)

    # =========================================================
    # Línea 284
    # LATERAL DERECHO
    # =========================================================

    # Línea 288
    add_line(x_front_right, y_tapa_top_front, x_front_right, y_suelo)
    add_line(x_inner_right_front, y_tapa_top_front, x_inner_right_front, y_suelo)
    add_line(x_inner_right_front, y_suelo, x_front_right, y_suelo)

    # Línea 292
    add_line(x_inner_right_front, y_tapa_top_front, x_front_right, y_tapa_top_front)
    add_line(x_inner_right_front, y_tapa_top_front, x_back_right - espesor_px_x, y_tapa_top_back)
    add_line(x_back_right - espesor_px_x, y_tapa_top_back, x_back_right, y_tapa_top_back)
    add_line(x_front_right, y_tapa_top_front, x_back_right, y_tapa_top_back)

    # Línea 298
    add_line(x_right_side_outer_back, y_tapa_top_back, x_right_side_outer_back, y_suelo - dy_fondo)
    add_line(x_front_right, y_suelo, x_right_side_outer_back, y_suelo - dy_fondo)

    # =========================================================
    # Línea 304
    # BALDAS
    # =========================================================

    # Línea 308
    for balda in baldas:
        add_line(x_inner_left_front, balda["y_sup_front"], x_inner_right_front, balda["y_sup_front"])
        add_line(x_inner_left_front, balda["y_sup_front"], x_inner_back_left, balda["y_sup_back"])
        add_line(x_inner_back_left, balda["y_sup_back"], x_inner_right_front, balda["y_sup_back"])

        add_line(x_inner_left_front, balda["y_inf_front"], x_inner_right_front, balda["y_inf_front"])
        add_line(x_inner_left_front, balda["y_sup_front"], x_inner_left_front, balda["y_inf_front"])
        add_line(x_inner_right_front, balda["y_sup_front"], x_inner_right_front, balda["y_inf_front"])

    # =========================================================
    # Línea 319
    # BASE
    # =========================================================

    # Línea 323
    add_line(x_inner_left_front, y_base_top_front, x_inner_right_front, y_base_top_front)
    add_line(x_inner_left_front, y_base_top_front, x_inner_back_left, y_base_top_back)
    add_line(x_inner_back_left, y_base_top_back, x_inner_right_front, y_base_top_back)

    # Línea 328
    add_line(x_inner_left_front, y_base_bottom_front, x_inner_right_front, y_base_bottom_front)
    add_line(x_inner_left_front, y_base_top_front, x_inner_left_front, y_base_bottom_front)
    add_line(x_inner_right_front, y_base_top_front, x_inner_right_front, y_base_bottom_front)

    # =========================================================
    # Línea 334
    # ZÓCALO
    # =========================================================

    # Línea 338
    if zocalo is not None:
        # Solo cara frontal visible
        add_line(
            zocalo["x_left"],
            zocalo["y_left_top_visible"],
            zocalo["x_left"],
            zocalo["y_left_bottom"],
        )
        add_line(
            zocalo["x_left"],
            zocalo["y_left_bottom"],
            zocalo["x_right"],
            zocalo["y_right_bottom"],
        )
        add_line(
            zocalo["x_right"],
            zocalo["y_right_bottom"],
            zocalo["x_right"],
            zocalo["y_right_top_visible"],
        )

    # =========================================================
    # Línea 354
    # AGUJEROS DE COLGAR
    # =========================================================

    # Línea 358
    if colgado:
        diametro_mm = 17.0
        radio_x = (diametro_mm * px_por_mm_x) / 2.0
        radio_y = (diametro_mm * px_por_mm_y) / 2.0

        offset_superior_mm = 75.0
        offset_lateral_mm = 16.5

        # Línea 365
        # Referidos SOLO a la pieza trasera real
        cx_izq = x_trasera_left + offset_lateral_mm * px_por_mm_x
        cy = y_trasera_top + offset_superior_mm * px_por_mm_y

        add_ellipse(cx_izq, cy, radio_x, radio_y, clase=clase_agujero)

        # Línea 371
        # El agujero derecho solo si visualmente se ve.
        # Con esta perspectiva normalmente queda oculto, así que no se dibuja.
        pass

    # =========================================================
    # Línea 377
    # SVG FINAL
    # =========================================================

    # Línea 381
    min_x = min(0.0, x_front_left - 110.0)
    min_y = min(0.0, y_back_left - 40.0)
    max_x = max(x_back_right + 120.0, x_front_right + 140.0)
    max_y = max(y_suelo + 110.0, y_base_bottom_front + 150.0)

    # Línea 386
    view_w = max_x - min_x
    view_h = max_y - min_y

    # Línea 389
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{min_x:.1f} {min_y:.1f} {view_w:.1f} {view_h:.1f}">',
        "<style>",
        f'.{clase_relleno}{{fill:{color_relleno};stroke:none;}}',
        f'.{clase_linea}{{stroke:{color_linea};stroke-width:2.2;fill:none;stroke-linecap:round;stroke-linejoin:round;}}',
        f'.{clase_agujero}{{fill:#FFFFFF;stroke:{color_linea};stroke-width:2.0;}}',
        "</style>",
        *rellenos,
        *lineas,
        *agujeros,
        "</svg>",
    ]

    # Línea 397
    return "\n".join(svg)


# Línea 400
def _validar_inputs(
    ancho_mm: float,
    alto_mm: float,
    fondo_mm: float,
    num_baldas: int,
    zocalo_mm: float,
    espesor_mm: float,
) -> None:
    if ancho_mm <= 0:
        raise ValueError("ancho_mm debe ser mayor que 0.")
    if alto_mm <= 0:
        raise ValueError("alto_mm debe ser mayor que 0.")
    if fondo_mm <= 0:
        raise ValueError("fondo_mm debe ser mayor que 0.")
    if num_baldas < 0:
        raise ValueError("num_baldas no puede ser negativo.")
    if zocalo_mm < 0:
        raise ValueError("zocalo_mm no puede ser negativo.")
    if espesor_mm <= 0:
        raise ValueError("espesor_mm debe ser mayor que 0.")


def _normalizar_hex(color_hex: str) -> str:
    valor = str(color_hex).strip()
    if re.fullmatch(r"#([0-9a-fA-F]{6})", valor):
        return valor.upper()
    return "#FFFFFF"


def _color_contraste(hex_color: str) -> str:
    color = _normalizar_hex(hex_color)
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    luminancia = (0.299 * r) + (0.587 * g) + (0.114 * b)
    return "#111111" if luminancia >= 150 else "#F5F5F5"
