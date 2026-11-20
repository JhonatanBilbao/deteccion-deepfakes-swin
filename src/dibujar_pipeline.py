"""Dibuja docs/pipeline.png con un rostro real de FaceForensics++.

Uso: python dibujar_pipeline.py RUTA_VIDEO RUTA_FACE_LANDMARKER_TASK [SALIDA]
"""
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DF_BACKEND", "numpy")
from augment import bilateral_diff_map  # noqa: E402
from data import FaceAligner  # noqa: E402

VIDEO, TASK = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "../docs/pipeline.png"

# Rostro del dataset
cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 3)
ok, frame_bgr = cap.read()
cap.release()
assert ok, f"No pude leer {VIDEO}"
aligner = FaceAligner(224, task_model_path=TASK)
face, info = aligner.align(frame_bgr)
assert face is not None, "MediaPipe no encontró un rostro"
frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
pts = info["landmarks"]

# Recorte cuadrado del fotograma alrededor del rostro (sin alinear)
x0, y0 = pts.min(0)
x1, y1 = pts.max(0)
cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
half = 0.75 * max(x1 - x0, y1 - y0)
a, b = int(max(cx - half, 0)), int(max(cy - half, 0))
c, d = int(min(cx + half, frame.shape[1])), int(min(cy + half, frame.shape[0]))
crop = frame[b:d, a:c]
crop_pts = pts - np.array([a, b])

# Mapa de calor de la rama lateral (residuos de textura, como en el entrenamiento)
f = face.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
heat = np.asarray(bilateral_diff_map(f, mode="residual"))[0].mean(0)
heat = cv2.GaussianBlur(heat, (0, 0), 2.0)
heat = np.clip(heat / np.percentile(heat, 99.5), 0, 1)

# Lienzo en coordenadas de píxel (896 x 1200), eje y hacia abajo
W, H = 896, 1262
DY = 0  # desplazamiento vertical de las etapas 2 y 3
fig = plt.figure(figsize=(W / 100, H / 100))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.set_aspect("equal")
ax.axis("off")
LINE = "#2b2b2b"


def panel(x, y, w, h, title, fc, ec, fs=13.5, ty=26, ls="-"):
    y += DY
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=16", fc=fc, ec=ec, lw=1.6, ls=ls))
    ax.text(x + w / 2, y + ty, title, ha="center", va="center", fontsize=fs, weight="bold", color="#111111")


def box(x, y, w, h, text, fs=11.2):
    y += DY
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10", fc="#f3f3f1",
                                ec="#5a5a5a", lw=1.3, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#111111",
            linespacing=1.15, zorder=4)
    return x, y - DY, w, h


def text(x, y, s, fs=11.2, va="top"):
    ax.text(x, y + DY, s, ha="center", va=va, fontsize=fs, color="#111111", linespacing=1.15, zorder=4)


def arrow(p, q, rad=0.0):
    p, q = (p[0], p[1] + DY), (q[0], q[1] + DY)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=15, lw=2.0, color=LINE,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=5))


def route(pts_, r=14):
    """Polilínea con esquinas redondeadas y flecha en el último tramo."""
    p = [np.array((q[0], q[1] + DY), float) for q in pts_]
    xs, ys = [p[0][0]], [p[0][1]]
    for k in range(1, len(p) - 1):
        u = (p[k] - p[k - 1]) / np.linalg.norm(p[k] - p[k - 1])
        v = (p[k + 1] - p[k]) / np.linalg.norm(p[k + 1] - p[k])
        s, e = p[k] - u * r, p[k] + v * r
        xs.append(s[0]); ys.append(s[1])
        for t in np.linspace(0, 1, 8)[1:]:
            q = (1 - t) ** 2 * s + 2 * (1 - t) * t * p[k] + t ** 2 * e
            xs.append(q[0]); ys.append(q[1])
    end = p[-1] - (p[-1] - p[-2]) / np.linalg.norm(p[-1] - p[-2]) * 12
    xs.append(end[0]); ys.append(end[1])
    ax.plot(xs, ys, color=LINE, lw=2.0, zorder=5, solid_capstyle="round")
    arrow((end[0], end[1] - DY), (p[-1][0], p[-1][1] - DY))


def image(im, x, y, w, h, cmap=None):
    y += DY
    ax.imshow(im, extent=(x, x + w, y + h, y), cmap=cmap, interpolation="lanczos", zorder=3)


R = lambda bx: (bx[0] + bx[2], bx[1] + bx[3] / 2)
L = lambda bx: (bx[0], bx[1] + bx[3] / 2)
T = lambda bx: (bx[0] + bx[2] / 2, bx[1])
B = lambda bx: (bx[0] + bx[2] / 2, bx[1] + bx[3])

# Etapa 1
panel(3, 3, 890, 268, "Etapa 1: Preprocesamiento y Extracción por Frame", "#dce8f7", "#7fa5d6")
iy, isz = 86, 104
# claqueta
vx = 55
ax.add_patch(FancyBboxPatch((vx, iy + 26), 98, 70, boxstyle="round,pad=0,rounding_size=6", fc="white",
                            ec="black", lw=3.2, zorder=3))
ax.add_patch(Polygon([(vx + 40, iy + 46), (vx + 40, iy + 76), (vx + 64, iy + 61)], fc="black", zorder=4))
ax.add_patch(Polygon([(vx - 2, iy + 22), (vx + 100, iy + 22), (vx + 100, iy + 6), (vx - 2, iy + 6)],
                     fc="black", zorder=3))
for k in range(5):
    ax.add_patch(Polygon([(vx + 8 + 20 * k, iy + 21), (vx + 18 + 20 * k, iy + 7), (vx + 26 + 20 * k, iy + 7),
                          (vx + 16 + 20 * k, iy + 21)], fc="white", zorder=4))
ax.add_patch(Polygon([(vx, iy + 2), (vx + 96, iy - 16), (vx + 98, iy - 6), (vx + 2, iy + 12)], fc="black",
                     zorder=3, transform=ax.transData))
text(vx + 49, 202, "Entrada de Video", va="top")
# fotogramas apilados
fx = 218
frame_small = cv2.resize(frame, (160, 120))
for k in range(4, 0, -1):
    ax.add_patch(Rectangle((fx + 6 * k, iy - 6 * k + 6), 112, 84, fc="#555555", ec="#222222", lw=1, zorder=2))
image(frame_small, fx, iy + 6, 112, 84)
ax.add_patch(Rectangle((fx, iy + 6), 112, 84, fill=False, ec="#222222", lw=1, zorder=4))
text(fx + 62, 202, "Muestreo Uniforme\n(N=16 frames)")
# landmarks
lx = 402
image(crop, lx, iy, isz, isz)
sel = crop_pts[::6] * (isz / crop.shape[1])
ax.scatter(lx + sel[:, 0], iy + sel[:, 1], s=3, c="white", edgecolors="none", zorder=5)
text(lx + isz / 2, 202, "Landmarks\n(MediaPipe +\nrespaldo Haar)")
# alineación
ax_ = 567
image(face, ax_, iy, isz, isz)
ax.plot([ax_ + isz / 2] * 2, [iy - 14, iy + isz + 14], color="black", lw=1.4, zorder=5)
text(ax_ + isz / 2, 202, "Alineación Facial\n(Eje Central,\nescala relativa)")
# rostro normalizado
nx = 740
image(face, nx, iy, isz, isz)
text(nx + isz / 2, 202, "Rostro Normalizado\n(224x224)\n+ índice de giro")
for p, q in ((vx + 108, fx - 4), (fx + 140, lx - 4), (lx + isz + 8, ax_ - 4), (ax_ + isz + 8, nx - 4)):
    arrow((p, iy + isz / 2), (q, iy + isz / 2))

# Entrenamiento (entre etapas)
ax.add_patch(FancyBboxPatch((150, 285), 600, 52, boxstyle="round,pad=0,rounding_size=10", fc="#fbfaf6",
                            ec="#777777", lw=1.3, ls="--", zorder=6))
ax.text(450, 311, "Solo en entrenamiento: aumento de calidad (baja resolución, desenfoque, ruido, brillo)\n"
                  "y normalización robusta con mediana / IQR calculados solo sobre train",
        ha="center", va="center", fontsize=10, color="#111111", linespacing=1.3, zorder=7)
DY = 62

# Etapa 2
panel(3, 318, 890, 564, "Etapa 2: Extracción en Dos Ramas y Fusión Balanceada", "#ede4d1", "#c7ab82", ty=27)
panel(28, 372, 677, 156, "Rama Global (Swin Transformer)", "#d8ebd4", "#8bb886", fs=12.5, ty=25)
gy, gh = 422, 90
g = [box(45, gy, 115, gh, "Rostro\nCompleto\n(224x224)"),
     box(205, gy, 170, gh, "Swin Transformer\nBackbone (Swin-T,\npesos ImageNet-1k)", fs=10.6),
     box(415, gy, 92, gh, "Vector\nGlobal\n(768d)"),
     box(560, gy, 125, gh, "Proyección\n(256d)")]
for p, q in zip(g, g[1:]):
    arrow(R(p), L(q))

panel(28, 550, 677, 307, "Rama Lateral (Asimetría Diferencial)", "#fde3c6", "#d9a64e", fs=12.5, ty=25)
rt = box(45, 596, 107, 70, "Residuos de\nTextura\n(x − blur σ=1)", fs=10.4)
rd = box(45, 690, 107, 60, "Rostro\nDividido")
arrow(B(rt), T(rd))
mi = box(192, 600, 94, 62, "Mitad\nIzquierda")
md = box(192, 747, 94, 64, "Mitad\nDerecha")
text(239, 668, "(224x112)\nnorma RMS", fs=10.4)
text(239, 813, "(224x112)\nnorma RMS", fs=10.2)
fl = box(320, 596, 107, 70, "Inversión\nEspejo\n(Flip)")
route([(152, 720), (170, 720), (170, 631), (192, 631)], r=12)
route([(152, 720), (170, 720), (170, 779), (192, 779)], r=12)
arrow(R(mi), L(fl))
mx, my = 322, 719
ax.add_patch(Circle((mx, my + DY), 14, fc="white", ec=LINE, lw=1.4, zorder=6))
ax.text(mx, my + DY, "(-)", ha="center", va="center", fontsize=9.5, zorder=7)
route([(373, 666), (373, 684), (mx, 684), (mx, my - 14)], r=10)
route([(286, 779), (mx, 779), (mx, my + 14)], r=10)
hx, hy, hw, hh = 386, 679, 40, 80
image(heat, hx, hy, hw, hh, cmap="jet")
ax.add_patch(Rectangle((hx, hy + DY), hw, hh, fill=False, ec="#333333", lw=0.8, zorder=4))
arrow((mx + 14, my), (hx, my))
text(hx + hw / 2, 764, "Mapa de\nDiferencias\n|flip(izq) − der|", fs=10.4)
cn = box(452, 682, 113, 74, "Codificador\nCNN\n32→64→128", fs=10.6)
pj = box(585, 688, 105, 69, "Proyección\n(256d)")
arrow((hx + hw, my), (458, my))
arrow(R(cn), L(pj))

fu = box(730, 557, 140, 88, "Fusión\nBalanceada\n(Concatenación)")
de = box(731, 746, 129, 89, "Descriptor\nHíbrido\n(512d)")
route([R(g[3]), (795, R(g[3])[1]), (795, 557)])
route([R(pj), (745, R(pj)[1]), (745, 645)])
arrow((795, 645), (795, 746))

route([(nx + isz / 2, 271 - DY), (nx + isz / 2, 352 - DY), (20, 352 - DY), (20, 631), (45, 631)])
route([(100, 352 - DY), (100, 422)])

# Etapa 3
panel(3, 942, 890, 255, "Etapa 3: Clasificación y Agregación Temporal", "#e8def0", "#a78bc4", ty=27)
cy3 = 1074
d3 = box(33, 1031, 126, 87, "Descriptor\nHíbrido\n(512d)")
route([(795, 835), (795, 910), (97, 910), (97, 1031)])
layers = [(222, [1022, 1048, 1074, 1100, 1126]), (272, [1009, 1035, 1061, 1087, 1113, 1139]),
          (326, [1048, 1100])]
for (xa, ya), (xb, yb) in zip(layers, layers[1:]):
    for p in ya:
        for q in yb:
            ax.plot([xa, xb], [p + DY, q + DY], color="#222222", lw=0.9, zorder=3)
for xa, ya in layers:
    for p in ya:
        ax.add_patch(Circle((xa, p + DY), 10, fc="white", ec="#111111", lw=1.4, zorder=4))
arrow(R(d3), (205, cy3))
text(275, 1153, "MLP 512→256→2\n(Focal Loss α=0.5, γ=2)", fs=10.4)
sc = box(386, 1031, 111, 87, "Score por\nFrame\n(p$_i$)")
arrow((343, cy3), L(sc))
ag = box(523, 1031, 168, 87, "Agregación\nTemporal\n(Promedio N=16)")
arrow(R(sc), L(ag))
ax.add_patch(Rectangle((727, 1018 + DY), 138, 24, fc="#4c9a3a", ec="#2f6b24", lw=1, zorder=3))
ax.add_patch(Rectangle((727, 1106 + DY), 138, 24, fc="#d10f0f", ec="#8f0a0a", lw=1, zorder=3))
text(796, 1003, "Video Real", va="center")
text(796, 1090, "Video Deepfake", va="center")
route([R(ag), (706, cy3), (706, 1030), (727, 1030)], r=12)
route([R(ag), (706, cy3), (706, 1118), (727, 1118)], r=12)
text(796, 1139, "Decisión Final del Video\n(umbral fijado en validación)", fs=10)

fig.savefig(OUT, dpi=300, facecolor="white")
print("guardado", OUT)
