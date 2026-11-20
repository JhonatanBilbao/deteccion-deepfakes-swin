"""Lectura de videos, extracción y alineación de rostros, cachés y particiones."""
from __future__ import annotations

import os
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")
LABELS = {"real": 0, "fake": 1}


@dataclass
class VideoItem:
    path: str
    label: int
    method: str
    video_id: str
    identities: Tuple[str, ...]
    group: int = -1
    split: str = ""


def parse_name(stem: str, label: int) -> Tuple[str, Tuple[str, ...]]:
    parts = stem.split("_")
    if label == 0 and stem.isdigit():
        return "real", (stem,)
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2]), (parts[-2], parts[-1])
    return ("real" if label == 0 else "fake"), (stem,)


def list_videos(root: str) -> List[VideoItem]:
    items: List[VideoItem] = []
    for name, lab in LABELS.items():
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(VIDEO_EXT):
                stem = os.path.splitext(f)[0]
                method, ids = parse_name(stem, lab)
                items.append(VideoItem(os.path.join(d, f), lab, method, stem, ids))
    if not items:
        raise FileNotFoundError(f"No hay videos en {root}/real ni {root}/fake")
    return items


FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")  # protocolo estándar de 4 métodos


def list_videos_ffpp(root: str, compression: str = "c23",
                     methods: Sequence[str] = FFPP_METHODS) -> List[VideoItem]:
    """Lee la estructura original de FF++: original_sequences/youtube/<comp>/videos y manipulated_sequences/<metodo>/<comp>/videos."""
    items: List[VideoItem] = []
    d = os.path.join(root, "original_sequences", "youtube", compression, "videos")
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(VIDEO_EXT):
                stem = os.path.splitext(f)[0]
                items.append(VideoItem(os.path.join(d, f), 0, "real", stem, (stem,)))
    for m in methods:
        d = os.path.join(root, "manipulated_sequences", m, compression, "videos")
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(VIDEO_EXT):
                stem = os.path.splitext(f)[0]
                t, s_ = stem.split("_")[:2]
                items.append(VideoItem(os.path.join(d, f), 1, m, f"{m}_{stem}", (t, s_)))
    if not items:
        raise FileNotFoundError(f"No hay videos FF++ {compression} en {root}")
    return items


def list_videos_celebdf(root: str, list_file: str = "List_of_testing_videos.txt") -> List[VideoItem]:
    """Lista oficial de test de Celeb-DF v2 (518 videos)."""
    items: List[VideoItem] = []
    with open(os.path.join(root, list_file)) as f:
        for line in f:
            if not line.strip():
                continue
            flag, rel = line.split()
            label = 0 if flag == "1" else 1
            folder, fname = rel.split("/")
            assert (label == 1) == ("synthesis" in folder.lower()), f"Etiqueta inconsistente: {line}"
            stem = os.path.splitext(fname)[0]
            parts = stem.split("_")
            if folder.lower().startswith("youtube"):
                ids = (f"cdf_yt{stem}",)
            elif label == 1:
                ids = (f"cdf_{parts[0]}", f"cdf_{parts[1]}")
            else:
                ids = (f"cdf_{parts[0]}",)
            items.append(VideoItem(os.path.join(root, folder, fname), label,
                                   "CelebDF" if label else "real", f"cdf_{folder}_{stem}", ids))
    missing = [v.path for v in items if not os.path.exists(v.path)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} videos de la lista de Celeb-DF no existen, p. ej. {missing[0]}")
    return items


def detect_layout(root: str) -> str:
    """'ffpp' (estructura oficial) o 'simple' (carpetas real/ y fake/)."""
    if os.path.isdir(os.path.join(root, "original_sequences")) and \
            os.path.isdir(os.path.join(root, "manipulated_sequences")):
        return "ffpp"
    if os.path.isdir(os.path.join(root, "real")) and os.path.isdir(os.path.join(root, "fake")):
        return "simple"
    raise FileNotFoundError(f"{root}: no tiene real/+fake/ ni original_sequences/+manipulated_sequences/")


def load_videos(root: str, compression: str = "c23") -> List[VideoItem]:
    return list_videos_ffpp(root, compression) if detect_layout(root) == "ffpp" else list_videos(root)


def split_official(videos: List[VideoItem]) -> Dict[str, List[VideoItem]]:
    """Partición oficial de FF++ (720/140/140 por video original)."""
    from ff_splits import official_split_of_id

    sp = official_split_of_id()
    for v in videos:
        s = {sp[i] for i in v.identities if i in sp}
        if len(s) != 1:
            raise ValueError(f"{v.video_id}: identidades {v.identities} fuera de la partición oficial o en "
                             f"conjuntos distintos ({s})")
        v.split = s.pop()
    assign_identity_groups(videos)
    return {n: [v for v in videos if v.split == n] for n in ("train", "val", "test")}


def find_dataset_dir(search_roots: Sequence[str]) -> Optional[str]:
    """Busca la primera carpeta con real/+fake/ o con la estructura oficial de FF++."""
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for cur, dirs, _ in os.walk(root):
            if ("real" in dirs and "fake" in dirs) or \
                    ("original_sequences" in dirs and "manipulated_sequences" in dirs):
                return cur
            if cur.count(os.sep) - root.count(os.sep) >= 4:
                dirs[:] = []
    return None


def quick_subset(videos: List[VideoItem], n_real: int = 25) -> List[VideoItem]:
    """n_real videos reales y un fake por cada uno, rotando el método. Solo para pruebas rápidas."""
    reals = sorted([v for v in videos if v.label == 0], key=lambda v: v.video_id)[:n_real]
    targets = [v.video_id for v in reals]
    by_target: Dict[str, List[VideoItem]] = defaultdict(list)
    for v in videos:
        if v.label == 1:
            by_target[v.identities[0]].append(v)
    fakes = []
    for i, t in enumerate(targets):
        cands = sorted(by_target.get(t, []), key=lambda v: v.method)
        if cands:
            fakes.append(cands[i % len(cands)])
    return reals + fakes


def smoke_subset(videos: List[VideoItem], n_per_class: int = 3) -> List[VideoItem]:
    """n reales y n fakes con identidades distintas; con n=3 queda 1 real y 1 fake por conjunto."""
    fakes, used = [], set()
    for v in sorted((v for v in videos if v.label == 1), key=lambda v: (v.identities[0], v.method)):
        if not used & set(v.identities) and v.method not in {f.method for f in fakes}:
            fakes.append(v)
            used |= set(v.identities)
        if len(fakes) == n_per_class:
            break
    fake_ids = {i for v in videos if v.label == 1 for i in v.identities}  # evita reales ligados a cualquier fake
    reals = [v for v in sorted((v for v in videos if v.label == 0), key=lambda v: v.video_id)
             if not set(v.identities) & fake_ids][:n_per_class]
    if len(fakes) < n_per_class or len(reals) < n_per_class:
        raise ValueError("No hay suficientes videos con identidades disjuntas para el smoke test")
    return reals + fakes


# Partición por identidad

class _UnionFind:
    def __init__(self) -> None:
        self.p: Dict[str, str] = {}

    def find(self, a: str) -> str:
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def assign_identity_groups(videos: List[VideoItem]) -> int:
    uf = _UnionFind()
    for v in videos:
        for i in v.identities:
            uf.find(i)
        for a, b in zip(v.identities, v.identities[1:]):
            uf.union(a, b)
    roots = sorted({uf.find(v.identities[0]) for v in videos})
    rid = {r: k for k, r in enumerate(roots)}
    for v in videos:
        v.group = rid[uf.find(v.identities[0])]
    return len(roots)


def split_by_identity(videos: List[VideoItem], ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
                      seed: int = 42) -> Dict[str, List[VideoItem]]:
    """Reparte grupos de identidad completos entre train, val y test, cuidando que haya reales y fakes en cada uno."""
    assert abs(sum(ratios) - 1.0) < 1e-6
    assign_identity_groups(videos)
    groups: Dict[int, List[VideoItem]] = defaultdict(list)
    for v in videos:
        groups[v.group].append(v)
    strata: Dict[Tuple[bool, bool], List[int]] = defaultdict(list)
    for g, vs in groups.items():
        strata[(any(v.label == 0 for v in vs), any(v.label == 1 for v in vs))].append(g)
    names = ("train", "val", "test")
    rng = np.random.RandomState(seed)
    for key in sorted(strata):
        gids = sorted(strata[key])
        rng.shuffle(gids)
        total = sum(len(groups[g]) for g in gids)
        target = np.array(ratios) * total
        count = np.zeros(3)
        for g in gids:
            k = int(np.argmax((target - count) / np.maximum(target, 1e-9)))
            count[k] += len(groups[g])
            for v in groups[g]:
                v.split = names[k]
    return {n: [v for v in videos if v.split == n] for n in names}


def check_identity_leakage(splits: Dict[str, List[VideoItem]]) -> Dict[str, int]:
    """Cuenta identidades y videos compartidos entre conjuntos."""
    ids = {k: {i for v in vs for i in v.identities} for k, vs in splits.items()}
    paths = {k: {v.path for v in vs} for k, vs in splits.items()}
    out = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        out[f"identidades {a}-{b}"] = len(ids[a] & ids[b])
        out[f"videos {a}-{b}"] = len(paths[a] & paths[b])
    return out


# Alineación facial (MediaPipe)

_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
_LE_IN, _LE_OUT, _RE_IN, _RE_OUT = 133, 33, 362, 263
ALIGN_VERSION = 3
_NOSE_TIP, _CHEEK_L, _CHEEK_R = 1, 234, 454


def yaw_index(pts: np.ndarray) -> float:
    """Giro de la cabeza en [-1, 1]: posición de la nariz respecto al centro de las mejillas (0 = frontal)."""
    xl, xr, xn = float(pts[_CHEEK_L, 0]), float(pts[_CHEEK_R, 0]), float(pts[_NOSE_TIP, 0])
    half = abs(xr - xl) / 2.0
    return float(np.clip((xn - (xl + xr) / 2.0) / max(half, 1e-6), -1.0, 1.0))


class FaceAligner:
    """Alinea el rostro como preprocess.py: nivela los ojos y recorta un cuadrado centrado de 224x224."""

    def __init__(self, output_size: int = 224, margin_ratio: float = 1.35,
                 task_model_path: str = "face_landmarker.task", small_face_fallback: bool = True) -> None:
        import mediapipe as mp

        self.output_size, self.margin = output_size, margin_ratio
        self.small_face_fallback = small_face_fallback
        self._haar = None
        if small_face_fallback:
            try:
                xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                self._haar = cv2.CascadeClassifier(xml)
                if self._haar.empty():
                    self._haar = None
            except Exception:  # build de OpenCV sin cv2.data
                self._haar = None
        if hasattr(mp, "solutions"):
            self.kind = "solutions"
            self._det = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5)
        else:
            from mediapipe.tasks.python import BaseOptions, vision

            if not os.path.exists(task_model_path):
                urllib.request.urlretrieve(_TASK_URL, task_model_path)
            self.kind = "tasks"
            self._mp = mp
            opts = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=task_model_path),
                running_mode=vision.RunningMode.IMAGE, num_faces=1,
                min_face_detection_confidence=0.5)
            self._det = vision.FaceLandmarker.create_from_options(opts)

    def landmarks(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Devuelve (468+, 2) en píxeles o None."""
        pts = self._landmarks_raw(frame_rgb)
        if pts is not None or not self.small_face_fallback:
            return pts
        return self._landmarks_small_face(frame_rgb)

    def _landmarks_small_face(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        if self._haar is None:
            return None
        H, W = frame_rgb.shape[:2]
        gray = cv2.equalizeHist(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY))
        boxes = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        for (x, y, w, h) in sorted(boxes, key=lambda b: -b[2] * b[3])[:3]:  # los 3 más grandes
            cx, cy, half = x + w / 2.0, y + h / 2.0, 1.1 * max(w, h)
            xa, ya = int(max(0, cx - half)), int(max(0, cy - half))
            xb, yb = int(min(W, cx + half)), int(min(H, cy + half))
            crop = frame_rgb[ya:yb, xa:xb]
            if crop.size == 0:
                continue
            s = 480.0 / max(crop.shape[:2])
            big = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            pts = self._landmarks_raw(big)
            if pts is not None:
                return pts / s + np.array([xa, ya], dtype=np.float32)
        return None

    def _landmarks_raw(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        H, W = frame_rgb.shape[:2]
        if self.kind == "solutions":
            res = self._det.process(frame_rgb)
            if not res.multi_face_landmarks:
                return None
            lms = res.multi_face_landmarks[0].landmark
        else:
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
            res = self._det.detect(img)
            if not res.face_landmarks:
                return None
            lms = res.face_landmarks[0]
        return np.array([[l.x * W, l.y * H] for l in lms], dtype=np.float32)

    def align(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pts = self.landmarks(rgb)
        if pts is None:
            return None, {}
        lc = (pts[_LE_IN] + pts[_LE_OUT]) / 2.0
        rc = (pts[_RE_IN] + pts[_RE_OUT]) / 2.0
        eye_c = (lc + rc) / 2.0
        angle = float(np.degrees(np.arctan2(rc[1] - lc[1], rc[0] - lc[0])))
        M = cv2.getRotationMatrix2D((float(eye_c[0]), float(eye_c[1])), angle, 1.0)
        aligned = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        rp = (M[:, :2] @ pts.T + M[:, 2:]).T
        nl = (rp[_LE_IN] + rp[_LE_OUT]) / 2.0
        nr = (rp[_RE_IN] + rp[_RE_OUT]) / 2.0
        y0, y1 = float(rp[:, 1].min()), float(rp[:, 1].max())
        x0, x1 = float(rp[:, 0].min()), float(rp[:, 0].max())
        cx, cy = float((nl[0] + nr[0]) / 2.0), (y0 + y1) / 2.0
        size = int(round(max((y1 - y0) * self.margin, (x1 - x0) * self.margin,
                             float(np.linalg.norm(nr - nl)) * 2.8)))
        # Sin tamaño mínimo: el encuadre es relativo al rostro (misma escala en todos los videos).
        # (preprocess.py forzaba al menos 224 px y en planos abiertos el rostro quedaba pequeño)
        size = max(size, 16)
        xa, ya = int(round(cx - size / 2.0)), int(round(cy - size / 2.0))
        xb, yb = xa + size, ya + size
        pt, pb, pl, pr = max(0, -ya), max(0, yb - H), max(0, -xa), max(0, xb - W)
        if pt or pb or pl or pr:
            aligned = cv2.copyMakeBorder(aligned, pt, pb, pl, pr, borderType=cv2.BORDER_REFLECT_101)
        crop = aligned[ya + pt:yb + pt, xa + pl:xb + pl]
        if crop.size == 0:
            return None, {}
        face = cv2.resize(crop, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
        return face, {"angle": angle, "landmarks": pts, "crop": (xa, ya, xb, yb), "yaw": yaw_index(rp)}

    def close(self) -> None:
        try:
            self._det.close()
        except Exception:
            pass


def sample_frame_indices(total: int, n: int) -> List[int]:
    """Muestreo uniforme de N índices (igual que dataset.py, sin duplicar frames)."""
    if total <= 0:
        return []
    if total <= n:
        return list(range(total))
    step = total / n
    return [int(i * step) for i in range(n)]


def read_frames(path: str, indices: Sequence[int]) -> List[Tuple[int, np.ndarray]]:
    wanted, out = set(indices), []
    cap = cv2.VideoCapture(path)
    fi = 0
    last = max(indices) if indices else -1
    while fi <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if fi in wanted:
            out.append((fi, frame))
        fi += 1
    cap.release()
    return out


def build_face_cache(videos: List[VideoItem], n_frames: int, aligner: FaceAligner,
                     cache_file: str, verbose: bool = True) -> Dict[str, np.ndarray]:
    """Extrae y alinea N frames por video."""
    paths = np.array([v.path for v in videos])
    if os.path.exists(cache_file):
        c = dict(np.load(cache_file, allow_pickle=False))
        if (c["paths"].shape == paths.shape and (c["paths"] == paths).all() and int(c["n_frames"]) == n_frames
                and int(c.get("align_version", 0)) == ALIGN_VERSION):
            if verbose:
                print(f"Caché reutilizada: {cache_file}")
            return c
    faces, fvid, fidx, fyaw = [], [], [], []
    n_detect = np.zeros(len(videos), dtype=np.int32)
    t0 = time.time()
    for vi, v in enumerate(videos):
        cap = cv2.VideoCapture(v.path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        for fi, frame in read_frames(v.path, sample_frame_indices(total, n_frames)):
            face, info = aligner.align(frame)
            if face is not None:
                faces.append(face)
                fyaw.append(info["yaw"])
                fvid.append(vi)
                fidx.append(fi)
                n_detect[vi] += 1
        if verbose and ((vi + 1) % 10 == 0 or vi + 1 == len(videos)):
            print(f"  {vi + 1:4d}/{len(videos)} videos | {len(faces)} rostros | {time.time() - t0:.0f}s")
    c = {
        "faces": np.stack(faces).astype(np.uint8),
        "frame_video": np.array(fvid, dtype=np.int32),
        "frame_index": np.array(fidx, dtype=np.int32),
        "frame_yaw": np.array(fyaw, dtype=np.float32),
        "n_detect": n_detect,
        "labels": np.array([v.label for v in videos], dtype=np.int64),
        "paths": paths,
        "n_frames": np.array(n_frames),
        "align_version": np.array(ALIGN_VERSION),
    }
    os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
    np.savez(cache_file, **c)
    return c


# Caché en carpeta (FF++ completo, memoria mapeada)
# Con FF++ completo los rostros ocupan varios GB: se guardan en faces.npy y se leen con
# se reparte entre procesos (MediaPipe corre en CPU; Kaggle ofrece 4 núcleos).

_WORKER_ALIGNER: Optional[FaceAligner] = None


def _worker_init(output_size: int) -> None:
    global _WORKER_ALIGNER
    _WORKER_ALIGNER = FaceAligner(output_size)


def _worker_video(args: Tuple[str, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Un video -> (rostros (k,224,224,3) uint8, índices de frame, yaw)."""
    path, n_grid, budget = args
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    grid = sample_frame_indices(total, n_grid)
    if budget < len(grid):
        grid = [grid[i] for i in np.linspace(0, len(grid) - 1, budget).round().astype(int)]
    faces, idx, yaw = [], [], []
    for fi, frame in read_frames(path, grid):
        face, info = _WORKER_ALIGNER.align(frame)
        if face is not None:
            faces.append(face)
            idx.append(fi)
            yaw.append(info["yaw"])
    sz = _WORKER_ALIGNER.output_size
    return (np.stack(faces) if faces else np.zeros((0, sz, sz, 3), np.uint8),
            np.array(idx, np.int32), np.array(yaw, np.float32))


def build_face_cache_dir(videos: List[VideoItem], cache_dir: str, n_grid: int = 16,
                         budgets: Optional[Sequence[int]] = None, workers: int = 1,
                         output_size: int = 224, verbose: bool = True) -> Dict[str, np.ndarray]:
    """Extrae rostros alineados a cache_dir/{faces.npy, meta.npz}."""
    budgets = np.array(budgets if budgets is not None else [n_grid] * len(videos), dtype=np.int32)
    paths = np.array([v.path for v in videos])
    meta_f, faces_f = os.path.join(cache_dir, "meta.npz"), os.path.join(cache_dir, "faces.npy")
    if os.path.exists(meta_f) and os.path.exists(faces_f):
        old = load_face_cache_dir(cache_dir)
        if (old["paths"].shape == paths.shape and (old["paths"] == paths).all()
                and (old["budgets"] == budgets).all() and int(old["n_grid"]) == n_grid
                and int(old["align_version"]) == ALIGN_VERSION):
            if verbose:
                print(f"Caché reutilizada: {cache_dir} ({len(old['frame_video'])} rostros)")
            return old
    os.makedirs(cache_dir, exist_ok=True)
    tmp_f = faces_f + ".tmp.npy"
    big = np.lib.format.open_memmap(tmp_f, mode="w+", dtype=np.uint8,
                                    shape=(int(budgets.sum()), output_size, output_size, 3))
    fvid, fidx, fyaw = [], [], []
    n_detect = np.zeros(len(videos), np.int32)
    jobs = [(v.path, n_grid, int(b)) for v, b in zip(videos, budgets)]
    t0, m = time.time(), 0
    if workers > 1:
        import multiprocessing as mp
        FaceAligner(output_size).close()  # descarga el modelo .task una sola vez
        # "spawn": procesos limpios. MediaPipe/TFLite no es seguro tras fork (corrompe el heap).
        pool = mp.get_context("spawn").Pool(workers, initializer=_worker_init, initargs=(output_size,))
        it = pool.imap(_worker_video, jobs, chunksize=4)
    else:
        _worker_init(output_size)
        pool, it = None, map(_worker_video, jobs)
    for vi, (f, i, y) in enumerate(it):
        k = len(f)
        big[m:m + k] = f
        fvid += [vi] * k
        fidx += list(i)
        fyaw += list(y)
        n_detect[vi] = k
        m += k
        if verbose and ((vi + 1) % 100 == 0 or vi + 1 == len(videos)):
            print(f"  {vi + 1:5d}/{len(videos)} videos | {m} rostros | {time.time() - t0:.0f}s")
    if pool is not None:
        pool.close()
        pool.join()
    big.flush()
    out = np.lib.format.open_memmap(faces_f, mode="w+", dtype=np.uint8, shape=(m, output_size, output_size, 3))
    for s in range(0, m, 2048):
        out[s:s + 2048] = big[s:s + 2048]
    out.flush()
    del big, out
    os.remove(tmp_f)
    np.savez(meta_f, frame_video=np.array(fvid, np.int32), frame_index=np.array(fidx, np.int32),
             frame_yaw=np.array(fyaw, np.float32), n_detect=n_detect,
             labels=np.array([v.label for v in videos], np.int64), paths=paths, budgets=budgets,
             n_grid=np.array(n_grid), align_version=np.array(ALIGN_VERSION),
             video_ids=np.array([v.video_id for v in videos]))
    return load_face_cache_dir(cache_dir)


def load_face_cache_dir(cache_dir: str) -> Dict[str, np.ndarray]:
    c = dict(np.load(os.path.join(cache_dir, "meta.npz"), allow_pickle=False))
    c["faces"] = np.load(os.path.join(cache_dir, "faces.npy"), mmap_mode="r")
    assert len(c["faces"]) == len(c["frame_video"]), "faces.npy y meta.npz no coinciden"
    return c
