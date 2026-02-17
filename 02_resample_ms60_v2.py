#!/usr/bin/env python3
"""
Ресэмплинг MS60: приведение модальностей к единому пространству FLAIR.

Стратегия:
  1. Reference = FLAIR каждого пациента (in-plane shape)
  2. T1, T2 → resize in-plane к FLAIR
  3. Z-срезы → center-crop ВСЕХ модальностей к min(Z_flair, Z_t1, Z_t2)
     (безопаснее интерполяции для толстых 2D-срезов с неизвестным spacing)
  4. Маски → nearest-neighbor resize + бинаризация
  5. Сохраняем с чистым affine (RAS+, qform=sform=1)

Использование:
    python 02_resample_ms60_v2.py /path/to/MS60 /path/to/output

    Опции:
      --skip-existing     Не перезаписывать уже обработанных пациентов
      --single-mask MOD   Использовать маску только одной модальности (flair/t1/t2)
                          и применить её ко всем. По умолчанию: каждая маска своя.

Выход (для каждого пациента):
    Patient-N/
      N-Flair.nii.gz           # in-plane = оригинал, Z = center-crop
      N-T1.nii.gz              # in-plane resized к FLAIR, Z = center-crop
      N-T2.nii.gz              # in-plane resized к FLAIR, Z = center-crop
      N-LesionSeg-Flair.nii.gz # аналогично
      N-LesionSeg-T1.nii.gz
      N-LesionSeg-T2.nii.gz
"""

import sys
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import zoom as scipy_zoom
from typing import Tuple, Optional


# ──────────────────────────────────────────────
#  Вспомогательные функции
# ──────────────────────────────────────────────

def center_crop_z(data: np.ndarray, target_z: int) -> np.ndarray:
    """Центральная обрезка по оси Z."""
    current_z = data.shape[2]
    if current_z == target_z:
        return data
    if current_z < target_z:
        # Паддинг нулями (не должно случаться при min-стратегии, но на всякий)
        pad_total = target_z - current_z
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return np.pad(data, ((0, 0), (0, 0), (pad_before, pad_after)), mode="constant")
    # Обрезка
    start = (current_z - target_z) // 2
    return data[:, :, start : start + target_z]


def resize_in_plane(
    data: np.ndarray,
    target_xy: Tuple[int, int],
    order: int = 3,
) -> np.ndarray:
    """
    Resize in-plane (оси 0 и 1), ось Z не трогаем.
    order=3 для изображений, order=0 для масок.
    """
    if (data.shape[0], data.shape[1]) == target_xy:
        return data.copy()

    zoom_x = target_xy[0] / data.shape[0]
    zoom_y = target_xy[1] / data.shape[1]

    resampled = scipy_zoom(data.astype(np.float32), (zoom_x, zoom_y, 1.0), order=order)

    # Коррекция на ±1 пиксель из-за округления scipy
    if resampled.shape[0] != target_xy[0] or resampled.shape[1] != target_xy[1]:
        result = np.zeros((target_xy[0], target_xy[1], resampled.shape[2]), dtype=np.float32)
        sx = min(resampled.shape[0], target_xy[0])
        sy = min(resampled.shape[1], target_xy[1])
        result[:sx, :sy, :] = resampled[:sx, :sy, :]
        resampled = result

    if order > 0:
        resampled = np.clip(resampled, data.min(), data.max())

    return resampled


def make_clean_affine(shape: Tuple[int, ...]) -> np.ndarray:
    """Чистая RAS+ affine с origin в центре объёма, pixdim=1.0."""
    affine = np.eye(4)
    affine[0, 3] = -(shape[0] - 1) / 2.0
    affine[1, 3] = -(shape[1] - 1) / 2.0
    affine[2, 3] = -(shape[2] - 1) / 2.0
    return affine


def save_nifti(data: np.ndarray, affine: np.ndarray, path: Path):
    """Сохраняет NIfTI с корректными заголовками."""
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    img.header["qform_code"] = 1
    img.header["sform_code"] = 1
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, str(path))


def load_nii(path: Path) -> Optional[np.ndarray]:
    """Загружает NIfTI, пробует .nii и .nii.gz."""
    for p in [path, path.parent / (path.name + ".gz")]:
        if p.exists():
            return np.asarray(nib.load(str(p)).dataobj).astype(np.float32)
    return None


# ──────────────────────────────────────────────
#  Обработка пациента
# ──────────────────────────────────────────────

def process_patient(patient_dir: Path, output_dir: Path, skip_existing: bool = False) -> dict:
    """
    Обрабатывает одного пациента. Возвращает dict со статусом.
    """
    pid = patient_dir.name
    num = pid.split("-")[-1]
    out_dir = output_dir / pid

    if skip_existing and out_dir.exists():
        nfiles = len(list(out_dir.glob("*.nii.gz")))
        if nfiles >= 6:
            return {"status": "skipped", "reason": "already exists"}

    # ── Загрузка ──
    modality_files = {
        "Flair":          f"{num}-Flair.nii",
        "T1":             f"{num}-T1.nii",
        "T2":             f"{num}-T2.nii",
        "LesionSeg-Flair": f"{num}-LesionSeg-Flair.nii",
        "LesionSeg-T1":   f"{num}-LesionSeg-T1.nii",
        "LesionSeg-T2":   f"{num}-LesionSeg-T2.nii",
    }

    data = {}
    for mod, fname in modality_files.items():
        arr = load_nii(patient_dir / fname)
        if arr is None:
            return {"status": "error", "reason": f"{mod} not found"}
        data[mod] = arr

    # ── Определяем target shape ──
    flair_shape = data["Flair"].shape
    target_xy = (flair_shape[0], flair_shape[1])

    # Z: min по основным модальностям
    z_flair = data["Flair"].shape[2]
    z_t1 = data["T1"].shape[2]
    z_t2 = data["T2"].shape[2]
    target_z = min(z_flair, z_t1, z_t2)

    target_shape = (target_xy[0], target_xy[1], target_z)

    # ── Ресэмплинг ──
    out_dir.mkdir(parents=True, exist_ok=True)
    affine = make_clean_affine(target_shape)

    for mod, arr in data.items():
        is_mask = mod.startswith("LesionSeg")

        # Определяем, к какой модальности относится маска
        if is_mask:
            # Маска должна быть того же размера, что и родительская модальность
            # Ресэмплим in-plane к target_xy
            processed = resize_in_plane(arr, target_xy, order=0)
        else:
            # Изображение: resize in-plane к FLAIR
            processed = resize_in_plane(arr, target_xy, order=3)

        # Center-crop по Z
        processed = center_crop_z(processed, target_z)

        # Финализация масок
        if is_mask:
            processed = (processed > 0.5).astype(np.float32)

        # Сохраняем
        out_fname = modality_files[mod].replace(".nii", ".nii.gz")
        save_nifti(processed, affine, out_dir / out_fname)

    return {
        "status": "ok",
        "original_shapes": {
            "Flair": data["Flair"].shape,
            "T1": data["T1"].shape,
            "T2": data["T2"].shape,
        },
        "target_shape": target_shape,
        "z_cropped": {
            "Flair": z_flair - target_z,
            "T1": z_t1 - target_z,
            "T2": z_t2 - target_z,
        },
    }


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MS60: ресэмплинг модальностей к пространству FLAIR"
    )
    parser.add_argument("input_dir", help="Директория MS60 с папками Patient-*")
    parser.add_argument("output_dir", help="Выходная директория")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Пропускать уже обработанных пациентов")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("Patient")],
        key=lambda x: int(x.name.split("-")[-1]),
    )

    if not patient_dirs:
        print(f"Не найдены Patient-* в {input_dir}")
        sys.exit(1)

    print(f"Пациентов: {len(patient_dirs)}")
    print(f"Выход: {output_dir}")
    print(f"Стратегия: in-plane → FLAIR, Z → center-crop к min\n")

    stats = {"ok": 0, "error": 0, "skipped": 0}
    z_crop_total = {"Flair": [], "T1": [], "T2": []}

    for i, pdir in enumerate(patient_dirs):
        result = process_patient(pdir, output_dir, skip_existing=args.skip_existing)
        status = result["status"]
        stats[status] += 1

        if status == "ok":
            orig = result["original_shapes"]
            tgt = result["target_shape"]
            zcrop = result["z_cropped"]
            for m in ["Flair", "T1", "T2"]:
                z_crop_total[m].append(zcrop[m])

            print(
                f"  [{i+1:2d}/{len(patient_dirs)}] {pdir.name}: "
                f"FLAIR {orig['Flair']} | T1 {orig['T1']} | T2 {orig['T2']} "
                f"→ {tgt}  (Z crop: F={zcrop['Flair']}, T1={zcrop['T1']}, T2={zcrop['T2']})"
            )
        elif status == "error":
            print(f"  [{i+1:2d}/{len(patient_dirs)}] {pdir.name}: ERROR — {result['reason']}")
        else:
            print(f"  [{i+1:2d}/{len(patient_dirs)}] {pdir.name}: skipped")

    # Итоги
    print(f"\n{'='*60}")
    print(f"  ИТОГО: ok={stats['ok']}, errors={stats['error']}, skipped={stats['skipped']}")
    print(f"{'='*60}")

    if z_crop_total["Flair"]:
        print(f"\n  Статистика обрезки по Z:")
        for m in ["Flair", "T1", "T2"]:
            vals = z_crop_total[m]
            print(
                f"    {m:6s}: среднее обрезано={np.mean(vals):.1f} срезов, "
                f"макс={max(vals)}, без обрезки={vals.count(0)}/{len(vals)}"
            )


if __name__ == "__main__":
    main()