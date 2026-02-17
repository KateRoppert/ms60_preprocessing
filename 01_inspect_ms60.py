#!/usr/bin/env python3
"""
Шаг 1: Инспекция геометрии MS60 датасета.
Запусти перед ресэмплингом, чтобы подтвердить предположения о данных.

Использование:
    python 01_inspect_ms60.py /path/to/MS60_dataset

Ожидаемая структура:
    Patient-1/1-Flair.nii, 1-T1.nii, 1-T2.nii, 1-LesionSeg-Flair.nii, ...
"""

import sys
import os
import glob
import nibabel as nib
import numpy as np
from pathlib import Path


def inspect_patient(patient_dir: Path) -> dict:
    """Извлекает геометрию всех модальностей одного пациента."""
    patient_id = patient_dir.name
    # Определяем номер пациента из имени директории
    num = patient_id.split("-")[-1]

    modalities = {
        "Flair": f"{num}-Flair.nii",
        "T1": f"{num}-T1.nii",
        "T2": f"{num}-T2.nii",
        "LesionSeg-Flair": f"{num}-LesionSeg-Flair.nii",
        "LesionSeg-T1": f"{num}-LesionSeg-T1.nii",
        "LesionSeg-T2": f"{num}-LesionSeg-T2.nii",
    }

    info = {"patient_id": patient_id}
    for mod_name, filename in modalities.items():
        fpath = patient_dir / filename
        if not fpath.exists():
            # Попробуем .nii.gz
            fpath = patient_dir / (filename + ".gz")
        if not fpath.exists():
            info[mod_name] = None
            continue

        img = nib.load(str(fpath))
        hdr = img.header
        info[mod_name] = {
            "shape": img.shape,
            "pixdim": tuple(np.round(hdr.get_zooms(), 4)),
            "affine_diag": tuple(np.round(np.diag(img.affine)[:3], 4)),
            "qform_code": int(hdr["qform_code"]),
            "sform_code": int(hdr["sform_code"]),
            "origin": tuple(np.round(img.affine[:3, 3], 2)),
            "dtype": str(img.get_data_dtype()),
            "data_range": (
                float(np.min(np.asarray(img.dataobj))),
                float(np.max(np.asarray(img.dataobj))),
            ),
        }
    return info


def main():
    if len(sys.argv) < 2:
        print(f"Использование: python {sys.argv[0]} /path/to/MS60_dataset")
        sys.exit(1)

    dataset_dir = Path(sys.argv[1])
    patient_dirs = sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("Patient")],
        key=lambda x: int(x.name.split("-")[-1]),
    )

    if not patient_dirs:
        print(f"Не найдены директории Patient-* в {dataset_dir}")
        sys.exit(1)

    print(f"Найдено пациентов: {len(patient_dirs)}\n")

    # Собираем статистику по всем пациентам
    all_shapes = {"Flair": [], "T1": [], "T2": []}
    problems = []

    for pdir in patient_dirs:
        info = inspect_patient(pdir)
        pid = info["patient_id"]

        for mod in ["Flair", "T1", "T2"]:
            if info[mod] is not None:
                all_shapes[mod].append(info[mod]["shape"])

        # Детальный вывод для первых 5 пациентов
        if int(pid.split("-")[-1]) <= 5:
            print(f"{'='*60}")
            print(f"  {pid}")
            print(f"{'='*60}")
            for mod in ["Flair", "T1", "T2", "LesionSeg-Flair", "LesionSeg-T1", "LesionSeg-T2"]:
                d = info[mod]
                if d is None:
                    print(f"  {mod:20s}: ФАЙЛ НЕ НАЙДЕН")
                else:
                    print(
                        f"  {mod:20s}: shape={str(d['shape']):20s} "
                        f"pixdim={str(d['pixdim']):20s} "
                        f"aff_diag={d['affine_diag']}  "
                        f"qf={d['qform_code']} sf={d['sform_code']}  "
                        f"origin={d['origin']}  "
                        f"range=[{d['data_range'][0]:.1f}, {d['data_range'][1]:.1f}]"
                    )
            print()

    # Сводная статистика
    print(f"\n{'='*60}")
    print("  СВОДКА ПО ВСЕМУ ДАТАСЕТУ")
    print(f"{'='*60}\n")

    for mod in ["Flair", "T1", "T2"]:
        shapes = all_shapes[mod]
        if not shapes:
            print(f"  {mod}: нет данных")
            continue

        unique_shapes = set(shapes)
        print(f"  {mod} ({len(shapes)} файлов):")
        print(f"    Уникальных shape: {len(unique_shapes)}")

        # In-plane sizes
        in_plane = set((s[0], s[1]) for s in shapes)
        z_slices = [s[2] for s in shapes]
        print(f"    In-plane размеры: {sorted(in_plane)}")
        print(f"    Z (срезы): min={min(z_slices)}, max={max(z_slices)}, "
              f"median={np.median(z_slices):.0f}, unique={len(set(z_slices))}")

        # Проверяем: если бы voxel size = 1.0, какой бы был FOV?
        for ip in sorted(in_plane):
            fake_fov = (ip[0] * 1.0, ip[1] * 1.0)
            print(f"    → FOV при pixdim=1mm: {fake_fov[0]:.0f}×{fake_fov[1]:.0f} mm "
                  f"(реально для мозга ~180-240mm)")
        print()

    # Проверяем соответствие Z-срезов между модальностями
    print("  Совпадение количества срезов (Flair vs T1 vs T2):")
    match_count = 0
    mismatch_patients = []
    for pdir in patient_dirs:
        info = inspect_patient(pdir)
        pid = info["patient_id"]
        z_vals = {}
        for mod in ["Flair", "T1", "T2"]:
            if info[mod]:
                z_vals[mod] = info[mod]["shape"][2]
        if len(set(z_vals.values())) == 1:
            match_count += 1
        else:
            mismatch_patients.append((pid, z_vals))

    print(f"    Совпадают: {match_count}/{len(patient_dirs)}")
    if mismatch_patients:
        print(f"    Не совпадают ({len(mismatch_patients)}):")
        for pid, zv in mismatch_patients[:10]:
            print(f"      {pid}: {zv}")
        if len(mismatch_patients) > 10:
            print(f"      ... и ещё {len(mismatch_patients) - 10}")


if __name__ == "__main__":
    main()