#!/usr/bin/env python3
"""
Шаг 3: Верификация результатов ресэмплинга.
Проверяет, что все модальности одного пациента имеют одинаковый shape,
маски бинарны, и данные выглядят осмысленно.

Использование:
    python 03_verify_resample.py /path/to/output_dir
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path


def verify_patient(patient_dir: Path) -> list:
    """Проверяет одного пациента, возвращает список проблем."""
    pid = patient_dir.name
    num = pid.split("-")[-1]
    problems = []

    files = {
        "Flair": f"{num}-Flair.nii.gz",
        "T1": f"{num}-T1.nii.gz",
        "T2": f"{num}-T2.nii.gz",
        "Mask-Flair": f"{num}-LesionSeg-Flair.nii.gz",
        "Mask-T1": f"{num}-LesionSeg-T1.nii.gz",
        "Mask-T2": f"{num}-LesionSeg-T2.nii.gz",
    }

    shapes = {}
    for name, fname in files.items():
        fpath = patient_dir / fname
        if not fpath.exists():
            problems.append(f"{name}: файл отсутствует")
            continue

        img = nib.load(str(fpath))
        data = np.asarray(img.dataobj)
        shapes[name] = img.shape

        # Проверка заголовков
        if img.header["qform_code"] == 0:
            problems.append(f"{name}: qform_code = 0")

        # Проверка масок
        if name.startswith("Mask"):
            unique_vals = set(np.unique(data))
            if not unique_vals.issubset({0.0, 1.0}):
                problems.append(f"{name}: не бинарная, значения: {unique_vals}")

        # Проверка на пустые данные
        if data.max() == 0:
            problems.append(f"{name}: пустой (все нули)")

        # Проверка на NaN
        if np.any(np.isnan(data)):
            problems.append(f"{name}: содержит NaN")

    # Все shapes должны совпадать
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        problems.append(f"Разные shapes: {dict(zip(shapes.keys(), shapes.values()))}")

    return problems


def main():
    if len(sys.argv) < 2:
        print(f"Использование: python {sys.argv[0]} /path/to/output_dir")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    patient_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("Patient")],
        key=lambda x: int(x.name.split("-")[-1]),
    )

    print(f"Проверяю {len(patient_dirs)} пациентов...\n")

    ok_count = 0
    all_shapes = []

    for pdir in patient_dirs:
        problems = verify_patient(pdir)
        if problems:
            print(f"  {pdir.name}: ПРОБЛЕМЫ")
            for p in problems:
                print(f"    - {p}")
        else:
            ok_count += 1

        # Собираем shape FLAIR для статистики
        num = pdir.name.split("-")[-1]
        flair = pdir / f"{num}-Flair.nii.gz"
        if flair.exists():
            all_shapes.append(nib.load(str(flair)).shape)

    print(f"\nРезультат: {ok_count}/{len(patient_dirs)} пациентов OK")

    if all_shapes:
        unique = set(all_shapes)
        print(f"Уникальных shapes: {len(unique)}")
        for s in sorted(unique):
            count = all_shapes.count(s)
            print(f"  {s}: {count} пациентов")


if __name__ == "__main__":
    main()