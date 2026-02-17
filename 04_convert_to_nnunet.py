#!/usr/bin/env python3
"""
Конвертация препроцессированного MS60 в формат nnU-Net raw dataset.

Структура выхода:
    nnUNet_raw/
      DatasetNNN_MS60/
        dataset.json
        imagesTr/
          MS60_001_0000.nii.gz   (FLAIR)
          MS60_001_0001.nii.gz   (T1)
          MS60_001_0002.nii.gz   (T2)
        labelsTr/
          MS60_001.nii.gz        (LesionSeg-Flair)
        imagesTs/
          MS60_049_0000.nii.gz
        labelsTs/
          MS60_049.nii.gz

Использование:
    python 04_convert_to_nnunet.py /path/to/resampled /path/to/nnUNet_raw --dataset-id 100

    --dataset-id N    Номер датасета для nnU-Net (по умолчанию 100)
    --seed S          Random seed для split (по умолчанию 42)
    --test-count N    Количество пациентов в тесте (по умолчанию 12)
"""

import sys
import json
import shutil
import random
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path


def get_patient_num(patient_dir: Path) -> int:
    """Извлекает номер пациента из имени директории."""
    return int(patient_dir.name.split("-")[-1])


def case_id(patient_num: int) -> str:
    """Формирует case ID в формате nnU-Net: MS60_001."""
    return f"MS60_{patient_num:03d}"


def copy_as_channel(src: Path, dst: Path):
    """Копирует файл. Если src .nii (не .gz), сохраняет как .nii.gz."""
    if src.suffix == ".gz":
        shutil.copy2(src, dst)
    else:
        # Пересохраняем как .nii.gz
        img = nib.load(str(src))
        nib.save(img, str(dst))


def validate_labels(label_path: Path) -> dict:
    """Проверяет маску: бинарность, наличие лезий."""
    img = nib.load(str(label_path))
    data = np.asarray(img.dataobj)
    unique = np.unique(data)
    n_lesion_voxels = int(np.sum(data > 0))
    return {
        "unique_values": unique.tolist(),
        "n_lesion_voxels": n_lesion_voxels,
        "is_binary": set(unique.tolist()).issubset({0.0, 1.0}),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Конвертация MS60 → nnU-Net raw dataset"
    )
    parser.add_argument("input_dir", help="Директория с препроцессированными данными")
    parser.add_argument("output_dir", help="Путь к nnUNet_raw")
    parser.add_argument("--dataset-id", type=int, default=100,
                        help="ID датасета (по умолчанию 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed для train/test split")
    parser.add_argument("--test-count", type=int, default=12,
                        help="Количество пациентов в тесте")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_base = Path(args.output_dir)
    dataset_name = f"Dataset{args.dataset_id:03d}_MS60"
    dataset_dir = output_base / dataset_name

    # Находим пациентов
    patient_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("Patient")],
        key=lambda x: get_patient_num(x),
    )
    n_patients = len(patient_dirs)
    assert n_patients == 60, f"Ожидалось 60 пациентов, найдено {n_patients}"

    # ── Train/Test split ──
    random.seed(args.seed)
    indices = list(range(n_patients))
    random.shuffle(indices)
    test_indices = set(indices[: args.test_count])
    train_indices = set(indices[args.test_count :])

    print(f"Датасет: {dataset_name}")
    print(f"Пациентов: {n_patients} (train={len(train_indices)}, test={len(test_indices)})")
    print(f"Seed: {args.seed}")
    print(f"Каналы: 0=FLAIR, 1=T1, 2=T2")
    print(f"Ground truth: LesionSeg-Flair\n")

    # ── Создаём директории ──
    for subdir in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]:
        (dataset_dir / subdir).mkdir(parents=True, exist_ok=True)

    # ── Конвертация ──
    train_cases = []
    test_cases = []
    empty_label_cases = []

    for i, pdir in enumerate(patient_dirs):
        num = get_patient_num(pdir)
        cid = case_id(num)
        is_test = i in test_indices

        img_subdir = "imagesTs" if is_test else "imagesTr"
        lbl_subdir = "labelsTs" if is_test else "labelsTr"

        # Каналы: FLAIR=0000, T1=0001, T2=0002
        channels = {
            "0000": f"{num}-Flair.nii.gz",
            "0001": f"{num}-T1.nii.gz",
            "0002": f"{num}-T2.nii.gz",
        }

        ok = True
        for ch_id, fname in channels.items():
            src = pdir / fname
            if not src.exists():
                print(f"  [ERROR] {cid}: {fname} не найден")
                ok = False
                continue
            dst = dataset_dir / img_subdir / f"{cid}_{ch_id}.nii.gz"
            copy_as_channel(src, dst)

        # Label: LesionSeg-Flair
        label_src = pdir / f"{num}-LesionSeg-Flair.nii.gz"
        label_dst = dataset_dir / lbl_subdir / f"{cid}.nii.gz"
        if not label_src.exists():
            print(f"  [ERROR] {cid}: label не найден")
            ok = False
        else:
            copy_as_channel(label_src, label_dst)
            # Валидация
            lbl_info = validate_labels(label_dst)
            if not lbl_info["is_binary"]:
                print(f"  [WARN] {cid}: маска не бинарная! values={lbl_info['unique_values']}")
            if lbl_info["n_lesion_voxels"] == 0:
                empty_label_cases.append(cid)

        if ok:
            if is_test:
                test_cases.append(cid)
            else:
                train_cases.append(cid)

        split_tag = "test" if is_test else "train"
        print(f"  [{i+1:2d}/{n_patients}] {cid} ({pdir.name}) → {split_tag}")

    # ── dataset.json ──
    dataset_json = {
        "channel_names": {
            "0": "FLAIR",
            "1": "T1",
            "2": "T2",
        },
        "labels": {
            "background": 0,
            "lesion": 1,
        },
        "numTraining": len(train_cases),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }

    json_path = dataset_dir / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    # ── Сохраняем split info ──
    split_info = {
        "seed": args.seed,
        "train": sorted(train_cases),
        "test": sorted(test_cases),
    }
    split_path = dataset_dir / "split_info.json"
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)

    # ── Итоги ──
    print(f"\n{'='*60}")
    print(f"  Готово!")
    print(f"  Датасет: {dataset_dir}")
    print(f"  Train: {len(train_cases)} cases")
    print(f"  Test:  {len(test_cases)} cases")
    print(f"  dataset.json: {json_path}")
    print(f"  split_info.json: {split_path}")
    if empty_label_cases:
        print(f"\n  ⚠ Пустые маски (0 лезий): {empty_label_cases}")
    print(f"{'='*60}")

    # Подсказка по запуску nnU-Net
    print(f"""
Следующие шаги:

  1. Задать переменные окружения:
     export nnUNet_raw="{output_base}"
     export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
     export nnUNet_results="/path/to/nnUNet_results"

  2. Планирование и препроцессинг:
     nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity

  3. Обучение (5-fold CV):
     nnUNetv2_train {args.dataset_id} 3d_fullres FOLD

  4. Инференс на тесте:
     nnUNetv2_predict -i {dataset_dir}/imagesTs -o /path/to/predictions -d {args.dataset_id} -c 3d_fullres -f all
""")


if __name__ == "__main__":
    main()