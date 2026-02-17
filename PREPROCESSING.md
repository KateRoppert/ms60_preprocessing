# MS60 Dataset Preprocessing: Cross-Modal Spatial Alignment

## Problem

The MS60 dataset (Muslim et al., 2022) provides multi-sequence brain MRI (FLAIR, T1, T2) for 60 MS patients collected from 20 different 1.5T MRI centers in Baghdad, Iraq. The data is distributed in NIfTI format with the following issues that prevent standard cross-modal analysis and multi-channel model training:

1. **Missing spatial metadata.** All files have `qform_code = 0` and `sform_code = 0`, meaning no valid coordinate system is defined. Voxel sizes are uniformly recorded as 1.0×1.0×1.0 mm, which is incorrect — for example, a 512×512-matrix image at 1.0 mm/pixel would imply a 51.2 cm FOV, far exceeding realistic brain dimensions.

2. **Heterogeneous acquisition matrices.** In-plane matrix sizes vary substantially across modalities and patients: FLAIR ranges from 224×224 to 512×512, T1 from 224×224 to 512×512, T2 from 224×224 to 1024×1024 (13, 11, and 15 unique in-plane sizes, respectively).

3. **Mismatched slice counts.** The number of axial slices differs between modalities in 33 out of 60 patients (e.g., FLAIR: 23 slices, T1: 19 slices, T2: 19 slices for Patient-1).

These issues make standard intensity-based registration tools (ANTs, FSL FLIRT) inapplicable: they operate in physical coordinates derived from the affine matrix, which here are meaningless. Our empirical attempts at registration (rigid, SyN) consistently produced corrupted outputs.

## Approach

Since the original DICOM files are not publicly available and the NIfTI headers lack valid spatial information, we adopted a geometry-based resampling strategy that operates directly on the image arrays without relying on coordinate metadata.

**Reference space.** For each patient, the FLAIR image defines the in-plane reference dimensions, as FLAIR is the primary modality for MS lesion visualization and segmentation.

**In-plane alignment.** T1 and T2 volumes are resized to match the FLAIR in-plane matrix using cubic interpolation (order 3) for images and nearest-neighbor interpolation (order 0) for lesion masks. Interpolation is applied independently per axial slice (the Z axis is not interpolated during this step) to avoid introducing artifacts along the through-plane direction, where the true slice spacing is unknown.

**Slice count harmonization.** When modalities have different numbers of axial slices, all volumes are center-cropped along Z to `min(Z_FLAIR, Z_T1, Z_T2)`. Center-cropping is preferred over Z-interpolation because the data consists of thick 2D-acquired slices with unknown inter-slice spacing, making interpolation along this axis unreliable. The average number of slices removed was 1.9 for FLAIR, 0.3 for T1, and 0.7 for T2 (maximum: 9 slices); the cropped slices correspond to the superior and inferior margins of the brain, which are typically lesion-sparse in MS.

**Header reconstruction.** Output files are saved with a clean RAS-oriented affine matrix, isotropic 1.0 mm voxel size, origin at the volume center, and `qform_code = sform_code = 1`. While the recorded voxel size does not reflect the true physical dimensions (which are unrecoverable without the original DICOM metadata), it provides a consistent and valid spatial reference for downstream tools.

**Mask integrity.** All lesion segmentation masks are binarized after resampling (threshold > 0.5) to eliminate interpolation artifacts.

## Result

After preprocessing, all three modalities and their corresponding lesion masks share identical dimensions within each patient (60/60 patients processed successfully). The 34 unique volume shapes across patients are handled by nnU-Net's internal resampling during training.

## Scripts

| Script | Purpose |
|---|---|
| `01_inspect_ms60.py` | Dataset geometry inspection and diagnostics |
| `02_resample_ms60_v2.py` | Cross-modal resampling (main preprocessing) |
| `03_verify_resample.py` | Output verification (shape consistency, mask integrity) |

## Dependencies

- Python ≥ 3.8
- nibabel
- scipy
- numpy

## Reference

Muslim, A. M., Mashohor, S., Al Gawwam, G., Mahmud, R., Hanafi, M. B., Alnuaimi, O., Josephine, R., & Almutairi, A. D. (2022). Brain MRI dataset of multiple sclerosis with consensus manual lesion segmentation and patient meta information. *Data in Brief*, 42, 108139. https://doi.org/10.1016/j.dib.2022.108139