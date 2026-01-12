biomedparse_datasets/convert_acdc_to_npz.py
- Read image and gt files from `<dataset_root>/<split>/img/` and `<dataset_root>/<split>/gt/`.
- For each case: load with nibabel, move slice axis to first dimension (D,H,W),
  remove slices with all-zero ground truth, scale image intensity to [0,255] (uint8),
  keep gt as integer labels, include `spacing` (voxel zooms) reordered to match axes,
  save to `<output_root>/<split>/*.npz` containing keys: `imgs`, `gts`, `spacing`.

scripts/eval_acdc.py
