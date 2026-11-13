# Third-party dependencies

The reproduction entry point depends on pinned external source repositories listed in `rescene/reproduction/upstream-lock.json`. These are fetched during setup, not relicensed by this repository.

- LTGS and its 3D Gaussian Splatting fork: renderer, initial training, camera and geometry utilities.
- Graphdeco 3D Gaussian Splatting, rasterizer and simple-knn: retain their upstream research/non-commercial terms where applicable.
- MASt3R, DUSt3R and CroCo: pretrained depth and matching; retain upstream model/code license terms, including MASt3R's non-commercial restrictions.
- Hierarchical Localization: COLMAP binary-format readers; no HLoc camera-estimation stage is used in this synthetic benchmark.
- FlashSplat rasterization and fused-SSIM: runtime imports/build dependencies of the upstream 3DGS fork.

The extracted `rescene/pipeline/mast3r_depth_support.py` function and class originate from the pinned LTGS `src/utils/localization_utils.py`. Integration patches are separately identified under `rescene/patches/`. Check each fetched repository's license before use or redistribution. No blanket permissive license is asserted for this combined research package or the supplied datasets.
