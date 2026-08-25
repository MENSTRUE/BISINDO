BISINDO LOCAL REALTIME — MULTIMODAL V1 UPDATE
==============================================

ROOT CAUSE ERROR LAMA
---------------------
Folder models/v1 sudah berisi model baru, tetapi check_setup/realtime lama
masih menganggap v1 = Hand134 V4 lama dan mencari:
  wl_bisindo_hand134_transformer_traced.pt
  feature_mean.npy
  feature_std.npy

Update ini mengubah v1 supaya membaca deployment_config.json hasil Kaggle
serta file normalisasi per-cabang yang benar.

RUNTIME V1 BARU
---------------
Webcam
  -> MediaPipe Pose
  -> MediaPipe Hands
  -> Hand134 sequence
  -> Pose36 sequence
  -> FaceHead52 sequence (dari MediaPipe Pose face/head points)
  -> pose-guided 48x48 face crop (disediakan untuk signature model)
  -> normalisasi branch-specific
  -> TorchScript multimodal model
  -> 32 kata

Winner training saat ini: C = Hand134 + Pose36 + FaceHead52.

V2 alphabet tetap kompatibel dan tidak diubah arsitekturnya.
