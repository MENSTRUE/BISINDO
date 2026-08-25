BISINDO LOCAL V8.4 — DETECTOR/DEBUG FIX
=======================================

INI PATCH LOCAL, BUKAN TRAINING ULANG.

ROOT CAUSE YANG DIBERESKAN
--------------------------
1. Fallback config v2 lama masih menunjuk alphabet V7 ONNX single-frame.
   Patch sekarang fallback v2 = V8.4 TorchScript sequence 48x134.
2. Webcam pada screenshot sangat backlit/dark di area subjek.
   MediaPipe bisa gagal sebelum model V8.4 sempat inference.
3. Runtime lama hanya menampilkan "Valid 0%" tanpa info apakah Pose,
   detector utama, atau ROI recovery yang gagal.

PERUBAHAN
---------
- detector-only adaptive CLAHE + mild gamma untuk frame gelap/backlight
- whole-frame low-threshold fallback MediaPipe Hands bila tracker utama 0 hand
- ROI recovery juga memakai detector-only enhancement
- threshold Pose tracking diturunkan 0.40 -> 0.30
- debug overlay:
    Model / runtime / buffer
    Pose YES/NO
    Hand L/R YES/NO
    primary/fallback hand count
    Valid total/L/R
    center light + RAW/ENH
    confidence + margin + FPS
- active_model.txt dalam patch = v2
- models/v2/model_config.json = V8.4 TorchScript sequence

CARA PASANG
-----------
1. TUTUP run.bat / aplikasi realtime.
2. Backup folder local jika mau.
3. Extract ZIP ini ke folder `local` dan izinkan replace file:
       realtime_bisindo.py
       check_setup.py
       run.bat
       setup_env.bat
       requirements.txt
       active_model.txt
       models/v2/model_config.json
       models/v2/recommended_realtime_v8_4.json

4. JANGAN HAPUS file model V8.4 yang sudah ada di models/v2:
       alphabet_temporal_v8_4_traced.pt
       feature_mean_v8_4.npy
       feature_std_v8_4.npy
       class_mapping_v8_4.json
       metrics_v8_4.json

5. Jalankan:
       .\run.bat

STARTUP YANG BENAR
------------------
Active model   : v2
Runtime        : torchscript
Inference mode : sequence
Expected input : 1 x 48 x 134
Expected output: 1 x 26

DEBUG OVERLAY
-------------
Kalau tangan masih tidak terbaca, lihat baris:

Pose YES | Hand L YES R NO | P/F 1/0
Valid 83% L 83% R 0% | Light 68 ENH

Interpretasi:
- Pose NO + Hand L/R NO -> masalah kamera/lighting/detector.
- Pose YES + Hand NO -> tangan terlalu kecil/blur/tertutup; fallback akan dicoba.
- Hand YES tapi Valid rendah -> tahan gesture sampai buffer 48 frame terisi.
- Valid tinggi tapi kandidat tidak lolos -> baru masalah confidence/model/domain shift.

OPSIONAL
--------
Matikan enhancement bila kamera sudah terang:
    .\run.bat --no-detector-enhance

Matikan fallback detector untuk FPS maksimum:
    .\run.bat --no-detector-fallback

Untuk test awal, pencahayaan depan tetap lebih baik daripada jendela terang di belakang.
