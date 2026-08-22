@echo off
setlocal
cd /d "%~dp0"

echo ================================================================
echo WL-BISINDO LOCAL REALTIME - SETUP WINDOWS
echo ================================================================
echo.

where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python launcher "py" tidak ditemukan.
    echo Install Python 3.11 x64 terlebih dahulu.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/6] Membuat .venv dengan Python 3.11...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo.
        echo [ERROR] Python 3.11 tidak ditemukan.
        echo Cek instalasi dengan: py -0p
        pause
        exit /b 1
    )
) else (
    echo [1/6] .venv sudah ada - skip create.
)

echo [2/6] Mengaktifkan environment...
call .venv\Scripts\activate.bat
if errorlevel 1 goto :error

echo [3/6] Upgrade pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :error

echo [4/6] Install dependency...
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [5/6] Menyiapkan folder model...
if not exist "model" mkdir model

echo [6/6] Cek dependency...
python check_setup.py --allow-missing-model
if errorlevel 1 goto :error

echo.
echo ================================================================
echo SETUP SELESAI
echo ================================================================
echo.
echo Setelah training Kaggle selesai, copy ke folder model:
echo   wl_bisindo_hand134_transformer_traced.pt
echo   feature_mean.npy
echo   feature_std.npy
echo   class_mapping.json
echo.
echo Setelah itu jalankan:
echo   run.bat
echo.
pause
exit /b 0

:error
echo.
echo [ERROR] Setup gagal. Baca error di atas.
pause
exit /b 1
