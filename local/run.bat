@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] .venv belum ada.
    echo Jalankan setup_env.bat terlebih dahulu.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

python check_setup.py
if errorlevel 1 (
    echo.
    echo [ERROR] Runtime belum siap.
    echo Pastikan 4 file model dari Kaggle sudah berada di folder model.
    pause
    exit /b 1
)

echo.
echo Menjalankan WL-BISINDO realtime...
echo.

python realtime_bisindo.py %*

if errorlevel 1 (
    echo.
    echo Program berhenti karena error.
    pause
)
