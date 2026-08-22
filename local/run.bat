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

echo.
echo Active model:
type active_model.txt
echo.

python check_setup.py
if errorlevel 1 (
    echo.
    echo [ERROR] Active model belum siap.
    echo Cek active_model.txt dan folder models\VERSION\
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
