@echo off
REM ============================================================
REM  AdaptiveFusion TPAMI - v6: BIR XIL epoch to'ri bilan qayta o'qitish
REM  Barcha metodlar uchun epoch to'ri: 25, 50, 100, 200, 400
REM
REM  Ishlatish:
REM     run1.bat            to'liq tajriba (natijalar: results\v6)
REM     run1.bat --quick    tezkor tekshiruv (2 seed, kichik to'r)
REM
REM  Eski natijalar (results\unified, results\v5) o'zgarmaydi.
REM  To'xtab qolsa, qayta ishga tushiring: tugagan qismlar qayta
REM  hisoblanmaydi. Oxirida "results_v6_for_review_....zip" yaratiladi.
REM ============================================================
setlocal
cd /d "%~dp0"
chcp 65001 >nul
if not exist logs mkdir logs

REM ---------- 1. Python ni topish ----------
set "PY="
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if defined PY goto :havepy
where python >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY goto :havepy
echo.
echo [XATO] Python topilmadi.
echo   https://www.python.org/downloads/ dan Python 3.11 yoki 3.12 ni o'rnating.
echo   O'rnatishda "Add python.exe to PATH" belgisini qo'ying, so'ng run.bat ni qayta bosing.
pause
exit /b 1

:havepy
%PY% -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 1)"
if errorlevel 1 (
    echo [XATO] Python 3.9 dan eski. Python 3.11 yoki 3.12 ni o'rnating.
    %PY% --version
    pause
    exit /b 1
)
echo Python:
%PY% --version

REM ---------- 2. Virtual muhit ----------
if exist ".venv\Scripts\python.exe" goto :haveenv
echo Virtual muhit yaratilmoqda...
%PY% -m venv .venv
if errorlevel 1 goto :err
:haveenv
set "VPY=%~dp0.venv\Scripts\python.exe"

REM ---------- 3. Kutubxonalar (faqat birinchi marta) ----------
if exist ".venv\installed.flag" goto :run
echo.
echo Kutubxonalar o'rnatilmoqda. Birinchi marta 5-20 daqiqa olishi mumkin...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :err

where nvidia-smi >nul 2>nul
if errorlevel 1 goto :cputorch

echo NVIDIA GPU topildi - PyTorch CUDA versiyasi o'rnatiladi.
"%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if not errorlevel 1 goto :torchok
"%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu126
if not errorlevel 1 goto :torchok
echo CUDA versiyasi o'rnatilmadi - CPU versiyasiga o'tilmoqda.

:cputorch
echo PyTorch CPU versiyasi o'rnatilmoqda.
"%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if not errorlevel 1 goto :torchok
"%VPY%" -m pip install torch
if errorlevel 1 goto :err

:torchok
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :err
"%VPY%" -c "import torch, sklearn, scipy, pandas, matplotlib; print('PyTorch', torch.__version__, '| GPU:', torch.cuda.is_available())"
if errorlevel 1 goto :err
echo ok> ".venv\installed.flag"

REM ---------- 4. Tajribalar ----------
:run
echo.
echo ============================================================
echo  Tajribalar boshlandi. Kompyuterni uyqu rejimiga qo'ymang.
echo  Jarayon "logs" papkasiga yozib boriladi.
echo ============================================================
echo.
"%VPY%" run1.py %*
if errorlevel 1 goto :err

echo.
echo ============================================================
echo  TAYYOR. "results_v6_for_review_....zip" faylini yuboring.
echo ============================================================
pause
exit /b 0

:err
echo.
echo [XATO] Nimadir ishlamadi. "logs" papkasidagi oxirgi faylni
echo yoki shu oynadagi xabarni yuboring.
echo Kutubxonalarni qayta o'rnatish uchun ".venv" papkasini o'chirib, run.bat ni qayta bosing.
pause
exit /b 1
