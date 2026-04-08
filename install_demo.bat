@echo off
set VIRTUAL_ENV=
echo === LLaVA-Med Demo Installer ===
echo.

REM Check that uv is available
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: uv is not installed. Install it with:
    echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo Then restart this terminal and run install_demo.bat again.
    pause
    exit /b 1
)

mkdir llavamed_demo
cd llavamed_demo

REM Clone the repo and install all dependencies (auto-downloads Python 3.11)
git clone https://github.com/helena-intel/LLaVA-Med.git --single-branch --branch helena/uv .
uv sync

REM Download model files
uv run hf download helenai/llava-med-imf16-llmint8 --local-dir llava-med-imf16-llmint8

REM Download sample images
git clone https://github.com/LiangXin1001/LLaVA_Med_qa50_images.git 
tar -xf LLaVA_Med_qa50_images\qa50_images.zip
move qa50_images data

REM Create update script
echo @echo off > update.bat
echo set VIRTUAL_ENV= >> update.bat
echo git pull >> update.bat
echo uv sync >> update.bat
echo uv run hf download helenai/llava-med-imf16-llmint8 --local-dir llava-med-imf16-llmint8 >> update.bat

echo.
echo === Installation complete! Run launch_app.bat to start the demo. ===

