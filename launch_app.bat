@echo off
set VIRTUAL_ENV=
if exist LLaVA-Med\pyproject.toml pushd LLaVA-Med
start "LLaVA-Med Demo" /wait uv run python app.py
popd 2>nul
