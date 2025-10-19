@echo off
REM Install Steno CLI tool in development mode

echo Installing Steno CLI tool...
echo.

REM Check if in virtual environment
if "%VIRTUAL_ENV%"=="" (
    echo WARNING: Not in a virtual environment!
    echo It's recommended to use a virtual environment.
    echo.
    choice /C YN /M "Continue anyway"
    if errorlevel 2 exit /b 1
)

REM Install in editable mode
pip install -e .

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo Installation successful!
    echo ========================================
    echo.
    echo You can now run 'steno' from anywhere in your terminal
    echo Example: steno -m -s -o transcript.jsonl
    echo.
) else (
    echo.
    echo Installation failed!
    exit /b 1
)
