@echo off
title RoomSense AI — Push to GitHub Helper
cls
echo =====================================================================
echo              RoomSense AI — GitHub Push Automation
echo =====================================================================
echo.

where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git is not installed or not in your system PATH.
    echo.
    echo Please follow these 2 quick steps:
    echo 1. Download Git for Windows from: https://git-scm.com/download/win
    echo 2. Run the installer (keep default options), then re-run this script.
    echo.
    pause
    exit /b 1
)

echo [OK] Git is detected on your system!
echo.
echo Please create a new repository on GitHub (https://github.com/new).
echo Make sure "Add a README file" is UNCHECKED.
echo.
set /p REPO_URL="Enter your GitHub Repository URL (e.g. https://github.com/username/RoomSense-AI.git): "

if "%REPO_URL%"=="" (
    echo.
    echo [ERROR] No URL provided. Aborting.
    pause
    exit /b 1
)

echo.
echo Initializing Git repository and committing project files...
if not exist .git (
    git init
)

git add .
git commit -m "Initial commit: RoomSense AI full project with ML/DL notebooks & dashboard"
git branch -M main

git remote remove origin >nul 2>nul
git remote add origin %REPO_URL%

echo.
echo Pushing project to GitHub (%REPO_URL%)...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =====================================================================
    echo [SUCCESS] Your project has been successfully pushed to GitHub!
    echo =====================================================================
) else (
    echo.
    echo [NOTE] If git push asked for credentials, please sign in via browser
    echo or use your GitHub Personal Access Token (PAT).
)

echo.
pause

