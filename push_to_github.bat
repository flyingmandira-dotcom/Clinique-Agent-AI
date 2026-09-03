@echo off
setlocal enabledelayedexpansion
title Push Clinique-Agent-AI to GitHub

echo =======================================================
echo     Clinique-Agent-AI: GitHub Auto-Push Tool
echo =======================================================
echo.

:: Check if git is installed
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not found in PATH.
    echo Please install Git from https://git-scm.com/
    goto :end
)

:: Repository configuration
set REPO_URL=https://github.com/flyingmandira-dotcom/Clinique-Agent-AI.git
set BRANCH=main

:: Initialize git repository if not already initialized
if not exist .git (
    echo [INFO] Initializing Git repository...
    git init
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to initialize Git repository.
        goto :end
    )
)

:: Ensure default branch is set to main
git branch -M %BRANCH% >nul 2>nul

:: Check and configure remote origin
git remote get-url origin >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Adding remote origin: %REPO_URL%
    git remote add origin %REPO_URL%
) else (
    echo [INFO] Updating remote origin: %REPO_URL%
    git remote set-url origin %REPO_URL%
)

:: Prompt for commit message or use default
echo.
set /p "COMMIT_MSG=Enter commit message (Press Enter for default message): "
if "%COMMIT_MSG%"=="" (
    set "COMMIT_MSG=Update Clinique Agent AI: %date% %time%"
)

echo.
echo [1/3] Staging changes...
git add .

:: Check if there are changes to commit
git diff --cached --quiet
if %errorlevel% equ 0 (
    echo [INFO] No new changes to commit.
) else (
    echo [2/3] Committing changes with message: "%COMMIT_MSG%"
    git commit -m "%COMMIT_MSG%"
)

echo.
echo [3/3] Pushing to GitHub (%BRANCH% branch)...
git push -u origin %BRANCH%

if %errorlevel% equ 0 (
    echo.
    echo =======================================================
    echo [SUCCESS] Repository successfully pushed to GitHub!
    echo URL: https://github.com/flyingmandira-dotcom/Clinique-Agent-AI
    echo =======================================================
) else (
    echo.
    echo [WARNING] Direct push failed. Attempting to pull remote changes and retry...
    git pull origin %BRANCH% --rebase
    git push -u origin %BRANCH%
    if !errorlevel! equ 0 (
        echo.
        echo [SUCCESS] Repository updated and successfully pushed!
    ) else (
        echo.
        echo [ERROR] Push failed. Please check your GitHub credentials or remote repository permissions.
    )
)

:end
echo.
pause
