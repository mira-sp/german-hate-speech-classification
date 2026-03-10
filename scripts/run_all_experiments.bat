@echo off

REM  Hate Speech Classification - Alle Experimente ausführen
REM ============================================================
REM  Dieses Script führt alle Experimente sequentiell aus.
REM  Geschätzte Gesamtzeit (RTX 3060): ~12-15 Stunden


echo -------------------------------------------------------------
echo   Hate Speech Classification - Experiment Pipeline

echo.

REM Sicherheitscheck: GPU
python -c "import torch; assert torch.cuda.is_available(), 'CUDA nicht verfügbar!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"
if %ERRORLEVEL% NEQ 0 (
    echo FEHLER: Keine GPU verfügbar!
    pause
    exit /b 1
)

echo.
echo === Schritt 0: Daten vorverarbeiten ===
python -m src.preprocessing --variant full_preprocessing
if %ERRORLEVEL% NEQ 0 (
    echo FEHLER bei Preprocessing!
    pause
    exit /b 1
)

echo.
echo === Schritt 1: Baseline-Modelle (Experiment 1) ===
python -m experiments.01_baselines
if %ERRORLEVEL% NEQ 0 (
    echo FEHLER bei Experiment 1!
    pause
    exit /b 1
)

echo.
echo === Schritt 2: BERT Fine-Tuning (Experiment 2) ===
echo --- mBERT ---
python -m experiments.02_bert_full_data --model mBERT --folds 5 --epochs 3
echo --- GBERT ---
python -m experiments.02_bert_full_data --model GBERT --folds 5 --epochs 3
echo --- HateBERT ---
python -m experiments.02_bert_full_data --model HateBERT --folds 5 --epochs 3
if %ERRORLEVEL% NEQ 0 (
    echo FEHLER bei Experiment 2!
    pause
    exit /b 1
)

echo.
echo === Schritt 3: Data Size Variation (Experiment 3) ===
python -m experiments.03_data_size_variation --model GBERT --sizes 0.25 0.5 0.75 1.0 --folds 5
if %ERRORLEVEL% NEQ 0 (
    echo FEHLER bei Experiment 3!
    pause
    exit /b 1
)

echo.
echo === Schritt 4: Preprocessing Ablation (Experiment 4) ===
python -m experiments.04_preprocessing_ablation --model GBERT --folds 5
if %ERRORLEVEL% NEQ 0 (
    echo FEHLER bei Experiment 4!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   ALLE EXPERIMENTE ABGESCHLOSSEN!
echo ============================================================
echo   Ergebnisse in: results/metrics/
echo   Plots in:      results/plots/
echo   Modelle in:    results/models/
echo ============================================================

pause
