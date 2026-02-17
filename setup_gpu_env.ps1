# Setup GPU-enabled Python Environment for BanglaNLP
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GPU Environment Setup for BanglaNLP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Step 1: Create virtual environment
Write-Host "`n[1/5] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".\venv_gpu") {
    Write-Host "Removing existing venv_gpu directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".\venv_gpu"
}
python -m venv venv_gpu

# Step 2: Activate environment and upgrade pip
Write-Host "`n[2/5] Activating environment and upgrading pip..." -ForegroundColor Yellow
& .\venv_gpu\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Step 3: Install PyTorch with CUDA support
Write-Host "`n[3/5] Installing PyTorch with CUDA 12.4 support..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Gray
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Step 4: Install other required libraries
Write-Host "`n[4/5] Installing other required libraries..." -ForegroundColor Yellow
pip install ultralytics opencv-python-headless pillow numpy pandas matplotlib seaborn pyyaml

# Step 5: Verify installation
Write-Host "`n[5/5] Verifying GPU availability..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nTo activate this environment in the future, run:" -ForegroundColor Cyan
Write-Host "  .\venv_gpu\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "`nTo deactivate, run:" -ForegroundColor Cyan
Write-Host "  deactivate" -ForegroundColor White
