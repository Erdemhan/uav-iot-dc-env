# setup_worker.ps1
# Laboratuvar Bilgisayarları İçin Otomatik Ray Worker Kurulum Betiği

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   Ray Cluster - İşçi (Worker) Kurulum Sihirbazı   " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Python Kontrolü
Write-Host "`n[1/3] Python 3.11 Kontrol Ediliyor..." -ForegroundColor Yellow
$pythonInstalled = $false
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python 3\.11") {
        Write-Host "[OK] Python 3.11 zaten kurulu: $pythonVersion" -ForegroundColor Green
        $pythonInstalled = $true
    }
} catch {}

if (-not $pythonInstalled) {
    Write-Host "[BİLGİ] Python 3.11 bulunamadı veya farklı bir sürüm kurulu. İndiriliyor..." -ForegroundColor Cyan
    $installerPath = "$env:TEMP\python-3.11.9-amd64.exe"
    $url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
    
    Write-Host "Python indirme adresi: $url"
    Invoke-WebRequest -Uri $url -OutFile $installerPath
    
    Write-Host "Sessiz kurulum başlatılıyor (Yönetici yetkisi gerekebilir)..." -ForegroundColor Yellow
    Start-Process -FilePath $installerPath -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1" -Wait
    
    # Path güncellemesi yapalım
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    # Tekrar kontrol edelim
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python 3\.11") {
            Write-Host "[OK] Python 3.11 başarıyla kuruldu!" -ForegroundColor Green
        } else {
            Write-Host "[HATA] Kurulum tamamlandı ama Python 3.11 bulunamadı. Lütfen PC'yi yeniden başlatıp bu scripti tekrar çalıştırın." -ForegroundColor Red
            Exit
        }
    } catch {
        Write-Host "[HATA] Python kurulumu doğrulanamadı. Manuel kurmanız gerekebilir." -ForegroundColor Red
        Exit
    }
}

# 2. Sanal Ortam (Virtual Environment) Oluşturma ve Kütüphane Kurulumu
Write-Host "`n[2/3] Sanal Ortam ve Kütüphaneler Kuruluyor..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    Write-Host "Sanal ortam (.venv) oluşturuluyor..."
    python -m venv .venv
} else {
    Write-Host "[OK] .venv zaten mevcut."
}

Write-Host "Sanal ortam aktifleştiriliyor ve pip güncelleniyor..."
& .\.venv\Scripts\activate.ps1
python -m pip install --upgrade pip

if (Test-Path "requirements.txt") {
    Write-Host "Gereksinimler (requirements.txt) yükleniyor..." -ForegroundColor Cyan
    pip install -r requirements.txt
    # Optuna'nın görselleştirme kütüphanesini de kuralım
    pip install optuna plotly
} else {
    Write-Host "[BİLGİ] requirements.txt bulunamadı! Temel Ray paketleri kuruluyor..." -ForegroundColor Yellow
    pip install "ray[rllib]>=2.53.0" pettingzoo==1.24.3 gymnasium torch numpy<2.0.0 pandas matplotlib seaborn optuna plotly
}

Write-Host "[OK] Kütüphane kurulumları tamamlandı!" -ForegroundColor Green

# 3. Ray Cluster Bağlantı Kurulumu
Write-Host "`n[3/3] Ray Cluster Bağlantı Sihirbazı..." -ForegroundColor Yellow
$headIp = Read-Host "Lütfen Ana Bilgisayarın (Head Node) IP adresini girin (Örn: 192.168.1.50)"

if (-not $headIp) {
    Write-Host "[HATA] IP adresi boş bırakılamaz." -ForegroundColor Red
    Exit
}

Write-Host "`nRay Worker başlatılıyor..." -ForegroundColor Cyan
Write-Host "Çalıştırılan Komut: ray start --address='$headIp:6379'" -ForegroundColor DarkGray

& .\.venv\Scripts\activate.ps1
ray start --address="$headIp:6379"

Write-Host "`n==================================================" -ForegroundColor Green
Write-Host "   Tebrikler! Bu bilgisayar Ray Cluster'a katıldı. " -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host "Eğitimi durdurmak veya bilgisayarı kümeden ayırmak için console ekranında 'ray stop' yazabilirsiniz.`n"
