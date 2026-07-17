#!/usr/bin/env bash

# setup_worker_wsl.sh
# Laboratuvar Bilgisayarlari Icin Otomatik Ray Worker Kurulum Betigi (WSL2 / Ubuntu)

# Renkler
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Betigin BASH ile calistirildigindan emin olalim
if [ -z "$BASH_VERSION" ]; then
    echo -e "${RED}[HATA] Bu betik yalnizca BASH kabugu ile calistirilmalidir.${NC}"
    echo -e "Lutfen betigi su sekilde calistirin: ${YELLOW}bash setup_worker_wsl.sh${NC} veya ${YELLOW}./setup_worker_wsl.sh${NC}"
    exit 1
fi

echo -e "${CYAN}==================================================${NC}"
echo -e "${CYAN}   Ray Cluster - WSL2 Isci (Worker) Kurulumu     ${NC}"
echo -e "${CYAN}==================================================${NC}"

# 1. Gerekli Sistem Paketlerinin Kontrolu
echo -e "\n${YELLOW}[1/3] Sistem Paketleri Kontrol Ediliyor...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[HATA] python3 bulunamadi. Lutfen WSL2 terminalinde python3 kurun.${NC}"
    echo -e "Kurmak icin: sudo apt update && sudo apt install -y python3 python3-pip python3-venv"
    exit 1
fi

# venv ve ensurepip modullerinin kurulu oldugundan emin olalim
python3 -c "import venv, ensurepip" &> /dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}[BILGI] python3-venv (ensurepip) paketi eksik. Yukleniyor (Sudo yetkisi gerekebilir)...${NC}"
    sudo apt update && sudo apt install -y python3-venv python3-pip
fi

echo -e "${GREEN}[OK] Python3 ve gerekli araclar hazir: $(python3 --version)${NC}"

# 2. Sanal Ortam (Virtual Environment) Kurulumu
echo -e "\n${YELLOW}[2/3] Sanal Ortam (.venv) ve Bagimliliklar Kuruluyor...${NC}"
if [ ! -f ".venv/bin/activate" ]; then
    # Eksik veya bozuk eski kurulumlari temizleyelim
    if [ -d ".venv" ]; then
        echo "Eksik veya bozuk .venv klasoru temizleniyor..."
        rm -rf .venv
    fi
    echo "Sanal ortam olusturuluyor (.venv)..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}[HATA] Sanal ortam (.venv) olusturulamadi.${NC}"
        echo -e "Lutfen python3-venv paketinin tam kuruldugundan emin olun:"
        echo -e "  sudo apt update && sudo apt install -y python3-venv"
        exit 1
    fi
else
    echo -e "${GREEN}[OK] .venv zaten mevcut ve gecerli.${NC}"
fi

echo "Sanal ortam aktiflestiriliyor ve pip guncelleniyor..."
source .venv/bin/activate
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo -e "${CYAN}requirements.txt uzerinden paketler kuruluyor...${NC}"
    pip install -r requirements.txt && pip install optuna plotly scikit-learn
else
    echo -e "${YELLOW}[BİLGİ] requirements.txt bulunamadi. Temel Ray paketleri yukleniyor...${NC}"
    pip install "ray[default,rllib]>=2.53.0" pettingzoo==1.24.3 gymnasium torch "numpy<2.0.0" pandas matplotlib seaborn optuna plotly scikit-learn
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}[HATA] Paket kurulumlari basarisiz oldu. Lutfen internet baglantinizi veya pip hatasini kontrol edin.${NC}"
    exit 1
fi

echo -e "${GREEN}[OK] Kutuphane kurulumlari tamamlandi!${NC}"

# 2.5. GPU ve CUDA Kontrolü
echo -e "\n${YELLOW}[2.5] GPU ve CUDA Durumu Kontrol Ediliyor...${NC}"

# WSL2 üzerinde GPU kütüphanelerinin ve nvidia-smi'nin bulunabilmesi için PATH ve LD_LIBRARY_PATH'i güncelleyelim
if [ -d "/usr/lib/wsl/lib" ]; then
    export PATH="/usr/lib/wsl/lib:$PATH"
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
fi

# A. nvidia-smi Kontrolü (WSL'in GPU'yu görüp görmediği)
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}[UYARI] nvidia-smi bulunamadi! WSL2 ekran kartinizi (GPU) algilayamiyor.${NC}"
    echo -e "Bu durum genellikle sunlardan kaynaklanir:"
    echo -e "  1. Windows ana makinede güncel NVIDIA sürücüsü yüklü degil."
    echo -e "  2. WSL sürümünüz güncel degil (Windows PowerShell'de 'wsl --update' calistirin)."
    echo -e "  3. WSL sürümü WSL1 olarak kalmis olabilir ('wsl --set-default-version 2' ile güncelleyin)."
    echo -e "${YELLOW}Devam ediliyor (Yalnizca CPU kullanilacaktir)...${NC}"
else
    echo -e "${GREEN}[OK] WSL2 ekran kartinizi algiladi:${NC}"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    
    # B. PyTorch CUDA Kontrolü (Sanal ortam icinde)
    echo -e "PyTorch CUDA erisimi kontrol ediliyor..."
    CUDA_CHECK=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    
    if [ "$CUDA_CHECK" = "True" ]; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        echo -e "${GREEN}[OK] PyTorch CUDA erisimi basarili! Kullanilabilir GPU: $GPU_NAME${NC}"
    else
        echo -e "${YELLOW}[BILGI] PyTorch CUDA'ya erisemiyor (torch.cuda.is_available() = False).${NC}"
        echo -e "CUDA uyumlu PyTorch sürümü otomatik olarak kuruluyor (bu islem internet hizina bagli olarak zaman alabilir)...${NC}"
        pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
        
        # Tekrar kontrol et
        CUDA_CHECK_NEW=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        if [ "$CUDA_CHECK_NEW" = "True" ]; then
            GPU_NAME_NEW=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
            echo -e "${GREEN}[OK] Kurulum basarili! PyTorch artik GPU kullanabilir: $GPU_NAME_NEW${NC}"
        else
            echo -e "${RED}[HATA] PyTorch hala CUDA'yi göremiyor. Sürücü ve WSL2 entegrasyonunu kontrol edin. Betik CPU modunda devam edecek.${NC}"
        fi
    fi
fi

# 2.6. SSH Sunucu (OpenSSH) Kurulumu ve Yapılandırması
echo -e "\n${YELLOW}[2.6] SSH Sunucu (OpenSSH) Kurulumu ve Yapılandırması...${NC}"

# A. OpenSSH Server Kurulumu
if ! dpkg -s openssh-server &> /dev/null; then
    echo "openssh-server yukleniyor (Sudo yetkisi gerekebilir)..."
    sudo apt update && sudo apt install -y openssh-server
else
    echo -e "${GREEN}[OK] openssh-server zaten kurulu.${NC}"
fi

# B. Port ve Sifreyle Giris Yapılandırması (Port 2222 - Windows ile cakismamasi icin)
echo "SSH Portu (2222) ve Sifreyle Kimlik Dogrulama yapilandiriliyor..."

# sshd_config dosyasını guncelle
# Port 2222 olarak ayarla veya varsa degistir
sudo sed -i 's/#\?Port [0-9]*/Port 2222/' /etc/ssh/sshd_config
# PasswordAuthentication yes olarak ayarla veya varsa degistir
sudo sed -i 's/#\?PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Gerekirse host key'leri uret
sudo ssh-keygen -A &> /dev/null

# C. SSH Servisini Baslat ve Etkinlestir
echo "SSH servisi baslatiliyor..."
sudo service ssh restart
SSH_STATUS=$?
sudo systemctl enable ssh &> /dev/null || true

if [ $SSH_STATUS -eq 0 ]; then
    echo -e "${GREEN}[OK] SSH Sunucu Port 2222 uzerinde basariyla aktif edildi!${NC}"
    echo -e "${YELLOW}[BILGI] Bu makineye erismek icin: ssh kullanici_adi@<IP_Adresi> -p 2222${NC}"
else
    echo -e "${RED}[HATA] SSH servisi baslatilamadi.${NC}"
fi

# 3. Ray Cluster Baglanti Kurulumu
echo -e "\n${YELLOW}[3/3] Ray Cluster Baglanti Asamasi...${NC}"
read -p "Lutfen Ana Bilgisayarin (Head Node) Windows IP adresini girin (Orn: 192.168.1.50): " head_ip

if [ -z "$head_ip" ]; then
    echo -e "${RED}[HATA] IP adresi bos birakilamaz.${NC}"
    exit 1
fi

echo -e "Calistirilan Komut: ${CYAN}ray start --address=\"$head_ip:6379\" --node-manager-port=10004 --object-manager-port=10002 --dashboard-agent-listen-port=10003 --min-worker-port=10010 --max-worker-port=10099${NC}"

# Check if ray exists in virtual env or PATH
RAY_CMD=""
if [ -f ".venv/bin/ray" ]; then
    RAY_CMD=".venv/bin/ray"
elif command -v ray &> /dev/null; then
    RAY_CMD="ray"
fi

if [ -z "$RAY_CMD" ]; then
    echo -e "${RED}[HATA] 'ray' komutu bulunamadi. Sanal ortamda '.venv/bin/ray' mevcut degil.${NC}"
    echo -e "Lutfen kurulum adimlarini ve bagimliliklarin yuklendigini kontrol edin.${NC}"
    exit 1
fi

# Ray baslatilir
$RAY_CMD start --address="$head_ip:6379" \
    --node-manager-port=10004 \
    --object-manager-port=10002 \
    --dashboard-agent-listen-port=10003 \
    --min-worker-port=10010 \
    --max-worker-port=10099

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}==================================================${NC}"
    echo -e "${GREEN}   Tebrikler! Bu WSL2 makinesi Ray Cluster'a katildi. ${NC}"
    echo -e "${GREEN}==================================================${NC}"
    echo -e "Kumeyi durdurmak veya ayrilmak icin: ${YELLOW}$RAY_CMD stop${NC}\n"
else
    echo -e "${RED}[HATA] Ray baslatilamadi veya Head Node'a baglanamadi.${NC}"
    echo -e "Lutfen sunlari kontrol edin:"
    echo -e "  1. Head Node IP adresinin dogrulugu ($head_ip)."
    echo -e "  2. Ag baglantisi (Ping atilabiliyor mu?)."
    echo -e "  3. Windows Firewall ayarlarinda Ray portlarinin (6379) acik oldugunu."
    exit 1
fi
