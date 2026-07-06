#!/usr/bin/env bash

# setup_worker_wsl.sh
# Laboratuvar Bilgisayarlari Icin Otomatik Ray Worker Kurulum Betigi (WSL2 / Ubuntu)

# Renkler
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

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

# venv modulunun kurulu oldugundan emin olalim
python3 -c "import venv" &> /dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}[BILGI] python3-venv paketi eksik. Yukleniyor (Sudo yetkisi gerekebilir)...${NC}"
    sudo apt update && sudo apt install -y python3-venv python3-pip
fi

echo -e "${GREEN}[OK] Python3 ve gerekli araclar hazir: $(python3 --version)${NC}"

# 2. Sanal Ortam (Virtual Environment) Kurulumu
echo -e "\n${YELLOW}[2/3] Sanal Ortam (.venv) ve Bagimliliklar Kuruluyor...${NC}"
if [ ! -d ".venv" ]; then
    echo "Sanal ortam olusturuluyor (.venv)..."
    python3 -m venv .venv
else
    echo -e "${GREEN}[OK] .venv zaten mevcut.${NC}"
fi

echo "Sanal ortam aktiflestiriliyor ve pip guncelleniyor..."
source .venv/bin/activate
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo -e "${CYAN}requirements.txt uzerinden paketler kuruluyor...${NC}"
    pip install -r requirements.txt
    pip install optuna plotly
else
    echo -e "${YELLOW}[BİLGİ] requirements.txt bulunamadi. Temel Ray paketleri yukleniyor...${NC}"
    pip install "ray[rllib]>=2.53.0" pettingzoo==1.24.3 gymnasium torch numpy<2.0.0 pandas matplotlib seaborn optuna plotly
fi

echo -e "${GREEN}[OK] Kutuphane kurulumlari tamamlandi!${NC}"

# 3. Ray Cluster Baglanti Kurulumu
echo -e "\n${YELLOW}[3/3] Ray Cluster Baglanti Asamasi...${NC}"
read -p "Lutfen Ana Bilgisayarin (Head Node) Windows IP adresini girin (Orn: 192.168.1.50): " head_ip

if [ -z "$head_ip" ]; then
    echo -e "${RED}[HATA] IP adresi bos birakilamaz.${NC}"
    exit 1
fi

echo -e "\nRay Worker baslatiliyor..."
echo -e "Calistirilan Komut: ${CYAN}ray start --address=\"$head_ip:6379\"${NC}"

# Ray baslatilir
ray start --address="$head_ip:6379"

echo -e "\n${GREEN}==================================================${NC}"
echo -e "${GREEN}   Tebrikler! Bu WSL2 makinesi Ray Cluster'a katildi. ${NC}"
echo -e "${GREEN}==================================================${NC}"
echo -e "Kumeyi durdurmak veya ayrilmak icin: ${YELLOW}ray stop${NC}\n"
