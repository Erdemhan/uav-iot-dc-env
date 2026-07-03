# Dağıtık Ray Cluster Kurulum Kılavuzu
## UAV-IoT Hiper-Parametre Optimizasyonu — Lab Bilgisayarları

---

## Mimari Özet

```
┌────────────────────────────────────────────────┐
│  SEN (Head Node)                               │
│  IP: 192.168.X.X                               │
│  • tune_models.py çalıştırır                   │
│  • Optuna denemeleri yönetir                   │
│  • Dashboard: http://localhost:8000/opt.html   │
└────────────────┬───────────────────────────────┘
                 │ Ray Protocol (port 6379)
     ┌───────────┼───────────┐
     ▼           ▼           ▼
  Worker 1    Worker 2  ... Worker 29
  (CPU+GPU)  (CPU+GPU)      (CPU+GPU)
  Denemeleri paralel çalıştırır
```

---

## ADIM 1 — Kendi Bilgisayarında (Head Node) Yap

### 1.1 IP Adresini Öğren
```powershell
ipconfig
# "IPv4 Address" satırındaki adres (örn: 192.168.1.50)
```

### 1.2 Ray Head Node'u Başlat (Sadece Koordinatör)
```powershell
# --num-cpus=0 --num-gpus=0 → Ray bu makineye hiçbir trial/env-runner yerleştirmez
# Tüm hesaplama yükü worker'lara gider; sen sadece yönetirsin.
ray start --head --port=6379 --dashboard-host=0.0.0.0 --num-cpus=0 --num-gpus=0
```
Çıktıda şunu göreceksin:
```
Ray runtime started.
To add another node to this Ray cluster, run:
  ray start --address='192.168.1.50:6379'  ← Bu adresi worker'lara ver
```

### 1.3 Windows Firewall Portlarını Aç (Yönetici olarak PowerShell'de)
```powershell
netsh advfirewall firewall add rule name="Ray Head" protocol=TCP dir=in localport=6379 action=allow
netsh advfirewall firewall add rule name="Ray Dashboard" protocol=TCP dir=in localport=8265 action=allow
netsh advfirewall firewall add rule name="Ray Workers" protocol=TCP dir=in localport=10001-10100 action=allow
```

---

## ADIM 2 — Her Worker Bilgisayarda Yap

### 2.1 Projeyi Git ile İndir
```powershell
git clone https://github.com/<KULLANICI_ADI>/uav-iot-dc-env.git
cd uav-iot-dc-env

# Proje zaten varsa güncelle:
# git pull origin main
```

### 2.2 Sanal Ortam Oluştur ve Kütüphaneleri Kur
```powershell
# Sanal ortam oluştur
python -m venv .venv

# Aktifleştir
.\.venv\Scripts\Activate.ps1

# Pip güncelle
python -m pip install --upgrade pip

# Tüm bağımlılıkları kur
pip install -r requirements.txt
```

> **RTX 3060 için CUDA'lı PyTorch (daha hızlı eğitim):**
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2.3 Kurulumu Test Et
```powershell
python -c "import ray; import torch; print('Ray:', ray.__version__); print('CUDA:', torch.cuda.is_available())"
# Beklenen çıktı: Ray: 2.x.x  |  CUDA: True
```

### 2.4 Ray Cluster'a Katıl
```powershell
# HEAD_IP = ADIM 1.1'de öğrendiğin IP
ray start --address="192.168.1.50:6379" --num-cpus=22 --num-gpus=1
```

> `--num-cpus=22` = Intel Core Ultra 9'un toplam thread sayısı

---

## ADIM 3 — Otomatik Script ile Tek Komutta Kur

Yukarıdaki adımları otomatikleştirmek için proje klasöründe şunu çalıştır:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_worker.ps1
```

Script sırasıyla:
1. Python kurulumunu kontrol eder
2. `.venv` sanal ortamı oluşturur ve `requirements.txt`'i kurar
3. CUDA'lı PyTorch seçeneği sunar
4. Firewall portlarını otomatik açar
5. Head node IP'sini sorar ve `ray start` ile cluster'a bağlanır

---

## ADIM 4 — Kümenin Hazır Olduğunu Kontrol Et (Head'de)

```powershell
ray status
```

Örnek çıktı:
```
======== Autoscaler status ========
Node status
---------------------------------------------------------------
Healthy:
 1 node(s) with resources: {'CPU': 0.0, 'GPU': 0.0}    ← Head (koordinatör, çalışmaz)
 29 node(s) with resources: {'CPU': 22.0, 'GPU': 1.0}  ← Workers (tüm yük burada)
```

Ayrıca Ray Dashboard: `http://localhost:8265` adresinden de izlenebilir.

---

## ADIM 5 — Phase 1: Model HPO Başlat (Sadece Head'de)

```powershell
.\.venv\Scripts\Activate.ps1

# PPO — Model Hiperparametreleri (30 trial × 1000 iterasyon)
python scripts/tune_models.py --algo PPO --num-samples 30 --iterations 1000 --num-workers 14 --use-gpu True

# DQN — Model Hiperparametreleri
python scripts/tune_models.py --algo DQN --num-samples 30 --iterations 1000 --num-workers 14 --use-gpu True

# QJC (Baseline) — Tabular Model Hiperparametreleri
python scripts/tune_models.py --algo QJC --num-samples 30 --iterations 1000
```

> **Phase 1 bütçesi:**
> - Her trial → `iterations=1000` training adımı
> - ASHA: ilk 500 iterasyon garantili çalışır, sonrası erken kesilebilir
> - Her algo bağımsız çalıştırılır; sonuçlar `confs/tuned_configs.json`'a kaydedilir

---

## ADIM 6 — Phase 2: Ödül Ağırlığı Optimizasyonu (Phase 1 Bittikten Sonra)

> [!IMPORTANT]
> Phase 2'yi başlatmadan önce PPO, DQN ve QJC için Phase 1'in **tamamlanmış** olması gerekir.
> `confs/tuned_configs.json` içinde `ppo`, `dqn`, `qjc` anahtarları mevcut olmalıdır.

```powershell
# Phase 2 — AYRI script, AYRI dashboard
# W_SUCCESS ve W_COST'u 3 algoritmanın ortalamasına göre optimize eder
python scripts/tune_reward.py --num-samples 20 --iterations 500 --num-workers 14 --use-gpu True
```

**Phase 2 dashboard'unu aç:**
```powershell
# Ayrı terminalde:
python scripts/dashboard_server.py
# http://localhost:5000/reward_opt.html
```

> **Phase 2 bütçesi:**
> - Her trial içinde PPO + DQN + QJC **sıralı** eğitilir
> - Objective = mean(JSR_ppo, JSR_dqn, JSR_qjc)
> - ASHA yok — her trial tam çalışır
> - Sonuç `confs/tuned_configs.json["reward"]`'a kaydedilir


---

## Optimizasyon Sonucunu İzle

```powershell
# Ayrı bir terminal'de dashboard'u başlat:
python scripts/dashboard_server.py

# Tarayıcıda aç:
# http://localhost:8000/opt.html
```

---

## Sorun Giderme

| Sorun | Çözüm |
|---|---|
| `Could not find any running Ray instance` | Head'de `ray start --head --port=6379` çalıştırıldığından emin ol |
| Worker bağlanamıyor | Firewall portlarını (6379) kontrol et; aynı ağda mı? |
| `CUDA: False` çıkıyor | `--index-url cu121` ile PyTorch'u yeniden kur, NVIDIA sürücüsünü güncelle |
| `ExecutionPolicy` hatası | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` çalıştır |
| Worker beklenmedik kapandı | `ray start --address=...` komutunu tekrar çalıştır |
| Script yetki hatası | PowerShell'i **Yönetici olarak** aç |
| `git clone` hatası | Git kurulu değilse: https://git-scm.com/download/win |

---

## Oturum Sonu

```powershell
# Her worker'da çalıştır (PC kapatmadan önce):
ray stop

# Head'de çalıştır:
ray stop
```
