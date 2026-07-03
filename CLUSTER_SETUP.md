# Dağıtık Ray Cluster Kurulum Kılavuzu
## UAV-IoT Hiper-Parametre Optimizasyonu — Lab Bilgisayarları

---

## Mimari Özet

```
┌────────────────────────────────────────────────┐
│  SEN (Head Node)                               │
│  IP: 192.168.X.X                               │
│  • tune_models.py ve tune_reward.py çalıştırır │
│  • Optuna denemelerini yönetir                   │
│  • Dashboard: http://localhost:5000/opt.html   │
└────────────────┬───────────────────────────────┘
                 │ Ray Protocol (port 6379)
     ┌───────────┼───────────┐
     ▼           ▼           ▼
  Worker 1    Worker 2  ... Worker 12
  (CPU+GPU)  (CPU+GPU)      (CPU+GPU)
  Denemeleri paralel çalıştırır (STRICT_PACK ile her trial 1 makinede izole)
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

> [!NOTE]
> **Git Klonlamaya Gerek Yoktur!**
> Ray, kodların dağıtımını otomatik olarak `runtime_env` (çalışma ortamı) ile Head node'dan Worker'lara zipleyip taşır.
> Worker makinelerde sadece Python, PyTorch ve Ray'in kurulu olması yeterlidir.

İşçileri (Worker) kurmak için iki yöntemden birini seçebilirsiniz:

### Yöntem A: Otomatik Script İle (Önerilen)
1. Head makinesindeki `scripts/setup_worker.ps1` dosyasını bir flash bellek veya yerel ağ üzerinden worker makineye kopyalayın (herhangi bir geçici klasöre, örn: Masaüstü).
2. Worker makinede PowerShell'i açıp o klasöre gidin ve çalıştırın:
   ```powershell
   powershell -ExecutionPolicy Bypass -File setup_worker.ps1
   ```
3. Script sizden Head IP'sini isteyecek, otomatik sanal ortamı kuracak, bağımlılıkları yükleyecek ve Ray'e bağlanacaktır.

---

### Yöntem B: Manuel Kurulum (Script Kullanmadan)
Eğer script kullanmak istemiyorsanız, worker makinede boş bir klasör oluşturup şu komutları çalıştırın:

1. **Sanal Ortamı Kur ve Aktifleştir:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

2. **Gerekli Kütüphaneleri Kur:**
   ```powershell
   pip install "ray[rllib]>=2.53.0" pettingzoo==1.24.3 gymnasium torch numpy<2.0.0 pandas matplotlib seaborn optuna plotly
   ```
   *(RTX 3060/3080ti için CUDA PyTorch kurmak isterseniz: `pip install torch --index-url https://download.pytorch.org/whl/cu121`)*

3. **Ray Cluster'a Katıl (Worker olarak):**
   ```powershell
   # HEAD_IP = ADIM 1.1'de öğrendiğin IP
   ray start --address="192.168.1.50:6379" --num-cpus=22 --num-gpus=1
   ```


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
 12 node(s) with resources: {'CPU': 22.0, 'GPU': 1.0}  ← Workers (tüm yük burada)
```

Ayrıca Ray Dashboard: `http://localhost:8265` adresinden de izlenebilir.
Canlı sonuç paneli: `http://localhost:5000/opt.html` adresinden izlenir.

---

## ADIM 5 — Phase 1: Model HPO Başlat (Sadece Head'de)

```powershell
.\.venv\Scripts\Activate.ps1

# PPO — Model Hiperparametreleri (30 trial × 1000 iterasyon)
python scripts/tune_models.py --algo PPO --num-samples 30 --iterations 1000 --num-workers 10 --use-gpu True

# DQN — Model Hiperparametreleri
python scripts/tune_models.py --algo DQN --num-samples 30 --iterations 1000 --num-workers 10 --use-gpu True

# QJC (Baseline) — Tabular Model Hiperparametreleri
python scripts/tune_models.py --algo QJC --num-samples 30 --iterations 1000
```

> **Phase 1 bütçesi:**
> - Her trial → `iterations=1000` training adımı
> - ASHA: ilk 500 iterasyon garantili çalışır, sonrası erken kesilebilir
> - Her trial `STRICT_PACK` ile **tek bir makinede** (11 CPU, 1 GPU) izole şekilde çalışır.
> - Sonuçlar `confs/tuned_configs.json`'a kaydedilir.


---

## ADIM 6 — Phase 2: Ödül Ağırlığı Optimizasyonu (Phase 1 Bittikten Sonra)

> [!IMPORTANT]
> Phase 2'yi başlatmadan önce PPO, DQN ve QJC için Phase 1'in **tamamlanmış** olması gerekir.
> `confs/tuned_configs.json` içinde `ppo`, `dqn`, `qjc` anahtarları mevcut olmalıdır.

```powershell
# Phase 2 — AYRI script, AYRI dashboard
# W_SUCCESS ve W_COST'u 3 algoritmanın ortalamasına göre optimize eder
python scripts/tune_reward.py --num-samples 20 --iterations 500 --num-workers 10 --use-gpu True
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
> - Her trial yine `STRICT_PACK` ile tek bir makinede çalıştırılır (11 CPU, 1 GPU).
> - ASHA yok — her trial tam çalışır
> - Sonuç `confs/tuned_configs.json["reward"]`'a kaydedilir



---

## Optimizasyon Sonucunu İzle

```powershell
# Ayrı bir terminal'de dashboard'u başlat:
python scripts/dashboard_server.py

# Tarayıcıda aç:
# Phase 1 İzleme: http://localhost:5000/opt.html
# Phase 2 İzleme: http://localhost:5000/reward_opt.html
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
