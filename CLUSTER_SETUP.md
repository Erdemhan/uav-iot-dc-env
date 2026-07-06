# Dağıtık Ray Cluster Kurulum Kılavuzu (WSL2 Mimarisi)
## UAV-IoT Hiper-Parametre Optimizasyonu — Lab Bilgisayarları (Windows + WSL2)

---

## Mimari Özet

Windows işletim sistemlerinde dağıtık Ray kümesinin (multi-node cluster) doğrudan çalıştırılması resmi olarak desteklenmemekte ve bellek yönetimi (shared memory), süreç çatallama (forking) gibi konularda kararsızlıklara neden olmaktadır. Akademik düzeyde güvenilir, kararlı ve tekrarlanabilir (reproducible) sonuçlar elde etmek için laboratuvardaki bilgisayarlarda **WSL2 (Windows Subsystem for Linux - Ubuntu)** altyapısı tercih edilmiştir.

Makinelerin birbirleriyle haberleşebilmesi için WSL2 üzerinde **Mirrored Networking** (Aynalanmış Ağ Modu) aktifleştirilmektedir. Bu sayede WSL2, ana bilgisayarın (Windows) fiziksel ağ kartını doğrudan kullanır ve ağda Windows IP adresiyle görünür.

```
┌────────────────────────────────────────────────────────┐
│  SEN (Head Node) - Windows + WSL2                      │
│  Host IP: 192.168.X.X (Aynı IP WSL2 için de geçerli)   │
│  • WSL2 Ubuntu üzerinde koordinasyon                   │
│  • tune_models.py ve tune_reward.py çalıştırır        │
│  • Dashboard: http://localhost:8265                    │
│  • Canlı Sonuç Paneli: http://localhost:5000/opt.html  │
└─────────────────────────┬──────────────────────────────┘
                          │ Ray Protocol (port 6379)
              ┌───────────┼───────────┐
              ▼           ▼           ▼
           Worker 1    Worker 2  ... Worker 12
          (WSL2 Ubuntu)(WSL2 Ubuntu)(WSL2 Ubuntu)
          Denemeleri paralel çalıştırır (STRICT_PACK ile her trial 1 makinede izole)
```

---

## ADIM 1 — Ön Gereksinimler (Tüm Bilgisayarlarda Yapılacak)

Laboratuvardaki **tüm bilgisayarlarda (hem Head hem Worker'larda)** WSL2 kurulumu, Mirrored Networking yapılandırması ve Güvenlik Duvarı kurallarının ayarlanması gerekmektedir.

Bu adımı otomatik olarak gerçekleştirmek için bir PowerShell betiği hazırlanmıştır:

1. Windows PowerShell terminalini **Yönetici Olarak (Run as Administrator)** açın.
2. Proje dizinine giderek aşağıdaki komutu çalıştırın:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\setup_host_wsl.ps1
   ```
3. Script; Windows Güvenlik Duvarı kurallarını otomatik açacak, `%USERPROFILE%\.wslconfig` dosyasını oluşturarak Mirrored Networking modunu aktif edecek ve **bilgisayarın toplam RAM miktarını otomatik algılayıp 3GB güvenli Windows payı çıkararak kalan tüm bellek limitini (örn. 32GB RAM için 29GB) WSL2'ye tahsis edecektir.** Bu sayede paralel RLlib işçilerinin (workers) bellek yetersizliğinden çökmesi (OOM) tamamen engellenir.
4. **Not:** Eğer sistemde WSL2 ilk defa kurulduysa, script sizden bilgisayarı yeniden başlatmanızı isteyecektir. Yeniden başlattıktan sonra scripti bir kez daha çalıştırarak kurulumu tamamlayın.

---

## ADIM 2 — Koordinatör Bilgisayarda (Head Node) Yapılacaklar 

> [!IMPORTANT]
> **Dosya Sistemi Hızı ve Performans Uyarısı (Head Node İçin):** Proje dosyalarınızı doğrudan Windows dosya yolu üzerinden (`/mnt/c/...`) WSL2 içinde çalıştırırsanız, Windows ve Linux arasındaki disk geçişi sebebiyle eğitim hızı son derece yavaş olacaktır. **Eğitimlerin hızlı ve kararlı tamamlanması için proje klasörünü mutlaka WSL2'nin kendi yerel dosya sistemine (örneğin `/home/kullaniciadi/uav-iot-dc-env/` altına) kopyalayın veya orada yeniden klonlayın.** Tüm eğitim ve çalıştırma işlemlerini bu dizinde yapın.

### 2.1. IP Adresini Öğrenin
Windows CMD veya PowerShell üzerinde IP adresinizi öğrenin:
```powershell
ipconfig
# "IPv4 Address" satırındaki yerel ağ adresini not edin (örn: 192.168.1.50)
```

### 2.2. WSL Ubuntu Terminaline Giriş Yapın
Başlat menüsünden veya terminalden Ubuntu'yu açın:
```bash
wsl
```

### 2.3. Ray Head Node'u Başlatın
WSL2 terminali içinde koordinatör servisini başlatın:
```bash
# --num-cpus=0 --num-gpus=0 -> Koordinatör makinede trial çalıştırılmaz, tüm yük işçilere dağıtılır.
ray start --head --port=6379 --dashboard-host=0.0.0.0 --num-cpus=0 --num-gpus=0
```
Başarıyla çalıştığında konsolda şu çıktıyı göreceksiniz:
```text
Ray runtime started.
To add another node to this Ray cluster, run:
  ray start --address='192.168.1.50:6379'
```

---

## ADIM 3 — İşçi Bilgisayarlarda (Worker Nodes) Yapılacaklar

> [!TIP]
> **Neden WSL2 Ev Dizinine ( ~/ ) Taşıyoruz?** Betiği ve Python sanal ortamını (`.venv`) doğrudan WSL2'nin yerel Linux dosya sisteminde (`~/` yani `/home/kullaniciadi/`) çalıştırmak, Windows disk geçiş yavaşlığını önler ve disk okuma/yazma hızını maksimuma çıkararak eğitim performansını korur.

İşçi bilgisayarlarda projenin tamamını Git üzerinden klonlamaya gerek yoktur. Ray'in `runtime_env` yapısı, kodları Head makinesinden Worker'lara otomatik olarak zipleyip taşır. İşçi bilgisayarlarda sadece `setup_worker_wsl.sh` scriptini çalıştırmamız yeterlidir.

Bu scripti işçi bilgisayarın WSL2 (Ubuntu) ortamına aktarıp çalıştırmak için aşağıdaki yöntemlerden birini seçebilirsiniz:

### Yöntem A: Windows Üzerinden Kopyalama (Flash Bellek veya Ağ Paylaşımı)
1. Head (ana) bilgisayardaki `scripts/setup_worker_wsl.sh` dosyasını bir flash bellek veya ağ üzerinden işçi bilgisayardaki Windows ortamına kopyalayın (Örn: `C:\` dizinine veya Masaüstüne).
2. İşçi bilgisayarda WSL Ubuntu terminalini açın:
   ```bash
   wsl
   ```
3. WSL2, Windows disklerinizi `/mnt/` altında otomatik olarak görür. Dosyayı kendi WSL2 ev dizininize kopyalayın:
   ```bash
   # Eğer Windows C:\ dizinine kopyaladıysanız:
   cp /mnt/c/setup_worker_wsl.sh ~/
   
   # Eğer Masaüstüne kopyaladıysanız (KullaniciAdi kısmını güncelleyin):
   # cp /mnt/c/Users/KullaniciAdi/Desktop/setup_worker_wsl.sh ~/
   ```
4. Ev dizinine gidip betiği çalıştırın:
   ```bash
   cd ~/
   chmod +x setup_worker_wsl.sh
   ./setup_worker_wsl.sh
   ```

---

### Yöntem B: Doğrudan WSL2 Terminalinde Dosya Oluşturma (En Hızlı Yöntem)
Herhangi bir dosya taşıma işlemiyle uğraşmak istemiyorsanız:
1. İşçi bilgisayarda WSL Ubuntu terminalini açın (`wsl`).
2. Ev dizininizde boş bir betik dosyası açın:
   ```bash
   nano ~/setup_worker_wsl.sh
   ```
3. Projedeki `scripts/setup_worker_wsl.sh` dosyasının içeriğini kopyalayın ve terminale yapıştırın. `Ctrl+O` ardından `Enter` ile kaydedip `Ctrl+X` ile çıkın.
4. Betiği çalıştırın:
   ```bash
   chmod +x ~/setup_worker_wsl.sh
   ~/setup_worker_wsl.sh
   ```

---

### Yöntem C: Windows Gezgini (Explorer) ile Sürükle-Bırak
1. İşçi bilgisayarda WSL Ubuntu terminalini açın:
   ```bash
   wsl
   ```
2. Terminale şu komutu yazarak Ubuntu ev dizinini Windows Dosya Gezgini'nde açın:
   ```bash
   explorer.exe .
   ```
3. Açılan Windows klasörü içine `setup_worker_wsl.sh` dosyasını sürükleyip bırakın.
4. WSL2 terminaline dönüp betiği çalıştırın:
   ```bash
   chmod +x setup_worker_wsl.sh
   ./setup_worker_wsl.sh
   ```

---

## ADIM 4 — Kümenin Hazır Olduğunu Kontrol Edin (Head'de)

Koordinatör makinenin WSL terminalinde küme durumunu doğrulayın:
```bash
ray status
```
Ayrıca, tarayıcınızdan `http://localhost:8265` adresine giderek Ray Dashboard üzerinden bağlı işçileri ve kaynak durumlarını (CPU/GPU) canlı izleyebilirsiniz.

---

## ADIM 5 — Phase 1 ve Phase 2 Optimizasyonlarını Başlatma (Head WSL'de)

Sanal ortamı aktifleştirdikten sonra optimizasyon scriptlerini koordinatör terminalinden normal şekilde çalıştırabilirsiniz:

```bash
source .venv/bin/activate

# Phase 1: Model HPO (Örnek PPO araması)
python scripts/tune_models.py --algo PPO --num-samples 30 --iterations 1000 --num-workers 10 --use-gpu True --max-concurrent 4

# Phase 2: Ödül Ağırlığı Optimizasyonu (Tüm Phase 1 tamamlandıktan sonra)
python scripts/tune_reward.py --num-samples 20 --iterations 500 --num-workers 10 --use-gpu True
```

---

## Sorun Giderme (WSL2 Özel)

| Sorun | Neden | Çözüm |
|---|---|---|
| `Connection timed out` veya `Worker unable to connect` | Güvenlik duvarı engeli veya Mirrored Networking'in aktif olmaması. | 1. `wsl --shutdown` komutu ile WSL'i yeniden başlatın.<br>2. Windows host üzerinde firewall portlarının açık olduğunu denetleyin.<br>3. Windows IP'sini pingleyebildiğinizden emin olun. |
| GPU'lar WSL içinde görünmüyor | NVIDIA CUDA WSL sürücüsünün eksik veya güncel olmaması. | Windows tarafında güncel NVIDIA Game Ready / Studio Driver kurulu olmalıdır. WSL içinde `nvidia-smi` komutunu çalıştırarak GPU algılamasını doğrulayın. |
| `.wslconfig` dosyası algılanmıyor | Dosyanın uzantısının `.txt` olarak kalmış olması. | Windows dosya ayarlarından "Dosya adı uzantılarını göster" seçeneğini açıp dosya adının tam olarak `.wslconfig` olduğunu (sonunda `.txt` olmadığını) doğrulayın. |
| `Out of memory` veya yetersiz kaynak hatası | WSL'in çok fazla bellek tüketmesi. | `%USERPROFILE%\.wslconfig` içinde WSL'e maksimum bellek sınırı koyabilirsiniz (örn: `memory=16GB`). |

---

## Oturum Sonu

Küme çalışmasını sonlandırmak için:
```bash
# Her worker makinede:
ray stop

# Head makinesinde:
ray stop
```

---

## ADIM 7 — Temizlik ve Kaldırma (İşiniz Bittiğinde)

Çalışma tamamlandıktan sonra, laboratuvar bilgisayarlarını eski orijinal durumuna döndürmek, açılan portları kapatmak ve WSL2/Ubuntu kurulumlarını kaldırmak için aşağıdaki adımları izleyin:

1. Windows PowerShell terminalini **Yönetici Olarak** açın.
2. Proje dizinine giderek şu komutu çalıştırın:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\cleanup_host_wsl.ps1
   ```
3. Script; açılan güvenlik duvarı kurallarını silecek, `.wslconfig` içindeki ağ yapılandırmasını temizleyecek ve isteğinize bağlı olarak Ubuntu dağıtımını tamamen kaldırarak disk alanı boşaltacaktır.
