# Dağıtık Parametre Optimizasyonu Yol Haritası (Ray + Optuna)
**Tarih**: 2 Temmuz 2026  
**Durum**: ✅ Uygulama tamamlandı — Yerel doğrulama testleri başarılı  
**Hedef Donanım**: 30 × [Intel Core Ultra 9 CPU (16 Cores/22 Threads) + NVIDIA RTX 3060 GPU]

Bu rapor, laboratuvardaki 30 adet bilgisayardan oluşan yerel bir Ray kümesi üzerinde gerçekleştirilecek iki aşamalı (iki fazlı) hiperparametre ve ödül fonksiyonu arama sürecinin tüm detaylarını ve uygulama planını sunmaktadır.

---

## 1. Donanım ve Paralelleştirme Stratejisi

Mevcut eğitim süreleri tek bilgisayarda (kısıtlı kaynak kullanımı ile) yaklaşık 7-8 saattir. Bu süreyi laboratuvar donanımının tamamını kullanarak optimize edeceğiz:

1. **CPU Dağıtımı (`num_workers`)**:
   - Intel Ultra 9 işlemcilerin yüksek çekirdek sayısından yararlanarak her deneme için `num_workers = 14` olarak ayarlanacaktır. Simülasyon veri toplama süreci CPU üzerinde 14 paralel kanaldan yürütülecektir.
2. **GPU Hızlandırması (`use_gpu`)**:
   - RTX 3060 ekran kartlarının CUDA çekirdekleri sinir ağı (PPO, DQN, PPO-LSTM) güncellemeleri için aktif edilecektir (`use_gpu = True`).
   - Her deneme GPU'nun yarısını (`num_gpus = 0.5`) kullanacak; böylece bir bilgisayar iki denemeyi paralel destekleyebilecektir.
   - **GPU olmayan makinelerde otomatik CPU fallback**: `torch.cuda.is_available()` kontrolü ile CUDA yoksa uyarı verilerek CPU'ya geçiş yapılır.
3. **Hız Kazancı Tahmini**:
   - Çekirdek sayısının artırılması ve GPU desteği sayesinde tek bir tam eğitim süresinin **1.5 saat veya daha az bir süreye** düşmesi beklenmektedir.
4. **Dağıtık Ölçekleme (30 PC)**:
   - 30 bilgisayarlık Ray kümesinde aynı anda **30 bağımsız deneme paralel olarak koşturulacaktır.**
   - 90 denemeden oluşan bir arama süreci, sadece **3 turda (yaklaşık 3.5 - 4.5 saat)** tamamen sonlanacaktır.

---

## 2. İki Aşamalı Optimizasyon Mimarisi ve Hedef (Objective) Fonksiyonları

Karşılaştırmanın bilimsel geçerliliğini ve hakem savunmasını en üst düzeye çıkarmak için iki aşamalı bir optimizasyon metodu izlenecektir. Her denemenin (trial) sonunda, kararlılık ve dayanıklılığı garanti etmek adına **30 tohumlu dayanıklılık testi (30-seed robustness evaluation)** çalıştırılacak ve Optuna'ya döndürülecek skorlar bu 30 testin ortalamasından alınacaktır.

---

### FAZ 1: Model Hiperparametrelerinin Birleşik Optimizasyonu
Bu aşamada çevre ve ödül parametreleri sabitlenecektir (`W_SUCCESS = 0.8`, `W_TRACKING = 0.2`, `W_COST = 0.03`). Her algoritma için en iyi çalışan sinir ağı yapısı ve öğrenme hızları **birleşik (joint)** olarak aranacaktır.

* **Hedef (Objective) Fonksiyonu**: **30 tohum üzerindeki Ortalama Ödül (30-seed Mean Episode Reward)** değerini maksimize etmek.
* **Nedeni**: Modelin hem İHA'yı iyi takip etmesini (Tracking), hem başarılı karıştırma yapmasını (JSR) hem de gereksiz güç harcamamasını (Power Cost) dengeli şekilde öğrenen en iyi sinir ağını bulabilmek için ödül fonksiyonunun tamamına odaklanılmalıdır.

#### Arama Uzayları:

* **PPO Arama Uzayı (3 Parametre)**:
  - `learning_rate` (LR): `1e-5` ile `1e-3` arasında sürekli (logaritmik)
  - `gamma`: `0.85` ile `0.99` arasında sürekli (düzgün)
  - `architecture`: 18 önceden tanımlı sinir ağı mimarisi (kategorik)

* **DQN Arama Uzayı (4 Parametre)**:
  - `learning_rate` (LR): `1e-5` ile `1e-3` arasında sürekli (logaritmik)
  - `gamma`: `0.85` ile `0.99` arasında sürekli (düzgün)
  - `architecture`: 18 önceden tanımlı sinir ağı mimarisi (kategorik)
  - `target_network_update_freq`: `[200, 500, 1000, 2000]` (DQN kararlılığı için kritik)

* **PPO-LSTM Arama Uzayı (5 Parametre)**:
  - `learning_rate` (LR): `1e-5` ile `5e-4` arasında sürekli (logaritmik)
  - `gamma`: `0.85` ile `0.99` arasında sürekli (düzgün)
  - `architecture`: 18 önceden tanımlı sinir ağı mimarisi (kategorik) (LSTM öncesi MLP)
  - `lstm_cell_size`: `[128, 256, 512]` (LSTM hücre boyutu)
  - `max_seq_len`: `[10, 20, 30]` (Geriye dönük zaman penceresi)

* **QJC (Tabular Q-Learning Baseline - 4 Parametre)**:
  - `tau_0`: `1e-5` ile `1e-3` arasında sürekli (logaritmik)
  - `gamma`: `0.85` ile `0.99` arasında sürekli (düzgün)
  - `temp_xi`: `1.0` ile `10.0` arasında sürekli (Softmax sıcaklığı)
  - `mu_offset`: `1.0` ile `2.0` arasında sürekli
  - *Not*: QJC sinir ağı eğitmediği için, bu arama ana kümede saniyeler içinde tamamlanacaktır.

#### Sinir Ağı Mimarisi Arama Uzayı (18 Seçenek)

Eski tasarımda `num_layers × layer_size` kombinasyonu kullanılıyordu (örn. `num_layers=3`, `layer_size=256` → `[256, 256, 256]`). Bu yaklaşımda tüm katmanlar aynı boyutta olduğundan `[128, 256, 512]` gibi zengin mimarilere ulaşmak mümkün değildi.

Yeni tasarımda tek bir `architecture` kategorik parametresi kullanılmaktadır. 18 mimari dört tipte gruplanmıştır:

| Tip | Örnekler | Açıklama |
|---|---|---|
| **Sığ (1 katman)** | `[128]`, `[256]`, `[512]` | Basit problemler için hızlı yakınsama |
| **Homojen (2-3 katman)** | `[256, 256]`, `[512, 512, 512]` | Klasik derin ağ |
| **Genişleyen (Expanding)** | `[128, 256]`, `[256, 512]`, `[128, 256, 512]` | Düşük seviyeden yüksek soyutlamaya |
| **Daralan/Funnel** | `[256, 128]`, `[512, 256]`, `[512, 256, 128]` | Bilgi sıkıştırma/kompresyon |
| **Bottleneck** | `[256, 128, 256]`, `[512, 256, 512]` | Özellik çıkarımı + tekrar genişleme |

**Faz 1 Çıktısı**: Her algoritma için en optimum ağ yapısı ve öğrenme parametreleri `{project_root}/confs/tuned_configs.json` dosyasına kaydedilecektir.

---

### FAZ 2: Çok Amaçlı Ödül Fonksiyonu Optimizasyonu
Faz 1'de bulunan en iyi model parametreleri dondurulacaktır. Bu aşamada, her algoritma kendi optimize edilmiş ayarlarını yükleyecek ve ortak çevre/ödül ağırlıkları aranacaktır:

* **Ödül Yapısı**:
  `Reward = W_success * Jamming_Success + W_tracking * Tracking_Acc - W_cost * Power_Cost`
* **Hedef (Objective) Fonksiyonu**: **30 tohum üzerindeki Ortalama Karıştırma Başarı Oranı (30-seed Mean Jamming Success Rate - JSR)** değerini maksimize etmek.
* **Nedeni (Çok Kritik)**: Ağırlıklar (`W_success` ve `W_cost`) değiştikçe ödülün sayısal ölçeği bozulur ve farklı parametrelerin ödülleri birbiriyle kıyaslanamaz hale gelir. Bu yüzden ağırlık aramasında fiziksel metrik olan **JSR** başarısını hedef fonksiyon yapmak en adil ve tutarlı çözümdür.
* **Aranacak Parametreler**:
  - `W_success`: `0.5` ile `0.95` arasında sürekli (JSR önceliği)
  - `W_cost`: `0.005` ile `0.1` arasında sürekli (Güç koruma önceliği)
  - `W_tracking`: Dinamik olarak `1.0 - W_success` hesaplanacaktır.
* **Adillik Garantisi**: Tüm modeller kendi en optimum parametreleriyle bu ortak ödülü çözmeye çalışacağı için, elde edilen karşılaştırma sonuçları akademik olarak **%100 adil ve hatasız** olacaktır.

---

## 3. Dağıtık Altyapı ve Bağlantı Kurulumu

Laboratuvardaki 30 bilgisayarı birbirine bağlamak için ayrıntılı kurulum adımları `CLUSTER_SETUP.md` dosyasında belgelenmiştir. Özet:

### A. Ana Bilgisayarda (Head Node) Yapılacaklar:
1. IP adresini öğren: `ipconfig`
2. Firewall portlarını aç (6379, 8265, 10001-10100)
3. Ray Head'i başlat:
   ```powershell
   ray start --head --port=6379 --dashboard-host=0.0.0.0
   ```

### B. İşçi Bilgisayarlarda (Worker Nodes - Diğer 29 PC) Yapılacaklar:
1. Projeyi Git ile indir:
   ```powershell
   git clone https://github.com/<KULLANICI_ADI>/uav-iot-dc-env.git
   cd uav-iot-dc-env
   ```
2. Otomatik kurulum scriptini çalıştır (Python kurulumu, `.venv`, kütüphaneler, bağlantı):
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\setup_worker.ps1
   ```
   Script Head Node IP'sini sorar ve `ray start --address=<IP>:6379 --num-cpus=22 --num-gpus=1` ile kümeye katılır.

3. Küme durumunu kontrol et (Head'de):
   ```powershell
   ray status
   ```

> Ayrıntılı adımlar ve sorun giderme tablosu: [CLUSTER_SETUP.md](CLUSTER_SETUP.md)

---

## 4. Optimizasyon Süreci Görselleştirme ve Veri Kayıt Sistemi

Optimizasyon sonuçlarını hem canlı takip etmek, hem kalıcı arşivlemek hem de makale yazımında kullanabileceğimiz veri ve grafikleri elde etmek için gelişmiş bir kayıt sistemi kurulmuştur:

1. **Fiziksel Veri Kaydı (Disk Üzerine)**:
   - Arama süreci tamamlandığında tüm süreç verileri `{run_dir}/optuna/` dizini altına kaydedilir:
     - `optuna_trials.json`: Tüm denemelerin ID, parametre değerleri, ara skorları, durumları ve sürelerini içeren detaylı veri tabanı.
     - `best_params.json`: Bulunan en optimum parametre setinin ve ulaşılan en iyi skorun özeti.

2. **Otomatik Grafik Üretimi**:
   - Optuna'nın `study` nesnesi üzerinden şu yüksek kaliteli grafikler otomatik çizilip `{run_dir}/optuna/` klasörüne kaydedilir:
     - `optimization_history.png` (Denemeler boyunca en iyi ödül değerinin gelişim eğrisi)
     - `param_importances.png` (Hangi parametrenin başarı üzerinde ne kadar etkili olduğunu gösteren bar grafiği)
     - `parallel_coordinate.png` (Parametreler arası çok boyutlu ilişki grafiği)
     - `slice_plot.png` (Parametrelerin tekil bazda ödül değerine etkisini gösteren saçılım grafiği)
   - **Not**: `architecture` parametresi liste tipinde olduğundan (örn. `[128, 256, 512]`), `parallel_coordinate` ve `slice_plot` grafikleri çizilirken liste değerleri otomatik olarak string'e dönüştürülür; bu sayede Optuna'nın hash kısıtlaması aşılmaktadır.

3. **Web Arayüz Entegrasyonu (Dashboard)**:
   - `dashboard_server.py` sunucusu `/opt.html` adresinde yeni bir rapor sayfası sunar.
   - Bu sayfa diskteki JSON ve PNG dosyalarını okuyarak 5 saniyelik polling ile canlı güncellenir:
     - Toplam/tamamlanan/budanan deneme sayıları
     - En yüksek skora göre sıralanabilir interaktif deneme tablosu
     - Optuna grafikleri sekme geçişli olarak görüntüleme
   - Dashboard yeni bir run başlatıldığında `dashboard_active_run.txt` dosyası güncellenerek otomatik olarak o run'ı hedefler.

---

## 5. Optimizasyonu Başlatma Komutları

```powershell
# Sanal ortamı aktifleştir
.\.venv\Scripts\Activate.ps1

# PPO Phase 1 — Model Hiperparametreleri
python scripts/tune_models.py --algo PPO --num-samples 30 --iterations 100 --num-workers 14 --use-gpu True

# PPO Phase 2 — Reward Ağırlıkları (Phase 1 tamamlandıktan sonra)
python scripts/tune_models.py --algo PPO --phase 2 --num-samples 20 --iterations 100 --num-workers 14 --use-gpu True

# DQN Phase 1
python scripts/tune_models.py --algo DQN --num-samples 30 --iterations 100 --num-workers 14 --use-gpu True

# QJC (Baseline)
python scripts/tune_models.py --algo QJC --num-samples 30 --iterations 100

# Dashboard (ayrı terminalde)
python scripts/dashboard_server.py
# → http://localhost:8000/opt.html
```

---

## 6. Dosyalar ve Uygulama Durumu

| Dosya | Durum | Açıklama |
|---|---|---|
| `scripts/tune_models.py` | ✅ Tamamlandı | Optuna TPE + ASHA Scheduler, PlacementGroupFactory, 30-seed eval, architecture kategorik arama |
| `scripts/setup_worker.ps1` | ✅ Tamamlandı | Worker bilgisayar otomatik kurulum scripti (venv + pip + ray join) |
| `scripts/dashboard/opt.html` | ✅ Tamamlandı | Canlı Optuna trial izleme paneli, grafik görüntüleyici |
| `scripts/dashboard_server.py` | ✅ Tamamlandı | `/api/optuna`, `/api/plots/optuna/`, `/opt.html` endpoint'leri eklendi |
| `scripts/dashboard/index.html` | ✅ Tamamlandı | Header'a "Optimization Panel" butonu eklendi |
| `confs/tuned_configs.json` | ⏳ Run sonrası oluşur | Faz 1 en iyi parametrelerinin kalıcı konfigürasyon dosyası |
| `CLUSTER_SETUP.md` | ✅ Tamamlandı | Lab küme kurulum kılavuzu |
