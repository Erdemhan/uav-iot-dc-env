# PROJE GELİŞİM RAPORU VE TEKNİK DOKÜMANTASYON

**Proje Başlığı:** Nesnelerin İnterneti Tabanlı İHA Uygulamalarında Güvenlik Hassasiyetli Akıllı Yöntemlerin Geliştirilmesi
**Rapor Tarihi:** 02.02.2026 01:07
**Versiyon:** 1.0.0 (Başlangıç Sürümü)

---

## 1. GİRİŞ VE SİSTEM GENEL BAKIŞI

Bu proje, Doktora Tezi kapsamında İnsansız Hava Araçları (İHA) ve Nesnelerin İnterneti (IoT) ağlarının entegre çalıştığı senaryolarda, siber güvenlik tehditlerinin (özellikle Jamming saldırıları) etkilerini analiz etmek ve bunlara karşı dayanıklı akıllı yöntemler geliştirmek amacıyla tasarlanmıştır.

Geliştirilen simülasyon ortamı, literatürdeki standartlara uygun olarak Python tabanlı, modüler, genişletilebilir ve bilimsel geçerliliği olan matematiksel modellere dayalı bir altyapıya sahiptir. OpenAI Gymnasium arayüzü benimsenerek, ileride Derin Pekiştirmeli Öğrenme (Deep Reinforcement Learning - DRL) algoritmalarının entegrasyonuna hazır hale getirilmiştir.

## 2. SİSTEM MİMARİSİ

Simülasyon altyapısı, Nesne Yönelimli Programlama (OOP) prensipleri çerçevesinde, her biri spesifik bir görevi üstlenen gevşek bağlı (loose-coupled) modüllerden oluşmaktadır.

### 2.1. Konfigürasyon Modülleri (`confs/`)

*   **`confs/config.py` (Sistem Konfigürasyonu):** Sistemin fiziksel bant genişliği, frekans, gürültü seviyesi gibi temel donanım parametrelerini tutar.
*   **`confs/env_config.py` (Ortam ve Senaryo Konfigürasyonu):** Simülasyonun senaryo parametrelerini (Düğüm sayısı, alan boyutu, adım süresi, saldırgan konumu vb.) barındırır. Bu ayrım sayesinde fiziksel altyapı değiştirilmeden farklı senaryolar test edilebilir.

### 2.2. Çekirdek Modüller (`core/`)
*   **`core/physics.py` (Fizik Motoru):** Sistemin "stateless" (durumsuz) matematiksel hesaplama çekirdeğidir. Haberleşme kanalı (Path Loss, SINR, Shannon Kapasitesi) ve enerji tüketim modelleri (İHA uçuş gücü, IoT iletim enerjisi) burada saf fonksiyonlar (pure functions) olarak implemente edilmiştir.
*   **`simulation/entities.py` (Varlık Modellemesi):** Simülasyon dünyasındaki aktörlerin (İHA, IoT Düğümü, Saldırgan) davranışlarını ve durumlarını modelleyen sınıfları içerir.
    *   *Miras Yapısı:* `BaseEntity` -> `MobileEntity` / `TransceiverEntity` -> `UAVAgent` / `IoTNode` şeklinde hiyerarşik bir yapı kurgulanmıştır.

### 2.2. Simülasyon ve Ortam
*   **`simulation/pettingzoo_env.py` (PettingZoo Ortamı):** `UAV_IoT_PZ_Env` sınıfı, simülasyonun çoklu ajan (multi-agent) yapısını destekleyen `pettingzoo.utils.ParallelEnv` tabanlı ortamdır. İHA, Saldırgan ve her bir IoT düğümü ayrı birer ajan olarak modellenmiştir.
*   **`simulation/controllers.py` (Kural Tabanlı Kontrolcüler):** İHA gibi belirli kurallara (örn. navigasyon) dayalı hareket eden ajanların davranış mantığını kapsüller.
*   **`main.py` (Yürütücü):** Simülasyon döngüsünü PettingZoo API'sine uygun şekilde (sözlük yapılı aksiyon/gözlem) yönetir. `confs/config.py` içerisindeki `SIMULATION_DELAY` parametresi ile simülasyon akış hızı kontrol edilebilir.

### 2.3. Veri Yönetimi ve Analiz

*   **`core/logger.py` (Telemetri Kaydı):** Simülasyon sırasında üretilen ham verileri (konumlar, SINR değerleri, enerji tüketimleri) periyodik olarak CSV formatında kayıt altına alır.

*   **`visualization/visualizer.py` (Görsel Analiz):** Simülasyon sonrası elde edilen verileri işleyerek akademik kalitede (SCIE standartlarında) grafikler ve yörünge analizleri üretir.

### 2.5. Kullanılan Altyapı ve Teknolojiler

Projenin geliştirilmesinde, akademik standartlara uygunluk ve yüksek performans gereksinimleri gözetilerek aşağıdaki açık kaynaklı kütüphaneler kullanılmıştır:

*   **PettingZoo (Python):** Çoklu ajan (Multi-Agent) takviyeli öğrenme ortamları için endüstri standardı olan bu kütüphane, projemizin temel yapı taşıdır. `ParallelEnv` API'si kullanılarak, İHA, Jammer ve IoT düğümlerinin eş zamanlı olarak etkileşime girdiği, ölçeklenebilir ve oyun teorik analizlere uygun bir simülasyon ortamı oluşturulmuştur.
*   **OpenAI Gymnasium:** PettingZoo'nun üzerine inşa edildiği temel API yapısıdır. Ajanların durum-aksiyon uzaylarının (Box, Discrete) tanımlanmasında standartları belirler.
*   **NumPy:** Yüksek performanslı vektörel matematik işlemleri için kullanılmıştır. Fizik motorundaki (`physics.py`) sinyal gücü, SINR ve enerji hesaplamaları, döngüler yerine NumPy vektör operasyonları ile optimize edilerek simülasyon hızı artırılmıştır.
*   **Matplotlib:** Simülasyon verilerinin görselleştirilmesi ve analiz grafiklerinin (`trajectory.png`, `metrics_analysis.png`) oluşturulması için kullanılmıştır.
*   **Pandas:** Simülasyon loglarının (`history.csv`) işlenmesi, filtrelenmesi ve zaman serisi analizlerinin yapılması amacıyla veri manipülasyonu için tercih edilmiştir.

### 2.6. Mevcut Simülasyon Senaryosu (v1.0.0)

Bu sürümde kullanılan senaryo, temel sistem dinamiklerini doğrulamak amacıyla oluşturulmuş "Baseline" (Taban) senaryosudur.

*   **Operasyonel Alan:** 1000m x 1000m boyutlarında 2 boyutlu düzlem.
*   **İHA Davranışı (Blue Team):**
    *   **Waypoint Navigasyonu:** İHA, simülasyon alanındaki IoT düğümlerini sırasıyla (Node 0 -> Node 1 -> ...) ziyaret eder.
    *   Bir düğüme ulaştığında (mesafe < 10m), bir sonraki düğümü hedef olarak belirler. Bu sayede tüm sensörlerden yakından veri toplamayı hedefler.
*   **Saldırgan Davranışı (Red Team):**
    *   Alan merkezine yakın sabit bir konumda bulunmaktadır.
    *   **Stokastik Jamming:** Her zaman adımında 0 ile 2 Watt arasında rastgele bir güç seviyesi belirleyerek iletişim kanalını karıştırmaktadır.
*   **Ağ Dinamikleri:**
    *   5 adet IoT düğümü alana rastgele dağıtılmıştır.
    *   Fizik motoru her adımda anlık SINR değerini hesaplar.
    *   **Kesinti Kriteri:** SINR < 1.0 (0 dB) durumunda iletişim kopar ve düğümün "Bilgi Yaşı" (AoI) artmaya başlar.

---

## 3. MATEMATİKSEL MODELLER

Sistemin gerçekçiliği, tez önerisinde belirtilen aşağıdaki modellerin entegrasyonu ile sağlanmıştır:

### 3.1. Haberleşme Kanalı (Air-to-Ground)
Hava-Yer kanalı için serbest uzay yol kaybı modeli temel alınmış ve aşağıdaki SINR (Sinyal-Gürültü ve Girişim Oranı) denklemi kullanılmıştır:

$$ SINR = \frac{P_{rx}}{N_0 + I_{jam}} $$

Burada $P_{rx}$ alınan güç, $N_0$ termal gürültü ve $I_{jam}$ saldırganın oluşturduğu girişim gücüdür. Veri hızı ise Shannon-Hartley teoremi ile hesaplanmaktadır (Denklem 245, 248).

### 3.2. Enerji Modelleri
*   **İHA:** Döner kanatlı İHA enerji tüketimi, ileri uçuş hızı ($v$) ve askıda kalma (hover) durumlarını içeren kapsamlı bir aerodinamik model ile hesaplanmaktadır (Denklem 263, 272).
*   **IoT Düğüm:** Veri toplama, şifreleme ve iletim süreçlerinin toplam enerji maliyeti modellenmiştir (Denklem 288).

---

## 4. ANALİZ VE GÖRSELLEŞTİRME YÖNTEMLERİ

Sistem performansını değerlendirmek ve senaryo çıktılarını yorumlamak için `visualizer.py` modülü tarafından üretilen, SCIE makale formatına uygun iki temel grafik seti kullanılmaktadır.

### 4.1. Yörünge ve Olay Analizi (`trajectory.png`)
Bu grafik, simülasyonun mekansal (spatial) analizini ve ağ topolojisini gösterir.

*   **İçerik:**
    *   **İHA Yörüngesi (Mavi Çizgi):** İHA'nın görev süresince izlediği fiziksel rotayı gösterir.
    *   **IoT Düğümleri (Renkli Kareler):** Sahadaki sabit sensör düğümlerinin konumlarını; her biri kendine özgü renkle (Node 0 Mavi, Node 1 Turuncu vb.) gösterir.
    *   **Başarılı İletişim (Renkli Noktalar):** İHA'nın iletişim kurduğu anları, ilgili düğümün renginde işaretler. Aynı anda çoklu bağlantı varsa noktalar kaydırılarak (offset) çizilir.
    *   **Karıştırma Kesintileri (Kırmızı Çarpı):** İletişimin sadece **Jamming kaynaklı** kesildiği (ve başka hiçbir aktif bağlantının olmadığı) anları gösterir.
    *   **Mesafe Kesintileri (Gri Nokta):** İletişimin sadece mesafe nedeniyle koptuğu anları gösterir.
    *   **Saldırgan Konumu (Kırmızı 'X'):** Jamming kaynağının konumunu işaret eder.

*   **Yorumlama:**
    *   Renkli noktaların yoğunluğu, İHA'nın hangi düğümle iletişimde olduğunu netleştirir.
    *   Kırmızı çarpıların sadece "tam kesinti" anlarında çıkması, görsel analizi sadeleştirir.

### 4.2. Metrik Analizi (`metrics_analysis.png`)
Bu grafik, sistemin zamansal (temporal) performansını üç alt panelde inceler.

1.  **Sinyal Kalitesi ve Saldırı Gücü (Üst Panel):**
    *   *Sol Eksen (Mavi):* Anlık SINR (dB) değerini gösterir.
    *   *Sağ Eksen (Kırmızı):* Saldırganın jamming gücünü (Watt) gösterir.
    *   *Yorum:* Kırmızı eğrinin yükseldiği (Saldırı gücü arttığı) anlarda, mavi eğrinin (SINR) düşüşü gözlemlenir. Eşik değerin (Örn: 0 dB) altına inilen anlar, iletişimin koptuğu anlardır.
2.  **Bilgi Yaşı - Age of Information (Orta Panel):**
    *   Verinin tazeliğini (Freshness) ifade eder.
    *   *Yorum:* Grafik "testere dişi" (sawtooth) formundadır. AoI değerinin lineer olarak arttığı (yukarı tırmandığı) süreler, verinin alınamadığı kesinti süreleridir. Ani düşüşler (sıfırlanma), başarılı paket alımını gösterir. Tepe noktalarının yüksekliği, ağdaki gecikme performansının en kötü durumunu gösterir.
3.  **Enerji Tüketimi (Alt Panel):**
    *   İHA'nın toplam kümülatif enerji tüketimini gösterir.
    *   *Yorum:* Eğimin (slope) artması, İHA'nın daha fazla güç tükettiği manevraları veya yüksek hızları işaret eder.

### 4.3. İletişim İstatistikleri (`advanced_metrics.png`)
Bu grafik seti, her bir IoT düğümünün operasyonel performansını detaylandırır.

1.  **Toplam Başarılı İletişim Süresi (Üst Panel):**
    *   Her bir düğümün (Node 0, Node 1...) simülasyon boyunca toplam kaç saniye boyunca İHA ile başarılı bağlantı kurduğunu gösteren sütun grafiğidir.
    *   *Kırmızı Kesik Çizgi:* Tüm düğümlerin ortalama başarılı iletişim süresini gösterir.
2.  **Maksimum Kesintisiz İletişim (Alt Panel):**
    *   Her bir düğümün bağlantı kopmadan (AoI resetlenmeden) sürdürebildiği en uzun iletişim süresini (Streak) gösterir.
    *   Bu metrik, sistemin kararlılığını ve jamming'in iletişim sürekliliği üzerindeki etkisini ölçmek için kritiktir.

### 4.4. Dashboard Analizi
Simülasyon tamamlandığında, yukarıdaki tüm analizler (`Trajectory`, `Metrics`, `Advanced Stats`) tek bir **"Dashboard"** penceresinde operatöre sunulur. Bu sayede simülasyon sonuçlarına bütüncül (holistic) bir bakış açısı sağlanır.

---

## 5. GELİŞİM GÜNLÜĞÜ (CHANGE LOG)

### [02.02.2026 01:07] - Başlangıç Sürümü (v1.0.0)
**Yapılan Değişiklikler:**
1.  **Altyapı Kurulumu:** Tüm temel modüller (`main`, `config`, `physics`, `entities`, `env`) sıfırdan kodlandı.
2.  **Model Entegrasyonu:** Tez önerisindeki matematiksel formüller `physics.py` içerisine fonksiyonel olarak gömüldü.
3.  **Senaryo Tasarımı:** İHA'nın dairesel devriye gezdiği ve saldırganın rastgele jamming uyguladığı temel "Baseline" senaryo kurgulandı.
4.  **Hata Düzeltmeleri:**
    *   `pandas` ve `numpy` kütüphanelerindeki sürüm uyumsuzluğu giderildi.
    *   `entities.py` içindeki çoklu kalıtım (Diamond Problem) yapısında `super()` kullanımı yerine açık sınıf çağrıları (Explicit init calls) yapılarak `__init__` hatası düzeltildi.
5.  **Loglama Sistemi:** Simülasyon verilerinin anlık kaydı için `SimulationLogger` sınıfı geliştirildi.
6.  **Görselleştirme Modülü:** Sonuçların analiz için CSV verilerini okuyup grafik üreten `visualizer.py` modülü sisteme eklendi.

**Amaç:**
Tez çalışmasının simülasyon gereksinimlerini karşılayan, doğrulanmış (verified) ve veri üretebilen kararlı bir sürümün oluşturulması.

### [02.02.2026 01:40] - Otomasyon Sürümü (v1.1.0)
**Yapılan Değişiklikler:**
1.  **Tam Otomasyon:** `main.py` güncellenerek simülasyon bitiminde `visualizer` modülünün otomatik tetiklenmesi sağlandı.
2.  **Görselleştirme İyileştirmesi:** İHA rotası üzerinde başarılı/başarısız iletişim ve IoT düğüm konumları eklendi.
3.  **Veri Zenginleştirme:** `environment.py` loglarına düğüm konumları ve bağlantı durumu eklendi.

### [02.02.2026 01:42] - Parametre Revizyonu (v1.1.1)
**Yapılan Değişiklikler:**
1.  **Saldırgan Gücü Revizyonu:** `MAX_JAMMING_POWER` 2.0W -> 1.0W düşürüldü.
2.  **Konfigürasyon Ayrımı:** Ortam parametreleri `core/env_config.py` dosyasına taşınarak fiziksel parametrelerden (`core/config.py`) izole edildi.

### [02.02.2026 02:40] - Navigasyon ve Görselleştirme Paketi (v1.2.0)
**Yapılan Değişiklikler:**
1.  **Waypoint Navigasyonu:** İHA'nın dairesel uçuşu yerine, düğümleri sırasıyla (0->N) ziyaret ettiği "Target Tracking" modeline geçildi.
2.  **Gerçekçi Jamming Alanı:** Anlık görselleştirmede (`visualization.py`), basit daire yerine SINR < 0dB olan bölgeleri tarayan **Dinamik Kontur Grafiği** eklendi.
3.  **Gelişmiş Trajectory Analizi:**
    *   **Düğüm Renkleri:** Her düğüm benzersiz bir renge (tab10) atandı.
    *   **Offset Logic:** Çoklu bağlantı durumunda noktaların üst üste binmesi, koordinat kaydırma (offset) yöntemiyle engellendi.
    *   **Akıllı Filtreleme:** Eğer İHA en az bir düğüme bağlıysa, görsel kirliliği önlemek için diğer düğümlerin Jammed/Out-of-Range sembolleri gizlendi.

### [02.02.2026 03:00] - Mimari Refaktör (v1.2.1)
**Yapılan Değişiklikler:**
1.  **Dizin Yapısı Düzenlemesi:** Konfigürasyon dosyaları (`config.py`, `env_config.py`) `core/` klasöründen yeni oluşturulan `confs/` klasörüne taşındı.
2.  **Modülarite:** Konfigürasyon ve Çekirdek mantığı birbirinden tamamen izole edildi. Tüm modüller (`simulation`, `visualization`, `core`) yeni yapıya uygun olarak güncellendi.

### [06.02.2026 15:55] - Multi-Agent Mimari Göçü (v1.3.0)
**Yapılan Değişiklikler:**
1.  **PettingZoo Geçişi:** OpenAI Gymnasium (`gym.Env`) yapısından PettingZoo (`ParallelEnv`) yapısına geçildi. Bu sayede simülasyon, tek ajanlı yapıdan çok ajanlı (Multi-Agent) yapıya evrildi.
2.  **Ölçeklenebilir Ajan Tanımı:** İHA (`uav_0`), Jammer (`jammer_0`) ve IoT Düğümleri (`node_0`,`node_1`,...) artık sistemde birer "ajan" olarak tanımlanmıştır.
3.  **Kural Tabanlı Kontrolcü:** İHA'nın navigasyon mantığı, çevre kodundan (`env.step`) çıkarılarak harici bir kontrolcü sınıfına (`UAVRuleBasedController`) taşındı. Bu, İHA'nın ileride farklı politika algoritmalarıyla (RL vb.) değiştirilebilmesine olanak tanımaktadır.
4.  **Ağ ve Oyun Teorisi Altyapısı:** Yeni mimari, oyun teorik yaklaşımların (örn. Jammer ve İHA arasındaki Stackelberg oyunları) uygulanabilmesi için gerekli olan eş zamanlı aksiyon (simultaneous action) altyapısını sağlamaktadır.

### [06.02.2026 16:15] - Gelişmiş Metrikler ve Dashboard (v1.4.0)
**Yapılan Değişiklikler:**
1.  **Yeni Haberleşme Metrikleri:** Her düğüm için "Toplam Başarılı İletişim Süresi" ve "Maksimum Kesintisiz İletişim Süresi" (Max Continuous Streak) metrikleri sisteme eklendi.
2.  **Dashboard Arayüzü:** Simülasyon sonunda üretilen tüm grafikleri (Yörünge, Zaman Serileri, İstatistikler) tek bir pencerede birleştiren `show_dashboard()` özelliği `visualizer.py` modülüne entegre edildi.
3.  **Varlık Güncellemesi:** `IoTNode` sınıfı, kendi iletişim tarihçesini (History) tutacak şekilde akıllandırıldı.
