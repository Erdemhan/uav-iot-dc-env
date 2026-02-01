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

### 2.1. Çekirdek Modüller (`core/`)

*   **`core/config.py` (Konfigürasyon Yönetimi):** Sistemin fiziksel bant genişliği, frekans, gürültü seviyesi gibi temel donanım parametrelerini tutar.
*   **`core/env_config.py` (Ortam ve Senaryo Konfigürasyonu):** Simülasyonun senaryo parametrelerini (Düğüm sayısı, alan boyutu, adım süresi, saldırgan konumu vb.) barındırır. Bu ayrım sayesinde fiziksel altyapı değiştirilmeden farklı senaryolar test edilebilir.
*   **`core/physics.py` (Fizik Motoru):** Sistemin "stateless" (durumsuz) matematiksel hesaplama çekirdeğidir. Haberleşme kanalı (Path Loss, SINR, Shannon Kapasitesi) ve enerji tüketim modelleri (İHA uçuş gücü, IoT iletim enerjisi) burada saf fonksiyonlar (pure functions) olarak implemente edilmiştir.
*   **`simulation/entities.py` (Varlık Modellemesi):** Simülasyon dünyasındaki aktörlerin (İHA, IoT Düğümü, Saldırgan) davranışlarını ve durumlarını modelleyen sınıfları içerir.
    *   *Miras Yapısı:* `BaseEntity` -> `MobileEntity` / `TransceiverEntity` -> `UAVAgent` / `IoTNode` şeklinde hiyerarşik bir yapı kurgulanmıştır.

### 2.2. Simülasyon ve Ortam

*   **`simulation/environment.py` (OpenAI Gym Ortamı):** `UAV_IoT_Env` sınıfı, simülasyonun durum uzayı (state space), aksiyon uzayı (action space) ve ödül mekanizmasını (reward function) tanımlar. Zaman adımlı (time-stepped) bir akış içerisinde fizik motorunu ve varlıkları koordine eder.
*   **`main.py` (Yürütücü):** Simülasyonun başlatılması, döngünün yönetimi ve kaynakların (logger, visualizer) serbest bırakılmasından sorumludur. `config.py` içerisindeki `SIMULATION_DELAY` parametresi ile simülasyon akış hızı kontrol edilebilir.

### 2.3. Veri Yönetimi ve Analiz

*   **`core/logger.py` (Telemetri Kaydı):** Simülasyon sırasında üretilen ham verileri (konumlar, SINR değerleri, enerji tüketimleri) periyodik olarak CSV formatında kayıt altına alır.

*   **`visualization/visualizer.py` (Görsel Analiz):** Simülasyon sonrası elde edilen verileri işleyerek akademik kalitede (SCIE standartlarında) grafikler ve yörünge analizleri üretir.

### 2.4. Mevcut Simülasyon Senaryosu (v1.0.0)

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
