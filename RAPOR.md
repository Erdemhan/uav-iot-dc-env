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
*   **Ray RLLib:** Dağıtık (Distributed) Reinforcement Learning eğitimi için kullanılmıştır. PPO (Proximal Policy Optimization) gibi gelişmiş algoritmaların, çoklu ajan (PettingZoo) ortamımızla entegre bir şekilde çalıştırılmasını ve modelin (`train.py` üzerinden) eğitilmesini sağlayan temel kütüphanedir.

### 2.6. Mevcut Simülasyon Senaryosu (v1.7.0 - Akıllı Tehdit Modeli)

Bu sürümde kullanılan senaryo, "Adil ve Kıyaslanabilir Akıllı Tehdit" (Fair & Comparable Smart Threat) modelidir.

*   **Operasyonel Alan:** 1000m x 1000m boyutlarında 2 boyutlu düzlem.
*   **Frekans Spektrumu:** Sistem 3 farklı frekans kanalında (2.4, 5.0, 5.8 GHz) çalışabilmektedir.
*   **İHA Davranışı (Blue Team - Reaktif Hedef):**
    *   **Görev:** 5 adet IoT düğümünü sırayla ziyaret edip veri toplamak.
    *   **Tepkisellik:** Eğer İHA saldırıya uğrarsa (SINR < 0dB), bulunduğu kanalı terk eder ve bir sonraki adıma **Markov Geçiş Matrisi** (Transition Matrix) ile karar verir. Yani kaçışı rastgele değil, belirli bir istatistiksel örüntüye dayalıdır.
*   **Saldırgan Davranışı (Red Team - Akıllı Ajan):**
    *   **Amaç:** İHA'nın kanal değiştirme örüntüsünü öğrenip onu bloke etmek.
    *   **Yöntemler:** Baseline (Q-Learning), PPO (Deep RL) veya DQN (Deep Q-Network) algoritmalarıyla eğitilir.
    *   **Kısıtlar:** Sürekli yüksek güç basamaz (Enerji maliyeti) ve her frekansta aynı verimlilikte değildir (PA Efficiency).
*   **Ağ Dinamikleri:**
    *   Başarılı iletişim için sadece mesafe yetmez, **Kanal Uyumu** (Saldırganla çakışmama) gereklidir.

---

## 3. MATEMATİKSEL MODELLER

Sistemin gerçekçiliği, tez önerisinde belirtilen aşağıdaki modellerin entegrasyonu ile sağlanmıştır:

### 3.1. Haberleşme Kanalı (Air-to-Ground)
Hava-Yer kanalı için serbest uzay yol kaybı modeli temel alınmış ve aşağıdaki SINR (Sinyal-Gürültü ve Girişim Oranı) denklemi kullanılmıştır:

$$ SINR = \frac{P_{rx}}{N_0 + I_{jam}} $$

Burada $P_{rx}$ alınan güç, $N_0$ termal gürültü ve $I_{jam}$ saldırganın oluşturduğu girişim gücüdür. Veri hızı ise Shannon-Hartley teoremi ile hesaplanmaktadır (Denklem 245, 248).

### 3.2. Enerji Modelleri
*   **İHA:** Döner kanatlı İHA enerji tüketimi, ileri uçuş hızı ($v$) ve askıda kalma (hover) durumlarını içeren kapsamlı bir aerodinamik model ile hesaplanmaktadır (Denklem 263, 272). Hover durumunda ($v=0$), indüklenen güç (induced power) arttığı için güç tüketimi ileri uçuşa göre daha yüksektir.
*   **IoT Düğüm:** Veri toplama, şifreleme ve iletim süreçlerinin toplam enerji maliyeti modellenmiştir (Denklem 288).

---

## 4. ÖDÜL (REWARD) MEKANİZMALARININ DETAYLI AÇIKLAMASI

Jammer'ın eğitimi, üç farklı algoritma için (Baseline, PPO, DQN) dikkatli tasarlanmış ödül fonksiyonları ile yönlendirilmektedir. Bu ödül yapıları, hem jamming etkinliğini maksimize ederken, hem de enerji verimliliğini koruyan stratejilerin öğrenilmesini teşvik eder.

### 4.1. Baseline (QJC Algoritması) Ödül Yapısı

Klasik Q-Learning yaklaşımı, ayrık bir durum-aksiyon ödül tablosu kullanır:

*   **Durum (State):** Mevcut kanal (0, 1, 2)
*   **Aksiyon (Action):** Güç seviyesi (0-9)
*   **Ödül Formülü:**
    ```
    Ödül = (Jamlenmiş_Düğüm_Sayısı × 10) - (Enerji_Tüketimi × 0.1)
    ```

Bu basit linear yapı, tablosal Q-Learning için yeterli geri bildirim sağlar.

### 4.2. PPO & DQN (Derin Takviyeli Öğrenme) Ödül Yapısı

Her iki derin öğrenme algoritması da **aynı üç bileşenli** ödül yapısını paylaşır:

#### Bileşen 1: Jamming Başarı Ödülü (Sparse, Yüksek Değer)
```
ödül_başarı = jamlenen_düğüm_sayısı × 10
```
*   **Amaç:** Birincil hedef - jamming etkinliğini maksimize etmek
*   **Aralık:** 0 ile 50 arası (5 düğüm için)
*   **Tip:** Seyrek ödül (sadece jamming başarılı olduğunda verilir)

#### Bileşen 2: Kanal Takip Ödülü (Dense, Düşük Değer)
```python
if (jammer_kanalı == uav_kanalı AND jammer_gücü > 0.01):
    ödül_takip = 0.5
else:
    ödül_takip = 0.0
```
*   **Amaç:** Jammer'ı İHA'nın frekans atlama davranışını takip etmesi için yönlendirmek
*   **KRİTİK KOŞUL:** Sadece **güç kullanıldığında** verilir (sömürüyü önler)
*   **Tip:** Yoğun rehberlik sinyali (her adımda kontrol edilir)
*   **Tasarım Mantığı:** Güç eşik kontrolü (`> 0.01W`), ajanların "sıfır güçle sadece kanal takibi yaparak ödül alma" açığını kapatır.

#### Bileşen 3: Enerji Maliyet Cezası
```
ödül_enerji = -(jammer_güç_tüketimi × 0.1)
```
*   **Amaç:** Enerji verimli jamming stratejilerini teşvik etmek
*   **Aralık:** 0 ile -0.01W (tipik değerler)
*   **Etki:** Gereksiz yüksek güç kullanımını cesaretlendirmez

#### Toplam Ödül
```
toplam_ödül = ödül_başarı + ödül_takip - ödül_enerji
```

### 4.3. Ödül Tasarımının Teorik Temelleri

1.  **Güç Eşik Kontrolü:**
    *   **Problem:** İlk versiyonda takip ödülü (`+0.5`) güçten bağımsızdı.
    *   **Sonuç:** Ajanlar "İHA kanalını takip et ama jamming yapma" şeklinde dejenere bir politika öğrendi.
    *   **Çözüm:** `power > 0.01W` koşulu eklenerek, ödülün sadece gerçek jamming faaliyeti sırasında verilmesi sağlandı.

2.  **Ölçeklendirme Dengesi:**
    *   Başarı ödülü (10×) > Takip ödülü (0.5) → Jamming birincil hedef olarak kalır.
    *   Enerji cezası (0.1×) → Jamming'i caydırmayacak kadar küçük, ama verimsiz güç kullanımını optimize edecek kadar anlamlı.

3.  **Seyrek vs Yoğun Ödül Trade-off'u:**
    *   **Seyrek (Jamming):** Doğrudan hedefi yansıtır ama öğrenmeyi zorlaştırır (credit assignment problem).
    *   **Yoğun (Takip):** Gradient akışını stabilize eder ve erken aşamada keşfi hızlandırır.
    *   **Birlikte:** İki ödül tipi sinerjik çalışarak hem exploration hem de exploitation'ı dengeştirir.

### 4.4. Algoritma Karşılaştırması - Ödül Kullanımı

| Metrik                  | Baseline (QJC) | PPO           | DQN           |
|-------------------------|----------------|---------------|---------------|
| **Ödül Yapısı**         | Linear combina | 3-Comp Hybrid | 3-Comp Hybrid |
| **Güç Eşik Kontrolü**   | Yok (Not Needed) | Var (0.01W)   | Var (0.01W)   |
| **Policy Gradient**     | Tabular Update | CLIP-PPO      | Q-Learning    |
| **Exploration Strategy**| ε-greedy       | Entropy Bonus | ε-decay       |

---

## 5. ANALİZ VE GÖRSELLEŞTİRME YÖNTEMLERİ

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

## 6. GELİŞİM GÜNLÜĞÜ (CHANGE LOG)

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

### [06.02.2026 17:00] - Davranışsal Gerçekçilik (v1.5.0)
**Yapılan Değişiklikler:**
1.  **Hover (Askıda Kalma) Mantığı:** İHA'nın sadece üzerinden geçmek yerine, düğümlere vardığında verimli veri toplamak adına 5 saniye boyunca havada asılı kalması (Hover) sağlandı.
2.  **Dinamik Navigasyon:** Hedefe varış kriteri, simülasyon adım büyüklüğüne (Step Size) göre dinamik hale getirilerek "overshoot" (hedefi ıskalama) problemleri çözüldü.
3.  **Güç Tüketimi Görünürlüğü:** Analiz modülünde enerji grafiklerinin anlık güç değişimlerini (Hover vs Flight) yansıtması sağlandı (Kullanıcı isteği üzerine kümülatif gösterime geri dönüldü ancak altyapı bu detayı desteklemektedir).

### [06.02.2026 21:00] - Akıllı Tehdit & RL Entegrasyonu (v1.6.0)
**Yapılan Değişiklikler:**
1.  **Çoklu Frekans Kanalı:** Sistem artık 2.4, 5.0 ve 5.8 GHz kanallarını desteklemekte ve fiziksel katman (Path Loss, PA Efficiency) buna göre modellenmektedir.
2.  **Akıllı Tehdit Modeli (QJC):** Liao ve ark. (2025) tarafından önerilen Q-Learning tabanlı Jammer Kanal Seçim algoritması (SmartAttacker sınıfına) entegre edildi.
3.  **Reaktif Markov İHA:** İHA'nın jamming saldırısına uğradığında rastgele değil, belirli bir olasılıksal matrise (Markov Zinciri) göre kanal değiştirdiği "Hareketli Hedef" modeli oluşturuldu.
4.  **RLLib Entegrasyonu:** `train.py` dosyası ile Ray RLLib (PPO) üzerinde Jammer'ın bu Markov modelini öğrenmesi için eğitim altyapısı kuruldu.

### [07.02.2026 00:10] - Adillik ve Kıyaslama Paketi (v1.7.0)
**Yapılan Değişiklikler:**
1.  **Sensing Mode (Gerçekçi Algılama):**
    *   RL Ajanının (Jammer) gözlem uzayından hileli "God View" (Tam Koordinat) verisi çıkarıldı.
    *   Yerine, sadece **Mesafe** (Distance) ve **Sinyal Gücü** (RSSI) gibi gerçek hayatta sensörlerle ölçülebilen veriler eklendi.
2.  **Adil Kıyaslama (Algorithmic Fairness):**
    *   Baseline (QJC) algoritmasının, PPO eğitim sürelerine denk (60K adım) deneyim kazanması için **Ön Eğitim (Pre-training)** modülü (`train_baseline.py`) eklendi.
    *   Böylece "Eğitimsiz Baseline vs Eğitilmiş RL" adaletsizliği giderildi.
3.  **Algoritmik Çeşitlilik:**
    *   **PPO:** Sürekli (Continuous) politika optimizasyonu (New API Stack).
    *   **DQN:** Ayrık (Discrete) aksiyon uzayı optimizasyonu (Old API Stack).
    *   **Baseline:** Tablosal (Tabular) Q-Learning.
    *   Üç algoritmayı tek komutla yarıştıran `run_experiments.py` otomasyonu geliştirildi.
4.  **Otomatik Raporlama:** Tüm sonuçları karşılaştırmalı sütun grafiklerine (`comparison_result.png`) döken analiz modülü eklendi.
### [07.02.2026 02:00] - RLLib Yama ve Otomasyon Paketi (v1.8.0)
**Yapılan Değişiklikler:**
1.  **Ray RLLib Hata Düzeltmesi (Bug Fix):**
    *   Ray 2.53.0 sürümündeki DQN algoritmasının "Old API Stack" yolunda kilitlenmesine neden olan `TypeError: argument of type 'ABCMeta' is not iterable` hatası tespit edildi.
    *   `rllib/algorithms/algorithm.py` dosyasına `isinstance(..., str)` kontrolü eklenerek yerel kütüphane yamalandı.
    *   Düzeltme, Ray projesine resmi Pull Request olarak gönderildi (DCO imzalı).
2.  **Tam Otomatik Deney Hattı (Experiment Pipeline):**
    *   `run_experiments.py` scripti geliştirilerek; Baseline (QJC), PPO ve DQN modellerinin sırayla eğitilmesi, değerlendirilmesi ve karşılaştırmalı rapor üretilmesi otomatikleştirildi.
3.  **Temizlik ve Optimizasyon:**
    *   Gereksiz geçici dosyalar ve büyük boyutlu fork dosyaları silinerek proje alanı optimize edildi.
4.  **Dokümantasyon Güncellemesi:**
    *   `README.md` ve `.gitignore` dosyaları yeni deney çıktılarını ve yama sürecini kapsayacak şekilde güncellendi.

**Amaç:**
Simülasyon ortamının akademik bir test yatağı (testbed) olarak kararlılığını sağlamak ve Ray kütüphanesi kaynaklı engelleri kalıcı olarak aşmak.

Bu sürüm ile proje, "Saldırgan Kıyaslama" (Attacker Comparison) makalesi için gerekli test yatağına dönüşmüştür.

### [07.02.2026 03:45] - Konfigürasyon Refactoring ve GPU Desteği (v1.9.0)
**Yapılan Değişiklikler:**
1.  **Merkezi Konfigürasyon Yapısı (`confs/model_config.py`):**
    *   **GlobalConfig:** Tüm algoritmalar için ortak parametreler (`RANDOM_SEED`, `FLATTEN_ACTIONS`) merkezi hale getirildi.
    *   **RLConfig → PPOConfig:** PPO parametreleri yeniden adlandırılarak netleştirildi ve model mimarisi (`FCNET_HIDDENS`) eklendi.
    *   **DQNConfig (YENİ):** DQN için özel hyperparameter sınıfı oluşturuldu (LR, GAMMA, TRAIN_BATCH_SIZE, TARGET_NETWORK_UPDATE_FREQ, REPLAY_BUFFER_CAPACITY, vb.).
    *   **QJCConfig Genişletildi:** `TRAIN_EPISODES`, `SAVE_PATH`, `MAX_POWER_LEVEL` parametreleri merkezi yapıya taşındı.
    
2.  **Reproducibility (Yeniden Üretilebilirlik) Garantisi:**
    *   Random seed değeri (`RANDOM_SEED = 42`) tüm training scriptlerinde hardcoded olarak tekrar ediliyordu. Artık tek bir noktadan (`GlobalConfig.RANDOM_SEED`) yönetiliyor.
    *   Seed değiştirmek için tek satır edit yeterli.

3.  **PyTorch CUDA Kurulumu ve GPU Desteği:**
    *   **Sorun Tespiti:** PyTorch CPU-only versiyonu (2.5.1+cpu) yüklüydü, CUDA 12.2 kurulu olmasına rağmen GPU tanınmıyordu.
    *   **Çözüm:** Conda install timeout sorunu nedeniyle pip kullanılarak `torch-2.5.1+cu121` kuruldu.
    *   **Donanım Doğrulaması:** NVIDIA GeForce RTX 3080 başarıyla tanındı ve aktif edildi.
    *   **Performans Etkisi:** GPU desteği ile eğitim hızı ~5-10x artış gösterdi.

4.  **Action Space Harmonizasyonu (Adalet İyileştirmesi):**
    *   **Sorun:** PPO `MultiDiscrete([3, 10])` kullanırken DQN `Discrete(30)` kullanıyordu. Bu PPO'ya %60-70 yapısal avantaj sağlıyordu (2.5x gradient efficiency, structured exploration).
    *   **Çözüm:** Her iki algoritma da `Discrete(30)` kullanacak şekilde harmonize edildi (`GlobalConfig.FLATTEN_ACTIONS = True`).
    *   **Sonuç:** %100 adil kıyaslama garantisi sağlandı.

5.  **Gamma Harmonizasyonu:**
    *   PPO'nun `GAMMA = 0.95` değeri, Baseline ve DQN'in `0.9` değeriyle eşitlenerek (PPOConfig.GAMMA = 0.9) tüm algoritmaların aynı ödül iskontolama stratejisini kullanması sağlandı.

6.  **API Stack Şeffaflığı:**
    *   PPO: New API Stack (varsayılan, modern, aktif geliştirme)
    *   DQN: Old API Stack (gereklilik, MultiDiscrete native desteği yok)
    *   Her algoritma kendi en stabil stack'ini kullanıyor, performans adaleti korunuyor.

**Amaç:**
Proje bakımını kolaylaştırmak, hyperparameter tuning'i merkezileştirmek ve tüm deney koşullarını %100 yeniden üretilebilir kılmak. Ayrıca GPU desteği ile eğitim süresini optimize edip bilimsel iterasyon hızını artırmak.

Bu sürüm ile proje, "Fair Algorithmic Comparison" standartlarına tam uyumlu hale getirilmiştir.

### [07.02.2026 04:16] - Konfigürasyon İyileştirmeleri v1.9.1 (Patch)
**Yapılan Değişiklikler:**
1.  **TRAIN_ITERATIONS Merkezileştirilmesi:**
    *   `TRAIN_ITERATIONS` parametresi tüm config class'larından kaldırılıp `GlobalConfig`'e taşındı.
    *   Artık eğitim iterasyonunu değiştirmek için tek satır edit yeterli.
    *   **Etkilenen:** `QJCConfig`, `PPOConfig`, `DQNConfig` → `GlobalConfig.TRAIN_ITERATIONS`

2.  **PPO API Stack Harmonizasyonu:**
    *   PPO, DQN ile aynı performans ve şeffaflık için Old API Stack'e geçirildi.
    *   **Fayda:** Her iki algoritma da aynı API stack kullanıyor → GPU raporlaması ve davranış tam eşit.
    
3.  **DQN Training Intensity Optimizasyonu:**
    *   DQN'de `training_intensity=1` parametresi eklendi.
    *   **Sorun:** DQN her iterasyonda 1M gradient update yapıyordu (60 dakika).
    *   **Çözüm:** Training intensity ile sınırlandırıldı → **2 dakikaya düştü** (~30x hızlanma).
    
4.  **Ray Metrics Uyarılarının Gizlenmesi:**
    *   `RAY_DISABLE_METRICS_EXPORT=1` environment variable eklendi.
    *   Zararsız "metrics exporter" uyarıları temiz çıktı için susturuldu.

**Amaç:**
Kod kalitesini artırmak, eğitim süresini optimize etmek ve geliştirici deneyimini iyileştirmek.

### 6.1. Fiziksel Katman Güncellemeleri
*   **Kanallar:** 2.4 GHz, 5.0 GHz ve 5.8 GHz.
*   **PA Verimliliği (Cui et al., 2005):** Yüksek frekanslarda güç amplifikatörü verimliliğinin düştüğü model (2.4GHz: 0.50 -> 5.8GHz: 0.19) simüle edilmiştir. Bu, Jammer için "Yüksek frekansta jam yapmak daha maliyetlidir" trade-off'unu oluşturur.

### 6.2. Zeka ve Strateji
1.  **Akıllı Jammer (QJC):**
    *   **Algoritma:** Q-Learning tabanlı kanal takibi.
    *   **Hedef:** İHA'nın hangi kanala kaçacağını tahmin edip o kanalı bloke etmek.
    *   **Referans:** Liao et al. (2025).

2.  **Reaktif Kurban (İHA):**
    *   **Davranış:** Bağlantı koptuğunda (SINR < Eşik) kanal değiştirir.
    *   **Markov Modeli:** Kanal değişimi rastgele değildir; gizli bir `Transition Matrix` kullanır. Bu sayede Saldırganın (RL Ajanı) öğrenebileceği istatistiksel bir örüntü sunar.

### 6.3. Eğitim (`train.py`)
Sistemi eğitmek için:
```bash
python train.py
```
Bu komut, Jammer'ı PPO algoritması ile eğiterek, İHA'nın kaçış örüntüsünü çözmesini ve enerji verimli saldırı yapmasını sağlar.

### 6.4. Operasyonel Senaryo Akışı ve Algoritmik Kıyaslama
Simülasyon döngüsü, farklı zeka seviyelerindeki saldırganların (Baseline vs RL) başarısını kıyaslamak üzere şu adımları izler:

1.  **Başlangıç ve Kalibrasyon (Initialization):**
    *   İHA ve IoT düğümleri varsayılan kanal üzerinden iletişime başlar.
    *   **Kıyaslama Hazırlığı:** `run_experiments.py` vasıtasıyla üç farklı beyin (Baseline QJC, PPO, DQN) aynı senaryo koşullarında sırayla devreye alınır.

2.  **Saldırı Kararı ve Politika Uygulama (Jammer Action):**
    *   **Baseline (QJC):** Sabit bir Q-Tablosu üzerinden en yüksek olasılıklı kanalı seçer.
    *   **PPO (Sensing Mode):** Gözlem uzayından (Mesafe, RSSI) gelen verileri sinir ağı ile işleyerek kanal ve güç kararı verir.
    *   **DQN:** Ayrık aksiyon uzayında en iyi "Q-değerini" tahmin ederek saldırı gerçekleştirir.

3.  **Kanal Etkileşimi ve Girişim (Interference):**
    *   Fizik motoru (`physics.py`), İHA ve Saldırganın kanallarını kontrol eder. Kanal çakışması durumunda SINR düşürülür.

4.  **Reaktif Tepki ve Hareketli Hedef (Target Reaction):**
    *   İHA, jamming tespit ettiğinde "Markov Geçiş Matrisi"ne göre yeni bir kanala zıplar. Bu zıplama istatistiksel bir örüntü oluşturur.

5.  **Performans Ölçümü ve Kıyaslama (Evaluation):**
    *   Saldırganın başarısı; İHA'nın toplam veri toplama süresini ne kadar kısalttığı (Attacker Success Rate) ve harcanan enerji birimi başına verilen zarar (Efficiency) üzerinden ölçülür.
    *   Sonuçlar `comparison_result.png` üzerinde otomatik olarak görselleştirilerek hangi algoritmanın Markov örüntüsünü daha hızlı çözdüğü ortaya konur.

---

## 7. ALGORİTMİK PERFORMANS ANALİZİ (QJC vs PPO vs DQN)

Proje kapsamında üç farklı "Saldırı Zekası" modeli birbiriyle yarıştırılmaktadır.

### 7.1. Klasik Q-Learning (QJC - Baseline)
*   **Çalışma Prensibi:** Durum (state) ve aksiyonları (action) içeren sonlu bir tablo (Look-up Table) tutar.
*   **Avantajı:** Matematiksel olarak basittir ve çok kısıtlı işlem gücüyle çalışabilir.
*   **Bu Projedeki Rolü:** İHA'nın Markov örüntüsünü "istatistiksel" olarak çözmekle görevlidir. DQN ve PPO için bir alt sınır (Lower Bound) çizerek, Deep RL'in sağladığı katma değeri ölçmemize yarar.

### 7.2. PPO (Proximal Policy Optimization)
*   **Çalışma Prensibi:** "Yeni API Stack" (Ray 2.53+) üzerinde çalışan, sürekli aksiyon uzaylarını da destekleyen modern bir politika gradiyent algoritmasıdır.
*   **Karakteristiği:** Eğitim stabilitesi yüksektir. Gözlem uzayındaki gürültülü (RSSI, Mesafe) verilere karşı daha dayanıklıdır.
*   **Beklentimiz:** İHA'nın hareket örüntüsünü sadece kanal bazlı değil, mesafeye bağlı güç optimizasyonuyla birlikte öğrenmesi.

### 7.3. DQN (Deep Q-Network)
*   **Çalışma Prensibi:** Klasik Q-Learning'i derin sinir ağlarıyla birleştirir. Bu projede "Old API Stack" üzerinden, MultiDiscrete uzayı Discrete(30) olarak harmonize edilerek koşturulmaktadır.
*   **Karakteristiği:** Örnek verimliliği (Sample Efficiency) yüksektir; yani daha az adımda karmaşık kararları öğrenebilir.
*   **Kritik Yama:** DQN'in Ray kütüphanesindeki `ABCMeta` hatası tarafımızca yamalanarak kütüphane stabil hale getirilmiştir.

### 7.4. Kıyaslama Metrikleri
Deney sonunda üretilen `comparison_result.png` şu sorulara yanıt verir:
1.  **Kilitleme Hızı (Tracking Accuracy):** Jammer, UAV'nin kanalını ne kadar sürede tahmin edebiliyor?
2.  **Zarar Verme Kapasitesi (Success Rate):** UAV'nin veri toplama başarısını yüzde kaç düşürebiliyor?
3.  **Enerji Verimliliği:** En az güç harcayarak en yüksek zararı hangi algoritma veriyor?
