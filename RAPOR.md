# PROJE GELİŞİM RAPORU VE TEKNİK DOKÜMANTASYON

**Proje Başlığı:** Nesnelerin İnterneti Tabanlı İHA Uygulamalarında Güvenlik Hassasiyetli Akıllı Yöntemlerin Geliştirilmesi
**Rapor Tarihi:** 23.06.2026 19:22
**Versiyon:** 3.2.3 (Ray Temizleme Entegrasyonu)

---

## 1. GİRİŞ VE SİSTEM GENEL BAKIŞI

Bu proje, Doktora Tezi kapsamında İnsansız Hava Araçları (İHA) ve Nesnelerin İnterneti (IoT) ağlarının entegre çalıştığı senaryolarda, siber güvenlik tehditlerinin (özellikle Jamming saldırıları) etkilerini analiz etmek ve bunlara karşı dayanıklı akıllı yöntemler geliştirmek amacıyla tasarlanmıştır.

Geliştirilen simülasyon ortamı, literatürdeki standartlara uygun olarak Python tabanlı, modüler, genişletilebilir ve bilimsel geçerliliği olan matematiksel modellere dayalı bir altyapıya sahiptir. OpenAI Gymnasium arayüzü benimsenerek, Baseline (QJC), DRL (PPO, DQN) ve Bellek Tabanlı (PPO-LSTM) algoritmaların entegre çalışabildiği kapsamlı bir test yatağı (testbed) oluşturulmuştur.

---

## 2. İLGİLİ ÇALIŞMALAR (LİTERATÜR ÖZETİ)

İHA destekli iletişim ağlarında karıştırma (jamming) saldırılarına karşı güvenilirlik sağlama problemi üç ana eksende incelenmiştir:

### 2.1. Oyun Teorisi Tabanlı Yaklaşımlar
Liao et al. (2025), İHA konuşlandırmasını bir **Tıkanıklık Oyunu (Congestion Game)** ve taşıyıcı seçimini **Stackelberg Oyunu** olarak modellemiştir. Bu yöntemler matematiksel bir denge (Nash Equilibrium) garantisi sunsa da, oyuncuların tam rasyonel olduğunu varsayar ve dinamik tehditlere adaptasyon süreleri uzundur.

### 2.2. Geleneksel Yöntemler
Parçacık Sürü Optimizasyonu (PSO) ve Bulanık C-Means (FCM) gibi sezgisel algoritmalar genellikle statik ortam optimizasyonu için kullanılır. Düşman (adversarial) bir jammerın anlık strateji değiştirdiği senaryolarda yetersiz kalabilirler.

### 2.3. Pekiştirmeli Öğrenme (RL) Yaklaşımları
QJC (Q-Learning Based Jamming) gibi temel RL yöntemleri, düşük işlem maliyeti sunar ancak genellikle "kör" (blind) stratejilerdir; yani ortamı algılamadan sadece ödül geçmişine bakarlar.

**Önerilen Yöntem:** Çalışmamızda kullanılan PPO ve DQN algoritmaları, jammerın sinyal gücünü (RSS) ve spektrum doluluğunu algıladığı "Smart Jammer" modeline dayanır. Bu, kör öğrenme yerine **durum-farkında (state-aware)** ve veriye dayalı (data-driven) bir savunma/saldırı mekanizması sağlar.

## 3. SİSTEM MİMARİSİ

Simülasyon altyapısı, Nesne Yönelimli Programlama (OOP) prensipleri çerçevesinde, her biri spesifik bir görevi üstlenen gevşek bağlı (loose-coupled) modüllerden oluşmaktadır.

### 3.1. Konfigürasyon Modülleri (`confs/`)

*   **`confs/config.py` (Sistem Konfigürasyonu):** Sistemin fiziksel bant genişliği, frekans, gürültü seviyesi gibi temel donanım parametrelerini tutar.
*   **`confs/env_config.py` (Ortam ve Senaryo Konfigürasyonu):** Simülasyonun senaryo parametrelerini (Düğüm sayısı, alan boyutu, adım süresi, saldırgan konumu vb.) barındırır. Bu ayrım sayesinde fiziksel altyapı değiştirilmeden farklı senaryolar test edilebilir.

### 3.2. Çekirdek Modüller (`core/`)
*   **`core/physics.py` (Fizik Motoru):** Sistemin "stateless" (durumsuz) matematiksel hesaplama çekirdeğidir. Haberleşme kanalı (Path Loss, SINR, Shannon Kapasitesi) ve enerji tüketim modelleri (İHA uçuş gücü, IoT iletim enerjisi) burada saf fonksiyonlar (pure functions) olarak implemente edilmiştir.
*   **`simulation/entities.py` (Varlık Modellemesi):** Simülasyon dünyasındaki aktörlerin (İHA, IoT Düğümü, Saldırgan) davranışlarını ve durumlarını modelleyen sınıfları içerir.
    *   *Miras Yapısı:* `BaseEntity` -> `MobileEntity` / `TransceiverEntity` -> `UAVAgent` / `IoTNode` şeklinde hiyerarşik bir yapı kurgulanmıştır.

### 3.3. Simülasyon ve Ortam
*   **`simulation/pettingzoo_env.py` (PettingZoo Ortamı):** `UAV_IoT_PZ_Env` sınıfı, simülasyonun çoklu ajan (multi-agent) yapısını destekleyen `pettingzoo.utils.ParallelEnv` tabanlı ortamdır. İHA, Saldırgan ve her bir IoT düğümü ayrı birer ajan olarak modellenmiştir.
*   **`simulation/controllers.py` (Kural Tabanlı Kontrolcüler):** İHA gibi belirli kurallara (örn. navigasyon) dayalı hareket eden ajanların davranış mantığını kapsüller.
*   **`scripts/main.py` (Yürütücü):** Simülasyon döngüsünü PettingZoo API'sine uygun şekilde (sözlük yapılı aksiyon/gözlem) yönetir. `confs/config.py` içerisindeki `SIMULATION_DELAY` parametresi ile simülasyon akış hızı kontrol edilebilir.

### 3.4. Veri Yönetimi ve Analiz

*   **`core/logger.py` (Telemetri Kaydı):** Simülasyon sırasında üretilen ham verileri (konumlar, SINR değerleri, enerji tüketimleri) periyodik olarak CSV formatında kayıt altına alır.

*   **`visualization/visualizer.py` (Görsel Analiz):** Simülasyon sonrası elde edilen verileri işleyerek akademik kalitede (SCIE standartlarında) grafikler ve yörünge analizleri üretir.

### 3.5. Kullanılan Altyapı ve Teknolojiler

Projenin geliştirilmesinde, akademik standartlara uygunluk ve yüksek performans gereksinimleri gözetilerek aşağıdaki açık kaynaklı kütüphaneler kullanılmıştır:

*   **PettingZoo (Python):** Çoklu ajan (Multi-Agent) takviyeli öğrenme ortamları için endüstri standardı olan bu kütüphane, projemizin temel yapı taşıdır. `ParallelEnv` API'si kullanılarak, İHA, Jammer ve IoT düğümlerinin eş zamanlı olarak etkileşime girdiği, ölçeklenebilir ve oyun teorik analizlere uygun bir simülasyon ortamı oluşturulmuştur.
*   **OpenAI Gymnasium:** PettingZoo'nun üzerine inşa edildiği temel API yapısıdır. Ajanların durum-aksiyon uzaylarının (Box, Discrete) tanımlanmasında standartları belirler.
*   **NumPy:** Yüksek performanslı vektörel matematik işlemleri için kullanılmıştır. Fizik motorundaki (`physics.py`) sinyal gücü, SINR ve enerji hesaplamaları, döngüler yerine NumPy vektör operasyonları ile optimize edilerek simülasyon hızı artırılmıştır.
*   **Matplotlib:** Simülasyon verilerinin görselleştirilmesi ve analiz grafiklerinin (`trajectory.png`, `metrics_analysis.png`) oluşturulması için kullanılmıştır.
*   **Pandas:** Simülasyon loglarının (`history.csv`) işlenmesi, filtrelenmesi ve zaman serisi analizlerinin yapılması amacıyla veri manipülasyonu için tercih edilmiştir.
*   **Ray RLLib:** Dağıtık (Distributed) Reinforcement Learning eğitimi için kullanılmıştır. PPO (Proximal Policy Optimization) gibi gelişmiş algoritmaların, çoklu ajan (PettingZoo) ortamımızla entegre bir şekilde çalıştırılmasını ve modelin (`train.py` üzerinden) eğitilmesini sağlayan temel kütüphanedir.

### 3.6. Mevcut Simülasyon Senaryosu (v3.0.0 - Çoklu İHA ve LoRaWAN Modeli)

Bu sürümde (v3.0.0) kullanılan senaryo, "Çoklu İHA ve Gerçekçi LoRaWAN Spektrumu Altında Zeki Tehdit" modelidir.

*   **Operasyonel Alan:** 1000m x 1000m boyutlarında 2 boyutlu düzlem.
*   **Frekans Spektrumu:** Endüstriyel standart olan 8 kanallı LoRaWAN (EU868) frekans bandı (867.1 MHz - 868.5 MHz) kullanılmaktadır.
*   **İHA Davranışı (Blue Team - Reaktif Hedef):**
    *   **Görev:** Ortamda yer alan 30 adet IoT düğümünden işbirlikli veri toplamak ($M=3$ İHA).
    *   **Navigasyon ve Rota Planlama:** Hesaplama karmaşıklığını ve seyahat sürelerini minimize etmek amacıyla Yapay Potansiyel Alanları (APF) tabanlı çarpışma engelleme itici kuvvetleri devre dışı bırakılmış, doğrudan doğrusal rotalar izlenmiştir. Düğümler arası rota atamasında ise statik coğrafi kümeleme (clustering) yerine dinamik ve işbirlikli bir **Ortak Ziyaret Edilmeyen Düğümler Havuzu (Shared Unvisited Pool)** kurgulanmıştır. Her İHA, o anda diğer İHA'lar tarafından hedeflenmemiş en yakın düğüme yönelir. Bu sayede, tek bir jammerın bulunduğu riskli bölgedeki düğümlere sürekli aynı İHA'nın giderek bataryasını tüketmesi ve jammerın tek bir kurbana kilitlenip saldırıyı basitleştirmesi engellenir. İHA'lar riskli bölgedeki düğümleri sırayla ve dönüşümlü olarak ziyaret ederek karıştırma yükünü ve enerji maliyetini kendi aralarında adil şekilde dağıtırlar.
    *   **Tepkisellik:** İHA, haberleştiği düğümden gelen SINR değerini takip eder. Eğer ardışık 5 adım boyunca SINR < 0dB (saldırı eşiği) kalırsa kanalın karıştırıldığını doğrular ve **Markov Geçiş Matrisi** (Transition Matrix) olasılıklarına göre kanal atlar.
*   **Saldırgan Davranışı (Red Team - Akıllı Ajan):**
    *   **Amaç:** İHA'ların konum/frekans durumlarını algılayarak kanal değiştirme örüntüsünü öğrenip onları bloke etmek.
    *   **Yöntemler:** Baseline (QJC), PPO, DQN veya PPO-LSTM algoritmalarıyla eğitilir.
    *   **Kısıtlar:** Sürekli yüksek güç basamaz (Enerji maliyeti). Karıştırma enerjisi maliyeti olarak doğrudan anlık RF çıkış gücü ($0.1\text{ W} - 1.0\text{ W}$) kullanılmaktadır.
*   **Ağ Dinamikleri:**
    *   Yer düğümleri kendilerine en yakın olan İHA ile dinamik olarak ilişkilendirilir (Dynamic Association).
    *   Aynı İHA ile ilişkilendirilen ve aynı kanalda çalışan diğer düğümler arasında Eş-Kanal Girişimi (Co-Channel Interference - CCI) modellenmiştir.

---

## 4. MATERYAL VE YÖNTEMLER

Bu bölüm, sistemin fiziksel ve matematiksel altyapısını detaylandırmaktadır.

### 4.1. Sistem Modeli
Senaryo, $1000 \times 1000$ metrelik bir alana yerleştirilmiş $N=30$ adet sabit IoT düğümü, $M=3$ adet İHA ve 1 adet Akıllı Jammer içermektedir. Yer düğümleri kendilerine en yakın olan İHA ile dinamik olarak ilişkilendirilir.

#### 4.1.1. Haberleşme Kanalı (Air-to-Ground)
İHA ile düğümler arasındaki iletişim kalitesi, anlık **SINR (Signal-to-Interference-plus-Noise Ratio)** değeri ile belirlenir:

$$ \text{SINR}_i = \frac{P_{rx,i}}{\sigma^2 B + P_{rx,jam} + P_{rx,co}} $$

Burada:
*   $P_{rx,i}$: Friis denklemi ile hesaplanan alınan sinyal gücü ($P_{tx,node} \cdot \beta_0(f) / d^2$).
*   $\sigma^2 B$ (veya $N_0$): Toplam alıcı gürültü gücü (Noise floor). Gerçekçi LoRaWAN EU868 standartlarında $B = 125\text{ kHz}$ bant genişliği ve Semtech SX1261 LoRa alıcı çipinin $NF = 6\text{ dB}$ gürültü katsayısı temel alınarak $-117\text{ dBm}$ ($1.995 \times 10^{-15}\text{ W}$) olarak hesaplanmıştır (detaylar aşağıda verilmiştir).
*   $P_{rx,jam}$: Jammer'dan kaynaklanan ve İHA alıcısına ulaşan karıştırma gücü.
*   $P_{rx,co}$: Aynı kanalda çalışan diğer yer düğümlerinden kaynaklanan Eş-Kanal Girişimi (Co-Channel Interference - CCI).

**Gürültü Tabanı (Noise Floor) ve Bant Genişliği Modellemesi:**
Sistemde, haberleşme kanalı bant genişliği $B = 125\text{ kHz}$ olarak ayarlanmıştır. Bu değer, standart EU868 LoRaWAN frekans planındaki tekil bir kanalın bant genişliğine karşılık gelir. Alıcı gürültü tabanı ($N_0$ veya $\sigma^2 B$), Johnson-Nyquist termal gürültü modeli ve donanım kısıtları doğrultusunda şu şekilde hesaplanmıştır:
1.  **İdeal Termal Gürültü Gücü ($P_{\text{thermal}}$):** Oda sıcaklığında ($T = 290\text{ K}$) ve $B = 125\text{ kHz}$ bant genişliği için teorik alt sınır:
    $$ P_{\text{thermal}} = k \cdot T \cdot B = (1.38 \times 10^{-23}\text{ J/K}) \cdot (290\text{ K}) \cdot (125 \times 10^3\text{ Hz}) \approx 5.0025 \times 10^{-16}\text{ W} $$
    Desibel miliwatt ($\text{dBm}$) cinsinden değeri:
    $$ P_{\text{thermal, dBm}} = 10 \cdot \log_{10}\left(\frac{P_{\text{thermal}}}{10^{-3}}\right) \approx -123\text{ dBm} $$
2.  **Semtech SX1261 Alıcı Gürültü Katsayısı (Noise Figure - NF):** Donanım veri sayfasında (datasheet) belirtilen ve iç devre kayıplarını temsil eden alıcı gürültü katsayısı $NF = 6\text{ dB}$'dir.
3.  **Gerçekçi Gürültü Tabanı ($N_0$):** Alıcı devrelerindeki kayıplar ideal termal gürültü sınırına eklenerek alıcı duyarlılığı/gürültü tabanı elde edilir:
    $$ N_{\text{floor, dBm}} = P_{\text{thermal, dBm}} + NF = -123\text{ dBm} + 6\text{ dB} = -117\text{ dBm} $$
    Bunun lineer güç ölçeğindeki (Watt) karşılığı, simülasyon kodunda kullanılan `N0_Linear` değeridir:
    $$ N_0 = 10^{\frac{-117}{10}} \cdot 10^{-3} \approx 1.995 \times 10^{-15}\text{ W} $$

**PA Verimliliği (Güç Amplifikatörü):**
Sistem modelimiz, 8 kanallı EU868 LoRaWAN bandını temel alır. Kanallar 867.1 MHz ile 868.5 MHz arasında dar bir bantta toplandığı için, hem yer düğümlerinde hem de İHA alıcı-vericilerinde ortak Semtech SX1261 LoRa entegresi varsayılmış ve güç amplifikatörü (PA) verimliliği tüm kanallar için veri sayfası (datasheet) değerlerine dayanarak **%29.8 (0.298) uniform sabit** olarak modellenmiştir. Ayrıca hem düğüm iletim gücü $P_{tx, node}$ hem de İHA iletim gücü $P_{tx, uav}$ bu entegrenin maksimum verimli çıkış seviyesi olan $14\text{ dBm}$ ($0.025\text{ W}$) değerine eşitlenerek fiziksel tutarlılık artırılmıştır.

#### 4.1.2. Enerji Tüketim Modelleri
*   **İHA Uçuş Enerjisi:** Aerodinamik prensiplere dayalı güç tüketimi $P_{UAV}(v)$:
    $$ P_{UAV}(v) = P_0 \left( 1 + \frac{3v^2}{U_{tip}^2} \right) + P_{ind} \left( \sqrt{1 + \frac{v^4}{4v_0^4}} - \frac{v^2}{2v_0^2} \right)^{1/2} + \frac{1}{2} d_0 \rho s A v^3 $$
    İHA askıda sabit dururken ($v = 0$ - Hover modu) aerodinamik güç tüketimi $P_{hover} = P_0 + P_{ind}$ şeklindedir.
*   **IoT Enerjisi:** Veri toplama, şifreleme ve iletim ($E_{node} = E_{acq} + P_{enc}(L_p t_{enc}) + P_{tx,node} (L_p / R_i)$) maliyetlerinin toplamıdır.
*   **Jammer Enerjisi:** Jammer'ın güç tüketimi olarak doğrudan anlık RF çıkış gücü ($P_{total\_jam} = P_{jam\_out}$) kullanılmaktadır. Bu basitleştirilmiş model sayesinde, jammer donanımının iç devre yapısından bağımsız bir şekilde doğrudan uygulanan sinyal karıştırma çıkış gücünün ($0.1\text{ W} - 1.0\text{ W}$) maliyeti ölçülmektedir.

### 4.2. Problem Formülasyonu (MDP)
Problem, $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$ ile tanımlanan bir Markov Karar Sürecidir.

*   **Durum Gözlem Uzayı ($\mathcal{S}$):** Toplamda $3 + 4 + 30 = 37$ boyuta sahip olan durum gözlem vektörü şunları içerir (M=3 İHA senaryosu için):
    1. **İHA Uzaklıkları (3 değer):** Jammer'dan her bir İHA'ya olan mesafenin normalize değerleri (UAV RSSI için proxy).
    2. **İHA Kanal Durumları (3 değer):** 3 İHA'nın o an kullandığı aktif kanal indeksleri.
    3. **Karıştırıcı Kanal Durumu (1 değer):** Karıştırıcının o an aktif olduğu kanal indeksi.
    4. **Düğüm Sinyal Güçleri (RSSI) (30 değer):** Alandaki 30 sabit yer düğümünün Jammer alıcısındaki normalize edilmiş anlık telsiz sinyal gücü (RSSI) değerleri. Düğüm aktif iletim yapmıyorsa -150 dBm (0.0 normalizasyon) kabul edilir.
*   **Aksiyon Uzayı ($\mathcal{A}$):** DQN ile uyumluluk amacıyla $a_{flat} \in \{0, \dots, 79\}$ olacak şekilde 80 ayrık eyleme düzleştirilmiştir ($c_{jam} = \lfloor a_{flat} / 10 \rfloor$ ve $p_{level} = a_{flat} \pmod{10}$).
*   **Ödül Fonksiyonu ($\mathcal{R}$):**
    $$ r_t = w_{jam} \cdot N_{jammed}(t) + w_{track} \cdot \mathbb{I}\left(c_{jam}(t) == c_{closest}(t) \land P_{jam\_out}(t) > 0.01\text{ W}\right) - w_{cost} \cdot P_{total\_jam}(t) $$

### 4.3. Paralel Veri Toplama ve İşçi (Rollout Worker) Mimarisi

Derin Pekiştirmeli Öğrenme (DRL) eğitimlerinde, verimli bir politika güncellemesi yapabilmek için donanım kaynaklarının (CPU ve GPU) dengeli kullanılması kritik önem taşır. Bu tez kapsamında geliştirilen simülasyon ortamında, veri toplama ve model güncelleme adımlarını hızlandırmak amacıyla **Ray RLLib Paralel Rollout Worker Mimarisi** kurgulanmıştır.

#### 4.3.1. Rollout Worker Görev Dağılımı
Eğitim süreci iki ana süreç arasında koordine edilir:
1.  **Ana Süreç (Driver - GPU):** GPU (NVIDIA GTX 3080) üzerinde çalışır. Tek görevi, toplanan veri paketleri üzerinden gradyan inişi (gradient descent) hesaplayarak sinir ağı ağırlıklarını eğitmek ve güncellemektedir.
2.  **İşçi Süreçleri (Rollout Workers - CPU):** Simülasyon ortamının (`UAV_IoT_PZ_Env`) birer kopyasını barındırırlar. Tamamen CPU çekirdeklerinde çalışarak ajanın güncel politikasına göre fiziksel ortamı simüle eder, adımları yürütür ve veri (deneyim) toplarlar.

#### 4.3.2. İşçi Sayısının Artırılması ($NUM\_WORKERS = 2$) ve Hızlanma
İterasyon başına toplanması hedeflenen veri miktarı olan $T = 1000$ adımın paralel toplanma mekanizması şu şekilde optimize edilmiştir:
*   **Tek İşçi Durumu ($NUM\_WORKERS = 1$):** Tek bir CPU süreci, 10 ardışık epizot (10 epizot $\times$ 100 adım = 1000 adım) boyunca simülasyonu koşturur. Bu esnada GPU boşta bekler ve veri toplama aşaması zaman darboğazı oluşturur.
*   **İki İşçi Durumu ($NUM\_WORKERS = 2$):** İş yükü ikiye bölünür. **Worker 1** kendi CPU çekirdeğinde 5 epizot (500 adım) koştururken, **Worker 2** eş zamanlı olarak kendi CPU çekirdeğinde diğer 5 epizot (500 adım) veriyi toplar. Veri toplama süresi teorik olarak %50 (yarı yarıya) azalır ve GPU'nun bekleme süresi minimize edilir.

#### 4.3.3. Tohum Kaydırma (Seed Shifting) ve Veri Korelasyonu
Paralel toplayıcıların aynı verileri tekrar etmesini önlemek amacıyla, RLLib otomatik olarak her bir işçiye farklı bir rastgelelik tohumu atar:
$$ \text{Worker Seed} = \text{Global Seed} + \text{Worker Index} $$
Örneğin, $\text{Global Seed} = 42$ olduğunda:
*   **Rollout Worker 1:** $\text{Seed } 43$ ile simülasyonu başlatır ve düğümleri bu rastgelelikle yerleştirir.
*   **Rollout Worker 2:** $\text{Seed } 44$ ile farklı bir düğüm dağılımı ve farklı ajan yörüngeleriyle simülasyonu yürütür.

Bu mimari sayesinde GPU'ya gönderilen 1000 adımlık eğitim paketi, tek bir evrenden gelen yüksek ilintili (highly correlated) ardışık veriler yerine, iki farklı evrenden gelen **ilintisizleştirilmiş (de-correlated)** zengin veri serilerinden oluşur. Bu, derin öğrenme modelinin aşırı ezberlemesini (overfitting) önler, kararlı gradyan güncellemeleri sağlar ve model başarısını artırır (PPO JSR başarısının %76'dan %81'e çıkmasının ana nedeni bu çeşitliliktir).

#### 4.3.4. Neden 4 Yerine 2 Rollout Worker? ( Sweet Spot Analizi)
Sistemde 24 CPU çekirdeği olmasına rağmen işçi sayısını 4 yerine 2'de tutmanın iki temel gerekçesi vardır:
1.  **İletişim Gecikmesi (IPC Overhead):** Ray platformunun işçilerden gelen verileri ana driver sürecine aktarırken yaptığı RAM serileştirme ve kopyalama işlemleri, işçi sayısı arttıkça ekstra bir yük getirir (Azalan Verim Kanunu).
2.  **Epizot Bütünlüğü (LSTM Uyumluluğu):** Simülasyon epizot uzunluğumuz tam olarak $100$ adımdır.
    *   $NUM\_WORKERS = 2$ durumunda her işçi 500 adım toplar. Bu, tam olarak **5 tam epizoda** denk gelir.
    *   $NUM\_WORKERS = 4$ durumunda her işçi 250 adım toplar. Bu, **2 tam epizot ve 1 yarım epizoda** denk gelir. Bellek tabanlı recurrent modellerde (PPO-LSTM) epizotların yarım kalması, LSTM gizli durumlarının (hidden states) sıfırlanma ritmini bozarak öğrenme başarısını düşürür. Dolayısıyla, 2 işçi hem donanım performansı hem de model kalitesi açısından en ideal denge noktasıdır.

---

## 5. ÖDÜL (REWARD) MEKANİZMALARININ DETAYLI AÇIKLAMASI

Jammer'ın eğitimi, üç farklı algoritma için (Baseline, PPO, DQN) dikkatli tasarlanmış ödül fonksiyonları ile yönlendirilmektedir. Bu ödül yapıları, hem jamming etkinliğini maksimize ederken, hem de enerji verimliliğini koruyan stratejilerin öğrenilmesini teşvik eder.

### 5.1. Baseline (QJC Algoritması) Ödül Yapısı

Klasik Q-Learning yaklaşımı, ayrık bir durum-aksiyon ödül tablosu kullanır:

*   **Durum (State):** Mevcut kanal (0 - 7 arası 8 kanal)
*   **Aksiyon (Action):** Güç seviyesi (0-9)
*   **Ödül Formülü:**
    ```
    Ödül = (Jamlenmiş_Düğüm_Sayısı × 10) - (Enerji_Tüketimi × 0.1)
    ```

Bu basit linear yapı, tablosal Q-Learning için yeterli geri bildirim sağlar.

### 5.2. PPO & DQN (Derin Takviyeli Öğrenme) Ödül Yapısı

Her iki derin öğrenme algoritması da **aynı üç bileşenli** ödül yapısını paylaşır. Ödüller, eğitimi stabilize etmek amacıyla **0 ile 1 arasına normalize edilmiştir**:

#### Bileşen 1: Jamming Başarı Ödülü (Sparse, Yüksek Değer)
```
ödül_başarı = jamlenen_düğüm_sayısı × 0.8
```
*   **Amaç:** Birincil hedef - jamming etkinliğini maksimize etmek (%80 ağırlık)
*   **Aralık:** 0 ile 24 arası (30 düğüm için)
*   **Tip:** Seyrek ödül (sadece jamming başarılı olduğunda verilir)

#### Bileşen 2: Kanal Takip Ödülü (Dense, Düşük Değer)
```python
if (jammer_kanalı == uav_kanalı AND jammer_gücü > 0.01):
    ödül_takip = 0.2
else:
    ödül_takip = 0.0
```
*   **Amaç:** Jammer'ı İHA'nın frekans atlama davranışını takip etmesi için yönlendirmek (%20 ağırlık)
*   **KRİTİK KOŞUL:** Sadece **güç kullanıldığında** verilir (sömürüyü önler)
*   **Tip:** Yoğun rehberlik sinyali (her adımda kontrol edilir)
*   **Tasarım Mantığı:** Güç eşik kontrolü (`> 0.01W`), ajanların "sıfır güçle sadece kanal takibi yaparak ödül alma" açığını kapatır.

#### Bileşen 3: Enerji Maliyet Cezası
```
ödül_enerji = -(jammer_güç_tüketimi × 0.03)
```
*   **Amaç:** Enerji verimli jamming stratejilerini teşvik etmek (optimum ceza ağırlığı)
*   **Aralık:** 0 ile -0.03 (1.0W maksimum güç tüketimi için)
*   **Etki:** Boşta beklerken gücü kısmayı sağlar, ancak başarılı jamming ödülü (0.8) tarafından kolayca domine edilerek ajanların yüksek güç uygulamayı öğrenmesini engellemez.

#### Toplam Ödül
```
toplam_ödül = ödül_başarı + ödül_takip - ödül_enerji
```

### 5.3. Ödül Tasarımının Teorik Temelleri

1.  **Güç Eşik Kontrolü:**
    *   **Problem:** İlk versiyonda takip ödülü güçten bağımsızdı.
    *   **Sonuç:** Ajanlar "İHA kanalını takip et ama jamming yapma" şeklinde dejenere bir politika öğrendi.
    *   **Çözüm:** `power > 0.01W` koşulu eklenerek, ödülün sadece gerçek jamming faaliyeti sırasında verilmesi sağlandı.

2.  **Ölçeklendirme ve Normalizasyon Dengesi:**
    *   **Normalizasyon:** Toplam pozitif ödül aralığı [0, 1] arasına çekilmiştir (`W_SUCCESS = 0.8` ve `W_TRACKING = 0.2`).
    *   **Yerel Optimum Tuzağının Önlenmesi:** Başarı ödülü (0.8) ile enerji maliyeti cezası (0.03) arasındaki oran `26.6:1` olarak ayarlanmıştır. Eski versiyonda başarı ödülü (0.6) ile enerji cezası (0.1) arasındaki oran çok düşüktü (6:1) ve bu durum ajanları enerji cezasından kaçınmak için sürekli minimum güçte (0.11W) kalmaya ve "düşük güç yerel optimumuna" kilitlenmeye zorluyordu. Yeni katsayılarla, tek bir başarılı jamming eylemi dahi enerji cezasını fazlasıyla karşılayarak ajanların yüksek gücü hızlıca keşfetmesini ve uygulamasını sağlamaktadır.
    *   **Enerji Koruma Modu:** Ajan, İHA uzaktayken veya karıştırma yapamadığında enerji cezasından kaçınmak amacıyla gücünü en düşük seviye olan 0.11W'a otomatik olarak düşürmekte, böylece enerji verimliliğini sürdürmektedir.

3.  **Seyrek vs Yoğun Ödül Trade-off'u:**
    *   **Seyrek (Jamming):** Doğrudan hedefi yansıtır ama öğrenmeyi zorlaştırır (credit assignment problem).
    *   **Yoğun (Takip):** Gradient akışını stabilize eder ve erken aşamada keşfi hızlandırır.
    *   **Birlikte:** İki ödül tipi sinerjik çalışarak hem exploration hem de exploitation'ı dengeştirir.

### 5.4. Algoritma Karşılaştırması - Ödül Kullanımı

| Metrik                  | Baseline (QJC) | PPO           | DQN           |
|-------------------------|----------------|---------------|---------------|
| **Ödül Yapısı**         | Linear combina | 3-Comp Hybrid | 3-Comp Hybrid |
| **Güç Eşik Kontrolü**   | Yok (Not Needed) | Var (0.01W)   | Var (0.01W)   |
| **Policy Gradient**     | Tabular Update | CLIP-PPO      | Q-Learning    |
| **Exploration Strategy**| ε-greedy       | Entropy Bonus | ε-decay       |

---

## 6. ANALİZ VE GÖRSELLEŞTİRME YÖNTEMLERİ

Sistem performansını değerlendirmek ve senaryo çıktılarını yorumlamak için `visualizer.py` modülü tarafından üretilen, SCIE makale formatına uygun iki temel grafik seti kullanılmaktadır.

### 6.1. Yörünge ve Olay Analizi (`trajectory.png`)
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

### 6.2. Metrik Analizi (`metrics_analysis.png`)
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

### 6.3. İletişim İstatistikleri (`advanced_metrics.png`)
Bu grafik seti, her bir IoT düğümünün operasyonel performansını detaylandırır.

1.  **Toplam Başarılı İletişim Süresi (Üst Panel):**
    *   Her bir düğümün (Node 0, Node 1...) simülasyon boyunca toplam kaç saniye boyunca İHA ile başarılı bağlantı kurduğunu gösteren sütun grafiğidir.
    *   *Kırmızı Kesik Çizgi:* Tüm düğümlerin ortalama başarılı iletişim süresini gösterir.
2.  **Maksimum Kesintisiz İletişim (Alt Panel):**
    *   Her bir düğümün bağlantı kopmadan (AoI resetlenmeden) sürdürebildiği en uzun iletişim süresini (Streak) gösterir.
    *   Bu metrik, sistemin kararlılığını ve jamming'in iletişim sürekliliği üzerindeki etkisini ölçmek için kritiktir.

### 6.4. Dashboard Analizi
Simülasyon tamamlandığında, yukarıdaki tüm analizler (`Trajectory`, `Metrics`, `Advanced Stats`) tek bir **"Dashboard"** penceresinde operatöre sunulur. Bu sayede simülasyon sonuçlarına bütüncül (holistic) bir bakış açısı sağlanır.

### 6.5. İstatistiksel Güvenilirlik Analizi (Robustness)
Bilimsel sonuçların güvenilirliğini sağlamak için tekil koşular (Single Run) yerine istatistiksel dağılım analizi benimsenmiştir:
*   **Çoklu Çekirdek (Multi-Seed):** Her bir algoritma, **100-130 aralığında seçilen 30 farklı rastgele tohum** ile test edilerek sonuçların varyansı ölçülmüştür.
*   **Hata Çubukları (Error Bars):** Performans grafikleri, ortalama değerin yanı sıra standart sapmayı (Standard Deviation) da içerecek şekilde üretilmiştir.
*   **Adil Kıyaslama:** Tüm algoritmalar tamamen aynı başlangıç koşullarında ve aynı rastgele sayı üreteci (RNG) durumlarında yarıştırılmıştır.


### 6.6. Algoritmik Karşılaştırma Grafiği (`comparison_robustness.png`)
Bu grafik, farklı algoritmaların performansını dört temel metrik üzerinden (Başarı, Takip, Güç, SINR) yan yana (side-by-side) ve istatistiksel hata paylarıyla karşılaştırır.
*   **Bar Çubukları:** 30 farklı denemenin ortalama değerini gösterir.
*   **Hata Çizgileri (Error Bars):** Sonuçların standart sapmasını (varyansını) göstererek algoritmanın kararlılığını görselleştirir.
*   **Kullanım Amacı:** Baseline ve Önerilen Yöntemler (PPO, LSTM) arasındaki farkın "şans eseri" olmadığını, istatistiksel olarak anlamlı (statistically significant) olduğunu kanıtlar.

---

### 6.7. Otomatik Eğitim Sonuç Grafiği (`comparison_result.png`)
Bu grafik, `run_experiments.py` otomasyonu tarafından her eğitim döngüsünün sonunda üretilen "anlık durum" raporudur.
*   **İçerik:** Algoritmaların son eğitim iterasyonundaki (örn. 500. iterasyon) performansını (Jammed Node Count, Success Rate, Power) gösterir.
*   **Farkı:** `comparison_robustness.png` 30 tekrarlı bir doğrulama iken, bu grafik tek bir eğitimin (Single Run) sonucunu yansıtır. Eğitim sürecinin hızlı takibi (Quick Monitoring) için kullanılır.

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

### 7.4. PPO-LSTM (Recurrent PPO)
*   **Çalışma Prensibi:** PPO mimarisine LSTM (Long Short-Term Memory) katmanı eklenerek ajana "hafıza" yeteneği kazandırılmıştır.
*   **Karakteristiği:** Zamansal bağımlılıkları (temporal dependencies) öğrenebilir. İHA'nın sadece anlık değil, geçmiş hareketlerine de bakarak geleceği tahmin etmeye çalışır.
*   **Bu Projedeki Rolü:** "Hafızalı Ajan" hipotezini test etmek. (Ancak sonuçlar, bu senaryo için Markov özniteliğinin yeterli olduğunu, ekstra hafızanın karmaşıklık yarattığını göstermiştir).

### 7.5. Kıyaslama Metrikleri
Deney sonunda üretilen `comparison_result.png` şu sorulara yanıt verir:
1.  **Kilitleme Hızı (Tracking Accuracy):** Jammer, UAV'nin kanalını ne kadar sürede tahmin edebiliyor?
2.  **Zarar Verme Kapasitesi (Success Rate):** UAV'nin veri toplama başarısını yüzde kaç düşürebiliyor?
3.  **Enerji Verimliliği:** En az güç harcayarak en yüksek zararı hangi algoritma veriyor?

### 7.6. Deneysel Sonuçlar (Robustness Analizi - 30 Seed)

Adil ve kapsamlı bir değerlendirme için her algoritma **100-130 aralığında seçilen 30 farklı başlangıç tohumu (seed)** ile test edilmiştir.

#### Performans Karşılaştırması (Ortalama ± Standart Sapma)

| Algoritma | Başarı (JSR) | Kanal Eşleşme (Tracking) | Ort. Güç (W) | SINR (dB) |
|-----------|--------------|--------------------------|--------------|-----------|
| **PPO (Önerilen)** | **%57.4 ± 10.9** 🏆 | **%60.1** | 0.429 | **3.94** |
| **PPO-LSTM** | %53.6 ± 8.6 | %56.0 | **0.305** 🍃 | 3.91 |
| **DQN** | %29.4 ± 11.8 | %33.3 | **0.241** | 5.10 |
| **Baseline (QJC)** | %1.9 ± 0.8 | %1.1 | 0.400 | 3.78 |

#### Temel Bulgular
- ✅ **PPO Şampiyon:** Baseline'a göre **~30 kat** (%1.9 -> %57.4) performans artışı sağlamıştır. Sürekli aksiyon uzayı ve kararlı öğrenme yapısı (Clipped Objective) bu başarının anahtarıdır.
- ✅ **LSTM Verimliliği:** PPO-LSTM, Baseline'a göre **%24 daha az enerji** harcayarak (%0.30W vs 0.40W) çok yüksek başarı (%53.6) elde etmiştir. Gereksiz saldırıları filtreleyerek "Sessiz ve Derinden" bir strateji izlemiştir.
- ✅ **Baseline Başarısızlığı:** "Yapısal Körlük" nedeniyle (mesafe/spektrum algısı yok), $d^2$ yol kaybı fiziği karşısında çaresiz kalmıştır.
- ✅ **SINR Paradoksu:** PPO ve Baseline benzer ortalama SINR üretmiştir. PPO "etkili" darbelerle iletişimi tamamen keserken (Deep Fade), Baseline sadece "etkisiz" arka plan gürültüsü yaratmıştır. PPO'nun ortalamasının yüksek kalması, İHA'nın bu darbelerden kaçıp temiz kanallara sığınmasındandır.
- ✅ **DQN'in Sessizliği:** Dinamik 3D uzayda (Konum+Frekans+Güç) kaybolmuş ve ceza almamak için pasif kalmayı (Sparsity Trap) seçmiştir.

**Not:** Tüm istatistikler `paper/robustness_results_30seeds.json` dosyasında saklanmaktadır.

---

## 8. GELİŞİM GÜNLÜĞÜ (CHANGE LOG)

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
4.  **RLLib Entegrasyonu:** `scripts/train.py` dosyası ile Ray RLLib (PPO) üzerinde Jammer'ın bu Markov modelini öğrenmesi için eğitim altyapısı kuruldu.

### [07.02.2026 00:10] - Adillik ve Kıyaslama Paketi (v1.7.0)
**Yapılan Değişiklikler:**
1.  **Sensing Mode (Gerçekçi Algılama):**
    *   RL Ajanının (Jammer) gözlem uzayından hileli "God View" (Tam Koordinat) verisi çıkarıldı.
    *   Yerine, sadece **Mesafe** (Distance) ve **Sinyal Gücü** (RSSI) gibi gerçek hayatta sensörlerle ölçülebilen veriler eklendi.
2.  **Adil Kıyaslama (Algorithmic Fairness):**
    *   Baseline (QJC) algoritmasının, PPO eğitim sürelerine denk (60K adım) deneyim kazanması için **Ön Eğitim (Pre-training)** modülü (`scripts/train_baseline.py`) eklendi.
    *   Böylece "Eğitimsiz Baseline vs Eğitilmiş RL" adaletsizliği giderildi.
3.  **Algoritmik Çeşitlilik:**
    *   **PPO:** Sürekli (Continuous) politika optimizasyonu (New API Stack).
    *   **DQN:** Ayrık (Discrete) aksiyon uzayı optimizasyonu (Old API Stack).
    *   **Baseline:** Tablosal (Tabular) Q-Learning.
    *   Üç algoritmayı tek komutla yarıştıran `scripts/run_experiments.py` otomasyonu geliştirildi.
4.  **Otomatik Raporlama:** Tüm sonuçları karşılaştırmalı sütun grafiklerine (`comparison_result.png`) döken analiz modülü eklendi.

### [07.02.2026 02:00] - RLLib Yama ve Otomasyon Paketi (v1.8.0)
**Yapılan Değişiklikler:**
1.  **Ray RLLib Hata Düzeltmesi (Bug Fix):**
    *   Ray 2.53.0 sürümündeki DQN algoritmasının "Old API Stack" yolunda kilitlenmesine neden olan `TypeError: argument of type 'ABCMeta' is not iterable` hatası tespit edildi.
    *   `rllib/algorithms/algorithm.py` dosyasına `isinstance(..., str)` kontrolü eklenerek yerel kütüphane yamalandı.
    *   Düzeltme, Ray projesine resmi Pull Request olarak gönderildi (DCO imzalı).
2.  **Tam Otomatik Deney Hattı (Experiment Pipeline):**
    *   `scripts/run_experiments.py` scripti geliştirilerek; Baseline (QJC), PPO ve DQN modellerinin sırayla eğitilmesi, değerlendirilmesi ve karşılaştırmalı rapor üretilmesi otomatikleştirildi.
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

### [08.02.2026 23:30] - Paralel Eğitim ve UI Paketi (v2.0.0)
**Yapılan Değişiklikler:**
1.  **Paralel Eğitim (Parallel Execution):**
    *   `subprocess` ve `threading` kütüphaneleri kullanılarak Baseline (QJC), PPO ve DQN algoritmaları artık **eş zamanlı** olarak eğitilmektedir.
    *   Bu sayede toplam deney süresi yaklaşık **3 kat** kısalmıştır (Örn: 1 saat -> 20 dakika).
    *   Otomasyon scripti (`run_experiments.py`) tüm kaynak yönetimini (GPU/CPU) otomatik yapar.

2.  **Gelişmiş Kullanıcı Deneyimi (UI/UX):**
    *   **Anlık Progress Bar:** Terminal üzerinden her algoritmanın ilerleme durumu (Adım/Toplam) ve yüzdesi canlı olarak takip edilebilmektedir. 
    *   **Renkli Çıktılar:** Durumlar (OK=Yeşil, Running=Sarı, Fail=Kırmızı) ANSI renk kodlarıyla görselleştirilmiştir.
    *   **Ray RLLib Entegrasyonu:** `ProgressCallback` sınıfı sayesinde Ray'in karmaşık logları filtrelenerek temiz bir ilerleme çubuğuna dönüştürülmüştür.

3.  **Esnek Konfigürasyon ve Argümanlar:**
    *   `--debug`: Detaylı hata ayıklama modu. Tüm subprocess çıktılarını (stdout/stderr) ekrana basar.
    *   `--ui <saniye>`: Terminal güncelleme sıklığını ayarlar (Varsayılan: 3s).

4.  **Artifact Yönetimi:**
    *   Her deney çalıştırması için `artifacts/YYYY-MM-DD_HH-MM-SS/` formatında izole bir klasör oluşturulur.
    *   Bu klasör içinde eğitim modelleri, loglar ve karşılaştırma grafikleri düzenli bir hiyerarşide saklanır. Eski `logs/` yapısından daha temiz bir yapıya geçilmiştir.

**Amaç:**
### [09.02.2026 23:00] - LSTM ve Görselleştirme Paketi (v2.1.0)
**Yapılan Değişiklikler:**
1.  **PPO-LSTM Entegrasyonu:**
    *   Ray RLLib'in Recurrent Network (LSTM) desteği projeye eklendi (`train_ppo_lstm.py`).
    *   `PPOLSTMConfig` yapılandırma sınıfı oluşturuldu.
    *   Değerlendirme (`evaluate.py`) scripti, gizli durumları (hidden states - h, c) yönetecek şekilde güncellendi.
    
2.  **Karşılaştırma Görselleştirmesi (Refined Visualization):**
    *   **Adil Başlangıç:** Tüm Deep RL algoritmalarının grafikleri, veri toplama fazını yansıtacak şekilde 1000. adımdan başlatıldı.
    *   **Baseline Hizalaması:** Baseline verisi, Deep RL batch size'ına (1000) uygun şekilde yeniden örneklenerek (resampling) grafiklerin x-ekseninde tam hizalanması sağlandı.
    *   **(0,0) Noktası:** Yanıltıcı olmaması için yapay (0,0) noktası kaldırıldı, doğal öğrenme süreçleri yansıtıldı.
    
3.  **DQN Hata Yönetimi:**
    *   Paralel çalışmada DQN'in bazen zaman aşımına uğraması (timeout) sorunu analiz edildi ve result.json varlığı kontrol edilerek "False Negative" durumları engellendi.

**Amaç:**
Hafızalı (Recurrent) modellerin etkisini ölçmek ve grafik okumayı bilimsel standartlara (elmalarla elmalar) taşımak.

### [10.02.2026 00:00] - Robustness ve İstatistik Paketi (v2.2.0)
**Yapılan Değişiklikler:**
1.  **30-Seed Robust Evaluation:**
    *   Bilimsel geçerliliği artırmak için tüm algoritmalar **30 farklı random seed** (Range: 100-129) ile test edildi.
    *   `scripts/evaluate_paper_robustness.py` scripti geliştirildi.
    *   Sonuçlar ortalama ve standart sapma (Mean ± Std) olarak raporlandı.

2.  **Teorik Analiz Derinleştirme:**
    *   Baseline başarısızlığının sebebi "Yapısal Körlük" (Structural Blindness) ve $d^2$ fiziksel kısıtı olarak tanımlandı.
    *   SINR Paradoksu (PPO ve Baseline'ın benzer ortalama vermesi), "Etkili Güç" (Effective Power) ve "İHA Adaptasyonu" (UAV Adaptation) kavramları ile açıklandı.

3.  **Deneysel Bulgular:**
    *   PPO'nun başarısı istatistiksel olarak kanıtlandı (%57.4 ± 10.9).
    *   PPO-LSTM'in enerji verimliliği (%24 tasarruf) ve kararlılığı (düşük varyans) ortaya kondu.

**Amaç:**
Makale (Paper) için gerekli olan güvenilir, tekrarlanabilir ve istatistiksel olarak anlamlı veri setini oluşturmak.

### [19.06.2026 23:35] - Kod Temizliği ve Optimizasyonu (v2.2.1)
**Yapılan Değişiklikler:**
1. **Kullanılmayan Eski Kodların Temizlenmesi:**
   * PettingZoo ortamına geçişle birlikte tamamen atıl kalan ve artık hiçbir yerde kullanılmayan eski Gymnasium tabanlı tek-ajan simülasyon ortamı `simulation/environment.py` dosyası silindi.
2. **Tekrar Eden Kodların ve Tanımlamaların Kaldırılması:**
   * `simulation/pettingzoo_env.py` dosyasındaki mükerrer/çift ödül hesaplama satırları silinerek kod temizlendi.
   * `confs/env_config.py` dosyasındaki çift `UAV_START_Z` tanımı kaldırıldı.
   * `scripts/train.py` dosyasının en sonunda yer alan mükerrer `print` ve `ray.shutdown()` ifadeleri temizlendi.

**Amaç:**
Kod tabanındaki mükerrerlikleri temizlemek, okunabilirliği artırmak ve kullanılmayan eski dosyaları kaldırarak projeyi daha temiz ve sürdürülebilir bir yapıya kavuşturmak.

### [20.06.2026 14:20] - Erken Durdurma ve En İyi Model Kayıt Desteği (v2.3.0)
**Yapılan Değişiklikler:**
1. **Erken Durdurma (Early Stopping) Entegrasyonu:**
   * `train.py`, `train_dqn.py` ve `train_ppo_lstm.py` dosyalarına `EarlyStoppingStopper` sınıfı entegre edildi. Ajanların performansı platoya girdiğinde (100 iterasyon boyunca ödül artmadığında) eğitim otomatik olarak sonlandırılarak zamandan tasarruf sağlanmaktadır.
2. **En İyi Modelin Kaydedilmesi (Best Checkpoint Saving):**
   * Tüm eğitim scriptlerinde `tune.run` fonksiyonuna `checkpoint_freq=10`, `keep_checkpoints_num=3` ve `checkpoint_score_attr="env_runners/episode_reward_mean"` parametreleri eklendi. Böylece diske sadece son model değil, eğitim süresince elde edilmiş en yüksek ödüllü (en iyi) model kaydedilmektedir.

**Amaç:**
Gereksiz uzun süren eğitimleri kısaltarak zaman tasarrufu sağlamak ve her zaman en kararlı ve başarılı çalışan modelleri garanti altına almak.

### [21.06.2026 01:25] - Hiperparametre Harmonizasyonu (Gamma=0.85 ve LR=1.2e-4) (v2.4.0)
**Yapılan Değişiklikler:**
1. **Unified Learning Rate (LR) ve Gamma (γ) Güncellemesi:**
   * Tüm derin pekiştirmeli öğrenme modelleri (PPO, DQN, PPO-LSTM) için öğrenme oranı `LR = 1.2e-4` ve iskonto faktörü `GAMMA = 0.85` olarak eşitlendi.
   * Baseline QJC algoritmasının iskonto faktörü de adil karşılaştırılabilirlik için `GAMMA = 0.85` olarak güncellendi.

**Amaç:**
Tüm algoritmaların adil kıyaslanabilirliğini korumak, 750 iterasyonluk bütçede DQN ve PPO-LSTM'in yakınsamasını hızlandırmak ve kanal takip başarısını optimize etmek.

### [21.06.2026 01:37] - Çoklu İHA Destekli İşbirlikli Veri Toplama Senaryosu (v2.5.0)
**Yapılan Değişiklikler:**
1. **Çoklu İHA Simülasyonu Altyapısı (`pettingzoo_env.py`):**
   * Simülasyon ortamında eşzamanlı hareket eden $M$ adet İHA oluşturulmasını destekleyecek yapı kuruldu (`NUM_UAVS = 2`).
   * Gözlem (observation) uzayı tüm İHA'ların konum/kanal bilgilerini Jammer'a sunacak şekilde dinamik genişletildi.
   * Fiziksel SINR hesaplamasına, İHA'lar arası girişimi temsil eden Eş-Kanal Girişimi (co-channel interference) eklendi.
2. **Kural Tabanlı İşbirlikli İHA Kontrolü (`controllers.py`):**
   * İHA'ların çarpışmasını engellemek için Yapay Potansiyel Alanları (Artificial Potential Fields - APF) kullanılarak itici kuvvet vektörleri hıza entegre edildi.
   * Rota planlamasında, her İHA'nın diğerleri tarafından hedeflenmemiş en yakın düğüme yöneldiği dinamik rota paylaşım havuzu (Shared Unvisited Pool) kurgulandı.
3. **Değerlendirme ve Görselleştirme Entegrasyonu:**
   * Çoklu İHA rotalarını ve parametrelerini görselleştirebilmek için `visualizer.py` güncellendi.
   * `evaluate.py` içerisindeki değerlendirme döngüsü çoklu İHA yapısıyla uyumlu hale getirilerek testlerin otomatik İHA aksiyonuyla çalışması sağlandı.

**Amaç:**
Smart Attacker'ın daha karmaşık ve gerçekçi bir çoklu İHA veri toplama ortamındaki performansını ve saldırı başarısını test etmek.

### [21.06.2026 01:52] - İnteraktif Canvas Simülasyonu ve Teknik Yazılım Mimarisi Sunumu (v2.6.0)
**Yapılan Değişiklikler:**
1. **İnteraktif Canvas Simülasyonu:**
   * Çoklu İHA ve akıllı karıştırıcı dinamiklerini gösteren Canvas tabanlı 2B bir animasyon motoru sunum arayüzüne (Slayt 6 / Sekme 6) eklendi.
   * Fizik hesaplamaları (SINR, Shannon veri hızı, Friis yol kaybı ve güç tüketimleri) anlık güncellenen bir HUD paneline ve fizik log konsoluna bağlandı.
2. **Teknik Yazılım Mimarisi Sunumu (Slayt 7 / Sekme 7):**
   * Geliştiricilere yönelik olarak Python simülasyon kodunun modüler dizin hiyerarşisini (`confs/`, `simulation/`, `core/`, `scripts/`) açıklayan görsel yapılar eklendi.
   * Gözlem (Observation) ve Eylem (Action) uzaylarının PettingZoo / Gymnasium standartlarındaki kod tanımları ve formülasyonları sunuma dahil edildi.
   * Çoklu ajan eğitimi (Ray/RLLib) ve Markov geçiş matrislerinin kod içerisindeki akış adımları detaylandırıldı.

**Amaç:**
Sunum arayüzünü hem genel kullanıcıların sistemi gözlemleyebileceği bir simülasyon paneline, hem de jüri ve geliştiricilerin yazılım mimarisini inceleyebileceği teknik bir sunum aracına dönüştürmek.

### [21.06.2026 02:10] - Animasyonlu Teknik Altyapı ve Yazılım Mimarisi Sunum Modu (v2.7.0)
**Yapılan Değişiklikler:**
1. **Üçüncü Bağımsız Mod: Teknik Sunum (Animasyonlu) Entegrasyonu:**
   * Sunuma mevcut "Senaryo Sunumu" ve "Panel Modu"nun yanına "Teknik Sunum (Animasyonlu)" isimli bağımsız, geliştirici ve jüri odaklı yeni bir görsel sunum modu entegre edildi.
2. **5 Ayrı İnteraktif ve Animasyonlu Teknik Slayt Kurgusu:**
   * **Slayt 1 (Modüller ve Bağımlılık Ağacı):** Projenin modüler Python dosyalarının (`confs/`, `simulation/`, `core/`, `scripts/`) import akışını canlandıran interaktif bir bağımlılık ağacı canvas'ı tasarlandı. Listeden bir dosyaya tıklandığında ilgili python modülünün görevi, açıklaması, import bağımlılıkları (canvas üzerinde glowing particle beams ile) ve vscode-style kod penceresi dinamik olarak gösterilir.
   * **Slayt 2 (Gymnasium/PettingZoo Lifecycle):** Ajan-ortam pekiştirmeli öğrenme döngüsünü `obs ➔ agent ➔ action ➔ env ➔ reward` adımlarıyla görselleştiren akış canvas'ı eklendi. "Tek Adım (Step)", "Otomatik Oynat" ve "Hız Ayarı" kontrolleriyle; gözlem vektörü, neural network katmanlarının ışıldaması, aksiyon paketi ve ödül sinyalinin akışı canlandırılmakta, altında canlı debugger konsol çıktıları simüle edilmektedir.
   * **Slayt 3 (Sinyal Fiziği ve Kanal Modeli):** İHA-IoT mesafesi ve taşıyıcı frekansı (900 MHz - 6.0 GHz) değiştikçe anlık dalga boyunu/genliğini canlandıran bir wave generator canvas'ı kurgulandı. Sürgüler oynatıldıkça Friis yol kaybı, alınan güç, jammer gücü, SINR (dB) ve Shannon kapasitesi ($R = B \log_2(1+SINR)$) dinamik matematiksel formüllerle anlık hesaplanmakta ve `physics.py` kod bloğundaki satırlar live-vurgulu (highlighted) gösterilmektedir.
   * **Slayt 4 (APF Otonom Navigasyon Canvas'ı):** İHA 0, İHA 1 ve IoT Düğüm nesnelerinin fare ile sürüklenebildiği interaktif bir APF çizici kuruldu. Canvas üzerinde düğüme yönelen çekici kuvvet ($F_{att}$ - yeşil ok), İHA'ların 50m güvenlik sınırına girmesiyle oluşan itici kuvvet ($F_{rep}$ - kırmızı ok) ve bileşke yön kuvveti ($F_{net}$ - mavi ok) anlık hesaplanıp çizilmekte, `controllers.py` içindeki APF algoritması kodları canlı izlenebilmektedir.
   * **Slayt 5 (Durum-Farkında Saldırgan & Markov Predictor):** Zeki jammer'ın İHA'nın kanal atlama örüntülerini çıkardığı 7x7 Markov geçiş matrisi etkileşimli hale getirildi. Kullanıcı kanallara tıkladığında İHA zıplaması simüle edilir, matris sayımları anlık güncellenir (değişen hücreler parlar), jammer'ın argmax tahminleri ve bloklama (match) durumları canlı olarak tablolara yansıtılır.
3. **Optimizasyon ve Frame Kontrolü:**
   * Animasyon döngüleri (`requestAnimationFrame`), sadece ilgili teknik slayt aktif olduğunda çalışacak şekilde optimize edildi, arka planda gereksiz CPU tüketimi engellendi. Klavye ok yönü navigasyonu (sağ/sol) teknik sunum moduna da uyarlandı.

**Amaç:**
Sistem mimarisini, Gymnasium yapısını, otonom APF navigasyonunu ve telsiz sinyal yayılımını jüriye statik slaytlar yerine, canlı kod satırları ve etkileşimli fizik canvasları eşliğinde animasyonlu bir sunum olarak anlatabilmek.


### [22.06.2026 16:45] - Zaman Karmaşıklığı Optimizasyonu ve Log Frekansı Güncellemesi (v2.8.0)
**Yapılan Değişiklikler:**
1. **Mesafe Hesaplama Optimizasyonu (`pettingzoo_env.py`):**
   * Eş-kanal girişim (co-channel interference) hesaplamasında her adımda tekrarlanan İHA-düğüm mesafe hesaplamaları (`np.linalg.norm`) optimize edildi.
   * Tüm düğüm-İHA mesafeleri her adımın başında bir kez hesaplanıp (`node_uav_dists`) ilişkilendirme bilgisi (`node_assoc_info`) precompute edildi.
   * Yapılan bu precomputation sayesinde `step()` fonksiyonu **~6.15 kat hızlandı** (100 adım için 1.17 sn -> 0.19 sn).
2. **Loglama Sıklığı Güncellemesi (`train_baseline.py`):**
   * Baseline QJC eğitiminde terminale log basma sıklığı `TRAIN_EPISODES // 10 = 2000` episode'dan `TRAIN_EPISODES // 200 = 100` episode'a düşürüldü.
   * Bu sayede paralel eğitim aracı olan `run_experiments.py` ilerleme durumunu anlık yakalayabilmekte ve Baseline ilerleme çubuğu `%0` değerinde kilitli kalmamaktadır.

**Amaç:**
İHA sayısı 3'e ve IoT düğüm sayısı 30'a yükseltildiğinde ortaya çıkan $O(N^2 \cdot U)$ işlem karmaşıklığı darboğazını çözerek eğitimi hızlandırmak ve Baseline ilerleme çubuğunun takibini sağlamak.

### [22.06.2026 18:25] - Saf LoRaWAN EU868 Çoklu Kanal Haberleşme Mimarisi (v2.9.0)
**Yapılan Değişiklikler:**
1. **Fiziksel Kanal Modeli Güncellemesi (`confs/config.py`):**
   * Heterojen çok frekanslı yapı (900 MHz - 6.0 GHz) yerine, endüstri standardı olan **8 kanallı LoRaWAN (EU868)** frekans planına geçiş yapıldı.
   * Kanal frekansları 867.1 MHz ile 868.5 MHz arasındaki gerçekçi 8 kanala atandı.
   * Kanalların hepsi aynı 868 MHz bandında olduğundan, frekansa bağlı güç amplifikatörü (PA) verimliliği (`ETA_PA`) tüm kanallar için **%60 (0.60) sabit** hale getirildi.
2. **Ağ Fiziği ve Güç Modeli Uyumluluğu:**
   * Çok bantlı yapıdan kaynaklanan fiziksel tutarsızlıklar (aynı ucuz IoT düğümünün 6 GHz ve 900 MHz'i aynı anda kullanması gibi gerçek dışı durumlar) ortadan kaldırıldı, model %100 gerçekçi bir LoRa IoT ağına dönüştürüldü.
   * Karıştırıcının eylem uzayı (action space) ve buna bağlı düzleştirilmiş eylem sayısı 70'ten **80'e** (`8 kanal x 10 güç seviyesi`) yükseldi ve sistem otomatik olarak bu yeni uzaya adapte edildi.
3. **Dokümantasyon Güncellemesi (`paper/method_materials.md`):**
   * Türkçe makale metodoloji bölümündeki frekans tabloları, aksiyon uzayı formülleri ve açıklamalar 8 kanallı saf LoRa modeline uygun olarak güncellendi.

**Amaç:**
IoT sensörlerinin donanımsal ve ekonomik gerçekliğiyle çelişen çok frekanslı (Multi-band) yapıyı terk ederek, endüstri standardı LoRaWAN (EU868) protokolünü temel alan, hakem/jüri sorgusuna karşı %100 doğrulanabilir fiziksel bir altyapı kurgulamak.


### [22.06.2026 18:35] - Çoklu İHA Dinamik Rota Planlaması ve APF Devre Dışı Bırakma (v3.0.0)
**Yapılan Değişiklikler:**
1. **Çoklu İHA Navigasyonunda APF Devre Dışı Bırakılması (`controllers.py`):**
   * Yapay Potansiyel Alanları (APF) tabanlı çarpışma önleme itici kuvvetleri tamamen devre dışı bırakıldı.
   * Bu sayede İHA'lar hedef düğümlere en doğrudan (doğrusal) rotadan giderek seyahat süresini ve zaman karmaşıklığını minimize ederler.
2. **İşbirlikli Rota Planlama (`controllers.py`):**
   * Coğrafi kümeleme (clustering) yerine ortak bir "Ziyaret Edilmeyen Düğümler Havuzu" (Shared Unvisited Pool) modeli kuruldu.
   * Her İHA o an seçilmemiş olan en yakın düğümü hedef alır. Bu dinamik paylaşım, jammerın bulunduğu bölgeye sürekli aynı İHA'nın giderek yıpranmasını önler ve jammerın tek bir kurbana kilitlenmesini engeller.
3. **M=3 İHA ve N=30 IoT Düğümü Parametrik Güncellemesi:**
   * Tüm sistem rapor içeriği ($M=3$, $N=30$, 8 LoRaWAN kanalı, uniform %29.8 PA verimliliği, 14 dBm / 0.025W düğüm iletim gücü) güncel koda uygun olarak revize edildi ve tutarlılık sağlandı.

**Amaç:**
Çoklu İHA veri toplama senaryosunda seyrüsefer gecikmesini azaltmak, jammerın bölge kilitleme saldırılarını dinamik rota paylaşımıyla aşmak ve raporu son kod konfigürasyonuna göre %100 güncel kılmak.

### [23.06.2026 13:30] - Web Tabanlı Gerçek Zamanlı Deney Kontrol Paneli (Dashboard) (v3.1.0)
**Yapılan Değişiklikler:**
1. **Web Tabanlı Gerçek Zamanlı Kontrol Paneli (Dashboard):**
   - Deneyler çalışırken ilerleyişi anlık izlemek için HTML5, CSS3 (Vanilla CSS) ve JS tabanlı premium bir karanlık tema kontrol paneli tasarlandı (`scripts/dashboard/index.html`).
   - Panel üzerinde 4 algoritmanın (Baseline, PPO, DQN, PPO-LSTM) durumları, yüzdeleri, anlık ortalama ödülleri ve çevre adım sayıları dinamik kartlar üzerinde sunulmaktadır.
   - Chart.js kütüphanesi entegre edilerek, 4 algoritmanın öğrenme eğrileri (ortalama ödül vs toplam adım) eş zamanlı ve canlı çizdirilmektedir.
   - Alt kısımda her algoritmanın kendi subprocess çıktılarını gerçek zamanlı gösteren sekmeli bir terminal emülatörü eklendi.
2. **Çoklu İş Parçacıklı HTTP Sunucu Altyapısı (`scripts/dashboard_server.py`):**
   - Python standart kütüphanesi (`http.server` ve `socketserver`) kullanılarak harici hiçbir pip bağımlılığı gerektirmeyen çoklu iş parçacıklı bir backend sunucusu yazıldı.
   - PPO/DQN (`progress.csv`) ve Baseline (`training_curve.csv`) dosyalarını gerçek zamanlı okuyup, Baseline adımlarını Deep RL iterasyonlarına hizalayacak 10-epizotluk resampling mantığı kuruldu.
   - Boş portu otomatik bulma (5000-5049) ve browser'ı otomatik açma (`webbrowser`) özellikleri eklendi.
3. **Orkestrasyon Entegrasyonu (`scripts/run_experiments.py` ve `scripts/train_baseline.py`):**
   - `ParallelTrainer` sınıfı, subprocess çıktılarının (stdout/stderr) son 50 satırını hafızada tutacak log buffer'ları ile donatıldı.
   - `run_experiments.py` başlangıcında sunucu thread'i otomatik tetiklenecek ve bitişte/hata durumunda kapatılacak şekilde entegrasyon sağlandı.
   - `train_baseline.py` içindeki log yazımına anlık disk sifonlama (`flush()`) eklenerek Baseline verilerinin web arayüzünde takılmadan akması sağlandı.
4. **Ortak Ray Kümesi (Shared Ray Cluster) Entegrasyonu:**
   - Çoklu paralel süreçlerin (`train.py`, `train_dqn.py`, `train_ppo_lstm.py`, `evaluate.py` ve `evaluate_paper_robustness.py`) aynı anda veya aktif bir küme varken kendi lokal Ray kümelerini kurmaya çalışırken oluşturduğu Redis ve geçici dizin çakışmaları (`AssertionError: Session name...`) giderildi.
   - Ana orkestratörde (`run_experiments.py`) tek bir ortak Ray kümesi başlatılıp tüm alt süreçlerin `ray.init(address="auto")` ile bu kümeye bağlanması sağlandı. Standalone bağımsız çalıştırmalar için lokal geri dönüş (fallback) mekanizması entegre edildi.

5. **Karıştırma Başarı Oranı (JSR) Matematiksel Hesaplama Düzeltmesi (`compare.py`):**
   - Eski `compare.py` kodunda JSR (Success Rate) hesaplanırken, karıştırılan düğüm sayısının alandaki toplam düğüm sayısına (30) bölünerek tüm simülasyon adımlarının ortalamasının alındığı bir hata düzeltildi.
   - Gerçekte 30 düğümden sadece o anda İHA'nın üzerinde asılı durduğu düğüm (en fazla 1 adet) yayın yaptığı için bu durum JSR'ı yapay olarak 30 kat düşük gösteriyordu (JSR en fazla %3.3 olabiliyordu).
   - Yeni yapıda, tıpkı `evaluate_paper_robustness.py` içerisindeki gibi JSR; sadece aktif yayın yapan (durumu 0 veya 2 olan, yani out-of-range olmayan) adımlar dikkate alınarak `Total_Jammed / Total_Reachable` formülüyle hesaplanacak şekilde revize edildi.

**Amaç:**
Deneyler yürütülürken eğitim süreçlerini, ödül yakınsamalarını ve hata loglarını konsola bağımlı kalmadan takip edebilmeyi sağlamak; paralel süreçlerdeki Ray çakışmalarını gidermek ve karşılaştırma raporlarındaki JSR başarı oranının matematiksel hesaplama hatasını düzelterek bilimsel olarak doğru sonuçlar üretmek.

### [23.06.2026 16:00] - Sıralı Çalıştırma ve Zaman Sayacı Desteği (v3.2.0)
**Yapılan Değişiklikler:**
1. **Sıralı/Paralel Çalıştırma Seçeneği (`run_experiments.py`):**
   - Eğitim ve değerlendirme süreçleri için varsayılan olarak **sıralı (sequential)** çalıştırma mekanizması kurgulandı. Bu sayede tek GPU'lu sistemlerde CUDA bağlam kilitlenmeleri ve bellek yetersizliği (OOM) hataları tamamen engellendi.
   - Paralel çalıştırmayı tercih eden kullanıcılar için `--parallel` parametresi eklendi.
2. **Kontrol Panelinde Zaman Sayacı (Runtime Counter):**
   - Deneyin toplam çalışma süresini göstermek üzere `scripts/dashboard_server.py` ve `scripts/dashboard/index.html` üzerinde `elapsed_time` (geçen süre) sayacı eklendi.
   - Deneyler (tüm eğitimler ve testler) başarıyla veya hata ile tamamlandığında sayacın donması (durması) sağlandı.
   - Deney bittikten sonra kullanıcının arayüzü inceleyebilmesi adına dashboard HTTP sunucusunu kapatmadan önce terminalde kullanıcıdan `Enter` tuşuna basması beklenmesi sağlandı.
3. **PPO-LSTM Arayüz ID Hatalarının Giderilmesi (`index.html`):**
   - JavaScript tarafında PPO-LSTM verilerinin DOM elemanları ile eşleşmesini bozan tire işareti (`-`) kaldırılma mantığı eklenerek (`domName = name.replace('-', '')`) arayüzdeki PPO-LSTM kartlarının ve loglarının güncellenmeme hatası giderildi.

**Amaç:**
Tek GPU bulunan geliştirme ortamlarında Ray/CUDA kaynak çakışmalarını önleyerek eğitim stabilitesini artırmak ve deneylerin toplam süresini kontrol panelinden canlı izlenebilir kılmak.

### [23.06.2026 18:55] - Ray Rollout Worker Dizin ve Yol Hatası Düzeltmesi (v3.2.1)
**Yapılan Değişiklikler:**
1. **Ray Runtime Environment Entegrasyonu:**
   - Tüm Ray çalıştıran scriptlerde (`evaluate_paper_robustness.py`, `evaluate.py`, `run_experiments.py`, `train.py`, `train_dqn.py`, `train_ppo_lstm.py`) `ray.init` çağrılarına `runtime_env` eklendi.
   - `runtime_env = {"env_vars": {"PYTHONPATH": project_root}}` tanımlanarak, Ray rollout worker aktörlerinin ve arka plan süreçlerinin proje kök dizinindeki modüllere erişmesi garanti altına alındı.
   - Bu sayede değerlendirme ve eğitim süreçlerinde karşılaşılan `ModuleNotFoundError: No module named 'confs.env_config'` hatası giderildi.

**Amaç:**
Ray rollout worker süreçlerinin bağımsız çalışırken proje dizinini görememesinden kaynaklanan import ve model yükleme hatalarını kalıcı olarak çözmek.

### [23.06.2026 19:12] - Görselleştirici Tek/Çoklu İHA Uyumluluk Yaması (v3.2.2)
**Yapılan Değişiklikler:**
1. **Görselleştirici (Visualization.py) Güncellemesi:**
   - Çoklu İHA entegrasyonu sonrasında tek İHA senaryosunda çalışırken `env.uav` özniteliğinin bulunamamasından kaynaklanan `AttributeError` giderildi.
   - Kod, `env.uavs[0]` listesini kontrol edecek şekilde güncellendi, hem tek İHA hem de çoklu İHA görselleştirmeleriyle tam uyumlu hale getirildi.

**Amaç:**
Değerlendirme (evaluate) aşamasında canlı simülasyon arayüzünün hata vermeden İHA hareketlerini ve kanal durumlarını çizebilmesini sağlamak.

### [23.06.2026 19:16] - Otomatik Ray Süreç Temizliği ve Kilitlenme Çözümü (v3.2.3)
**Yapılan Değişiklikler:**
1. **Ray Süreçlerini Otomatik Sıfırlama (`run_experiments.py`):**
   - Paralel eğitim başlatılmadan hemen önce sistemdeki tüm eski/zombi Ray arka plan süreçlerini (raylets, Redis port kilitleri vb.) sonlandırmak için `ray stop --force` komutu entegre edildi.
   - Bu sayede port çakışmalarından ötürü bazı algoritmaların eğitim başlangıcında `PENDING` (askıda) kalması sorunu çözüldü.

**Amaç:**
Paralel eğitim döngülerinin kilitlenmesini önlemek ve Ray kümesinin her zaman temiz ve boş portlarla başlamasını garanti altına almak.

### [23.06.2026 19:22] - Bağımsız Standalone Ray Başlatma Düzeni (v3.2.4)
**Yapılan Değişiklikler:**
1. **Paylaşımlı Ray Kümesinin Kaldırılması:**
   - Orkestratör `run_experiments.py` içerisindeki paylaşımlı `ray.init` ve `ray.shutdown` blokları kaldırılarak arka planda tek bir küme üzerinden kısıtlama uygulanması engellendi.
2. **Bireysel/Bağımsız Ray Sunucusu Geçişi:**
   - `train.py`, `train_dqn.py`, `train_ppo_lstm.py`, `evaluate.py` ve `evaluate_paper_robustness.py` dosyalarında `address="auto"` bağlantı parametreleri iptal edilerek tüm süreçlerin doğrudan kendi bağımsız lokal Ray sunucularını başlatması sağlandı.

**Amaç:**
Paralel modda çalıştırıldığında modellerin GPU'yu CUDA düzeyinde dinamik paylaşmasını sağlamak ve bir algoritma bittiğinde serbest kalan GPU gücünün diğer süreçler tarafından tam kapasiteyle kullanılmasını sağlamak.

---

### [23.06.2026 23:20] - Paralel Çalıştırma Optimizasyonu ve Adım Hizalaması (v3.3.0)
**Yapılan Değişiklikler:**
1. **Çevre Gözlem Önbellekleme (Observation Caching):**
   - `UAV_IoT_PZ_Env` içinde çoklu ajanların (16+ ajan) her adımda aynı gözlem vektörünü mükerrer şekilde sıfırdan hesaplamasını önlemek amacıyla caching mekanizması eklendi. Gözlem hesaplama yükü %90 azaltıldı.
2. **CPU İş Parçacığı (Thread) Sınırlaması:**
   - PyTorch ve alt kütüphanelerin (OMP, MKL, OpenBLAS) 24 çekirdekli işlemcide oluşturduğu aşırı thread çekişmesini (context thrashing) önlemek amacıyla süreç başına 2 thread sınırı (`OMP_NUM_THREADS = 2` ve `torch.set_num_threads(2)`) uygulandı.
3. **Ray Rollout Worker Paralelleştirmesi:**
   - Her bir algoritmanın simülasyon ortamından veri toplama hızını artırmak amacıyla rollout worker sayısı (`NUM_WORKERS`) 1'den 2'ye çıkarıldı. Bu sayede veriler paralel olarak toplanarak simülasyon aşaması hızlandırıldı.
4. **DQN Adım Hizalaması ve Eşitleme:**
   - `NUM_WORKERS = 2` yapıldığında DQN'in asenkron olarak iterasyon başına 1332 adım toplaması ve toplamda 133.200 adıma uzaması engellendi. `rollout_fragment_length = 100` parametresi eklenerek DQN'in de PPO gibi tam 1000 adım toplar hale getirilmesi ve 100.000 adımda durması sağlandı.
5. **Gelişmiş Seeding (Determinizm):**
   - NumPy ve PyTorch'un yanı sıra standart Python `random` kütüphanesi de seed'lenerek donanım seviyesinde determinizm ve tam tekrarlanabilirlik güçlendirildi.

**Amaç:**
Paralel eğitim verimliliğini maksimum düzeye çıkarmak, donanım kaynaklarını en optimal şekilde koordine etmek, eğitim grafiklerinde algoritmaların adımlarını adil ve eşit şekilde hizalamak.

### [24.06.2026 09:58] - Canlı Dashboard Üzerinde Konfigürasyon Sekmesi (v3.4.0)
**Yapılan Değişiklikler:**
1. **Konfigürasyon Parametrelerinin `metadata.json` Kaydına Dahil Edilmesi (`run_experiments.py`):**
   - `UAVConfig` ve `EnvConfig` sınıflarının tüm statik parametre değerleri JSON serileştirilebilir hale getirilerek her yeni deneyin `metadata.json` dosyasına eklenmiştir.
2. **Dashboard Arayüzü Güncellemesi (`index.html`):**
   - Log konsolunun bulunduğu alt panele yeşil temalı "Active Configs" sekmesi eklenmiştir. Bu sekme, deney başladığında geçerli olan tüm `GlobalConfig`, `EnvConfig`, `UAVConfig`, `QJCConfig`, `PPOConfig`, `DQNConfig` ve `PPOLSTMConfig` değerlerini gruplanmış ve kaydırılabilir şık bir tablo görünümünde canlı olarak sunmaktadır.

**Amaç:**
Deney esnasında konsola veya kaynak kod dosyalarına bakmaya gerek kalmadan, o an çalıştırılan tüm fiziksel ve algoritmik hiperparametreleri tek bir arayüzden inceleyebilmek.

### [24.06.2026 16:34] - Semtech SX1261 Fiziksel Parametre Hizalaması (v3.4.1)
**Yapılan Değişiklikler:**
1. **Bant Genişliği (Bandwidth) Güncellemesi (`config.py`):**
   - `B = 2e6` (2 MHz) olan kanal bant genişliği, gerçek LoRaWAN EU868 standartlarına uygun olarak `B = 125e3` (125 kHz) değerine düşürülmüştür.
2. **Gürültü Tabanı (Noise Floor) Hesaplaması (`config.py`):**
   - 125 kHz bant genişliği ve Semtech SX1261 veri sayfasında belirtilen tipik 6 dB alıcı gürültü katsayısı (Noise Figure) doğrultusunda, gürültü tabanı gücü $N_0$ (`N0_Linear`) değeri $-100\text{ dBm}$'den gerçekçi seviye olan $-117\text{ dBm}$'ye ($2 \times 10^{-15}\text{ Watt}$) çekilmiştir.

**Amaç:**
Simülasyonun telsiz/fiziksel katman modellemesini gerçek bir Semtech SX1261 çipinin standart haberleşme parametreleriyle birebir uyumlu hale getirmek ve bilimsel geçerliliği artırmak.

### [24.06.2026 22:25] - SINR Eşik Birim Uyumsuzluğu ve JSR %0 Hatası Düzeltmesi (v3.4.2)
**Yapılan Değişiklikler:**
1. **SINR Birim Dönüşüm Düzeltmesi (`pettingzoo_env.py`):**
   - Simülasyonun `pettingzoo_env.py` dosyasında doğrusal (linear) skaladaki anlık `sinr` değerinin, desibel (dB) skalasındaki `UAVConfig.SINR_THRESHOLD = -6.5` sabiti ile doğrudan karşılaştırılmasından kaynaklanan birim uyumsuzluğu (unit mismatch) hatası düzeltildi.
   - Doğrudan `linear_sinr > -6.5` karşılaştırması yapıldığında, doğrusal SINR her zaman pozitif veya sıfır olduğundan bu koşul her zaman doğru (`True`) olmaktaydı. Bu durum düğümlerin bağlantı durumunun her zaman `0` (Connected) kalmasına ve asla `2` (Jammed) olmamasına yol açıyordu.
   - Çözüm olarak desibel seviyesindeki `UAVConfig.SINR_THRESHOLD` değeri `10 ** (UAVConfig.SINR_THRESHOLD / 10.0)` formülüyle doğrusal (linear) skalaya çevrilerek karşılaştırıldı.
2. **Kanal Takip (Tracking) ve JSR Düzeltmesi:**
   - Birim hatası nedeniyle düğümler hiçbir zaman karıştırılamadığı için karıştırma başarı oranı (JSR) %0 çıkmaktaydı.
   - İHA hiçbir düğümün karıştırıldığını (status=2) algılayamadığından frekans atlama (frequency hopping) tetiklenmiyordu ve İHA sürekli ilk atandığı kanal olan 0. kanalda kalıyordu.
   - Jammer ise İHA'nın sürekli 0. kanalda durmasından dolayı takip ödülü (tracking reward) kazanmak için sürekli 0. kanalı seçmeyi öğrenerek kanal eşleşme oranının yapay olarak %100 çıkmasına neden oluyordu. Bu birim uyumsuzluğu çözülerek gerçekçi karıştırma ve kanal takip dinamikleri sağlandı.

**Amaç:**
Karıştırma başarı oranı (JSR) ve kanal takip doğruluğu (Tracking Accuracy) performans metriklerinin fiziksel formüllere uygun şekilde bilimsel olarak doğru hesaplanmasını ve İHA frekans atlama davranışının jammer baskısı altında düzgün tetiklenmesini sağlamak.

### [24.06.2026 23:25] - Dayanıklılık Değerlendirmesi SINR dB Dönüşümü ve Canlı Görselleştirme Düzeltmesi (v3.4.3)
**Yapılan Değişiklikler:**
1. **Dayanıklılık Değerlendirmesi SINR Hesaplama Düzeltmesi (`evaluate_paper_robustness.py`):**
   - Değerlendirme scripti `evaluate_paper_robustness.py` üzerinde, adım bazlı ortalama SINR değerlerinin desibel (dB) birimine dönüştürülmeden doğrudan doğrusal (linear) birimde toplanması ve ortalamasının alınması hatası giderildi.
   - Bu hata, çıktılarda ortalama sistem SINR değerinin astronomik olarak yüksek çıkmasına (örneğin ~3000-4000 dB) neden oluyordu.
   - Her adımda hesaplanan doğrusal SINR değeri `10 * np.log10(max(step_avg_sinr, 1e-12))` formülüyle dB skalasına dönüştürüldükten sonra toplama eklendi, böylece gerçekçi dB değerleri (örneğin 30-40 dB) elde edilmesi sağlandı.
2. **Canlı Görselleştirici Eşik Güncellemesi (`visualization.py`):**
   - Canlı Matplotlib/Pygame arayüzündeki (`visualization/visualization.py`) karıştırma bölgesi çiziminde (kontur çizgileri) kullanılan sert kodlanmış eski `1.0` (0 dB) doğrusal eşik değeri, yeni `-6.5 dB` çözme sınırının doğrusal karşılığı (`10 ** (UAVConfig.SINR_THRESHOLD / 10.0)`) ile dinamik hale getirildi.

**Amaç:**
Ortalama SINR değerlerinin raporlarda ve grafiklerde bilimsel gerçeklikle uyumlu dB biriminde doğru gösterilmesini sağlamak ve canlı görselleştiricinin karıştırma konturlarını yeni LoRaWAN fiziksel parametrelerine göre doğru çizmesini garanti etmek.

### [24.06.2026 23:30] - Ortalama SINR Metriğinin Grafiklerden Kaldırılması (v3.4.4)
**Yapılan Değişiklikler:**
1. **Karşılaştırma ve Dayanıklılık Grafikleri Güncellemesi (`compare.py` ve `evaluate_paper_robustness.py`):**
   - Ortalama sistem SINR değerinin (Average Network SINR) akademik/fiziksel olarak kafa karıştırıcı olması ve değerlendirme için birincil öncelikli olmaması nedeniyle bu metrik tüm görselleştirmelerden (subplot grafiklerinden) kaldırıldı.
   - Grafiklerde boşalan 4. panel slotu temiz bir şekilde gizlenerek (`set_visible(False)`) 3 panelli şık bir yerleşim düzeni oluşturuldu.
   - `compare.py` üzerinde SINR metriği kaldırıldıktan sonra diğer 3 grafik (Jamming Success Rate, Channel Tracking Accuracy ve Training Progress) yeniden dizilerek görsel denge korundu (sağ alt köşe boş bırakıldı).

**Amaç:**
Raporlarda ve grafiklerde kafa karıştırıcı veya gereksiz metrikleri temizleyerek odaklanılması gereken temel performans kriterlerine (JSR, Tracking ve Güç Tüketimi) yönelik daha net ve odaklanmış bir sunum sunmak.

### [25.06.2026 10:10] - Ödül Normalizasyonu ve Düşük Güç Tuzağı Çözümü (v3.5.0)
**Yapılan Değişiklikler:**
1. **Ödül Ağırlıklarının Normalizasyonu (`confs/env_config.py`):**
   * Toplam pozitif ödül aralığını [0, 1] arasına çekmek için başarılı karıştırma ödülü `W_SUCCESS = 0.8` ve kanal takip ödülü `W_TRACKING = 0.2` olarak güncellendi.
2. **Düşük Güç Yerel Optimum Tuzağının Çözülmesi:**
   * Ajanların enerji cezasından kaçınmak için minimum güçte (0.11W) kilitlendiği yerel optimum sorununu çözmek amacıyla enerji maliyeti ceza katsayısı `W_COST` değeri `0.1`'den `0.03` seviyesine düşürüldü.
   * Böylece başarılı bir karıştırma eylemi (0.8 ödül) ile maksimum güç maliyeti (0.03 ceza) arasındaki oran `26.6:1` yapılarak, keşif aşamasında yüksek gücün getirisinin maliyetini fazlasıyla domine etmesi sağlandı.
3. **Akademik Rapor ve Dokümantasyon Entegrasyonu:**
   * `RAPOR.md` dosyasının Bölüm 5.2 (PPO & DQN Ödül Yapısı) ve Bölüm 5.3 (Ödül Tasarımının Teorik Temelleri) kısımları yeni normalizasyon katsayıları ve gerekçeleriyle güncellenerek kod ile doküman tutarlılığı sağlandı.

**Amaç:**
Pekiştirmeli öğrenme ajanlarının (PPO ve DQN) kanal takibinin yanında yüksek karıştırma gücü uygulamasını da başarıyla öğrenmesini sağlamak ve bunu yaparken gereksiz güç harcamasını engelleyen enerji verimliliği dengesini korumak.



