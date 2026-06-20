## İHA Destekli ve İletişim Anti-Jamming Simülasyon Senaryoları 

## **ve Simülasyon Senaryoları Kaynak Çalışmalar** 

## **Senaryo 1: Ortak Rota ve Kümeleme Optimizasyonu (Küçük ve Büyük Ölçekli)** _Kaynak Çalışma: Robust Anti-Jamming Technique for UAV Data Collection in IoT Using Landing Platforms and RIS_ 

Bu simülasyon, 1500m x 1500m boyutlarındaki bir alanda gerçekleştirilmiştir. Senaryo ikiye ayrılır: Alanda 25 olası konum ve 20’ye kadar IoT cihazı bulunan küçük ölçekli senaryo ile 100 olası konum ve 80’e kadar IoT cihazı yer alan büyük ölçekli senaryo test edilmiştir. İHA (UAV) uçuşa (0,0,0) koordinatından başlayıp görevini (1500,1500,0) noktasında tamamlamaktadır. İniş platformu sayısı (LP) 10 olarak belirlenmiş olup; bozucu (jammer), IoT cihazları ve Yansıtıcı Akıllı Yüzey (RIS) konumları her senaryoda rastgele atanmıştır. Simülasyon, önerilen Ant Kolonisi tabanlı AC-JRC algoritmasının K-means (AC-kmeans) ve standart Ant Koloni Sistemi (ACS) ile karşılaştırılması üzerine kuruludur. 

## **Senaryo 2: Ağaç Tohumu Algoritması (TSA) ile Anti-Jamming İHA Rotası** _Kaynak Çalışma: Anti-jamming Trajectory Design Method of UAV Based on Intelligent Optimization Algorithm_ 

Kare şeklindeki bir WSN (Kablosuz Sensör Ağı) alanında kurulan bu senaryoda, 4 adet Yer Sensörü (GS) ve 3 adet Yer Bozucusu (GJ - Jammer) bulunmaktadır. İHA’nın maksimum hızı 50 m/s, popülasyon boyutu 50 ve boyut (dim) 100 olarak ayarlanmıştır. Simülasyonda, UAV’nin bozuculardan uzaklaşıp sensörlere yaklaşarak veri toplaması hedeflenmiştir. Önerilen Ağaç Tohumu Optimizasyonu Algoritması (TSA), geleneksel Rastgele İteratif Algoritma ile 10 ve 2000 iterasyon sayıları üzerinden karşılaştırılarak test edilmiştir. 

## **Senaryo 3: DDPM-TD3 Tabanlı Çift İHA Destekli Güvenli Ağ** 

_Kaynak Çalışma: Secure Data Collection in UAV-Assisted IoT via Diffusion Model-Enabled Deep Reinforcement Learning_ 

Bu senaryoda hedef bölge 300m x 300m boyutlarındadır. Sistemde, 7 yerel IoT cihazından veri toplayan 1 "Birincil İHA" ve 2 adet yerel dinleyiciye (eavesdropper) karşı bozucu sinyaller yollayan 1 "Bozucu İHA" (Jamming UAV) bulunmaktadır. Birincil İHA (0, 299, 150), bozucu İHA ise (299, 299, 200) koordinatlarından göreve başlar ve her ikisi de sabit bir irtifada uçar. Önerilen Difüzyon Modeli destekli DDPM-TD3 algoritması, Bilgi Yaşı (AoI) ve enerji tüketimini optimize etmek amacıyla TD3, DDPG, A2C, PPO ve DQN algoritmalarıyla karşılaştırılmıştır. 

## **Senaryo 4: Sahte Düğüm (Eavesdropper) Karşısında Gizlilik ve SEE Optimizasyonu** 

_Kaynak Çalışma: Secrecy Energy Efficiency Maximization in UAV-Enabled Wireless Sensor Networks Without Eavesdropper’s CSI_ 

Bu senaryo, 4 adet Sensör Düğümü (SN) ve 1 adet Dinleyici (Eve) içerir. Sensör düğümlerinin koordinatları ( _±_ 50m, _±_ 50m) noktalarına, Eve’in koordinatı ise (20m, 20m) noktasına yerleştirilmiştir. İHA, başlangıç noktası olarak 1. Sensör Düğümünden (SN 1) hareket eder ve alanı saat 

1 

yönünün tersine dolaşır. Bu simülasyonda amaç, alt problem yaklaşımları (SCA, Dinkelbach vs.) kullanılarak Güvenli Enerji Verimliliğini (SEE) maksimize etmektir. 

## **Senaryo 5: VPPSA Algoritması ile AoI ve Enerji Farkındalıklı Çoklu Optimizasyon** 

_Kaynak Çalışma: AoI and Energy-Aware Data Collection for IRS-Assisted UAV-IoT Networks Under Jamming_ 

Bu test, 1000m x 1000m boyutlarındaki bir kare alan üzerinde gerçekleştirilmiş olup, IoT cihazları Poisson dağılımına göre yerleştirilmiştir. İHA sayısı 1’dir ve başlangıç-bitiş alanı sınırları 1000m olarak tanımlıdır. Alan içinde 1 Baz İstasyonu (1000, 500, 100), 1 Akıllı Yansıtıcı (IRS) (500, 0, 100) ve 1 Jammer (500, 500, 100) bulunmaktadır. Geliştirilen VPPSA algoritması; PSO, GWO, SDPSO, AWOA gibi kıyaslama algoritmalarına karşı, 30 ila 50 arasında değişen yoğunluktaki IoT cihazlarıyla sınanmış ve veri tazeliği (AoI) ile enerji tüketimi (EC) performansı test edilmiştir. 

## **Senaryo 6: DM-TD3 Tabanlı Kesintisiz Enerji Aktarımı ve Veri Toplama** 

## _Kaynak Çalışma: UAV-Enabled Secure Data Collection and Energy Transfer in IoT via DiffusionModel-Enhanced Deep Reinforcement Learning_ 

Hedef alan 400m x 400m kare olarak tasarlanmıştır. Bu alanda 10 adet IoT cihazı ve 1 adet jammer rastgele dağıtılmıştır. İHA (0, 0, 60) koordinatından başlayarak IoT cihazlarına sabit bir irtifada hem veri toplama hem de enerji transferi hizmeti verir. İHA’nın hızı ve uçuş dinamikleri ile IoT cihazlarının uyanık kalma/uyku durumları, yeni nesil DM-TD3 (Difüzyon destekli DRL) algoritması kullanılarak SAC, DDPG, sabit ve rastgele senaryolara karşı test edilmiştir. 

## **Senaryo 7: AGIFL (Hava-Yer Entegreli Federatif Öğrenme) Şehiriçi Algılama** 

## _Kaynak Çalışma: Covert Communications in Air-Ground Integrated Urban Sensing Networks Enhanced by Federated Learning_ 

Simülasyonda 100 adet Federatif Öğrenme (FL) destekli yer sensörü, 1000m x 1000m’lik alana rastgele yerleştirilmiştir. Senaryoda, ağın güvenliğini artırmak ve dinleyicileri engellemek amacıyla bir "Dost Jammer" (Friendly Jammer) ile sensörler ortak çalışır. Geliştirilen ortak yerel doğruluk ve iletim gücü stratejisi (OPUM&LA), sadece kullanıcıları veya sadece İHA’ları optimize eden temel yaklaşımlarla (OPU&LA, OPM&LA, OPUM) karşılaştırılmıştır. 

## **Senaryo 8: Endüstriyel IoT için Çoklu ABS-UAV ve JAM-UAV** 

## _Kaynak Çalışma: Efficient and Secure UAV-Assisted Industrial Internet of Things Based on Confidence-Weighted Reinforcement Learning_ 

Bu senaryoda 1 km x 1 km (1000m x 1000m) boyutlarındaki bir hedef alanda, rastgele dağılmış IoT cihazları ve kötü niyetli kullanıcılar (malicious users) yer alır. Sistemde veri toplamak için görev yapan "Havadan Baz İstasyonu (ABS-UAV)" adlı İHA’lar ile güvenliği sağlamak için bozucu sinyal yayını yapan "Bozucu İHA (JAM-UAV)"lar birlikte görev yapar. İHA’lar hedef alanın 50 m ila 140 m yüksekliğinde uçar. Tipik bir test kurulumunda 20 IoT cihazı, 3 ABSUAV ve 2 JAM-UAV bulunur. İHA çarpışmaları engellenerek AoI minimize edilir; geliştirilen CTD3 algoritması, TD3, PPO, NDQN gibi algoritmalarla karşılaştırılmıştır. 

## **Senaryo 9: Çift Küme Başlı (Double CH) NOMA Tabanlı İHA Veri Toplama** _Kaynak Çalışma: Secure Resource Allocation and Trajectory Design for UAV-Assisted IoT With Double Cluster Head_ 

Hedef alanda düğümler (nodes) kümeler halinde gruplanmıştır ve her kümede havada asılı duran bir İHA (UAV) bulunur. Testte normal düğümler (RN) ve her küme içinde veri güvenliğinin önemli olduğu Güvenli Düğüm (SMN) bulunmaktadır. Hedef, dinleyiciye (Eve) rağmen güvenli enerji verimliliğini (SEE) maksimize etmektir. Tasarlanan INDIC algoritması, havada asılı ka- 

2 

larak NOMA ile iletişim kuran (Fly-and-hover IE-NOMA) algoritmalarla test edilmiştir. 

## **Senaryo 10: Dost Jammer Seçimi ile BCD ve SCA Tabanlı Gizli Veri Toplama** _Kaynak Çalışma: Cooperative Jamming and Trajectory Design for UAV-Assisted Data Collection with Physical-Layer Security_ 

İHA (UAV), rastgele konumlandırılmış Sensör Düğümünden (SN) veri toplarken, Yardımcı Düğüm (AD), bir Eavesdropper’a (Eve) karşı dost jammer (bozucu) olarak kullanılmak üzere sistem tarafından seçilir. BCD (Blok Koordinat İnişi) ve SCA metodolojilerini birleştiren bu senaryo algoritması; yalnızca yörünge optimizasyonu yapan (ASDC) veya önceden tanımlanmış aktarım düzeni (PDT) kullanan sistemlerle karşılaştırılmıştır. 

## **Senaryo 11: RIS Destekli Çift İHA (Toplayıcı ve Dinleyici)** 

## _Kaynak Çalışma: Cooperative Jamming for RIS-Assisted UAV-WSN Against Aerial Malicious Eavesdropping_ 

Operasyon alanında geçen bu senaryoda sensörler (SN) ve bir dost jammer bulunur. UAV-L (veri toplayıcı) ve UAV-E (kötü niyetli İHA) zıt veya doğrusal bir rotada uçar. Geliştirilen CJJOA algoritması (Çerçeve Uzunluğu, Yörünge, Güç ve RIS fazı optimizasyonlarını içerir), yörünge ya da yansıtıcı yüzey eksikliklerini içeren kırpılmış algoritmalarla kıyaslanmıştır. 

**Senaryo 12: CFC Tabanlı Statik/Dinamik Jammer ve Engelden Kaçınma** _Kaynak Çalışma: Motion Planning in UAV-Aided Data Collection with Dynamic Jamming_ Sensörden (SN) veri toplamak amacıyla oluşturulan bu senaryoda statik ve dinamik jammer’lar ile fiziksel engeller (obstacles) kullanılmıştır. 100 kbits, 200 kbits, 500 kbits ve 1 Mbits verinin sensörlerden toplanması durumu simüle edilmiştir. Geliştirilen "İletişim Uçuş Koridoru" (CFC) temelli yol planlama (SCA ile lokal optimize edilen) yaklaşımı; DWA (Dinamik Pencere Yaklaşımı) ve AStar (A*) başlangıç algoritmalarıyla test edilmiştir. Statik jamming ortamında sensörün alanı "armut şeklinde" büzülmekte, İHA sadece güvenli "koridor" üzerinden veri toplayabilmektedir. 

3 

Tez İzleme Komitesi (TİK) raporunuza veya makalenize doğrudan ekleyebileceğiniz; literatürdeki diğer çalışmaları, oluşturduğumuz detaylı özeti (tabloyu), temel aldığınız (baseline) çalışmayı ve "çakışma ile maskeleme (masking by interference)" gibi kritik gerekçelendirmeleri barındıran kapsamlı LaTeX kodunu aşağıda sunuyorum. 

Bu kurgu, hem literatüre ne kadar hakim olduğunuzu gösterir hem de senaryonuzu neden baseline çalışmadan farklılaştırdığınızı (yapısal körlük ve maskeleme üzerinden) çok güçlü bir akademik dille savunur. 

## **1 Simülasyon Senaryosu Seçimi ve Kıyaslanabilirlik Gerekçelendirmesi** 

Bu tez çalışmasında geliştirilen _Durum-Farkında Zeki Jammer (State-Aware Intelligent Jammer)_ modelinin ve PPO/DQN tabanlı saldırı stratejilerinin performansını değerlendirmek için, literatür standartlarına dayalı ancak zeki ajanın öğrenme kapasitesini izole edecek özgün bir simülasyon senaryosu kurgulanmıştır. Senaryo parametreleri ve bu seçimlerin literatüre dayalı akademik gerekçeleri aşağıda detaylandırılmıştır. 

## **1.1 Literatürdeki Mevcut Senaryolar ve Makro Çevresel Standartlar** 

Son yıllarda İHA destekli IoT ağlarında veri toplama ve anti-jamming üzerine yapılan çalışmalar incelendiğinde, belirli operasyonel sınırların f li standart (de facto) haline geldiği görülmektedir. Tablo 2’de özetlenen güncel literatür analiz edildiğinde; özellikle _"Efficient and Secure UAV-Assisted Industrial Internet of Things Based on Confidence-Weighted Reinforcement Learning"_ ve _"AoI and Energy-Aware Data Collection for IRS-Assisted UAV-IoT Networks Under Jamming"_ isimli çalışmalarda test alanının 1000 _m ×_ 1000 _m_ olarak belirlendiği, veri toplayıcı İHA’ların 50m ile 150m arası sabit irtifalarda görev yaptığı ve ortamda 20 ila 50 arasında IoT düğümü bulunduğu görülmektedir. 

Adil bir kıyaslama (fair comparison) zemini oluşturmak adına, bu çalışmada da makro çevresel kısıtlar literatürle hizalanmış ve 1000 _m ×_ 1000 _m_ boyutlarında bir operasyon alanı seçilmiştir. Ayrıca, İHA’nın uçuş irtifası _H_ = 100 _m_ olarak sabitlenmiş ve mesafe odaklı sinyal zayıflaması ( _d_[2] ) gibi fiziksel kısıtlar, literatürdeki gerçekçi kanal modellerine sadık kalınarak sisteme entegre edilmiştir. 

## **1.2 Temel (Baseline) Çalışma ve Senaryo Uyuşmazlıkları** 

Çalışmamızda, literatürdeki en güncel makine öğrenmesi tabanlı zeki bozucu modellerinden biri olan _"UAV Deployment Optimization and Carrier Selection in Jamming Environments: A Game Learning Approach"_ (Liao vd., 2025) isimli çalışma temel (baseline) olarak alınmıştır. Bu referans çalışma, 1000 _m ×_ 1000 _m_ ’lik bir alanda 11 İHA’lık bir sürü (swarm) ile Q-öğrenme (QJC) tabanlı bozucular arasındaki frekans atlama mücadelesini modellemektedir. 

Çevresel ölçek bağlamında bu baseline çalışma ile tam bir uyum sağlanmış olsa da, referans çalışmanın ağ kurgusu (swarm mimarisi) tez kapsamında kurgulanan senaryoya birebir kopyalanmamıştır. Bu yapısal farklılaştırmanın temel nedeni, tezin araştırma odağı ile baseline çalışmanın ağ tıkanıklığı dinamikleri arasındaki metodolojik uyuşmazlıktır. 

4 

Tablo 1: Simülasyon Senaryoları Özeti ve Kaynak Çalışmalar 

|**Senaryo / Kaynak Çalışma**|**Simülasyon**<br>**Alanı / Uçuş**|**Varlıklar**<br>**(İHA,**<br>**IoT, Jammer, vb.)**|**Temel Odak / Algo-**<br>**ritmalar**|
|---|---|---|---|
|**Senaryo 1**<br>Robust Anti-Jamming Technique for UAV<br>Data Collection in IoT Using Landing<br>Platforms and RIS|1500m<br>x<br>1500m,<br>Başlangıç:<br>(0,0,0)<br>Bitiş: (1500,1500,0)|1 İHA, 20-80 IoT Ci-<br>hazı, 1 Jammer, 1 RIS,<br>10 İniş Platformu|AC-JRC algoritması vs<br>AC-kmeans,<br>standart<br>ACS. İHA rotası, veri<br>toplama ve kümeleme.|
|**Senaryo 2**<br>Anti-jamming Trajectory Design Method<br>of UAV Based on Intelligent Optimization<br>Algorithm|Kare Bölge, Sabit<br>İrtifa,<br>Max<br>İHA<br>Hızı: 50 m/s|1 İHA, 4 Yer Sensörü<br>(GS), 3 Yer Bozucusu<br>(GJ)|TSA (Ağaç Tohumu) vs.<br>Rastgele<br>İteratif<br>Algo-<br>ritma. Veri toplama ora-<br>nını maksimize etme.|
|**Senaryo 3**<br>Secure Data Collection in UAV-Assisted<br>IoT via Difusion Model-Enabled DRL|300m x 300m, Sa-<br>bit İrtifa (150m ve<br>200m)|1 Birincil İHA, 1 Bo-<br>zucu İHA, 7 IoT Dü-<br>ğümü, 2 Dinleyici|DDPM-TD3<br>vs.<br>TD3,<br>DDPG, A2C vb. Bilgi<br>Yaşı (AoI) ve enerji op-<br>timizasyonu.|
|**Senaryo 4**<br>Secrecy Energy Efciency Maximization in<br>UAV-Enabled WSNs Without Eavesdrop-<br>per’s CSI|SN:<br>(_±_50m,<br>_±_50m),<br>Dinle-<br>yici: (20m, 20m)|1 İHA, 4 Sensör Dü-<br>ğümü (SN), 1 Dinle-<br>yici (Eve)|SCA, Dinkelbach teknik-<br>leri. Güvenli Enerji Ve-<br>rimliliği (SEE) ve sahte<br>düğüm atlatma.|
|**Senaryo 5**<br>AoI and Energy-Aware Data Collection<br>for IRS-Assisted UAV-IoT Networks Un-<br>der Jamming|1000m x 1000m|1 İHA, 30-50 IoT Ci-<br>hazı, 1 BS, 1 Jammer,<br>1 RIS|VPPSA algoritması vs.<br>PSO,<br>GWO,<br>SDPSO.<br>Bilgi<br>Yaşı<br>(AoI),<br>İHA<br>enerjisi ve anti-jamming.|
|**Senaryo 6**<br>UAV-Enabled Secure Data Collection and<br>Energy Transfer in IoT via Difusion-<br>Model-Enhanced DRL|400m x 400m, Baş-<br>langıç (0,0,60), Sa-<br>bit İrtifa|1 İHA, 10 IoT Cihazı,<br>1 Jammer|DM-TD3<br>(Difüzyon<br>DRL) vs. DDPG, SAC,<br>TD3.<br>Kesintisiz<br>enerji<br>aktarımı, AoI ve güven-<br>lik.|
|**Senaryo 7**<br>Covert Communications in Air-Ground In-<br>tegrated Urban Sensing Networks Enhan-<br>ced by FL|1000m<br>x<br>1000m<br>kare test alanı|HAP<br>(Yüksek<br>İrtifa<br>İHA),<br>100<br>Sensör,<br>Dost Jammer|OPUM&LA<br>vs<br>OPM&LA<br>vb.<br>Hava-<br>Yer<br>entegreli<br>Federatif<br>Öğrenme (AGIFL) algı-<br>lama.|
|**Senaryo 8**<br>Efcient and Secure UAV-Assisted IIoT<br>Based on Confdence-Weighted Reinforce-<br>ment Learning|1000m<br>x<br>1000m,<br>Uçuş<br>Yüksekliği:<br>50m - 140m|3<br>Toplayıcı<br>İHA<br>(ABS), 2 Bozucu İHA<br>(JAM), 20 IoT|CTD3 vs. TD3, PPO,<br>NDQN. Çoklu İHA ağ-<br>larında çarpışma önleme,<br>AoI ve rol paylaşımı.|
|**Senaryo 9**<br>Secure Resource Allocation and Trajectory<br>Design for UAV-Assisted IoT With Double<br>Cluster Head|Alan içerisinde kü-<br>meler|3 İHA (Küme Baş-<br>ları),<br>RN<br>(Normal),<br>SMN<br>(Güvenli)<br>Dü-<br>ğümler|INDIC<br>algoritması<br>vs.<br>Fly-and-Hover<br>IE-<br>NOMA.|
|**Senaryo 10**<br>Cooperative Jamming and Trajectory De-<br>sign for UAV-Assisted Data Collection|Max İHA hızı 50<br>m/s|1 İHA, Sensörler (SN),<br>Yardımcı<br>Düğümler<br>(AD), Eve|BCD ve SCA algoritması<br>vs. ASDC, PDT. Dina-<br>mik jammer seçimi ve<br>gizlilik kapasitesi.|
|**Senaryo 11**<br>Cooperative<br>Jamming<br>for<br>RIS-Assisted<br>UAV-WSN<br>Against<br>Aerial<br>Malicious<br>Eavesdropping|Doğrusal<br>rotada<br>uçuş|1 Toplayıcı İHA (L), 1<br>Kötü Niyetli İHA (E),<br>SN, Jammer, RIS|CJJOA algoritması. Yö-<br>rünge, RIS faz şifti ve<br>çerçeve uzunluğu optimi-<br>zasyonları.|
|**Senaryo 12**<br>Motion Planning in UAV-Aided Data Col-<br>lection with Dynamic Jamming|Fiziksel engelli alan|1 İHA, Sensör (SN),<br>Dinamik Jammer’lar,<br>Engeller|CFC<br>(İletişim<br>Uçuş<br>Koridoru) tabanlı SCA<br>vs AStar, DWA. Engel-<br>den/bozucudan kaçınma.|



5 

## **1.3 Araştırma Odağı Uyuşmazlığı ve "Sürü Yoğunluğu" ile "Olasılıksal Maskeleme" Paradoksu** 

Baseline çalışmada (Liao vd., 2025) kullanılan Q-Learning tabanlı jammer (QJC), çevresel konum ve kanal durumlarını algılamayan durumsuz (stateless / yapısal kör) bir yapıda olmasına rağmen kalabalık sürü (swarm) senaryolarında görünürde yüksek bir başarı sergilemektedir. Ancak bu durum, aslında İHA sürüsünün kendi iç spektral tıkanıklığı ve kalabalık sürü boyutunun yarattığı **olasılıksal hedef bulma kolaylığı (probabilistic target hit masking)** yanılsamasıdır.

Referans çalışmada $N = 13$ kullanılabilir frekans kanalı, $J = 2$ jammer ve $M = 11$ İHA yer almaktadır. Jammer'lar kendi aralarında çakışmadığından spektrumun 2 kanalını bloke ederler. Bu kurguda, spektrumda 11 İHA için geriye tam olarak 11 temiz kanal kalmaktadır. Bu durum, **kanal doluluk oranını (occupancy density) tam olarak %100'e ($11/11$)** getirmektedir. İHA'ların kaçabileceği hiçbir boş yedek kanal (redundant backup channel) bulunmamaktadır. Herhangi bir İHA karıştırmadan kaçmaya çalıştığı anda, kaçtığı kanalda diğer bir İHA ile çakışarak co-frequency interference (eş-frekans girişimi) yaratacaktır. 

Daha da önemlisi, bozucular (jammers) rastgele veya körlemesine herhangi 2 kanalı seçtiğinde bile, kanalların İHA'lar tarafından doluluk oranı %84.6 ($11/13$) olduğundan, **hiçbir örüntü öğrenme zekası göstermeden her adımda en az bir İHA'yı vurma olasılıkları (overall random hit probability) %98.7'dir** (tek bir kanal bazında ise %84.6). Bu durum, jammer'ın "akıllı ve hedefe yönelik" bir saldırı yaptığı yanılsamasını doğurmaktadır; oysa bu başarı, zeki bir takipten değil, tamamen İHA sürüsünün alandaki yüksek spektral doluluğundan kaynaklanmaktadır.

Bu tezin temel hedefi, kaba kuvvet gürültüsüyle kalabalık bir sürüyü istatistiksel yoğunluktan faydalanarak olasılıksal olarak vurmak değil; tekil bir kurbanın reaktif frekans atlama (Markovian hopping) örüntüsünü çözebilen gerçek bir **saldırı zekası (state-aware tracking)** oluşturmaktır. Önerilen senaryoda ise $N=7$ kullanılabilir frekans kanalı, $J=1$ jammer ve $M=1$ İHA yer almaktadır. Bu yeni kurgu, hem jammer spektral kaplama oranını baseline çalışmaya paralel olarak **%14.3'e ($1/7$)** düşürmekte hem de kurbanın zeki takibini en üst düzeyde test etmektedir. Bu doğrultuda asıl belirleyici olan **rastgele hedef vurma olasılığı** ve **kurbanın kaçış yedekliliğidir (escape redundancy)**:
1. **Rastgele Hedef Vurma Olasılığının Düşürülmesi:** Baseline kurguda jammer'ın rastgele bir atışla bir İHA'yı vurma olasılığı **%84.6** iken, önerilen 7 kanallı senaryoda bu oran **%14.3'e ($1/7$)** indirilmiştir. Yani rastgele bir jammer'ın hedefi ıskalama olasılığı %15.4'ten **%85.7**'ye çıkarılmıştır.
2. **UAV Kaçış Yedekliliği (Escape Redundancy) ve Çözüm Kümesi:** $N=7$, $M=1$ ve $J=1$ kurgusunda jammer 1 kanalı kapattığında geriye 6 temiz kanal kalır ve 1 İHA bu kanallardan birini seçer. Böylece İHA'nın **%500 kaçış yedekliliği (5 boş yedek kanal)** bulunur. Baseline çalışmada ise temiz kanal doluluğu %100 olduğundan yedeklilik **%0**'dır.

Bu izole kurguda İHA reaktif olarak (sadece karıştırıldığında) kaçtığı için, jammer'ın kurbanı sürekli bloke edebilmesi için **UAV'nin 7 kanallı Markov geçiş örüntüsünü aktif olarak tahmin etmesi** zorunludur. Aksi takdirde, durumsuz QJC modelinde olduğu gibi jammer statik bir kanala odaklandığında İHA'yı en fazla 1 kez karıştırır; İHA hemen boş yedek kanallardan birine kaçarak orada sabit kalır ve jammer eski kanalda boş yayın yapmaya devam eder. Nitekim, durumsuz QJC modelinin bu kurguda **%1.1 ila %1.9 takip başarısında** kalarak tamamen çökmesi, önerilen durum-farkında PPO modelimizin (%57.4 - %60.1) başarısının tamamen "tahmin zekasına" dayandığını bilimsel olarak kanıtlamaktadır.

## **1.4 Enerji Verimli Saldırı ve Önerilen Senaryo Kurgusu** 

Sonuç olarak, literatürdeki _"Motion Planning in UAV-Aided Data Collection with Dynamic Jamming"_ gibi fiziksel engel (obstacle) barındıran veya kalabalık cihaz grupları içeren çalışmaların aksine; bu çalışmada zeki saldırganın durum-farkında (state-aware) öğrenme yeteneğini ölçen spesifik bir kurgu tasarlanmıştır. 

Önerilen Senaryo Parametreleri: 

- **Alan ve Varlıklar:** 1000 _m ×_ 1000 _m_ engelsiz (obstacle-free) alan; 5 IoT düğümü, 1 İHA ve 1 Zeki Jammer. 

- **İletişim Dinamikleri:** İHA, SINR seviyesi eşiğin ( _<_ 0 dB) altına düştüğünde statik bir kurban olarak kalmamakta, Markov geçiş matrisine bağlı istatistiksel bir örüntü ile reaktif olarak kanal değiştirmektedir (2.4, 5.0 ve 5.8 GHz). 

- **Fiziksel Kısıtlar ve Enerji:** Saldırgan; normalize mesafe (RSS proxy), spektrum doluluğu ve frekansa bağlı Güç Amplifikatörü (PA) verimliliğini ( _ηPA_ (2 _._ 4) _≈_ 0 _._ 50, _ηPA_ (5 _._ 8) _≈_ 0 _._ 19) dikkate almak zorundadır. 

Bu sayede jammer’ın yalnızca sınırsız güçle yayın yapması engellenmiş, "doğru anlarda derin SINR düşüşleri ( _SINR ≪_ 0)" yaratarak PPO-LSTM modeliyle yaklaşık %24 enerji tasarrufu sağlayan etkili saldırılar öğrenmesi garanti altına alınmıştır. 

6 

## **1.5 Simülasyon Senaryoları Özeti Tablosu** 

Mevcut çalışmanın literatürdeki yeri ve senaryo parametrelerinin karşılaştırması Tablo 2’de verilmiştir. 

Tablo 2: Literatürdeki Simülasyon Senaryoları ve Önerilen Özgün Senaryo Özeti 

|**Senaryo / Kaynak Çalışma**|**Simülasyon**<br>**Alanı / Uçuş**|**Varlıklar**<br>**(İHA,**<br>**IoT, Jammer, vb.)**|**Temel Odak / Algo-**<br>**ritmalar**|
|---|---|---|---|
|_UAV Deployment Optimization and Car-_<br>_rier Selection in Jamming Environments_<br>_(Baseline)_|1000m x 1000m, Iz-<br>gara (Grid) tabanlı|11<br>İHA<br>(Sürü),<br>2<br>Akıllı<br>Jammer<br>(Q-<br>Learning)|Congestion Game ve Q-<br>Learning tabanlı taşıyıcı<br>frekans (carrier) seçimi.|
|_Efcient and Secure UAV-Assisted IIoT_<br>_Based on Confdence-Weighted Reinforce-_<br>_ment Learning_|1000m<br>x<br>1000m,<br>Uçuş<br>Yüksekliği:<br>50m - 140m|3 Toplayıcı İHA, 2 Bo-<br>zucu İHA, 20 IoT|CTD3 vs. TD3, PPO,<br>NDQN. Çoklu İHA ağ-<br>larında çarpışma önleme,<br>AoI minimizasyonu.|
|_AoI and Energy-Aware Data Collection_<br>_for IRS-Assisted UAV-IoT Networks Un-_<br>_der Jamming_|1000m x 1000m|1 İHA, 30-50 IoT Ci-<br>hazı, 1 Jammer, 1 RIS|VPPSA algoritması vs.<br>PSO, GWO. Bilgi Yaşı<br>(AoI), enerji optimizas-<br>yonu ve anti-jamming.|
|_Secure Data Collection in UAV-Assisted_<br>_IoT via Difusion Model-Enabled DRL_|300m x 300m, Sa-<br>bit İrtifa (150m ve<br>200m)|1 Birincil İHA, 1 Bo-<br>zucu İHA, 7 IoT Dü-<br>ğümü, 2 Dinleyici|DDPM-TD3<br>vs.<br>TD3,<br>DDPG, A2C vb. Bilgi<br>Yaşı (AoI) ve güvenlik.|
|_Motion Planning in UAV-Aided Data Col-_<br>_lection with Dynamic Jamming_|Fiziksel engelli alan|1 İHA, Sensör, Dina-<br>mik Jammer’lar, En-<br>geller (Obstacles)<br>|CFC (İletişim Uçuş Ko-<br>ridoru) tabanlı SCA vs<br>AStar, DWA. Engelden<br>ve bozucudan kaçınma.|
|**Bu Çalışma (Önerilen Se-**<br>**naryo)**|**1000m x 1000m**,<br>Engelsiz (Obstacle-<br>Free), Sabit İrtifa:<br>100m|**1**<br>**İHA,**<br>**5**<br>**IoT**<br>**Cihazı,**<br>**1**<br>**Durum-**<br>**Farkında**<br>**(State-**<br>**Aware) Zeki Jam-**<br>**mer**|**PPO,**<br>**DQN,**<br>**PPO-**<br>**LSTM vs. QJC (Base-**<br>**line).** Çakışmadan izole<br>edilmiş kanallarda zeki<br>tehdidin<br>Markov<br>örün-<br>tüsü<br>takibi<br>ve<br>enerji-<br>verimli saldırı.|



. 

7 

