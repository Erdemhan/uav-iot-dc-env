# Raporlama İş Akışı (Documentation Workflow) Yönergesi

Bu iş akışı, projedeki geliştirici ajanın (Antigravity) her kod değişikliği, hata çözümü veya görev tamamlama adımından sonra otomatik olarak **Gelişim Raporu** ve **Gelişim Günlüğü** hazırlamasını kurallara bağlar.

---

## 1. Tetiklenme Koşulu
* Ajan, kullanıcı tarafından verilen her görevin tamamlanma aşamasında (Verification / Doğrulama fazı öncesinde veya sonrasında) bu iş akışını otomatik olarak tetikler.

## 2. Ajanın İzleyeceği Adımlar (Çalışma Mantığı)

Ajan sırasıyla şu adımları işletir:

### Adım 1: Değişiklikleri Analiz Etme
Ajan, yerelde en son yapılan kod değişikliklerini tam olarak görmek için Git araçlarını kullanır:
* `git status` ile değişen/yeni eklenen dosyaları tespit eder.
* `git diff` (veya `git log -1 -p` eğer commit yapılmışsa) çalıştırarak satır bazında hangi kod bloklarının güncellendiğini inceler.

### Adım 2: Gerekçelendirme ve Sohbet Geçmişi Analizi
Ajan, yapılan kod değişikliklerinin **neden** yapıldığını anlamak için son konuşmaları analiz eder:
* Kullanıcının taleplerini, motivasyonunu ve yönlendirmelerini inceler.
* Sohbet geçmişinden teknik kararların alınma sebeplerini çıkarır.

### Adım 3: Raporu Yazma ve Güncelleme
Elde ettiği teknik verileri ve gerekçeleri birleştirerek akademik bir Türkçe diliyle rapor hazırlar. Raporu şu adrese ekler:
* **[RAPOR.md](file:///e:/Projeler/TEZ/uav-iot-dc-env/RAPOR.md)** dosyasının **Section 6 (Gelişim Günlüğü)** bölümünün en altına kronolojik sıraya uygun olarak ekler.

---

## 3. Rapor Şablonu (Template)

Rapor eklenirken aşağıdaki standart Markdown şablonu kullanılmalıdır:

```markdown
### [GG.AA.YYYY SS:DD] - [Descriptive Title] (vX.Y.Z)
**Yapılan Değişiklikler:**
1. **[Değişen Kısım/Özellik Başlığı]:**
   * Yapılan kod değişikliğinin detayları ve etkilenen dosyalar.
   * `[dosya_adi.py](file:///path/to/file)` formatında ilgili kod dosyalarına tıklanabilir bağlantılar.
2. **[Eklenen/Silinen Diğer Detaylar]:**
   * ...

**Teknik Gerekçelendirme ve Motivasyon:**
* Bu değişikliğin yapılma nedeni (Örn: tezdeki kıyaslama doğruluğunu artırmak, bellek tüketimini azaltmak, çoklu İHA çarpışmalarını engellemek vb.).
* Sohbet sırasında kullanıcı ile kararlaştırılan tasarım kararları.

**Test ve Doğrulama Sonuçları:**
* Yapılan değişikliklerin nasıl test edildiği (Örn: `python scripts/run_experiments.py` veya `evaluate.py`).
* Elde edilen ilk performans metrikleri veya doğrulama çıktıları.
```

---

## 4. Dil ve Stil Kuralları
* Rapor dili **tamamen Türkçe** olmalıdır (Tez gereksinimleri nedeniyle).
* Anlatım birinci tekil şahıs ("yaptım", "ekledim") yerine **edilgen veya üçüncü şahıs** diliyle ("yapılmıştır", "eklenmiştir", "kurgulanmıştır") yazılmalıdır.
* Kod dosyalarına ve sınıflara mutlaka tıklanabilir bağlantılar (`[Dosya](file:///...)`) verilmelidir.
