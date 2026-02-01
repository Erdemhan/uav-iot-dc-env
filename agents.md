# Agent Protokolleri ve Çalışma Kuralları

Bu dosya, proje üzerinde çalışan yapay zeka ajanlarının (veya geliştiricilerin) uyması gereken temel dokümantasyon ve bakım kurallarını tanımlar.

## 1. Raporlama Protokolü (`RAPOR.md`)
Her teknik güncelleme, kod değişikliği veya parametre revizyonundan sonra `RAPOR.md` dosyası kontrol edilmelidir.

*   **Teknik Güncelleme:** Eğer yapılan değişiklik sistemin mimarisini, matematiksel modellerini veya senaryo akışını değiştiriyorsa, raporun ilgili teknik bölümleri ("2. Sistem Mimarisi", "3. Matematiksel Modeller" vb.) güncellenmelidir.
*   **Gelişim Günlüğü (Change Log):** Yapılan her anlamlı değişiklik (Bug fix, özellik ekleme, parametre ayarı), "5. Gelişim Günlüğü" bölümüne **yeni bir tarih/saat bloğu** açılarak eklenmelidir.
    *   Sıralama: **Eskiden Yeniye (Artan)** şekilde olmalıdır (En eski sürüm en üstte, en yeni sürüm en altta). *Not: Mevcut raporda bu yapı v1.0.0 -> v1.1.1 şeklindedir, buna uyulmalıdır.*

## 2. İş Akışı Dokümantasyonu (`README.md`)
Projenin çalışma mantığını ve adımlarını içeren dosya artık `README.md` olarak adlandırılmıştır (Eski adıyla Workflow).

*   **Süreklilik:** Sistemin çalışma şeklini değiştiren bir güncelleme yapıldığında (Örn: Yeni bir script eklendiğinde, görselleştirme adımı değiştiğinde), `README.md` dosyası **mevcut mantığı ve yapısı bozulmadan** güncellenmelidir.
*   **İçerik:** Başlatma, Simülasyon Döngüsü, Sonlandırma ve Analiz adımları her zaman güncel tutulmalıdır.

## 3. Görev Takibi (`todo.md`)
*   Planlanan özellikler ve düzeltilecek hatalar `todo.md` dosyasında takip edilmelidir.
*   Tamamlanan görevler işaretlenmeli, yeni ihtiyaçlar listeye eklenmelidir.
