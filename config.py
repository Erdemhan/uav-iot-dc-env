from tez_reporter import TezReporter

TezReporter("config.py", "Config Sınıfı Tanımlandı")

class UAVConfig:
    """
    Tez önerisi Tablo 3 ve varsayılan simülasyon parametreleri.
    """
    # Fiziksel Sabitler
    C = 3e8       # Işık hızı (m/s)
    G = 9.8       # Yerçekimi (m/s^2)

    # İHA Mekanik Parametreleri (Döner Kanatlı)
    H = 100.0     # Operasyon irtifası (m)
    U_TIP = 120.0 # Rotor ucu hızı (m/s)
    V0 = 4.03     # Ortalama rotor indüklenmiş hızı (m/s)
    P0 = 79.86    # Blade profil gücü (W)
    P_IND = 88.63 # İndüklenmiş güç (W)
    RHO = 1.225   # Hava yoğunluğu (kg/m^3)
    D0 = 0.6      # Gövde sürükleme oranı
    S = 0.05      # Rotor solidity
    A = 0.5       # Rotor disk alanı (m^2)

    # Haberleşme Parametreleri
    B = 2e6       # Bant genişliği (Hz) -> 2 MHz
    FC = 2.4e9    # Taşıyıcı frekansı (Hz) -> 2.4 GHz
    ETA = 2.0     # Yol kaybı üssü (Free space varsayımı)
    
    # Gürültü ve Güç
    N0_Linear = 10**(-100/10) * 1e-3 # Watt cinsinden gürültü (örnek -100 dBm) 
    # Not: Kullanıcı N0 + I_jam dediği için N0'ı lineer Watt olarak tutuyoruz.
    
    P_TX_NODE = 0.1 # Node iletim gücü (Watt)
    P_TX_UAV = 0.5  # UAV iletim gücü (Watt)

    # IoT ve Enerji Parametreleri
    NUM_NODES = 5
    AREA_SIZE = 1000 # 1000x1000m alan
    
    # Denklem 288 için Varsayılanlar
    L_P = 1024      # Paket uzunluğu (bit)
    E_ACQ = 1e-5    # Veri toplama enerjisi (J)
    P_ENC = 0.01    # Şifreleme güç harcaması (W)
    T_ENC = 1e-4    # Şifreleme süresi (s)
