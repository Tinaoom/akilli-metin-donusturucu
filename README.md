# ✍️ Akıllı Metin Dönüştürücü (LLM Multilingual Text Styler)

Bu proje, **LLM (Large Language Model)** tabanlı, çok dilli bir içerik dönüştürücü asistandır.  
Kullanıcı tarafından girilen metni **otomatik olarak diline göre algılar**, belirlenen **tona (ör. resmî, samimi, akademik)** uygun biçimde yeniden yazar  
ve **okunabilirlik / anlam koruma metriklerini** hesaplar.

---

## 🚀 Özellikler

✅ Otomatik dil desteği: **Türkçe 🇹🇷**, **İngilizce 🇬🇧**, **Almanca 🇩🇪**, **İspanyolca 🇪🇸**  
✅ Ton dönüştürme: Resmî, samimi, akademik, özür, tanıtım, iş yazışması vb.  
✅ Font seçimi ve biçim önerisi  
✅ Anlam korunumu (TF-IDF + Cosine Similarity)  
✅ Okunabilirlik analizi (Flesch Reading Ease)  
✅ Modern, responsive arayüz (Tailwind CSS)  
✅ Geçmiş kayıt desteği (Firebase Firestore ile entegre edilebilir)

> 📌 Not: Şu anki sürümde yalnızca **yerel analiz** yapılır. Firebase entegrasyonu isteğe bağlıdır.

## ⚙️ Kurulum

### 1️⃣ Sanal ortam oluşturun
```bash
cd akilli-metin-donusturucu-main
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux

#2️⃣ Gereksinimleri yükleyin
pip install -r requirements.txt

#3️⃣ Ortam değişkenini tanımlayın
Ana dizinde bir .env dosyası oluşturun ve Google AI Studio üzerinden aldığınız API anahtarını girin:
GEMINI_API_KEY=your_google_gemini_api_key 

#4️⃣ Uygulamayı çalıştırın
python app.py

| Katman          | Teknoloji                                                |
| --------------- | -------------------------------------------------------- |
| **Backend**     | Flask (Python)                                           |
| **LLM API**     | Google Gemini 2.5                                        |
| **Frontend**    | HTML5, TailwindCSS                                       |
| **NLP Analizi** | `textstat`, `nltk`, `scikit-learn`                       |
| **Veritabanı**  | (Opsiyonel) Firebase Firestore                           |
| **Metrikler**   | Okunabilirlik (Flesch), Anlam benzerliği (TF-IDF Cosine) |

🪪 Lisans
Bu proje MIT Lisansı ile lisanslanmıştır.
İstediğiniz gibi kullanabilir, geliştirebilir ve paylaşabilirsiniz.






