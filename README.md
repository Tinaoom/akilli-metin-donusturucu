# ✍️ Akıllı Metin Dönüştürücü (LLM Multilingual Text Styler)

Bu proje, **LLM (Large Language Model)** tabanlı, çok dilli bir içerik dönüştürücü asistandır.  
Kullanıcı tarafından girilen metni otomatik olarak diline göre algılar, belirlenen tona (ör. **resmî**, **samimi**, **akademik**) uygun biçimde yeniden yazar ve **okunabilirlik / anlam koruma metriklerini** hesaplar.

---

## 🚀 Özellikler

✅ **Otomatik dil algılama:** Türkçe 🇹🇷, İngilizce 🇬🇧, Almanca 🇩🇪, İspanyolca 🇪🇸  
✅ **Ton dönüştürme:** Resmî, samimi, akademik, özür, tanıtım, iş yazışması vb.  
✅ **Font seçimi ve biçim önerisi**  
✅ **Anlam korunumu (Cosine Similarity)**  
✅ **Okunabilirlik analizi (Flesch Reading Ease)**  
✅ **Firebase ile geçmiş kaydı (Firestore Realtime)**  
✅ **Modern, responsive arayüz (Tailwind CSS)**  

## ⚙️ Kurulum

### 1️⃣ Sanal ortam oluşturun
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux

#2️⃣ Gereksinimleri yükleyin
pip install -r requirements.txt

#3️⃣ Ortam değişkenini tanımlayın
GEMINI_API_KEY=your_google_gemini_api_key

#4️⃣ Uygulamayı çalıştırın
python app.py



| Katman          | Teknoloji                                                |
| --------------- | -------------------------------------------------------- |
| **Backend**     | Flask (Python)                                           |
| **LLM API**     | Google Gemini 2.5                                        |
| **Frontend**    | HTML5, TailwindCSS, Firebase Firestore                   |
| **NLP Analizi** | `textstat`, `nltk`, `scikit-learn`, `langdetect`         |
| **Veritabanı**  | Firebase Firestore                                       |
| **Metrikler**   | Okunabilirlik (Flesch), Anlam benzerliği (TF-IDF Cosine) |





