import os
import requests
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# NLP ve Metrik Hesaplama Kütüphaneleri
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import re

# -----------------------------------------------------
# KONFİGÜRASYON VE API AYARLARI
# -----------------------------------------------------

load_dotenv()
# .env dosyasındaki GEMINI_API_KEY değişkenini kullan
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # Bu hatayı terminalde göstererek uygulamanın başlamasını engeller.
    raise ValueError("❌ GEMINI_API_KEY bulunamadı. Lütfen .env dosyanıza ekleyin!")

# Gemini API Endpoint ve Model
API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
REWRITE_MODEL = "gemini-2.5-flash-preview-05-20"
CLASSIFY_MODEL = "gemini-2.5-flash-preview-05-20"

app = Flask(__name__)

# NLTK Türkçe Durak Kelimeleri (Stopwords) yükleme
try:
    # Kullanıcının daha önce indirdiği varsayılır (pip install ve python -c "...")
    TURKISH_STOPWORDS = set(stopwords.words('turkish'))
except LookupError:
    # NLTK verileri eksikse boş küme kullan (uygulamanın çökmesini engeller)
    TURKISH_STOPWORDS = set()
    print("UYARI: NLTK 'stopwords' bulunamadı. Metrikler etkilenebilir.")

# -----------------------------------------------------
# NLP METRİK FONKSİYONLARI
# -----------------------------------------------------

def clean_text(text):
    """Metni Kosinüs Benzerliği için hazırlar: küçük harf, noktalama kaldırma, stop words kaldırma."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    try:
        words = nltk.word_tokenize(text, language='turkish')
        words = [w for w in words if w not in TURKISH_STOPWORDS]
        return ' '.join(words)
    except Exception:
        # Tokenizer hatası durumunda temizlenmemiş metni döndür
        return text

def calculate_similarity(text1, text2):
    """İki metin arasındaki Kosinüs Benzerliğini hesaplar."""
    if not text1 or not text2:
        return 0.0

    cleaned_text1 = clean_text(text1)
    cleaned_text2 = clean_text(text2)

    if not cleaned_text1 or not cleaned_text2:
        return 0.0

    documents = [cleaned_text1, cleaned_text2]
    
    try:
        # TF-IDF Vektörleştirme
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Kosinüs Benzerliği hesaplama
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # text1 ve text2 arasındaki benzerlik
        return float(similarity_matrix[0, 1])
    except ValueError:
        # Vektörleştirme hatası (çok kısa, aynı kelimeler vb.)
        return 0.0

def calculate_readability(text):
    """Metnin okunabilirlik skorunu (ortalama kelime/cümle sayısı) hesaplar."""
    if not text:
        return 0
    
    try:
        sentence_count = len(nltk.sent_tokenize(text, language='turkish'))
        word_count = len(nltk.word_tokenize(text, language='turkish'))
    except Exception:
        # Tokenizer hatası
        return 0
    
    if sentence_count == 0 or word_count == 0:
        return 0
        
    # Basit okunabilirlik metriği: Ortalama kelime uzunluğu (daha karmaşık okunabilirlik formüllerinin yerine)
    avg_word_length = sum(len(word) for word in nltk.word_tokenize(text, language='turkish')) / word_count
    return round(avg_word_length, 2)


# -----------------------------------------------------
# GEMINI API İLETİŞİMİ
# -----------------------------------------------------

def make_gemini_request(model_name, prompt, system_instruction, response_schema=None):
    """Gemini API'ye genel çağrı yapan yardımcı fonksiyon."""
    
    api_url = f"{API_URL_BASE}{model_name}:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {
            "maxOutputTokens": 2048,
        }
    }

    if response_schema:
        payload["generationConfig"]["responseMimeType"] = "application/json"
        payload["generationConfig"]["responseSchema"] = response_schema
    
    try:
        response = requests.post(
            api_url, 
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"❌ API Bağlantı Hatası: {e}")
        return {"error": str(e)}

def classify_target_style(text):
    """Girilen metnin stilini Gemini'nin JSON yeteneği ile tespit eder."""
    
    candidate_labels = [
        "resmi", "is", "samimi", "akademik", "etik", 
        "ozur", "destek", "tanitim", "mail", "rapor", "nötr", "kaba"
    ]
    
    system_prompt = (
        "Sen, verilen metnin tonunu ve amacını analiz eden bir dilbilimcisinn. "
        "Yalnızca verilen etiklerden birini seç ve puanla. Çıktıyı zorunlu JSON şemasına göre oluştur."
    )
    
    prompt = f"""
    Aşağıdaki metnin en uygun ton/stil etiketini belirle.
    Metin: "{text}"
    Etiketler: {', '.join(candidate_labels)}
    """

    # JSON Çıktı Şeması
    classification_schema = {
        "type": "OBJECT",
        "properties": {
            "label": {"type": "STRING", "description": "Metne en uygun olan etiket."},
            "score": {"type": "NUMBER", "description": "Bu etiketin uygunluk puanı (0.0 ile 1.0 arası)."}
        },
        "propertyOrdering": ["label", "score"]
    }

    data = make_gemini_request(CLASSIFY_MODEL, prompt, system_prompt, classification_schema)

    if "error" in data:
        return {"label": "bağlantı_hatası", "score": 0.0}

    try:
        json_string = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        result = json.loads(json_string)
        return {"label": result.get("label", "belirlenemedi"), "score": result.get("score", 0.0)}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ JSON Ayrıştırma Hatası: {e}")
        return {"label": "ayrıştırma_hatası", "score": 0.0}

def rewrite_message(text, target_style, font):
    """Girilen metni hedef stile göre Gemini ile yeniden yazar."""
    
    system_prompt = (
        "Sen, metinleri verilen ton ve üsluba mükemmel şekilde dönüştüren profesyonel bir Türk metin yazarı botusun. "
        "Çıktı kurallarına kesinlikle uy. Font açıklaması sadece bir üslup hatırlatmasıdır, metni biçimlendirme."
    )

    prompt = f"""
    Kullanıcının metni: "{text}"
    
    Görevin: Bu metni, belirlenen kurallara ve '{target_style}' üslubuna uygun olarak yeniden yazmak.

    KURALLAR:
    1. Ton: Mutlaka '{target_style}' üslubunu yansıt.
    2. Argo/Kaba İçerik: Eğer orijinal metin argo veya kaba ise, bunu olabilecek **en kibar, nötr ve mesafeli ifadeyle** değiştir.
    3. Çıktı Biçimi: Metin çıktısının başında ve sonunda herhangi bir açıklama, giriş veya kapanış cümlesi **OLMAMALIDIR**. Sadece dönüştürülmüş metni ver.
    4. Font: Kullanıcının talebi üzerine {font} fontunun kullanıldığını hayal et.
    """
    
    data = make_gemini_request(REWRITE_MODEL, prompt, system_prompt)
    
    if "error" in data:
        return {"success": False, "error": f"API Hatası: {data['error']}"}

    candidate = data.get("candidates", [{}])[0]
    generated_text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "").strip()

    if not generated_text:
        # Güvenlik filtresinden dolayı boş yanıt gelmesi durumu
        return {"success": False, "error": "Model, girilen metin için bir yanıt oluşturmayı reddetti veya boş yanıt verdi. (Güvenlik filtresi etkin olabilir.)"}

    # Modelin bazen eklediği fazlalıkları temizleme
    for prefix in ["'metin:", "metin:", "çıktı:", "işte metin:", "'", '"']:
        if generated_text.lower().startswith(prefix):
            generated_text = generated_text[len(prefix):].strip()
    
    for suffix in ["'", '"']:
        if generated_text.endswith(suffix):
            generated_text = generated_text[:-len(suffix)].strip()
        
    return {"success": True, "rewritten": generated_text}


# -----------------------------------------------------
# FLASK ROTLARI
# -----------------------------------------------------

@app.route("/")
def index():
    """Ana sayfayı render eder."""
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    """Metin stilini otomatik olarak sınıflandırır."""
    try:
        data = request.get_json()
        user_message = data.get("text", "").strip()

        if not user_message:
            return jsonify({"success": True, "label": "boş", "score": 0.0})

        classification_result = classify_target_style(user_message)
        
        return jsonify({
            "success": True,
            "label": classification_result["label"],
            "score": classification_result["score"]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/rewrite", methods=["POST"])
def rewrite():
    """Metni yeniden yazar, NLP metriklerini hesaplar ve ton çakışması kontrolü yapar."""
    try:
        data = request.get_json()
        original_text = data.get("text", "").strip()
        selected_tone = data.get("tone", "nötr")
        font = data.get("font", "Arial")

        if not original_text:
            return jsonify({"success": False, "error": "Metin boş olamaz."})

        # 1. Otomatik Ton Tespiti (Çakışma Kontrolü için)
        classification_result = classify_target_style(original_text)
        detected_tone = classification_result["label"]
        
        conflict_warning = None
        # Ton çakışması kontrolü: Modelin tespit ettiği ton ile kullanıcının seçtiği ton farklıysa
        if detected_tone != "belirlenemedi" and detected_tone != "bağlantı_hatası" and selected_tone != detected_tone:
            # Sadece benzer kategoriler değilse uyarı ver
            if not (selected_tone in ["is", "mail", "rapor"] and detected_tone in ["is", "mail", "rapor"]):
                conflict_warning = (
                    f"Girdiğiniz metin, **'{detected_tone.upper()}'** tonuna daha uygun görünüyor. "
                    f"Seçtiğiniz **'{selected_tone.upper()}'** tonda devam etmek, anlamsal bir kaymaya neden olabilir."
                )


        # 2. Yeniden Yazma İşlemi
        rewrite_result = rewrite_message(original_text, selected_tone, font)
        
        if not rewrite_result["success"]:
            return jsonify({"success": False, "error": rewrite_result["error"], "warning": conflict_warning})
        
        rewritten_text = rewrite_result["rewritten"]

        # 3. NLP Metriklerini Hesaplama
        similarity_score = calculate_similarity(original_text, rewritten_text)
        readability_original = calculate_readability(original_text)
        readability_rewritten = calculate_readability(rewritten_text)

        return jsonify({
            "success": True,
            "rewritten": rewritten_text,
            "metrics": {
                "similarity": round(similarity_score, 4),
                "readability_original": readability_original,
                "readability_rewritten": readability_rewritten
            },
            "warning": conflict_warning # Ton çakışma uyarısını döndür
        })
        
    except Exception as e:
        print(f"❌ Ana İşlem Hatası: {e}")
        return jsonify({"success": False, "error": f"Beklenmedik bir sunucu hatası oluştu: {str(e)}", "warning": None})

if __name__ == "__main__":
    app.run(debug=True)
