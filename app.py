from flask import Flask, render_template, request, send_file, redirect, url_for, flash, session
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except Exception:
    # Allow the app to boot without TensorFlow; prediction route will handle gracefully
    load_model = None
    image = None
    TENSORFLOW_AVAILABLE = False
import numpy as np
import pandas as pd
import json
import os
try:
    from xhtml2pdf import pisa
    XHTML2PDF_AVAILABLE = True
except Exception:
    pisa = None
    XHTML2PDF_AVAILABLE = False
try:
    from googletrans import Translator
    translator = Translator()
    TRANSLATOR_AVAILABLE = True
except Exception:
    translator = None
    TRANSLATOR_AVAILABLE = False

app = Flask(__name__, template_folder='.', static_folder='static')
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')

SUPPORTED_LANGUAGES = {
    "en": "English",
    "bn": "Bengali",
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
    "gu": "Gujarati"
}
DEFAULT_LANG = "en"

translations = {
    "en": {
        "home_page_title": "Welcome to Plant Disease Detector",
        "app_title": "Plant Disease Detection System",
        "app_tagline": "Upload leaf images, predict diseases, and access remedy information",
        "predict_cta": "Predict Disease",
        "info_cta": "View Disease Info",
        "language_label": "Language",
        "language_helper": "Choose your preferred language",
        "predict_page_title": "Disease Prediction",
        "upload_heading": "Upload Leaf Image",
        "predict_button": "Predict",
        "supported_formats": "Supported formats: JPG, PNG. Image will be resized automatically.",
        "back_home": "Back to Home",
        "result_page_title": "Prediction Result",
        "result_heading": "Disease Predicted",
        "description_label": "Description",
        "steps_label": "Remedy Steps",
        "supplement_label": "Supplement",
        "download_pdf": "Download as PDF",
        "predict_again": "Predict Another",
        "info_page_title": "Disease Info",
        "list_title": "Plant Disease Information",
        "detail_page_title": "Disease Detail",
        "recommended_supplement": "Recommended Supplement",
        "back_to_list": "Back to Disease List",
        "pdf_title": "Plant Disease Report",
        "pdf_disease_label": "Disease",
        "pdf_description_label": "Description",
        "pdf_steps_label": "Remedy Steps",
        "pdf_supplement_label": "Supplement",
        "prediction_required": "Please make a prediction first.",
    },
    "bn": {
        "home_page_title": "উদ্ভিদ রোগ শনাক্তকরণে স্বাগতম",
        "app_title": "উদ্ভিদ রোগ শনাক্তকরণ ব্যবস্থা",
        "app_tagline": "পাতার ছবি আপলোড করুন, রোগ জানুন এবং প্রতিকার তথ্য পান",
        "predict_cta": "রোগ শনাক্ত করুন",
        "info_cta": "রোগের তথ্য দেখুন",
        "language_label": "ভাষা",
        "language_helper": "আপনার পছন্দের ভাষা নির্বাচন করুন",
        "predict_page_title": "রোগ পূর্বাভাস",
        "upload_heading": "পাতার ছবি আপলোড করুন",
        "predict_button": "পূর্বাভাস দিন",
        "supported_formats": "সমর্থিত ফরম্যাট: JPG, PNG। ছবি স্বয়ংক্রিয়ভাবে আকার বদলাবে।",
        "back_home": "হোমে ফিরুন",
        "result_page_title": "পূর্বাভাস ফলাফল",
        "result_heading": "সনাক্ত রোগ",
        "description_label": "বর্ণনা",
        "steps_label": "প্রতিকার ধাপ",
        "supplement_label": "সম্পূরক",
        "download_pdf": "পিডিএফ ডাউনলোড করুন",
        "predict_again": "আরও একটি ছবি যাচাই করুন",
        "info_page_title": "রোগের তথ্য",
        "list_title": "উদ্ভিদ রোগ সম্পর্কিত তথ্য",
        "detail_page_title": "রোগের বিস্তারিত",
        "recommended_supplement": "প্রস্তাবিত সম্পূরক",
        "back_to_list": "রোগ তালিকায় ফিরুন",
        "pdf_title": "উদ্ভিদ রোগ প্রতিবেদন",
        "pdf_disease_label": "রোগ",
        "pdf_description_label": "বর্ণনা",
        "pdf_steps_label": "প্রতিকার ধাপ",
        "pdf_supplement_label": "সম্পূরক",
        "prediction_required": "দয়া করে আগে একটি পূর্বাভাস তৈরি করুন।",
    },
    "hi": {
        "home_page_title": "पादप रोग पहचान प्रणाली में आपका स्वागत है",
        "app_title": "पादप रोग पहचान प्रणाली",
        "app_tagline": "पत्ती की तस्वीर अपलोड करें, रोग जानें और उपचार जानकारी पाएँ",
        "predict_cta": "रोग पहचानें",
        "info_cta": "रोग जानकारी देखें",
        "language_label": "भाषा",
        "language_helper": "अपनी पसंदीदा भाषा चुनें",
        "predict_page_title": "रोग पूर्वानुमान",
        "upload_heading": "पत्ती की छवि अपलोड करें",
        "predict_button": "पूर्वानुमान लगाएँ",
        "supported_formats": "समर्थित प्रारूप: JPG, PNG. छवि स्वतः री-साइज़ होगी।",
        "back_home": "होम पर लौटें",
        "result_page_title": "पूर्वानुमान परिणाम",
        "result_heading": "पूर्वानुमानित रोग",
        "description_label": "विवरण",
        "steps_label": "उपचार के चरण",
        "supplement_label": "अनुशंसित पूरक",
        "download_pdf": "पीडीएफ डाउनलोड करें",
        "predict_again": "दूसरी छवि जाँचें",
        "info_page_title": "रोग जानकारी",
        "list_title": "पादप रोग जानकारी",
        "detail_page_title": "रोग विवरण",
        "recommended_supplement": "अनुशंसित पूरक",
        "back_to_list": "रोग सूची पर लौटें",
        "pdf_title": "पादप रोग रिपोर्ट",
        "pdf_disease_label": "रोग",
        "pdf_description_label": "विवरण",
        "pdf_steps_label": "उपचार चरण",
        "pdf_supplement_label": "पूरक",
        "prediction_required": "कृपया पहले एक पूर्वानुमान बनाएँ।",
    },
    "mr": {
        "home_page_title": "वनस्पती रोग निदानात स्वागत",
        "app_title": "वनस्पती रोग निदान प्रणाली",
        "app_tagline": "पानाचे चित्र अपलोड करा, रोग ओळखा आणि उपाय जाणून घ्या",
        "predict_cta": "रोग भाकीत करा",
        "info_cta": "रोगांची माहिती पहा",
        "language_label": "भाषा",
        "language_helper": "आपली पसंतीची भाषा निवडा",
        "predict_page_title": "रोग भाकीत",
        "upload_heading": "पानाचे छायाचित्र अपलोड करा",
        "predict_button": "भाकीत करा",
        "supported_formats": "समर्थित स्वरूप: JPG, PNG. प्रतिमा आपोआप रीसाइज होईल.",
        "back_home": "मुख्य पृष्ठावर जा",
        "result_page_title": "भाकीत परिणाम",
        "result_heading": "ओळखलेला रोग",
        "description_label": "वर्णन",
        "steps_label": "उपाय पावले",
        "supplement_label": "सुचविलेले पूरक",
        "download_pdf": "PDF डाउनलोड करा",
        "predict_again": "पुन्हा तपासा",
        "info_page_title": "रोग माहिती",
        "list_title": "वनस्पती रोग माहिती",
        "detail_page_title": "रोग तपशील",
        "recommended_supplement": "शिफारस केलेले पूरक",
        "back_to_list": "रोग यादीकडे परत",
        "pdf_title": "वनस्पती रोग अहवाल",
        "pdf_disease_label": "रोग",
        "pdf_description_label": "वर्णन",
        "pdf_steps_label": "उपाय पावले",
        "pdf_supplement_label": "पूरक",
        "prediction_required": "कृपया आधी भाकीत तयार करा.",
    },
    "ta": {
        "home_page_title": "தாவர நோய் கண்டறிதலுக்கு வரவேற்கிறோம்",
        "app_title": "தாவர நோய் கண்டறிதல் அமைப்பு",
        "app_tagline": "இலையின் படத்தை பதிவேற்றி, நோயை அறிந்து, சிகிச்சை தகவலைப் பெறுங்கள்",
        "predict_cta": "நோயை கணிக்க",
        "info_cta": "நோய் தகவல் காண்க",
        "language_label": "மொழி",
        "language_helper": "உங்களுக்கு பிடித்த மொழியைத் தேர்ந்தெடுக்கவும்",
        "predict_page_title": "நோய் கணிப்பு",
        "upload_heading": "இலையின் படத்தை பதிவேற்றவும்",
        "predict_button": "கணித்து காண்க",
        "supported_formats": "ஆதரிக்கப்படும் வடிவங்கள்: JPG, PNG. படம் தானாக அளவுசெய்யப்படும்.",
        "back_home": "முகப்புக்கு திரும்ப",
        "result_page_title": "கணிப்பு முடிவு",
        "result_heading": "கணிக்கப்பட்ட நோய்",
        "description_label": "விளக்கம்",
        "steps_label": "சிகிச்சை நடவடிக்கைகள்",
        "supplement_label": "பரிந்துரைக்கப்பட்ட சேர்க்கை",
        "download_pdf": "PDF பதிவிறக்கவும்",
        "predict_again": "மற்றொன்றை முயற்சிக்க",
        "info_page_title": "நோய் தகவல்",
        "list_title": "தாவர நோய் தகவல்",
        "detail_page_title": "நோய் விவரம்",
        "recommended_supplement": "பரிந்துரைக்கப்பட்ட சேர்க்கை",
        "back_to_list": "நோய் பட்டியலுக்கு திரும்ப",
        "pdf_title": "தாவர நோய் அறிக்கை",
        "pdf_disease_label": "நோய்",
        "pdf_description_label": "விளக்கம்",
        "pdf_steps_label": "சிகிச்சை நடவடிக்கைகள்",
        "pdf_supplement_label": "சேர்க்கை",
        "prediction_required": "முதலில் ஒரு கணிப்பை செய்யவும்.",
    },
    "gu": {
        "home_page_title": "વનસ્પતિ રોગ શોધકમાં આપનું સ્વાગત છે",
        "app_title": "વનસ્પતિ રોગ શોધક સિસ્ટમ",
        "app_tagline": "પાંદડાની છબી અપલોડ કરો, રોગની આગાહી કરો અને ઉપચારની માહિતી મેળવો",
        "predict_cta": "રોગની આગાહી કરો",
        "info_cta": "રોગની માહિતી જુઓ",
        "language_label": "ભાષા",
        "language_helper": "તમારી પસંદગીની ભાષા પસંદ કરો",
        "predict_page_title": "રોગની આગાહી",
        "upload_heading": "પાંદડાની છબી અપલોડ કરો",
        "predict_button": "આગાહી કરો",
        "supported_formats": "સમર્થિત ફોર્મેટ: JPG, PNG. છબી આપમેળે રી-સાઇઝ થશે.",
        "back_home": "હોમ પર પાછા જાઓ",
        "result_page_title": "આગાહી પરિણામ",
        "result_heading": "આગાહી કરેલ રોગ",
        "description_label": "વર્ણન",
        "steps_label": "ઉપચારના પગલા",
        "supplement_label": "ભલામણ કરેલ પૂરક",
        "download_pdf": "PDF ડાઉનલોડ કરો",
        "predict_again": "બીજી છબી તપાસો",
        "info_page_title": "રોગની માહિતી",
        "list_title": "વનસ્પતિ રોગની માહિતી",
        "detail_page_title": "રોગની વિગત",
        "recommended_supplement": "ભલામણ કરેલ પૂરક",
        "back_to_list": "રોગ સૂચિ પર પાછા જાઓ",
        "pdf_title": "વનસ્પતિ રોગ અહેવાલ",
        "pdf_disease_label": "રોગ",
        "pdf_description_label": "વર્ણન",
        "pdf_steps_label": "ઉપચારના પગલા",
        "pdf_supplement_label": "પૂરક",
        "prediction_required": "કૃપા કરીને પહેલા આગાહી કરો.",
    }
}

TRANSLATION_CODE_LOOKUP = {
    "en": "en",
    "bn": "bn",
    "hi": "hi",
    "mr": "mr",
    "ta": "ta",
    "gu": "gu"
}

translation_cache = {}


def get_lang():
    lang = request.args.get('lang') or request.form.get('lang')
    if lang in SUPPORTED_LANGUAGES:
        session['lang'] = lang
        return lang
    return session.get('lang', DEFAULT_LANG)


def get_strings(lang_code: str):
    return translations.get(lang_code, translations[DEFAULT_LANG])


def translate_text(text: str, lang: str) -> str:
    if not text or lang == DEFAULT_LANG or not TRANSLATOR_AVAILABLE:
        return text
    target = TRANSLATION_CODE_LOOKUP.get(lang)
    if not target:
        return text
    cache_key = (target, text)
    if cache_key in translation_cache:
        return translation_cache[cache_key]
    try:
        translated_value = translator.translate(text, dest=target).text
        translation_cache[cache_key] = translated_value
        return translated_value
    except Exception:
        return text


def localize_content(disease_info: dict, supplement_info: dict, lang: str):
    disease_localized = {
        "Description": translate_text(disease_info.get("Description", ""), lang),
        "Steps": translate_text(disease_info.get("Steps", ""), lang)
    }
    supplement_localized = {
        "Name": translate_text(supplement_info.get("Name", ""), lang),
        "Link": supplement_info.get("Link", "#")
    }
    return disease_localized, supplement_localized

# Lazy-load the trained CNN model (so the app can start even if load fails)
model = None

def get_model():
    global model
    if model is not None:
        return model
    if not TENSORFLOW_AVAILABLE:
        return None
    try:
        model = load_model('Plant_disease_Predictive_System.h5')
        return model
    except Exception as e:
        # Keep model as None if loading fails
        return None

# Load class labels safely
labels = {}
try:
    with open('class_indices.json', 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
    # The format is {"index": "ClassName"}, so we can use it directly
    labels = {int(index): class_name for index, class_name in class_indices.items()}
except Exception as e:
    labels = {}

# Mapping between model labels and readable disease names
name_map = {
    "Apple___Apple_scab": "Apple : scab",
    "Apple___Black_rot": "Apple : Black rot",
    "Apple___Cedar_apple_rust": "Apple :Cedar rust",
    "Apple___healthy": "Apple :Healthy",
    "Blueberry___healthy": "Blueberry :Healthy",
    "Cherry_(including_sour)___Powdery_mildew": "Cherry : Powdery mildew",
    "Cherry_(including_sour)___healthy": "Cherry :Healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn : Cercospora leaf spot | Gray leaf spot",
    "Corn_(maize)___Common_rust_": "Corn : Common Rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn  :Northern Leaf Blight",
    "Corn_(maize)___healthy": "Corn :Healthy",
    "Grape___Black_rot": "Grape : Black rot",
    "Grape___Esca_(Black_Measles)": "Grape : Esca | Black Measles",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape : Leaf blight | Isariopsis Leaf Spot",
    "Grape___healthy": "Grape : Healthy",
    "Orange___Haunglongbing_(Citrus_greening)": "Orange : Haunglongbing | Citrus greening",
    "Peach___Bacterial_spot": "Peach : Bacterial spot",
    "Peach___healthy": "Peach : Healthy",
    "Pepper,_bell___Bacterial_spot": "Pepper bell : Bacterial spot",
    "Pepper,_bell___healthy": "Pepper bell : Healthy",
    "Potato___Early_blight": "Potato : Early blight",
    "Potato___Late_blight": "Potato : Late blight",
    "Potato___healthy": "Potato : Healthy",
    "Raspberry___healthy": "Raspberry : Healthy",
    "Soybean___healthy": "Soybean : Healthy",
    "Squash___Powdery_mildew": "Squash : Powdery mildew",
    "Strawberry___Leaf_scorch": "Strawberry : Leaf scorch",
    "Strawberry___healthy": "Strawberry : healthy",
    "Tomato___Bacterial_spot": "Tomato : Bacterial spot",
    "Tomato___Early_blight": "Tomato : Early blight",
    "Tomato___Late_blight": "Tomato : Late blight",
    "Tomato___Leaf_Mold": "Tomato : Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Tomato : Septoria leaf spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato : Spider mites | Two-spotted spider mite",
    "Tomato___Target_Spot": "Tomato : Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato : Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato : Mosaic virus",
    "Tomato___healthy": "Tomato : Healthy"
}

# Load CSV data safely
try:
    disease_info_df = pd.read_csv('disease_info.csv', encoding='ISO-8859-1')
except Exception:
    disease_info_df = pd.DataFrame(columns=['disease_name', 'description', 'Possible Steps'])

try:
    supplement_info_df = pd.read_csv('supplement_info.csv', encoding='ISO-8859-1')
except Exception:
    supplement_info_df = pd.DataFrame(columns=['disease_name', 'supplement name', 'buy link'])


# Home Page
@app.route('/')
def index():
    lang = get_lang()
    strings = get_strings(lang)
    images_directory = os.path.join(app.static_folder or 'static', 'plant_images')
    if os.path.isdir(images_directory):
        plant_images = os.listdir(images_directory)
    else:
        plant_images = []
    return render_template(
        'index.html',
        images=plant_images,
        lang=lang,
        strings=strings,
        language_options=SUPPORTED_LANGUAGES
    )


# Predict Disease from Uploaded Image
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    lang = get_lang()
    strings = get_strings(lang)
    if request.method == 'POST':
        img_file = request.files.get('image')
        if img_file:
            upload_directory = app.static_folder or 'static'
            os.makedirs(upload_directory, exist_ok=True)
            img_path = os.path.join(upload_directory, 'upload.jpg')
            img_file.save(img_path)
            prediction = 'Prediction unavailable'
            csv_name = None
            disease_info = {"Description": "", "Steps": ""}
            supplement_info = {"Name": "", "Link": ""}

            mdl = get_model()
            try:
                if TENSORFLOW_AVAILABLE and mdl is not None and image is not None:
                    img = image.load_img(img_path, target_size=(224, 224))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0

                    result = mdl.predict(img_array)
                    predicted_class = int(np.argmax(result))
                    prediction = labels.get(predicted_class, str(predicted_class))
                    csv_name = name_map.get(prediction, prediction)
                else:
                    if not TENSORFLOW_AVAILABLE:
                        prediction = 'TensorFlow not available. Install TensorFlow to enable predictions.'
                    elif mdl is None:
                        prediction = 'Model not loaded. Check model file.'
                    elif image is None:
                        prediction = 'Image processing not available.'
                    else:
                        prediction = 'Model not loaded. Install TensorFlow to enable predictions.'
                    csv_name = None
            except Exception as e:
                prediction = f'Prediction error: {str(e)}'
                csv_name = None

            # Fetch disease info if we have a csv_name
            if csv_name:
                disease_row = disease_info_df[
                    disease_info_df['disease_name'].str.strip().str.lower() == csv_name.strip().lower()
                ]
                if not disease_row.empty:
                    row = disease_row.iloc[0]
                    disease_info = {
                        "Description": row['description'],
                        "Steps": row['Possible Steps']
                    }

            # ✅ FIXED: Match supplement using csv_name (not raw prediction)
            if csv_name:
                supp_row = supplement_info_df[
                    supplement_info_df['disease_name'].str.strip().str.lower() == csv_name.strip().lower()
                ]
                if not supp_row.empty:
                    row = supp_row.iloc[0]
                    supplement_info = {
                        "Name": row['supplement name'],
                        "Link": row['buy link']
                    }

            session['last_result'] = {
                "prediction": csv_name if csv_name else prediction,
                "disease_info": disease_info,
                "supplement_info": supplement_info
            }
            return redirect(url_for('show_result', lang=lang))
    return render_template(
        'predict.html',
        lang=lang,
        strings=strings,
        language_options=SUPPORTED_LANGUAGES
    )


@app.route('/result')
def show_result():
    lang = get_lang()
    strings = get_strings(lang)
    result_payload = session.get('last_result')
    if not result_payload:
        flash(strings["prediction_required"], 'warning')
        return redirect(url_for('predict', lang=lang))
    disease_localized, supplement_localized = localize_content(
        result_payload["disease_info"],
        result_payload["supplement_info"],
        lang
    )
    return render_template(
        'submit.html',
        prediction=result_payload["prediction"],
        disease_info=disease_localized,
        supplement_info=supplement_localized,
        lang=lang,
        strings=strings,
        language_options=SUPPORTED_LANGUAGES
    )


# Disease Info List Page
@app.route('/disease-info')
def disease_info():
    lang = get_lang()
    strings = get_strings(lang)
    merged = pd.merge(
        disease_info_df,
        supplement_info_df[['disease_name', 'supplement name']],
        on='disease_name',
        how='left'
    )
    return render_template(
        'disease_info.html',
        diseases=merged.to_dict(orient='records'),
        lang=lang,
        strings=strings,
        language_options=SUPPORTED_LANGUAGES
    )


# Disease Detail Page
@app.route('/disease/<name>')
def disease_detail(name):
    lang = get_lang()
    strings = get_strings(lang)
    row = disease_info_df[
        disease_info_df['disease_name'].str.strip().str.lower() == name.strip().lower()
    ]
    supp_row = supplement_info_df[
        supplement_info_df['disease_name'].str.strip().str.lower() == name.strip().lower()
    ]
    
    if not row.empty:
        r = row.iloc[0]
        supplement = supp_row.iloc[0] if not supp_row.empty else None

        disease_info = {
            "Description": r['description'],
            "Steps": r['Possible Steps']
        }

        supplement_info = {
            "Name": supplement['supplement name'] if supplement is not None else "Not available",
            "Link": supplement['buy link'] if supplement is not None else "#"
        }
        disease_localized, supplement_localized = localize_content(disease_info, supplement_info, lang)

        return render_template(
            'disease_detail.html',
            disease_name=name,
            disease_info=disease_localized,
            supplement_info=supplement_localized,
            lang=lang,
            strings=strings,
            language_options=SUPPORTED_LANGUAGES
        )
    return redirect(url_for('disease_info', lang=lang))


# Generate and Download PDF Report
@app.route('/download-pdf')
def download_pdf():
    lang = get_lang()
    strings = get_strings(lang)
    prediction = request.args.get('prediction')
    desc = request.args.get('desc')
    steps = request.args.get('steps')
    supp_name = request.args.get('supp')
    buy_link = request.args.get('link')

    html = render_template('report.html',
                           prediction=prediction,
                           description=desc,
                           steps=steps,
                           supplement=supp_name,
                           link=buy_link,
                           lang=lang,
                           strings=strings,
                           language_options=SUPPORTED_LANGUAGES)
    
    if not XHTML2PDF_AVAILABLE:
        # Fallback: return HTML as a downloadable file if PDF library is not available
        fallback_path = os.path.join(app.static_folder or 'static', 'result.html')
        os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
        with open(fallback_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return send_file(fallback_path, as_attachment=True)

    pdf_path = os.path.join(app.static_folder or 'static', 'result.pdf')
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    with open(pdf_path, "w+b") as result_file:
        pisa.CreatePDF(src=html, dest=result_file)

    return send_file(pdf_path, as_attachment=True)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=8000)
