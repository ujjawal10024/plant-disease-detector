## Plant Disease Detector

This project is a TensorFlow-powered Flask app that predicts plant diseases from leaf photos and recommends remedies. It now offers a multilingual interface (English, Bengali, Hindi, Marathi, Tamil, Gujarati) so you can present it comfortably in class.

### Quick Start (Windows)

1. Double-click `run_app.bat` (or run it from PowerShell / Command Prompt).
2. The script will:
   - check that Python 3.10 is installed (`py -3.10 -V`),
   - upgrade `pip` for that interpreter,
   - install / update everything from `requirements.txt` globally for Python 3.10,
   - start the Flask dev server on `http://127.0.0.1:8000/`.
3. Keep the window open while presenting. Press `Ctrl+C` to stop the server.

### Manual Setup (optional)

```powershell
cd C:\Users\Ujjawal Shakya\Downloads\roproject\roproject
py -3.10 -m pip install --upgrade pip
py -3.10 -m pip install -r requirements.txt
py -3.10 app.py
```

### Language Toggle + Dynamic Translation

- Language dropdown on every page lets you pick Bengali, Marathi, Hindi, English, Tamil, or Gujarati.
- UI labels and PDF labels come from curated translations (works offline).
- Descriptions, remedy steps, and supplement names from the CSV files are translated live via Google Translate (through the `googletrans` helper). That means even the prediction/result content follows the selected language, as long as you have an internet connection.

### Presenting Tomorrow

- Home page → show gallery, switch languages.
- Predict page → upload `static\upload.jpg` or any plant image to demonstrate predictions.
- Result page → download the localized PDF and highlight the treatment steps/supplements.

You’re ready! Good luck with the presentation. 🎉

