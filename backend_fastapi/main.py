import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import pandas as pd
from datetime import datetime
from joblib import load as joblib_load
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Załaduj zmienne środowiskowe
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Pozwól na połączenia z frontendu React (dla dev; w prod przy tym samym originie CORS nie jest wymagany)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserText(BaseModel):
    text: str


def extract_user_data(user_input):
    """Wyciągnij dane użytkownika z tekstu używając GPT-4"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"""Jesteś ekspertem w analizie tekstu. Twoim zadaniem jest wyciągnięcie następujących informacji z podanego tekstu:\n1. Imię\n2. Wiek (w latach) lub rok urodzenia\n3. Płeć (M dla mężczyzny, K dla kobiety)\n4. Czas na 5km (w minutach, może być w formacie MM:SS lub jako liczba minut)\nZwróć odpowiedź w formacie JSON:\n{{\n  \"name\": \"imię lub null\",\n  \"age\": liczba_lat lub null,\n  \"birth_year\": rok_urodzenia lub null,\n  \"gender\": \"M\" lub \"K\" lub null,\n  \"time_5k_minutes\": liczba_minut lub null\n}}\nJeśli nie możesz określić płci z tekstu, spróbuj wywnioskować ją z imienia.\nJeśli podano czas w formacie MM:SS, przekonwertuj na minuty (np. 25:30 = 25.5).\nObecny rok: {datetime.now().year}"""
                },
                {"role": "user", "content": f"Tekst użytkownika: {user_input}"},
            ],
            temperature=0.1,
            max_tokens=200,
            timeout=25,
        )
        result = response.choices[0].message.content
        if not result:
            return None
        return json.loads(result)
    except Exception:
        return None


def infer_gender_from_name(name):
    """Wywnioskuj płeć na podstawie imienia używając AI"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "Jesteś ekspertem w rozpoznawaniu płci na podstawie imion. Zwróć tylko 'M' dla mężczyzny, 'K' dla kobiety lub 'NIEZNANA' jeśli nie możesz określić płci. Bierz pod uwagę imiona z różnych kultur i języków.",
                },
                {"role": "user", "content": f"Jaką płeć ma osoba o imieniu: {name}?"},
            ],
            temperature=0.1,
            max_tokens=10,
            timeout=10,
        )
        result = response.choices[0].message.content
        if result:
            result = result.strip().upper()
            if result in ["M", "K"]:
                return result
        return None
    except Exception:
        return None


def parse_time_5k(value):
    """Waliduje i przelicza czas 5 km na minuty (float). Obsługuje 'MM:SS' i liczby minut."""
    if value is None:
        return None
    try:
        v = float(value)
        if v > 0:
            return v
    except Exception:
        pass
    try:
        text = str(value).strip()
        if ":" in text:
            parts = text.split(":")
            if len(parts) == 2:
                mm = int(parts[0])
                ss = int(parts[1])
                if 0 <= ss < 60 and mm >= 0:
                    return mm + ss / 60.0
    except Exception:
        return None
    return None


def predict_half_marathon_time(model, gender, age, time_5k_seconds):
    birth_year = datetime.now().year - age
    gender_encoded = 1 if str(gender).upper().startswith("M") else 0
    input_data = pd.DataFrame(
        [
            {
                "Średni Czas na 5 km": float(time_5k_seconds),
                "Rocznik": int(birth_year),
                "Płeć_LE": int(gender_encoded),
            }
        ]
    )
    pred = model.predict(input_data)[0]
    return float(pred)


def format_time(seconds):
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# Ładujemy pickla modelu bez importu PyCaret
MODEL_PATH = os.path.join(os.path.dirname(__file__), "app_zad_dom_9_regressor.pkl")
model = joblib_load(MODEL_PATH)


@app.post("/analyze")
async def analyze(user_text: UserText):
    user_input = user_text.text.strip()
    if not user_input:
        return {"error": "Brak tekstu wejściowego"}

    extracted_data = extract_user_data(user_input)
    if not extracted_data:
        return {"error": "Nie udało się przetworzyć danych. Spróbuj ponownie lub podaj dane jaśniej."}

    name = extracted_data.get("name")
    age = extracted_data.get("age")
    birth_year = extracted_data.get("birth_year")
    gender = extracted_data.get("gender")
    time_5k = extracted_data.get("time_5k_minutes")

    if not age and birth_year:
        age = datetime.now().year - int(birth_year)
    if not gender and name:
        gender = infer_gender_from_name(name)

    time_5k_minutes = parse_time_5k(time_5k)

    missing = []
    if not name:
        missing.append("imię")
    if not age:
        missing.append("wiek")
    if not gender:
        missing.append("płeć")
    if time_5k_minutes is None:
        missing.append("czas na 5km (MM:SS lub minuty)")

    if missing:
        return {"error": f"Brakuje/niepoprawne: {', '.join(missing)}", "data": extracted_data}

    time_5k_seconds = float(time_5k_minutes) * 60.0
    predicted_time = predict_half_marathon_time(model, gender, int(age), time_5k_seconds)
    predicted_time_formatted = format_time(predicted_time)

    return {
        "name": name,
        "age": int(age),
        "birth_year": int(birth_year) if birth_year else (datetime.now().year - int(age)),
        "gender": gender,
        "time_5k": float(time_5k_minutes),
        "predicted_time_seconds": predicted_time,
        "predicted_time_formatted": predicted_time_formatted,
    }


# Serwowanie builda React (produkcja)
BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
if os.path.isdir(BUILD_DIR):
    app.mount("/", StaticFiles(directory=BUILD_DIR, html=True), name="static")

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        index_path = os.path.join(BUILD_DIR, "index.html")
        if os.path.isfile(index_path):
            return FileResponse(index_path)
        return {"status": "ok"}
