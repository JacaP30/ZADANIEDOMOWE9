import streamlit as st
import openai
import pandas as pd
import numpy as np
import pickle
import os
import json
from dotenv import load_dotenv
import re
from datetime import datetime
import base64

from langfuse.decorators import observe
from langfuse.openai import OpenAI


# Langfuse import
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False



def set_bg(png_file):
    with open(png_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}
    
    /* Poprawa czytelności tekstu */
    .stApp > div > div > div > div {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        backdrop-filter: blur(5px);
        margin: 10px 0;
    }}
    
    /* Style dla głównego contentu */
    .main .block-container {{
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 2px 4px 4px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin-top: 1rem;
    }}
    
    /* Style dla tytułów */
    h1, h2, h3 {{
        color: #0b1e52ff !important;
        text-shadow: 4px 4px 8px #75cfcfff;
        font-weight: bold !important;
    }}
    
    /* Style dla zwykłego tekstu */
    p, div, span {{
        color: #1f2937 !important;
        text-shadow: 1px 1px 1px rgba(255, 255, 255, 0.8);
    }}
    
    /* Style dla formularzy */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div {{
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1f2937 !important;
        border: 2px solid #e5e7eb !important;
        caret-color: #3b82f6 !important;
    }}
    
    /* Kursor w polach tekstowych */
    .stTextArea > div > div > textarea:focus,
    .stTextInput > div > div > input:focus {{
        caret-color: #1e40af !important;
        border-color: #3b82f6 !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }}
    
    /* Style dla etykiet pól tekstowych */
    .stTextArea > label,
    .stTextInput > label {{
        color: #02040aff !important;   
        font-weight: bold !important;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.9) !important;
        font-size: 16px !important;
    }}
    
    /* Style dla przycisków */
    .stButton > button {{
        background-color: #3b82f6 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        text-shadow: none !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }}
    
    .stButton > button:hover {{
        background-color: #2563eb !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    }}
    
    /* Style dla metrics */
    .metric-container {{
        background-color: rgba(255, 255, 255, 0.9) !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border: 1px solid #e5e7eb !important;
    }}
    
    /* Style dla alertów i komunikatów */
    .stAlert {{
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 8px !important;
        backdrop-filter: blur(5px) !important;
    }}
    
    /* Style dla success/error/info/warning */
    .stSuccess, .stError, .stInfo, .stWarning {{
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1f2937 !important;
        border-radius: 8px !important;
        backdrop-filter: blur(5px) !important;
    
    }}
    
    /* Style dla stopki */
    .footer {{
        background-color: rgba(255, 255, 255, 0.9) !important;
        padding: 10px !important;
        border-radius: 8px !important;
        margin-top: 20px !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Użycie:
set_bg("images/background.png")# import matplotlib.patheffects


# Załaduj zmienne środowiskowe
load_dotenv()

# Konfiguracja OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Konfiguracja Langfuse (opcjonalna)
langfuse_client = None
if LANGFUSE_AVAILABLE:
    # Sprawdź zmienne środowiskowe
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    print(f"🔍 Langfuse config check:")
    print(f"  - SECRET_KEY: {'✅ Set' if secret_key else '❌ Missing'}")
    print(f"  - PUBLIC_KEY: {'✅ Set' if public_key else '❌ Missing'}")
    print(f"  - HOST: {host}")
    
    if secret_key and public_key:
        try:
            langfuse_client = Langfuse(
                secret_key=secret_key,
                public_key=public_key,
                host=host
            )
            print("✅ Langfuse initialized successfully")
        except Exception as e:
            print(f"⚠️ Langfuse initialization failed: {e}")
            langfuse_client = None
    else:
        print("⚠️ Langfuse keys missing - skipping initialization")
else:
    print("⚠️ Langfuse not available - library not installed")


def log_to_langfuse(function_name, input_data, output_data, metadata=None):
    """Loguj wywołanie funkcji do Langfuse"""
    if langfuse_client is None:
        return
    
    try:
        trace = langfuse_client.trace(
            name=function_name,
            input=input_data,
            output=output_data,
            metadata=metadata or {}
        )
        # Flush natychmiast aby dane zostały wysłane
        langfuse_client.flush()
        return trace
    except Exception as e:
        print(f"Langfuse logging error: {e}")
        return None

def load_model():
    """Załaduj wytrenowany model regresji PyCaret"""
    try:
        # Import PyCaret functions
        from pycaret.regression import load_model as pycaret_load_model, predict_model as pycaret_predict_model
        
        # Załaduj model PyCaret
        model = pycaret_load_model('model/app_zad_dom_9_regressor')
        return model
    except Exception as e:
        st.error(f"Błąd podczas ładowania modelu: {e}")
        return None

def extract_user_data(user_input):
    """Wyciągnij wszystkie dane użytkownika z tekstu używając AI"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """Jesteś ekspertem w analizie tekstu. Twoim zadaniem jest wyciągnięcie następujących informacji z podanego tekstu:
                    1. Imię
                    2. Wiek (w latach) lub rok urodzenia
                    3. Płeć (M dla mężczyzny, K dla kobiety)
                    4. Czas na 5km (w minutach, może być w formacie MM:SS lub jako liczba minut)
                    
                    Zwróć odpowiedź w formacie JSON:
                    {
                        "name": "imię lub null",
                        "age": liczba_lat lub null,
                        "birth_year": rok_urodzenia lub null,
                        "gender": "M" lub "K" lub null,
                        "time_5k_minutes": liczba_minut lub null
                    }
                    
                    Jeśli nie możesz określić płci z tekstu, spróbuj wywnioskować ją z imienia.
                    Jeśli podano czas w formacie MM:SS, przekonwertuj na minuty (np. 25:30 = 25.5).
                    Obecny rok: {datetime.now().year}"""
                },
                {
                    "role": "user", 
                    "content": f"Tekst użytkownika: {user_input}"
                }
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        result = response.choices[0].message.content
        if not result:
            return None
        result = result.strip()
        
        # Spróbuj sparsować JSON
        try:
            data = json.loads(result)
            
            # Loguj do Langfuse
            log_to_langfuse(
                function_name="extract_user_data",
                input_data={"user_input": user_input},
                output_data=data,
                metadata={
                    "model": "gpt-4",
                    "temperature": 0.1,
                    "max_tokens": 200
                }
            )
            
            return data
        except json.JSONDecodeError:
            # Jeśli nie udało się sparsować JSON, zwróć None
            return None
            
    except Exception as e:
        st.error(f"Błąd podczas komunikacji z AI: {e}")
        return None

def infer_gender_from_name(name):
    """Wywnioskuj płeć na podstawie imienia używając AI"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """Jesteś ekspertem w rozpoznawaniu płci na podstawie imion. 
                    Zwróć tylko 'M' dla mężczyzny, 'K' dla kobiety lub 'NIEZNANA' jeśli nie możesz określić płci.
                    Bierz pod uwagę imiona z różnych kultur i języków."""
                },
                {
                    "role": "user",
                    "content": f"Jaką płeć ma osoba o imieniu: {name}?"
                }
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        if result:
            result = result.strip().upper()
            if result in ['M', 'K']:
                # Loguj do Langfuse
                log_to_langfuse(
                    function_name="infer_gender_from_name",
                    input_data={"name": name},
                    output_data={"gender": result},
                    metadata={
                        "model": "gpt-4",
                        "temperature": 0.1,
                        "max_tokens": 10
                    }
                )
                return result
        return None
            
    except Exception as e:
        st.error(f"Błąd podczas rozpoznawania płci: {e}")
        return None

def predict_half_marathon_time(model, gender, age, time_5k):
    """Przewiduj czas półmaratonu na podstawie danych użytkownika"""
    try:
        from pycaret.regression import predict_model as pycaret_predict_model
        
        # Oblicz rok urodzenia z wieku
        birth_year = datetime.now().year - age
        
        # Kodowanie płci: M=1, K=0 (zgodnie z treningiem)
        gender_encoded = 1 if gender == 'M' else 0
        
        # Przygotuj dane wejściowe zgodnie z formatem z notebooka
        # Model oczekuje: 'Średni Czas na 5 km', 'Rocznik', 'Płeć_LE'
        input_data = pd.DataFrame([{
            'Średni Czas na 5 km': time_5k,  # czas w sekundach
            'Rocznik': birth_year,
            'Płeć_LE': gender_encoded
        }])
        
        # Dokonaj predykcji używając PyCaret
        prediction_df = pycaret_predict_model(model, data=input_data)
        prediction = prediction_df['prediction_label'].iloc[0]
        
        return prediction
        
    except Exception as e:
        st.error(f"Błąd podczas predykcji: {e}")
        return None

def format_time(seconds):
    """Formatuj czas z sekund na format HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
    st.title("Predyktor Czasu Półmaratonu")
    # st.markdown("---")
    
    # Załaduj model
    model = load_model()
    if model is None:
        st.stop()
    
    st.markdown("### Opowiedz o sobie:")
    
    # Instrukcje dla użytkownika
    st.info("""
    📝 **Podaj następujące informacje w dowolnej formie:**
    - **Imię**,  **Wiek**,  **Czas na 5km**,  **Płeć** (jeśli chcesz)
    
    💡 **Przykłady:**
    - "Nazywam się Kasia, urodziłam się w 1990 roku, 5km w 26.5 minuty"
    - "Jestem Anna, mam 28 lat i biegam 5km w 24 minuty"
    - "Marek, 35 lat, czas na 5km: 22:45"    
    - Możesz też po prostu "Janek 75 25"  :)
    """)
    
    # Formularz dla użytkownika
    with st.form("user_data_form"):
        user_input = st.text_area(
            "## **Wpisz informacje o sobie:**",
            height=100,
            placeholder="Napisz coś o sobie... ",
            help="Podaj swoje dane w dowolnej formie - AI wyciągnie potrzebne informacje"
        )
        
        submitted = st.form_submit_button("🔍 Analizuj i przewiduj czas półmaratonu", use_container_width=True)
    
    if submitted:
        if not user_input.strip():
            st.error("Proszę podać informacje o sobie!")
            st.stop()
        
        # Analiza danych przez AI
        with st.spinner("🤖 AI analizuje Twoje dane..."):
            extracted_data = extract_user_data(user_input.strip())
            
            if not extracted_data:
                st.error("Nie udało się przetworzyć danych. Spróbuj podać informacje w innej formie.")
                st.stop()
        
        # Wyświetl wyciągnięte dane
        st.markdown("### 🔍 Dane wyciągnięte przez AI:")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            name = extracted_data.get('name')
            if name:
                st.success(f"**Imię:** {name}")
            else:
                st.warning("**Imię:** nie rozpoznano")
        
        with col2:
            age = extracted_data.get('age')
            birth_year = extracted_data.get('birth_year')
            
            if age:
                st.success(f"**Wiek:** {age} lat")
                if not birth_year:
                    birth_year = datetime.now().year - age
            elif birth_year:
                age = datetime.now().year - birth_year
                st.success(f"**Wiek:** {age} lat (ur. {birth_year})")
            else:
                st.error("**Wiek:** nie rozpoznano")
        
        with col3:
            gender = extracted_data.get('gender')
            if gender:
                gender_text = "Mężczyzna" if gender == 'M' else "Kobieta"
                st.success(f"**Płeć:** {gender_text}")
            else:
                # Spróbuj wywnioskować z imienia
                if name:
                    st.info("Rozpoznawanie płci z imienia...")
                    gender = infer_gender_from_name(name)
                    if gender:
                        gender_text = "Mężczyzna" if gender == 'M' else "Kobieta"
                        st.success(f"**Płeć:** {gender_text} (z imienia)")
                    else:
                        st.error("**Płeć:** nie rozpoznano")
                else:
                    st.error("**Płeć:** nie rozpoznano")
        
        with col4:
            time_5k = extracted_data.get('time_5k_minutes')
            if time_5k:
                st.success(f"**5km:** {time_5k} min")
            else:
                st.error("**Czas 5km:** nie rozpoznano")
        
        # Sprawdź czy mamy wszystkie potrzebne dane
        missing_data = []
        if not name:
            missing_data.append("imię")
        if not age:
            missing_data.append("wiek")
        if not gender:
            missing_data.append("płeć")
        if not time_5k:
            missing_data.append("czas na 5km")
        
        if missing_data:
            st.error(f"❌ Brakuje następujących danych: {', '.join(missing_data)}")
            st.info("💡 Spróbuj podać informacje w bardziej szczegółowy sposób lub w innym formacie.")
            st.stop()
        
        # Predykcja
        # st.markdown("---")
        with st.spinner("🏃‍♂️ Przewidywanie czasu półmaratonu..."):
            
            # Konwersja czasu 5km na sekundy
            time_5k_seconds = time_5k * 60
            
            # Predykcja
            predicted_time = predict_half_marathon_time(model, gender, age, time_5k_seconds)
            
            if predicted_time is not None:
                # Loguj całkowitą predykcję do Langfuse
                log_to_langfuse(
                    function_name="half_marathon_prediction",
                    input_data={
                        "name": name,
                        "age": age,
                        "gender": gender,
                        "time_5k_minutes": time_5k,
                        "original_input": user_input
                    },
                    output_data={
                        "predicted_time_seconds": predicted_time,
                        "predicted_time_formatted": format_time(predicted_time)
                    },
                    metadata={
                        "model_type": "pycaret_regression",
                        "features": ["Średni Czas na 5 km", "Rocznik", "Płeć_LE"]
                    }
                )
                
                # Główny wynik
                # st.markdown("---")
                predicted_time_formatted = format_time(predicted_time)
                
                st.markdown(f"""
                <div style="
                    text-align: center; 
                    padding: 25px; 
                    border: 3px solid #4CAF50; 
                    border-radius: 15px; 
                    background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.95));
                    backdrop-filter: blur(10px);
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
                    margin: 20px 0;
                ">
                    <h2 style="
                        color: #2E7D32 !important; 
                        margin-bottom: 15px;
                        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.8);
                        font-weight: bold;
                    ">🏃‍♂️ Przewidywany czas półmaratonu:</h2>
                    <h1 style="
                        color: #1B5E20 !important; 
                        font-size: 3.5em; 
                        margin: 10px 0;
                        text-shadow: 3px 3px 6px rgba(255, 255, 255, 0.9);
                        font-weight: bold;
                        letter-spacing: 2px;
                    ">{predicted_time_formatted}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Dodatkowe informacje
                # st.markdown("---")
                st.markdown("### 📊 Analiza:")
                
                # Oblicz tempo na km dla półmaratonu
                pace_per_km = predicted_time / 21.1
                pace_minutes = int(pace_per_km // 60)
                pace_seconds = int(pace_per_km % 60)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Tempo na km:** {pace_minutes}:{pace_seconds:02d} min/km")
                
                with col2:
                    # Porównanie z czasem 5km
                    pace_5k = (time_5k * 60) / 5
                    pace_5k_min = int(pace_5k // 60)
                    pace_5k_sec = int(pace_5k % 60)
                    st.info(f"**Tempo 5km:** {pace_5k_min}:{pace_5k_sec:02d} min/km")
                
                # Motywujący komentarz
                st.success("💪 Powodzenia w treningu! Pamiętaj, że regularne treningi są kluczem do sukcesu.")
                
                # Opcja ponownej analizy
                #st.markdown("---")
                if st.button("🔄 Analizuj inne dane"):
                    st.rerun()

if __name__ == "__main__":
    main()
