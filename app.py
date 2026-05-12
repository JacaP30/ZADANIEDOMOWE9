import streamlit as st
import openai
import pandas as pd
import os
import json
from dotenv import load_dotenv
from datetime import datetime
import base64

try:
    from langfuse.decorators import observe
    from langfuse.openai import OpenAI as LangfuseOpenAI
    from langfuse import Langfuse  # Dodajemy dla inicjalizacji klienta
    LANGFUSE_AVAILABLE = True
    USE_LANGFUSE_OPENAI = True
except ImportError:
    try:
        from langfuse import Langfuse
        LANGFUSE_AVAILABLE = True
        USE_LANGFUSE_OPENAI = False
    except ImportError:
        LANGFUSE_AVAILABLE = False
        USE_LANGFUSE_OPENAI = False

if "observe" not in globals():

    def observe(**_kwargs):
        def _decorator(fn):
            return fn

        return _decorator



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
set_bg("images/background.png")


# Załaduj zmienne środowiskowe
load_dotenv()


def _read_streamlit_secret_openai_key() -> str | None:
    try:
        s = st.secrets.get("OPENAI_API_KEY", "")  # type: ignore[attr-defined]
        if s:
            return str(s).strip() or None
    except Exception:
        pass
    return None


def get_env_openai_key() -> str | None:
    k = (os.getenv("OPENAI_API_KEY") or "").strip()
    return k or None


def validate_openai_api_key(api_key: str) -> tuple[bool, str]:
    """Sprawdza format i wywołuje lekkie żądanie do API OpenAI (lista modeli)."""
    key = (api_key or "").strip()
    if not key:
        return False, "Klucz API nie może być pusty."
    if not key.startswith("sk-"):
        return False, 'Klucz OpenAI powinien zaczynać się od prefiksu "sk-".'
    try:
        from openai import OpenAI as OpenAIKeyCheck

        client = OpenAIKeyCheck(api_key=key)
        client.models.list()
        return True, ""
    except Exception as e:
        msg = str(e).strip() or repr(e)
        return False, f"Klucz nie został zaakceptowany przez OpenAI: {msg}"


def build_openai_client(api_key: str):
    """Tworzy klienta chat completions (wrapper Langfuse lub standardowy OpenAI)."""
    if USE_LANGFUSE_OPENAI:
        return LangfuseOpenAI(api_key=api_key)  # type: ignore[misc]
    openai.api_key = api_key
    return openai


def get_openai_client_from_session():
    """Klient OpenAI na podstawie zwalidowanego klucza w sesji (None w trybie demo)."""
    if st.session_state.get("demo_mode"):
        return None
    key = st.session_state.get("openai_api_key")
    if not key:
        return None
    cache_key = "_openai_client_for_key"
    if st.session_state.get(cache_key) == key and "_openai_client_obj" in st.session_state:
        return st.session_state["_openai_client_obj"]
    client = build_openai_client(key)
    st.session_state[cache_key] = key
    st.session_state["_openai_client_obj"] = client
    return client


def render_api_setup_gate():
    """
    Jeśli brak trybu demo i brak poprawnego klucza — pokaż ekran startowy.
    Klucz z .env / secrets walidujemy przy pierwszym wejściu.
    """
    if st.session_state.get("demo_mode"):
        return
    if st.session_state.get("openai_api_key"):
        return

    env_key = get_env_openai_key()
    secret_key = _read_streamlit_secret_openai_key()
    auto_key = env_key or secret_key
    if auto_key:
        if st.session_state.get("_env_key_validated") == auto_key:
            st.session_state["openai_api_key"] = auto_key
            return
        if st.session_state.get("_env_key_invalid") == auto_key:
            st.info(
                "Klucz z pliku `.env` / zmiennych środowiska lub Streamlit Secrets "
                "wcześniej nie przeszedł walidacji — wprowadź poprawny klucz poniżej lub wybierz tryb demo."
            )
        else:
            ok, err = validate_openai_api_key(auto_key)
            if ok:
                st.session_state["openai_api_key"] = auto_key
                st.session_state["_env_key_validated"] = auto_key
                st.session_state.pop("_env_key_invalid", None)
                return
            st.error(
                "Zmienna OPENAI_API_KEY (lub klucz w Streamlit Secrets) jest ustawiona, "
                f"ale walidacja nie powiodła się: {err}"
            )
            st.session_state["_env_key_invalid"] = auto_key

    st.title("Konfiguracja OpenAI")
    st.markdown(
        "Aby korzystać z analizy tekstu przez AI, potrzebny jest klucz API OpenAI. "
        "Możesz też włączyć **tryb demo** i obejrzeć interfejs bez wywołań API."
    )
    choice = st.radio(
        "Wybierz opcję",
        ("Wprowadzę klucz API OpenAI", "Tryb demo (bez API)"),
        horizontal=True,
    )

    if choice.startswith("Tryb demo"):
        if st.button("Uruchom w trybie demo", type="primary", use_container_width=True):
            st.session_state["demo_mode"] = True
            st.session_state["openai_api_key"] = None
            st.rerun()
        st.stop()

    key_in = st.text_input(
        "Klucz API OpenAI",
        type="password",
        help="Klucz zaczyna się zwykle od sk-; nie jest zapisywany na dysku, tylko w pamięci sesji przeglądarki.",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        validate_clicked = st.button("Sprawdź klucz i kontynuuj", type="primary", use_container_width=True)
    with col_b:
        if st.button("Tryb demo", use_container_width=True):
            st.session_state["demo_mode"] = True
            st.session_state["openai_api_key"] = None
            st.rerun()

    if validate_clicked:
        ok, err = validate_openai_api_key(key_in)
        if ok:
            st.session_state["openai_api_key"] = key_in.strip()
            st.session_state["demo_mode"] = False
            st.session_state.pop("_openai_client_obj", None)
            st.session_state.pop("_openai_client_for_key", None)
            st.session_state.pop("_env_key_invalid", None)
            st.success("Klucz poprawny. Ładowanie aplikacji…")
            st.rerun()
        else:
            st.error(err)

    st.stop()

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
            langfuse_client = Langfuse( # type: ignore
                secret_key=secret_key,
                public_key=public_key,
                host=host
            )
            print("✅ Langfuse initialized successfully")
            
            # Wyświetl dostępne metody do debugowania
            available_methods = [method for method in dir(langfuse_client) if not method.startswith('_') and callable(getattr(langfuse_client, method))]
            print(f"🔧 Available methods: {', '.join(available_methods[:10])}")  # Pierwsze 10
            
            # Sprawdź wersję jeśli dostępna
            if hasattr(langfuse_client, '__version__'):
                print(f"📦 Langfuse version: {langfuse_client.__version__}") # type: ignore
                
        except Exception as e:
            print(f"⚠️ Langfuse initialization failed: {e}")
            langfuse_client = None
    else:
        print("⚠️ Langfuse keys missing - skipping initialization")
else:
    print("⚠️ Langfuse not available - library not installed")


def log_to_langfuse(function_name, input_data, output_data, metadata=None):
    """Loguj wywołanie funkcji do Langfuse 2.51.4+"""
    if langfuse_client is None:
        return
    
    try:
        # Langfuse 2.51.4 ma metodę event() (nie create_event)
        event = langfuse_client.event(
            name=function_name,
            input=input_data,
            output=output_data,
            metadata={
                **(metadata or {}),
                "function": function_name,
                "model": "gpt-4" if "extract" in function_name or "infer" in function_name else "ml-model",
                "app": "half_marathon_predictor"
            }
        )
        
        print(f"✅ Logged to Langfuse: {function_name}")
        
        # Flush natychmiast
        if hasattr(langfuse_client, 'flush'):
            langfuse_client.flush()
        
        return event
        
    except Exception as e:
        print(f"❌ Langfuse logging error: {e}")
        print(f"Function: {function_name}")
        # Pokazuj dostępne metody dla debugowania
        available = [method for method in dir(langfuse_client) if not method.startswith('_') and callable(getattr(langfuse_client, method))]
        print(f"Available methods: {available[:10]}")
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

@observe(name="extract_user_data") # type: ignore
def extract_user_data(user_input):
    """Wyciągnij wszystkie dane użytkownika z tekstu używając AI"""
    openai_client = get_openai_client_from_session()
    if openai_client is None:
        st.error("Analiza AI jest niedostępna (brak klucza API lub tryb demo).")
        return None
    try:
        response = openai_client.chat.completions.create(
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
            return data
        except json.JSONDecodeError:
            # Jeśli nie udało się sparsować JSON, zwróć None
            return None
            
    except Exception as e:
        st.error(f"Błąd podczas komunikacji z AI: {e}")
        return None

@observe(name="infer_gender_from_name")  # type: ignore
def infer_gender_from_name(name):
    """Wywnioskuj płeć na podstawie imienia używając AI"""
    openai_client = get_openai_client_from_session()
    if openai_client is None:
        st.error("Rozpoznawanie płci przez AI jest niedostępne (brak klucza API lub tryb demo).")
        return None
    try:
        response = openai_client.chat.completions.create(
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
    if "demo_mode" not in st.session_state:
        st.session_state.demo_mode = False

    render_api_setup_gate()

    with st.sidebar:
        mode_label = "Tryb demo (bez API)" if st.session_state.get("demo_mode") else "Tryb z kluczem OpenAI"
        st.caption(f"Aktywny: **{mode_label}**")
        if st.button("Zmień klucz API / tryb startowy", use_container_width=True):
            keys_to_clear = (
                "demo_mode",
                "openai_api_key",
                "_env_key_validated",
                "_env_key_invalid",
                "_openai_client_obj",
                "_openai_client_for_key",
            )
            for k in keys_to_clear:
                st.session_state.pop(k, None)
            st.session_state.demo_mode = False
            st.rerun()

    st.title("Predyktor Czasu Półmaratonu")
    # st.markdown("---")
    
    # Załaduj model
    model = load_model()
    if model is None:
        st.stop()

    if st.session_state.get("demo_mode"):
        st.warning(
            "Jesteś w **trybie demo**: możesz przeglądać opis i formularz, "
            "ale analiza tekstu przez AI i predykcja na tej podstawie są wyłączone. "
            "Wybierz „Zmień klucz API / tryb startowy” na pasku bocznym i wprowadź klucz OpenAI, aby w pełni korzystać z aplikacji."
        )
    
    st.markdown("### Opowiedz o sobie żeby uzyskać prawdopodobny czas ukończenia pułmaratonu:")
    
    # Instrukcje dla użytkownika
    st.info("""
    📝 **Podaj następujące informacje w dowolnej formie:**
    - **Imię**,  **Wiek**,  **Czas na 5km**,  **Płeć** (jeśli chcesz)
    
    💡 **Przykłady:**
    - "Nazywam się Kasia, urodziłam się w 1990 roku, biegam 5 km w 26.5 minuty"
    - "Jestem Anna, mam 28 lat i biegam 5 km w 24 minuty"
    - "Marek, 35 lat, czas na 5km: 22:45"
    - Możesz też po prostu "Janek 75 25"  😉
    """)
    
    # Formularz dla użytkownika
    demo = st.session_state.get("demo_mode", False)
    with st.form("user_data_form"):
        user_input = st.text_area(
            "## **Wpisz informacje o sobie:**",
            height=100,
            placeholder="Napisz coś o sobie... ",
            help="Podaj swoje dane w dowolnej formie - AI wyciągnie potrzebne informacje"
            if not demo
            else "W trybie demo pole jest tylko do podglądu — bez klucza API analiza nie zostanie uruchomiona.",
            disabled=demo,
        )

        submitted = st.form_submit_button(
            "🔍 Analizuj i przewiduj czas półmaratonu",
            use_container_width=True,
            disabled=demo,
        )

    if submitted:
        if demo:
            st.info("W trybie demo przycisk analizy jest wyłączony. Dodaj klucz OpenAI w konfiguracji startowej.")
            st.stop()
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
        
        # Sprawdź czy mamy wszystkie potrzebne dan
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
        with st.spinner("Przewidywanie czasu półmaratonu..."):
            
            # Konwersja czasu 5km na sekundy
            time_5k_seconds = time_5k * 60
            
            # Predykcja
            predicted_time = predict_half_marathon_time(model, gender, age, time_5k_seconds)
            
            if predicted_time is not None:
                # Główny wynik
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
                        text-shadow: 3px 2px 6px rgba(55, 255, 55, 0.9);
                        font-weight: bold;
                        letter-spacing: 2px;
                    ">{predicted_time_formatted}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Dodatkowe informacje
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
    # Konfiguracja dla Digital Ocean
    port = int(os.environ.get("PORT", 8501))
    main()
