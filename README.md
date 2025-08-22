# Aplikacja Streamlit – Predykcja czasu półmaratonu

Wprowadź w czacie imię, płeć (opcjonalnie), wiek lub rok urodzenia oraz czas na 5 km.
Aplikacja wyciągnie brakujące pola przy pomocy GPT-4 i przewidzi czas półmaratonu używając modelu `model/app_zad_dom_9_regressor.pkl`.

## Uruchomienie lokalne

1. (Opcjonalnie) aktywuj środowisko conda/venv.
2. Zainstaluj zależności:

```bash
pip install -r modul_9/ZADANIEDOMOWE9/requirements.txt
```

3. Uruchom aplikację:

```bash
streamlit run modul_9/ZADANIEDOMOWE9/app.py
```

Aby użyć ekstrakcji GPT ustaw `OPENAI_API_KEY` w zmiennych środowiskowych lub w pliku `.env` w katalogu `modul_9/ZADANIEDOMOWE9/`.  

