import React, { useState } from 'react';
import './App.css';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 25000);

    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || data.error) {
        setError(data.error || 'Błąd serwera');
      } else {
        setResult(data);
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        setError('Przekroczono limit czasu. Spróbuj ponownie.');
      } else {
        setError('Nie udało się połączyć z backendem');
      }
    } finally {
      clearTimeout(timeoutId);
      setLoading(false);
    }
  };

  return (
    <div className="App" style={{ maxWidth: '100%', margin: '0 auto', padding: 24 }}>
      <h2>Predyktor Czasu Półmaratonu</h2>
      <p>Wpisz w dowolnej formie: imię, wiek/rok urodzenia, płeć (opcjonalnie) i czas na 5 km.</p>

      <form onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={5}
          style={{ width: '50%', padding: 12, fontSize: 16 }} 
          placeholder={`Np. Mam na imię Kasia, mam 32 lata, jestem kobietą, 5 km w 28 minut

Jestem Jan, 28 lat, 5 km w 18 min

Lub po prostu: Anna 30 15 :)`}
        />
        <div style={{ marginTop: 12 }}>
          <button type="submit" disabled={loading || !text.trim()}>
            {loading ? 'Analizuję...' : 'Analizuj i przewiduj'}
          </button>
        </div>
      </form>

      {error && (
        <div style={{ marginTop: 16, color: 'crimson' }}>
          <strong>Błąd:</strong> {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: 24, textAlign: 'center', width: '70%', margin: '24px auto 0 auto', backgroundColor: 'rgba(255, 255, 255, 0)', padding: '20px', borderRadius: '10px' }}>
          <h3 style={{ fontSize: '2em', color: '#141b4d', textShadow: '1px 1px 0 #59b3f3a7, -1px -1px 0 #59b3f3a7, 1px -1px 0 #59b3f3a7, -1px 1px 0 #59b3f3a7' }}>🔍 Dane wyciągnięte i wynik</h3>
          <div style={{ marginBottom: 16, fontSize: 16, color: '#141b4d', textShadow: '1px 1px 0 #9fd2f7a7, -1px -1px 0 #9fd2f7a7, 1px -1px 0 #9fd2f7a7, -1px 1px 0 #9fd2f7a7' }}>
            <strong>Imię:</strong> {result.name ?? '-'} | <strong>Wiek:</strong> {result.age ?? '-'}{result.birth_year ? ` (ur. ${result.birth_year})` : ''} | <strong>Płeć:</strong> {result.gender === 'M' ? 'Mężczyzna' : result.gender === 'K' ? 'Kobieta' : '-'} | <strong>Czas 5 km:</strong> {result.time_5k ? `${result.time_5k} min` : '-'}
          </div>

          <div style={{ marginTop: 16, padding: 16, border: '2px solid #4CAF50', borderRadius: 8, backgroundColor: 'rgba(255, 255, 255, 0.1)' }}>
            <h4 style={{ fontSize: '2em', color: '#141b4d', textShadow: '1px 1px 0 #59b3f3a7, -1px -1px 0 #59b3f3a7, 1px -1px 0 #59b3f3a7, -1px 1px 0 #59b3f3a7' }}>🏁 Przewidywany czas półmaratonu</h4>
            <div style={{ fontSize: 50, color: '#014005ff', fontWeight: 700, textShadow: '1px 1px 0 #68eda6ff, -1px -1px 0 #68eda6ff, 1px -1px 0 #68eda6ff, -1px 1px 0 #68eda6ff' }}>{result.predicted_time_formatted}</div>
            <div style={{ marginTop: 12, color: '#000000ff', textShadow: '1px 1px 0 #59b3f3a7, -1px -1px 0 #59b3f3a7, 1px -1px 0 #59b3f3a7, -1px 1px 0 #59b3f3a7' }}>
              ({Math.round(result.predicted_time_seconds)} s)
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
