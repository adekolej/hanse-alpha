# Hanse Alpha

Aplikacja webowa do analizy akcji, sektorów i łańcuchów dostaw — zbudowana na
**Streamlit** + **yfinance**, z danymi GPW ze **stooq.pl**.

🔗 **Live:** _wstaw tu link po wdrożeniu_ (np. `https://hanse-alpha.streamlit.app`)

---

## Funkcje

| Zakładka | Opis | Źródło danych |
|----------|------|---------------|
| **Stocks** | Wykres świecowy + wskaźniki (SMA/EMA/Bollinger/RSI/MACD), fundamenty, news, kalendarz, cele cenowe, wynik kwartalny | yfinance (Yahoo) |
| **Watchlist** | Lista obserwowanych spółek z notowaniami na żywo | yfinance |
| **Sectors** | Przegląd sektorów Yahoo (overview, top companies, raporty) | yfinance |
| **GPW (Stooq)** | Notowania Giełdy Papierów Wartościowych (indeksy + spółki) oraz wykresy historyczne | stooq.pl |
| **Supply Chain** | Mapa kluczowych dostawców (Apple, Tesla, Boeing, Lockheed Martin) — diagram Sankey + notowania dostawców na żywo | skurowana baza + yfinance |

### Uwagi o danych
- **Relacje w łańcuchu dostaw są skurowane** (z publicznych list dostawców i raportów) — to ilustracyjna mapa Tier-1, nie pełny BOM. Dane rynkowe dostawców są na żywo.
- **Historia GPW** ze stooq.pl wymaga `apikey` (endpoint masowego pobierania jest za captchą). Notowania na żywo działają bez klucza.

---

## Uruchomienie lokalne

```bash
# 1. Wirtualne środowisko (Python 3.11+)
python3 -m venv venv
source venv/bin/activate          # macOS/Linux

# 2. Zależności
pip install -r requirements.txt

# 3. Start
streamlit run app.py
```

Aplikacja ruszy na <http://localhost:8501>.

> **Ważne:** po edycji `dicts.py` zrób **pełny restart** serwera (Ctrl+C i ponownie
> `streamlit run app.py`). Sam przycisk „Rerun" w przeglądarce nie przeładowuje
> zaimportowanych modułów — Streamlit trzyma je w pamięci.

---

## Wdrożenie na Streamlit Community Cloud (darmowe, publiczny link)

To najlepszy sposób, żeby wysłać komuś działający link — apka stoi w chmurze,
**Twój komputer nie musi być włączony**.

1. Wejdź na **<https://share.streamlit.io>** i zaloguj się przez GitHub.
2. Kliknij **Create app** → **Deploy a public app from GitHub**.
3. Ustaw:
   - **Repository:** `adekolej/hanse-alpha`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. _(opcjonalnie)_ **Advanced settings → Secrets** — jeśli chcesz wykresy
   historyczne GPW, dodaj klucz stooq:
   ```toml
   stooq_apikey = "TWÓJ_KLUCZ"
   ```
   Klucz zdobędziesz na `https://stooq.pl/q/d/?s=pkn&get_apikey`.
5. **Deploy**. Po chwili dostaniesz publiczny adres typu
   `https://hanse-alpha.streamlit.app` — wklej go na górze tego README.

🔄 **Auto-deploy:** każdy `git push` na `main` automatycznie aktualizuje wdrożoną
aplikację. Jeśli apka jest już na Streamlit Cloud, samo wypchnięcie zmian ją odświeży.

---

## Udostępnianie — co działa, a co nie

| Link | Kto otworzy |
|------|-------------|
| `https://...streamlit.app` | **Każdy w internecie** ✅ (zalecane) |
| `http://localhost:8501` | tylko Ty, na swoim komputerze |
| `http://<IP-lokalny>:8501` | tylko osoby w tej samej sieci WiFi/LAN |
| `http://<IP-publiczny>:8501` | zwykle **nie zadziała** — wymaga port forwardingu, zapory i braku CGNAT |

---

## Stack
Streamlit · yfinance · pandas · numpy · plotly · stooq.pl (CSV API)
