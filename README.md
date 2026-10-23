# Lab2

## Pliki

- `data/pan-tadeusz.txt` — polski tekst z Wolnych Lektur, analizowany w A i E.
- `data/hamlet.txt` — angielski tekst z Project Gutenberg, analizowany w B i D.
- `lab2.py` — uruchamia analizę; `lab2_impl.py` — kod części A–F.
- `wyniki/A_tekst_oczyszczony.txt`, `A_zdania.csv`, `A_tokeny.csv` — kolejne etapy przetwarzania w A.
- `wyniki/A_*.csv`, `A_*.png` — częstości n-gramów i histogramy.
- `wyniki/B_token_rdzen_lemat.csv`, `D_rozne_stemmery.csv`, `E_polski_token_rdzen_lemat.csv`, `F_miary.csv` — tabele B, D, E i F.
- `wyniki/wyniki_lab2.txt` — krótkie wyniki, wyjaśnienia i wnioski A–F, w tym część C.
- `Sprawozdanie_Lab2_4_0.pdf` — gotowe sprawozdanie PDF.

Pliki CSV mają separator `;` i kodowanie UTF-8 z BOM, aby poprawnie otwierały się w Excelu.

Źródła: [Pan Tadeusz](https://wolnelektury.pl/media/book/txt/pan-tadeusz.txt) i [Hamlet](https://www.gutenberg.org/cache/epub/27761/pg27761.txt).

## Uruchomienie na Windows

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m spacy download pl_core_news_sm
.venv\Scripts\python.exe -c "import nltk; [nltk.download(x, download_dir='data/nltk_data') for x in ['wordnet','omw-1.4','averaged_perceptron_tagger_eng']]"
.venv\Scripts\python.exe lab2.py
```
