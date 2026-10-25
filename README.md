# Lab2

## Pliki

- `data/pan-tadeusz.txt` — polski tekst z Wolnych Lektur, analizowany w A i E.
- `data/hamlet.txt` — angielski tekst z Project Gutenberg, analizowany w B i D.
- `lab2.py` — uruchamia analizę; `lab2_impl.py` — kod części A–I.
- `tokenizer_pl.json` — własny tokenizator BPE (5 000 tokenów) wytrenowany na „Panu Tadeuszu”.
- `wyniki/A_tekst_oczyszczony.txt`, `A_zdania.csv`, `A_tokeny.csv` — kolejne etapy przetwarzania w A.
- `wyniki/A_*.csv`, `A_*.png` — częstości n-gramów i histogramy.
- `wyniki/B_token_rdzen_lemat.csv`, `D_rozne_stemmery.csv`, `E_polski_token_rdzen_lemat.csv`, `F_miary.csv` — tabele B, D, E i F.
- `wyniki/G_*.csv`, `G_*.png` — n-gramy bez stop words, porównanie obu wariantów i nazwy własne.
- `wyniki/H_*.csv`, `H_*.png`, `tokenizer_pl_1000.json`, `tokenizer_pl_30000.json` — podział słów, porównanie tokenizatorów i wpływ rozmiaru słownika.
- `wyniki/I_porownanie.csv` — porównanie czterech sposobów przetwarzania tego samego fragmentu.
- `wyniki/wyniki_lab2.txt` — krótkie wyniki, wyjaśnienia i wnioski A–I.
- `sprawozdanie2.docx`, `Sprawozdanie_Lab2_5_0.pdf` — sprawozdanie (części A–I) w wersji edytowalnej i PDF.

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

Część H pobiera z Hugging Face wyłącznie pliki tokenizatorów (`gpt2`, `allegro/herbert-base-cased`, `speakleash/Bielik-1.5B-v3`). Repozytorium Bielika v3 ma ograniczony dostęp: trzeba zaakceptować warunki na jego stronie i zalogować się (`.venv\Scripts\hf.exe auth login`). Bez dostępu skrypt używa zastępczo tokenizatora Bielika v2.3 i oznacza to w wynikach.

`preshed` jest przypięty do 3.0.13, bo 3.0.14 nie wczytuje modelu `pl_core_news_sm` w spaCy 3.8.16.
