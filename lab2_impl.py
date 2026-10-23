from __future__ import annotations

import csv
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "wyniki"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import spacy
import textdistance
from nltk.corpus import wordnet
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from pystempel import Stemmer

DATA = ROOT / "data"
nltk.data.path.insert(0, str(DATA / "nltk_data"))
PL_WORD = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+")
EN_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT = re.compile(r"(?<=[.!?…])\s+")
notes: list[str] = []


def say(message=""):
    print(message)
    notes.append(str(message))


def heading(label):
    say(f"\n## {label}")


def csvfile(name, headers, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(headers)
        writer.writerows(rows)


def grams(words, n):
    return Counter(zip(*(words[i:] for i in range(n))))


def top(counter):
    return ", ".join(f"{' '.join(k) if isinstance(k, tuple) else k} ({v})"
                     for k, v in counter.most_common(10))


def histogram(counter, title, filename):
    items = list(reversed(counter.most_common(10)))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh([" ".join(k) for k, _ in items], [v for _, v in items])
    ax.set(xlabel="Liczba wystąpień", title=title)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def pos(tag):
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN,
            "R": wordnet.ADV}.get(tag[0], wordnet.NOUN)


def edit_path(a, b):
    d = [[0] * (len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): d[i][0] = i
    for j in range(len(b)+1): d[0][j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1,
                          d[i-1][j-1]+(a[i-1] != b[j-1]))
    i, j, ops = len(a), len(b), []
    while i or j:
        if i and j and d[i][j] == d[i-1][j-1]+(a[i-1] != b[j-1]):
            if a[i-1] != b[j-1]: ops.append(f"zamiana {a[i-1]!r} → {b[j-1]!r} na pozycji {i}")
            i -= 1; j -= 1
        elif j and d[i][j] == d[i][j-1]+1:
            ops.append(f"wstawienie {b[j-1]!r} po pozycji {i}"); j -= 1
        else:
            ops.append(f"usunięcie {a[i-1]!r} z pozycji {i}"); i -= 1
    return d[-1][-1], list(reversed(ops))


def main():
    raw_pl = (DATA / "pan-tadeusz.txt").read_text(encoding="utf-8-sig")
    raw_en = (DATA / "hamlet.txt").read_text(encoding="utf-8-sig")
    pl = "Litwo! Ojczyzno moja!" + raw_pl.split("Litwo! Ojczyzno moja!", 1)[1].split("\n-----", 1)[0]
    en = raw_en.split("*** START OF THE PROJECT GUTENBERG EBOOK HAMLET ***", 1)[1]
    en = en.split("*** END OF THE PROJECT GUTENBERG EBOOK HAMLET ***", 1)[0]
    say("LAB2 wyniki")
    say("PL: Pan Tadeusz bez nagłówka i stopki; EN: Hamlet bez nagłówka i licencji.")

    heading("A - n-gramy w tekście polskim")
    sentences = [s.strip() for s in SENT.split(pl) if s.strip()]
    cleaned = re.sub(r"\s+", " ", re.sub(r"[^A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż\s]", " ", pl).lower()).strip()
    words = PL_WORD.findall(cleaned)
    (OUT / "A_tekst_oczyszczony.txt").write_text(cleaned + "\n", encoding="utf-8")
    csvfile("A_zdania.csv", ["nr", "zdanie"],
            ((i, re.sub(r"\s+", " ", sentence)) for i, sentence in enumerate(sentences, 1)))
    csvfile("A_tokeny.csv", ["nr", "token"], enumerate(words, 1))
    original_grams = {n: grams(words, n) for n in (1, 2, 3)}
    say(f"A1–A3: {len(pl)} znaków, {len(sentences)} zdań, {len(words)} wyrazów; tekst oczyszczony: {len(cleaned)} znaków.")
    for n, name in ((1, "unigramy"), (2, "bigramy"), (3, "trigramy")):
        c = original_grams[n]
        csvfile(f"A_{name}.csv", ["ngram", "liczba"], [(" ".join(k), v) for k, v in c.most_common()])
        say(f"A4–A5: {name}: {sum(c.values())} wystąpień, {len(c)} typów; top 10: {top(c)}")
    histogram(original_grams[2], "A: 10 najczęstszych bigramów", "A_bigramy.png")
    histogram(original_grams[3], "A: 10 najczęstszych trigramów", "A_trigramy.png")
    say("A6–A7: histogramy A_bigramy.png i A_trigramy.png. Bigramy dominujące są pokazane powyżej;")
    say("trigramy wskazują powtarzane frazy, lecz nie dowodzą unikalnego stylu autora. Te frazy mogły po prostu być popularne w tamtym czasie dla każdego autora, trzebaby porównać z innymi autorami i ksiązkami.")
    say("Unigramy są mocniej skupione na częstych wyrazach, kilka pojedynczych wyrazów występuje bardzo często, a większość rzadko; bigramy mają więcej typów i kombinacji i dłuższy ogon rzadkich kombinacji, tez konkretne pary się rzadzeiej powtarzają, mamy 18760 róznych unigramów, a aż 58785 bigramów.")

    heading("B angielski stemming i lematyzacja")
    quotes = ["Who's there?", "Long live the king!", "I have seen nothing.",
              "The graves stood tenantless, and the sheeted dead Did squeak and gibber in the Roman streets.",
              "I am sick at heart."]
    normalized_en = re.sub(r"\s+", " ", en).lower()
    assert all(s.lower() in normalized_en for s in quotes)
    porter, lancaster, wordnet_lemma = PorterStemmer(), LancasterStemmer(), WordNetLemmatizer()
    b_rows = []
    for number, quote in enumerate(quotes, 1):
        for token, tag in nltk.pos_tag(EN_WORD.findall(quote)):
            b_rows.append((number, quote, token, porter.stem(token.lower()),
                           wordnet_lemma.lemmatize(token.lower(), pos(tag))))
    csvfile("B_token_rdzen_lemat.csv", ["zdanie", "tekst", "token", "Porter", "WordNet_lemat"], b_rows)
    say(f"B1–B4: 5 cytatów; pełna tabela B_token_rdzen_lemat.csv. Przykłady różnic: {[(r[2], r[3], r[4]) for r in b_rows if r[3] != r[4]][:8]}")
    say("Stemming mechanicznie skraca i może zwrócić nieistniejący wyraz; lematyzacja próbuje znaleźć formę słownikową z uwzględnieniem części mowy.")
    say("WordNet może błędnie rozstrzygnąć wieloznaczność, np. graves → graf; to ograniczenie metody w tym kontekście(WordNet znalazł więcej niż jedną możliwą formę podstawową dla graves)")

    heading("C odległość Levenshteina")
    x, y = "Ala ma kota", "Ala posiada kota"
    distance, operations = edit_path(x, y)
    assert distance == textdistance.levenshtein.distance(x, y) == len(operations)
    say(f"C1–C3: {x!r} → {y!r}: {distance} edycji. Jedna optymalna ścieżka: {'; '.join(operations)}.")

    heading("D porównanie stemmerów")
    en_vocab = sorted(set(EN_WORD.findall(en.lower())))
    differences = [(w, porter.stem(w), lancaster.stem(w)) for w in en_vocab
                   if porter.stem(w) != lancaster.stem(w)]
    csvfile("D_rozne_stemmery.csv", ["wyraz", "Porter", "Lancaster"], differences)
    sample_words = ["being", "graves", "nature", "streets", "wonder"]
    sample = [r for r in differences if r[0] in sample_words]
    assert len(sample) == 5
    say(f"D1–D2: różne rdzenie dla {len(differences)} typów; pięć przykładów: {sample}.")
    say("Porter zachowuje czytelniejszy rdzeń np. nature → natur (Lancaster: nat) i wonder → wonder (Lancaster: wond). Lancaster jest agresywny - 100 reguł dopóki kazda nie będzie pasowac")
    assert {"man", "manner", "see", "seen"}.issubset(en_vocab)
    say(f"D3: nadmierne obcinanie Lancastera: man i manner → {lancaster.stem('man')}; różne znaczenia dostają ten sam rdzeń.")
    say(f"D3: niedostateczne zbieganie Portera: see → {porter.stem('see')}, seen → {porter.stem('seen')}; to formy jednego czasownika.")

    heading("E polskie formy")
    nlp = spacy.load("pl_core_news_sm")
    stem_pl = Stemmer.default()
    fragment = pl[:4500]
    doc = nlp(fragment)
    snow_en = SnowballStemmer("english")
    e_rows = [(t.text, stem_pl(t.text.lower()) or "", t.lemma_.lower(), snow_en.stem(t.text.lower()))
              for t in doc if t.is_alpha]
    csvfile("E_polski_token_rdzen_lemat.csv",
            ["token", "polski_stemming_Stempel", "polska_lematyzacja_spaCy", "angielski_stemming_Snowball"],
            e_rows)
    say(f"E1: fragment 4500 znaków, {len(e_rows)} wyrazów; tabela E_polski_token_rdzen_lemat.csv.")
    say(f"E2: pięć polskich słów po angielskim Snowball: {list(dict.fromkeys((r[0], r[3]) for r in e_rows))[:5]}.")
    changed_by_english = list(dict.fromkeys((r[0],r[3]) for r in e_rows if r[0].lower()!=r[3]))
    say(f"E2: pięć sztucznych zmian przez angielski stemmer: {changed_by_english[:5]}.")
    say(f"E3: języki NLTK SnowballStemmer: {', '.join(SnowballStemmer.languages)}.")
    say("Polskiego brak na liście; angielski algorytm nie rozpoznaje polskiej fleksji.")
    say("E: np. jesteś → być w lematyzacji spaCy, podczas gdy Stempel daje y; agresywny rdzeń bywa trudny do interpretacji.")

    heading("F miary podobieństwa")
    pairs = [("Ala ma kota", "Ala posiada kota"), ("kot je bulke", "kto je bułke"),
             ("zolw", "żółw"), ("Warszawa", "Warzawa"), ("dom", "tom")]
    f_rows = []
    for a, b in pairs:
        lev = textdistance.levenshtein.distance(a, b)
        dam = textdistance.damerau_levenshtein.distance(a, b)
        maximum = max(len(a), len(b))
        f_rows.append((a, b, lev, round(1-lev/maximum, 4), dam, round(1-dam/maximum, 4),
                       round(textdistance.jaro_winkler.similarity(a,b), 4),
                       round(textdistance.Jaccard(qval=2).similarity(a,b), 4)))
    csvfile("F_miary.csv", ["tekst1", "tekst2", "Levenshtein_odleglosc", "Levenshtein_podobienstwo",
                             "Damerau_odleglosc", "Damerau_podobienstwo", "Jaro_Winkler", "Jaccard_bigramy"], f_rows)
    widest = max(f_rows, key=lambda r:max(r[3],r[5],r[6],r[7])-min(r[3],r[5],r[6],r[7]))
    say("F1–F3: Levenshtein liczy zamiany, wstawienia i usunięcia; Damerau uwzglednia przestawienie dwoch sąsiednich znaków (kot kto).")
    say("Jaro-Winkler mierzy dopasowanie znaków i premiuje wspólny początek; Jaccard porównuje zbiory bigramów znakowych.")
    say("Pełna tabela F_miary.csv; podobieństwa są w skali 0–1, Levenshtein i Damerau dodatkowo podano jako odległości.")
    say(f"F4: największa rozpiętość: {widest[0]!r}/{widest[1]!r}, podobieństwa {widest[3]}, {widest[5]}, {widest[6]}, {widest[7]} — transpozycja kontra zbiory bigramów.")

    (OUT / "wyniki_lab2.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
