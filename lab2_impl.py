from __future__ import annotations

import csv
import importlib.resources
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "wyniki"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import spacy
import textdistance
from nltk.corpus import wordnet
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
# pystempel 2.0.0 importuje typy usunięte z importlib.resources w Pythonie 3.13
importlib.resources.Package = getattr(importlib.resources, "Package", str)
importlib.resources.Resource = getattr(importlib.resources, "Resource", str)
from pystempel import Stemmer
from stop_words import get_stop_words
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import AutoTokenizer

DATA = ROOT / "data"
nltk.data.path.insert(0, str(DATA / "nltk_data"))
PL_WORD = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+")
EN_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT = re.compile(r"(?<=[.!?…])\s+")
PROPER = re.compile(r"(?<=[a-ząćęłńóśźż,;] )[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+")
BPE_SIZES = (500, 1000, 2000, 5000, 10000, 20000, 30000)
BPE_ALPHABET = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ0123456789.,;:!?'\"()-—„”«»…")
PRETRAINED = (("gpt2", ("gpt2",)),
              ("HerBERT", ("allegro/herbert-base-cased",)),
              # repozytorium v3 jest bramkowane na HF; v2.3 tylko zastępczo, dopóki nie ma dostępu
              ("Bielik", ("speakleash/Bielik-1.5B-v3", "speakleash/Bielik-11B-v2.3-Instruct")))
MODERN_PL = ("Rząd zapowiedział, że od przyszłego roku wszystkie urzędy będą przyjmować wnioski przez internet. "
             "Nowa aplikacja pozwoli sprawdzić stan sprawy na smartfonie i otrzymać powiadomienie, gdy decyzja będzie gotowa. "
             "Eksperci zwracają uwagę, że cyfryzacja administracji wymaga szkoleń dla pracowników oraz zabezpieczenia danych osobowych. "
             "Pierwsze testy systemu przeprowadzono w kilku województwach, a użytkownicy ocenili go pozytywnie. "
             "Ministerstwo planuje także udostępnić otwarte dane o czasie załatwiania spraw.")
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
    ax.barh([k if isinstance(k, str) else " ".join(k) for k, _ in items], [v for _, v in items])
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


def flat(text):
    return re.sub(r"\s+", " ", text).strip()


def per_word(count, text):
    found = PL_WORD.findall(text)
    return round(count(" ".join(found)) / len(found), 2)


def method_variants(tokens, stop, tokenizer):
    return [("same wyrazy", [t.text.lower() for t in tokens]),
            ("wyrazy po lematyzacji (spaCy)", [t.lemma_.lower() for t in tokens]),
            ("wyrazy bez stop words", [t.text.lower() for t in tokens if t.text.lower() not in stop]),
            ("tokeny BPE 5000", tokenizer.encode(" ".join(t.text for t in tokens)).tokens)]


def train_bpe(lines, size):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Metaspace(), pre_tokenizers.Punctuation()])
    tokenizer.decoder = decoders.Metaspace()
    tokenizer.train_from_iterator(lines, trainers.BpeTrainer(
        vocab_size=size, special_tokens=["[UNK]"], initial_alphabet=BPE_ALPHABET, show_progress=False))
    return tokenizer


def pretrained(names):
    for name in names:
        try:
            return name, AutoTokenizer.from_pretrained(name)
        except OSError as error:
            say(f"H: brak dostępu do {name}: {str(error).splitlines()[0]}")
    raise RuntimeError(f"Nie udało się pobrać żadnego tokenizatora z {names}")


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
    pairs = [("Ala ma kota", "Ala posiada kota"), ("kot", "kto"),
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

    heading("G stop words i nazwy własne")
    stop = set(get_stop_words("polish"))
    content = [w for w in words if w not in stop]
    stop_grams = {n: grams(content, n) for n in (1, 2, 3)}
    removed = len(words) - len(content)
    say(f"G1: lista stop-words (polish): {len(stop)} wyrazów; usunięto {removed} z {len(words)} wyrazów "
        f"({removed / len(words):.1%}), zostało {len(content)}.")
    g_rows = []
    for n, name in ((1, "unigramy"), (2, "bigramy"), (3, "trigramy")):
        c = stop_grams[n]
        csvfile(f"G_{name}.csv", ["ngram", "liczba"], [(" ".join(k), v) for k, v in c.most_common()])
        say(f"G2: {name} bez stop words: {sum(c.values())} wystąpień, {len(c)} typów; top 10: {top(c)}")
        for variant, counter in (("ze stop words", original_grams[n]), ("bez stop words", c)):
            total = sum(counter.values())
            g_rows.append((name, variant, total, len(counter),
                           round(100 * sum(v for _, v in counter.most_common(10)) / total, 2), top(counter)))
    csvfile("G_porownanie.csv", ["ngramy", "wariant", "wystapienia", "typy", "udzial_top10_proc", "top10"], g_rows)
    histogram(stop_grams[2], "G: 10 najczęstszych bigramów bez stop words", "G_bigramy.png")
    histogram(stop_grams[3], "G: 10 najczęstszych trigramów bez stop words", "G_trigramy.png")
    say("G2–G3: histogramy G_bigramy.png i G_trigramy.png; zestawienie obu wariantów w G_porownanie.csv.")
    say(f"G3: usunięcie stop words zabiera {removed / len(words):.0%} wyrazów, ale tylko "
        f"{len(original_grams[1]) - len(stop_grams[1])} typów — lista to głównie bardzo częste wyrazy funkcyjne (także pan, pani).")
    say("G3: różnice są istotne. Ze stop words top bigramy to połączenia funkcyjne (się w, i z, się z), które opisują składnię i styl;")
    say("bez stop words wychodzą postacie i sytuacje (rzekł sędzia, ksiądz robak, jenerał dąbrowski, rzekł hrabia), czyli treść.")
    say("G3: wariant bez stop words lepiej pokazuje treść, wariant pełny — styl (spójniki, zaimek się, zwroty dialogowe).")
    say("G3: koszt: powstają pary, które w tekście nie sąsiadowały (idąc kłaniał ← idąc kłaniał się), a trigramy prawie się nie")
    say("powtarzają — top trigramy to w większości kawałki jednego powtarzanego dwuwiersza (Z wieku mu i z urzędu ten zaszczyt należy, /")
    say("Idąc kłaniał się...). Bigram dal j to artefakt czyszczenia z części A: dawna pisownia daléj traci literę é.")

    case_counts = Counter(PL_WORD.findall(pl))
    candidates = Counter(PROPER.findall(pl))
    names = Counter({w: c for w, c in candidates.items() if c > case_counts[w.lower()]})
    csvfile("G_nazwy_wlasne.csv", ["wyraz", "wielka_litera_w_srodku_zdania", "mala_litera", "nazwa_wlasna"],
            [(w, c, case_counts[w.lower()], "tak" if w in names else "nie") for w, c in candidates.most_common()])
    histogram(names, "G: 10 najczęstszych nazw własnych", "G_nazwy_wlasne.png")
    rejected = [(w, c, case_counts[w.lower()]) for w, c in candidates.most_common() if w not in names][:8]
    say(f"G4: wyrażenie regularne {PROPER.pattern} — wielka litera tylko po małej literze, przecinku lub średniku i spacji,")
    say("więc pomija początki zdań, wersów, wypowiedzi po myślniku i cytatów.")
    say(f"G4: {sum(candidates.values())} trafień, {len(candidates)} różnych form; po odrzuceniu form, które częściej "
        f"występują małą literą, zostało {len(names)} form ({sum(names.values())} wystąpień).")
    say(f"G4: najczęstsze odrzucone (forma, wielką, małą): {rejected}.")
    say(f"G5: 10 najczęstszych nazw własnych: {top(names)}; wykres G_nazwy_wlasne.png.")
    say("G5: ograniczenia: odmiana rozbija jedną nazwę na kilka form (Tadeusz/Tadeusza, Sędzia/Sędziego); tytuły pisane wielką")
    say("literą (Sędzia, Hrabia, Wojski) pełnią w utworze rolę imion postaci; nazwy na początku zdania lub wersu są pomijane,")
    say("a filtr małych liter odrzuca formy grzecznościowe i podniosłe (Pan, Waść, Jego, Ojczyzny).")

    heading("H własny tokenizator BPE")
    lines = [line.strip() for line in pl.splitlines() if line.strip()]
    bpe = {size: train_bpe(lines, size) for size in BPE_SIZES}
    own = bpe[5000]
    own.save(str(ROOT / "tokenizer_pl.json"))
    for size in (1000, 30000):
        bpe[size].save(str(OUT / f"tokenizer_pl_{size}.json"))
    say(f"H1–H2: BPE (tokenizers) uczony na Panu Tadeuszu, {len(lines)} wersów; zadany → rzeczywisty rozmiar słownika: "
        + ", ".join(f"{s} → {bpe[s].get_vocab_size()}" for s in (1000, 5000, 30000))
        + "; wariant 5000 zapisany jako tokenizer_pl.json, pozostałe w wyniki/.")
    h_rows = [[word] + [f"{' | '.join(t)} ({len(t)})" for t in (bpe[s].encode(word).tokens for s in (1000, 5000, 30000))]
              for word in ("nieprzetłumaczalny", "hiperparametryzacja", "kotkowi", "Soplicowo")]
    csvfile("H_podzial_slow.csv", ["slowo", "BPE_1000", "BPE_5000", "BPE_30000"], h_rows)
    for row in h_rows:
        say(f"H3: {row[0]}: 1000: {row[1]}; 5000: {row[2]}; 30000: {row[3]}")

    parallel = [("Być albo nie być, oto jest pytanie.", "To be, or not to be, that is the question."),
                ("Słabości, twe imię jest kobieta!", "Frailty, thy name is woman!"),
                ("Nie bądź ani dłużnikiem, ani wierzycielem.", "Neither a borrower nor a lender be."),
                ("Więcej jest rzeczy na niebie i ziemi, Horacjo, niż się śniło waszym filozofom.",
                 "There are more things in heaven and earth, Horatio, than are dreamt of in your philosophy."),
                ("Reszta jest milczeniem.", "The rest is silence.")]
    assert all(en_s.lower().rstrip(".!") in normalized_en for _, en_s in parallel)
    counters = {"BPE_5000": lambda s: len(own.encode(s).tokens)}
    for label, repos in PRETRAINED:
        repo, tok = pretrained(repos)
        if repo != repos[0]:
            label = f"{label} v2.3 (zastępczo)"
        counters[label] = lambda s, tok=tok: len(tok.encode(s, add_special_tokens=False))
        say(f"H4: {label} = {repo}")
    labels = list(counters)
    h4_rows = [(lang, s, len((PL_WORD if lang == "PL" else EN_WORD).findall(s)), *(count(s) for count in counters.values()))
               for lang, s in [("PL", p) for p, _ in parallel] + [("EN", e) for _, e in parallel]]
    csvfile("H_liczba_tokenow.csv", ["jezyk", "zdanie", "wyrazy", *labels], h4_rows)
    mean = {lang: [sum(r[3 + i] for r in h4_rows if r[0] == lang) / len(parallel) for i in range(len(labels))]
            for lang in ("PL", "EN")}
    summary = [(label, round(mean["PL"][i], 2), round(mean["EN"][i], 2), round(mean["PL"][i] / mean["EN"][i], 2),
                per_word(counters[label], pl), per_word(counters[label], MODERN_PL))
               for i, label in enumerate(labels)]
    csvfile("H_srednie.csv", ["tokenizator", "srednia_PL", "srednia_EN", "PL_do_EN", "tokeny_na_wyraz_Pan_Tadeusz",
                              "tokeny_na_wyraz_tekst_wspolczesny"], summary)
    say("H4: zdania EN to cytaty z Hamleta, zdania PL to ich własne tłumaczenia (te same treści); tabela H_liczba_tokenow.csv.")
    say("H4: tokeny na wyraz liczone na samych wyrazach (bez interpunkcji): cały Pan Tadeusz i współczesny akapit spoza książki.")
    for row in summary:
        say(f"H5: {row[0]}: średnio PL {row[1]}, EN {row[2]} tokenów/zdanie; PL/EN = {row[3]}; "
            f"tokeny na wyraz: Pan Tadeusz {row[4]}, tekst współczesny {row[5]}")
    fig, ax = plt.subplots(figsize=(10, 5))
    for shift, lang in ((-0.2, "PL"), (0.2, "EN")):
        bars = ax.bar([i + shift for i in range(len(labels))], mean[lang], width=0.4, label=lang)
        ax.bar_label(bars, fmt="%.1f")
    ax.set_xticks(range(len(labels)), labels)
    ax.set(ylabel="Średnia liczba tokenów na zdanie", title="H: średnia liczba tokenów na zdanie")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "H_tokenizery.png", dpi=180)
    plt.close(fig)

    whole, pl_test = flat(pl), " ".join(p for p, _ in parallel)
    curve = [(size, bpe[size].get_vocab_size(), len(bpe[size].encode(whole).tokens), len(bpe[size].encode(pl_test).tokens),
              per_word(lambda s, t=bpe[size]: len(t.encode(s).tokens), pl),
              per_word(lambda s, t=bpe[size]: len(t.encode(s).tokens), MODERN_PL))
             for size in BPE_SIZES]
    csvfile("H_tokeny_vs_slownik.csv", ["zadany_rozmiar", "rzeczywisty_rozmiar", "tokeny_Pan_Tadeusz", "tokeny_zdania_PL",
                                        "tokeny_na_wyraz_Pan_Tadeusz", "tokeny_na_wyraz_tekst_wspolczesny"], curve)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, column, title in ((axes[0], 2, "Pan Tadeusz (tekst treningowy)"), (axes[1], 3, "5 zdań PL spoza książki")):
        ax.plot([r[1] for r in curve], [r[column] for r in curve], marker="o")
        ax.set(xlabel="Rozmiar słownika", ylabel="Liczba tokenów", title=title)
    fig.suptitle("H: liczba tokenów a rozmiar słownika BPE")
    fig.tight_layout()
    fig.savefig(OUT / "H_tokeny_vs_slownik.png", dpi=180)
    plt.close(fig)
    say("H6: rozmiar słownika → tokeny (Pan Tadeusz, zdania PL): "
        + "; ".join(f"{r[1]} → {r[2]}, {r[3]}" for r in curve) + "; wykres H_tokeny_vs_slownik.png.")
    say("H6: tokeny na wyraz (Pan Tadeusz / tekst współczesny): "
        + "; ".join(f"{r[1]}: {r[4]} / {r[5]}" for r in curve) + ".")
    rows = {r[0]: r for r in summary}
    best = {lang: labels[min(range(len(labels)), key=lambda i: mean[lang][i])] for lang in ("PL", "EN")}
    say(f"H6a: najmniej tokenów na zdanie: dla polskiego {best['PL']}, dla angielskiego {best['EN']}.")
    doubling = [1 - (b[2] / a[2]) ** (1 / math.log2(b[1] / a[1])) for a, b in zip(curve, curve[1:])]
    by_size = {r[0]: r[2] for r in curve}
    say(f"H6b: większy słownik daje mniej tokenów, ale nie liniowo: każde podwojenie słownika zmniejsza liczbę tokenów Pana Tadeusza")
    say(f"o podobny procent ({min(doubling):.0%}–{max(doubling):.0%}), więc zależność jest w przybliżeniu logarytmiczna. "
        f"Przejście 1000→5000 oszczędza {(by_size[1000] - by_size[5000]) / 4000:.1f} tokenu na każdy nowy wpis,")
    say(f"a 5000→30000 już tylko {(by_size[5000] - by_size[30000]) / 25000:.1f}; kolejne scalenia dotyczą coraz rzadszych ciągów.")
    say(f"H6c: tokenizator uczony na jednej książce zapamiętuje jej słownictwo (imiona, archaizmy, częste formy): na Panu Tadeuszu")
    say(f"daje {rows['BPE_5000'][4]} tokenu na wyraz, na współczesnym tekście {rows['BPE_5000'][5]}; wariant 30000: "
        f"{curve[-1][4]} wobec {curve[-1][5]} (Soplicowo to 1 token, hiperparametryzacja 8). HerBERT, uczony na dużym")
    say(f"korpusie, zachowuje się odwrotnie ({rows['HerBERT'][4]} / {rows['HerBERT'][5]}), bo to archaiczny wiersz jest dla niego rzadszy.")
    say(f"H6d: podatek językowy to dodatkowe tokeny (koszt, czas, krótszy kontekst) za tę samą treść w języku słabo reprezentowanym")
    say(f"w danych tokenizatora. Na zdaniach równoległych gpt2 potrzebuje dla polskiego {rows['gpt2'][3]}× tyle tokenów co dla angielskiego;")
    say("pozostałe PL/EN: " + ", ".join(f"{r[0]} {r[3]}" for r in summary if r[0] != "gpt2")
        + ". Tokenizatory uczone na polszczyźnie odwracają podatek — płaci go wtedy angielski.")
    if any("zastępczo" in label for label in labels):
        say("H6d: Bielik v2.3 dziedziczy tokenizator po Mistralu (głównie angielski), dlatego płaci podatek; wyniki Bielika v3")
        say("trzeba uzupełnić po uzyskaniu dostępu do repozytorium na Hugging Face.")

    heading("I porównanie metod")
    book = [t for d in nlp.pipe(lines, batch_size=256, disable=["parser", "ner"]) for t in d if t.is_alpha]
    levels = [method_variants([t for t in doc if t.is_alpha], stop, own), method_variants(book, stop, own)]
    bases = [len(set(level[0][1])) for level in levels]
    i_rows = [(name, *(x for level, base in zip(levels, bases)
                       for x in (len(level[i][1]), len(set(level[i][1])),
                                 round(100 * (len(set(level[i][1])) - base) / base, 1))),
               " ".join(units[:12]))
              for i, (name, units) in enumerate(levels[0])]
    csvfile("I_porownanie.csv", ["wariant", "jednostki_fragment", "slownik_fragment", "zmiana_slownika_fragment_proc",
                                 "jednostki_ksiazka", "slownik_ksiazka", "zmiana_slownika_ksiazka_proc", "poczatek"], i_rows)
    say(f"I1–I2: ten sam fragment co w E ({len(fragment)} znaków) oraz dla skali cały Pan Tadeusz; tabela I_porownanie.csv.")
    for row in i_rows:
        say(f"I2: {row[0]}: fragment {row[1]} jednostek, słownik {row[2]} ({row[3]:+}%); "
            f"cała książka {row[4]} jednostek, słownik {row[5]} ({row[6]:+}%); początek: {row[7]}")
    words_i, lemma_i, stop_i, bpe_i = i_rows
    say(f"I3: usunięcie stop words na fragmencie zmniejsza słownik o {-stop_i[3]}%, ale w książce tylko o {-stop_i[6]}%: usuwa {words_i[5] - stop_i[5]}")
    say("typów, więc skraca głównie tekst, a nie słownik; kosztem jest utrata składni i negacji (nie też jest na liście).")
    say(f"I3: lematyzacja łączy formy fleksyjne: {lemma_i[3]:+}% na fragmencie i {lemma_i[6]:+}% w książce; koszt to utrata informacji")
    say("gramatycznej (przypadek, liczba, czas) i błędy lematyzatora (litwo, ojczyzno zostają bez zmian).")
    say(f"I3: BPE na fragmencie powiększa słownik ({bpe_i[3]:+}%), bo jeden wyraz daje kilka różnych podsłów, ale w skali książki")
    say(f"ma najmniejszy słownik ({bpe_i[5]} jednostek, {bpe_i[6]:+}%), z góry ograniczony do 5000 i bez słów spoza słownika.")
    say(f"Kosztem są dłuższe sekwencje ({bpe_i[4]} zamiast {words_i[4]} jednostek) i podsłowa bez własnego znaczenia (▁Lit wo, ▁Oj czyz no).")

    (OUT / "wyniki_lab2.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
