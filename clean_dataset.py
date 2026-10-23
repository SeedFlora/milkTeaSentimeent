# -*- coding: utf-8 -*-
"""
Pipeline pembersihan & penyeimbangan dataset sentimen review milk tea.

Tahapan:
1. Perbaiki mojibake (encoding rusak) dan buang emoji/simbol.
2. Normalisasi slang -> bahasa Indonesia formal (kamus ~400 entri).
3. Buang URL, angka, tanda baca, huruf berulang (enakkk -> enak).
4. Hapus duplikat & teks kosong; buang teks yang labelnya konflik.
5. Augmentasi kelas minoritas (negatif) dengan penggantian sinonim
   (pola kata diambil dari komentar yang ada) sampai label seimbang.
"""

import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

random.seed(42)

BASE = Path(__file__).resolve().parent
SRC = BASE / "dataset_preprocessed (2).csv"
OUT_CLEAN = BASE / "dataset_clean.csv"
OUT_BALANCED = BASE / "dataset_clean_balanced.csv"

# ---------------------------------------------------------------- mojibake
def fix_mojibake(s: str) -> str:
    """Perbaiki teks UTF-8 yang terlanjur didekode sebagai cp1252 (mis. 'â€¢')."""
    for _ in range(2):
        try:
            fixed = s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if fixed == s:
            break
        s = fixed
    return s

# ------------------------------------------------------- kamus slang->formal
# Diturunkan dari frekuensi token korpus ini (yg, ga, bgt, kalo, cepet, ...).
SLANG = {
    # kata fungsi / singkatan umum
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "utk": "untuk",
    "dr": "dari", "dlm": "dalam", "pd": "pada", "sm": "sama",
    "jg": "juga", "jgn": "jangan", "jd": "jadi", "jdi": "jadi",
    "tp": "tetapi", "tpi": "tetapi", "tapi": "tetapi",
    "krn": "karena", "karna": "karena", "krna": "karena", "soalnya": "karena",
    "kalo": "kalau", "klo": "kalau", "kl": "kalau",
    "sdh": "sudah", "udh": "sudah", "udah": "sudah", "uda": "sudah", "dah": "sudah",
    "blm": "belum", "blum": "belum", "belom": "belum", "blom": "belum",
    "tdk": "tidak", "tak": "tidak", "ndak": "tidak", "nda": "tidak",
    "ga": "tidak", "gak": "tidak", "gk": "tidak", "ngga": "tidak",
    "nggak": "tidak", "engga": "tidak", "enggak": "tidak",
    "kaga": "tidak", "kagak": "tidak", "gx": "tidak", "gada": "tidak ada",
    "gaada": "tidak ada", "gatau": "tidak tahu", "gabisa": "tidak bisa",
    "gamau": "tidak mau", "gajelas": "tidak jelas", "gpp": "tidak apa-apa",
    "gapapa": "tidak apa-apa",
    "bs": "bisa", "bsa": "bisa", "hrs": "harus", "msti": "harus",
    "mesti": "harus", "musti": "harus",
    "lg": "lagi", "lgi": "lagi", "dpt": "dapat", "byk": "banyak",
    "bnyk": "banyak", "bnyak": "banyak",
    "org": "orang", "orng": "orang", "sgt": "sangat",
    "bgt": "sangat", "bngt": "sangat", "banget": "sangat", "banged": "sangat",
    "bet": "sangat", "pisan": "sangat",
    "skrg": "sekarang", "skrng": "sekarang", "kmrn": "kemarin",
    "kemaren": "kemarin", "bsk": "besok", "hr": "hari",
    "mlm": "malam", "pgi": "pagi", "mnt": "menit", "dtg": "datang",
    "dateng": "datang", "dtng": "datang", "plg": "pulang",
    "lbh": "lebih", "lbih": "lebih", "spt": "seperti", "sprt": "seperti",
    "kyk": "seperti", "kayak": "seperti", "kek": "seperti", "kayaknya": "sepertinya",
    "kyknya": "sepertinya", "keknya": "sepertinya",
    "gt": "begitu", "gtu": "begitu", "gitu": "begitu", "gini": "begini",
    "gmn": "bagaimana", "gmna": "bagaimana", "gimana": "bagaimana",
    "knp": "kenapa", "knpa": "kenapa",
    "emg": "memang", "emang": "memang", "mmg": "memang", "emng": "memang",
    "trs": "terus", "trus": "terus", "tros": "terus",
    "aja": "saja", "aj": "saja", "ajah": "saja", "doang": "saja", "doank": "saja",
    "cuman": "hanya", "cmn": "hanya", "cuma": "hanya",
    "bkn": "bukan", "bnr": "benar", "bner": "benar", "bener": "benar",
    "beneran": "benar-benar", "bnran": "benar-benar", "sumpah": "sungguh",
    "dikit": "sedikit", "sdikit": "sedikit", "sdkt": "sedikit",
    "tsb": "tersebut", "dll": "dan lain-lain", "dsb": "dan sebagainya",
    "spy": "supaya", "biar": "agar", "bgmn": "bagaimana",
    "trnyata": "ternyata", "ternyta": "ternyata", "pdhl": "padahal",
    "pokonya": "pokoknya", "pkknya": "pokoknya",
    "tmpt": "tempat", "tmpat": "tempat", "tmptnya": "tempatnya",
    "hrg": "harga", "hrga": "harga", "hrganya": "harganya",
    "tmbh": "tambah", "nambah": "menambah",
    # kata ganti orang
    "sy": "saya", "sya": "saya", "gw": "saya", "gue": "saya", "gua": "saya",
    "gwe": "saya", "aku": "saya", "ak": "saya", "aq": "saya", "ku": "saya",
    "lo": "anda", "lu": "anda", "loe": "anda", "elo": "anda", "km": "kamu",
    "mba": "mbak",
    # kata kerja informal
    "pake": "pakai", "pakek": "pakai", "pk": "pakai", "make": "memakai",
    "pesen": "pesan", "mesen": "memesan", "order": "pesan", "orderan": "pesanan",
    "nyobain": "mencoba", "cobain": "mencoba", "nyoba": "mencoba",
    "dicobain": "dicoba", "coba": "mencoba",
    "nunggu": "menunggu", "nungguin": "menunggu", "nunggunya": "menunggunya",
    "ngantri": "mengantre", "ngantre": "mengantre",
    "antri": "antre", "antrian": "antrean", "antriannya": "antreannya",
    "liat": "melihat", "ngeliat": "melihat", "liatin": "melihat",
    "ngerasa": "merasa", "kerasa": "terasa", "krasa": "terasa",
    "ngerti": "mengerti", "ngasih": "memberi", "dikasih": "diberi",
    "dikasi": "diberi", "ngomong": "berbicara", "ngobrol": "mengobrol",
    "nongkrong": "bersantai", "nangkring": "bersantai",
    "bikin": "membuat", "bkin": "membuat", "ngebantah": "membantah",
    "beliin": "membelikan", "dibeliin": "dibelikan", "nyari": "mencari",
    "dapet": "dapat", "dpet": "dapat", "ketemu": "bertemu",
    "ngambil": "mengambil", "ngasi": "memberi", "naro": "menaruh",
    "nanya": "bertanya", "ngecek": "memeriksa", "dianter": "diantar",
    "nganter": "mengantar", "anter": "antar", "dibales": "dibalas",
    "bales": "balas", "ngelayanin": "melayani", "ngelayani": "melayani",
    "dilayanin": "dilayani", "ngehargain": "menghargai",
    "tau": "tahu", "ketauan": "ketahuan", "ngantuk": "mengantuk",
    "ilang": "hilang", "nyesel": "menyesal", "kesini": "ke sini",
    "kesana": "ke sana", "disini": "di sini", "disitu": "di situ",
    "disana": "di sana", "dimana": "di mana", "diatas": "di atas",
    "dibawah": "di bawah", "didalam": "di dalam", "diluar": "di luar",
    # kata sifat / sentimen informal
    "cepet": "cepat", "gercep": "cepat", "lemot": "lambat", "lelet": "lambat",
    "lambreta": "lambat", "seger": "segar", "anget": "hangat",
    "adem": "sejuk", "rame": "ramai", "ramenya": "ramainya",
    "deket": "dekat", "dkt": "dekat", "enk": "enak", "enaak": "enak",
    "endul": "enak", "endol": "enak", "endeus": "enak", "maknyus": "enak",
    "mantul": "mantap", "mantab": "mantap", "mantep": "mantap",
    "manteb": "mantap", "jos": "mantap", "joss": "mantap", "gandos": "mantap",
    "kece": "keren", "kereen": "keren", "cakep": "bagus", "bgs": "bagus",
    "murce": "murah", "murmer": "murah meriah", "mehong": "mahal",
    "kemahalan": "terlalu mahal", "kemanisan": "terlalu manis",
    "keasinan": "terlalu asin", "kepanasan": "terlalu panas",
    "kemurahan": "terlalu murah", "kelamaan": "terlalu lama",
    "judes": "ketus", "jutek": "ketus", "songong": "sombong",
    "belagu": "sombong", "nyebelin": "menyebalkan", "ngeselin": "menyebalkan",
    "kesel": "kesal", "sebel": "kesal", "sebal": "kesal", "kzl": "kesal",
    "bete": "kesal", "bt": "kesal", "zonk": "mengecewakan",
    "males": "malas", "mls": "malas", "mager": "malas",
    "asik": "asyik", "asyikk": "asyik", "happy": "senang",
    "nagih": "ketagihan", "favorit": "favorit", "fav": "favorit",
    "recommended": "direkomendasikan", "recomended": "direkomendasikan",
    "rekomended": "direkomendasikan", "rekomen": "direkomendasikan",
    "recommend": "merekomendasikan", "recomend": "merekomendasikan",
    "oke": "baik", "okey": "baik", "okay": "baik", "ok": "baik", "oce": "baik",
    "okelah": "cukup baik", "lumayanlah": "lumayan",
    # bahasa inggris umum -> indonesia
    "best": "terbaik", "good": "bagus", "nice": "bagus", "great": "bagus",
    "bad": "buruk", "worst": "terburuk", "slow": "lambat", "fast": "cepat",
    "friendly": "ramah", "clean": "bersih", "dirty": "kotor",
    "comfy": "nyaman", "cozy": "nyaman", "comfortable": "nyaman",
    "delicious": "enak", "tasty": "enak", "yummy": "enak", "yumm": "enak",
    "fresh": "segar", "sweet": "manis", "expensive": "mahal",
    "cheap": "murah", "crowded": "ramai", "love": "suka", "lovee": "suka",
    "service": "layanan", "staff": "staf", "place": "tempat",
    "price": "harga", "taste": "rasa", "perfect": "sempurna",
    "excellent": "sangat baik", "exxcellent": "sangat baik",
    "pricy": "mahal", "pricey": "mahal", "overprice": "terlalu mahal",
    "overpriced": "terlalu mahal", "slowly": "lambat",
    "thx": "terima kasih", "tq": "terima kasih", "thanks": "terima kasih",
    "thankyou": "terima kasih", "makasih": "terima kasih",
    "makasi": "terima kasih", "mksh": "terima kasih", "trims": "terima kasih",
    "terimakasih": "terima kasih",
    # partikel percakapan -> dihapus (tidak formal, tidak membawa makna)
    "sih": "", "deh": "", "dong": "", "kok": "", "lho": "", "loh": "",
    "lah": "", "nih": "", "tuh": "", "yah": "", "eh": "", "hmm": "",
    "hm": "", "duh": "", "wkwk": "", "wkwkwk": "", "wkwkwkwk": "",
    "haha": "", "hahaha": "", "hehe": "", "hehehe": "", "hihi": "",
    "xixi": "", "lol": "", "anjir": "", "anjay": "", "njir": "",
    "anjrit": "", "btw": "", "guys": "", "gan": "", "bro": "", "sis": "",
    "kak": "", "min": "", "ya": "", "yaa": "",
}

# frasa multi-kata yang harus diganti sebelum pemetaan per token
PHRASES = [
    (r"\bworth\s*it\b", "layak"),
    (r"\bby\s+the\s+way\b", ""),
    (r"\bso\s+so\b", "biasa saja"),
    (r"\bnot\s+bad\b", "lumayan"),
    (r"\bcustomer\s+service\b", "layanan pelanggan"),
    (r"\bbintang\s*5\b", "bintang lima"),
    (r"\bbintang\s*4\b", "bintang empat"),
    (r"\bbintang\s*1\b", "bintang satu"),
]

# kata yang boleh berakhiran huruf ganda (jangan dipangkas)
DOUBLE_OK = {"less", "press", "express", "all", "small", "still", "free",
             "coffee", "tea", "see", "too", "good", "sweet", "ee", "stress"}

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_NONLETTER = re.compile(r"[^a-z\s-]")
RE_REPEAT3 = re.compile(r"(.)\1{2,}")
RE_SPACE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = fix_mojibake(str(s))
    s = s.lower()
    s = RE_URL.sub(" ", s)
    for pat, rep in PHRASES:
        s = re.sub(pat, rep, s)
    s = RE_NONLETTER.sub(" ", s)          # buang angka, tanda baca, emoji
    s = RE_REPEAT3.sub(r"\1", s)          # enakkk -> enak
    tokens = []
    for tok in s.split():
        tok = tok.strip("-")
        if not tok:
            continue
        # pangkas huruf ganda di akhir kata: baikk -> baik, baruu -> baru
        if len(tok) > 2 and tok[-1] == tok[-2] and tok not in DOUBLE_OK \
                and tok not in SLANG:
            tok = tok[:-1]
        tok = SLANG.get(tok, tok)
        if tok:
            tokens.append(tok)
    out = " ".join(tokens)
    # pemetaan ulang utk hasil frasa (mis. "tidak tahu" dari gatau) sudah formal
    return RE_SPACE.sub(" ", out).strip()


# ------------------------------------------------ augmentasi sinonim (EDA)
# Grup sinonim diambil dari pola kata yang benar-benar muncul pada komentar
# (terutama komentar negatif) setelah normalisasi.
SYNONYMS = [
    {"enak", "lezat", "nikmat", "sedap"},
    {"tidak", "tak"},
    {"sangat", "amat", "sungguh"},
    {"buruk", "jelek"},
    {"lama", "lambat", "lamban"},
    {"kotor", "jorok"},
    {"ramah", "sopan"},
    {"cepat", "sigap"},
    {"pelayanan", "layanan", "servis"},
    {"tempat", "lokasi"},
    {"staf", "pegawai", "karyawan"},
    {"toko", "gerai", "outlet", "kedai"},
    {"suka", "senang", "gemar"},
    {"menunggu", "menanti"},
    {"salah", "keliru"},
    {"kasar", "ketus"},
    {"mengecewakan", "buruk"},
    {"bagus", "baik", "memuaskan"},
    {"murah", "terjangkau"},
    {"kecewa", "tidak puas"},
    {"marah", "kesal"},
    {"mahal", "kurang terjangkau"},
    {"banyak", "melimpah"},
    {"sedikit", "minim"},
    {"panas", "hangat"},
    {"dingin", "sejuk"},
    {"manis", "legit"},
    {"sombong", "angkuh"},
    {"menyebalkan", "menjengkelkan"},
    {"sempurna", "luar biasa"},
]
SYN_MAP = {}
for group in SYNONYMS:
    for w in group:
        SYN_MAP.setdefault(w, set()).update(group - {w})


def synonym_variant(text: str, p_replace: float = 0.4) -> str:
    """Buat satu varian teks dengan mengganti kata bersinonim secara acak."""
    tokens = text.split()
    candidates = [i for i, t in enumerate(tokens) if t in SYN_MAP]
    if not candidates:
        return text
    n_target = max(1, round(len(candidates) * p_replace))
    chosen = random.sample(candidates, min(n_target, len(candidates)))
    for i in chosen:
        tokens[i] = random.choice(sorted(SYN_MAP[tokens[i]]))
    # sesekali tukar dua kata bertetangga agar varian lebih beragam
    if len(tokens) > 4 and random.random() < 0.3:
        j = random.randrange(len(tokens) - 1)
        tokens[j], tokens[j + 1] = tokens[j + 1], tokens[j]
    return " ".join(tokens)


def main():
    df = pd.read_csv(SRC)
    n0 = len(df)

    # --- bersihkan (basis: text_id = teks asli yang sudah berbahasa Indonesia)
    df["text_clean"] = df["text_id"].map(clean_text)

    # --- buang kosong / terlalu pendek (tanpa kata sama sekali)
    df = df[df["text_clean"].str.len() > 0]
    df = df[df["text_clean"].str.split().str.len() >= 2]
    n_short = n0 - len(df)

    # --- label dari kolom yang ada (konsisten: bintang 1-3 negatif, 4-5 positif)
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["positif", "negatif"])]

    # --- hapus duplikat teks; buang teks yang muncul dengan dua label berbeda
    conflict = df.groupby("text_clean")["label"].nunique()
    conflict_texts = set(conflict[conflict > 1].index)
    df = df[~df["text_clean"].isin(conflict_texts)]
    n_conflict = len(conflict_texts)
    before_dedup = len(df)
    df = df.drop_duplicates(subset="text_clean", keep="first")
    n_dup = before_dedup - len(df)

    clean = df[["text", "text_clean", "stars", "label"]].copy()
    clean["label_id"] = (clean["label"] == "positif").astype(int)
    clean.to_csv(OUT_CLEAN, index=False, encoding="utf-8")

    counts = Counter(clean["label"])
    print(f"baris awal               : {n0}")
    print(f"dibuang (kosong/<2 kata) : {n_short}")
    print(f"teks konflik label       : {n_conflict}")
    print(f"duplikat dibuang         : {n_dup}")
    print(f"baris bersih             : {len(clean)}")
    print(f"distribusi label bersih  : {dict(counts)}")

    # --- augmentasi sinonim untuk kelas minoritas
    major = counts.most_common(1)[0][0]
    minor = "negatif" if major == "positif" else "positif"
    target = counts[major]

    minority_texts = clean.loc[clean["label"] == minor, "text_clean"].tolist()
    existing = set(clean["text_clean"])
    augmented = []
    need = target - counts[minor]
    attempts = 0
    max_attempts = need * 60
    while len(augmented) < need and attempts < max_attempts:
        base = random.choice(minority_texts)
        var = synonym_variant(base)
        attempts += 1
        if var not in existing:
            existing.add(var)
            augmented.append(var)

    print(f"\naugmentasi '{minor}'     : butuh {need}, dihasilkan {len(augmented)}")

    aug_df = pd.DataFrame({
        "text_clean": augmented,
        "label": minor,
        "is_augmented": 1,
    })
    base_df = clean[["text_clean", "label"]].copy()
    base_df["is_augmented"] = 0
    final = pd.concat([base_df, aug_df], ignore_index=True)

    # jika augmentasi tidak cukup, samakan dengan undersampling kelas mayoritas
    fc = Counter(final["label"])
    if fc[major] > fc[minor]:
        major_rows = final[final["label"] == major].sample(fc[minor], random_state=42)
        final = pd.concat([major_rows, final[final["label"] == minor]],
                          ignore_index=True)

    final["label_id"] = (final["label"] == "positif").astype(int)
    final = final.sample(frac=1, random_state=42).reset_index(drop=True)
    final = final[["text_clean", "label", "label_id", "is_augmented"]]
    final.to_csv(OUT_BALANCED, index=False, encoding="utf-8")

    print(f"\ndistribusi label final   : {dict(Counter(final['label']))}")
    print(f"total baris final        : {len(final)}")
    print(f"\nfile tersimpan:\n  {OUT_CLEAN}\n  {OUT_BALANCED}")

    print("\ncontoh hasil pembersihan:")
    sample = clean.sample(5, random_state=7)
    for _, r in sample.iterrows():
        print(f"  [{r['label']}] {r['text_clean'][:110]}")
    print("\ncontoh hasil augmentasi:")
    for t in augmented[:5]:
        print(f"  [negatif] {t[:110]}")


if __name__ == "__main__":
    main()
