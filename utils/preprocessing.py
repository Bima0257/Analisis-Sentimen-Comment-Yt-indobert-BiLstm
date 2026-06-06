import re
import csv
import html
import emoji


def clean_text(text: str) -> str:
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_repeated_chars(text: str) -> str:
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def remove_emoji_text(text: str) -> str:
    return emoji.replace_emoji(text, replace=' ')


def load_slang_dict(filepath="slang_indo.csv"):
    slang_dict = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    slang = row[0].strip().lower()
                    formal = row[1].strip().lower()
                    if slang and formal:
                        slang_dict[slang] = formal
    except FileNotFoundError:
        slang_dict = {
            "gk": "tidak", "ga": "tidak", "gak": "tidak", "ngga": "tidak",
            "nggak": "tidak", "enggak": "tidak", "gapapa": "tidak apa",
            "bgt": "sangat", "bngt": "sangat", "banget": "sangat",
            "yg": "yang", "dr": "dari", "tp": "tapi", "jg": "juga",
            "udh": "sudah", "udah": "sudah", "sdh": "sudah",
            "lg": "lagi", "blm": "belum",
            "dgn": "dengan", "utk": "untuk", "tdk": "tidak",
            "klo": "kalau", "kalo": "kalau", "krn": "karena",
            "emg": "memang", "emang": "memang", "aja": "saja",
            "deh": "", "sih": "", "dong": "", "nih": "",
            "koplak": "bodoh", "anjir": "astaga", "anjay": "astaga",
            "wkwk": "", "wkwkwk": "", "haha": "", "hehe": "",
            "mantap": "bagus", "mantul": "bagus sekali",
            "mksh": "terima kasih", "makasih": "terima kasih",
            "thx": "terima kasih", "tq": "terima kasih",
            "sy": "saya", "aq": "saya", "gw": "saya", "gue": "saya",
            "lo": "kamu", "lu": "kamu", "km": "kamu",
        }
    return slang_dict


slang_dict = load_slang_dict("slang_indo.csv")


def normalize_slang(text: str) -> str:
    words = text.split()
    normalized = []
    for word in words:
        stripped = re.sub(r'[^\w]', '', word)
        replacement = slang_dict.get(word, slang_dict.get(stripped, word))
        if replacement:
            normalized.append(replacement)
    return " ".join(normalized).strip()


def preprocess(text: str) -> str:
    text = clean_text(text)
    text = remove_emoji_text(text)
    text = normalize_repeated_chars(text)
    text = normalize_slang(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
