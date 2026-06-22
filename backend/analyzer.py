import re
from typing import Dict, List

from natasha import Doc, Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[^\w\s!?.,:;—\-ёа-яА-Я]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> List[str]:
    normalized = normalize_text(text)
    doc = Doc(normalized)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    lemmas = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        lemma = (token.lemma or token.text).lower().strip()
        if lemma and re.search(r"[а-яё]", lemma):
            lemmas.append(lemma)
    return lemmas


def matches_stem(word: str, stems: List[str]) -> bool:
    return any(word.startswith(stem) for stem in stems)


def has_any_in_window(tokens: List[str], start: int, end: int, items: set) -> bool:
    return any(tok in items for tok in tokens[max(0, start):end])


def analyze_text(text: str) -> Dict:
    original = text or ""
    lemmas = tokenize_and_lemmatize(original)
    word_count = len(lemmas)

    exclamations = original.count("!")
    questions = original.count("?")
    negation_count = sum(1 for w in lemmas if w in NEGATORS)
    intensity_count = sum(1 for w in lemmas if w in INTENSIFIERS)

    contrast_index = -1
    for i, w in enumerate(lemmas):
        if w in CONTRAST_WORDS:
            contrast_index = i

    pos = 0.0
    neg = 0.0
    found_keywords = []
    seen = set()

    for i, word in enumerate(lemmas):

        weight = 1.0

        if has_any_in_window(lemmas, i - 3, i, INTENSIFIERS):
            weight *= 1.35

        negated = has_any_in_window(lemmas, i - 2, i, NEGATORS)

        if contrast_index != -1 and i > contrast_index:
            weight *= 1.12

        is_positive = matches_stem(word, POSITIVE_STEMS)
        is_negative = matches_stem(word, NEGATIVE_STEMS)

        if is_positive:
            if negated:
                neg += weight
                keyword = f"не {word}"
            else:
                pos += weight
                keyword = word

            if keyword not in seen:
                seen.add(keyword)
                found_keywords.append(keyword)

        elif is_negative:
            if negated:
                pos += weight
                keyword = f"не {word}"
            else:
                neg += weight
                keyword = word

            if keyword not in seen:
                seen.add(keyword)
                found_keywords.append(keyword)

    base = (pos - neg) / max(word_count / 2, 1)
    punctuation_boost = (min(0.22, exclamations * 0.05) - min(0.08, questions * 0.02))
    intensity_boost = min(0.30, intensity_count * 0.08)

    direction = 0 if base == 0 else (1 if base > 0 else -1)
    raw_polarity = clamp(base + direction * (punctuation_boost + intensity_boost), -1, 1)

    score = int(clamp(round((raw_polarity + 1) * 50), 0, 100))

    if raw_polarity <= -0.65:
        tone = "Очень негативно"
        emoji = "😡"
        description = "Сильная негативная окраска и высокая эмоциональная напряжённость."
    elif raw_polarity < -0.25:
        tone = "Негативно"
        emoji = "☹️"
        description = "Негатив преобладает над позитивом."
    elif raw_polarity <= 0.25:
        tone = "Нейтрально"
        emoji = "😐"
        description = "Эмоциональный баланс близок к нейтральному."
    elif raw_polarity < 0.65:
        tone = "Позитивно"
        emoji = "🙂"
        description = "Преобладает позитивная и поддерживающая лексика."
    else:
        tone = "Очень позитивно"
        emoji = "😁"
        description = "Текст выраженно доброжелательный и эмоционально светлый."

    total_emotion_hits = pos + neg

    intensity = clamp((abs(raw_polarity) * 70) + (intensity_count * 6) + (exclamations * 3), 0, 100)
    subjectivity = clamp((total_emotion_hits / max(word_count, 1)) * 220, 0, 100)
    confidence = clamp(
        100 - (questions * 6) - (18 if word_count < 8 else 0) - max(0, 22 - word_count),
        20,
        99
    )
    lexical_density = clamp(
        ((word_count - sum(1 for w in lemmas if w in STOP_WORDS)) / max(word_count, 1)) * 100,
        0,
        100
    )
    negation_share = clamp((negation_count / max(word_count, 1)) * 100, 0, 100)
    punctuation_pressure = clamp(((exclamations * 2) + questions + intensity_count) * 12, 0, 100)
    emotional_contrast = clamp((abs(pos - neg) / max(total_emotion_hits, 1)) * 100, 0, 100)

    return {
        "tone": tone,
        "emoji": emoji,
        "description": description,
        "score": score,
        "rawPolarity": round(raw_polarity, 2),
        "wordCount": word_count,
        "pos": int(round(pos)),
        "neg": int(round(neg)),
        "intensity": round(intensity),
        "subjectivity": round(subjectivity),
        "confidence": round(confidence),
        "lexicalDensity": round(lexical_density),
        "negationShare": round(negation_share),
        "punctuationPressure": round(punctuation_pressure),
        "emotionalContrast": round(emotional_contrast),
        "keywords": found_keywords[:8],
    }


if __name__ == "__main__":
    sample = "Мне очень нравится этот проект, но местами он не совсем удобный."
    print(analyze_text(sample))