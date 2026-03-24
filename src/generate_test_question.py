        )

        if hard_boundary:
            flush_block()
            continue

        current_lines.append(line)

        if len(current_lines) >= 8:
            flush_block()

    flush_block()
    return blocks

def split_sentences(text):
    parts = re.split(r'(?<=[\.\?!;:])\s+', text)
    out = []
    for p in parts:
        p = p.strip()
        p = re.sub(r"\s+", " ", p)
        if 45 <= len(p) <= 450:
            out.append(p)
    return out

def extract_topic(text, language, article_title=None):
    if article_title and len(article_title.split()) <= 7:
        return article_title.lower()

    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-']{3,}", text.lower())
    stop = FR_STOP if language == "French" else EN_STOP
    words = [w for w in words if w not in stop]

    if not words:
        return "this matter" if language == "English" else "cette question"

    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    top = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))[:4]
    phrase = " ".join([w for w, _ in top[:2]])
    return phrase if phrase else ("this matter" if language == "English" else "cette question")

def classify_difficulty(answer):
    wc = len(answer.split())
    if wc < 18:
        return "easy"
    elif wc < 35:
        return "medium"
    return "hard"

def category_from_article(article_title, language):
    if not article_title:
        return None

    title = article_title.lower()
    hint_map = FR_ARTICLE_CATEGORY_HINTS if language == "French" else EN_ARTICLE_CATEGORY_HINTS

    best = None
    best_len = -1
    for hint, category in hint_map.items():
        if hint in title and len(hint) > best_len:
            best = category
            best_len = len(hint)
    return best

def map_category(text, language, article_title=None):
    article_category = category_from_article(article_title, language)
    if article_category:
        return article_category

    text_l = text.lower()
    keyword_map = FR_CATEGORY_KEYWORDS if language == "French" else EN_CATEGORY_KEYWORDS
    fallback = "Dispositions générales" if language == "French" else "General Provisions"

    scores = {}
    for category, keywords in keyword_map.items():
        score = 0
        for kw in keywords:
            if kw.lower() in text_l:
                score += 2 if " " in kw else 1
        if score > 0:
            scores[category] = score

    if scores:
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[0][0]

    return fallback

def make_question(answer, language, article_title=None):
    topic = extract_topic(answer, language, article_title=article_title)
    template = random.choice(FR_Q_TEMPLATES if language == "French" else EN_Q_TEMPLATES)
    q = template.format(topic=topic)
    return q[0].upper() + q[1:] if q else q

def normalize_answer(text):
    text = text.strip(' "\'')
    text = re.sub(r"\s+", " ", text)
    return text[:450].rstrip()

def dedupe_key(question, answer):
    q = re.sub(r"\W+", "", question.lower())
    a = re.sub(r"\W+", "", answer.lower())[:160]
    return q + "|" + a

def load_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f]

def build_candidates(blocks, expected_language):
    candidates = []

    for block in blocks:
        text = block["text"]
        article_number = block["article_number"]
        article_title = block["article_title"]

        sents = split_sentences(text)

        for sent in sents:
            sent = normalize_answer(sent)
            if 50 <= len(sent) <= 300:
                candidates.append({
                    "answer": sent,
                    "article_number": article_number,
                    "article_title": article_title,
                    "language": expected_language
                })

        if len(sents) >= 2:
            for i in range(len(sents) - 1):
                combined = normalize_answer(" ".join(sents[i:i+2]))
                if 60 <= len(combined) <= 420:
                    candidates.append({
                        "answer": combined,
                        "article_number": article_number,
                        "article_title": article_title,
                        "language": expected_language
                    })

    return candidates

def sample_qa(candidates, n, language, prefix):
    rows = []
    seen = set()
    attempts = 0
    idx = 1

    while len(rows) < n and attempts < n * 100:
        attempts += 1
        item = random.choice(candidates)

        answer = item["answer"]
        article_number = item["article_number"]
        article_title = item["article_title"]

        question = make_question(answer, language, article_title=article_title)
        category = map_category(answer + " " + question, language, article_title=article_title)
        key = dedupe_key(question, answer)

        if key in seen:
            continue

        seen.add(key)
        row_id = f"{prefix}_{idx:03d}"

        rows.append([
            row_id,
            question,
            answer,
            category,
            classify_difficulty(answer),
            language
        ])
        idx += 1

    return rows

def main():
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(f"Could not find {INPUT_FILE}")

    all_lines = load_lines(INPUT_FILE)

    en_lines = all_lines[EN_START - 1:EN_END]
    fr_lines = all_lines[FR_START - 1:FR_END]

    en_blocks = reconstruct_paragraphs_with_articles(en_lines, "English")
    fr_blocks = reconstruct_paragraphs_with_articles(fr_lines, "French")

    en_candidates = build_candidates(en_blocks, "English")
    fr_candidates = build_candidates(fr_blocks, "French")

    if len(en_candidates) < EN_TARGET:
        raise ValueError(f"Not enough English candidates: {len(en_candidates)}")
    if len(fr_candidates) < FR_TARGET:
        raise ValueError(f"Not enough French candidates: {len(fr_candidates)}")

    en_rows = sample_qa(en_candidates, EN_TARGET, "English", "test_en")
    fr_rows = sample_qa(fr_candidates, FR_TARGET, "French", "test_fr")

    rows = en_rows + fr_rows

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "expected_answer", "category", "difficulty", "language"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUTPUT_FILE}")
    print(f"English blocks: {len(en_blocks)}")
    print(f"French blocks: {len(fr_blocks)}")
    print(f"English candidates: {len(en_candidates)}")
    print(f"French candidates: {len(fr_candidates)}")

if __name__ == "__main__":
    main()