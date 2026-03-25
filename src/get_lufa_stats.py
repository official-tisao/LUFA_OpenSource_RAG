    text = build_marked(pages)
    lines = text.splitlines()
    out = []
    cur_article = '0';
    cur_clause = '0';
    cur_title = 'PREAMBLE';
    cur_page = 1;
    buf = []

    def flush():
        nonlocal buf
        t = '\n'.join(buf).strip()
        t = PAGE_MARK_RE.sub('', t).strip()
        if t:
            out.append({'article_number': cur_article, 'clause_id': cur_clause, 'section_title': cur_title, 'text': t,
                        'language': detect_lang(t, forced_lang), 'page_no': cur_page, 'tokens': tokens(t)})
        buf = []

    for line in lines:
        m = PAGE_MARK_RE.match(line)
        if m:
            cur_page = int(m.group(1));
            buf.append(line);
            continue
        s = line.strip()
        if not s:
            buf.append(line);
            continue
        if SECTION_RE.match(s):
            cur_title = s;
            buf.append(line);
            continue
        am = ARTICLE_RE.match(s)
        if am:
            flush();
            cur_article = am.group(1);
            cur_clause = cur_article;
            buf.append(line);
            continue
        cm = CLAUSE_RE.match(s)
        if cm and cm.group(1).startswith(cur_article + '.'):
            flush();
            cur_clause = cm.group(1).replace('(', ' .').replace(')', '').replace(' ', '').replace('..', '.').replace(
                '. ', ' .').replace(' .', '.')
            cur_clause = re.sub(r'\(([a-z]+)\)', r'.\1', cur_clause)
            buf.append(line);
            continue
        buf.append(line)
    flush()

    # merge short clauses into next clause
    merged = [];
    pending = [];
    pending_page = None
    for c in out:
        if c['tokens'] < min_tokens:
            pending.append(c)
            pending_page = pending_page or c['page_no']
        else:
            if pending:
                prefix = '\n'.join(x['text'] for x in pending)
                c['text'] = (prefix + '\n' + c['text']).strip()
                c['tokens'] = tokens(c['text'])
                c['page_no'] = pending_page or c['page_no']
                pending = [];
                pending_page = None
            merged.append(c)
    if pending and merged:
        last = merged[-1]
        last['text'] = (last['text'] + '\n' + '\n'.join(x['text'] for x in pending)).strip()
        last['tokens'] = tokens(last['text'])
    return merged


def get_pdf_stats(pdf_path, lang):
    if not pdf_path or not Path(pdf_path).exists():
        return {'articles': 0, 'clauses': 0, 'chunks': 0, 'avg_tokens': 0, 'min_tokens': 0, 'max_tokens': 0,
                'total_tokens': 0, 'appendices': 0}

    nodes = chunk_pdf(Path(pdf_path), lang)
    df = pd.DataFrame(nodes)
    if len(df) == 0:
        return {'articles': 0, 'clauses': 0, 'chunks': 0, 'avg_tokens': 0, 'min_tokens': 0, 'max_tokens': 0,
                'total_tokens': 0, 'appendices': 0}

    return {
        'articles': df['article_number'].replace('0', pd.NA).dropna().nunique(),
        'clauses': df['clause_id'].replace('0', pd.NA).dropna().nunique(),
        'chunks': len(df),
        'avg_tokens': round(df['tokens'].mean()),
        'min_tokens': int(df['tokens'].min()),
        'max_tokens': int(df['tokens'].max()),
        'total_tokens': int(df['tokens'].sum()),
        'appendices': int(
            df['section_title'].str.contains('SCHEDULE|APPENDIX|ANNEXE|PARTIE', case=False, na=False).sum())
    }


def scan_chroma_db_bilingual(db_path):
    db_dir = Path(db_path)
    if not db_dir.exists():
        print(f"ChromaDB directory not found at {db_path}")
        return None, None

    print(f"Scanning ChromaDB SQLite database at {db_path}...")
    client = chromadb.PersistentClient(path=str(db_path))

    stats_en = {'chunks': 0, 'total_tokens': 0, 'articles': set(), 'clauses': set(), 'token_list': [], 'appendices': 0}
    stats_fr = {'chunks': 0, 'total_tokens': 0, 'articles': set(), 'clauses': set(), 'token_list': [], 'appendices': 0}

    collections = client.list_collections()
    if not collections:
        print("No collections found in the database.")
        return None, None

    for coll_meta in collections:
        coll_name = coll_meta.name if hasattr(coll_meta, 'name') else coll_meta
        coll = client.get_collection(coll_name)
        data = coll.get(include=['documents', 'metadatas'])

        docs = data.get('documents', [])
