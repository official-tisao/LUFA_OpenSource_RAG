        metas = data.get('metadatas', [])

        for doc, meta in zip(docs, metas):
            if not doc or not meta: continue

            # Determine target dictionary based on language metadata
            lang = meta.get('language', 'en').lower()
            target = stats_fr if lang == 'fr' else stats_en

            target['chunks'] += 1
            tok_count = len(doc.split())
            target['total_tokens'] += tok_count
            target['token_list'].append(tok_count)

            # Track unique articles and clauses (ignoring '0')
            art_no = str(meta.get('article_number', '0'))
            cl_id = str(meta.get('clause_id', '0'))

            if art_no != '0': target['articles'].add(art_no)
            if cl_id != '0': target['clauses'].add(cl_id)

            # Track Appendices and schedules
            sec_title = str(meta.get('section_title', ''))
            if re.search(r'SCHEDULE|APPENDIX|ANNEXE|PARTIE', sec_title, re.IGNORECASE):
                target['appendices'] += 1

    def summarize(s):
        t_list = s['token_list']
        return {
            'articles': len(s['articles']),
            'clauses': len(s['clauses']),
            'chunks': s['chunks'],
            'avg_tokens': round(sum(t_list) / len(t_list)) if t_list else 0,
            'min_tokens': min(t_list) if t_list else 0,
            'max_tokens': max(t_list) if t_list else 0,
            'total_tokens': s['total_tokens'],
            'appendices': s['appendices']
        }

    return summarize(stats_en), summarize(stats_fr)


def generate_comparison_table(stats_en, stats_fr):
    data = [
        {
            'Property': 'Total articles',
            'English Version': stats_en['articles'],
            'French Version': stats_fr['articles']
        },
        {
            'Property': 'Total numbered clauses',
            'English Version': stats_en['clauses'],
            'French Version': stats_fr['clauses']
        },
        {
            'Property': 'Total chunks after clause-boundary segmentation',
            'English Version': stats_en['chunks'],
            'French Version': stats_fr['chunks']
        },
        {
            'Property': 'Average tokens per chunk',
            'English Version': stats_en['avg_tokens'],
            'French Version': stats_fr['avg_tokens']
        },
        {
            'Property': 'Minimum tokens per chunk',
            'English Version': stats_en['min_tokens'],
            'French Version': stats_fr['min_tokens']
        },
        {
            'Property': 'Maximum tokens per chunk',
            'English Version': stats_en['max_tokens'],
            'French Version': stats_fr['max_tokens']
        },
        {
            'Property': 'Total corpus tokens',
            'English Version': f"{stats_en['total_tokens']:,}",
            'French Version': f"{stats_fr['total_tokens']:,}"
        },
        {
            'Property': 'Appendices and schedules',
            'English Version': stats_en['appendices'],
            'French Version': stats_fr['appendices']
        }
    ]
    return pd.DataFrame(data)


def main():
    ap = argparse.ArgumentParser(description="Generate bilingual stats from PDFs or ChromaDB")
    ap.add_argument('--en-pdf', help='Path to the English PDF file')
    ap.add_argument('--fr-pdf', help='Path to the French PDF file')
    ap.add_argument('--db-path', default='./db/chroma_db',
                    help='Path to ChromaDB sqlite directory (used if PDFs are not provided)')
    ap.add_argument('--out', default='lufa_stats_bilingual.csv', help='Output CSV file path')
    args = ap.parse_args()

    if args.en_pdf or args.fr_pdf:
        print("Processing PDFs directly...")
        stats_en = get_pdf_stats(args.en_pdf, 'en')
        stats_fr = get_pdf_stats(args.fr_pdf, 'fr')
    else:
        stats_en, stats_fr = scan_chroma_db_bilingual(args.db_path)
        if stats_en is None:
            return

    # Generate the requested comparative DataFrame
    out_df = generate_comparison_table(stats_en, stats_fr)

    # Save to CSV
    out_df.to_csv(args.out, index=False)

    # Print the table to terminal matching the screenshot
    print("\n" + "=" * 80)
    print(out_df.to_string(index=False, justify='left'))
    print("=" * 80 + "\n")
    print(f"Saved comparative data to {args.out}")


if __name__ == '__main__':
    main()
    generate_system_table()