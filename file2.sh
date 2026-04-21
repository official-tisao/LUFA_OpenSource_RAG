    "test_reflector.py:deleted" "CLAUDE.md:created" "combine_corpus.py:created"
    "src/api.py:created" "src/translator.py:created" "src/clause_chunker.py:created"
)

is_text() { [[ "$1" =~ \.(py|md|csv|txt|sh|html|css|js|json)$ ]] && return 0 || return 1; }

for ENTRY in "${FILES[@]}"; do
    FILE_PATH="${ENTRY%%:*}"
    OPERATION="${ENTRY##*:}"
    B_NAME=$(basename "$FILE_PATH")

    if [ "$OPERATION" == "deleted" ]; then
        rm -f "$FILE_PATH"
        git add "$FILE_PATH"
        export GIT_AUTHOR_DATE="$(date -d "@$CUR_SEC" +"%Y-%m-%d 10:00:00")"
        export GIT_COMMITTER_DATE="$GIT_AUTHOR_DATE"
        git commit -m "$OPERATION $B_NAME"
    elif is_text "$FILE_PATH"; then
        LINES=$(wc -l < "$FILE_PATH")
        [ "$LINES" -lt 450 ] && CHUNKS=3 || CHUNKS=6
