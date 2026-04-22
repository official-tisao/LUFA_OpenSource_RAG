        split -n l/"$CHUNKS" "$FILE_PATH" .tmp_
        for CHUNK in .tmp_*; do
            mv "$CHUNK" "$FILE_PATH"
            export GIT_AUTHOR_DATE="$(date -d "@$CUR_SEC" +"%Y-%m-%d 10:00:00")"
            export GIT_COMMITTER_DATE="$GIT_AUTHOR_DATE"
            git add "$FILE_PATH"
            git commit -m "$OPERATION $B_NAME"
            CUR_SEC=$(( CUR_SEC + 86400 ))
        done
        rm -f .tmp_*
    else
        export GIT_AUTHOR_DATE="$(date -d "@$CUR_SEC" +"%Y-%m-%d 10:00:00")"
        export GIT_COMMITTER_DATE="$GIT_AUTHOR_DATE"
        git add "$FILE_PATH"
        git commit -m "$OPERATION $B_NAME"
    fi
    CUR_SEC=$(( CUR_SEC + 86400 ))
    [ "$CUR_SEC" -gt "$END_SEC" ] && CUR_SEC=$END_SEC
done