def split_text(text, max_tokens):
    words = text.split()
    chunks = []
    current_chunk = []
    current_token_count = 0

    for word in words:
        token_count = len(word.split())

        if current_token_count + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_token_count = token_count
        else:
            current_chunk.append(word)
            current_token_count += token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks