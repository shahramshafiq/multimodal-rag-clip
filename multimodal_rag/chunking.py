"""Step 2: cut the text into chunks that fit CLIP's limit of 77 tokens.

CLIP silently cuts longer text, so every chunk is measured with CLIP's own tokenizer. A chunk is built from whole
sentences, and only a single sentence that is too long is cut between words.
"""
import re
from dataclasses import dataclass

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class TextChunk:
    chunk_id: str
    doc: str
    page: int
    text: str


def _split_long_sentence(sentence, count_tokens, max_tokens):
    pieces, current = [], []
    for word in sentence.split():
        if current and count_tokens(" ".join(current + [word])) > max_tokens:
            pieces.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_block(text, count_tokens, max_tokens):
    """Cut one block of text into pieces of at most max_tokens tokens."""
    pieces = []
    for sentence in SENTENCE_END.split(text):
        if count_tokens(sentence) > max_tokens:
            pieces.extend(_split_long_sentence(sentence, count_tokens, max_tokens))
        else:
            pieces.append(sentence)

    chunks, current = [], ""
    for piece in pieces:
        joined = f"{current} {piece}".strip()
        if current and count_tokens(joined) > max_tokens:
            chunks.append(current)
            current = piece
        else:
            current = joined
    if current:
        chunks.append(current)
    return chunks


def chunk_blocks(blocks, count_tokens, max_tokens):
    chunks = []
    for block in blocks:
        for text in chunk_block(block.text, count_tokens, max_tokens):
            chunks.append(TextChunk(f"{block.doc}_p{block.page}_{len(chunks)}", block.doc, block.page, text))
    return chunks
