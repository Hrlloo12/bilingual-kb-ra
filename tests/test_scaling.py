from rag.config import load_serving_config
from rag.ingestion import CHUNKS_FILE_NAME, read_chunks
from rag.scaling import make_distractors

CHUNKS = read_chunks(load_serving_config().paths.corpus_processed / CHUNKS_FILE_NAME)


def test_distractors_are_deterministic_unique_and_marked_synthetic():
    first = make_distractors(CHUNKS, 600)
    second = make_distractors(CHUNKS, 600)
    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]
    assert len({chunk.chunk_id for chunk in first}) == 600
    assert all(chunk.chunk_id.startswith("synthetic_") and chunk.source.startswith("synthetic/") for chunk in first)
    real_ids = {chunk.chunk_id for chunk in CHUNKS}
    assert not real_ids & {chunk.chunk_id for chunk in first}


def test_distractors_change_names_and_numbers_but_keep_language():
    base = next(chunk for chunk in CHUNKS if "QH-" in chunk.text)
    index = CHUNKS.index(base)
    copy = make_distractors(CHUNKS, index + 1)[index]
    assert copy.text != base.text
    assert copy.language == base.language
