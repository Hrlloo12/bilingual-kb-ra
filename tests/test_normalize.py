from rag.normalize import (
    count_script_letters,
    extract_numbers,
    fold_digits,
    normalize_display,
    normalize_for_dense,
    normalize_for_search,
    to_eastern_digits,
)


def test_fold_digits_handles_eastern_and_persian_digits():
    assert fold_digits("١٤ يومًا و۲۴ ساعة") == "14 يومًا و24 ساعة"


def test_fold_digits_handles_arabic_separators():
    assert fold_digits("١٨٬٥٠٠ و٣٫٥") == "18,500 و3.5"


def test_to_eastern_digits_round_trip():
    assert fold_digits(to_eastern_digits("QH-BR-210 7,800")) == "QH-BR-210 7,800"


def test_search_normalization_removes_diacritics_tatweel_and_letter_variants():
    assert normalize_for_search("سِيَاسَةُ الاسـتـرجاع") == "سياسه الاسترجاع"
    assert normalize_for_search("أإآٱ مستشفى") == "اااا مستشفي"


def test_search_normalization_lowercases_latin_and_folds_digits():
    assert normalize_for_search("Premium SLA ١ ساعة") == "premium sla 1 ساعه"


def test_dense_normalization_keeps_letter_variants():
    assert normalize_for_dense("الطلبات المتأخرة ١٠ أيام") == "الطلبات المتأخرة 10 أيام"


def test_display_normalization_converts_presentation_forms_and_strips_bidi_marks():
    assert normalize_display("ﻷ‏‫سلام‬") == "لأسلام"


def test_display_normalization_preserves_paragraphs():
    assert normalize_display("line one  \n\n\n\n  line   two") == "line one\n\nline two"


def test_extract_numbers_handles_thousands_times_and_codes():
    assert extract_numbers("18,500 SAR from 8:00, code QH-OC-022, ١٬١٥٠") == [18500.0, 8.0, 0.0, 22.0, 1150.0]


def test_count_script_letters():
    assert count_script_letters("أبغى أعرف الـ pricing") == (10, 7)
