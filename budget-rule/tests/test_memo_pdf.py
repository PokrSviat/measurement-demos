from pathlib import Path

from budget_rule.memo_pdf import write_memo_pdf


def test_write_memo_pdf_keeps_title_and_headings(tmp_path: Path) -> None:
    text = "Title line\n\nEstimand\nThe estimand.\n\nRecommendation\nHold."
    path = tmp_path / "memo.pdf"
    write_memo_pdf(text, path)
    payload = path.read_bytes()
    assert payload.startswith(b"%PDF")
    assert path.stat().st_size > 0
