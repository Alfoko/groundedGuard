from groundguard.chunking import Chunk
from groundguard.citations import CitationStatus, check_citations
from groundguard.generation import Answer, Claim, StructuredAnswer


def chunk(cid: str) -> Chunk:
    return Chunk(id=cid, source=f"{cid}.md", section=cid, start=0, end=4, text="text")


SOURCES = [chunk("a"), chunk("b"), chunk("c")]


def answer_with(*claims: Claim) -> Answer:
    return Answer("q", SOURCES, StructuredAnswer(answerable=True, claims=list(claims)), [])


def test_valid_citations_resolve_to_chunks_in_order() -> None:
    [checked] = check_citations(answer_with(Claim(text=" Pods are small. ", sources=[3, 1])))
    assert checked.status is CitationStatus.OK
    assert [c.id for c in checked.evidence] == ["c", "a"]
    assert checked.text == "Pods are small."
    assert checked.invalid == []


def test_claim_without_citation_is_flagged() -> None:
    [checked] = check_citations(answer_with(Claim(text="Unsourced.", sources=[])))
    assert checked.status is CitationStatus.NO_CITATION
    assert not checked.citations_ok


def test_out_of_range_numbers_are_invalid() -> None:
    [checked] = check_citations(answer_with(Claim(text="Made up.", sources=[0, 4, 4])))
    assert checked.status is CitationStatus.INVALID_ONLY
    assert checked.invalid == [0, 4]
    assert checked.evidence == []


def test_mixed_citations_keep_valid_evidence() -> None:
    [checked] = check_citations(answer_with(Claim(text="Partly sourced.", sources=[2, 9])))
    assert checked.citations_ok
    assert [c.id for c in checked.evidence] == ["b"]
    assert checked.invalid == [9]


def test_duplicate_citations_are_collapsed() -> None:
    [checked] = check_citations(answer_with(Claim(text="Twice.", sources=[1, 1, 2])))
    assert [c.id for c in checked.evidence] == ["a", "b"]
    assert checked.cited == [1, 1, 2]


def test_claims_keep_their_position() -> None:
    claims = check_citations(
        answer_with(Claim(text="First.", sources=[1]), Claim(text="Second.", sources=[]))
    )
    assert [(c.index, c.status) for c in claims] == [
        (0, CitationStatus.OK),
        (1, CitationStatus.NO_CITATION),
    ]


def test_unanswerable_answer_has_no_claims() -> None:
    answer = Answer("q", SOURCES, StructuredAnswer(answerable=False), [])
    assert check_citations(answer) == []
