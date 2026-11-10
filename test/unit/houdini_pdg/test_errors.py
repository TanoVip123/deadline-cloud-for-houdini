# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""The text an artist reads for a refused graph."""

from __future__ import annotations

from typing import List, Tuple

from hypothesis import given
from hypothesis import strategies as st

from deadline.houdini_pdg.errors import GraphRefusedError, artist_message, format_refusals
from deadline.houdini_pdg.validation import Refusal, RefusalScenario, Verdict

# One line of text with no line break, for a subject, detail or remedy.
_LINE: st.SearchStrategy[str] = st.text(
    alphabet=st.characters(blacklist_categories=("Cc", "Cs", "Zl", "Zp")), min_size=1, max_size=20
)


def _refusal(subject: str, detail: str = "is wrong.", remedy: str = "Fix it.") -> Refusal:
    """A refusal with the given text."""
    return Refusal(
        scenario=RefusalScenario.ROP_BATCH, subject=subject, detail=detail, remedy=remedy
    )


def test_each_refusal_is_on_its_own_line_after_a_header() -> None:
    """Each refusal is a "- " bullet on its own line, so the artist can tell where one ends."""
    verdict: Verdict = Verdict(refusals=(_refusal("a", "x.", "y."), _refusal("b", "p.", "q.")))
    assert str(GraphRefusedError(verdict)).split("\n") == [
        "The graph was refused (2 refusal(s)):",
        "- a: x. y.",
        "- b: p. q.",
    ]


@given(st.lists(st.tuples(_LINE, _LINE, _LINE), min_size=1, max_size=8))
def test_n_refusals_give_a_header_and_n_lines_each_starting_with_its_subject(
    texts: List[Tuple[str, str, str]],
) -> None:
    """For any refusals, the text has one header and one bullet line per refusal, in order."""
    # Build the refusals, then split the text and match each line to its subject.
    refusals: Tuple[Refusal, ...] = tuple(_refusal(s, d, r) for s, d, r in texts)
    lines: List[str] = format_refusals(refusals).split("\n")
    assert len(lines) == len(refusals) + 1
    assert all(line.startswith(f"- {s}: ") for line, (s, _, _) in zip(lines[1:], texts))


def test_the_node_error_carries_the_same_lines() -> None:
    """The node's error flag shows the same lines as the popup, under the scheduler's name."""
    # The node error ends with the same refusal lines, and adds no line of its own.
    verdict: Verdict = Verdict(refusals=(_refusal("a"), _refusal("b")))
    message: str = artist_message(GraphRefusedError(verdict))
    assert message.endswith("\n" + "\n".join(format_refusals(verdict.refusals).split("\n")[1:]))
    assert message.count("\n") == 2
