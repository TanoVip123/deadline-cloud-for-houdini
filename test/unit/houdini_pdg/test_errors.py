# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""The text an artist reads for a refused graph."""

from __future__ import annotations


from deadline.houdini_pdg.errors import GraphRefusedError, artist_message, format_refusals
from deadline.houdini_pdg.validation import Refusal, RefusalScenario, Verdict


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


def test_the_node_error_carries_the_same_lines() -> None:
    """The node's error flag shows the same lines as the popup, under the scheduler's name."""
    # The node error ends with the same refusal lines, and adds no line of its own.
    verdict: Verdict = Verdict(refusals=(_refusal("a"), _refusal("b")))
    message: str = artist_message(GraphRefusedError(verdict))
    assert message.endswith("\n" + "\n".join(format_refusals(verdict.refusals).split("\n")[1:]))
    assert message.count("\n") == 2
