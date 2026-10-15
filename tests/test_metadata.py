"""Scope filters: parsing, normalisation, and matching semantics."""

from __future__ import annotations

import pytest

from app.core.metadata import (
    Condition,
    InvalidFilterError,
    ScopeFilter,
    collect_facets,
    normalize_metadata,
)


# --------------------------------------------------------------- normalising


def test_metadata_is_normalized_to_lowercase():
    assert normalize_metadata({"Course": "CS1428"}) == {"course": "cs1428"}


def test_metadata_collapses_whitespace():
    assert normalize_metadata({"  the   course ": " Fall  2025 "}) == {
        "the course": "fall 2025"
    }


def test_metadata_accepts_non_string_values():
    """A user tagging {"year": 2025} means something obvious; don't reject it."""
    assert normalize_metadata({"year": 2025}) == {"year": "2025"}


def test_metadata_renders_booleans_lowercase():
    assert normalize_metadata({"archived": True}) == {"archived": "true"}


def test_metadata_rejects_list_values():
    with pytest.raises(InvalidFilterError):
        normalize_metadata({"course": ["CS1428", "CS2308"]})


def test_metadata_rejects_overlong_keys():
    with pytest.raises(InvalidFilterError):
        normalize_metadata({"k" * 100: "v"})


def test_metadata_rejects_overlong_values():
    with pytest.raises(InvalidFilterError):
        normalize_metadata({"k": "v" * 500})


def test_metadata_rejects_too_many_keys():
    with pytest.raises(InvalidFilterError):
        normalize_metadata({"k{}".format(i): "v" for i in range(100)})


def test_empty_metadata_is_an_empty_dict():
    assert normalize_metadata(None) == {}
    assert normalize_metadata({}) == {}


def test_blank_keys_are_dropped():
    assert normalize_metadata({"  ": "value", "ok": "yes"}) == {"ok": "yes"}


# ------------------------------------------------------------------ parsing


def test_bare_value_is_an_exact_match():
    f = ScopeFilter.parse({"course": "CS1428"})
    assert f.matches("d1", {"course": "cs1428"})
    assert not f.matches("d1", {"course": "cs2308"})


def test_list_value_matches_any():
    f = ScopeFilter.parse({"course": ["CS1428", "CS2308"]})
    assert f.matches("d1", {"course": "cs1428"})
    assert f.matches("d1", {"course": "cs2308"})
    assert not f.matches("d1", {"course": "cs3358"})


def test_in_operator_matches_any():
    f = ScopeFilter.parse({"course": {"in": ["CS1428", "CS2308"]}})
    assert f.matches("d1", {"course": "cs2308"})
    assert not f.matches("d1", {"course": "cs3358"})


def test_not_operator_excludes():
    f = ScopeFilter.parse({"course": {"not": "CS1428"}})
    assert not f.matches("d1", {"course": "cs1428"})
    assert f.matches("d1", {"course": "cs2308"})


def test_not_in_operator_excludes_several():
    f = ScopeFilter.parse({"course": {"not_in": ["CS1428", "CS2308"]}})
    assert not f.matches("d1", {"course": "cs1428"})
    assert f.matches("d1", {"course": "cs3358"})


def test_conditions_are_anded_across_keys():
    f = ScopeFilter.parse({"course": "CS1428", "term": "Fall2025"})
    assert f.matches("d1", {"course": "cs1428", "term": "fall2025"})
    assert not f.matches("d1", {"course": "cs1428", "term": "spring2026"})


def test_missing_key_fails_a_positive_condition():
    """Unknown keys must not silently widen the scope."""
    f = ScopeFilter.parse({"course": "CS1428"})
    assert not f.matches("d1", {"term": "fall2025"})
    assert not f.matches("d1", {})


def test_missing_key_satisfies_a_negated_condition():
    """A document with no course is indeed 'not course=CS1428'."""
    f = ScopeFilter.parse({"course": {"not": "CS1428"}})
    assert f.matches("d1", {})


def test_filter_matching_is_case_insensitive():
    f = ScopeFilter.parse({"COURSE": "cs1428"})
    assert f.matches("d1", {"course": "cs1428"})


def test_empty_filter_matches_everything():
    f = ScopeFilter.parse()
    assert f.is_empty
    assert f.matches("anything", {})


def test_doc_ids_restrict_matching():
    f = ScopeFilter.parse(doc_ids=["a", "b"])
    assert f.matches("a", {})
    assert not f.matches("c", {})


def test_doc_ids_combine_with_conditions():
    f = ScopeFilter.parse({"course": "CS1428"}, doc_ids=["a"])
    assert f.matches("a", {"course": "cs1428"})
    assert not f.matches("b", {"course": "cs1428"})
    assert not f.matches("a", {"course": "cs2308"})


# ------------------------------------------------------------------- errors


def test_unknown_operator_is_rejected():
    with pytest.raises(InvalidFilterError):
        ScopeFilter.parse({"course": {"startswith": "CS"}})


def test_multiple_operators_are_rejected():
    with pytest.raises(InvalidFilterError):
        ScopeFilter.parse({"course": {"in": ["a"], "not": "b"}})


def test_empty_operator_object_is_rejected():
    with pytest.raises(InvalidFilterError):
        ScopeFilter.parse({"course": {}})


def test_empty_value_list_is_rejected():
    with pytest.raises(InvalidFilterError):
        ScopeFilter.parse({"course": []})


def test_blank_filter_key_is_rejected():
    with pytest.raises(InvalidFilterError):
        ScopeFilter.parse({"   ": "value"})


def test_non_mapping_filter_is_rejected():
    with pytest.raises(InvalidFilterError):
        ScopeFilter.parse(["course", "CS1428"])


# ---------------------------------------------------------------- describing


def test_describe_reads_as_prose():
    f = ScopeFilter.parse({"course": "CS1428"})
    assert f.describe() == "course is cs1428"


def test_describe_negation():
    f = ScopeFilter.parse({"course": {"not": "CS1428"}})
    assert "is not" in f.describe()


def test_describe_empty_filter():
    assert ScopeFilter.parse().describe() == "the whole knowledge base"


def test_describe_mentions_doc_id_restriction():
    f = ScopeFilter.parse(doc_ids=["a", "b"])
    assert "2 selected document" in f.describe()


# ------------------------------------------------------------------- facets


def test_collect_facets_counts_values():
    facets = collect_facets(
        [
            {"course": "cs1428", "term": "fall2025"},
            {"course": "cs1428", "term": "spring2026"},
            {"course": "cs2308"},
        ]
    )
    assert facets["course"] == {"cs1428": 2, "cs2308": 1}
    assert facets["term"] == {"fall2025": 1, "spring2026": 1}


def test_collect_facets_on_untagged_corpus_is_empty():
    assert collect_facets([{}, {}]) == {}


def test_condition_is_hashable():
    """Frozen so a filter can be cached or de-duplicated if that is ever useful."""
    c = Condition(key="course", values=frozenset({"cs1428"}))
    assert {c, c} == {c}
