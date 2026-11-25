"""
PR 42 Commit 6, Step 6A — feature-contract recommendations and research handoff.

``research_logs/pr42/feature_contract_recommendations.md`` consolidates the
accepted PR 42 records into dispositions and handoffs. These tests hold it to
those records:

- every accepted run it names passes its integrity check (sidecar and content
  digest), states its own content digest and registration, and is the first run
  that wrote its record; the latest accepted run reproduces every earlier one;
- every row of its evidence index resolves, in the named accepted record, to the
  value it states (at the precision written), and every citation names a row;
- the production feature code is the code every accepted run recorded, and every
  code-computable frozen reference is the one the latest accepted run recorded,
  so Step 6A changes no feature code, contract or identity;
- the record index, the notebook map and the notebook point to the document,
  and the notebook stays output-free with its 68 code cells.

The records are tracked byte for byte (``research_logs/pr42/.gitattributes``), so
nothing here needs a local artifact, the database or a test row.
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path

import pytest

from feature_research import frozen_references as FR
from feature_research import research_records as RR

BANK = Path(__file__).resolve().parents[2]
DOC = BANK / "research_logs" / "pr42" / "feature_contract_recommendations.md"
NOTEBOOK = BANK / "feature_research" / "feature_research.ipynb"

#: The accepted run that holds all seven records (Step 5B).
LATEST = "c5_20261124_122029"
RECORDS = ("feature_inventory", "encoding_audit", "temporal_validity", "binning_drift",
           "binning_candidates", "stage3_simplification", "interaction_selection")

#: Production feature code: Stage 1-3 feature engineering, the Stage 2 binning, the
#: Stage 3 DAG and block identity, and the EBM interaction definition.
PRODUCTION_FEATURE_CODE = (
    "shared/stage1/feature_engineering.py",
    "shared/stage2/feature_engineering.py",
    "shared/stage2/binning.py",
    "shared/stage3/feature_engineering.py",
    "shared/stage3/data.py",
    "shared/stage3/block_cache.py",
    "feature_research/config.py",
    "feature_research/feature_engineering/__init__.py",
    "feature_research/feature_engineering/_checks.py",
    "feature_research/feature_engineering/crisis.py",
    "feature_research/feature_engineering/integrals.py",
    "feature_research/feature_engineering/derivatives.py",
    "feature_research/feature_engineering/temporal.py",
    "feature_research/feature_engineering/prior.py",
    "feature_research/feature_engineering/overlap.py",
    "feature_research/feature_engineering/pipeline.py",
    "glass_pipeline/ebm/interactions.py",
    "blackbox_pipeline/models/mlp/features.py",
    "stage2_crossfit/inner_features.py",
)


# ======================================================================
# reading the document
# ======================================================================
def _text() -> str:
    return DOC.read_text(encoding="utf-8")


def _block(name: str) -> list:
    """The table rows between ``<!-- BEGIN name -->`` and ``<!-- END name -->`` (header removed)."""
    text = _text()
    begin, end = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
    assert text.count(begin) == 1 and text.count(end) == 1, name
    lines = [ln for ln in text.split(begin, 1)[1].split(end, 1)[0].splitlines() if ln.strip()]
    assert lines[1].replace("|", "").replace("-", "").strip() == "", "a table header and separator"
    return [_cells(ln) for ln in lines[2:]]


def _cells(line: str) -> list:
    """One table row's cells: split on unescaped pipes, backticks removed, ``\\|`` unescaped."""
    parts = re.split(r"(?<!\\)\|", line.strip())[1:-1]
    return [p.strip().replace("`", "").replace("\\|", "|") for p in parts]


@lru_cache(maxsize=None)
def _record(run_id: str, name: str) -> dict:
    return RR.read_run_record(run_id, name)          # checks the sidecar and the content digest


def _accepted() -> list:
    return [dict(zip(("step", "run", "record", "content", "registration", "review"), row))
            for row in _block("ACCEPTED RUNS")]


def _index() -> list:
    return [dict(zip(("id", "claim", "record", "key", "value"), row))
            for row in _block("EVIDENCE INDEX")]


# ======================================================================
# resolving an evidence key
# ======================================================================
AGGREGATES = {"all": lambda v: all(v), "any": lambda v: any(v), "len": len, "sum": sum,
              "max": max, "min": min, "unique": lambda v: sorted(set(v))}


def _select(obj, seg: str):
    if seg.startswith("[") and seg.endswith("]"):
        many = seg.startswith("[?")
        conditions = [c.split("=", 1) for c in seg[2 if many else 1:-1].split(";")]
        hits = [e for e in obj if isinstance(e, dict)
                and all(k in e and str(e[k]) == v for k, v in conditions)]
        if many:
            return hits
        if len(hits) != 1:
            raise LookupError(f"{seg}: {len(hits)} elements match, not one")
        return hits[0]
    if isinstance(obj, list):
        return obj[int(seg)]
    return obj[seg]


def resolve(obj, key: str):
    """The value at ``key`` (the evidence index's path rules, document section 10)."""
    base, *ops = key.split("/@")

    def walk(o, segs):
        if not segs:
            return o
        if segs[0] == "*":
            return [walk(e, segs[1:]) for e in o]
        return walk(_select(o, segs[0]), segs[1:])

    out = walk(obj, base.split("/") if base else [])
    for op in ops:
        out = AGGREGATES[op](out)
    return out


def matches(actual, stated: str) -> bool:
    """Whether a record value is the stated one: a prefix, a literal, or a number at its precision."""
    s = stated.strip()
    if s.endswith("…"):
        return isinstance(actual, str) and len(s) > 9 and actual.startswith(s[:-1])
    if s in ("true", "false"):
        return actual is (s == "true")
    if s[:1] in "[{\"":
        return actual == json.loads(s)
    try:
        number = Decimal(s)
    except InvalidOperation:
        return actual == s
    if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not number.is_finite():
        return False
    quantum = Decimal(1).scaleb(number.as_tuple().exponent)
    return abs(Decimal(repr(float(actual))) - number) <= quantum / 2 + Decimal("1e-15")


def test_the_resolver_and_matcher_follow_the_documented_rules():
    rows = [{"a": "x", "b": 1, "v": 0.12345, "ok": True}, {"a": "y & z", "b": 2, "v": 2.5e-17, "ok": True}]
    obj = {"t": rows, "s": "0817825b9eecbd11"}
    assert resolve(obj, "t/[a=x;b=1]/v") == 0.12345
    assert resolve(obj, "t/*/ok/@all") is True and resolve(obj, "t/@len") == 2
    assert resolve(obj, "t/[?a=y & z]/*/b/@sum") == 2 and resolve(obj, "t/*/a/@unique") == ["x", "y & z"]
    with pytest.raises(LookupError):
        resolve(obj, "t/[ok=True]/v")                 # two matches where one is required
    assert matches(0.12345, "0.123") and not matches(0.12345, "0.124") and matches(0.12345, "0.12")
    assert matches(-0.0089999, "-0.0090") and matches(17.0, "17") and matches(2.5e-17, "2.5e-17")
    assert not matches(True, "1") and not matches("1", "1.0") and not matches(0.1, "nan")
    assert matches("0817825b9eecbd11", "0817825b9ee…") and not matches("0817825b9eecbd11", "08…")
    assert matches(["True"], '["True"]') and matches("misses", "misses") and matches(False, "false")


# ======================================================================
# the accepted runs
# ======================================================================
def test_every_accepted_run_is_intact_and_states_its_digest_and_registration():
    accepted = _accepted()
    assert [a["record"] for a in accepted] == list(RECORDS)
    for a in accepted:
        rec = _record(a["run"], a["record"])
        assert rec["run"]["run_id"] == a["run"]
        assert matches(rec["content_sha256"], a["content"]), a
        assert rec["environment"]["os"] == "Windows", a          # the accepted runs are the Windows runs
        if a["registration"] == "—":
            assert "registration" not in rec, a
            continue
        reg_id, version, digest = a["registration"].split()
        content = rec["registration"]["content"]
        assert (content["id"], content["version"]) == (reg_id, version), a
        assert matches(rec["registration"]["sha256"], digest), a


def test_each_accepted_run_is_the_first_run_that_wrote_its_record():
    root = RR.RECORD_ROOT
    runs = sorted((d for d in root.iterdir() if d.is_dir() and RR.RUN_ID.match(d.name)),
                  key=lambda d: d.name.split("_", 1)[1])
    for a in _accepted():
        first = next(d.name for d in runs if (d / f"{a['record']}.json").is_file())
        assert first == a["run"], a


def test_the_latest_accepted_run_reproduces_every_earlier_accepted_record():
    accepted = {a["record"]: a["run"] for a in _accepted()}
    assert accepted["interaction_selection"] == LATEST
    for name, run in accepted.items():
        latest = _record(LATEST, name)
        if name == "feature_inventory":
            # The Commit 2 inventory differs only in its callers: later PR 42 modules import the
            # feature-engineering modules. The Step 3A run's inventory is the one reproduced.
            rows = RR.compare_records(latest, _record(run, name))
            assert {r["section"] for r in rows if not r["same"]} == {"duplicates"}
            assert {r["part"] for r in rows if not r["same"] and r["part"]} == {"callers"}
            assert latest["content_sha256"] == _record(accepted["encoding_audit"], name)["content_sha256"]
            continue
        assert latest["content_sha256"] == _record(run, name)["content_sha256"], name


# ======================================================================
# the evidence index
# ======================================================================
def test_the_evidence_index_ids_are_unique_and_every_citation_names_a_row():
    rows = _index()
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)) and len(ids) > 100
    accepted_runs = {a["run"] for a in _accepted()}
    for r in rows:
        run, name = r["record"].split("/")
        assert run in accepted_runs and name in RECORDS, r
    body = _text().split("<!-- BEGIN EVIDENCE INDEX -->")[0]
    cited = set()
    for group in re.findall(r"\[(S\d[^\]]*)\]", body):
        for part in group.split(","):
            ends = [e.strip() for e in part.split("–")]
            assert all(re.fullmatch(r"S\d+\.\d+[a-z]?", e) for e in ends), group
            cited.update(ends)
    assert cited and not cited - set(ids), sorted(cited - set(ids))


@pytest.mark.parametrize("row", _index(), ids=lambda r: r["id"])
def test_each_evidence_row_holds_in_its_accepted_record(row):
    run, name = row["record"].split("/")
    actual = resolve(_record(run, name), row["key"])
    assert matches(actual, row["value"]), (row["id"], actual, row["value"])


# ======================================================================
# no feature code, contract or identity changed
# ======================================================================
def _lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_the_production_feature_code_is_the_code_every_accepted_run_recorded():
    assert set(PRODUCTION_FEATURE_CODE) <= set(RR.CODE_FILES)
    now = {f: _lf_sha256(BANK / f) for f in PRODUCTION_FEATURE_CODE}
    for a in _accepted():
        recorded = _record(a["run"], a["record"])["code"]["files_sha256_lf"]
        changed = sorted(f for f in PRODUCTION_FEATURE_CODE if recorded.get(f) != now[f])
        assert changed == [], (a["run"], changed)


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def test_the_code_computable_frozen_references_are_the_ones_the_accepted_run_recorded():
    recorded = {r["name"]: r for r in _record(LATEST, "feature_inventory")["frozen_references"]}
    # As in ``frozen_references``, the population is recorded from the training rows, not statically.
    computed = [r for r in FR.static_references() if r["name"] != "Train-side population"]
    assert len(computed) >= 15
    for ref in computed:
        rec = recorded[ref["name"]]
        for key in ("computed", "recorded", "status"):
            assert _canonical(ref[key]) == _canonical(rec[key]), (ref["name"], key)
    assert {r["status"] for r in computed} >= {"verified"}
    assert all(r["status"] != "MISMATCH" for r in computed)


# ======================================================================
# where the document is read from
# ======================================================================
def test_the_record_index_and_the_notebook_map_point_to_the_document():
    readme = (BANK / "research_logs" / "pr42" / "README.md").read_text(encoding="utf-8")
    assert "[`feature_contract_recommendations.md`](feature_contract_recommendations.md)" in readme
    nmap = (BANK / "documentation" / "notebook_map.md").read_text(encoding="utf-8")
    section5 = nmap.split("## 5. Deferred utility work", 1)[1]
    row = next(ln for ln in section5.splitlines() if ln.startswith("| Feature-engineering copies"))
    assert "feature_contract_recommendations.md" in row and "PR 45" in row


def test_the_notebook_concludes_with_the_document_and_stays_output_free():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in code)
    headers = [re.findall(r"^# CELL (\S+?):", "".join(c["source"]), re.M) for c in code]
    assert [h[0] for h in headers] == [str(i) for i in range(1, 69)]      # Step 6A adds no code cell
    last = "".join(nb["cells"][-1]["source"])
    assert nb["cells"][-1]["cell_type"] == "markdown"
    assert last.startswith("## PR 42 conclusion") and "feature_contract_recommendations.md" in last
    assert "feature_contract_recommendations.md" in "".join(nb["cells"][0]["source"])
