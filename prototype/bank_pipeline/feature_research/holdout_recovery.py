"""One scoped M3 recovery for the accepted Step 5B black-box execution."""
from shared.pinned_artifacts import sha256_file
from upstream_regeneration import records

BLACK = "c5_20261126_104038"
BLACK_SHA = "416ab0a324d07fc08d120c62b697810b69ca5a404c667625bafaa7c788e816c7"
FAILURE = "glass_t0xr25k0.txt"
# LF digests of the two reviewed replacements, and the only additions to the historical tree.
REPLACEMENTS = {
    "feature_research/holdout_producers.py": "d6ce978cf7d016e1091bccf4abffc7f5cad6fd49ace8d01dc1800695f408e191",
    "feature_research/stage4_holdout.py": "de30405cfce8bb1ed5504ab9440bbc154f52202bb3dd5c4979f6c5e95cdfa433",
}
ADDITIONS = ("feature_research/holdout_recovery.py", "tests/feature_research/test_pr43_ilp_recovery.py")


def original_code(root, current, black):
    if black["run"]["run_id"] != BLACK or black["content_sha256"] != BLACK_SHA:
        raise ValueError("Unexpected black-box recovery anchor")
    files = dict(current["files_sha256_lf"])
    old = black["code"]
    overrides = dict.fromkeys(ADDITIONS)
    for path, expected in REPLACEMENTS.items():
        if files[path] != expected:
            raise ValueError(f"Unreviewed recovery source: {path}")
        files[path] = overrides[path] = old["files_sha256_lf"][path]
    tree = records.code_identity(root, overrides=overrides)
    if (files != old["files_sha256_lf"] or tree["code_sha256"] != old["tree"]["code_sha256"]
            or tree["files"] != old["tree"]["files"] or tree["head"] != old["head"]):
        raise ValueError("Cannot reconstruct the accepted pre-M3 code identity")
    return files, old["tree"]


def failure_identity(cache):
    path = cache / "failures" / FAILURE
    message = path.read_text(encoding="utf8")
    if "Router 0 requires review:" not in message or "'pass2': 'valid_unproven_undocumented'" not in message:
        raise ValueError("Unexpected original GLASS router failure")
    return {"file": FAILURE, "sha256": sha256_file(path)}
