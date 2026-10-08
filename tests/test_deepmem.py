import json
import os
import tempfile

from deepmem import DeepMemoryIndex, entry_hash, entry_text, sidecar_path_for


def _fact(topic, content, created="2026-01-01T00:00:00Z"):
    return {"topic": topic, "content": content, "created_at": created, "updated_at": created}


def test_entry_text_variants():
    assert entry_text(_fact("Alice", "Lives in Berlin"), "memories") == "Alice: Lives in Berlin"
    assert entry_text({"members": ["Alice", "Bob"], "relation": "Friends"}, "dynamics") == "Alice & Bob: Friends"
    assert entry_text({"title": "Tab War", "context": "Tabs vs spaces"}, "inside_jokes") == "Tab War: Tabs vs spaces"
    assert entry_text({"topic": "", "content": ""}, "unknown") == ""


def test_entry_hash_stability_and_edits():
    e = _fact("Alice", "Lives in Berlin")
    h = entry_hash(e, "memories")
    assert h == entry_hash(dict(e), "memories")
    assert h != entry_hash(dict(e, content="Moved to Munich"), "memories")
    assert h != entry_hash(dict(e), "inside_jokes")
    # created_at disambiguates duplicate texts archived at different times
    assert h != entry_hash(_fact("Alice", "Lives in Berlin", created="2026-02-01T00:00:00Z"), "memories")


def test_query_ranking_and_filters():
    with tempfile.TemporaryDirectory() as td:
        idx = DeepMemoryIndex(os.path.join(td, "emb.json"))
        e1, e2, e3 = _fact("Alpha", "first"), _fact("Beta", "second"), _fact("Gamma", "third")
        flat = [("memories", e1), ("memories", e2), ("memories", e3)]
        assert idx.sync(flat, lambda texts: [[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]]) == 3

        hits = idx.query([1.0, 0.0], flat, top_k=2, min_score=0.0)
        assert [h[2]["topic"] for h in hits] == ["Alpha", "Gamma"]
        assert hits[0][0] > hits[1][0]

        hits = idx.query([1.0, 0.0], flat, top_k=3, min_score=0.9)
        assert [h[2]["topic"] for h in hits] == ["Alpha"]

        # Zero-norm query is safe; entries without stored vectors are skipped
        assert idx.query([0.0, 0.0], flat) == []
        extended = flat + [("memories", _fact("Delta", "fourth"))]
        # Delta has no stored vector; Beta's score of 0.0 passes min_score=0.0
        assert [h[2]["topic"] for h in idx.query([1.0, 0.0], extended, min_score=0.0)] == ["Alpha", "Gamma", "Beta"]


def test_sync_adds_drops_and_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "emb.json")
        idx = DeepMemoryIndex(path)
        e1, e2 = _fact("A", "one"), _fact("B", "two")
        calls = []

        def embed(texts):
            calls.append(list(texts))
            return [[1.0, 0.0] for _ in texts]

        assert idx.sync([("memories", e1), ("memories", e2)], embed) == 2
        assert calls == [["A: one", "B: two"]]  # one call, all missing texts

        # Nothing missing -> 0 without an embed call
        assert idx.sync([("memories", e1), ("memories", e2)], embed) == 0
        assert len(calls) == 1

        # e2 removed -> its vector dropped (no new embeddings)
        assert idx.sync([("memories", e1)], lambda texts: [[9.0, 9.0]]) == 0
        assert len(calls) == 1

        # Editing e1 changes its hash -> re-embedded
        e1b = dict(e1, content="one edited")
        assert idx.sync([("memories", e1b)], embed) == 1
        assert calls[-1] == ["A: one edited"]

        # Roundtrip: a fresh instance loads the saved vectors (nothing missing)
        idx2 = DeepMemoryIndex(path)
        idx2.load()
        assert idx2.sync([("memories", e1b)], embed) == 0
        assert len(calls) == 2


def test_sync_rewrites_sidecar_only_when_dirty():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "emb.json")
        idx = DeepMemoryIndex(path)
        e1 = _fact("A", "one")
        calls = []

        def embed(texts):
            calls.append(list(texts))
            return [[1.0, 0.0] for _ in texts]

        assert idx.sync([("memories", e1)], embed) == 1
        assert os.path.exists(path)

        # Nothing missing, nothing dropped -> no embed call and NO rewrite
        os.unlink(path)
        assert idx.sync([("memories", e1)], embed) == 0
        assert calls == [["A: one"]]
        assert not os.path.exists(path)

        # A dropped entry dirties the cache -> persisted again
        assert idx.sync([], embed) == 0
        assert os.path.exists(path)


def test_sync_failure_keeps_state():
    with tempfile.TemporaryDirectory() as td:
        idx = DeepMemoryIndex(os.path.join(td, "emb.json"))
        e1, e2, e3 = _fact("A", "one"), _fact("B", "two"), _fact("C", "three")
        all_entries = [("memories", e1), ("memories", e2), ("memories", e3)]
        assert idx.sync([("memories", e1)], lambda texts: [[1.0, 0.0]]) == 1

        # embed_fn raising -> 0, old state kept
        assert idx.sync(all_entries, lambda texts: (_ for _ in ()).throw(RuntimeError("boom"))) == 0
        # wrong vector count (1 vector for 2 missing texts) -> 0, old state kept
        assert idx.sync(all_entries, lambda texts: [[1.0, 0.0]]) == 0

        # e1's vector survived both failures: only e2 and e3 are still missing
        calls = []

        def embed(texts):
            calls.append(list(texts))
            return [[1.0, 0.0] for _ in texts]

        assert idx.sync(all_entries, embed) == 2
        assert calls == [["B: two", "C: three"]]


def test_model_mismatch_and_malformed_tolerated():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "emb.json")
        e1 = _fact("A", "one")
        real_hash = entry_hash(e1, "memories")

        # Model mismatch: stored vectors are discarded -> nothing found, re-embedded on sync
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"model": "other-model", "vectors": {real_hash: [1.0, 0.0]}}, f)
        idx = DeepMemoryIndex(path, embed_model="google/gemini-embedding-2")
        idx.load()
        assert idx.query([1.0, 0.0], [("memories", e1)]) == []
        assert idx.sync([("memories", e1)], lambda texts: [[1.0, 0.0]]) == 1

        # Malformed file tolerated (cache discarded, no crash)
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json")
        idx2 = DeepMemoryIndex(path)
        idx2.load()
        assert idx2.sync([("memories", e1)], lambda texts: [[1.0, 0.0]]) == 1


def test_sidecar_path_derivation():
    # Default store keeps the historical sidecar name (existing caches stay valid)
    assert sidecar_path_for("pletykas-deepmemory.json") == "pletykas-deepmemory-embeddings.json"
    # Relocated store keeps its cache beside it
    assert sidecar_path_for("/data/store/deep.json") == "/data/store/deep-embeddings.json"
    # Extension-less and dotted directory names
    assert sidecar_path_for("deepmem") == "deepmem-embeddings.json"
    assert sidecar_path_for("dir.v2/deep") == "dir.v2/deep-embeddings.json"
    # Blank input falls back to the default store name
    assert sidecar_path_for("  ") == "pletykas-deepmemory-embeddings.json"


def test_identity_mismatch_discards_on_dim_or_base_change():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "emb.json")
        e1 = _fact("A", "one")
        real_hash = entry_hash(e1, "memories")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"model": "m", "dim": 1536, "base": "https://x/v1", "vectors": {real_hash: [1.0, 0.0]}},
                f,
            )

        # Matching identity -> cache kept (nothing re-embedded)
        idx = DeepMemoryIndex(path, embed_model="m", embed_dim=1536, embed_base="https://x/v1")
        idx.load()
        assert idx.count_vectors() == 1
        assert idx.sync([("memories", e1)], lambda texts: [[9.0, 9.0]]) == 0
        assert idx.query([1.0, 0.0], [("memories", e1)], min_score=0.0)

        # Dim-only change, model name unchanged -> discarded and re-embedded
        idx2 = DeepMemoryIndex(path, embed_model="m", embed_dim=0, embed_base="https://x/v1")
        idx2.load()
        assert idx2.count_vectors() == 0
        assert idx2.sync([("memories", e1)], lambda texts: [[1.0, 0.0]]) == 1
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert (saved["model"], saved["dim"], saved["base"]) == ("m", 0, "https://x/v1")

        # Base-only change -> discarded as well
        idx3 = DeepMemoryIndex(path, embed_model="m", embed_dim=0, embed_base="https://other/v1")
        idx3.load()
        assert idx3.count_vectors() == 0

        # Legacy sidecar without dim/base fields -> not trusted, discarded
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"model": "m", "vectors": {real_hash: [1.0, 0.0]}}, f)
        idx4 = DeepMemoryIndex(path, embed_model="m", embed_dim=0, embed_base="https://x/v1")
        idx4.load()
        assert idx4.count_vectors() == 0


def test_format_hits():
    idx = DeepMemoryIndex()
    assert idx.format_hits([]) == ""
    hits = [
        (0.9, "memories", _fact("Alice", "Lives in Berlin")),
        (0.8, "inside_jokes", {"title": "Tab War", "context": "Tabs vs spaces"}),
    ]
    out = idx.format_hits(hits)
    assert out.splitlines()[0] == "[Recalled Deep Memory]"
    assert "- (Alice): Lives in Berlin" in out
    assert "- (Tab War): Tabs vs spaces" in out
