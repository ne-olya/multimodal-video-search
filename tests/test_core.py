import numpy as np

from videoscope.encoders import HashTextEncoder
from videoscope.evaluate import ranking_metrics
from videoscope.index import VectorIndex, VideoRecord, fuse_embeddings
from videoscope.sampling import uniform_positions
from videoscope.retrieval import MultimodalSegmentIndex, SegmentRecord


def test_uniform_positions_include_bounds():
    assert uniform_positions(100, 3).tolist() == [0, 49, 99]


def test_index_roundtrip_and_search(tmp_path):
    records = [VideoRecord("a", "a.mp4", 1, []), VideoRecord("b", "b.mp4", 1, [])]
    index = VectorIndex(np.eye(2), records)
    index.save(tmp_path)
    found = VectorIndex.load(tmp_path).search([1, 0], 1)
    assert found[0][0].video_id == "a"


def test_fusion_renormalizes_missing_modalities():
    result = fuse_embeddings(np.array([1.0, 0.0]), speech=np.array([0.0, 1.0]))
    assert np.isclose(np.linalg.norm(result), 1)


def test_metrics_and_hash_encoder():
    assert ranking_metrics(["x", "y"], {"y"}, 2) == {"recall@2": 1, "precision@2": 0.5, "mrr": 0.5}
    assert HashTextEncoder(8).encode_texts(["один запрос"]).shape == (1, 8)


def test_segment_retrieval_exposes_modality_contributions():
    records = [
        SegmentRecord("a:0", "a", "a.mp4", 0, 10, "a.jpg"),
        SegmentRecord("b:0", "b", "b.mp4", 0, 10, "b.jpg", "speech"),
    ]
    vectors = {"visual": np.eye(2), "speech": np.array([[0, 0], [1, 0]]), "ocr": np.zeros((2, 2))}
    available = {"visual": [True, True], "speech": [False, True], "ocr": [False, False]}
    result = MultimodalSegmentIndex(records, vectors, available).search_videos([1, 0], 2)
    assert "speech" in result[0]["contributions"] or "speech" in result[1]["contributions"]
    assert {row["record"].video_id for row in result} == {"a", "b"}
