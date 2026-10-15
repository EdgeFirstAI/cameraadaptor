import numpy as np

from edgefirst.cameraadaptor.tiling import nms_multilabel

BOX = [0.0, 0.0, 10.0, 10.0]


def test_one_box_yields_one_detection_per_class_above_the_floor():
    b, s, c = nms_multilabel([BOX], [[0.9, 0.001, 0.4]], score_threshold=0.001)
    assert c.tolist() == [0, 2]
    np.testing.assert_allclose(s, [0.9, 0.4])
    np.testing.assert_allclose(b, [BOX, BOX])


def test_suppression_is_class_aware_unless_agnostic():
    boxes = [BOX, [1.0, 0.0, 11.0, 10.0]]  # IoU 9/11
    scores = [[0.9, 0.0], [0.0, 0.8]]
    assert len(nms_multilabel(boxes, scores, iou=0.5)[0]) == 2
    assert (
        len(nms_multilabel(boxes, scores, iou=0.5, class_agnostic=True)[0])
        == 1
    )


def test_only_iou_above_the_threshold_suppresses():
    boxes = [BOX, [0.0, 0.0, 10.0, 20.0]]  # IoU exactly 0.5
    scores = [[0.9], [0.8]]
    assert len(nms_multilabel(boxes, scores, iou=0.5)[0]) == 2
    assert len(nms_multilabel(boxes, scores, iou=0.49)[0]) == 1


def test_candidates_and_detections_are_capped_by_score():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 5000, (200, 2))
    boxes = np.concatenate([xy, xy + 5], axis=1)
    scores = rng.random((200, 1)).astype(np.float32)
    top = np.sort(scores[:, 0])[::-1]
    _, s, _ = nms_multilabel(boxes, scores, max_candidates=20)
    np.testing.assert_array_equal(s, top[:20])
    _, s, _ = nms_multilabel(boxes, scores, max_det=7)
    np.testing.assert_array_equal(s, top[:7])


def test_no_candidates_returns_empty_arrays():
    b, s, c = nms_multilabel([BOX], [[0.0005]])
    assert b.shape == (0, 4)
    assert s.shape == (0,)
    assert c.shape == (0,)
