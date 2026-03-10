"""Tests for comic.py — comic rendering, title creation and orientation selection."""

from pathlib import Path

import pytest

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


# ---------------------------------------------------------------------------
# render_comic — integration (requires cairosvg)
# ---------------------------------------------------------------------------


def test_render_comic(tmp_path):
    pytest.importorskip("cairosvg", reason="cairosvg required")
    from xyzrender.comic import render_comic
    from xyzrender.readers import load_molecule, load_trajectory_frames
    from xyzrender.types import RenderConfig

    graph, _ = load_molecule(str(STRUCTURES / "sn2.v000.xyz"))
    frames = load_trajectory_frames(str(STRUCTURES / "sn2.v000.xyz"))
    cfg = RenderConfig(auto_orient=False)
    out = str(tmp_path / "comic.svg")
    render_comic(
        frames,
        reference_graph=graph,
        num_comic_frames=5,
        config=cfg,
        output=out,
    )

    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


def test_render_comic_orientation(tmp_path):
    pytest.importorskip("cairosvg", reason="cairosvg required")
    from xyzrender.comic import render_comic
    from xyzrender.readers import load_molecule, load_trajectory_frames
    from xyzrender.types import RenderConfig

    graph, _ = load_molecule(str(STRUCTURES / "sn2.v000.xyz"))
    frames = load_trajectory_frames(str(STRUCTURES / "sn2.v000.xyz"))
    cfg = RenderConfig()
    out = str(tmp_path / "comic.svg")
    render_comic(frames, reference_graph=graph, num_comic_frames=5, config=cfg, output=out, pca_orient_frame=-1)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


def test_render_comic_with_titles(tmp_path):
    pytest.importorskip("cairosvg", reason="cairosvg required")
    from xyzrender.comic import render_comic
    from xyzrender.readers import load_molecule, load_trajectory_frames
    from xyzrender.types import RenderConfig

    graph, _ = load_molecule(str(STRUCTURES / "sn2.v000.xyz"))
    frames = load_trajectory_frames(str(STRUCTURES / "sn2.v000.xyz"))
    cfg = RenderConfig()
    out = str(tmp_path / "comic.svg")
    render_comic(
        frames,
        reference_graph=graph,
        num_comic_frames=5,
        config=cfg,
        output=out,
        comic_titles=["t0", "t5", "t10", "t15", "t20"],
    )
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0
