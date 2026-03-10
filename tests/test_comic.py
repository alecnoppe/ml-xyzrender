"""Tests for comic.py — comic rendering, title creation and orientation selection."""

from pathlib import Path

import pytest

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


# ---------------------------------------------------------------------------
# render_comic — input validation
# ---------------------------------------------------------------------------


def test_render_input(tmp_path):
    from xyzrender.comic import _stitch_comic, render_comic
    from xyzrender.readers import load_molecule, load_trajectory_frames
    from xyzrender.types import RenderConfig

    graph, _ = load_molecule(str(STRUCTURES / "sn2.v000.xyz"))
    frames = load_trajectory_frames(str(STRUCTURES / "sn2.v000.xyz"))
    cfg = RenderConfig()
    out = str(tmp_path / "comic.svg")

    # Test what happens when number of titles != num frames - should raise error
    with pytest.raises(
        Exception,
        match="If you want to label the frames at each timesteps, the length of comic_titles must match"
        " the num_comic_frames",
    ):
        render_comic(
            frames,
            reference_graph=graph,
            num_comic_frames=5,
            config=cfg,
            output=out,
            comic_titles=["t0", "t5", "t10", "t15"],
        )

    # Test what happens when number of sample frames > number of trajectory frames - should raise error
    with pytest.raises(Exception, match="The num_comic_frames must be between 0 and #frames"):
        render_comic(frames, reference_graph=graph, num_comic_frames=50, config=cfg, output=out)

    # Test what happens when _stitch_comic is called on a list of empty strings
    with pytest.raises(ValueError, match="Could not find width/height in SVG"):
        _stitch_comic(svgs=["", "", "", "", ""], config=cfg, output=out)


# ---------------------------------------------------------------------------
# render_comic — integration
# ---------------------------------------------------------------------------


def test_render_comic(tmp_path):
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


def test_render_comic_no_reference(tmp_path):
    from xyzrender.comic import render_comic
    from xyzrender.readers import load_trajectory_frames
    from xyzrender.types import RenderConfig

    frames = load_trajectory_frames(str(STRUCTURES / "sn2.v000.xyz"))
    cfg = RenderConfig()
    out = str(tmp_path / "comic.svg")
    render_comic(frames, num_comic_frames=5, config=cfg, output=out, pca_orient_frame=-1)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


def test_render_comic_nci(tmp_path):
    from xyzrender.comic import render_comic
    from xyzrender.readers import load_molecule, load_trajectory_frames
    from xyzrender.types import RenderConfig

    graph, _ = load_molecule(str(STRUCTURES / "sn2.v000.xyz"))
    frames = load_trajectory_frames(str(STRUCTURES / "sn2.v000.xyz"))
    cfg = RenderConfig()
    out = str(tmp_path / "comic.svg")
    render_comic(frames, reference_graph=graph, num_comic_frames=5, config=cfg, output=out, detect_nci=True)

    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


def test_render_comic_no_recompute_bonds(tmp_path):
    from xyzrender.comic import render_comic
    from xyzrender.readers import load_molecule, load_trajectory_frames
    from xyzrender.types import RenderConfig

    graph, _ = load_molecule(str(STRUCTURES / "sn2.v000.xyz"))
    frames = load_trajectory_frames(str(STRUCTURES / "sn2.v000.xyz"))
    cfg = RenderConfig()
    out = str(tmp_path / "comic.svg")
    render_comic(frames, reference_graph=graph, num_comic_frames=5, config=cfg, output=out, recompute_bonds=False)

    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


def test_render_comic_orientation(tmp_path):
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


def test_render_comic_auto_orientation(tmp_path):
    from xyzrender.comic import render_comic
    from xyzrender.readers import load_molecule, load_trajectory_frames
    from xyzrender.types import RenderConfig

    graph, _ = load_molecule(str(STRUCTURES / "sn2.v000.xyz"))
    frames = load_trajectory_frames(str(STRUCTURES / "sn2.v000.xyz"))
    cfg = RenderConfig(auto_orient=True)
    out = str(tmp_path / "comic.svg")
    render_comic(frames, reference_graph=graph, num_comic_frames=5, config=cfg, output=out)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


def test_render_comic_with_titles(tmp_path):
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
