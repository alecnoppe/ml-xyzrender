"""Plot a cartoon of a trajectory."""
from __future__ import annotations

import logging
import sys
from io import BytesIO
from typing import TYPE_CHECKING, List

import numpy as np
import re

from xyzrender.gif import _orient_frames, _compute_rotation, _rotate_frames, _rotation_axis, \
    _rotation_config, _progress
from xyzrender.utils import pca_matrix
from xyzrender.io import build_graph
from xyzrender.bond_builder import build_distance_based_graph
from xyzrender.renderer import render_svg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import networkx as nx
    from xyzgraph.nci import NCIAnalyzer

    from xyzrender.types import RenderConfig



SVG_SIZE_RE = re.compile(
    r'<svg[^>]*\bwidth="([^"]+)"[^>]*\bheight="([^"]+)"',
    re.IGNORECASE
)

# TODO: DOCUMENTATION
# TODO: CHECK IF 'TITLE' ROW CAN BE **ABOVE** THE PLOTS, INSTEAD OF **ON** THE PLOTS.
def _stitch_cartoon(svgs: list[str], output: str, config: RenderConfig, *, cartoon_titles=None) -> None:
    cartoon_svg = '<?xml version="1.0" encoding="UTF-8"?>\n'

    widths = []
    heights = []

    for svg in svgs:
        match = SVG_SIZE_RE.search(svg)
        if not match:
            raise ValueError("Could not find width/height in SVG")

        width, height = match.groups()

        width = int(float(width.replace("px", "")))
        height = int(float(height.replace("px", "")))

        widths.append(width)
        heights.append(height)

    total_width = sum(widths)
    max_height = max(heights)
    start_svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{total_width}"
     height="{max_height}"
     viewBox="0 0 {total_width} {max_height}">\n
    """
    cartoon_svg += start_svg
    cartoon_svg += f'<rect width="100%" height="100%" fill="{config.background}"/>\n'
    widths_til_i = 0
    
    for i, svg in enumerate(svgs):
        
        center_height = 0.5 * (max_height - heights[i])
        cartoon_svg += f'<g transform="translate({widths_til_i},{center_height})">\n'
        cartoon_svg += svg + "\n"
        cartoon_svg += "</g>\n"
        
        if cartoon_titles:
            title_x = widths_til_i + widths[i] / 2  # center of the frame
            cartoon_svg += (
                f'<text x="{title_x}" y="{config.padding/2:.1f}" '
                f'text-anchor="middle" dominant-baseline="hanging" '
                f'font-size="{config.title_font_size}" '
                f'font-family="{config.title_font_family}" '
                f'fill="{config.title_color}">'
                f'{cartoon_titles[i]}</text>\n'
            )

        widths_til_i += widths[i]
        
    cartoon_svg += "</svg>"
    
    with open(output, "w") as f:
        f.write(cartoon_svg)

# TODO: DOCUMENTATION
def _render_cartoon_frames(
    graph: nx.Graph,
    frames: list[dict],
    config: RenderConfig,
    *,
    nci_analyzer: NCIAnalyzer | None = None,
    fixed_ncis: list | None = None,
    rotation_axis: np.ndarray | None = None,
    rotation_sign: float = 1.0,
) -> list[bytes]:
    """Render each trajectory frame to PNG, keeping graph topology fixed.

    If *nci_analyzer* is provided, NCI interactions are re-detected per
    frame and the graph is decorated with the frame-specific NCI edges.
    If *fixed_ncis* is provided, the same NCI set is applied to every frame
    (centroids recomputed from current atom positions each frame).
    If *rotation_axis* is provided, each frame is incrementally rotated
    around that axis for a full 360° over all frames.
    """
    if nci_analyzer is not None or fixed_ncis is not None:
        from xyzgraph.nci import build_nci_graph
    if rotation_axis is not None:
        from xyzrender.io import apply_axis_angle_rotation

    total = len(frames)
    step = 360.0 / total if rotation_axis is not None else 0
    svgs = []
    for idx, frame in enumerate(frames):
        positions = frame["positions"]
        for i, (x, y, z) in enumerate(positions):
            graph.nodes[i]["position"] = (float(x), float(y), float(z))

        if nci_analyzer is not None:
            ncis = nci_analyzer.detect(np.array(positions))
            render_graph = build_nci_graph(graph, ncis)
        elif fixed_ncis is not None:
            render_graph = build_nci_graph(graph, fixed_ncis)
        else:
            render_graph = graph

        if rotation_axis is not None:
            apply_axis_angle_rotation(render_graph, rotation_axis, rotation_sign * step * idx)

        svg = render_svg(render_graph, config, _log=False)
        svgs.append(svg)
        _progress(idx + 1, total)
    return svgs

# TODO: DOCUMENTATION
def plot_cartoon(
    frames: list[dict],
    num_cartoon_frames: int,
    config: RenderConfig,
    output: str,
    *,
    charge: int = 0,
    multiplicity: int | None = None,
    cartoon_titles: List[str] | None = None,
    reference_graph: nx.Graph | None = None,
    detect_nci: bool = False,
    axis: str | None = None,
    kekule: bool = False,
    graph_builder: str = "distance-based"
) -> None:
    """Render optimization/trajectory path as an animated GIF.
    
    Args:
        graph_builder: choose graph builder from `distance-based` and `default`.
            NOTE: for QM9 `distance-based` is prefered.

    Builds the molecular graph once from the last frame (optimized geometry)
    to get correct bond orders, then updates positions per frame.
    If ``reference_graph`` is provided, all frames are rotated to match.
    If ``detect_nci`` is True, NCI interactions are re-detected per frame
    using xyzgraph's NCIAnalyzer (topology built once, geometry per frame).
    If ``axis`` is provided, the molecule rotates 360° around that axis
    over the course of the trajectory.
    """
    if cartoon_titles and len(cartoon_titles) != num_cartoon_frames:
        raise Exception("If you want to label the frames at each timesteps, the length of `cartoon_titles` must match \
            the `num_cartoon_frames`.")
    if num_cartoon_frames > len(frames) or num_cartoon_frames <= 0:
        raise Exception("The `num_cartoon_frames` must be lie in interval (0, `len(frames)`] ")
    # Build graph from last frame (optimized geometry → correct bond orders)
    last = frames[-1]
    last_atoms = list(zip(last["symbols"], [tuple(p) for p in last["positions"]], strict=True))
    match graph_builder:
        case "distance-based":
            graph = build_distance_based_graph(last_atoms)
        case "default":
            graph = build_graph(last_atoms, charge=charge, multiplicity=multiplicity, kekule=kekule)
        case _:
            raise Exception(f"Options for graph_builder are `distance-based` or `default`, not {graph_builder}")

    # Subsample frames to include the first frame, the last frame and n-2 frames in-between 
    indices = np.linspace(0, len(frames) - 1, num_cartoon_frames, dtype=int)
    frames = [frames[i] for i in indices]
    
    # Copy TS/NCI edge attributes from reference graph
    if reference_graph is not None:
        for i, j, d in reference_graph.edges(data=True):
            if graph.has_edge(i, j):
                for attr in ("TS", "NCI", "bond_type"):
                    if attr in d:
                        graph[i][j][attr] = d[attr]

    # Build NCI analyzer once from topology, detect per frame later
    nci_analyzer = None
    if detect_nci:
        from xyzgraph.nci import NCIAnalyzer

        nci_analyzer = NCIAnalyzer(graph)

    # Apply viewer rotation if reference orientation was given
    if reference_graph is not None:
        logger.debug("Applying Kabsch rotation from viewer orientation")
        rot = _compute_rotation(graph, reference_graph)
        frames = _rotate_frames(frames, rot)

    # PCA: compute once from first frame, apply consistently to all
    if config.auto_orient:
        import copy

        vt = pca_matrix(np.array(frames[0]["positions"]))
        frames = _orient_frames(frames, vt)
        config = copy.copy(config)
        config.auto_orient = False

    # Fixed viewport for rotation
    axis_vec = None
    axis_sign = 1.0
    if axis:
        pos0 = np.array(frames[0]["positions"])
        config = _rotation_config(pos0, config)
        axis_vec, axis_sign = _rotation_axis(axis)

    logger.info("Rendering trajectory cartoon (%d frames%s)", len(frames), f", axis={axis}" if axis else "")
    pngs = _render_cartoon_frames(
        graph, frames, config, nci_analyzer=nci_analyzer, rotation_axis=axis_vec, rotation_sign=axis_sign
    )
    _stitch_cartoon(pngs, output, config, cartoon_titles=cartoon_titles)
    logger.info("Wrote %s", output)
