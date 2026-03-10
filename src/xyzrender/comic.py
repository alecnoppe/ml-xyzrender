"""Plot a comic of a trajectory."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, List

import numpy as np
from xyzgraph import build_graph

from xyzrender.gif import _compute_rotation, _fixed_viewport, _orient_frames, _progress, _rotate_frames
from xyzrender.renderer import render_svg
from xyzrender.utils import pca_matrix

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import networkx as nx
    from xyzgraph.nci import NCIAnalyzer

    from xyzrender.types import RenderConfig


def _stitch_comic(svgs: list[str], output: str, config: RenderConfig, *, comic_titles=None) -> None:
    """
    Stitch multiple SVGs together.

    Horizontally stitch together a list of SVGS. Additionally ensures that the individual SVGs are centered vertically
    and that the entire plot has a

    Parameters
    ----------
        svgs:
            The XML strings corresponding to the individual svgs (to be plotted in the comic).
        output:
            Path to store the comic .svg file.
        config:
            RenderConfig - used only for stylizing the title (right now).
        comic_titles:
            Optionally, plot a title for each frame in the comic.
    """
    # Create regex for extracting width/height from individual svgs
    svg_size_re = re.compile(r'<svg[^>]*\bwidth="([^"]+)"[^>]*\bheight="([^"]+)"', re.IGNORECASE)

    # Iterate over SVGs and store the width/height for each one
    widths = []
    heights = []
    for svg in svgs:
        match = svg_size_re.search(svg)
        if not match:
            raise ValueError("Could not find width/height in SVG")
        width, height = match.groups()
        width = int(float(width.replace("px", "")))
        height = int(float(height.replace("px", "")))
        widths.append(width)
        heights.append(height)
    # Select total width and maximum height, for row width and height respectively.
    # Max height is also used to vertically center the individual plots - in case they do not all have the same height
    total_width = sum(widths)
    max_height = max(heights)

    # Create the header of the final svg
    comic_svg = '<?xml version="1.0" encoding="UTF-8"?>\n'
    start_svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{total_width}"
     height="{max_height}"
     viewBox="0 0 {total_width} {max_height}">\n
    """
    comic_svg += start_svg
    comic_svg += f'<rect width="100%" height="100%" fill="{config.background}"/>\n'

    # Iterate over all svgs, and for each svg do the following
    # Shift to the right by 'widths_til_i' and center the svg vertically with the tallest svg in the list.
    # Optionally write a title in a row along the top of the comic.
    widths_til_i = 0
    for i, svg in enumerate(svgs):
        # Add main svg content
        center_height = 0.5 * (max_height - heights[i])
        comic_svg += f'<g transform="translate({widths_til_i},{center_height})">\n'
        comic_svg += svg + "\n"
        comic_svg += "</g>\n"
        # Add titles
        if comic_titles:
            title_x = widths_til_i + widths[i] / 2  # center of the frame
            comic_svg += (
                f'<text x="{title_x}" y="{config.padding:.1f}" '
                f'text-anchor="middle" dominant-baseline="hanging" '
                f'font-size="{config.title_font_size}" '
                f'font-family="{config.title_font_family}" '
                f'fill="{config.title_color}">'
                f"{comic_titles[i]}</text>\n"
            )
        # Update width offset
        widths_til_i += widths[i]

    comic_svg += "</svg>"

    # Write final svg to disk
    with open(output, "w") as f:
        f.write(comic_svg)


def _render_comic_frames(
    graph: nx.Graph,
    frames: list[dict],
    config: RenderConfig,
    *,
    nci_analyzer: NCIAnalyzer | None = None,
    recompute_bonds: bool = False,
    charge: int = 0,
    multiplicity: int | None = None,
    kekule: bool = False,
) -> list[bytes]:
    """Render each trajectory frame to SVG, allowing graph topology to be updated.

    If *nci_analyzer* is provided, NCI interactions are re-detected per
    frame and the graph is decorated with the frame-specific NCI edges.
    """
    if nci_analyzer is not None:
        from xyzgraph.nci import build_nci_graph

    total = len(frames)
    svgs = []
    for idx, frame in enumerate(frames):
        positions = frame["positions"]
        for i, (x, y, z) in enumerate(positions):
            graph.nodes[i]["position"] = (float(x), float(y), float(z))

        if recompute_bonds:
            atoms = list(zip(frame["symbols"], [tuple(p) for p in frame["positions"]], strict=True))
            graph = build_graph(atoms, charge=charge, multiplicity=multiplicity, kekule=kekule)

        if nci_analyzer is not None:
            ncis = nci_analyzer.detect(np.array(positions))
            render_graph = build_nci_graph(graph, ncis)
        else:
            render_graph = graph

        svg = render_svg(render_graph, config, _log=False, _id_prefix=str(idx))
        svgs.append(svg)
        _progress(idx + 1, total)
    return svgs


def render_comic(
    frames: list[dict],
    num_comic_frames: int,
    config: RenderConfig,
    output: str,
    *,
    charge: int = 0,
    multiplicity: int | None = None,
    comic_titles: List[str] | None = None,
    reference_graph: nx.Graph | None = None,
    detect_nci: bool = False,
    kekule: bool = False,
    pca_orient_frame: int = 0,
    recompute_bonds: bool = True,
) -> None:
    """Plot optimization/trajectory path as a comic.

    Renders multiple SVG files and stitches them together horizontally. The orientation can be fitted to any frame,
    by using pca_orient_frame.

    NOTE: Though we could theoretically render each frame in a trajectory, it is highly recommended to subsample using
    `num_comic_frames` << #frames for large #frames.

    Parameters
    ----------
        num_comic_frames:
            Number of frames to plot in the comic. This will plot the first frame, the last frame and n-2 equidistant
            frames in between.
        comic_titles:
            Optional list of titles which can be plotted above each frame in the comic.
        pca_orient_frame:
            Which frame to use to determine the pca orientation for the frames in the comic.

    Builds the molecular graph once from the last frame (optimized geometry)
    to get correct bond orders, then updates positions per frame.
    If ``reference_graph`` is provided, all frames are rotated to match.
    If ``detect_nci`` is True, NCI interactions are re-detected per frame
    using xyzgraph's NCIAnalyzer (topology built once, geometry per frame).
    """
    if comic_titles and len(comic_titles) != num_comic_frames:
        raise Exception(
            "If you want to label the frames at each timesteps, the length of comic_titles must match"
            " the num_comic_frames"
        )
    if num_comic_frames > len(frames) or num_comic_frames <= 0:
        raise Exception("The num_comic_frames must be between 0 and #frames")

    # Build graph from last frame (optimized geometry → correct bond orders)
    last = frames[-1]
    last_atoms = list(zip(last["symbols"], [tuple(p) for p in last["positions"]], strict=True))
    graph = build_graph(last_atoms, charge=charge, multiplicity=multiplicity, kekule=kekule)

    # Subsample frames to include the first frame, the last frame and n-2 frames in-between
    indices = np.linspace(0, len(frames) - 1, num_comic_frames, dtype=int)
    frames = [frames[i] for i in indices]

    # Copy TS/NCI edge attributes from reference graph
    if reference_graph is not None:
        shared_edges = [(i, j, d) for i, j, d in reference_graph.edges(data=True) if graph.has_edge(i, j)]
        for i, j, d in shared_edges:
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

        vt = pca_matrix(np.array(frames[pca_orient_frame]["positions"]))
        frames = _orient_frames(frames, vt)
        config = copy.copy(config)
        config.auto_orient = False

    # Fixed viewport across all frames so every PNG has identical dimensions
    config = _fixed_viewport(frames, config)

    logger.info("Rendering trajectory comic (%d frames)", len(frames))
    svgs = _render_comic_frames(
        graph,
        frames,
        config,
        nci_analyzer=nci_analyzer,
        recompute_bonds=recompute_bonds,
        charge=charge,
        multiplicity=multiplicity,
        kekule=kekule,
    )
    _stitch_comic(svgs, output, config, comic_titles=comic_titles)
    logger.info("Wrote %s", output)
