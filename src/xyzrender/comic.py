"""Plot a comic of a trajectory."""
from __future__ import annotations

import logging
import sys
import math
from io import BytesIO
from typing import TYPE_CHECKING, List

import numpy as np
import re

from xyzrender.gif import _orient_frames, _compute_rotation, _rotate_frames, _rotation_axis, \
    _progress, _fixed_viewport
from xyzrender.utils import pca_matrix
from xyzrender.io import build_graph
from xyzrender.bond_builder import build_distance_based_graph
from xyzrender.renderer import render_svg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import networkx as nx
    from xyzgraph.nci import NCIAnalyzer

    from xyzrender.types import RenderConfig


def _stitch_grid(svgs: list[str], output: str, config: RenderConfig, *, frame_titles:list[str] | None = None, 
                 _flat: bool = False, n_rows: int | None = None, n_cols: int | None = None) -> None:
    """
    Stitch SVGs into a grid.

    If the number of frames does not fill the grid exactly, the remaining
    cells are padded with white boxes.
    """
    # If `_flat` is specified, use the comic code to create essentially a flattened grid.
    if _flat:
        _stitch_comic(svgs, output, config, comic_titles=frame_titles)
        return
    # Verify the arguments
    if (n_rows or n_cols) and not(n_rows and n_cols):
        raise ValueError(
            "To use n_rows or n_cols, you must specify both."
        )
    if (n_rows and n_cols) and n_rows * n_cols < len(svgs):
        raise ValueError(
            "n_rows*n_cols must be bigger than the number of svgs."
        )
    if frame_titles and len(frame_titles) != len(svgs):
        raise ValueError(
            "Length of frame_titles must match number of frames."
        )
    # Determine grid size
    n_frames = len(svgs)
    n_cols = math.ceil(math.sqrt(n_frames)) if not n_cols else n_cols
    n_rows = math.ceil(n_frames / n_cols) if not n_rows else n_rows
    n_cells = n_rows * n_cols

    # Create regex for extracting width/height from individual svgs
    SVG_SIZE_RE = re.compile(
        r'<svg[^>]*\bwidth="([^"]+)"[^>]*\bheight="([^"]+)"',
        re.IGNORECASE
    )

    # Iterate over SVGs and store the width/height for each one
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

    # Create padding frames if n_frames is not a perfect square
    if n_cells > n_frames:
        max_w = max(widths)
        max_h = max(heights)
        pad_count = n_cells - n_frames
        widths.extend([max_w] * pad_count)
        heights.extend([max_h] * pad_count)
        svgs = svgs + [None] * pad_count  # placeholder for empty cells

    # Reshape the w/h into square grid
    widths_grid = np.array(widths).reshape(n_rows, n_cols)
    heights_grid = np.array(heights).reshape(n_rows, n_cols)
    # Find max w/h
    col_widths = widths_grid.max(axis=0)
    row_heights = heights_grid.max(axis=1)
    # Compute total widths
    total_width = int(col_widths.sum())
    total_height = int(row_heights.sum())

    # Create the header of the final svg
    grid_svg = '<?xml version="1.0" encoding="UTF-8"?>\n'
    grid_svg += f"""
    <svg xmlns="http://www.w3.org/2000/svg"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    width="{total_width}"
    height="{total_height}"
    viewBox="0 0 {total_width} {total_height}">
    """
    grid_svg += f'<rect width="100%" height="100%" fill="{config.background}"/>\n'
    
    # Iterate over all svgs, and for each svg do the following
    # For N rows: For each frame, shift to the right by `x_offset`, and down by `y_offset` and center the svg within 
    # the frame determined by the maximum width/height of any SVG in the list.
    # Optionally write a title in a row along the top of the comic. 
    idx = 0
    while idx < n_frames:
        # Compute row/col widths and offsets.
        row = idx // n_cols
        col = idx % n_cols
        x_offset = col_widths[:col].sum()
        y_offset = row_heights[:row].sum()
        cell_w = col_widths[col]
        cell_h = row_heights[row]

        # Add main svg content
        dx = 0.5 * (cell_w - widths_grid[row, col])
        dy = 0.5 * (cell_h - heights_grid[row, col])
        grid_svg += (
            f'<g transform="translate({x_offset + dx},{y_offset + dy})">\n'
        )
        grid_svg += svgs[idx] + "\n"
        grid_svg += "</g>\n"
        # Add titles
        if frame_titles:
            title_x = x_offset + cell_w / 2
            title_y = y_offset + config.padding / 2
            grid_svg += (
                f'<text x="{title_x}" y="{title_y:.1f}" '
                f'text-anchor="middle" dominant-baseline="hanging" '
                f'font-size="{config.title_font_size}" '
                f'font-family="{config.title_font_family}" '
                f'fill="{config.title_color}">'
                f'{frame_titles[idx]}</text>\n'
            )
        idx += 1

    # Pad the grid with empty plots, if the number of frames is not a perfect square
    while idx < n_rows * n_cols:
        row = idx // n_cols
        col = idx % n_cols
        x_offset = col_widths[:col].sum()
        y_offset = row_heights[:row].sum()
        cell_w = col_widths[col]
        cell_h = row_heights[row]
        # Empty padded cell → draw background color rectangle
        grid_svg += (
            f'<rect x="{x_offset}" y="{y_offset}" '
            f'width="{cell_w}" height="{cell_h}" '
            f'fill="{config.background}"/>\n'
        )
        idx += 1

    grid_svg += "</svg>"

    with open(output, "w") as f:
        f.write(grid_svg)


def _stitch_comic(svgs: list[str], output: str, config: RenderConfig, *, comic_titles=None) -> None:
    """
    Horizontally together a list of SVGS, such that they are vertically centered and may feature individual titles.
        
    Args:
        svgs: The XML strings corresponding to the individual svgs (to be plotted in the comic).
        output: Path to store the comic .svg file.
        config: RenderConfig - used only for stylizing the title (right now).
        comic_titles: Optionally, can plot a title for each frame in the comic.
    """
    # Create regex for extracting width/height from individual svgs
    SVG_SIZE_RE = re.compile(
        r'<svg[^>]*\bwidth="([^"]+)"[^>]*\bheight="([^"]+)"',
        re.IGNORECASE
    )
    
    # Iterate over SVGs and store the width/height for each one
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
    # Select total width and maximum height, for row width and height respcetively. 
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
                f'<text x="{title_x}" y="{config.padding/2:.1f}" '
                f'text-anchor="middle" dominant-baseline="hanging" '
                f'font-size="{config.title_font_size}" '
                f'font-family="{config.title_font_family}" '
                f'fill="{config.title_color}">'
                f'{comic_titles[i]}</text>\n'
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
    fixed_ncis: list | None = None,
    rotation_axis: np.ndarray | None = None,
    rotation_sign: float = 1.0,
    recompute_bonds: bool = False,
    charge: int = 0,
    multiplicity: int | None = None,
    kekule: bool = False,
    graph_builder: str = "default"
) -> list[bytes]:
    """Render each trajectory frame to SVG, allowing graph topology to be updated.

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
        
        if recompute_bonds:
            atoms = list(zip(frame["symbols"], [tuple(p) for p in frame["positions"]], strict=True))
            match graph_builder:
                case "distance-based":
                    graph = build_distance_based_graph(atoms)
                case "default":
                    graph = build_graph(atoms, charge=charge, multiplicity=multiplicity, kekule=kekule)
                case _:
                    raise Exception(f"Options for graph_builder are `distance-based` or `default`, not {graph_builder}")
                
        if nci_analyzer is not None:
            ncis = nci_analyzer.detect(np.array(positions))
            render_graph = build_nci_graph(graph, ncis)
        elif fixed_ncis is not None:
            render_graph = build_nci_graph(graph, fixed_ncis)
        else:
            render_graph = graph

        if rotation_axis is not None:
            apply_axis_angle_rotation(render_graph, rotation_axis, rotation_sign * step * idx)

        svg = render_svg(render_graph, config, _log=False, _id_prefix=str(idx))
        svgs.append(svg)
        _progress(idx + 1, total)
    return svgs


def plot_comic(
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
    axis: str | None = None,
    kekule: bool = False,
    graph_builder: str = "default",
    pca_orient_frame: int = 0
) -> None:
    """Render optimization/trajectory path as a comic.
    NOTE: Though we could theoretically render each frame in a trajectory, it is highly recommended to subsample using
    `num_comic_frames` << #frames for large #frames.
    
    Args:
        num_comic_frames: Number of frames to plot in the comic. This will plot the first frame, the last \
            frame and n-2 equidistant frames in between.
        graph_builder: Choose graph builder from `distance-based` and `default`.
            NOTE: for QM9 `distance-based` is prefered.
        comic_titles: Optional list of titles which can be plotted above each frame in the comic.
        pca_orient_frame: which frame to use to determine the pca orientation for the frames in the comic.

    Builds the molecular graph once from the last frame (optimized geometry)
    to get correct bond orders, then updates positions per frame.
    If ``reference_graph`` is provided, all frames are rotated to match.
    If ``detect_nci`` is True, NCI interactions are re-detected per frame
    using xyzgraph's NCIAnalyzer (topology built once, geometry per frame).
    If ``axis`` is provided, the molecule rotates 360° around that axis
    over the course of the trajectory.
    """
    if comic_titles and len(comic_titles) != num_comic_frames:
        raise Exception("If you want to label the frames at each timesteps, the length of `comic_titles` must match \
            the `num_comic_frames`.")
    if num_comic_frames > len(frames) or num_comic_frames <= 0:
        raise Exception("The `num_comic_frames` must be lie in interval (0, `len(frames)`] ")
    
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
    indices = np.linspace(0, len(frames) - 1, num_comic_frames, dtype=int)
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

        vt = pca_matrix(np.array(frames[pca_orient_frame]["positions"]))
        frames = _orient_frames(frames, vt)
        config = copy.copy(config)
        config.auto_orient = False

    axis_vec = None
    axis_sign = 1.0
    if axis:
        axis_vec, axis_sign = _rotation_axis(axis)

    # Fixed viewport across all frames so every PNG has identical dimensions
    config = _fixed_viewport(frames, config, rotation_axis=axis_vec)

    logger.info("Rendering trajectory comic (%d frames%s)", len(frames), f", axis={axis}" if axis else "")
    svgs = _render_comic_frames(
        graph, frames, config, nci_analyzer=nci_analyzer, rotation_axis=axis_vec, rotation_sign=axis_sign, 
        recompute_bonds=True, charge=charge, multiplicity=multiplicity, kekule=kekule, graph_builder=graph_builder
    )
    _stitch_comic(svgs, output, config, comic_titles=comic_titles)
    logger.info("Wrote %s", output)
    
    
def plot_grid(graphs: list[nx.Graph],
    config: RenderConfig,
    output: str,
    *,
    frame_titles: List[str] | None = None,
    _flat: bool = False,
    n_rows: int | None = None,
    n_cols: int | None = None):
    """Render multiple frames in a square grid. Looks best if |frames| = N^2 for some integer N.
    
    Args:
        graph_builder: Choose graph builder from `distance-based` and `default`.
            NOTE: for QM9 `distance-based` is prefered.
        frame_titles: Optional list of titles which can be plotted above each frame in the grid.
        _flat: Whether to flatten the grid; plot a single row of svgs.
        
    Builds the molecular graph once from the last frame (optimized geometry)
    to get correct bond orders, then updates positions per frame.
    If ``reference_graph`` is provided, all frames are rotated to match.
    If ``detect_nci`` is True, NCI interactions are re-detected per frame
    using xyzgraph's NCIAnalyzer (topology built once, geometry per frame).
    If ``axis`` is provided, the molecule rotates 360° around that axis
    over the course of the trajectory.
    """
    svgs = [render_svg(graph, config=config, _id_prefix=str(i)) for i,graph in enumerate(graphs)]
    _stitch_grid(svgs, output, config, frame_titles=frame_titles, _flat=_flat, n_rows=n_rows, n_cols=n_cols)
    logger.info("Wrote %s", output)
