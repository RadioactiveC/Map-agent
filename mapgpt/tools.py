from __future__ import annotations

from copy import deepcopy
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from langchain.tools import tool

import matplotlib

# Use non-interactive backend suitable for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt


try:
    import geopandas as gpd  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    gpd = None  # type: ignore

try:
    import rasterio  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    rasterio = None  # type: ignore


logger = logging.getLogger("mapgpt.tools")
logging.basicConfig(level=logging.INFO)

# Simple mapping for common Chinese color names to matplotlib-compatible names
_CN_COLOR_MAP: Dict[str, str] = {
    "白": "white",
    "白色": "white",
    "黑": "black",
    "黑色": "black",
    "红": "red",
    "红色": "red",
    "蓝": "blue",
    "蓝色": "blue",
    "绿": "green",
    "绿色": "green",
    "黄": "yellow",
    "黄色": "yellow",
    "灰": "gray",
    "灰色": "gray",
    "紫": "purple",
    "紫色": "purple",
    "橙": "orange",
    "橙色": "orange",
}


def _normalize_color(color: str) -> str:
    key = color.strip().lower()
    # Try direct match in Chinese map first (keys are Chinese, but we compare lowercase anyway)
    if key in _CN_COLOR_MAP:
        return _CN_COLOR_MAP[key]
    # Also try removing the trailing '色'
    if key.endswith("色") and key[:-1] in _CN_COLOR_MAP:
        return _CN_COLOR_MAP[key[:-1]]
    return color


@dataclass
class LineStyle:
    color: str = "#ffffff"
    width: float = 1.5
    alpha: float = 1.0


@dataclass
class PolygonStyle:
    edgecolor: str = "#333333"
    facecolor: Optional[str] = None  # None -> no fill
    linewidth: float = 0.8
    alpha: float = 1.0


@dataclass
class PointStyle:
    color: str = "#ff5555"
    size: float = 8.0
    alpha: float = 1.0


@dataclass
class MapSession:
    figure: Optional[plt.Figure] = None
    axis: Optional[plt.Axes] = None
    initialized: bool = False
    legend_entries: List[Dict] = field(default_factory=list)
    data_paths: List[str] = field(default_factory=list)
    # field() solves this problem perfectly through the default_factory parameter. default_factory accepts a callable object (usually a function or a class) that takes no arguments and returns the initial value of the field.
    # The key point is: this default_factory will be called every time a new instance is created, thus generating a brand new, independent object for each instance.
    # Current styles
    line_style: LineStyle = field(default_factory=LineStyle)
    polygon_style: PolygonStyle = field(default_factory=PolygonStyle)
    point_style: PointStyle = field(default_factory=PointStyle)

    title_color: str = "#0a0a0a"
    background_color: str = "#ffffff"
    title: Optional[str] = None

    def ensure_initialized(self) -> None:
        if not self.initialized or self.figure is None or self.axis is None:
            raise RuntimeError(
                "Map is not initialized yet. Please call map_initial first."
            )


# A simple singleton session for now
_SESSION: MapSession = MapSession()

# Parsing string input
def _parse_input_list(input_str: str) -> List[str]:
    input_str = input_str.strip()
    if not input_str:
        return []
    # Try JSON array first
    try:
        value = json.loads(input_str)
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return [v.strip() for v in value]
    except Exception:
        pass
    # Fallback: comma/semicolon/whitespace separated
    separators = [",", ";", "\n"]
    for sep in separators:
        if sep in input_str:
            return [p.strip() for p in input_str.split(sep) if p.strip()]
    return [input_str]

# Check if a file exists
def _safe_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False

# Traverse the file to find the maximum coordinate axis range
def _clean_path_arg(path_str: str) -> str:
    if not path_str:
        return ""
    return path_str.strip().strip("'").strip('"').strip()


def _collect_bounds_from_paths(paths: List[str]) -> Optional[Tuple[float, float, float, float]]:
    bounds: Optional[Tuple[float, float, float, float]] = None
    for p in paths:
        if not _safe_exists(p):
            continue
        ext = os.path.splitext(p)[1].lower()
        try:
            if gpd is not None and ext in {".shp", ".geojson", ".json", ".gpkg"}:
                gdf = gpd.read_file(p)
                if gdf.empty:
                    continue
                minx, miny, maxx, maxy = gdf.total_bounds
            elif rasterio is not None and ext in {".tif", ".tiff"}:
                with rasterio.open(p) as src:
                    left, bottom, right, top = src.bounds
                minx, miny, maxx, maxy = left, bottom, right, top
            else:
                continue
            if bounds is None:
                bounds = (minx, miny, maxx, maxy)
            else:
                bminx, bminy, bmaxx, bmaxy = bounds
                bounds = (
                    min(bminx, minx),
                    min(bminy, miny),
                    max(bmaxx, maxx),
                    max(bmaxy, maxy),
                )
        except Exception as exc:  # pragma: no cover - robustness
            logger.warning("Failed to read bounds from %s: %s", p, exc)
            continue
    return bounds


# ---------------------
# Tool implementations
# ---------------------

@tool
def map_initial(action_input: str) -> str:
    """
    Initialize the map canvas and store provided data paths.

    Action Input format:
    - JSON array of strings, or
    - Comma/semicolon/newline separated paths, or
    - Empty string to just initialize a blank map

    Behavior:
    - Creates a new matplotlib Figure/Axes
    - Sets a dark background for contrast
    - If valid data paths are provided, attempts to compute a combined extent
      and sets the axis to that extent.
    """
    # The global keyword is used to declare a variable's "account" in the global scope. If the account already exists, you modify it; if it does not exist, you create a new account for it in the global scope.
    # It is a good programming practice to first define an initial state of _SESSION globally and then modify it in the map_initial function.
    global _SESSION

    raw_paths = _parse_input_list(action_input)
    paths = [_clean_path_arg(p) for p in raw_paths]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    fig.patch.set_facecolor(_SESSION.background_color)
    ax.set_facecolor(_SESSION.background_color)

    _SESSION = MapSession(
        figure=fig,
        axis=ax,
        initialized=True,
        data_paths=paths,
        line_style=_SESSION.line_style,
        polygon_style=_SESSION.polygon_style,
        point_style=_SESSION.point_style,
        title_color=_SESSION.title_color,
        background_color=_SESSION.background_color,
        title=_SESSION.title,
    )

    # Try to set extent if we can infer bounds
    bounds = _collect_bounds_from_paths(paths) # get the range
    if bounds is not None:
        minx, miny, maxx, maxy = bounds
        try:
            _SESSION.axis.set_xlim(minx, maxx)
            _SESSION.axis.set_ylim(miny, maxy)
        except Exception:
            pass

    return (
        "Map initialized successfully with given data. "
        f"Paths count: {len(paths)}"
    )

@tool
def modify_line_width(action_input: str) -> str:
    """
    Set line width. Input: float string, e.g., '1.5'.
    """
    _SESSION.ensure_initialized()
    try:
        width = float(action_input.split('\n')[0].strip())
    except Exception:
        return "Error: line width must be a float"
    _SESSION.line_style.width = width
    return f"Line width set to {width}"

@tool
def modify_line_color(action_input: str) -> str:
    """
    Set line color. Input: color string or hex, e.g., 'white' or '#ffffff'.
    """
    _SESSION.ensure_initialized()
    color = _normalize_color(action_input.split('\n')[0].strip())
    if not color:
        return "Error: line color must be a non-empty string"
    _SESSION.line_style.color = color
    return f"Line color set to {color}"

@tool
def modify_polygon_edge_color(action_input: str) -> str:
    """
    Set polygon edge color. Input: color string.
    """
    _SESSION.ensure_initialized()
    color = _normalize_color(action_input.split('\n')[0].strip())
    if not color:
        return "Error: polygon edge color must be a non-empty string"
    _SESSION.polygon_style.edgecolor = color
    return f"Polygon edge color set to {color}"

@tool
def modify_polygon_face_color(action_input: str) -> str:
    """
    Set polygon face color. Input: color string or 'none'.
    """
    _SESSION.ensure_initialized()
    color = action_input.split('\n')[0].strip()
    if color.lower() in {"none", "transparent", ""}:
        _SESSION.polygon_style.facecolor = None
        return "Polygon face color set to none"
    color = _normalize_color(color)
    _SESSION.polygon_style.facecolor = color
    return f"Polygon face color set to {color}"

@tool
def modify_point_color(action_input: str) -> str:
    """
    Set point color. Input: color string.
    """
    _SESSION.ensure_initialized()
    color = _normalize_color(action_input.split('\n')[0].strip())
    if not color:
        return "Error: point color must be a non-empty string"
    _SESSION.point_style.color = color
    return f"Point color set to {color}"

@tool
def modify_title_color(action_input: str) -> str:
    """
    Set title color. Input: color string.
    """
    _SESSION.ensure_initialized()
    color = _normalize_color(action_input.split('\n')[0].strip() or _SESSION.background_color)
    _SESSION.title_color = color
    if not color:
        return "Error: Title color must be a non-empty string"
    _SESSION.title_color = color
    return f"Title color set to {color}"

@tool
def modify_point_size(action_input: str) -> str:
    """
    Set point size. Input: number string.
    """
    _SESSION.ensure_initialized()
    try:
        size = float(action_input.split('\n')[0].strip())
    except Exception:
        return "Error: point size must be a number"
    _SESSION.point_style.size = size
    return f"Point size set to {size}"


@tool
def map_add_legend(action_input: str) -> str:
    """
    Add a legend for all layers that were given a label. Input: optional JSON for styling, e.g., '{\"title\": \"Legend\", \"loc\": \"Location\"}'.
    """
    _SESSION.ensure_initialized()
    if not _SESSION.legend_entries:
        return "Warning: No labeled layers were added to create a legend from."

    params = {}
    if action_input.split('\n')[0].strip():
        try:
            params = json.loads(action_input.split('\n')[0].strip())
        except Exception as e:
            return f"Error: Invalid JSON for legend parameters: {e}"

    params.setdefault('loc', 'lower right')

    try:
        handles, labels = [], []
        for entry in _SESSION.legend_entries:
            style = entry["style"]
            labels.append(entry["label"])

            if entry["type"] == "polygon":
                handle = mpatches.Patch(
                    facecolor=style.facecolor or "none",
                    edgecolor=style.edgecolor,
                    linewidth=style.linewidth,
                    alpha=style.alpha,
                )
            elif entry["type"] == "line":
                handle = mlines.Line2D(
                    [], [],
                    color=style.color,
                    linewidth=style.width,
                    alpha=style.alpha,
                )
            elif entry["type"] == "point":
                handle = mlines.Line2D(
                    [], [],
                    color=style.color,
                    marker='o',
                    markersize=style.size / 2,  # markersize in legend is different from plot
                    linestyle='None',
                    alpha=style.alpha,
                )
            else:
                continue  # Skip unknown types

            handles.append(handle)

        # Add handles and labels to the user-provided parameters
        params['handles'] = handles
        params['labels'] = labels
        _SESSION.axis.legend(**params)

        return f"Legend added for {len(labels)} entries."
    except Exception as e:
        logger.exception("Failed to add legend: %s", e)
        return f"Error: Failed to add legend: {e}"


@tool
def map_add_layer(action_input: str) -> str:
    """
    Add a layer. Input: path string, OR a JSON string like '{\"path\": \"/path/to/file.shp\", \"label\": \"My Label\"}' to add a legend label.
    """
    _SESSION.ensure_initialized()

    path, label = "", None
    try:
        data = json.loads(action_input.split('\n')[0].strip())
        if isinstance(data, dict):
            path, label = data.get("path"), data.get("label")
        else:
            path = action_input.split('\n')[0].strip()
    except json.JSONDecodeError:
        path = action_input.split('\n')[0].strip()

    path = _clean_path_arg(path)

    if not path: return "Error: layer path is empty"
    if not _safe_exists(path): return f"Error: layer path does not exist: {path}"

    ext = os.path.splitext(path)[1].lower()

    try:
        if gpd is not None and ext in {".shp", ".geojson", ".json", ".gpkg"}:
            gdf = gpd.read_file(path)
            if gdf.empty: return f"Warning: vector layer is empty: {path}"

            geom_type = gdf.geom_type.iloc[0].lower()
            legend_info = None

            if "line" in geom_type:
                gdf.plot(
                    ax=_SESSION.axis,
                    linewidth=_SESSION.line_style.width,
                    color=_SESSION.line_style.color,
                    alpha=_SESSION.line_style.alpha,
                )
                if label:
                    legend_info = {"label": label, "type": "line", "style": deepcopy(_SESSION.line_style)}
            elif "polygon" in geom_type:
                gdf.plot(
                    ax=_SESSION.axis,
                    edgecolor=_SESSION.polygon_style.edgecolor,
                    facecolor=_SESSION.polygon_style.facecolor or "none",
                    linewidth=_SESSION.polygon_style.linewidth,
                    alpha=_SESSION.polygon_style.alpha,
                )
                if label:
                    legend_info = {"label": label, "type": "polygon", "style": deepcopy(_SESSION.polygon_style)}
            else:  # Treat as points
                gdf.plot(
                    ax=_SESSION.axis,
                    markersize=_SESSION.point_style.size,
                    color=_SESSION.point_style.color,
                    alpha=_SESSION.point_style.alpha,
                )
                if label:
                    legend_info = {"label": label, "type": "point", "style": deepcopy(_SESSION.point_style)}

            if legend_info:
                _SESSION.legend_entries.append(legend_info)

            _SESSION.axis.set_aspect("equal", adjustable="datalim")
            return f"Vector layer added: {path}" + (f" with label '{label}'" if label else "")

        elif rasterio is not None and ext in {".tif", ".tiff"}:
            with rasterio.open(path) as src:
                data = src.read()
                if data.ndim == 3 and data.shape[0] in (3, 4):
                    img = data[:3, :, :].transpose(1, 2, 0)
                else:
                    img = data[0, :, :]
                _SESSION.axis.imshow(img, extent=src.bounds)
            return f"Raster layer added: {path}" + (
                " (Note: labels on raster layers are not standard)" if label else "")
        else:
            return f"Error: unsupported layer type: {path}"

    except Exception as exc:
        logger.exception("Failed to add layer %s: %s", path, exc)
        return f"Error: failed to add layer: {exc}"


@tool
def map_set_title(action_input: str) -> str:
    """
    Set map title. Input: a string.
    """
    _SESSION.ensure_initialized()
    title = action_input.split('\n')[0].strip()
    _SESSION.title = title if title else None
    if _SESSION.title:
        _SESSION.axis.set_title(_SESSION.title, color=_SESSION.title_color)
        return f"Title set to: {_SESSION.title}"
    return "Title cleared"


@tool
def map_set_background_color(action_input: str) -> str:
    """
    Set background. Input: color string.
    """
    _SESSION.ensure_initialized()
    color = _normalize_color(action_input.split('\n')[0].strip() or _SESSION.background_color)
    _SESSION.background_color = color
    if _SESSION.figure is not None and _SESSION.axis is not None:
        _SESSION.figure.patch.set_facecolor(color)
        _SESSION.axis.set_facecolor(color)
    return f"Background color set to {color}"


@tool
def map_save(action_input: str) -> str:
    """
    Save the map image. Input: output path (e.g., './map_output.jpg').
    """
    _SESSION.ensure_initialized()
    raw_path = action_input.split('\n')[0].strip()
    output = _clean_path_arg(raw_path) or "./map_output.jpg"
    out_dir = os.path.dirname(output) or "."
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    try:
        _SESSION.axis.set_xticks([])
        _SESSION.axis.set_yticks([])
        _SESSION.axis.set_xlabel("")
        _SESSION.axis.set_ylabel("")
        _SESSION.axis.spines["top"].set_visible(False)
        _SESSION.axis.spines["right"].set_visible(False)
        _SESSION.axis.spines["left"].set_visible(False)
        _SESSION.axis.spines["bottom"].set_visible(False)
    except Exception:
        pass

    _SESSION.figure.savefig(output, bbox_inches="tight")
    return f"Map saved to: {os.path.abspath(output)}"


# ---------------------
# Tool registry & prompt specs
# ---------------------

ToolFunc = Union[
    # All tools take a single input string and return a string Observation
    callable,
]

TOOLS: Dict[str, ToolFunc] = {
    "map_initial": map_initial,
    "modify_line_width": modify_line_width,
    "modify_line_color": modify_line_color,
    "modify_polygon_edge_color": modify_polygon_edge_color,
    "modify_polygon_face_color": modify_polygon_face_color,
    "modify_point_color": modify_point_color,
    "modify_point_size": modify_point_size,
    "modify_title_color": modify_title_color,
    "map_add_layer": map_add_layer,
    "map_add_legend": map_add_legend,
    "map_set_title": map_set_title,
    "map_set_background_color": map_set_background_color,
    "map_save": map_save,
}


def get_tool_names() -> List[str]:
    return list(TOOLS.keys())


def get_tools_prompt_string() -> str:
    """Return a concise human-readable description of tools for prompt injection.

    Keep it simple: one line per tool with input expectations.
    """
    return (
        "\n".join(
            [
                "map_initial: Initialize the map. Input: JSON array or comma-separated data paths (optional).",
                "modify_line_width: Set line width. Input: float, e.g., '1.5'.",
                "modify_line_color: Set line color. Input: color string or hex, e.g., 'white' or '#ffffff'.",
                "modify_polygon_edge_color: Set polygon edge color. Input: color string.",
                "modify_polygon_face_color: Set polygon face color. Input: color string or 'none'.",
                "modify_point_color: Set point color. Input: color string.",
                "modify_point_size: Set point size. Input: number.",
                "modify_title_color: Set title color. Input: color string.",
                # "map_add_layer: Add a layer by path (.shp/.geojson/.gpkg/.tif). Input: absolute path.",
                "map_add_layer: Add a layer. Input: path string, OR a JSON string like '{\"path\": \"/path/to/file.shp\", \"label\": \"My Label\"}' to add a legend label.",
                "map_add_legend: Add a legend for all layers that were given a label. Input: optional JSON for styling, e.g., '{\"title\": \"Legend\", \"loc\": \"Location\"}'.",
                "map_set_title: Set map title. Input: a string.",
                "map_set_background_color: Set background. Input: color string.",
                "map_save: Save the map image. Input: output path (e.g., './map_output.jpg').",
            ]
        )
    )


def call_tool(action_name: str, action_input: str) -> str:
    tool = TOOLS.get(action_name)
    if tool is None:
        return f"Error: unknown tool '{action_name}'"
    return tool(action_input)


def get_tools_list() -> List:
    return [
        map_initial,
        modify_line_width,
        modify_line_color,
        modify_polygon_edge_color,
        modify_polygon_face_color,
        modify_point_color,
        modify_point_size,
        modify_title_color,
        map_add_layer,
        map_add_legend,
        map_set_title,
        map_set_background_color,
        map_save,
    ]


def get_tools_by_name() -> Dict[str, any]:
    tools = get_tools_list()
    return {t.name: t for t in tools}