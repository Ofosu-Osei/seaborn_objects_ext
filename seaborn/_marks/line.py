from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Mapping, Any
import collections
import logging
import numpy as np
import matplotlib as mpl
import scipy.optimize
from seaborn._marks.base import (
    Mark,
    Mappable,
    MappableFloat,
    MappableString,
    MappableColor,
    resolve_properties,
    resolve_color,
    document_properties,
)


@document_properties
@dataclass
class Path(Mark):
    """
    A mark connecting data points in the order they appear.

    See also
    --------
    Line : A mark connecting data points with sorting along the orientation axis.
    Paths : A faster but less-flexible mark for drawing many paths.

    Examples
    --------
    .. include:: ../docstrings/objects.Path.rst

    """

    color: MappableColor = Mappable("C0")
    alpha: MappableFloat = Mappable(1)
    linewidth: MappableFloat = Mappable(rc="lines.linewidth")
    linestyle: MappableString = Mappable(rc="lines.linestyle")
    marker: MappableString = Mappable(rc="lines.marker")
    pointsize: MappableFloat = Mappable(rc="lines.markersize")
    fillcolor: MappableColor = Mappable(depend="color")
    edgecolor: MappableColor = Mappable(depend="color")
    edgewidth: MappableFloat = Mappable(rc="lines.markeredgewidth")

    _sort: ClassVar[bool] = False

    def _plot(self, split_gen, scales, orient):
        for keys, data, ax in split_gen(keep_na=not self._sort):
            vals = resolve_properties(self, keys, scales)
            vals["color"] = resolve_color(self, keys, scales=scales)
            vals["fillcolor"] = resolve_color(self, keys, prefix="fill", scales=scales)
            vals["edgecolor"] = resolve_color(self, keys, prefix="edge", scales=scales)

            if self._sort:
                data = data.sort_values(orient, kind="mergesort")

            artist_kws = self.artist_kws.copy()
            self._handle_capstyle(artist_kws, vals)

            line = mpl.lines.Line2D(
                data["x"].to_numpy(),
                data["y"].to_numpy(),
                color=vals["color"],
                linewidth=vals["linewidth"],
                linestyle=vals["linestyle"],
                marker=vals["marker"],
                markersize=vals["pointsize"],
                markerfacecolor=vals["fillcolor"],
                markeredgecolor=vals["edgecolor"],
                markeredgewidth=vals["edgewidth"],
                **artist_kws,
            )
            ax.add_line(line)

    def _legend_artist(self, variables, value, scales):
        keys = {v: value for v in variables}
        vals = resolve_properties(self, keys, scales)
        vals["color"] = resolve_color(self, keys, scales=scales)
        vals["fillcolor"] = resolve_color(self, keys, prefix="fill", scales=scales)
        vals["edgecolor"] = resolve_color(self, keys, prefix="edge", scales=scales)

        artist_kws = self.artist_kws.copy()
        self._handle_capstyle(artist_kws, vals)

        return mpl.lines.Line2D(
            [],
            [],
            color=vals["color"],
            linewidth=vals["linewidth"],
            linestyle=vals["linestyle"],
            marker=vals["marker"],
            markersize=vals["pointsize"],
            markerfacecolor=vals["fillcolor"],
            markeredgecolor=vals["edgecolor"],
            markeredgewidth=vals["edgewidth"],
            **artist_kws,
        )

    def _handle_capstyle(self, kws, vals):
        # Work around for this matplotlib issue:
        # https://github.com/matplotlib/matplotlib/issues/23437
        if vals["linestyle"][1] is None:
            capstyle = kws.get("solid_capstyle", mpl.rcParams["lines.solid_capstyle"])
            kws["dash_capstyle"] = capstyle


@document_properties
@dataclass
class Line(Path):
    """
    A mark connecting data points with sorting along the orientation axis.

    See also
    --------
    Path : A mark connecting data points in the order they appear.
    Lines : A faster but less-flexible mark for drawing many lines.

    Examples
    --------
    .. include:: ../docstrings/objects.Line.rst

    """

    _sort: ClassVar[bool] = True


@document_properties
@dataclass
class Paths(Mark):
    """
    A faster but less-flexible mark for drawing many paths.

    See also
    --------
    Path : A mark connecting data points in the order they appear.

    Examples
    --------
    .. include:: ../docstrings/objects.Paths.rst

    """

    color: MappableColor = Mappable("C0")
    alpha: MappableFloat = Mappable(1)
    linewidth: MappableFloat = Mappable(rc="lines.linewidth")
    linestyle: MappableString = Mappable(rc="lines.linestyle")

    _sort: ClassVar[bool] = False

    def __post_init__(self):
        # LineCollection artists have a capstyle property but don't source its value
        # from the rc, so we do that manually here. Unfortunately, because we add
        # only one LineCollection, we have the use the same capstyle for all lines
        # even when they are dashed. It's a slight inconsistency, but looks fine IMO.
        self.artist_kws.setdefault("capstyle", mpl.rcParams["lines.solid_capstyle"])

    def _plot(self, split_gen, scales, orient):
        line_data = {}
        for keys, data, ax in split_gen(keep_na=not self._sort):
            if ax not in line_data:
                line_data[ax] = {
                    "segments": [],
                    "colors": [],
                    "linewidths": [],
                    "linestyles": [],
                }

            segments = self._setup_segments(data, orient)
            line_data[ax]["segments"].extend(segments)
            n = len(segments)

            vals = resolve_properties(self, keys, scales)
            vals["color"] = resolve_color(self, keys, scales=scales)

            line_data[ax]["colors"].extend([vals["color"]] * n)
            line_data[ax]["linewidths"].extend([vals["linewidth"]] * n)
            line_data[ax]["linestyles"].extend([vals["linestyle"]] * n)

        for ax, ax_data in line_data.items():
            lines = mpl.collections.LineCollection(**ax_data, **self.artist_kws)
            # Handle datalim update manually
            # https://github.com/matplotlib/matplotlib/issues/23129
            ax.add_collection(lines, autolim=False)
            if ax_data["segments"]:
                xy = np.concatenate(ax_data["segments"])
                ax.update_datalim(xy)

    def _legend_artist(self, variables, value, scales):
        key = resolve_properties(self, {v: value for v in variables}, scales)

        artist_kws = self.artist_kws.copy()
        capstyle = artist_kws.pop("capstyle")
        artist_kws["solid_capstyle"] = capstyle
        artist_kws["dash_capstyle"] = capstyle

        return mpl.lines.Line2D(
            [],
            [],
            color=key["color"],
            linewidth=key["linewidth"],
            linestyle=key["linestyle"],
            **artist_kws,
        )

    def _setup_segments(self, data, orient):
        if self._sort:
            data = data.sort_values(orient, kind="mergesort")

        # Column stack to avoid block consolidation
        xy = np.column_stack([data["x"], data["y"]])

        return [xy]


@document_properties
@dataclass
class Lines(Paths):
    """
    A faster but less-flexible mark for drawing many lines.

    See also
    --------
    Line : A mark connecting data points with sorting along the orientation axis.

    Examples
    --------
    .. include:: ../docstrings/objects.Lines.rst

    """

    _sort: ClassVar[bool] = True


@document_properties
@dataclass
class Range(Paths):
    """
    An oriented line mark drawn between min/max values.

    Examples
    --------
    .. include:: ../docstrings/objects.Range.rst

    """

    def _setup_segments(self, data, orient):
        # TODO better checks on what variables we have
        # TODO what if only one exist?
        val = {"x": "y", "y": "x"}[orient]
        if not set(data.columns) & {f"{val}min", f"{val}max"}:
            agg = {f"{val}min": (val, "min"), f"{val}max": (val, "max")}
            data = data.groupby(orient).agg(**agg).reset_index()

        cols = [orient, f"{val}min", f"{val}max"]
        data = data[cols].melt(orient, value_name=val)[["x", "y"]]
        segments = [d.to_numpy() for _, d in data.groupby(orient)]
        return segments


@document_properties
@dataclass
class Dash(Paths):
    """
    A line mark drawn as an oriented segment for each datapoint.

    Examples
    --------
    .. include:: ../docstrings/objects.Dash.rst

    """

    width: MappableFloat = Mappable(0.8, grouping=False)

    def _setup_segments(self, data, orient):
        ori = ["x", "y"].index(orient)
        xys = data[["x", "y"]].to_numpy().astype(float)
        segments = np.stack([xys, xys], axis=1)
        segments[:, 0, ori] -= data["width"] / 2
        segments[:, 1, ori] += data["width"] / 2
        return segments


@dataclass
class LineLabel(Mark):
    text: MappableString = Mappable("")
    color: MappableColor = Mappable("k")
    alpha: MappableFloat = Mappable(1)
    fontsize: MappableFloat = Mappable(rc="font.size")
    offset: float = 4
    additional_distance_offset: float = 0

    def _compute_target_positions(
        self,
        data: dict[mpl.axes.Axes, list[Mapping[str, Any]]],  # pyright: ignore
        scales,
        other: str,
        *,
        offset: float = 0,
    ) -> dict[mpl.axes.Axes, np.ndarray]:  # pyright: ignore
        """Solves a constrained optimization problem to determine the optimal target point positions."""
        # https://github.com/nschloe/matplotx/blob/main/src/matplotx/_labels.py
        target_positions: dict[mpl.axes.Axes, np.ndarray] = {}  # pyright: ignore
        point_dtype = np.dtype([("x", "f8"), ("y", "f8")])

        def _resolve_fontsize(keys) -> float:
            return resolve_properties(self, keys, scales)["fontsize"]

        for ax, rows in data.items():
            # Calculate offsets for each target based on the fontsize and additional offset.
            offsets = np.array(
                [[_resolve_fontsize(row["_keys"]) * (5 / 3) + offset] for row in rows]
            )
            # Transform points to screen coordinates so min_distance_apart is scale-agnostic.
            points = ax.transData.transform([(row["x"], row["y"]) for row in rows])
            points = points.view(point_dtype)
            # Record the sorting indices so we can recover the original order.
            sorted_indexes = np.argsort(points, axis=0, order=other)
            original_indexes = np.argsort(sorted_indexes, axis=0)
            sorted_offsets = np.take_along_axis(offsets, sorted_indexes, axis=0)
            sorted_points = np.take_along_axis(points, sorted_indexes, axis=0)

            # Calculate min y0 position to bootstrap the first index.
            num_points = points.size
            min_point = sorted_points[other][0] - num_points * sorted_offsets[0]

            # Solve non-negative least squares problem
            A = np.tril(np.ones((num_points, num_points)))
            b = sorted_points[other].squeeze(1) - (
                min_point + np.arange(num_points) * sorted_offsets.squeeze(1)
            )
            sol, objective_value = scipy.optimize.nnls(A, b)
            # Recover points
            sol = (
                np.cumsum(sol)
                + min_point
                + np.arange(num_points) * sorted_offsets.squeeze(1)
            )
            sol = np.take_along_axis(sol[:, np.newaxis], original_indexes, axis=0)
            logging.info(
                "Found line label positions with final objective value: %f",
                objective_value,
            )

            # Update original points
            points[other] = sol

            # Transform back to data coordinates
            screen_to_data = ax.transData.inverted()
            target_positions[ax] = screen_to_data.transform(points.view("f8")).view(
                point_dtype
            )

        return target_positions

    def _plot(self, split_gen, scales, orient):
        data_by_axes: dict[
            mpl.axes.Axes, list[Mapping[str, Any]]
        ] = collections.defaultdict(
            list
        )  # pyright: ignore

        other = {"x": "y", "y": "x"}[orient]
        for keys, data, ax in split_gen():
            records = data.query(f"`{orient}` == {orient}.max()").to_dict("records")
            records = collections.ChainMap(*records, {"_keys": keys})
            data_by_axes[ax].append(records)

        target_positions = self._compute_target_positions(
            data_by_axes,
            scales,
            other,
            offset=self.additional_distance_offset,
        )
        for ax, data in data_by_axes.items():
            for idx, row in enumerate(data):
                vals = resolve_properties(self, row["_keys"], scales)
                color = resolve_color(self, row["_keys"], "", scales)
                fontsize = vals["fontsize"]

                transform = mpl.transforms.offset_copy(  # pyright: ignore
                    ax.transData,
                    fig=ax.figure,
                    x=self.offset if orient == "x" else 0,
                    y=self.offset if orient == "y" else 0,
                    units="points",
                )

                target_position = target_positions[ax][idx]
                ax.add_artist(
                    mpl.text.Text(  # pyright: ignore
                        x=target_position["x"],
                        y=target_position["y"],
                        text=str(row.get("text", vals["text"])),
                        color=color,
                        fontsize=fontsize,
                        horizontalalignment="left" if orient == "x" else "center",
                        verticalalignment="center" if orient == "x" else "bottom",
                        transform=transform,
                        rotation=90 if orient == "y" else 0,
                        zorder=2,
                        clip_on=False,
                        in_layout=True,
                        **self.artist_kws,
                    )
                )
