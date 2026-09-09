"""A spectral optics playground using actual simulated photon trajectories.

Synthetic scenes exercise the general optical transport. Path images are x/y
projections, not camera radiance. Full statistics and sampled paths are labeled
separately. No trajectories or fluorescence delays are invented for display.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from io import BytesIO
from pathlib import Path
import time

import numpy as np

from chroma.event import RAYLEIGH_SCATTER, SURFACE_REEMIT, SURFACE_DETECT
from chroma.triton.optical_response import TabulatedCDF, uniform, validate_seed
from chroma.triton.runtime import PhotonBatch

SCENES = {
    "prism": "01 · Dispersive prism",
    "fluorescence": "02 · Fluorescent coating",
    "rayleigh": "03 · Rayleigh scattering",
}


@dataclass(frozen=True)
class PlaygroundScene:
    name: str
    title: str
    geometry: object
    wavelengths: np.ndarray
    source_center: tuple
    source_band: tuple
    outline: np.ndarray | None
    explanation: str
    fluorescence_time: TabulatedCDF | None = None

    def photons(self, count, seed=901, polarization="random"):
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("photon count must be a positive integer")
        if polarization not in ("random", "y", "z"):
            raise ValueError("polarization must be random, y, or z")
        seed = validate_seed(seed)
        ids = np.arange(count, dtype=np.int64)
        pos = np.broadcast_to(self.source_center, (count, 3)).astype(np.float32).copy()
        pos[:, 1] += (uniform(ids, seed, 0x10000000) - 0.5) * 2.0
        pos[:, 2] += (uniform(ids, seed, 0x10000001) - 0.5) * 2.0
        direction = np.zeros_like(pos)
        direction[:, 0] = 1.0
        pol = np.zeros_like(pos)
        if polarization == "random":
            angle = 2 * np.pi * uniform(ids, seed, 0x10000002)
            pol[:, 1], pol[:, 2] = np.cos(angle), np.sin(angle)
        else:
            pol[:, 1 if polarization == "y" else 2] = 1.0
        low, high = self.source_band
        wavelengths = low + (high - low) * uniform(ids, seed, 0x10000003)
        return PhotonBatch(
            pos=pos,
            direction=direction,
            polarization=pol,
            wavelengths=wavelengths,
            times=np.zeros(count, np.float32),
            last_hit_triangles=np.full(count, -1, np.int32),
            flags=np.zeros(count, np.uint32),
            weights=np.ones(count, np.float32),
            event_indices=np.zeros(count, np.uint32),
            global_photon_ids=ids,
            channels=np.zeros(count, np.uint32),
        )


def build_playground_scene(name="prism"):
    """Three explicit synthetic experiments; no external optics database."""
    from chroma import make
    from chroma.detector import Detector
    from chroma.geometry import Material, Solid, Surface

    if name not in SCENES:
        raise ValueError("unknown optical experiment: " + str(name))
    wavelengths = np.arange(280.0, 741.0, 2.0, dtype=np.float32)

    def material(label, index=1.0, scattering=np.inf):
        result = Material(label)
        result.set("refractive_index", index, wavelengths)
        result.set("absorption_length", np.inf, wavelengths)
        result.set("scattering_length", scattering, wavelengths)
        return result

    air = material("synthetic_air")
    bulk, outline, delay = air, None, None
    source, band = (-260.0, -10.0, 0.0), (390.0, 710.0)
    if name == "rayleigh":
        bulk = material("synthetic_rayleigh_medium", scattering=100.0 * (wavelengths / 450.0) ** 4)
        source = (-290.0, 0.0, 0.0)
        title = "Rayleigh scattering"
        explanation = "Polarized Rayleigh scattering · scattering length = 100 mm × (λ / 450 nm)⁴"
    elif name == "fluorescence":
        source, band = (-260.0, 0.0, 0.0), (300.0, 310.0)
        title = "Fluorescent coating"
        explanation = "Effective fluorescent coating · UV absorption · 85% reemission probability · tabulated delay"
    else:
        title = "Dispersive prism"
        explanation = "Dispersive dielectric · polarized Fresnel reflection / transmission · spectral group velocity"
    geometry = Detector(bulk)
    monitor = Surface("ideal_boundary_monitor")
    monitor.set("detect", 1.0, wavelengths)
    # One ideal channel records the enclosing measurement box, without PMT
    # response or electronics. All transport uses the real spectral kernel.
    geometry.add_pmt(
        Solid(make.box(600.0, 400.0, 240.0), bulk, air, surface=monitor),
        displacement=(0.0, 0.0, 0.0),
    )
    if name == "prism":
        glass = material("synthetic_dispersive_glass", 1.40 + 0.035 / (wavelengths / 1000.0) ** 2)
        # c / (n - λ dn/dλ), with the same micrometre units as the Cauchy law.
        glass.set(
            "group_velocity", 299.792458 / (1.40 + 0.105 / (wavelengths / 1000.0) ** 2), wavelengths
        )
        outline = np.array([[-70.0, -80.0], [70.0, -80.0], [0.0, 90.0]])
        mesh = make.linear_extrude(outline[:, 0], outline[:, 1], 120.0)
        geometry.add_solid(Solid(mesh, glass, air))
    elif name == "fluorescence":
        fluorescent = Surface("synthetic_fluorescent_coating", model=2)
        fluorescent.set("absorb", np.where(wavelengths < 380.0, 1.0, 0.0), wavelengths)
        fluorescent.set("reemit", 0.85, wavelengths)
        emission = TabulatedCDF.from_pdf([410, 450, 485, 530, 580], [0, 0.4, 1, 0.5, 0])
        fluorescent.set("reemission_cdf", emission.evaluate(wavelengths), wavelengths)
        delay = TabulatedCDF(
            [0, 1, 10, 50, 200], [0, 0.25, 0.65, 0.9, 1], "Synthetic fluorescent surface delay"
        )
        fluorescent.reemission_time_cdf = delay
        fluorescent.reemission_to_material1 = 0.5
        geometry.add_solid(Solid(make.box(5.0, 180.0, 120.0), air, air, surface=fluorescent))
        outline = np.array([[-2.5, -90.0], [2.5, -90.0], [2.5, 90.0], [-2.5, 90.0]])
    geometry.flatten()
    return PlaygroundScene(
        name, title, geometry, wavelengths, source, band, outline, explanation, delay
    )


def wavelength_rgb(wavelengths):
    """Spectral display palette, with UV assigned violet; not colorimetry."""
    knots = [280, 380, 440, 490, 510, 580, 620, 700, 740]
    colors = np.array(
        [
            [0.64, 0.25, 1],
            [0.64, 0.25, 1],
            [0.24, 0.35, 1],
            [0.12, 0.9, 1],
            [0.2, 1, 0.45],
            [1, 0.9, 0.15],
            [1, 0.35, 0.08],
            [1, 0.12, 0.13],
            [1, 0.12, 0.13],
        ]
    )
    return np.stack([np.interp(wavelengths, knots, colors[:, i]) for i in range(3)], axis=-1)


@dataclass(frozen=True)
class PhotonPaths:
    positions: np.ndarray  # [snapshot, selected photon, xyz]
    wavelengths: np.ndarray  # AFTER the interaction at each vertex
    times: np.ndarray
    flags: np.ndarray
    photon_ids: np.ndarray
    complete: bool


def trace_photon_subset(simulation, batch, *, seed=901, count=512, max_steps=256):
    """Recover exact vertices using deterministic prefixes of the original batch.

    IDs and seed stay fixed. Prefix results are never fed back to transport:
    that would reset its random interaction index and retain STEP_LIMIT flags.
    """
    from chroma.triton.photon_input import slice_batch
    from chroma.triton.spectral import STEP_LIMIT

    if not isinstance(count, (int, np.integer)) or not 1 <= count <= batch.photon_count:
        raise ValueError("path count must lie between 1 and the full photon count")
    sample = slice_batch(batch, 0, int(count))
    position, wavelength, times = (
        [sample.pos.copy()],
        [sample.wavelengths.copy()],
        [sample.times.copy()],
    )
    flags = [sample.flags.copy()]
    completed = False
    for limit in range(1, max_steps + 1):
        result = simulation.simulate(sample, seed=seed, max_steps=limit)
        state = result.final_state
        position.append(state["pos"].copy())
        wavelength.append(state["wavelengths"].copy())
        times.append(state["times"].copy())
        flags.append(state["flags"] & ~STEP_LIMIT)
        if not result.step_limit_count:
            completed = True
            break
    return PhotonPaths(
        np.asarray(position),
        np.asarray(wavelength),
        np.asarray(times),
        np.asarray(flags),
        sample.global_photon_ids.copy(),
        completed,
    )


@dataclass(frozen=True)
class PlaygroundRun:
    scene: PlaygroundScene
    result: object
    paths: PhotonPaths
    source_wavelengths: np.ndarray
    source_seconds: float
    transport_seconds: float
    trace_seconds: float
    seed: int
    polarization: str
    fluorescence_delays: np.ndarray

    @property
    def photon_count(self):
        return len(self.result.final_state["times"])

    @property
    def photons_per_second(self):
        return self.photon_count / (self.source_seconds + self.transport_seconds)


def _path_segments(paths, time_limit=np.inf):
    """Flights use incoming wavelength; changes happen at reemission vertices."""
    begin, end = paths.positions[:-1], paths.positions[1:]
    moving = np.linalg.norm(end - begin, axis=-1) > 1.0e-5
    # A fluorescent endpoint includes a waiting time. Reveal completed segments
    # at vertices rather than inventing velocities by interpolating that delay.
    visible = moving & (paths.times[1:] <= time_limit)
    segments = np.stack((begin[..., :2], end[..., :2]), axis=2)[visible]
    return segments, wavelength_rgb(paths.wavelengths[:-1][visible])


def playground_figure(run, *, time_limit=np.inf):
    """Standalone scientific figure, with actual paths and full-count data."""
    from matplotlib.figure import Figure
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Polygon

    scene, state = run.scene, run.result.final_state
    panel, foreground, muted = "#0e1828", "#ebf1fb", "#8ca5c3"
    cyan, gold = "#55e1de", "#ffca83"
    figure = Figure(figsize=(15.5, 9.7), dpi=120, facecolor="#070c17")
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=(2.25, 1),
        wspace=0.36,
        hspace=0.34,
        left=0.055,
        right=0.965,
        top=0.84,
        bottom=0.075,
    )
    hero = figure.add_subplot(grid[0, :])
    lower = [figure.add_subplot(grid[1, i]) for i in range(3)]
    for ax in [hero, *lower]:
        ax.set_facecolor(panel)
        ax.tick_params(colors=muted, labelsize=8)
        ax.xaxis.label.set_color(muted)
        ax.yaxis.label.set_color(muted)
        ax.title.set_color(foreground)
        ax.title.set_fontsize(11)
        for spine in ax.spines.values():
            spine.set_color("#273851")
    figure.text(0.055, 0.955, scene.title, fontsize=25, color=foreground, weight="bold")
    figure.text(0.055, 0.915, scene.explanation, fontsize=10.5, color=muted)
    figure.text(0.965, 0.975, "TRICHROMA / SPECTRAL LAB", color=cyan, fontsize=9, ha="right")
    visible_time = (
        "all arrival times"
        if np.isinf(time_limit)
        else f"interactions completed by {time_limit:g} ns"
    )
    figure.text(
        0.055,
        0.874,
        f"{run.photon_count:,} simulated photons · {len(run.paths.photon_ids):,} actual paths displayed · "
        f"{visible_time} · x/y projection",
        color=muted,
        fontsize=9,
    )
    if scene.outline is not None:
        hero.add_patch(
            Polygon(
                scene.outline,
                facecolor="#243952",
                edgecolor="#8cafd1",
                alpha=0.65,
                linewidth=1.3,
                zorder=1,
            )
        )
    segments, colors = _path_segments(run.paths, time_limit)
    # Glow is an explicitly nonphysical display effect on the same real paths.
    for width, alpha in ((4.0, 0.025), (1.7, 0.06), (0.45, 0.35)):
        hero.add_collection(
            LineCollection(segments, colors=colors, linewidths=width, alpha=alpha, zorder=2)
        )
    hero.scatter(
        [scene.source_center[0]], [scene.source_center[1]], color="#ffffff", s=18, zorder=4
    )
    hero.set(xlim=(-300, 300), ylim=(-200, 200), xlabel="x [mm]", ylabel="y [mm]")
    hero.text(
        0.012,
        0.96,
        "MONTE CARLO PHOTON PATHS",
        transform=hero.transAxes,
        color=muted,
        fontsize=8,
        va="top",
    )
    hero.text(
        0.99,
        0.96,
        "Synthetic optics · ideal enclosing photon monitor",
        transform=hero.transAxes,
        color=muted,
        fontsize=8,
        va="top",
        ha="right",
    )
    if scene.name == "fluorescence":
        hero.text(15, 106, "fluorescent coating", color=cyan, fontsize=9)
        hero.text(-245, 25, "UV beam\n(violet display color)", color="#be9bff", fontsize=9)
    elif scene.name == "prism":
        hero.text(-30, 106, "dispersive glass", color=muted, fontsize=9)
    else:
        hero.text(-235, 145, "Short wavelengths scatter more often", color=muted, fontsize=10)

    ax = lower[0]
    bins = np.linspace(280, 740, 93)
    incoming, edges = np.histogram(run.source_wavelengths, bins=bins)
    monitor = (state["flags"] & SURFACE_DETECT) != 0
    outgoing, _ = np.histogram(state["wavelengths"][monitor], bins=bins)
    ax.stairs(incoming, edges, color="#a29fff", linewidth=1.2, label="Emitted")
    ax.stairs(outgoing, edges, color=cyan, fill=True, alpha=0.25)
    ax.stairs(outgoing, edges, color=cyan, linewidth=1.2, label="Reached monitor")
    ax.set(title="The spectrum", xlabel="Wavelength [nm]", ylabel="Photons / 5 nm", xlim=(280, 740))
    ax.legend(facecolor=panel, edgecolor="none", labelcolor=foreground, fontsize=8)

    ax = lower[1]
    if scene.name == "prism":
        forward = monitor & (state["pos"][:, 0] > 299.0)
        angle = np.rad2deg(
            np.arctan2(state["direction"][forward, 1], state["direction"][forward, 0])
        )
        density, x, y = np.histogram2d(
            state["wavelengths"][forward],
            angle,
            bins=[np.linspace(380, 720, 86), np.linspace(-80, 30, 111)],
        )
        if np.any(density):
            ax.pcolormesh(
                x,
                y,
                density.T,
                norm=LogNorm(vmin=1, vmax=max(density.max(), 2)),
                cmap="magma",
                rasterized=True,
            )
        ax.set(
            title="Dispersion at the forward monitor",
            xlabel="Wavelength [nm]",
            ylabel="Exit angle in x/y [degrees]",
        )
    elif scene.name == "fluorescence":
        values, edges = np.histogram(run.fluorescence_delays, bins=np.linspace(0, 200, 81))
        ax.stairs(values, edges, color=gold, fill=True, alpha=0.25)
        ax.stairs(values, edges, color=gold, linewidth=1.2)
        ax.set(
            title="Actual fluorescent delays",
            xlabel="Arrival − geometric flight time [ns]",
            ylabel="Reemitted photons / bin",
        )
    else:
        bins = np.linspace(390, 710, 33)
        total, _ = np.histogram(run.source_wavelengths, bins=bins)
        scattered = (state["flags"] & RAYLEIGH_SCATTER) != 0
        count, _ = np.histogram(run.source_wavelengths[scattered], bins=bins)
        mid = (bins[:-1] + bins[1:]) / 2
        ratio = np.divide(count, total, out=np.zeros_like(count, dtype=float), where=total > 0)
        ax.scatter(mid, ratio, c=wavelength_rgb(mid), s=17, label="Actual ≥1 scatter")
        path = 300.0 - scene.source_center[0]
        ax.plot(
            mid,
            1 - np.exp(-path / (100 * (mid / 450) ** 4)),
            color=foreground,
            linestyle="--",
            linewidth=1,
            label="1 − exp[−L / ℓ(λ)]",
        )
        ax.set(
            title="Blue scatters more often",
            xlabel="Wavelength [nm]",
            ylabel="Fraction scattered ≥1 time",
            ylim=(0, 1.05),
        )
        ax.legend(facecolor=panel, edgecolor="none", labelcolor=foreground, fontsize=7)
    ax = lower[2]
    times = state["times"][monitor]
    if scene.name == "fluorescence":
        edges, title = np.linspace(0, 205, 83), "Afterglow at the monitor"
    else:
        upper = max(float(np.percentile(times, 99.8)), 1.0) if len(times) else 5.0
        edges, title = np.linspace(0, upper, 81), "Photon flight times (central 99.8%)"
    values, edges = np.histogram(times, bins=edges)
    ax.stairs(values, edges, color=cyan, fill=True, alpha=0.25)
    ax.stairs(values, edges, color=cyan, linewidth=1.2)
    ax.set(title=title, xlabel="Arrival time [ns]", ylabel="Monitor photons / bin")
    for ax in lower:
        ax.grid(color="#31425a", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
    figure.text(
        0.055,
        0.018,
        "Colors encode wavelength; UV is violet. Glow and projection are display choices, not camera radiance. "
        "Histograms use all simulated photons; paths use the labeled subset.",
        color=muted,
        fontsize=8,
    )
    return figure


def figure_png(figure):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    buffer = BytesIO()
    FigureCanvasAgg(figure).print_png(buffer)
    return buffer.getvalue()


class OpticalShowcase:
    """Reuse scene resources; count changes rerun physics, time gates only redraw."""

    def __init__(self, *, device="cuda", backend="triton"):
        self.device, self.backend = device, backend
        self._simulations = {}
        self.event = None
        self._widget = None
        self._callbacks = []

    def experiment(self, name):
        if name not in self._simulations:
            from chroma.triton.spectral import SpectralSimulation

            scene = build_playground_scene(name)
            simulation = SpectralSimulation(
                scene.geometry,
                wavelengths=scene.wavelengths,
                backend=self.backend,
                device=self.device,
                tile_size=262144,
            )
            self._simulations[name] = (scene, simulation)
        return self._simulations[name]

    def run(
        self,
        name="prism",
        photons=2_500_000,
        *,
        seed=901,
        paths=512,
        polarization="random",
        max_steps=256,
    ):
        if (
            isinstance(photons, bool)
            or not isinstance(photons, (int, np.integer))
            or not 1 <= photons <= 30_000_000
        ):
            raise ValueError("optical photons must be an integer from 1 to 30,000,000")
        if not isinstance(paths, (int, np.integer)) or not 1 <= paths <= min(photons, 2048):
            raise ValueError("drawn path count must be from 1 to min(photons, 2048)")
        seed = validate_seed(seed)
        scene, simulation = self.experiment(name)
        warmup = simulation.simulate(
            scene.photons(16, seed, polarization), seed=seed, max_steps=max_steps
        )
        if warmup.step_limit_count:
            raise RuntimeError("warmup did not finish; increase max_steps")
        begin = time.perf_counter()
        batch = scene.photons(int(photons), seed, polarization)
        generated = time.perf_counter()
        result = simulation.simulate(batch, seed=seed, max_steps=max_steps)
        completed = time.perf_counter()
        if result.step_limit_count:
            raise RuntimeError(
                f"{result.step_limit_count} photons reached max_steps; increase the limit"
            )
        selected_paths = trace_photon_subset(
            simulation, batch, seed=seed, count=paths, max_steps=max_steps
        )
        traced = time.perf_counter()
        if not selected_paths.complete:
            raise RuntimeError("trajectory subset did not reach terminal states")
        for field, value in (
            ("pos", selected_paths.positions[-1]),
            ("wavelengths", selected_paths.wavelengths[-1]),
            ("times", selected_paths.times[-1]),
            ("flags", selected_paths.flags[-1]),
        ):
            if not np.array_equal(value, result.final_state[field][:paths]):
                raise RuntimeError(f"trajectory prefix and full event disagree in {field}")
        delays = np.empty(0)
        if name == "fluorescence":
            state = result.final_state
            reemitted = (state["flags"] & SURFACE_REEMIT) != 0
            # All UV first hits x=-2.5 mm, then travels straight in n=1. Visible
            # photons do not reabsorb or scatter in this synthetic scene.
            origin = batch.pos[reemitted].astype(float)
            conversion = origin.copy()
            conversion[:, 0] = -2.5
            distance = (
                conversion[:, 0]
                - origin[:, 0]
                + np.linalg.norm(state["pos"][reemitted] - conversion, axis=1)
            )
            delays = state["times"][reemitted] - distance / 299.792458
            if np.any(delays < -1.0e-4):
                raise RuntimeError("flight-subtracted fluorescent delay is unexpectedly negative")
            delays = np.maximum(delays, 0)
        self.event = PlaygroundRun(
            scene,
            result,
            selected_paths,
            batch.wavelengths.copy(),
            generated - begin,
            completed - generated,
            traced - completed,
            seed,
            polarization,
            delays,
        )
        return self.event

    def save_figure(self, path, **options):
        if self.event is None:
            raise RuntimeError("simulate an event before saving its figure")
        Path(path).write_bytes(figure_png(playground_figure(self.event, **options)))

    def show(self, *, photons=2_500_000, seed=901, auto_run=True):
        import ipywidgets as widgets
        from IPython.display import display

        if self._widget is not None:
            display(self._widget)
            return self._widget
        layout = widgets.Layout
        selection = widgets.Dropdown(
            options=[(title, name) for name, title in SCENES.items()],
            value="prism",
            description="Experiment",
            layout=layout(width="430px"),
        )
        count = widgets.BoundedIntText(
            value=photons,
            min=1000,
            max=30_000_000,
            step=100_000,
            description="Optical photons",
            style={"description_width": "105px"},
            layout=layout(width="290px"),
        )
        paths = widgets.IntSlider(
            value=512,
            min=32,
            max=2048,
            step=32,
            description="Drawn paths",
            continuous_update=False,
            layout=layout(width="300px"),
        )
        seed_widget = widgets.BoundedIntText(
            value=seed, min=0, max=2**32 - 1, description="Seed", layout=layout(width="190px")
        )
        polarization = widgets.Dropdown(
            options=[("Random linear", "random"), ("Linear y", "y"), ("Linear z", "z")],
            description="Polarization",
            layout=layout(width="260px"),
        )
        run_button = widgets.Button(
            description="Trace the light",
            button_style="info",
            icon="bolt",
            layout=layout(width="180px", height="36px"),
        )
        time_gate = widgets.FloatSlider(
            value=210,
            min=0,
            max=210,
            step=0.2,
            description="Time [ns]",
            continuous_update=False,
            layout=layout(width="420px"),
        )
        all_times = widgets.Checkbox(value=True, description="Show all times", indent=False)
        image = widgets.Image(format="png", layout=layout(width="100%"))
        metrics, status = widgets.HTML(), widgets.HTML()
        title = widgets.HTML(
            """<div style="background:linear-gradient(115deg,#10243b,#211635);padding:26px 30px;
                color:#e7effb;border-radius:14px;margin-bottom:14px;font-family:system-ui">
                <div style="font-size:11px;letter-spacing:3px;color:#58e4dd">TRICHROMA / SPECTRAL LAB</div>
                <div style="font-size:32px;font-weight:650;margin-top:6px">Let there be color.</div>
                <div style="color:#a6bfd4;margin-top:8px">Disperse a beam. Shift its wavelength.
                Watch blue light scatter. Every path comes from the optical simulation.</div></div>"""
        )
        controls = widgets.VBox(
            [
                widgets.HBox(
                    [selection, run_button], layout=layout(flex_flow="row wrap", gap="15px")
                ),
                widgets.HBox(
                    [count, paths, seed_widget], layout=layout(flex_flow="row wrap", gap="10px")
                ),
                widgets.HBox(
                    [polarization, time_gate, all_times],
                    layout=layout(flex_flow="row wrap", gap="10px"),
                ),
            ]
        )
        footer = widgets.HTML(
            "<p style='font-size:12px;color:#8097ad'>Synthetic optical tables. The photon count controls full "
            "Monte Carlo transport; drawn paths are a stable subset of those same photons. Colors encode "
            "wavelength and UV is assigned violet. Time controls reveal segments after their endpoint interaction and do not "
            "rerun physics. This is a projected photon-path visualization, not a camera-radiance prediction. "
            "Fluorescence uses the implemented effective surface model; bulk reemission is unsupported.</p>"
        )
        self._widget = widgets.VBox(
            [title, controls, status, metrics, image, footer], layout=layout(max_width="1500px")
        )
        self.controls = dict(
            experiment=selection,
            photons=count,
            paths=paths,
            seed=seed_widget,
            polarization=polarization,
            run=run_button,
            time=time_gate,
            all_times=all_times,
            image=image,
            status=status,
            metrics=metrics,
        )
        self._redraw_hold = False

        def redraw(_=None):
            if self.event is not None and not self._redraw_hold:
                limit = np.inf if all_times.value else time_gate.value
                image.value = figure_png(playground_figure(self.event, time_limit=limit))

        def simulate(_=None):
            run_button.disabled = True
            status.value = "<p>Tracing photons, then recording a subset of their actual interaction vertices…</p>"
            try:
                event = self.run(
                    selection.value,
                    count.value,
                    seed=seed_widget.value,
                    paths=min(paths.value, count.value),
                    polarization=polarization.value,
                )
                # A prism event lasts a few ns; fluorescence can last 200 ns.
                # Keep the time slider useful for the actual recorded paths.
                self._redraw_hold = True
                time_gate.max = max(1.0, float(np.ceil(event.paths.times[-1].max() * 1.01)))
                time_gate.step = max(0.001, time_gate.max / 500)
                time_gate.value = time_gate.max
                self._redraw_hold = False
                flags = event.result.final_state["flags"]
                features = [
                    ("SIMULATED", f"{event.photon_count:,}"),
                    ("MONITOR ARRIVALS", f"{int(np.count_nonzero(flags & SURFACE_DETECT)):,}"),
                    ("SCATTERED ≥1×", f"{int(np.count_nonzero(flags & RAYLEIGH_SCATTER)):,}"),
                    ("REEMITTED ≥1×", f"{int(np.count_nonzero(flags & SURFACE_REEMIT)):,}"),
                    (
                        "SOURCE + TRANSPORT",
                        f"{event.source_seconds + event.transport_seconds:.3f} s",
                    ),
                ]
                metrics.value = (
                    '<div style="display:flex;flex-wrap:wrap;gap:10px;margin:12px 0">'
                    + "".join(
                        f'<div style="background:#102033;border:1px solid #233950;border-radius:10px;padding:14px;flex:1">'
                        f'<div style="font:10px system-ui;color:#8ba5be;white-space:nowrap">{name}</div>'
                        f'<div style="font:600 21px system-ui;color:#e7effb;margin-top:5px">{value}</div></div>'
                        for name, value in features
                    )
                    + "</div>"
                )
                status.value = (
                    f"<p>Seed {event.seed} · {event.photon_count:,} optical photons · "
                    f"{event.photons_per_second / 1e6:.2f}M photons/s including host source + terminal-state download. "
                    f"Recording {len(event.paths.photon_ids):,} paths took {event.trace_seconds:.2f} s separately. "
                    "Setup, plotting and notebook transfer are excluded.</p>"
                )
                redraw()
            except Exception as error:
                status.value = "<p><b>Simulation failed:</b> " + escape(str(error)) + "</p>"
                raise
            finally:
                self._redraw_hold = False
                run_button.disabled = False

        for control in (time_gate, all_times):
            control.observe(redraw, names="value")
        run_button.on_click(simulate)
        self._callbacks.append((run_button, simulate))
        display(self._widget)
        if auto_run:
            simulate()
        elif self.event is not None:
            redraw()
        return self._widget

    def close(self):
        for button, callback in self._callbacks:
            button.on_click(callback, remove=True)
        self._callbacks.clear()
        if self._widget is not None:
            widgets = [self._widget]
            for widget in widgets:
                widgets.extend(getattr(widget, "children", ()))
            for widget in reversed(widgets):
                widget.unobserve_all()
                widget.close()
            self._widget = None
