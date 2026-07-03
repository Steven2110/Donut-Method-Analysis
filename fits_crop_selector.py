from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "processed" / "crops"


@dataclass
class CropSelection:
    parent_path: Path
    frame_index: int
    x_center: int
    y_center: int
    x_start: int
    x_stop: int
    y_start: int
    y_stop: int
    raw_crop: np.ndarray
    background_adu: float
    background_rms_adu: float
    signal: np.ndarray
    fit_mask: np.ndarray
    positive_flux_adu: float
    x_centroid_crop: float
    y_centroid_crop: float


def list_fits_files(data_dir: Path) -> list[Path]:
    return sorted(Path(data_dir).glob("*.fits"))


def load_fits_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                image = np.array(hdu.data, dtype=np.float32, copy=True)
                header = hdu.header.copy()
                break
        else:
            raise ValueError(f"No image data found in {path}")

    image = np.squeeze(image)
    while image.ndim > 2:
        image = image[0]

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D FITS image in {path}, got {image.shape}")

    return image, header


def display_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0

    vmin, vmax = np.percentile(finite, [1.0, 99.5])
    if np.isclose(vmin, vmax):
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def crop_bounds(
    image_shape: tuple[int, int],
    x_center: int,
    y_center: int,
    crop_size: int,
) -> tuple[int, int, int, int]:
    half = crop_size // 2
    x_start = int(x_center) - half
    x_stop = x_start + crop_size
    y_start = int(y_center) - half
    y_stop = y_start + crop_size

    ny, nx = image_shape
    if x_start < 0 or y_start < 0 or x_stop > nx or y_stop > ny:
        raise ValueError(
            f"Crop centered at x={x_center}, y={y_center} would leave image bounds."
        )

    return x_start, x_stop, y_start, y_stop


def preprocess_crop(raw_crop: np.ndarray) -> dict[str, object]:
    crop = np.asarray(raw_crop, dtype=np.float64)
    yy, xx = np.indices(crop.shape)
    x_seed = (crop.shape[1] - 1) / 2.0
    y_seed = (crop.shape[0] - 1) / 2.0
    radius_seed = np.hypot(xx - x_seed, yy - y_seed)

    background_mask = (
        np.isfinite(crop)
        & (radius_seed >= 24.0)
        & (radius_seed <= 30.0)
    )
    if background_mask.sum() < 50:
        raise ValueError("Too few background-annulus pixels in this crop.")

    _, background_adu, background_rms_adu = sigma_clipped_stats(
        crop[background_mask],
        sigma=3.0,
        maxiters=5,
    )

    signal = crop - background_adu
    moment_mask = np.isfinite(signal) & (radius_seed <= 25.0)
    weights = np.where(moment_mask, np.clip(signal, 0.0, None), 0.0)
    moment_flux = weights.sum()

    if not np.isfinite(moment_flux) or moment_flux <= 0:
        raise ValueError("No positive source signal found in the selected crop.")

    x_centroid = float((xx * weights).sum() / moment_flux)
    y_centroid = float((yy * weights).sum() / moment_flux)
    radius = np.hypot(xx - x_centroid, yy - y_centroid)
    fit_mask = np.isfinite(signal) & (radius <= 25.0)

    positive_flux = float(np.where(fit_mask, np.clip(signal, 0.0, None), 0.0).sum())

    return {
        "background_adu": float(background_adu),
        "background_rms_adu": float(background_rms_adu),
        "signal": signal.astype(np.float32),
        "fit_mask": fit_mask,
        "positive_flux_adu": positive_flux,
        "x_centroid_crop": x_centroid,
        "y_centroid_crop": y_centroid,
    }


def crop_header(selection: CropSelection, parent_header: fits.Header) -> fits.Header:
    header = fits.Header()
    keys_to_copy = [
        "DATE-OBS",
        "EXPTIME",
        "FILTER",
        "FOCUS",
        "GAIN",
        "RDNOISE",
        "SEEING",
        "XBINNING",
        "YBINNING",
        "XPIXSZ",
        "YPIXSZ",
        "TELESCOP",
        "INSTRUME",
    ]

    for key in keys_to_copy:
        if key in parent_header:
            header[key] = parent_header[key]

    header["PARENT"] = (selection.parent_path.name, "Original FITS image")
    header["SRCX"] = (selection.x_center, "Clicked source x in parent image")
    header["SRCY"] = (selection.y_center, "Clicked source y in parent image")
    header["XSTART"] = (selection.x_start, "Crop start x in parent image")
    header["XSTOP"] = (selection.x_stop, "Crop stop x, exclusive")
    header["YSTART"] = (selection.y_start, "Crop start y in parent image")
    header["YSTOP"] = (selection.y_stop, "Crop stop y, exclusive")
    header["CROPSIZE"] = (selection.raw_crop.shape[0], "Square crop side in pixels")
    header["BKGADU"] = (selection.background_adu, "Sigma-clipped background ADU")
    header["BKGRMS"] = (selection.background_rms_adu, "Sigma-clipped background RMS")
    return header


class FitsCropSelector:
    def __init__(
        self,
        paths: list[Path],
        output_dir: Path,
        crop_size: int = 64,
        save_normalized: bool = False,
    ):
        if not paths:
            raise FileNotFoundError(f"No .fits files found in {DEFAULT_DATA_DIR}")
        if crop_size % 2 != 0:
            raise ValueError("Use an even crop size so the saved bounds are simple.")

        self.paths = paths
        self.output_dir = Path(output_dir)
        self.crop_size = int(crop_size)
        self.save_normalized = bool(save_normalized)
        self.index = 0
        self.image: np.ndarray | None = None
        self.header: fits.Header | None = None
        self.selection: CropSelection | None = None
        self.image_artist = None
        self.preview_artist = None
        self.selection_marker: Rectangle | None = None
        self.selection_crosshair = []

        self.fig = plt.figure(figsize=(14, 8))
        grid = self.fig.add_gridspec(
            1,
            2,
            width_ratios=[4.0, 1.35],
            left=0.05,
            right=0.98,
            bottom=0.14,
            top=0.90,
            wspace=0.12,
        )
        self.image_ax = self.fig.add_subplot(grid[0, 0])
        self.preview_ax = self.fig.add_subplot(grid[0, 1])
        self.status_text = self.fig.text(0.05, 0.055, "", fontsize=10)

        self.previous_button = Button(self.fig.add_axes([0.34, 0.025, 0.10, 0.045]), "Previous")
        self.next_button = Button(self.fig.add_axes([0.46, 0.025, 0.10, 0.045]), "Next")
        self.save_button = Button(self.fig.add_axes([0.58, 0.025, 0.10, 0.045]), "Save")

        self.previous_button.on_clicked(lambda _event: self.previous_image())
        self.next_button.on_clicked(lambda _event: self.next_image())
        self.save_button.on_clicked(lambda _event: self.save_selection())
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def show(self) -> None:
        self.load_current_image()
        plt.show()

    def set_status(self, message: str) -> None:
        self.status_text.set_text(message)
        print(message)
        self.fig.canvas.draw_idle()

    def load_current_image(self) -> None:
        self.image, self.header = load_fits_image(self.paths[self.index])
        self.selection = None
        self.draw_current_image()
        self.clear_preview()
        self.set_status(
            f"Showing {self.index + 1}/{len(self.paths)}: {self.paths[self.index].name}. "
            f"Click a source to preview a {self.crop_size}x{self.crop_size} crop."
        )

    def draw_current_image(self) -> None:
        assert self.image is not None
        vmin, vmax = display_limits(self.image)

        if self.image_artist is None:
            self.image_artist = self.image_ax.imshow(
                self.image,
                origin="lower",
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
            )
        else:
            self.image_artist.set_data(self.image)
            self.image_artist.set_clim(vmin, vmax)

        self.clear_selection_overlay()

        self.image_ax.set_title(f"{self.index + 1}/{len(self.paths)}  {self.paths[self.index].name}")
        self.image_ax.set_xlabel("x [pixel]")
        self.image_ax.set_ylabel("y [pixel]")
        self.image_ax.set_xlim(-0.5, self.image.shape[1] - 0.5)
        self.image_ax.set_ylim(-0.5, self.image.shape[0] - 0.5)
        self.fig.canvas.draw_idle()

    def clear_selection_overlay(self) -> None:
        if self.selection_marker is not None:
            self.selection_marker.remove()
            self.selection_marker = None

        for artist in self.selection_crosshair:
            artist.remove()
        self.selection_crosshair = []

    def clear_preview(self) -> None:
        self.preview_ax.clear()
        self.preview_ax.set_title("Crop preview")
        self.preview_ax.set_xlabel("x [crop pixel]")
        self.preview_ax.set_ylabel("y [crop pixel]")
        self.preview_ax.text(
            0.5,
            0.5,
            "Click the image",
            transform=self.preview_ax.transAxes,
            ha="center",
            va="center",
        )
        self.preview_artist = None
        self.fig.canvas.draw_idle()

    def preview_crop(self, x_click: float, y_click: float) -> None:
        assert self.image is not None

        x_center = int(round(x_click))
        y_center = int(round(y_click))
        x_start, x_stop, y_start, y_stop = crop_bounds(
            self.image.shape,
            x_center,
            y_center,
            self.crop_size,
        )
        raw_crop = self.image[y_start:y_stop, x_start:x_stop]
        processed = preprocess_crop(raw_crop)

        self.selection = CropSelection(
            parent_path=self.paths[self.index],
            frame_index=self.index,
            x_center=x_center,
            y_center=y_center,
            x_start=x_start,
            x_stop=x_stop,
            y_start=y_start,
            y_stop=y_stop,
            raw_crop=raw_crop.astype(np.float32),
            background_adu=processed["background_adu"],
            background_rms_adu=processed["background_rms_adu"],
            signal=processed["signal"],
            fit_mask=processed["fit_mask"],
            positive_flux_adu=processed["positive_flux_adu"],
            x_centroid_crop=processed["x_centroid_crop"],
            y_centroid_crop=processed["y_centroid_crop"],
        )

        if self.selection_marker is not None:
            self.selection_marker.remove()
        for artist in self.selection_crosshair:
            artist.remove()
        self.selection_marker = Rectangle(
            (x_start - 0.5, y_start - 0.5),
            self.crop_size,
            self.crop_size,
            edgecolor="red",
            facecolor="none",
            linewidth=1.2,
        )
        self.image_ax.add_patch(self.selection_marker)
        crosshair_radius = max(5.0, self.crop_size * 0.10)
        self.selection_crosshair = [
            self.image_ax.plot(
                [x_center - crosshair_radius, x_center + crosshair_radius],
                [y_center, y_center],
                color="yellow",
                linewidth=1.2,
            )[0],
            self.image_ax.plot(
                [x_center, x_center],
                [y_center - crosshair_radius, y_center + crosshair_radius],
                color="yellow",
                linewidth=1.2,
            )[0],
        ]

        self.preview_ax.clear()
        vmin, vmax = display_limits(self.selection.raw_crop)
        self.preview_ax.imshow(
            self.selection.raw_crop,
            origin="lower",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        crop_center = self.crop_size // 2
        self.preview_ax.axvline(
            crop_center,
            color="yellow",
            linestyle="--",
            linewidth=1.0,
        )
        self.preview_ax.axhline(
            crop_center,
            color="yellow",
            linestyle="--",
            linewidth=1.0,
        )
        self.preview_ax.plot(
            self.selection.x_centroid_crop,
            self.selection.y_centroid_crop,
            "c+",
            markersize=12,
            markeredgewidth=1.3,
        )
        self.preview_ax.set_title(f"Preview x={x_center}, y={y_center}")
        self.preview_ax.set_xlabel("x [crop pixel]")
        self.preview_ax.set_ylabel("y [crop pixel]")

        self.set_status(
            f"Preview ready: parent x={x_center}, y={y_center}, "
            f"background={self.selection.background_adu:.2f} ADU, "
            "press Save or key 's'."
        )

    def move_selection(self, dx: int, dy: int) -> None:
        if self.selection is None:
            self.set_status("Click a source first, then use arrow keys to move the 64x64 finder.")
            return

        new_x = self.selection.x_center + dx
        new_y = self.selection.y_center + dy

        try:
            self.preview_crop(new_x, new_y)
        except ValueError as error:
            self.set_status(str(error))

    def save_selection(self) -> None:
        if self.selection is None:
            self.set_status("Nothing to save yet. Click a source first.")
            return
        assert self.header is not None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        base_name = (
            f"{self.selection.parent_path.stem}_"
            f"x{self.selection.x_center}_y{self.selection.y_center}_"
            f"{self.crop_size}"
        )
        crop_dir = self.output_dir / base_name
        crop_dir.mkdir(parents=True, exist_ok=True)
        header = crop_header(self.selection, self.header)

        raw_path = crop_dir / "raw.fits"
        signal_path = crop_dir / "background_subtracted.fits"
        mask_path = crop_dir / "fit_mask.fits"
        metadata_path = crop_dir / "metadata.json"

        raw_header = header.copy()
        raw_header["PROCSTAT"] = ("RAW_CROP", "Raw 64x64 selected crop")
        fits.writeto(raw_path, self.selection.raw_crop, header=raw_header, overwrite=True)

        signal_header = header.copy()
        signal_header["PROCSTAT"] = ("BKG_SUB", "Background-subtracted crop")
        fits.writeto(signal_path, self.selection.signal, header=signal_header, overwrite=True)

        mask_header = header.copy()
        mask_header["PROCSTAT"] = ("FIT_MASK", "Binary fit mask")
        fits.writeto(
            mask_path,
            self.selection.fit_mask.astype(np.uint8),
            header=mask_header,
            overwrite=True,
        )

        outputs = {
            "raw_crop": str(raw_path),
            "background_subtracted": str(signal_path),
            "fit_mask": str(mask_path),
        }

        if self.save_normalized:
            normalized_path = crop_dir / "normalized.fits"
            fit_signal = np.where(
                self.selection.fit_mask,
                np.clip(self.selection.signal, 0.0, None),
                0.0,
            )
            fit_flux = fit_signal.sum()
            if not np.isfinite(fit_flux) or fit_flux <= 0:
                self.set_status("Could not save normalized crop because positive fit flux is zero.")
                return
            normalized = fit_signal / fit_flux

            normalized_header = header.copy()
            normalized_header["PROCSTAT"] = ("NORM_OBS", "Positive masked flux normalized to 1")
            fits.writeto(
                normalized_path,
                normalized.astype(np.float32),
                header=normalized_header,
                overwrite=True,
            )
            outputs["normalized"] = str(normalized_path)

        metadata = {
            "parent_fits": str(self.selection.parent_path),
            "frame_index": self.selection.frame_index,
            "crop_size_pixels": self.crop_size,
            "clicked_center_parent_0_based": {
                "x": self.selection.x_center,
                "y": self.selection.y_center,
            },
            "crop_bounds_parent_0_based": {
                "x_start": self.selection.x_start,
                "x_stop_exclusive": self.selection.x_stop,
                "y_start": self.selection.y_start,
                "y_stop_exclusive": self.selection.y_stop,
            },
            "estimated_center_crop_0_based": {
                "x": self.selection.x_centroid_crop,
                "y": self.selection.y_centroid_crop,
            },
            "preprocessing": {
                "background_annulus_px": [24.0, 30.0],
                "fit_radius_px": 25.0,
                "background_adu": self.selection.background_adu,
                "background_rms_adu": self.selection.background_rms_adu,
                "positive_flux_after_background_adu": self.selection.positive_flux_adu,
                "fit_pixels": int(self.selection.fit_mask.sum()),
                "normalization_saved": self.save_normalized,
            },
            "outputs": outputs,
            "notes": [
                "Coordinate convention is NumPy zero-based image[y, x].",
                "Original FITS frame is not modified.",
                "The clicked center is a seed; the centroid is estimated from positive background-subtracted signal.",
                "Normalized FITS output is skipped by default. Use --save-normalized to create it.",
            ],
        }

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        self.set_status(f"Saved crop folder: {crop_dir}")

    def previous_image(self) -> None:
        self.index = (self.index - 1) % len(self.paths)
        self.load_current_image()

    def next_image(self) -> None:
        self.index = (self.index + 1) % len(self.paths)
        self.load_current_image()

    def on_click(self, event) -> None:
        if event.inaxes is not self.image_ax or event.xdata is None or event.ydata is None:
            return

        try:
            self.preview_crop(event.xdata, event.ydata)
        except ValueError as error:
            self.selection = None
            self.clear_preview()
            self.set_status(str(error))

    def on_key_press(self, event) -> None:
        move_keys = {
            "left": (-1, 0),
            "right": (1, 0),
            "down": (0, -1),
            "up": (0, 1),
            "shift+left": (-5, 0),
            "shift+right": (5, 0),
            "shift+down": (0, -5),
            "shift+up": (0, 5),
        }

        if event.key in move_keys:
            dx, dy = move_keys[event.key]
            self.move_selection(dx, dy)
        elif event.key in {"n", " "}:
            self.next_image()
        elif event.key in {"p", "backspace"}:
            self.previous_image()
        elif event.key == "s":
            self.save_selection()
        elif event.key in {"escape", "q"}:
            plt.close(self.fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse FITS files, click a source, preview a 64x64 crop, then save it."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument(
        "--save-normalized",
        action="store_true",
        help="Also save a positive-flux normalized FITS crop. Off by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = list_fits_files(args.data_dir)
    selector = FitsCropSelector(
        paths,
        args.output_dir,
        crop_size=args.crop_size,
        save_normalized=args.save_normalized,
    )
    selector.show()


if __name__ == "__main__":
    main()
