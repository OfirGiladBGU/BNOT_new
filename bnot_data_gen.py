#!/usr/bin/env python3
"""BNOT dataset generator.

Reads images from <data_path>/source/, generates target/ outputs,
and writes prompt.json JSONL entries.

Requires a source/ folder with input images. Does not use original/ fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import shutil

REPO_ROOT = Path(__file__).resolve().parent
PYTHON_SRC = REPO_ROOT / "python" / "src"
if str(PYTHON_SRC) not in sys.path:
	sys.path.insert(0, str(PYTHON_SRC))

from ibnot_cli_wrapper import (  # noqa: E402
	InferenceRequest,
	NativeConfig,
	OutputConfig,
	RenderConfig,
	find_default_executable,
	run_inference,
)
import tempfile

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_grayscale_image(image_path: Path, image_size: tuple[int, int] | None, invert: bool) -> np.ndarray:
	image = Image.open(image_path).convert("L")
	if image_size is not None and image.size != image_size:
		image = image.resize(image_size, resample=Image.Resampling.LANCZOS)
	if invert:
		image = ImageOps.invert(image)
	return np.asarray(image, dtype=np.float64) / 255.0


def save_source_image(image_array: np.ndarray, out_path: Path) -> None:
	# Deprecated: prefer save_source_copy to preserve original file colors.
	out_path.parent.mkdir(parents=True, exist_ok=True)
	source_u8 = np.clip(np.rint(image_array * 255.0), 0, 255).astype(np.uint8)
	Image.fromarray(source_u8, mode="L").save(out_path)


def save_source_copy(src_path: Path, out_path: Path) -> None:
	"""Copy original source file to `source/` without modification."""
	out_path.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(src_path, out_path)


def load_points(dat_path: Path) -> np.ndarray:
	points = np.loadtxt(dat_path, dtype=np.float64)
	if points.size == 0:
		return np.empty((0, 2), dtype=np.float64)
	if points.ndim == 1:
		points = points[np.newaxis, :]
	if points.shape[1] < 2:
		raise ValueError(f"expected at least two columns in point file: {dat_path}")
	return points[:, :2]


def save_target_png_from_points(points: np.ndarray, out_path: Path, width: int, height: int) -> None:
	canvas = np.full((height, width), 255, dtype=np.uint8)
	if points.size != 0:
		dx = 0.5
		dy = 0.5 * float(height) / float(width)
		# Map solver coordinates (centered domain) to pixel indices.
		# Normalized u,v in [0,1] -> pixel index should map to [0, width-1] / [0, height-1].
		for x, y in points:
			x_norm = 0.5 * (float(x) + dx) / dx
			y_norm = 0.5 * (dy - float(y)) / dy
			i = int(np.round(x_norm * (width - 1)))
			j = int(np.round(y_norm * (height - 1)))
			# clamp just in case of numerical edge cases
			if i < 0:
				i = 0
			elif i >= width:
				i = width - 1
			if j < 0:
				j = 0
			elif j >= height:
				j = height - 1
			canvas[j, i] = 0

	out_path.parent.mkdir(parents=True, exist_ok=True)
	Image.fromarray(canvas, mode="L").save(out_path)

def try_recover_from_crash(
	target_out_dir: Path,
	stem: str,
	width: int,
	height: int,
) -> tuple[bool, str]:
	"""
	Attempt to recover from a BNOT crash by using partial .dat file if it exists.
	If no partial data, skip the image.
	Returns (success: bool, recovery_type: str).
	"""
	dat_path = target_out_dir / f"{stem}.dat"
	target_png = target_out_dir / f"{stem}.png"

	# Try to use partial .dat file
	if dat_path.exists():
		try:
			points = load_points(dat_path)
			if points.shape[0] > 0:
				# Rasterize whatever points we have
				save_target_png_from_points(points, target_png, width=width, height=height)

				# Cleanup .dat and stats files
				try:
					dat_path.unlink()
				except Exception:
					pass
				try:
					stats_path = target_out_dir / f"{stem}.txt"
					if stats_path.exists():
						stats_path.unlink()
				except Exception:
					pass

				return True, "partial_data"
		except Exception:
			pass

	# No usable partial data; skip this image.
	return False, "failed"

def _write_pgm(path: Path, array: np.ndarray) -> None:
	# Write ASCII P2 PGM (matches wrapper behavior)
	arr = np.clip(np.rint(array * 255.0), 0, 255).astype(int)
	h, w = arr.shape
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		handle.write("P2\n")
		handle.write(f"{w} {h}\n")
		handle.write("255\n")
		for row in arr:
			handle.write(" ".join(str(int(v)) for v in row))
			handle.write("\n")


def build_request(
	image_array: np.ndarray,
	output_dir: Path,
	output_stem: str,
	*,
	image_path: Path | None = None,
	executable: Path,
	num_sites: int,
	seed: int,
	max_iters: int,
	max_newton_iters: int,
	keep_pgm: bool,
	keep_stats: bool,
	native_timer: bool,
	) -> InferenceRequest:
	# If an explicit image_path is provided, prefer that and leave image_array None
	if image_path is not None:
		img_arg = dict(image_path=image_path)
	else:
		img_arg = dict(image_array=image_array)

	return InferenceRequest(
		**img_arg,
		native=NativeConfig(
			num_sites=num_sites,
			seed=seed,
			max_iters=max_iters,
			max_newton_iters=max_newton_iters,
			invert=False,
			native_timer=native_timer,
		),
		render=RenderConfig(
			enabled=False,
			render_width=None,
			render_height=None,
			point_radius=0.002,
			png_enabled=False,
			dpi=300,
		),
		output=OutputConfig(
			output_dir=output_dir,
			output_stem=output_stem,
			keep_pgm=keep_pgm,
			keep_eps=False,
			keep_stats_txt=keep_stats,
		),
		executable=executable,
	)


def process_one(
	src_path: Path,
	source_out: Path,
	target_out_dir: Path,
	*,
	executable: Path,
	image_size: tuple[int, int] | None,
	invert: bool,
	num_sites: int,
	seed: int,
	max_iters: int,
	max_newton_iters: int,
	keep_pgm: bool,
	keep_stats: bool,
	track_time: bool,
	timestamps_dir: Path | None,
	rel_path: Path,
) -> str:
	target_png = target_out_dir / f"{src_path.stem}.png"
	if source_out.exists() and target_png.exists():
		return "skipped"

	# Load image as array but never overwrite the original file. If any
	# modification is required (resize/invert), create a temporary PGM file
	# and pass its path to the native CLI. The temp file is removed after use.
	image_array = load_grayscale_image(src_path, image_size=image_size, invert=invert)
	# Preserve the original file colors: copy into `source/` only if the
	# source path is different from the destination (i.e., when reading from
	# `original/`). If we are already reading from `source/`, leave it alone.
	try:
		if src_path.resolve() != source_out.resolve():
			save_source_copy(src_path, source_out)
	except Exception:
		# Fall back to copying if resolution check fails for any reason.
		save_source_copy(src_path, source_out)

	# Use the saved `source` image dimensions as authoritative for output size
	with Image.open(source_out) as _src_im:
		src_w, src_h = _src_im.size
	width = int(src_w)
	height = int(src_h)

	# Always use a temporary PGM so the original is never modified.
	with tempfile.TemporaryDirectory(prefix="bnot_tmp_") as tmpdir:
		tmp_pgm = Path(tmpdir) / f"{src_path.stem}_tmp_input.pgm"
		_write_pgm(tmp_pgm, image_array)

		request = build_request(
			image_array,
			target_out_dir,
			src_path.stem,
			image_path=tmp_pgm,
			executable=executable,
			num_sites=num_sites,
			seed=seed,
			max_iters=max_iters,
			max_newton_iters=max_newton_iters,
			keep_pgm=keep_pgm,
			keep_stats=keep_stats,
			native_timer=track_time,
		)

		result = run_inference(request)
	if result.resolved_paths.dat_path is None:
		raise RuntimeError("BNOT did not produce a .dat file for rasterization")

	points = load_points(result.resolved_paths.dat_path)
	# Use source image dimensions (read from saved file) for rasterization
	save_target_png_from_points(points, target_png, width=width, height=height)

	# Clean up native output files (.dat and stats .txt) — they are intermediate
	# and not needed after rasterization. Keep PGMs if requested.
	try:
		dat_path = result.resolved_paths.dat_path
		if dat_path is not None and dat_path.exists():
			dat_path.unlink()
	except Exception:
		pass
	try:
		stats_path = result.resolved_paths.stats_path
		if stats_path is not None and stats_path.exists():
			stats_path.unlink()
	except Exception:
		pass

	if track_time and timestamps_dir is not None:
		timestamps_dir.mkdir(parents=True, exist_ok=True)
		timestamp_path = timestamps_dir / rel_path.with_suffix(".txt")
		timestamp_path.parent.mkdir(parents=True, exist_ok=True)
		with timestamp_path.open("w", encoding="utf-8") as handle:
			handle.write(f"pgm_write_seconds: {result.timings.pgm_write_seconds:.6f}\n")
			handle.write(f"native_wall_seconds: {result.timings.native_wall_seconds:.6f}\n")
			handle.write(f"render_wall_seconds: {result.timings.render_wall_seconds:.6f}\n")
			handle.write(f"total_wall_seconds: {result.timings.total_wall_seconds:.6f}\n")

	return "processed"


def main() -> int:
	data_path = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/results/monkey"

	# data_path = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/results/quadratic_V2"

	# data_path = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/results/plant2"

	# ICONS
	# data_path = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/outputs/quantitative_advance_metrics"
	
	n = -1
	image_size = None
	num_sites = 1024
	seed = 7
	max_iters = 25
	max_newton_iters = 50
	invert = True
	keep_pgm = False
	keep_stats = True
	track_time = True
	overwrite = False
	executable = None

	parser = argparse.ArgumentParser(
		description="Generate BNOT source/target pairs from images under source/",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("--data_path", type=Path, default=data_path,
						help="Dataset root containing a source/ folder")
	parser.add_argument("--n", type=int, default=n, help="Number of images to process; -1 means all")
	parser.add_argument("--image_size", type=int, nargs=2, default=image_size,
						metavar=("W", "H"), help="Resize images before inference; default keeps original size")
	parser.add_argument("--num_sites", type=int, default=num_sites)
	parser.add_argument("--seed", type=int, default=seed)
	parser.add_argument("--max_iters", type=int, default=max_iters)
	parser.add_argument("--max_newton_iters", type=int, default=max_newton_iters)
	parser.add_argument("--invert", action=argparse.BooleanOptionalAction, default=invert,
						help="Invert the grayscale source before inference")
	parser.add_argument("--keep_pgm", action=argparse.BooleanOptionalAction, default=keep_pgm)
	parser.add_argument("--keep_stats", action=argparse.BooleanOptionalAction, default=keep_stats)
	parser.add_argument("--track_time", action=argparse.BooleanOptionalAction, default=track_time,
						help="Write per-image timing text files into timestamps/")
	parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=overwrite,
						help="Overwrite existing source/target outputs")
	parser.add_argument("--executable", type=Path, default=executable,
						help="Path to ibnot_new_cli; defaults to repo-local discovery")

	args = parser.parse_args()

	data_path = args.data_path.expanduser().resolve()
	source_dir = data_path / "source"
	target_dir = data_path / "target"
	json_path = data_path / "prompt.json"
	timestamps_dir = data_path / "timestamps" if args.track_time else None

	# Require source/ to exist.
	if not source_dir.is_dir():
		print(f"Error: 'source/' folder not found under: {data_path}", file=sys.stderr)
		return 1

	source_dir.mkdir(parents=True, exist_ok=True)
	target_dir.mkdir(parents=True, exist_ok=True)
	if timestamps_dir is not None:
		timestamps_dir.mkdir(parents=True, exist_ok=True)

	# Auto-set IBNOT_CLI_BIN to the repo-local built binary if present so the
	# user doesn't need to export it each time. If the user passes
	# --executable or has IBNOT_CLI_BIN set, that takes precedence.
	default_cli = REPO_ROOT / "ibnot_cli" / "build" / "ibnot_new_cli"
	if "IBNOT_CLI_BIN" not in os.environ and default_cli.exists():
		os.environ["IBNOT_CLI_BIN"] = str(default_cli.resolve())

	executable = find_default_executable(args.executable)

	# Read images from source/ directory only (no fallback to original/).
	image_files = sorted(
		[p.relative_to(source_dir) for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXT]
	)
	if not image_files:
		print(f"No images found under: {source_dir}", file=sys.stderr)
		return 1

	n = len(image_files) if args.n == -1 else min(args.n, len(image_files))

	print(f"Data path: {data_path}")
	print(f"Images found: {len(image_files)} | processing: {n}")
	print(f"Executable: {executable}")
	print(f"Num sites: {args.num_sites} | seed: {args.seed} | max iters: {args.max_iters}")

	entries: list[dict[str, str]] = []
	processed = 0
	skipped = 0

	for idx, rel_path in enumerate(tqdm(image_files[:n], desc="BNOT images"), 1):
		src_path = source_dir / rel_path
		source_out = source_dir / rel_path.with_suffix(".png")
		target_out_dir = target_dir / rel_path.parent
		target_out_dir.mkdir(parents=True, exist_ok=True)

		entries.append(
			{
				"source": f"source/{rel_path.with_suffix('.png').as_posix()}",
				"target": f"target/{rel_path.with_suffix('.png').as_posix()}",
				"prompt": "BNOT",
			}
		)

		target_png = target_out_dir / f"{rel_path.stem}.png"
		if not args.overwrite and source_out.exists() and target_png.exists():
			print(f"  [{idx}/{n}] Skipping {rel_path.name}: already processed")
			skipped += 1
			continue

		print(f"  [{idx}/{n}] Processing {rel_path.name}")
		try:
			status = process_one(
				src_path,
				source_out,
				target_out_dir,
				executable=executable,
				image_size=tuple(args.image_size) if args.image_size is not None else None,
				invert=args.invert,
				num_sites=args.num_sites,
				seed=args.seed,
				max_iters=args.max_iters,
				max_newton_iters=args.max_newton_iters,
				keep_pgm=args.keep_pgm,
				keep_stats=args.keep_stats,
				track_time=args.track_time,
				timestamps_dir=timestamps_dir,
				rel_path=rel_path,
			)
			if status == "processed":
				processed += 1
			else:
				skipped += 1
		except Exception as crash_error:
			# Attempt to recover using partial .dat file only; otherwise skip.
			print(f"    ⚠ Error processing {rel_path.name}: {crash_error}")
			
			# Infer dimensions from source image
			try:
				with Image.open(source_out) as src_img:
					width, height = src_img.size
			except Exception:
				print(f"    ✗ Could not determine image dimensions, skipping")
				skipped += 1
				continue
			
			# Try to recover using partial .dat only
			recovered, recovery_type = try_recover_from_crash(
				target_out_dir,
				rel_path.stem,
				width=width,
				height=height,
			)
			
			if recovered:
				if recovery_type == "partial_data":
					print(f"    ✓ Recovered using partial data ({rel_path.stem})")
				processed += 1
			else:
				print(f"    ✗ Skipped {rel_path.name} (recovery failed)")
				skipped += 1
			continue

	json_path.parent.mkdir(parents=True, exist_ok=True)
	with json_path.open("w", encoding="utf-8") as handle:
		for entry in entries:
			handle.write(json.dumps(entry) + "\n")

	print(f"\nSummary: processed={processed}, skipped={skipped}")
	print(f"prompt.json: {json_path}")
	if timestamps_dir is not None:
		print(f"timestamps: {timestamps_dir}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
