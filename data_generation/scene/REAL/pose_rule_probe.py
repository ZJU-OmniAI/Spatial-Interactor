from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


REPO_ROOT = Path("/path/to/workspace")
DA3_CACHE_ROOT = REPO_ROOT / "Depth-Anything-3" / "workspace" / "vsibench_pose_cache"
ARKIT_GT_ROOT = REPO_ROOT / "datasets" / "arkitscenes_gt" / "raw"
OUTPUT_ROOT = REPO_ROOT / "SCENEOUTPUT" / "REAL" / "pose_probe"


@dataclass
class CandidateResult:
    name: str
    num_frames: int
    center_mae: float
    step_translation_mae: float
    yaw_mae_deg: float
    pitch_mae_deg: float
    translation_score: float
    rotation_score: float
    score: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_da3_bundle(dataset: str, scene: str) -> Tuple[dict, np.ndarray]:
    scene_dir = DA3_CACHE_ROOT / dataset / scene
    bundle_path = scene_dir / "pose_bundle.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"DA3 pose_bundle not found: {bundle_path}")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    extr_path = Path(str(bundle["extrinsics_path"]).replace("/data/path/to/workspace", "/path/to/workspace"))
    if not extr_path.exists():
        alt_paths = [
            scene_dir / "extrinsics.npy",
            scene_dir / f"extrinsics_{bundle.get('num_frames', 32)}.npy",
            scene_dir / "extrinsics_32.npy",
            scene_dir / "extrinsics_64.npy",
        ]
        extr_path = next((p for p in alt_paths if p.exists()), extr_path)
    if not extr_path.exists():
        raise FileNotFoundError(f"DA3 extrinsics not found: {extr_path}")
    extr = np.load(extr_path)
    if extr.ndim == 3 and extr.shape[1:] == (3, 4):
        extr = to_homogeneous(extr)
    elif extr.ndim == 3 and extr.shape[1:] == (4, 4):
        pass
    else:
        raise ValueError(f"Unexpected DA3 extrinsics shape: {extr.shape}")
    return bundle, extr.astype(np.float64)


def to_homogeneous(ext: np.ndarray) -> np.ndarray:
    if ext.shape[-2:] == (4, 4):
        return ext
    if ext.shape[-2:] != (3, 4):
        raise ValueError(f"Unexpected extrinsic shape: {ext.shape}")
    out = np.repeat(np.eye(4, dtype=np.float64)[None], ext.shape[0], axis=0)
    out[:, :3, :] = ext
    return out


def camera_centers_from_w2c(ext_w2c: np.ndarray) -> np.ndarray:
    r = ext_w2c[:, :3, :3]
    t = ext_w2c[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(r, (0, 2, 1)), t)


def yaw_pitch_from_w2c(ext_w2c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = ext_w2c[:, :3, :3]
    forward_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward_world = np.einsum("nij,j->ni", np.transpose(r, (0, 2, 1)), forward_cam)
    yaw = np.degrees(np.arctan2(forward_world[:, 0], forward_world[:, 2]))
    horiz = np.linalg.norm(forward_world[:, [0, 2]], axis=1)
    pitch = np.degrees(np.arctan2(-forward_world[:, 1], np.maximum(horiz, 1e-8)))
    return yaw, pitch


def unwrap_deg(values: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(values)))


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    rot, _ = cv2.Rodrigues(axis_angle.astype(np.float64).reshape(3, 1))
    return rot.astype(np.float64)


def parse_arkitscenes_traj(path: Path) -> np.ndarray:
    mats: List[np.ndarray] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 7:
                raise ValueError(f"Unexpected traj row with {len(vals)} values: {line}")
            _ts, rx, ry, rz, tx, ty, tz = vals
            rot = axis_angle_to_matrix(np.array([rx, ry, rz], dtype=np.float64))
            m = np.eye(4, dtype=np.float64)
            m[:3, :3] = rot
            m[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
            mats.append(m)
    if not mats:
        raise ValueError(f"No poses parsed from {path}")
    return np.stack(mats, axis=0)


def parse_scannet_pose_dir(path: Path) -> np.ndarray:
    pose_files = sorted(path.glob("*.txt"))
    if not pose_files:
        raise FileNotFoundError(f"No txt pose files found under {path}")
    mats = []
    for pose_file in pose_files:
        mat = np.loadtxt(pose_file, dtype=np.float64)
        if mat.shape == (3, 4):
            mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
        if mat.shape != (4, 4):
            raise ValueError(f"Unexpected ScanNet pose shape in {pose_file}: {mat.shape}")
        mats.append(mat)
    return np.stack(mats, axis=0)


def parse_scannetpp_pose_intrinsic_imu(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("aligned_poses", "poses"):
        if key in payload:
            poses = payload[key]
            mats = []
            for pose in poses:
                mat = np.array(pose, dtype=np.float64)
                if mat.shape == (3, 4):
                    mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
                if mat.shape != (4, 4):
                    raise ValueError(f"Unexpected pose_intrinsic_imu shape: {mat.shape}")
                mats.append(mat)
            return np.stack(mats, axis=0)
    raise KeyError(f"aligned_poses / poses not found in {path}")


def parse_scannetpp_transforms_json(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise KeyError(f"frames not found in {path}")
    mats = []
    for frame in frames:
        mat = np.array(frame["transform_matrix"], dtype=np.float64)
        if mat.shape != (4, 4):
            raise ValueError(f"Unexpected transform_matrix shape: {mat.shape}")
        mats.append(mat)
    return np.stack(mats, axis=0)


def parse_scannetpp_colmap_images(path: Path) -> np.ndarray:
    mats = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        qvec = np.array([float(x) for x in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
        rot = qvec_to_rotmat(qvec)
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = rot
        mat[:3, 3] = tvec
        mats.append(mat)
        i += 1  # skip points2D line
    if not mats:
        raise ValueError(f"No COLMAP images parsed from {path}")
    return np.stack(mats, axis=0)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def sample_sequence(seq: np.ndarray, count: int) -> np.ndarray:
    if len(seq) == count:
        return seq
    if len(seq) < count:
        raise ValueError(f"Official pose sequence too short: {len(seq)} < {count}")
    idx = np.linspace(0, len(seq) - 1, num=count, dtype=np.int32)
    return seq[idx]


def gl_to_cv_c2w(seq: np.ndarray) -> np.ndarray:
    flip = np.eye(4, dtype=np.float64)
    flip[1, 1] = -1.0
    flip[2, 2] = -1.0
    return np.einsum("nij,jk->nik", seq, flip)


def invert_all(seq: np.ndarray) -> np.ndarray:
    return np.linalg.inv(seq)


def rotation_angle_deg(r: np.ndarray) -> float:
    trace = np.trace(r)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def umeyama_align(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = (dst_c.T @ src_c) / max(len(src), 1)
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1.0
    r = u @ s @ vt
    var = np.mean(np.sum(src_c * src_c, axis=1))
    scale = 1.0 if var < 1e-12 else float(np.sum(d * np.diag(s)) / var)
    t = dst_mean - scale * (r @ src_mean)
    return scale, r, t


def align_centers(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    scale, rot, trans = umeyama_align(src, dst)
    return (scale * (rot @ src.T)).T + trans


def evaluate_candidate(name: str, cand_w2c: np.ndarray, da3_w2c: np.ndarray) -> CandidateResult:
    cand_centers = camera_centers_from_w2c(cand_w2c)
    da3_centers = camera_centers_from_w2c(da3_w2c)
    cand_centers_aligned = align_centers(cand_centers, da3_centers)

    center_mae = float(np.mean(np.linalg.norm(cand_centers_aligned - da3_centers, axis=1)))
    step_translation_mae = float(
        np.mean(
            np.abs(
                np.linalg.norm(np.diff(cand_centers_aligned, axis=0), axis=1)
                - np.linalg.norm(np.diff(da3_centers, axis=0), axis=1)
            )
        )
    )

    cand_yaw, cand_pitch = yaw_pitch_from_w2c(cand_w2c)
    da3_yaw, da3_pitch = yaw_pitch_from_w2c(da3_w2c)
    cand_yaw = unwrap_deg(cand_yaw)
    da3_yaw = unwrap_deg(da3_yaw)
    cand_pitch = unwrap_deg(cand_pitch)
    da3_pitch = unwrap_deg(da3_pitch)

    cand_yaw_delta = np.diff(cand_yaw)
    da3_yaw_delta = np.diff(da3_yaw)
    cand_pitch_delta = np.diff(cand_pitch)
    da3_pitch_delta = np.diff(da3_pitch)

    yaw_mae_deg = float(np.mean(np.abs(cand_yaw_delta - da3_yaw_delta)))
    pitch_mae_deg = float(np.mean(np.abs(cand_pitch_delta - da3_pitch_delta)))

    translation_score = center_mae + 0.5 * step_translation_mae
    rotation_score = 0.01 * yaw_mae_deg + 0.01 * pitch_mae_deg
    score = translation_score + rotation_score
    return CandidateResult(
        name=name,
        num_frames=len(cand_w2c),
        center_mae=center_mae,
        step_translation_mae=step_translation_mae,
        yaw_mae_deg=yaw_mae_deg,
        pitch_mae_deg=pitch_mae_deg,
        translation_score=translation_score,
        rotation_score=rotation_score,
        score=score,
    )


def default_official_path(dataset: str, scene: str) -> Optional[Path]:
    if dataset == "scannet":
        candidate = REPO_ROOT / "DATA" / "REAL_OFFICIAL" / "scannet_pose_intrinsic_cache" / scene / "pose"
        if candidate.exists():
            return candidate
    if dataset == "arkitscenes":
        roots = [
            REPO_ROOT / "DATA" / "REAL_OFFICIAL" / "arkitscenes" / "raw",
            ARKIT_GT_ROOT,
        ]
        for root in roots:
            for split in ("Training", "Validation"):
                candidate = root / split / scene / "lowres_wide.traj"
                if candidate.exists():
                    return candidate
    return None


def build_candidates(dataset: str, official_pose_raw: np.ndarray, official_format: str) -> Dict[str, np.ndarray]:
    if official_format == "arkitscenes_traj":
        return {
            "arkit_traj_direct_as_w2c": official_pose_raw,
            "arkit_traj_inverse_c2w_to_w2c": invert_all(official_pose_raw),
        }
    if official_format == "scannet_pose_dir":
        return {
            "scannet_direct_as_w2c": official_pose_raw,
            "scannet_inverse_c2w_to_w2c": invert_all(official_pose_raw),
        }
    if official_format == "scannetpp_pose_intrinsic_imu":
        return {
            "scannetpp_pose_intrinsic_direct_as_w2c": official_pose_raw,
            "scannetpp_pose_intrinsic_inverse_c2w_to_w2c": invert_all(official_pose_raw),
        }
    if official_format == "scannetpp_transforms_json":
        seq_cv_c2w = gl_to_cv_c2w(official_pose_raw)
        return {
            "scannetpp_transforms_direct_as_w2c": official_pose_raw,
            "scannetpp_transforms_gl_c2w_to_cv_w2c": invert_all(seq_cv_c2w),
            "scannetpp_transforms_plain_inverse": invert_all(official_pose_raw),
        }
    if official_format == "scannetpp_colmap_images":
        return {
            "scannetpp_colmap_direct_w2c": official_pose_raw,
            "scannetpp_colmap_inverse_to_w2c": invert_all(official_pose_raw),
        }
    raise ValueError(f"Unsupported official_format: {official_format}")


def parse_official_pose(path: Path, official_format: str) -> np.ndarray:
    if official_format == "arkitscenes_traj":
        return parse_arkitscenes_traj(path)
    if official_format == "scannet_pose_dir":
        return parse_scannet_pose_dir(path)
    if official_format == "scannetpp_pose_intrinsic_imu":
        return parse_scannetpp_pose_intrinsic_imu(path)
    if official_format == "scannetpp_transforms_json":
        return parse_scannetpp_transforms_json(path)
    if official_format == "scannetpp_colmap_images":
        return parse_scannetpp_colmap_images(path)
    raise ValueError(f"Unsupported official_format: {official_format}")


def infer_default_format(dataset: str, official_path: Path) -> str:
    if dataset == "arkitscenes":
        return "arkitscenes_traj"
    if dataset == "scannet":
        return "scannet_pose_dir"
    if dataset == "scannetpp":
        name = official_path.name
        if name == "pose_intrinsic_imu.json":
            return "scannetpp_pose_intrinsic_imu"
        if name == "transforms.json":
            return "scannetpp_transforms_json"
        if name == "images.txt":
            return "scannetpp_colmap_images"
    raise ValueError(f"Cannot infer official format for dataset={dataset}, path={official_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe official pose conversion rules against DA3 reference.")
    parser.add_argument("--dataset", required=True, choices=["arkitscenes", "scannet", "scannetpp"])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--official-path", type=Path, default=None)
    parser.add_argument("--official-format", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    official_path = args.official_path or default_official_path(args.dataset, args.scene)
    if official_path is None or not official_path.exists():
        raise FileNotFoundError(
            f"Official pose path not found for dataset={args.dataset}, scene={args.scene}. "
            f"Please pass --official-path explicitly."
        )
    official_format = args.official_format or infer_default_format(args.dataset, official_path)

    bundle, da3_w2c = load_da3_bundle(args.dataset, args.scene)
    official_raw = parse_official_pose(official_path, official_format)
    official_raw = sample_sequence(official_raw, len(da3_w2c))

    candidates = build_candidates(args.dataset, official_raw, official_format)
    results = [evaluate_candidate(name, cand, da3_w2c) for name, cand in candidates.items()]
    results.sort(key=lambda item: item.score)
    best = results[0]

    out_dir = args.output_root / args.dataset / args.scene
    ensure_dir(out_dir)
    write_json(
        out_dir / "probe_result.json",
        {
            "dataset": args.dataset,
            "scene": args.scene,
            "official_path": str(official_path),
            "official_format": official_format,
            "da3_bundle_path": str(DA3_CACHE_ROOT / args.dataset / args.scene / "pose_bundle.json"),
            "num_da3_frames": int(len(da3_w2c)),
            "num_official_frames_before_sampling": int(len(parse_official_pose(official_path, official_format))),
            "recommended_rule": best.name,
            "candidate_scores": [
                {
                    "name": item.name,
                    "num_frames": item.num_frames,
                    "center_mae": item.center_mae,
                    "step_translation_mae": item.step_translation_mae,
                    "yaw_mae_deg": item.yaw_mae_deg,
                    "pitch_mae_deg": item.pitch_mae_deg,
                    "translation_score": item.translation_score,
                    "rotation_score": item.rotation_score,
                    "score": item.score,
                }
                for item in results
            ],
            "notes": [
                "score 越低越好。",
                "当前版本以平移轨迹一致性为主，旋转项只作为辅证。",
                "当前 probe 先按时间顺序均匀采样官方 pose 到 DA3 帧数。",
                "如果后续拿到精确的 frame timestamp / intrinsics，可把时序对齐进一步收紧。",
            ],
        },
    )
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "scene": args.scene,
                "recommended_rule": best.name,
                "score": best.score,
                "output": str(out_dir / "probe_result.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
