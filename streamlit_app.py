import base64
import json
import csv
import random
import re
import shutil
import zipfile
import subprocess
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import boto3
import streamlit as st
import streamlit.components.v1 as components

try:
    from imageio_ffmpeg import get_ffmpeg_exe
except Exception:
    get_ffmpeg_exe = None


def _get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return str(os.getenv(name, default))


APP_DEFAULTS = {
    "bucket": _get_secret("S3_BUCKET", "dataset-avt-fyp"),
    "workers_prefix": _get_secret("S3_WORKERS_PREFIX", "new35hoursfixedia_worker_strict_single/state"),
    "fallback_clips_prefix": _get_secret("S3_FALLBACK_CLIPS_PREFIX", ""),
    "region": _get_secret("AWS_REGION", "eu-north-1"),
    "access_key": _get_secret("AWS_ACCESS_KEY_ID", ""),
    "secret_key": _get_secret("AWS_SECRET_ACCESS_KEY", ""),
    "session_token": _get_secret("AWS_SESSION_TOKEN", ""),
    "url_expiry": 3600,
    "auto_load_on_start": True,
    "sample_output_dir": "testing/s3_review_samples",
}

PIPELINE_STAGES = [
    "talknet",
    "export_json",
    "extract_crops",
    "speaker_id",
    "clip_generation",
    "complete",
]

DEFAULT_STAGE_RANK = {
    "queued": 0,
    "talknet": 1,
    "export_json": 2,
    "extract_crops": 3,
    "speaker_id": 4,
    "clip_generation": 5,
    "complete": 6,
}

TIMESTAMP_KEYS = (
    "generated_at_utc",
    "updated_utc",
    "updated_at_utc",
    "updated_at",
    "timestamp",
    "last_updated",
    "modified_at",
)


@dataclass
class WorkerArtifacts:
    worker: str
    state_key: str
    manifest_key: str
    metadata_key: str
    folders_state_key: str
    state_last_modified: Optional[datetime]
    state_payload: Dict[str, Any]
    manifest_payload: Dict[str, Any]
    metadata_payload: Dict[str, Any]


@dataclass
class StageDecision:
    video_id: str
    stage: str
    stage_rank: int
    source_worker: str
    source_key: str
    decided_at: Optional[datetime]


@dataclass
class ClipChoice:
    clip_path: str
    clip_key: Optional[str]
    textgrid_key: Optional[str]
    wav_key: Optional[str]
    speaker_id: str
    source_video: str
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    worker: str


def normalize_prefix(prefix: str) -> str:
    return (prefix or "").strip().strip("/")


def parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def best_payload_ts(payload: Dict[str, Any]) -> Optional[datetime]:
    for key in TIMESTAMP_KEYS:
        dt = parse_ts(payload.get(key))
        if dt:
            return dt
    return None


def build_s3_client(region: str, access_key: str, secret_key: str, session_token: str):
    kwargs: Dict[str, Any] = {}
    if region:
        kwargs["region_name"] = region
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    if session_token:
        kwargs["aws_session_token"] = session_token
    return boto3.client("s3", **kwargs)


def worker_name_from_state_key(key: str) -> str:
    match = re.search(r"(^|/)(worker\d+)/processed_videos\.json$", key)
    if match:
        return match.group(2)
    return "worker_unknown"


def list_worker_state_files(client, bucket: str, workers_prefix: str) -> List[Tuple[str, datetime]]:
    prefix = normalize_prefix(workers_prefix)
    if prefix:
        prefix = f"{prefix}/"

    paginator = client.get_paginator("list_objects_v2")
    out: List[Tuple[str, datetime]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if re.search(r"(^|/)worker\d+/processed_videos\.json$", key):
                out.append((key, obj.get("LastModified")))
    out.sort(key=lambda x: x[0])
    return out


def list_object_keys_under_prefix(client, bucket: str, prefix: str) -> List[str]:
    normalized = normalize_prefix(prefix)
    scan_prefix = f"{normalized}/" if normalized else ""
    paginator = client.get_paginator("list_objects_v2")
    keys: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=scan_prefix):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if key:
                keys.append(key)
    return keys


def build_asset_key_cache(client, bucket: str, merged_clips: List[Dict[str, Any]], fallback_clips_prefix: str) -> Dict[str, Any]:
    prefixes: Set[str] = set()
    for item in merged_clips:
        base = normalize_prefix(str(item.get("_worker_base_prefix", "")))
        if base:
            prefixes.add(base)

    fallback = normalize_prefix(fallback_clips_prefix)
    if fallback:
        prefixes.add(fallback)

    key_set: Set[str] = set()
    prefix_counts: Dict[str, int] = {}
    for prefix in sorted(prefixes):
        keys = list_object_keys_under_prefix(client, bucket, prefix)
        prefix_counts[prefix] = len(keys)
        key_set.update(keys)

    return {
        "prefixes": sorted(prefixes),
        "counts_by_prefix": prefix_counts,
        "total_keys": len(key_set),
        "keys": key_set,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def key_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def read_json_object(client, bucket: str, key: str) -> Dict[str, Any]:
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    payload = json.loads(body.decode("utf-8"))
    if isinstance(payload, dict):
        return payload
    return {"_raw": payload}


def load_worker_artifacts(client, bucket: str, workers_prefix: str) -> List[WorkerArtifacts]:
    rows: List[WorkerArtifacts] = []
    for state_key, state_last_modified in list_worker_state_files(client, bucket, workers_prefix):
        worker = worker_name_from_state_key(state_key)
        base = state_key[: -len("/processed_videos.json")]
        manifest_key = f"{base}/vidout_boxed/boxed_clips_manifest.json"
        metadata_key = f"{base}/vidout_boxed/boxed_clips_metadata.json"
        folders_state_key = f"{base}/vidout_boxed/processed_folders.json"

        state_payload = read_json_object(client, bucket, state_key)
        manifest_payload: Dict[str, Any] = {}
        metadata_payload: Dict[str, Any] = {}

        if key_exists(client, bucket, manifest_key):
            manifest_payload = read_json_object(client, bucket, manifest_key)
        if key_exists(client, bucket, metadata_key):
            metadata_payload = read_json_object(client, bucket, metadata_key)

        rows.append(
            WorkerArtifacts(
                worker=worker,
                state_key=state_key,
                manifest_key=manifest_key,
                metadata_key=metadata_key,
                folders_state_key=folders_state_key,
                state_last_modified=state_last_modified,
                state_payload=state_payload,
                manifest_payload=manifest_payload,
                metadata_payload=metadata_payload,
            )
        )

    return rows


def stage_rank(stage: str, custom_rank: Dict[str, int]) -> int:
    normalized = (stage or "").strip().lower()
    if normalized in custom_rank:
        return custom_rank[normalized]
    return -1


def best_stage_from_v2_record(value: Dict[str, Any]) -> str:
    stages = value.get("stages", {})
    if not isinstance(stages, dict):
        return "queued"
    best = "queued"
    for stage in PIPELINE_STAGES:
        item = stages.get(stage, {})
        if isinstance(item, dict) and item.get("done"):
            best = stage
    return best


def best_stage_from_legacy_record(value: Dict[str, Any]) -> str:
    raw_stage = str(value.get("stage", "")).strip().lower()
    mapping = {
        "start": "queued",
        "face_diarization": "queued",
        "face_done": "queued",
        "talknet": "talknet",
        "talknet_done": "talknet",
        "export_json": "export_json",
        "export_done": "export_json",
        "extract_crops": "extract_crops",
        "extract_done": "extract_crops",
        "speaker_id": "speaker_id",
        "speaker_done": "speaker_id",
        "clip_generation": "clip_generation",
        "complete": "complete",
    }
    return mapping.get(raw_stage, raw_stage or "queued")


def iter_state_stage_candidates(wf: WorkerArtifacts, custom_rank: Dict[str, int]) -> List[StageDecision]:
    out: List[StageDecision] = []
    payload = wf.state_payload

    v2_videos = payload.get("videos", {}) if isinstance(payload, dict) else {}
    if isinstance(v2_videos, dict) and v2_videos:
        for video_id, value in v2_videos.items():
            if not isinstance(value, dict):
                continue
            stage = best_stage_from_v2_record(value)
            out.append(
                StageDecision(
                    video_id=str(video_id),
                    stage=stage,
                    stage_rank=stage_rank(stage, custom_rank),
                    source_worker=wf.worker,
                    source_key=wf.state_key,
                    decided_at=best_payload_ts(value) or wf.state_last_modified,
                )
            )
        return out

    if isinstance(payload, dict):
        for video_id, value in payload.items():
            if not isinstance(value, dict):
                continue
            stage = best_stage_from_legacy_record(value)
            out.append(
                StageDecision(
                    video_id=str(video_id),
                    stage=stage,
                    stage_rank=stage_rank(stage, custom_rank),
                    source_worker=wf.worker,
                    source_key=wf.state_key,
                    decided_at=best_payload_ts(value) or wf.state_last_modified,
                )
            )
    return out


def compare_decisions(current: StageDecision, candidate: StageDecision) -> StageDecision:
    if candidate.stage_rank > current.stage_rank:
        return candidate
    if candidate.stage_rank < current.stage_rank:
        return current

    cur_ts = current.decided_at or datetime.fromtimestamp(0, tz=timezone.utc)
    cand_ts = candidate.decided_at or datetime.fromtimestamp(0, tz=timezone.utc)
    if cand_ts > cur_ts:
        return candidate
    if cand_ts < cur_ts:
        return current

    if candidate.source_worker > current.source_worker:
        return candidate
    return current


def merge_stage_data(worker_files: List[WorkerArtifacts], custom_rank: Dict[str, int]) -> Dict[str, StageDecision]:
    merged: Dict[str, StageDecision] = {}
    for wf in worker_files:
        for candidate in iter_state_stage_candidates(wf, custom_rank):
            existing = merged.get(candidate.video_id)
            if existing is None:
                merged[candidate.video_id] = candidate
            else:
                merged[candidate.video_id] = compare_decisions(existing, candidate)
    return merged


def normalize_speaker_id(item: Dict[str, Any]) -> str:
    sid = (
        item.get("global_face_id")
        or item.get("speaker")
        or item.get("speaker_id")
        or item.get("global_speaker_id")
        or "unknown"
    )
    return str(sid)


def normalize_source_video(item: Dict[str, Any], clip_path: str) -> str:
    source_video = str(item.get("source_video") or "").strip()
    if source_video:
        return source_video
    stem = Path(clip_path).stem
    m = re.match(r"(.+?)_track_\d+_f\d+-\d+$", stem)
    if m:
        return m.group(1)
    return stem


def clip_signature(item: Dict[str, Any], clip_path: str, source_video: str, speaker_id: str) -> str:
    start = float(item.get("start_time_sec", item.get("requested_start_time_sec", 0.0)) or 0.0)
    end = float(item.get("end_time_sec", item.get("requested_end_time_sec", 0.0)) or 0.0)
    return "|".join(
        [
            source_video,
            clip_path,
            speaker_id,
            f"{start:.3f}",
            f"{end:.3f}",
        ]
    )


def merge_manifest_clips(worker_files: List[WorkerArtifacts]) -> List[Dict[str, Any]]:
    chosen: Dict[str, Dict[str, Any]] = {}

    for wf in worker_files:
        clips = wf.manifest_payload.get("clips", [])
        if not isinstance(clips, list):
            continue

        for raw in clips:
            if not isinstance(raw, dict):
                continue
            clip_path = str(raw.get("clip_path", "")).strip()
            if not clip_path:
                continue

            speaker_id = normalize_speaker_id(raw)
            source_video = normalize_source_video(raw, clip_path)
            sig = clip_signature(raw, clip_path, source_video, speaker_id)

            row = dict(raw)
            row["_speaker_id"] = speaker_id
            row["_source_video"] = source_video
            row["_worker"] = wf.worker
            row["_worker_base_prefix"] = wf.manifest_key[: -len("/boxed_clips_manifest.json")]
            row["_manifest_key"] = wf.manifest_key

            existing = chosen.get(sig)
            if existing is None:
                chosen[sig] = row
                continue

            existing_ts = parse_ts(existing.get("_merged_ts")) or datetime.fromtimestamp(0, tz=timezone.utc)
            candidate_ts = best_payload_ts(raw) or wf.state_last_modified or datetime.now(timezone.utc)
            row["_merged_ts"] = candidate_ts.isoformat()

            if candidate_ts > existing_ts:
                chosen[sig] = row
            elif candidate_ts == existing_ts and row["_worker"] > existing.get("_worker", ""):
                chosen[sig] = row

    rows = list(chosen.values())
    rows.sort(key=lambda x: (x.get("_source_video", ""), float(x.get("start_time_sec", 0.0)), x.get("clip_path", "")))
    return rows


def aggregate_speaker_stats(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for c in clips:
        sid = str(c.get("_speaker_id", "unknown"))
        source_video = str(c.get("_source_video", "unknown"))
        dur = float(c.get("duration_sec", 0.0) or 0.0)

        item = stats.setdefault(
            sid,
            {
                "speaker_id": sid,
                "clip_count": 0,
                "total_duration_sec": 0.0,
                "video_ids": set(),
            },
        )
        item["clip_count"] += 1
        item["total_duration_sec"] += dur
        item["video_ids"].add(source_video)

    rows: List[Dict[str, Any]] = []
    for sid, data in stats.items():
        rows.append(
            {
                "speaker_id": sid,
                "clip_count": int(data["clip_count"]),
                "total_duration_sec": round(float(data["total_duration_sec"]), 3),
                "total_duration_min": round(float(data["total_duration_sec"]) / 60.0, 4),
                "video_count": len(data["video_ids"]),
            }
        )

    rows.sort(key=lambda x: (-x["clip_count"], -x["total_duration_sec"], x["speaker_id"]))
    return rows


def aggregate_video_stats(clips: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    by_video: Dict[str, Dict[str, Any]] = {}
    by_video_speaker: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for c in clips:
        video_id = str(c.get("_source_video", "unknown"))
        speaker_id = str(c.get("_speaker_id", "unknown"))
        dur = float(c.get("duration_sec", 0.0) or 0.0)

        vrow = by_video.setdefault(
            video_id,
            {
                "video_id": video_id,
                "speaker_ids": set(),
                "clip_count": 0,
                "total_duration_sec": 0.0,
            },
        )
        vrow["speaker_ids"].add(speaker_id)
        vrow["clip_count"] += 1
        vrow["total_duration_sec"] += dur

        srow = by_video_speaker.setdefault(video_id, {}).setdefault(
            speaker_id,
            {
                "video_id": video_id,
                "speaker_id": speaker_id,
                "clip_count": 0,
                "total_duration_sec": 0.0,
            },
        )
        srow["clip_count"] += 1
        srow["total_duration_sec"] += dur

    video_rows: List[Dict[str, Any]] = []
    for video_id, item in by_video.items():
        video_rows.append(
            {
                "video_id": video_id,
                "speaker_count": len(item["speaker_ids"]),
                "clip_count": int(item["clip_count"]),
                "total_duration_sec": round(float(item["total_duration_sec"]), 3),
                "total_duration_min": round(float(item["total_duration_sec"]) / 60.0, 4),
            }
        )

    video_rows.sort(key=lambda x: x["video_id"])

    video_speaker_rows: Dict[str, List[Dict[str, Any]]] = {}
    for video_id, spk_map in by_video_speaker.items():
        rows = []
        for speaker_id, item in spk_map.items():
            rows.append(
                {
                    "video_id": video_id,
                    "speaker_id": speaker_id,
                    "clip_count": int(item["clip_count"]),
                    "total_duration_sec": round(float(item["total_duration_sec"]), 3),
                    "total_duration_min": round(float(item["total_duration_sec"]) / 60.0, 4),
                }
            )
        rows.sort(key=lambda x: (-x["clip_count"], x["speaker_id"]))
        video_speaker_rows[video_id] = rows

    return video_rows, video_speaker_rows


def clip_duration_sec(item: Dict[str, Any]) -> float:
    d = float(item.get("duration_sec", 0.0) or 0.0)
    if d > 0:
        return d
    start = float(item.get("start_time_sec", item.get("requested_start_time_sec", 0.0)) or 0.0)
    end = float(item.get("end_time_sec", item.get("requested_end_time_sec", 0.0)) or 0.0)
    return max(0.0, end - start)


def compute_der_estimate(clips: List[Dict[str, Any]], video_ids: Optional[set] = None) -> Dict[str, Any]:
    """
    Estimate diarization error rate (DER) under assumption of 2 speakers per video.
    For each video we pick the two speaker IDs with largest summed clip duration as
    the canonical speakers, and count all other speaker-duration as errors (fragmentation/confusion).

    Returns dict with per-video durations and error seconds and overall DER.
    """
    by_video: Dict[str, Dict[str, float]] = {}

    for c in clips:
        vid = str(c.get("_source_video", "unknown"))
        if video_ids is not None and vid not in video_ids:
            continue
        sid = str(c.get("_speaker_id", "unknown"))
        by_video.setdefault(vid, {}).setdefault(sid, 0.0)
        by_video[vid][sid] += clip_duration_sec(c)

    per_video_stats: Dict[str, Dict[str, Any]] = {}
    total_error = 0.0
    total_time = 0.0

    for vid, map_sid in by_video.items():
        total_vid = sum(map_sid.values())
        total_time += total_vid
        # pick top 2 speakers by duration
        top_two = sorted(map_sid.items(), key=lambda x: -x[1])[:2]
        top_two_dur = sum(x[1] for x in top_two)
        error = max(0.0, total_vid - top_two_dur)
        total_error += error
        per_video_stats[vid] = {
            "total_duration_sec": round(total_vid, 3),
            "top_two_duration_sec": round(top_two_dur, 3),
            "error_sec": round(error, 3),
            "der": round((error / total_vid) if total_vid > 0 else 0.0, 4),
            "top_two_speakers": [x[0] for x in top_two],
        }

    overall_der = round((total_error / total_time) if total_time > 0 else 0.0, 4)
    return {
        "overall_der": overall_der,
        "total_error_sec": round(total_error, 3),
        "total_time_sec": round(total_time, 3),
        "per_video": per_video_stats,
    }


def stage_counts(merged_stages: Dict[str, StageDecision]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in merged_stages.values():
        counts[item.stage] = counts.get(item.stage, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[0]))


def write_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    # Keep deterministic column order while still handling sparse rows.
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def export_merged_artifacts_to_codebase(
    export_root: Path,
    merged_stages: Dict[str, StageDecision],
    merged_clips: List[Dict[str, Any]],
    speaker_rows: List[Dict[str, Any]],
    video_rows: List[Dict[str, Any]],
    video_speaker_rows: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, str]:
    export_root.mkdir(parents=True, exist_ok=True)

    processed_json_path = export_root / "merged_processed_videos.json"
    clips_json_path = export_root / "merged_clips_metadata.json"
    speakers_csv_path = export_root / "speaker_totals.csv"
    videos_csv_path = export_root / "video_totals.csv"
    video_speakers_csv_path = export_root / "video_speaker_breakdown.csv"

    processed_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "videos": {
            vid: {
                "stage": dec.stage,
                "stage_rank": dec.stage_rank,
                "source_worker": dec.source_worker,
                "source_key": dec.source_key,
                "decided_at": dec.decided_at.isoformat() if dec.decided_at else None,
            }
            for vid, dec in sorted(merged_stages.items())
        },
    }
    processed_json_path.write_text(json.dumps(processed_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    clips_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_clips": len(merged_clips),
        "clips": merged_clips,
    }
    clips_json_path.write_text(json.dumps(clips_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    write_csv_rows(speakers_csv_path, speaker_rows)
    write_csv_rows(videos_csv_path, video_rows)

    flat_video_speaker_rows: List[Dict[str, Any]] = []
    for video_id in sorted(video_speaker_rows.keys()):
        flat_video_speaker_rows.extend(video_speaker_rows[video_id])
    write_csv_rows(video_speakers_csv_path, flat_video_speaker_rows)

    return {
        "processed_json": str(processed_json_path),
        "clips_json": str(clips_json_path),
        "speakers_csv": str(speakers_csv_path),
        "videos_csv": str(videos_csv_path),
        "video_speakers_csv": str(video_speakers_csv_path),
    }


def zip_directory(root_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(root_dir).as_posix())


def file_name_from_path(path: Path) -> str:
    return path.name


def worker_layout_from_state_key(state_key: str) -> Dict[str, str]:
    base = state_key[: -len("/processed_videos.json")]
    return {
        "state_json": state_key,
        "manifest_json": f"{base}/vidout_boxed/boxed_clips_manifest.json",
        "metadata_json": f"{base}/vidout_boxed/boxed_clips_metadata.json",
        "folders_state_json": f"{base}/vidout_boxed/processed_folders.json",
        "clips_root": f"{base}/vidout_boxed",
    }


def first_existing_key(client, bucket: str, keys: List[str], cached_keys: Optional[Set[str]] = None) -> Optional[str]:
    if cached_keys is not None:
        for key in keys:
            if key in cached_keys:
                return key
        return None

    for key in keys:
        if key_exists(client, bucket, key):
            return key
    return None


def relative_join(prefix: str, rel: str) -> str:
    pfx = normalize_prefix(prefix)
    r = rel.lstrip("/")
    if pfx:
        return f"{pfx}/{r}"
    return r


def resolve_clip_asset_keys(
    client,
    bucket: str,
    fallback_clips_prefix: str,
    item: Dict[str, Any],
    cached_keys: Optional[Set[str]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    base = normalize_prefix(str(item.get("_worker_base_prefix", "")))
    clip_path = str(item.get("clip_path", "")).strip()

    clip_candidates: List[str] = []
    if clip_path:
        clip_candidates.append(relative_join(base, clip_path))
        if fallback_clips_prefix:
            clip_candidates.append(relative_join(fallback_clips_prefix, clip_path))

    clip_key = first_existing_key(client, bucket, clip_candidates, cached_keys=cached_keys)

    textgrid_rel = str(item.get("textgrid_clip_path") or "").strip()
    if not textgrid_rel:
        maybe_name = str(item.get("textgrid_path") or "").strip()
        if maybe_name:
            textgrid_rel = str(Path(clip_path).parent / maybe_name)

    wav_rel = str(item.get("wav_path") or "").strip()
    if wav_rel and "/" not in wav_rel:
        wav_rel = str(Path(clip_path).parent / wav_rel)

    textgrid_candidates: List[str] = []
    if textgrid_rel:
        textgrid_candidates.append(relative_join(base, textgrid_rel))
        if fallback_clips_prefix:
            textgrid_candidates.append(relative_join(fallback_clips_prefix, textgrid_rel))

    wav_candidates: List[str] = []
    if wav_rel:
        wav_candidates.append(relative_join(base, wav_rel))
        if fallback_clips_prefix:
            wav_candidates.append(relative_join(fallback_clips_prefix, wav_rel))

    textgrid_key = first_existing_key(client, bucket, textgrid_candidates, cached_keys=cached_keys)
    wav_key = first_existing_key(client, bucket, wav_candidates, cached_keys=cached_keys)

    return clip_key, textgrid_key, wav_key


def get_clip_choices_for_video(
    video_id: str,
    merged_clips: List[Dict[str, Any]],
    client,
    bucket: str,
    fallback_clips_prefix: str,
    cached_keys: Optional[Set[str]] = None,
) -> List[ClipChoice]:
    out: List[ClipChoice] = []

    for item in merged_clips:
        if str(item.get("_source_video", "")) != video_id:
            continue

        clip_key, textgrid_key, wav_key = resolve_clip_asset_keys(
            client, bucket, fallback_clips_prefix, item, cached_keys=cached_keys
        )

        out.append(
            ClipChoice(
                clip_path=str(item.get("clip_path", "")),
                clip_key=clip_key,
                textgrid_key=textgrid_key,
                wav_key=wav_key,
                speaker_id=str(item.get("_speaker_id", "unknown")),
                source_video=video_id,
                start_time_sec=float(item.get("start_time_sec", item.get("requested_start_time_sec", 0.0)) or 0.0),
                end_time_sec=float(item.get("end_time_sec", item.get("requested_end_time_sec", 0.0)) or 0.0),
                duration_sec=float(item.get("duration_sec", 0.0) or 0.0),
                worker=str(item.get("_worker", "worker_unknown")),
            )
        )

    out.sort(key=lambda x: (x.start_time_sec, x.clip_path))
    return out


def presign_object_url(client, bucket: str, key: str, expiry_seconds: int) -> str:
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiry_seconds,
    )


def parse_textgrid_intervals(content: str) -> List[Dict[str, Any]]:
    lines = content.splitlines()
    words: List[Dict[str, Any]] = []

    xmin = None
    xmax = None
    text = None

    def flush():
        nonlocal xmin, xmax, text
        if xmin is None or xmax is None or text is None:
            xmin = None
            xmax = None
            text = None
            return
        token = text.strip()
        if token:
            words.append({"start": float(xmin), "end": float(xmax), "text": token})
        xmin = None
        xmax = None
        text = None

    for raw in lines:
        line = raw.strip()
        if line.startswith("intervals ["):
            flush()
        elif line.startswith("xmin ="):
            try:
                xmin = float(line.split("=", 1)[1].strip())
            except Exception:
                xmin = None
        elif line.startswith("xmax ="):
            try:
                xmax = float(line.split("=", 1)[1].strip())
            except Exception:
                xmax = None
        elif line.startswith("text ="):
            val = line.split("=", 1)[1].strip()
            if val.startswith('"') and val.endswith('"') and len(val) >= 2:
                val = val[1:-1]
            text = val

    flush()
    words.sort(key=lambda x: (x["start"], x["end"]))
    return words


def read_textgrid_words(client, bucket: str, key: str) -> List[Dict[str, Any]]:
    content = client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8", errors="replace")
    return parse_textgrid_intervals(content)


def build_player_iframe_src(video_url: str, words: List[Dict[str, Any]], object_fit: str = "contain", max_height_px: int = 1100) -> str:
    words_json = json.dumps(words)
    video_url_json = json.dumps(video_url)
    safe_fit = "cover" if object_fit == "cover" else "contain"
    safe_max_height = max(520, int(max_height_px))
    html = f"""
<!doctype html>
<html>
<head>
<meta charset=\"utf-8\" />
<style>
  body {{
    margin: 0;
    padding: 8px;
    background: #f7fbf8;
    font-family: "Segoe UI", Tahoma, sans-serif;
    color: #10231a;
  }}
  .wrap {{
    border: 1px solid #cfe1d6;
    border-radius: 12px;
    padding: 10px;
    background: #ffffff;
    height: {safe_max_height}px;
    box-sizing: border-box;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }}
  .video-shell {{
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    aspect-ratio: 16 / 9;
    border-radius: 8px;
    border: 1px solid #d6e4db;
    background: #000;
    overflow: hidden;
    flex: 0 0 auto;
  }}
  video {{
    width: 100%;
    height: 100%;
    background: #000;
    object-fit: {safe_fit};
    display: block;
  }}
  .caption {{
    line-height: 1.7;
    font-size: 18px;
    min-height: 110px;
    flex: 1 1 auto;
    overflow: auto;
    border: 1px solid #e1ebe4;
    border-radius: 8px;
    padding: 8px;
    background: #fbfdfb;
  }}
  .w {{
    display: inline-block;
    margin: 0 5px 8px 0;
    padding: 3px 8px;
    border-radius: 8px;
    background: #eef4f0;
    border: 1px solid #e1ebe4;
    transition: all 100ms ease-out;
  }}
  .w.past {{
    opacity: 0.62;
  }}
  .w.active {{
    background: #53dc6f;
    border-color: #1f8a38;
    color: #07210f;
    font-weight: 700;
    transform: translateY(-1px);
  }}
</style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"video-shell\">
      <video id=\"video\" controls preload=\"metadata\" playsinline></video>
    </div>
    <div id=\"caption\" class=\"caption\"></div>
  </div>

<script>
const words = {words_json};
const videoSrc = {video_url_json};
const caption = document.getElementById('caption');
const video = document.getElementById('video');
video.src = videoSrc;

function render(activeIdx) {{
  let left = 0;
  let right = words.length;
  if (activeIdx >= 0) {{
    left = Math.max(0, activeIdx - 16);
    right = Math.min(words.length, activeIdx + 17);
  }}
  const windowWords = words.slice(left, right);
  caption.innerHTML = windowWords.map((w, localIdx) => {{
    const idx = left + localIdx;
    const cls = idx === activeIdx ? 'w active' : (idx < activeIdx ? 'w past' : 'w');
    return `<span class="${{cls}}">${{w.text}}</span>`;
  }}).join(' ');
}}

function activeWordIndex(t) {{
  for (let i = 0; i < words.length; i += 1) {{
    if (t >= words[i].start && t <= words[i].end) return i;
  }}
  return -1;
}}

video.addEventListener('timeupdate', () => {{
  render(activeWordIndex(video.currentTime));
}});

render(-1);
</script>
</body>
</html>
"""
    return html


def video_bytes_to_data_url(video_bytes: bytes) -> str:
    return "data:video/mp4;base64," + base64.b64encode(video_bytes).decode("ascii")


def get_ffmpeg_binary() -> str:
    configured = os.getenv("FFMPEG_BINARY", "").strip()
    if configured:
        return configured
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    if get_ffmpeg_exe is not None:
        return get_ffmpeg_exe()
    raise RuntimeError(
        "ffmpeg is not available. Set FFMPEG_BINARY, install ffmpeg, or add imageio-ffmpeg."
    )


def extract_wav_from_mp4(mp4_path: Path, wav_path: Path, target_sr: int = 16000) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        get_ffmpeg_binary(),
        "-y",
        "-i",
        str(mp4_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(target_sr)),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ffmpeg wav extraction failed for {mp4_path}: {err[:500]}")


def transcode_mp4_for_web(input_mp4_path: Path, output_mp4_path: Path) -> None:
    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        get_ffmpeg_binary(),
        "-y",
        "-i",
        str(input_mp4_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(output_mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ffmpeg web transcode failed for {input_mp4_path}: {err[:700]}")


def get_browser_playable_video_bytes(
    client,
    bucket: str,
    clip_key: str,
    cache_root: Path,
    force_transcode: bool,
) -> bytes:
    cache_root.mkdir(parents=True, exist_ok=True)
    key_hash = hashlib.sha1(clip_key.encode("utf-8")).hexdigest()
    source_mp4 = cache_root / f"{key_hash}_src.mp4"
    web_mp4 = cache_root / f"{key_hash}_web.mp4"

    if not source_mp4.exists() or source_mp4.stat().st_size == 0:
        client.download_file(bucket, clip_key, str(source_mp4))

    if not force_transcode:
        return source_mp4.read_bytes()

    if not web_mp4.exists() or web_mp4.stat().st_size == 0:
        transcode_mp4_for_web(source_mp4, web_mp4)
    return web_mp4.read_bytes()


def sample_and_download_local(
    client,
    bucket: str,
    merged_clips: List[Dict[str, Any]],
    fallback_clips_prefix: str,
    output_root: Path,
    target_ratio: float,
    seed: int,
    include_sidecars: bool,
    cached_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    valid_pool = [item for item in merged_clips if clip_duration_sec(item) > 0.0]
    total_valid_duration_sec = sum(clip_duration_sec(item) for item in valid_pool)

    target_ratio = max(0.0, float(target_ratio))
    target_duration_sec = total_valid_duration_sec * target_ratio

    rng = random.Random(seed)

    chosen_items: List[Dict[str, Any]] = []
    chosen_duration_sec = 0.0
    if target_duration_sec > 0 and valid_pool:
        shuffled_pool = valid_pool[:]
        rng.shuffle(shuffled_pool)
        for item in shuffled_pool:
            d = clip_duration_sec(item)
            if d <= 0:
                continue

            if chosen_duration_sec < target_duration_sec <= (chosen_duration_sec + d):
                # Choose the closer point to target when crossing it.
                if abs(target_duration_sec - chosen_duration_sec) <= abs((chosen_duration_sec + d) - target_duration_sec):
                    break

            chosen_items.append(item)
            chosen_duration_sec += d

            # Soft upper bound to stay around target, not strict.
            if chosen_duration_sec >= target_duration_sec * 1.08:
                break

    by_video: Dict[str, List[Dict[str, Any]]] = {}
    for item in chosen_items:
        video_id = str(item.get("_source_video", "unknown"))
        by_video.setdefault(video_id, []).append(item)

    output_root.mkdir(parents=True, exist_ok=True)
    now_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = output_root / f"sample_{now_stamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": bucket,
        "fallback_clips_prefix": fallback_clips_prefix,
        "sampling_mode": "duration_target",
        "target_ratio": target_ratio,
        "target_duration_sec": round(target_duration_sec, 3),
        "target_duration_min": round(target_duration_sec / 60.0, 4),
        "total_valid_duration_sec": round(total_valid_duration_sec, 3),
        "total_valid_duration_min": round(total_valid_duration_sec / 60.0, 4),
        "seed": seed,
        "selected_video_count": len(by_video),
        "selected_clip_count": len(chosen_items),
        "selected_duration_sec": round(chosen_duration_sec, 3),
        "selected_duration_min": round(chosen_duration_sec / 60.0, 4),
        "videos": {},
        "flat_rows": [],
    }

    all_assets_dir = run_root / "all_sampled_assets"
    all_assets_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []

    for video_id in sorted(by_video.keys()):
        selected = by_video[video_id][:]

        manifest_rows: List[Dict[str, Any]] = []

        for item in selected:
            clip_key, textgrid_key, wav_key = resolve_clip_asset_keys(
                client, bucket, fallback_clips_prefix, item, cached_keys=cached_keys
            )
            if not clip_key:
                continue

            speaker_id = str(item.get("_speaker_id", "unknown"))
            clip_name = Path(str(item.get("clip_path", ""))).name
            safe_video = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_id)
            safe_speaker = re.sub(r"[^A-Za-z0-9_.-]+", "_", speaker_id)
            unique_stem = f"{safe_video}__{safe_speaker}__{Path(clip_name).stem}"
            local_clip_dir = all_assets_dir / unique_stem
            local_clip_dir.mkdir(parents=True, exist_ok=True)

            local_clip_path = local_clip_dir / "clip.mp4"
            client.download_file(bucket, clip_key, str(local_clip_path))

            local_tg_path = None
            local_wav_path = local_clip_dir / "clip.wav"

            if include_sidecars and textgrid_key:
                local_tg_path = local_clip_dir / "clip.TextGrid"
                client.download_file(bucket, textgrid_key, str(local_tg_path))

            extract_wav_from_mp4(local_clip_path, local_wav_path, target_sr=16000)

            row = {
                "video_id": video_id,
                "speaker_id": speaker_id,
                "worker": item.get("_worker"),
                "clip_path": item.get("clip_path", ""),
                "clip_s3_key": clip_key,
                "textgrid_s3_key": textgrid_key,
                "wav_s3_key": wav_key,
                "local_clip_path": str(local_clip_path),
                "local_textgrid_path": str(local_tg_path) if local_tg_path else None,
                "local_wav_path": str(local_wav_path),
                "start_time_sec": float(item.get("start_time_sec", item.get("requested_start_time_sec", 0.0)) or 0.0),
                "end_time_sec": float(item.get("end_time_sec", item.get("requested_end_time_sec", 0.0)) or 0.0),
                "duration_sec": float(item.get("duration_sec", 0.0) or 0.0),
                "source_video": item.get("_source_video"),
            }

            manifest_rows.append(row)
            all_rows.append(row)

        manifest["videos"][video_id] = {
            "requested_clips": len(selected),
            "downloaded_clips": len(manifest_rows),
            "clips": manifest_rows,
        }

    manifest["flat_rows"] = all_rows

    manifest_path = run_root / "sampling_manifest.json"
    manifest_csv_path = run_root / "sampling_manifest.csv"
    bundle_zip_path = run_root / "sample_bundle.zip"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv_rows(manifest_csv_path, all_rows)
    zip_directory(run_root, bundle_zip_path)

    return {
        "run_dir": str(run_root),
        "assets_dir": str(all_assets_dir),
        "manifest_path": str(manifest_path),
        "manifest_csv_path": str(manifest_csv_path),
        "bundle_zip_path": str(bundle_zip_path),
        "target_wav_sr": 16000,
        "videos_sampled": len(by_video),
        "target_ratio": target_ratio,
        "target_duration_sec": round(target_duration_sec, 3),
        "selected_duration_sec": round(chosen_duration_sec, 3),
        "total_downloaded_clips": sum(v["downloaded_clips"] for v in manifest["videos"].values()),
    }


def main() -> None:
    st.set_page_config(page_title="S3 TextGrid Review", layout="wide")
    st.title("S3 Video + TextGrid Review")
    st.caption("Worker-aware merge for processed state, boxed clip manifests, clip playback, and random local sampling.")

    with st.sidebar:
        st.header("S3 Connection")
        bucket = st.text_input("Bucket", value=APP_DEFAULTS.get("bucket", ""))
        workers_prefix = st.text_input("Workers prefix", value=APP_DEFAULTS.get("workers_prefix", ""))
        fallback_clips_prefix = st.text_input(
            "Fallback clips prefix (optional)",
            value=APP_DEFAULTS.get("fallback_clips_prefix", ""),
            help="Optional backup prefix for clip/textgrid/wav lookup if worker vidout_boxed path fails.",
        )
        region = st.text_input("Region (optional)", value=APP_DEFAULTS.get("region", ""))

        with st.expander("Optional explicit credentials"):
            access_key = st.text_input("AWS Access Key ID", value=APP_DEFAULTS.get("access_key", ""))
            secret_key = st.text_input("AWS Secret Access Key", value=APP_DEFAULTS.get("secret_key", ""), type="password")
            session_token = st.text_input("AWS Session Token", value=APP_DEFAULTS.get("session_token", ""), type="password")

        custom_rank_json = st.text_area(
            "Stage rank map (JSON)",
            value=json.dumps(DEFAULT_STAGE_RANK, indent=2),
            height=180,
        )
        url_expiry = st.number_input("Presigned URL expiry (sec)", min_value=60, max_value=86400, value=int(APP_DEFAULTS.get("url_expiry", 3600)))
        run_load = st.button("Load from S3", type="primary")

    if APP_DEFAULTS.get("auto_load_on_start", False) and not run_load and "s3_review_loaded" not in st.session_state:
        run_load = True

    if run_load:
        if not bucket:
            st.error("Bucket is required.")
            return

        try:
            custom_rank = json.loads(custom_rank_json)
            if not isinstance(custom_rank, dict):
                raise ValueError("Stage rank JSON must be an object.")
            custom_rank = {str(k).strip().lower(): int(v) for k, v in custom_rank.items()}
        except Exception as exc:
            st.error(f"Invalid stage rank JSON: {exc}")
            return

        try:
            client = build_s3_client(region, access_key, secret_key, session_token)
            worker_files = load_worker_artifacts(client, bucket, workers_prefix)
        except Exception as exc:
            st.exception(exc)
            return

        if not worker_files:
            st.warning("No worker processed_videos.json files found under the given prefix.")
            return

        merged_stages = merge_stage_data(worker_files, custom_rank)
        merged_clips = merge_manifest_clips(worker_files)
        speaker_rows = aggregate_speaker_stats(merged_clips)
        video_rows, video_speaker_rows = aggregate_video_stats(merged_clips)
        with st.spinner("Indexing clip/TextGrid/WAV keys once for fast lookup..."):
            asset_key_cache = build_asset_key_cache(client, bucket, merged_clips, fallback_clips_prefix)
        merged_export_paths = export_merged_artifacts_to_codebase(
            export_root=Path("testing") / "s3_review_exports",
            merged_stages=merged_stages,
            merged_clips=merged_clips,
            speaker_rows=speaker_rows,
            video_rows=video_rows,
            video_speaker_rows=video_speaker_rows,
        )

        st.session_state["s3_review_loaded"] = {
            "bucket": bucket,
            "workers_prefix": workers_prefix,
            "fallback_clips_prefix": fallback_clips_prefix,
            "region": region,
            "access_key": access_key,
            "secret_key": secret_key,
            "session_token": session_token,
            "custom_rank": custom_rank,
            "worker_files": worker_files,
            "merged_stages": merged_stages,
            "merged_clips": merged_clips,
            "speaker_rows": speaker_rows,
            "video_rows": video_rows,
            "video_speaker_rows": video_speaker_rows,
            "asset_key_cache": asset_key_cache,
            "merged_export_paths": merged_export_paths,
        }

    loaded = st.session_state.get("s3_review_loaded")
    if not loaded:
        st.info("Set S3 details and click Load from S3.")
        return

    if bucket != loaded.get("bucket") or workers_prefix != loaded.get("workers_prefix") or fallback_clips_prefix != loaded.get("fallback_clips_prefix"):
        st.warning("Inputs changed. Click Load from S3 to refresh with the new values.")

    worker_files: List[WorkerArtifacts] = loaded["worker_files"]
    merged_stages: Dict[str, StageDecision] = loaded["merged_stages"]
    merged_clips: List[Dict[str, Any]] = loaded["merged_clips"]
    speaker_rows: List[Dict[str, Any]] = loaded["speaker_rows"]
    video_rows: List[Dict[str, Any]] = loaded["video_rows"]
    video_speaker_rows: Dict[str, List[Dict[str, Any]]] = loaded["video_speaker_rows"]
    asset_key_cache: Dict[str, Any] = loaded.get("asset_key_cache", {})
    cached_asset_keys = asset_key_cache.get("keys")
    merged_export_paths: Dict[str, str] = loaded.get("merged_export_paths", {})

    client = build_s3_client(
        loaded.get("region", ""),
        loaded.get("access_key", ""),
        loaded.get("secret_key", ""),
        loaded.get("session_token", ""),
    )

    top1, top2, top3, top4 = st.columns(4)
    # compute valid videos: videos that have clips and are marked complete
    videos_with_clips = set([c.get("_source_video") for c in merged_clips if c.get("_source_video")])
    completed_videos = set([vid for vid, dec in merged_stages.items() if dec.stage == "complete"])
    valid_videos = videos_with_clips.intersection(completed_videos)
    der_stats = compute_der_estimate(merged_clips, video_ids=valid_videos)

    top1.metric("Worker state files", len(worker_files))
    top2.metric("Merged videos", len(merged_stages))
    top3.metric("Valid videos", len(valid_videos))
    top4.metric("Merged clips", len(merged_clips))
    top5, top6 = st.columns(2)
    top5.metric("Unique speakers", len(speaker_rows))
    top6.metric("DER (2-speaker assumption)", f"{der_stats['overall_der']:.4f}")

    with st.expander("S3 layout extracted from pipeline cell", expanded=False):
        if worker_files:
            sample_layout = worker_layout_from_state_key(worker_files[0].state_key)
            st.write("Expected structure per worker:")
            st.json(sample_layout)
        st.write("Detected worker state files:")
        st.dataframe(
            [
                {
                    "worker": w.worker,
                    "state_key": w.state_key,
                    "manifest_key": w.manifest_key,
                    "metadata_key": w.metadata_key,
                }
                for w in worker_files
            ],
            width="stretch",
            hide_index=True,
        )

    with st.expander("Merged exports saved in codebase", expanded=False):
        st.json(merged_export_paths)

        if merged_export_paths:
            st.write("One-click download merged exports:")
            for label, path_str in merged_export_paths.items():
                path = Path(path_str)
                if not path.exists() or not path.is_file():
                    continue
                st.download_button(
                    label=f"Download {label}",
                    data=path.read_bytes(),
                    file_name=file_name_from_path(path),
                    mime="application/octet-stream",
                    key=f"download_merged_{label}",
                )

    with st.expander("Cached asset key index", expanded=False):
        st.write(
            {
                "prefix_count": len(asset_key_cache.get("prefixes", [])),
                "total_indexed_keys": int(asset_key_cache.get("total_keys", 0)),
                "built_at_utc": asset_key_cache.get("built_at_utc"),
            }
        )

    tabs = st.tabs(["Overview", "Video Explorer", "Random Sampling"]) 

    with tabs[0]:
        with st.expander("Merged stage counts", expanded=True):
            st.json(stage_counts(merged_stages))

        with st.expander("DER estimate", expanded=True):
            st.caption("Assumption: each valid video has exactly 2 reference speakers. Any additional fragmented identities are treated as error against that 2-speaker reference.")
            st.write(
                {
                    "overall_der": der_stats["overall_der"],
                    "total_error_sec": der_stats["total_error_sec"],
                    "total_time_sec": der_stats["total_time_sec"],
                    "valid_videos_count": len(valid_videos),
                }
            )
            der_rows = [
                {"video_id": vid, **payload}
                for vid, payload in sorted(der_stats["per_video"].items(), key=lambda x: x[0])
            ]
            st.dataframe(der_rows, width="stretch", hide_index=True)

        with st.expander("Per-speaker totals across all workers", expanded=True):
            st.dataframe(speaker_rows, width="stretch", hide_index=True)

        with st.expander("Per-video totals", expanded=True):
            st.dataframe(video_rows, width="stretch", hide_index=True)

        with st.expander("Duration filter calculator", expanded=True):
            min_clip_sec = st.number_input(
                "Minimum seconds per clip",
                min_value=0.0,
                value=0.0,
                step=0.1,
                key="overview_min_clip_sec",
            )
            qualifying = [c for c in merged_clips if clip_duration_sec(c) >= float(min_clip_sec)]
            total_qualifying_sec = sum(clip_duration_sec(c) for c in qualifying)
            qualifying_videos = set(str(c.get("_source_video", "")) for c in qualifying if c.get("_source_video"))
            q1, q2, q3 = st.columns(3)
            q1.metric("Matching clips", len(qualifying))
            q2.metric("Videos with matching clips", len(qualifying_videos))
            q3.metric("Total valid time (min)", round(total_qualifying_sec / 60.0, 3))
            st.caption(f"Total valid time (sec): {round(total_qualifying_sec, 3)}")

    with tabs[1]:
        if not video_rows:
            st.warning("No clips found in worker manifests.")
        else:
            vctrl1, vctrl2 = st.columns(2)
            player_fit = vctrl1.selectbox("Player fit mode", ["contain", "cover"], index=1, help="Use cover to zoom in and make faces easier to see.")
            player_height = vctrl2.slider("Player height (px)", min_value=600, max_value=1400, value=1000, step=20)
            use_websafe_transcode = st.checkbox(
                "Web-safe playback (cached local H.264 transcode)",
                value=True,
                help="Fixes black-screen browser playback for incompatible MP4 codecs. First load per clip may be slower.",
            )

            video_ids = [row["video_id"] for row in video_rows]
            selected_video = st.selectbox("Select video", video_ids, key="selected_video")

            selected_video_row = next((r for r in video_rows if r["video_id"] == selected_video), None)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Video", selected_video)
            m2.metric("Speakers", int(selected_video_row.get("speaker_count", 0)) if selected_video_row else 0)
            m3.metric("Clips", int(selected_video_row.get("clip_count", 0)) if selected_video_row else 0)
            m4.metric("Duration (sec)", float(selected_video_row.get("total_duration_sec", 0.0)) if selected_video_row else 0.0)

            with st.expander("Speaker breakdown for selected video", expanded=True):
                st.dataframe(video_speaker_rows.get(selected_video, []), width="stretch", hide_index=True)

            with st.spinner("Loading clip/textgrid/wav keys from cache..."):
                clip_choices = get_clip_choices_for_video(
                    selected_video,
                    merged_clips,
                    client,
                    loaded["bucket"],
                    loaded.get("fallback_clips_prefix", ""),
                    cached_keys=cached_asset_keys,
                )

            if not clip_choices:
                st.warning("No clip entries found for the selected video.")
            else:
                labels = []
                for c in clip_choices:
                    if c.clip_key and c.textgrid_key:
                        status = "ok"
                    elif c.clip_key and not c.textgrid_key:
                        status = "video-only"
                    else:
                        status = "missing-video"
                    labels.append(
                        f"[{status}] {Path(c.clip_path).name} | worker={c.worker} | speaker={c.speaker_id} | dur={c.duration_sec:.2f}s | t={c.start_time_sec:.2f}-{c.end_time_sec:.2f}s"
                    )

                choice_idx = st.selectbox(
                    "Select clip",
                    list(range(len(clip_choices))),
                    format_func=lambda i: labels[i],
                    key=f"clip_select_{selected_video}",
                )
                chosen = clip_choices[choice_idx]

                with st.expander("Clip metadata", expanded=False):
                    st.write(
                        {
                            "clip_path": chosen.clip_path,
                            "worker": chosen.worker,
                            "clip_key": chosen.clip_key,
                            "textgrid_key": chosen.textgrid_key,
                            "wav_key": chosen.wav_key,
                            "speaker_id": chosen.speaker_id,
                            "start_time_sec": chosen.start_time_sec,
                            "end_time_sec": chosen.end_time_sec,
                            "duration_sec": chosen.duration_sec,
                        }
                    )

                if not chosen.clip_key:
                    st.error("Clip object not found in S3 for this manifest row.")
                else:
                    try:
                        video_url = presign_object_url(client, loaded["bucket"], chosen.clip_key, int(url_expiry))
                    except Exception as exc:
                        st.exception(exc)
                        return

                    textgrid_url = None
                    wav_url = None
                    if chosen.textgrid_key:
                        try:
                            textgrid_url = presign_object_url(client, loaded["bucket"], chosen.textgrid_key, int(url_expiry))
                        except Exception:
                            textgrid_url = None
                    if chosen.wav_key:
                        try:
                            wav_url = presign_object_url(client, loaded["bucket"], chosen.wav_key, int(url_expiry))
                        except Exception:
                            wav_url = None

                    link_cols = st.columns(3)
                    link_cols[0].markdown(f"[Download clip]({video_url})")
                    if textgrid_url:
                        link_cols[1].markdown(f"[Download TextGrid]({textgrid_url})")
                    else:
                        link_cols[1].markdown("TextGrid unavailable")
                    if wav_url:
                        link_cols[2].markdown(f"[Download WAV]({wav_url})")
                    else:
                        link_cols[2].markdown("WAV unavailable")

                    words: List[Dict[str, Any]] = []
                    if chosen.textgrid_key:
                        try:
                            words = read_textgrid_words(client, loaded["bucket"], chosen.textgrid_key)
                        except Exception:
                            st.warning("Failed to parse TextGrid. Video playback remains available.")
                    else:
                        st.warning("TextGrid missing for this clip. Playing video without highlighted words.")
                    try:
                        with st.spinner("Preparing browser-playable video..."):
                            video_bytes = get_browser_playable_video_bytes(
                                client=client,
                                bucket=loaded["bucket"],
                                clip_key=chosen.clip_key,
                                cache_root=Path("testing") / "s3_video_playback_cache",
                                force_transcode=bool(use_websafe_transcode),
                            )
                        player_src = build_player_iframe_src(
                            video_url=video_bytes_to_data_url(video_bytes),
                            words=words,
                            object_fit=player_fit,
                            max_height_px=int(player_height),
                        )
                        components.html(player_src, height=int(player_height), scrolling=False)
                    except Exception as exc:
                        st.warning("Enhanced highlighted player failed. Falling back to plain video playback.")
                        st.exception(exc)
                        st.video(video_url)

    with tabs[2]:
        st.write("Seeded statistical sampling around 1% of total valid clip duration (duration-based, not clip-count based).")

        total_valid_duration_sec = sum(clip_duration_sec(item) for item in merged_clips)
        target_ratio = 0.01
        target_duration_sec = total_valid_duration_sec * target_ratio

        c1, c2, c3 = st.columns(3)
        c1.metric("Total valid duration (min)", round(total_valid_duration_sec / 60.0, 2))
        c2.metric("Sampling target", "1.00%")
        c3.metric("Target duration (min)", round(target_duration_sec / 60.0, 2))

        sample_seed = st.number_input("Random seed", min_value=0, max_value=10**9, value=42, step=1)

        out_dir_str = st.text_input("Local output folder", value=APP_DEFAULTS.get("sample_output_dir", "testing/s3_review_samples"))
        include_sidecars = st.checkbox("Download TextGrid/WAV sidecars when available", value=True)

        if st.button("Create local sample set", type="primary"):
            try:
                output_root = Path(out_dir_str).resolve()
                with st.spinner("Sampling and downloading clips..."):
                    result = sample_and_download_local(
                        client=client,
                        bucket=loaded["bucket"],
                        merged_clips=merged_clips,
                        fallback_clips_prefix=loaded.get("fallback_clips_prefix", ""),
                        output_root=output_root,
                        target_ratio=float(target_ratio),
                        seed=int(sample_seed),
                        include_sidecars=bool(include_sidecars),
                        cached_keys=cached_asset_keys,
                    )
                st.session_state["latest_sampling_result"] = result
                st.success("Sample set created.")
                st.json(result)
            except Exception as exc:
                st.exception(exc)

        latest_sampling_result = st.session_state.get("latest_sampling_result")
        if latest_sampling_result:
            with st.expander("Latest sample set downloads", expanded=True):
                st.json(latest_sampling_result)

                for k in ("manifest_path", "manifest_csv_path", "bundle_zip_path"):
                    path_str = latest_sampling_result.get(k)
                    if not path_str:
                        continue
                    p = Path(path_str)
                    if not p.exists() or not p.is_file():
                        continue

                    mime = "application/json" if p.suffix.lower() == ".json" else "text/csv"
                    if p.suffix.lower() == ".zip":
                        mime = "application/zip"

                    st.download_button(
                        label=f"Download {p.name}",
                        data=p.read_bytes(),
                        file_name=p.name,
                        mime=mime,
                        key=f"download_sample_{k}",
                    )


if __name__ == "__main__":
    main()
