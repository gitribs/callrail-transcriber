#!/usr/bin/env python3
"""Batch-transcribe CallRail call recordings from a CSV of URLs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
from dotenv import load_dotenv

MAX_AUDIO_BYTES = 24 * 1024 * 1024
MAX_DOWNLOAD_HOPS = 3
RETRY_ATTEMPTS = 5
RETRY_BASE_SECONDS = 2.0
RETRY_MAX_SECONDS = 60.0
HTTP_TIMEOUT = httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0)
USER_AGENT = "CallRailTranscriber/1.0"

URL_HEADERS = ("recording_url", "recording_player", "recording", "url")
CALL_ID_HEADERS = ("call_id", "id", "call id")
NAME_HEADERS = ("caller_name", "customer_name", "name")
PHONE_HEADERS = ("phone number", "customer_phone_number", "caller_phone", "phone")
DATE_HEADERS = ("date", "start_time", "call_date")
TRACKING_HEADERS = ("tracking_id", "tracking id", "tracking_number", "tracking number")

AUDIO_CONTENT_TYPES = (
    "audio/",
    "video/mp4",
    "video/mpeg",
    "application/ogg",
    "application/octet-stream",
)
AUDIO_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg"}
DEFAULT_TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_SPEAKER_MODEL = "gpt-4o-mini"
ALLOWED_SPEAKERS = ("OPERATOR", "CALLER", "VOICEMAIL", "UNKNOWN")
MIN_DUPLICATE_BLOCK_WORDS = 12
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

SPEAKER_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "call_type": {"type": "string", "enum": ["conversation", "voicemail"]},
        "uncertain": {"type": "boolean"},
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "speaker": {"type": "string", "enum": list(ALLOWED_SPEAKERS)},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "text"],
            },
        },
    },
    "required": ["call_type", "uncertain", "turns"],
}

SPEAKER_SYSTEM_PROMPT = """You assign speaker labels to an existing phone-call transcript.

There are two possible speakers:
- OPERATOR: the person answering/handling the call for the business
- CALLER: the person who initiated the call

If the recording is a leftover voicemail or one-person message, use VOICEMAIL and do not invent a second speaker.

You are NOT transcribing and NOT rewriting. The input transcript is the source of truth for what was spoken.

You MAY:
- assign OPERATOR, CALLER, VOICEMAIL, or UNKNOWN labels
- split the transcript into logical speaker turns
- combine adjacent fragments that clearly belong to the same speaker
- omit a passage from the FINAL turns only when the original transcript contains an entire exchange duplicated consecutively as a processing artifact
- copy punctuation as-is, or make punctuation/capitalization/spacing changes that do not change any words

You must NOT:
- paraphrase, summarize, or rewrite
- improve grammar
- change statements into questions or questions into statements
- add words, remove words (except a clearly duplicated artifact block), or "correct" unusual words
- infer what someone "must have meant"
- substitute more natural-sounding language
- invent missing dialogue
- assume the first words are the operator solely because they come first

Use conversational context: who answers the phone, introductions, names, questions and responses, turn-taking, and statements that indicate someone works for the business.

If you cannot confidently assign a passage, keep the exact wording and label it UNKNOWN. Set uncertain=true in that case.

Return JSON only matching the schema. Each turns[].text must be copied from the original transcript, in order, covering the whole transcript (except one optional duplicated artifact block)."""

SPEAKER_RETRY_PROMPT = """Your previous labels changed the original words. Copy the transcript text EXACTLY.

Do not paraphrase. Do not add or remove words except one consecutive duplicated artifact block. Do not turn statements into questions. Split into speaker turns only."""


class RetryableError(Exception):
    """A transient error that is worth retrying."""


class FatalRowError(Exception):
    """A per-row error that should be recorded and skipped."""


@dataclass
class CallRow:
    row_number: int
    call_id: str
    recording_url: str
    caller_name: str
    phone: str
    date: str
    tracking_id: str = ""


@dataclass
class ProgressRecord:
    id: str
    url: str
    status: str
    caller_name: str
    phone: str
    date: str
    transcript: str = ""
    error: str = ""
    processed_at: str = ""
    audio_path: str = ""
    speaker_uncertain: bool = False
    raw_transcript: str = ""
    tracking_id: str = ""


@dataclass
class FormattedTranscript:
    text: str
    speaker_uncertain: bool = False
    raw_transcript: str = ""


def display_or_unknown(value: str) -> str:
    value = (value or "").strip()
    return value if value else "Unknown"


def normalize_header(name: str) -> str:
    return re.sub(r"\s+", " ", name.replace("_", " ").strip().lower())


def find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    by_normalized = {normalize_header(name): name for name in fieldnames if name}
    for candidate in candidates:
        match = by_normalized.get(normalize_header(candidate))
        if match:
            return match
    return None


def url_fallback_id(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"url-{digest}"


def load_calls(csv_path: Path) -> list[CallRow]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"No header row found in {csv_path}")

        url_col = find_column(list(reader.fieldnames), URL_HEADERS)
        if not url_col:
            expected = ", ".join(URL_HEADERS)
            raise SystemExit(
                f"Could not find a recording URL column in {csv_path}. "
                f"Expected one of: {expected}"
            )

        id_col = find_column(list(reader.fieldnames), CALL_ID_HEADERS)
        name_col = find_column(list(reader.fieldnames), NAME_HEADERS)
        phone_col = find_column(list(reader.fieldnames), PHONE_HEADERS)
        date_col = find_column(list(reader.fieldnames), DATE_HEADERS)
        tracking_col = find_column(list(reader.fieldnames), TRACKING_HEADERS)

        rows: list[CallRow] = []
        seen_ids: dict[str, int] = {}
        for index, raw in enumerate(reader, start=2):
            url = (raw.get(url_col) or "").strip()
            call_id = (raw.get(id_col) or "").strip() if id_col else ""
            if not call_id:
                call_id = url_fallback_id(url) if url else f"row-{index}"
            if call_id in seen_ids:
                call_id = f"{call_id}-row{index}"
            seen_ids[call_id] = index
            rows.append(
                CallRow(
                    row_number=index,
                    call_id=call_id,
                    recording_url=url,
                    caller_name=(raw.get(name_col) or "").strip() if name_col else "",
                    phone=(raw.get(phone_col) or "").strip() if phone_col else "",
                    date=(raw.get(date_col) or "").strip() if date_col else "",
                    tracking_id=(raw.get(tracking_col) or "").strip() if tracking_col else "",
                )
            )
    return rows


def progress_path(output_dir: Path) -> Path:
    return output_dir / "progress.jsonl"


def transcripts_path(output_dir: Path) -> Path:
    return output_dir / "transcripts.md"


def failures_path(output_dir: Path) -> Path:
    return output_dir / "failures.csv"


def load_progress(path: Path) -> dict[str, ProgressRecord]:
    latest: dict[str, ProgressRecord] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed progress line {line_number}", file=sys.stderr)
                continue
            latest[data["id"]] = ProgressRecord(
                id=data.get("id", ""),
                url=data.get("url", ""),
                status=data.get("status", ""),
                caller_name=data.get("caller_name", ""),
                phone=data.get("phone", ""),
                date=data.get("date", ""),
                transcript=data.get("transcript", ""),
                error=data.get("error", ""),
                processed_at=data.get("processed_at", ""),
                audio_path=data.get("audio_path", ""),
                speaker_uncertain=bool(data.get("speaker_uncertain", False)),
                raw_transcript=data.get("raw_transcript", "") or "",
                tracking_id=data.get("tracking_id", "") or "",
            )
    return latest


def append_progress(path: Path, record: ProgressRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def heading_date(value: str) -> str:
    text = display_or_unknown(value)
    if text == "Unknown":
        return text
    return text.split("T")[0].split(" ")[0]


def html_comment_raw(raw: str) -> str:
    safe = raw.replace("--", "- -")
    return f"<!-- RAW_TRANSCRIPT\n{safe}\n-->"


def render_call_section(record: ProgressRecord) -> str:
    title_id = record.tracking_id.strip() if record.tracking_id.strip() else record.id
    title = (
        f"{title_id} — {display_or_unknown(record.caller_name)} — {heading_date(record.date)}"
    )
    body = (record.transcript or "").strip() or "_No transcript text returned._"
    lines = [f"<!-- CALL_START id={record.id} -->"]
    if record.speaker_uncertain:
        lines.append("<!-- SPEAKER_UNCERTAIN -->")
    lines.extend(
        [
            f"## {title}",
            "",
            f"- **Caller:** {display_or_unknown(record.caller_name)}",
            f"- **Phone:** {display_or_unknown(record.phone)}",
            f"- **Date:** {display_or_unknown(record.date)}",
            f"- **Tracking ID:** {display_or_unknown(record.tracking_id)}",
            f"- **Recording URL:** {record.url or 'Unknown'}",
            "",
        ]
    )
    if record.speaker_uncertain:
        lines.extend(["_Speaker roles are uncertain; review this call._", ""])
    lines.append(body)
    raw = (record.raw_transcript or "").strip()
    if raw:
        lines.extend(["", html_comment_raw(raw)])
    lines.extend([f"<!-- CALL_END id={record.id} -->", ""])
    return "\n".join(lines)


def with_csv_metadata(record: ProgressRecord, row: CallRow) -> ProgressRecord:
    return ProgressRecord(
        id=record.id,
        url=record.url or row.recording_url,
        status=record.status,
        caller_name=row.caller_name or record.caller_name,
        phone=row.phone or record.phone,
        date=row.date or record.date,
        transcript=record.transcript,
        error=record.error,
        processed_at=record.processed_at,
        audio_path=record.audio_path,
        speaker_uncertain=record.speaker_uncertain,
        raw_transcript=record.raw_transcript,
        tracking_id=row.tracking_id or record.tracking_id,
    )


def rebuild_outputs(output_dir: Path, rows: list[CallRow], latest: dict[str, ProgressRecord]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    successes = [
        with_csv_metadata(latest[row.call_id], row)
        for row in rows
        if latest.get(row.call_id) and latest[row.call_id].status == "success"
    ]
    extras = [
        record
        for record_id, record in latest.items()
        if record.status == "success" and all(row.call_id != record_id for row in rows)
    ]
    sections = [render_call_section(record) for record in successes + extras]

    header = [
        "# Call transcripts",
        "",
        f"Generated {iso_now()}. {len(successes) + len(extras)} transcribed call(s).",
        "",
        "Each call is wrapped in `CALL_START` / `CALL_END` comments so you can split or filter this file later.",
        "",
    ]
    transcripts_path(output_dir).write_text("\n".join(header + sections), encoding="utf-8")

    failed = [latest[row.call_id] for row in rows if latest.get(row.call_id) and latest[row.call_id].status == "failed"]
    with failures_path(output_dir).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "url", "caller_name", "phone", "date", "error"],
        )
        writer.writeheader()
        for record in failed:
            writer.writerow(
                {
                    "id": record.id,
                    "url": record.url,
                    "caller_name": record.caller_name,
                    "phone": record.phone,
                    "date": record.date,
                    "error": record.error,
                }
            )


def recording_redirect_url(url: str) -> str | None:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path.endswith("/recording/redirect"):
        return None
    if path.endswith("/recording"):
        return urlunparse(parsed._replace(path=path + "/redirect"))
    return None


def looks_like_json(prefix: bytes) -> bool:
    stripped = prefix.lstrip()
    return stripped.startswith(b"{") or stripped.startswith(b"[")


def looks_like_html(prefix: bytes) -> bool:
    stripped = prefix.lstrip().lower()
    return stripped.startswith(b"<!doctype html") or stripped.startswith(b"<html") or stripped.startswith(b"<head")


def looks_like_audio(prefix: bytes, content_type: str, url: str) -> bool:
    lowered = content_type.lower()
    if any(lowered.startswith(prefix_type) or prefix_type in lowered for prefix_type in AUDIO_CONTENT_TYPES):
        if "json" not in lowered and "html" not in lowered:
            return True
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in AUDIO_EXTENSIONS:
        return True
    if prefix.startswith(b"ID3") or prefix.startswith(b"RIFF") or prefix.startswith(b"OggS") or prefix.startswith(b"fLaC"):
        return True
    if len(prefix) >= 2 and prefix[0] == 0xFF and prefix[1] in (0xFB, 0xF3, 0xF2, 0xFA, 0xF9):
        return True
    return False


def read_prefix(path: Path, size: int = 256) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)


def classify_error(exc: BaseException) -> Exception:
    try:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
    except ImportError:
        APIConnectionError = APITimeoutError = InternalServerError = RateLimitError = tuple()  # type: ignore[misc, assignment]

    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return RetryableError(str(exc))
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429 or status >= 500:
            return RetryableError(f"HTTP {status}: {exc}")
        if status in (401, 403):
            return FatalRowError(
                f"HTTP {status} downloading recording. The link may have expired, "
                "require a login, or be a CallRail API URL that needs an API key."
            )
        if status == 404:
            return FatalRowError("Recording URL returned 404 Not Found.")
        return FatalRowError(f"HTTP {status} downloading recording.")
    if RateLimitError and isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)):
        return RetryableError(str(exc))
    return exc


def call_with_retries(operation: str, fn, *, attempts: int = RETRY_ATTEMPTS):
    delay = RETRY_BASE_SECONDS
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except FatalRowError:
            raise
        except Exception as exc:  # noqa: BLE001 - classify then decide
            classified = classify_error(exc)
            if isinstance(classified, FatalRowError):
                raise classified from exc
            if not isinstance(classified, RetryableError):
                raise
            last_error = classified
            if attempt == attempts:
                break
            print(f"  {operation} failed (attempt {attempt}/{attempts}): {classified}. Retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay = min(delay * 2, RETRY_MAX_SECONDS)
    raise RetryableError(f"{operation} failed after {attempts} attempts: {last_error}")


def download_to_file(client: httpx.Client, url: str, destination: Path) -> tuple[str, int]:
    def _download() -> tuple[str, int]:
        with client.stream("GET", url) as response:
            if response.status_code == 429 or response.status_code >= 500:
                raise RetryableError(f"HTTP {response.status_code} from {url}")
            if response.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
            content_type = response.headers.get("content-type", "")
            content_length = response.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > MAX_AUDIO_BYTES:
                        raise FatalRowError(
                            f"Recording is {int(content_length)} bytes, over the {MAX_AUDIO_BYTES} byte OpenAI limit."
                        )
                except ValueError:
                    pass
            written = 0
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes():
                    written += len(chunk)
                    if written > MAX_AUDIO_BYTES:
                        raise FatalRowError(
                            f"Recording exceeded the {MAX_AUDIO_BYTES} byte OpenAI upload limit."
                        )
                    handle.write(chunk)
        return content_type, written

    return call_with_retries(f"Download {url}", _download)


def parse_json_url(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        candidate = payload.get("url")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def fetch_audio(client: httpx.Client, url: str, work_dir: Path) -> Path:
    current = url
    tried_redirect = False
    for hop in range(MAX_DOWNLOAD_HOPS):
        destination = work_dir / f"download-{hop}.bin"
        content_type, size = download_to_file(client, current, destination)
        if size == 0:
            raise FatalRowError(f"Download from {current} was empty.")
        prefix = read_prefix(destination)

        if looks_like_json(prefix) or "json" in content_type.lower():
            next_url = parse_json_url(destination)
            if not next_url:
                raise FatalRowError(
                    "URL returned JSON but no audio `url` field. "
                    "If this is a CallRail `/recording.json` API link, it needs a CallRail API key."
                )
            current = next_url
            continue

        if looks_like_html(prefix) or "html" in content_type.lower():
            redirect = recording_redirect_url(current)
            if redirect and not tried_redirect:
                print(f"  Got an HTML player page; trying {redirect}")
                current = redirect
                tried_redirect = True
                continue
            raise FatalRowError(
                "URL returned a web page instead of audio. "
                "Need a direct recording/download URL, not a CallRail dashboard link."
            )

        if looks_like_audio(prefix, content_type, current) or size > 1024:
            audio_path = destination.with_suffix(".mp3")
            destination.replace(audio_path)
            return audio_path

        raise FatalRowError(
            f"URL did not look like audio (content-type={content_type!r}, {size} bytes)."
        )

    raise FatalRowError(f"Too many redirects while fetching audio from {url}.")


def result_text(result: Any) -> str:
    text = getattr(result, "text", None)
    if text is None and isinstance(result, dict):
        text = result.get("text")
    return str(text or "").strip()


def word_tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text or "")]


def wording_matches_raw(raw: str, labeled_parts: list[str]) -> bool:
    raw_tokens = word_tokens(raw)
    labeled_tokens = word_tokens(" ".join(labeled_parts))
    if not raw_tokens:
        return not labeled_tokens
    if raw_tokens == labeled_tokens:
        return True
    removed = len(raw_tokens) - len(labeled_tokens)
    if removed < MIN_DUPLICATE_BLOCK_WORDS:
        return False
    n = len(raw_tokens)
    length = removed
    for start in range(0, n - 2 * length + 1):
        mid = start + length
        end = mid + length
        if raw_tokens[start:mid] != raw_tokens[mid:end]:
            continue
        stripped = raw_tokens[:mid] + raw_tokens[end:]
        if stripped == labeled_tokens:
            return True
    return False


def format_labeled_turns(turns: list[dict[str, str]]) -> str:
    lines = []
    for turn in turns:
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        speaker = (turn.get("speaker") or "UNKNOWN").strip().upper()
        if speaker not in ALLOWED_SPEAKERS:
            speaker = "UNKNOWN"
        lines.append(f"{speaker}: {text}")
    return "\n\n".join(lines)


def fallback_unlabeled(raw: str) -> FormattedTranscript:
    return FormattedTranscript(
        text=f"UNKNOWN: {raw.strip()}",
        speaker_uncertain=True,
        raw_transcript=raw,
    )


def parse_speaker_payload(content: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    turns = payload.get("turns")
    if not isinstance(turns, list) or not turns:
        return None
    cleaned = []
    for turn in turns:
        if not isinstance(turn, dict):
            return None
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        speaker = str(turn.get("speaker") or "UNKNOWN").strip().upper()
        cleaned.append({"speaker": speaker, "text": text})
    if not cleaned:
        return None
    payload["turns"] = cleaned
    payload["uncertain"] = bool(payload.get("uncertain"))
    return payload


def labeled_from_payload(raw: str, payload: dict[str, Any]) -> FormattedTranscript | None:
    turns = payload["turns"]
    if not wording_matches_raw(raw, [turn["text"] for turn in turns]):
        return None
    text = format_labeled_turns(turns)
    if not text:
        return None
    uncertain = bool(payload.get("uncertain")) or any(turn["speaker"] == "UNKNOWN" for turn in turns)
    if payload.get("call_type") == "voicemail" and all(turn["speaker"] in {"VOICEMAIL", "UNKNOWN"} for turn in turns):
        uncertain = bool(payload.get("uncertain"))
    return FormattedTranscript(text=text, speaker_uncertain=uncertain, raw_transcript=raw)


def request_speaker_labels(
    raw: str,
    caller_name: str,
    speaker_model: str,
    extra_user: str | None = None,
) -> dict[str, Any] | None:
    from openai import OpenAI

    client = OpenAI()
    user_parts = [
        "Assign speakers to this transcript. Copy the wording exactly.",
        f"Caller name from CallRail metadata (may be Unknown): {display_or_unknown(caller_name)}",
        "",
        "TRANSCRIPT:",
        raw,
    ]
    if extra_user:
        user_parts.extend(["", extra_user])

    def _label() -> dict[str, Any] | None:
        response = client.chat.completions.create(
            model=speaker_model,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "speaker_labeled_transcript",
                    "strict": True,
                    "schema": SPEAKER_JSON_SCHEMA,
                },
            },
            messages=[
                {"role": "system", "content": SPEAKER_SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            return None
        return parse_speaker_payload(content)

    return call_with_retries("Speaker labeling", _label)


def label_speakers(raw: str, caller_name: str, speaker_model: str) -> FormattedTranscript:
    payload = request_speaker_labels(raw, caller_name, speaker_model)
    formatted = labeled_from_payload(raw, payload) if payload else None
    if formatted:
        return formatted
    print("  Speaker labels changed the original words; retrying once...", flush=True)
    payload = request_speaker_labels(raw, caller_name, speaker_model, extra_user=SPEAKER_RETRY_PROMPT)
    formatted = labeled_from_payload(raw, payload) if payload else None
    if formatted:
        return formatted
    print("  Keeping the original transcript unlabeled after wording check failed.", flush=True)
    return fallback_unlabeled(raw)


def transcribe_file(audio_path: Path, model: str, language: str | None) -> str:
    from openai import OpenAI

    client = OpenAI()

    def _transcribe() -> str:
        with audio_path.open("rb") as handle:
            kwargs: dict[str, Any] = {"model": model, "file": handle}
            if language:
                kwargs["language"] = language
            result = client.audio.transcriptions.create(**kwargs)
        text = result_text(result)
        if not text:
            raise FatalRowError("OpenAI returned an empty transcript.")
        return text

    return call_with_retries("OpenAI transcription", _transcribe)


def should_process(row: CallRow, latest: dict[str, ProgressRecord], skip_failed: bool) -> bool:
    existing = latest.get(row.call_id)
    if existing is None:
        return True
    if existing.status == "success":
        return False
    if existing.status == "failed" and skip_failed:
        return False
    return True


def process_row(
    row: CallRow,
    client: httpx.Client,
    output_dir: Path,
    keep_audio: bool,
    model: str,
    language: str | None,
    speaker_model: str,
) -> ProgressRecord:
    if not row.recording_url:
        raise FatalRowError("Missing recording URL.")

    with tempfile.TemporaryDirectory(prefix="callrail-") as tmp:
        audio_path = fetch_audio(client, row.recording_url, Path(tmp))
        raw = transcribe_file(audio_path, model=model, language=language)
        formatted = label_speakers(raw, caller_name=row.caller_name, speaker_model=speaker_model)
        saved_audio = ""
        if keep_audio:
            audio_dir = output_dir / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            dest = audio_dir / f"{row.call_id}.mp3"
            dest.write_bytes(audio_path.read_bytes())
            saved_audio = str(dest)

    return ProgressRecord(
        id=row.call_id,
        url=row.recording_url,
        status="success",
        caller_name=row.caller_name,
        phone=row.phone,
        date=row.date,
        transcript=formatted.text,
        processed_at=iso_now(),
        audio_path=saved_audio,
        speaker_uncertain=formatted.speaker_uncertain,
        raw_transcript=formatted.raw_transcript or raw,
        tracking_id=row.tracking_id,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CallRail recordings from a CSV and transcribe them with OpenAI.",
    )
    parser.add_argument("csv_path", type=Path, help="CSV file with recording URLs and optional call metadata")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for transcripts.md, progress.jsonl, and failures.csv (default: ./output)",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep downloaded MP3s under output/audio/ instead of deleting them after transcription",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Do not retry rows that previously failed; only process new rows",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        metavar="N",
        help="Process at most N remaining calls this run (useful for a quick test)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    csv_path = args.csv_path.expanduser().resolve()
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    rows = load_calls(csv_path)
    print(f"Loaded {len(rows)} call(s) from {csv_path}", flush=True)
    if not rows:
        print("CSV has no data rows.")
        return 1

    output_dir = args.output_dir.expanduser().resolve()
    latest = load_progress(progress_path(output_dir))
    pending = [row for row in rows if should_process(row, latest, skip_failed=args.skip_failed)]
    already_done = len(rows) - len(pending)
    if already_done:
        print(f"Skipping {already_done} already completed call(s).", flush=True)

    if args.limit is not None:
        if args.limit < 1:
            print("Limit must be at least 1.", file=sys.stderr)
            return 1
        if len(pending) > args.limit:
            print(
                f"Limiting this run to {args.limit} of {len(pending)} remaining call(s).",
                flush=True,
            )
            pending = pending[: args.limit]

    if not pending:
        rebuild_outputs(output_dir, rows, latest)
        print(f"Nothing left to transcribe. Wrote {transcripts_path(output_dir)}")
        return 0

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print(
            "Missing OPENAI_API_KEY. Copy .env.example to .env and paste your OpenAI key, then re-run.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    model = os.getenv("OPENAI_TRANSCRIBE_MODEL", DEFAULT_TRANSCRIBE_MODEL).strip() or DEFAULT_TRANSCRIBE_MODEL
    speaker_model = os.getenv("OPENAI_SPEAKER_MODEL", DEFAULT_SPEAKER_MODEL).strip() or DEFAULT_SPEAKER_MODEL
    language = os.getenv("OPENAI_TRANSCRIBE_LANGUAGE", "").strip() or None
    print(f"Transcribing {len(pending)} call(s) with {model}; speaker labels with {speaker_model}", flush=True)

    http_client = httpx.Client(
        follow_redirects=True,
        timeout=HTTP_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    succeeded = 0
    failed = 0
    try:
        for index, row in enumerate(pending, start=1):
            print(f"[{index}/{len(pending)}] {row.call_id} — {display_or_unknown(row.caller_name)}", flush=True)
            try:
                record = process_row(
                    row,
                    client=http_client,
                    output_dir=output_dir,
                    keep_audio=args.keep_audio,
                    model=model,
                    language=language,
                    speaker_model=speaker_model,
                )
                succeeded += 1
                print(f"  transcribed ({len(record.transcript)} characters)", flush=True)
            except KeyboardInterrupt:
                raise
            except (FatalRowError, RetryableError) as exc:
                failed += 1
                record = ProgressRecord(
                    id=row.call_id,
                    url=row.recording_url,
                    status="failed",
                    caller_name=row.caller_name,
                    phone=row.phone,
                    date=row.date,
                    error=str(exc),
                    processed_at=iso_now(),
                )
                print(f"  FAILED: {exc}")
            except Exception as exc:  # noqa: BLE001 - keep the batch going
                failed += 1
                record = ProgressRecord(
                    id=row.call_id,
                    url=row.recording_url,
                    status="failed",
                    caller_name=row.caller_name,
                    phone=row.phone,
                    date=row.date,
                    error=f"Unexpected error: {exc}",
                    processed_at=iso_now(),
                )
                print(f"  FAILED: {exc}")

            append_progress(progress_path(output_dir), record)
            latest[record.id] = record
            rebuild_outputs(output_dir, rows, latest)
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; re-run the same command to resume.")
        rebuild_outputs(output_dir, rows, latest)
        return 130
    finally:
        http_client.close()

    print(
        f"Done. {succeeded} transcribed, {failed} failed, "
        f"{already_done} skipped. Transcripts: {transcripts_path(output_dir)}"
    )
    if failed:
        print(f"Failures listed in {failures_path(output_dir)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
