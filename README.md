# CallRail Batch Transcriber

A local Python utility that turns a CSV of CallRail call-recording URLs into one searchable markdown file of transcripts. It downloads each recording, transcribes it with OpenAI speech-to-text, then sends **that text** to a second model only to label OPERATOR vs CALLER. The original transcript wording is the source of truth.

This is a one-purpose script you run on your computer. There is no website, database, or installer.

## What CallRail is

CallRail is a call-tracking service. Businesses give out tracking phone numbers; CallRail logs who called, and (if recording is enabled) stores an audio file of the call. Each recording can be reached by a URL. This tool does **not** log into CallRail. You supply a spreadsheet of those URLs (and any caller name / phone / date / call ID columns), and it produces text.

CallRail’s normal Call Log Excel export often includes name, phone, and date, but **may not include a downloadable recording URL**. The CSV you feed this tool must have a column that is actually a recording or audio URL.

Typical CallRail URL shapes:

- Player page: `https://app.callrail.com/calls/{id}/recording?access_key=...`
- Direct-ish redirect: `https://app.callrail.com/calls/{id}/recording/redirect?access_key=...`
- A plain MP3 URL

If the spreadsheet only has call IDs and no URLs, this version cannot fetch the audio (that would need a CallRail API key). Get a CSV that includes recording URLs.

## Setup

You need Python 3.10+ and an OpenAI API key (add the key when you have it).

```bash
cd "Callrail Transcriber"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:

```
OPENAI_API_KEY=sk-...
```

Optional:

```
OPENAI_TRANSCRIBE_MODEL=gpt-4o-mini-transcribe
OPENAI_SPEAKER_MODEL=gpt-4o-mini
OPENAI_TRANSCRIBE_LANGUAGE=en
```

Never put the key in the source code. `.env` is gitignored.

## CSV columns

Headers are matched case-insensitively. Spaces and underscores are treated the same (`Call ID` = `call_id`).

| Purpose | Accepted names | Required? |
|---|---|---|
| Recording URL | `recording_url`, `recording_player`, `recording`, `url` | Yes |
| Call ID | `call_id`, `id`, `call id` | No (hash of the URL is used if missing) |
| Caller name | `name`, `caller_name`, `customer_name` | No |
| Phone | `phone number`, `phone`, `customer_phone_number`, `caller_phone` | No |
| Date | `start_time`, `date`, `call_date` | No |
| Tracking ID | `tracking number`, `tracking_id`, `tracking id` | No |

See `sample_calls.csv` for the expected shape. Those rows use fake example URLs so you can check CSV parsing; they will not download real audio. Other columns are ignored.

Save Excel exports as CSV (UTF-8) before running.

## Run

```bash
source .venv/bin/activate
python transcribe.py sample_calls.csv
python transcribe.py /path/to/client_calls.csv --output-dir ./output
python transcribe.py /path/to/client_calls.csv -l 3
```

Flags:

- `-l N`, `--limit N` — process at most N remaining calls this run (for a quick test)
- `--output-dir DIR` — where files are written (default: `./output`)
- `--keep-audio` — keep MP3s under `output/audio/` instead of deleting them after transcription
- `--skip-failed` — do not retry rows that already failed; only process new rows

`--help` works without an API key. If you run against a CSV without a key, the script will load and count the rows, then tell you to fill in `.env`.

## Output

| File | What it is |
|---|---|
| `output/transcripts.md` | One markdown file of every successful transcript, in CSV order |
| `output/progress.jsonl` | Resume log (one JSON object per attempt) |
| `output/failures.csv` | Current failures, if any |

Each call in `transcripts.md` looks like this, so you can split or filter the whole file later:

```markdown
<!-- CALL_START id=CAL111222333 -->
## CAL111222333 — Jane Doe — 2026-01-15

- **Caller:** Wile E. Coyote
- **Phone:** +1-555-123-4567
- **Date:** 2026-01-15 14:32
- **Tracking ID:** 877-223-3535
- **Recording URL:** https://...

OPERATOR: ACME Inc, this is Sam.

CALLER: Hi Sam, I've got a question about roadrunner traps.

OPERATOR: We've got that.
<!-- CALL_END id=CAL111222333 -->
```

Speaker labels are added in a **second, text-only** API call. It must not rewrite the speech-to-text. If the labeling model changes any words, the script rejects that output and keeps the original transcript as `UNKNOWN:` with a review flag.

- Leftover / one-person message: `VOICEMAIL:`
- Uncertain roles: `UNKNOWN:` plus `<!-- SPEAKER_UNCERTAIN -->`
- The raw STT text is stored in `progress.jsonl` as `raw_transcript` and as an HTML comment under each call so you can recover it

Missing metadata is shown as `Unknown`.

Already-completed calls from an earlier run stay skipped, including any diarize-era results. To redo those with the current pipeline, delete the matching lines from `output/progress.jsonl` (or remove `output/`) and run again (`-l 3` is fine for a test).

## Resume

Progress is saved after every call. If the run is interrupted (Ctrl+C, laptop sleep, network drop), run the **same command** again. Completed call IDs are skipped. Failed rows are retried unless you pass `--skip-failed`.

`transcripts.md` is rebuilt from `progress.jsonl` after each call, so it stays consistent.

## Limits and cost

- OpenAI rejects uploads over 25 MB. This tool skips anything over 24 MB and records it as a failure. Typical phone MP3s are much smaller.
- Calls are processed one at a time to stay under rate limits. Retries with backoff happen on 429 / 5xx / network errors.
- Each call is two API requests: speech-to-text, then a cheap text-only speaker-label pass. A 3-call test should take on the order of ~20 seconds plus a few seconds of labeling, not minutes.
- Ballpark STT cost for ~2,000 calls at ~3 minutes each with `gpt-4o-mini-transcribe` is on the order of **$20**. Speaker labeling is usually a few dollars more. That is not a quote.

The default model for transcription is `gpt-4o-mini-transcribe`, which produced the best results in testing. Do not set `OPENAI_TRANSCRIBE_MODEL` to `gpt-4o-transcribe-diarize`. It is slower and it rewrites wording. For long audio, `whisper-1` is the least likely to reject the file for length, and will still returns one block of text before the labeling step.

## How a recording is processed

For each URL the script:

1. Downloads the audio (following HTTP redirects, JSON wrappers, and CallRail player pages).
2. Transcribes the MP3 with `gpt-4o-mini-transcribe` (or whatever `OPENAI_TRANSCRIBE_MODEL` is set to). That text is kept unchanged.
3. Sends that text to `gpt-4o-mini` to assign `OPERATOR` / `CALLER` (or `VOICEMAIL` / `UNKNOWN`).
4. Checks that the labeled turns still contain the original words. If they do not, it retries once, then falls back to the raw transcript.

Audio is deleted after a successful transcription unless you pass `--keep-audio`.
