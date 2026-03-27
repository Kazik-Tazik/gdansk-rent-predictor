import json
import base64
import io
import time
import signal
import sys
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

# =========================
# CONFIG
# =========================
start_time = time.perf_counter()

IMAGES_ROOT = Path(r"C:\Users\kazim\PycharmProjects\notebook\otodom_images_best")
OUT_IMAGE_CSV = Path("image_scores.csv")
OUT_LISTING_CSV = Path("listing_image_scores.csv")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "openbmb/minicpm-v4"

REQUEST_TIMEOUT = 120
KEEP_ALIVE = "240m"
TEMPERATURE = 0
MAX_IMAGE_SIZE = 1024
JPEG_QUALITY = 85
MIN_USABLE_CONFIDENCE = 0.45

TEST_IMAGE_LIMIT = None

PROMPT = """
You are evaluating whether a real-estate photo is useful for estimating the overall quality of an apartment.

Return valid JSON only with these fields:
{
  "is_interior": true,
  "is_relevant": true,
  "is_overview": true,
  "junk_type": "string or null",
  "room_type": "string or null",
  "modernness": 0.0,
  "condition": 0.0,
  "brightness": 0.0,
  "furnishing_quality": 0.0,
  "spacious_feel": 0.0,
  "overall_quality": 0.0,
  "kitchen_visible": true,
  "bathroom_visible": false,
  "kitchen_quality": 0.0,
  "bathroom_quality": null,
  "confidence": 0.0,
  "short_reason": "string"
}

Rubric:
- 0.0 = very old, damaged, ugly, badly maintained, unattractive
- 0.5 = average / normal rental standard
- 1.0 = very modern, renovated, clean, premium, attractive

Rules:
- First decide if the image is useful for grading the apartment as a whole.
- is_relevant = false for exterior, building, map, floorplan, logo, person, agent photo, close-up object, blurry shot, or unusable image.
- is_interior = true only for interior apartment photos.
- is_overview = true only if the image shows a meaningful part of a room, not a tiny close-up.
- overall_quality should estimate the visible apartment standard from 0.0 to 1.0.
- If kitchen or bathroom is not visible, set the corresponding *_visible field to false and quality to null.
- If image is irrelevant, lower confidence and keep quality fields conservative.
- No markdown, JSON only.
"""

FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_interior": {"type": "boolean"},
        "is_relevant": {"type": "boolean"},
        "is_overview": {"type": "boolean"},
        "junk_type": {"type": ["string", "null"]},
        "room_type": {"type": ["string", "null"]},
        "modernness": {"type": "number"},
        "condition": {"type": "number"},
        "brightness": {"type": "number"},
        "furnishing_quality": {"type": "number"},
        "spacious_feel": {"type": "number"},
        "overall_quality": {"type": "number"},
        "kitchen_visible": {"type": "boolean"},
        "bathroom_visible": {"type": "boolean"},
        "kitchen_quality": {"type": ["number", "null"]},
        "bathroom_quality": {"type": ["number", "null"]},
        "confidence": {"type": "number"},
        "short_reason": {"type": "string"}
    },
    "required": [
        "is_interior",
        "is_relevant",
        "is_overview",
        "junk_type",
        "room_type",
        "modernness",
        "condition",
        "brightness",
        "furnishing_quality",
        "spacious_feel",
        "overall_quality",
        "kitchen_visible",
        "bathroom_visible",
        "kitchen_quality",
        "bathroom_quality",
        "confidence",
        "short_reason"
    ]
}

session = requests.Session()
stop_requested = False

def format_seconds(seconds):
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def encode_resized_image(path, max_size=1024, quality=85):
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def get_image_dimensions(path):
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None, None

def safe_float(x):
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def safe_bool(x):
    if x is None:
        return False
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        try:
            if pd.isna(x):
                return False
        except Exception:
            pass
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n", "null", "none", ""}:
            return False
    return False

def clamp01(x):
    if x is None:
        return None
    return max(0.0, min(1.0, x))

def parse_json_text(text):
    text = (text or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            if lines.strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


def ask_model(image_path):
    b64 = encode_resized_image(image_path, max_size=MAX_IMAGE_SIZE, quality=JPEG_QUALITY)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [b64]
            }
        ],
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "format": FORMAT_SCHEMA,
        "options": {
            "temperature": TEMPERATURE,
            "num_thread": 16
        }
    }

    response = session.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    result = response.json()
    text = result["message"]["content"]
    return parse_json_text(text)


def compute_photo_score(data):
    is_interior = safe_bool(data.get("is_interior"))
    is_relevant = safe_bool(data.get("is_relevant"))
    is_overview = safe_bool(data.get("is_overview"))
    kitchen_visible = safe_bool(data.get("kitchen_visible"))
    bathroom_visible = safe_bool(data.get("bathroom_visible"))

    modernness = clamp01(safe_float(data.get("modernness")))
    condition = clamp01(safe_float(data.get("condition")))
    brightness = clamp01(safe_float(data.get("brightness")))
    furnishing_quality = clamp01(safe_float(data.get("furnishing_quality")))
    spacious_feel = clamp01(safe_float(data.get("spacious_feel")))
    overall_quality = clamp01(safe_float(data.get("overall_quality")))
    kitchen_quality = clamp01(safe_float(data.get("kitchen_quality")))
    bathroom_quality = clamp01(safe_float(data.get("bathroom_quality")))
    confidence = clamp01(safe_float(data.get("confidence")))

    required = [
        modernness, condition, brightness, furnishing_quality,
        spacious_feel, overall_quality, confidence
    ]
    if any(v is None for v in required):
        raise ValueError(f"Missing required numeric field: {data}")

    junk_penalty = 0.0
    if not is_relevant:
        junk_penalty += 0.45
    if not is_interior:
        junk_penalty += 0.25
    if not is_overview:
        junk_penalty += 0.15

    room_bonus = 0.0
    if kitchen_visible and kitchen_quality is not None:
        room_bonus += 0.08 * kitchen_quality
    if bathroom_visible and bathroom_quality is not None:
        room_bonus += 0.08 * bathroom_quality

    base_score = (
        0.30 * overall_quality +
        0.18 * condition +
        0.16 * modernness +
        0.10 * furnishing_quality +
        0.08 * brightness +
        0.10 * spacious_feel +
        room_bonus
    )

    photo_score_raw = clamp01(base_score - junk_penalty)
    photo_score = clamp01(photo_score_raw * (0.35 + 0.65 * confidence))

    usable_for_listing = bool(
        is_relevant and
        is_interior and
        is_overview and
        confidence >= MIN_USABLE_CONFIDENCE
    )

    return {
        "is_interior": is_interior,
        "is_relevant": is_relevant,
        "is_overview": is_overview,
        "usable_for_listing": usable_for_listing,
        "junk_type": data.get("junk_type"),
        "room_type": data.get("room_type"),
        "modernness": modernness,
        "condition": condition,
        "brightness": brightness,
        "furnishing_quality": furnishing_quality,
        "spacious_feel": spacious_feel,
        "overall_quality": overall_quality,
        "kitchen_visible": kitchen_visible,
        "bathroom_visible": bathroom_visible,
        "kitchen_quality": kitchen_quality,
        "bathroom_quality": bathroom_quality,
        "confidence": confidence,
        "short_reason": data.get("short_reason"),
        "photo_score_raw": photo_score_raw,
        "photo_score": photo_score,
    }

def load_existing_rows(out_path):
    if out_path.exists():
        df = pd.read_csv(out_path)
        return df.to_dict(orient="records")
    return []

def make_done_set(rows):
    done = set()
    for r in rows:
        listing_id = str(r.get("listing_id", ""))
        image_name = str(r.get("image_name", ""))
        if listing_id and image_name:
            done.add((listing_id, image_name))
    return done

def weighted_mean(values, weights):
    pairs = []
    for v, w in zip(values, weights):
        v = safe_float(v)
        w = safe_float(w)
        if v is None or w is None or w <= 0:
            continue
        pairs.append((v, w))
    if not pairs:
        return None
    num = sum(v * w for v, w in pairs)
    den = sum(w for _, w in pairs)
    return num / den if den > 0 else None


def weighted_mean_col(df, value_col, weight_col):
    if len(df) == 0:
        return None
    return weighted_mean(df[value_col].tolist(), df[weight_col].tolist())


def build_listing_scores(image_df):
    if len(image_df) == 0:
        return pd.DataFrame(columns=[
            "listing_id",
            "total_images_seen",
            "scored_images",
            "relevant_image_count",
            "overview_image_count",
            "usable_image_count",
            "has_image_score",
            "avg_confidence_all",
            "avg_confidence_usable",
            "image_score_mean",
            "image_score_weighted",
            "image_score_top2_weighted",
            "image_score_max",
            "best_overall_quality",
            "best_condition",
            "best_modernness",
            "best_furnishing_quality"
        ])

    df = image_df.copy()

    for col in [
        "is_relevant", "is_interior", "is_overview", "usable_for_listing",
        "kitchen_visible", "bathroom_visible"
    ]:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].apply(safe_bool)

    for col in [
        "photo_score", "overall_quality", "condition", "modernness",
        "furnishing_quality", "confidence", "kitchen_quality", "bathroom_quality"
    ]:
        if col not in df.columns:
            df[col] = None

    rows = []

    for listing_id, g_all in df.groupby("listing_id"):
        g_all = g_all.copy()

        scored = g_all[g_all["photo_score"].notna()].copy()
        relevant = scored[scored["is_relevant"]].copy()
        usable = scored[scored["usable_for_listing"]].copy()
        overview = scored[scored["is_overview"]].copy()

        base_row = {
            "listing_id": listing_id,
            "total_images_seen": len(g_all),
            "scored_images": len(scored),
            "relevant_image_count": len(relevant),
            "overview_image_count": len(overview),
            "usable_image_count": len(usable),
            "has_image_score": int(len(usable) > 0),
            "avg_confidence_all": scored["confidence"].dropna().mean() if len(scored) else None,
            "avg_confidence_usable": usable["confidence"].dropna().mean() if len(usable) else None,
            "image_score_mean": None,
            "image_score_weighted": None,
            "image_score_top2_weighted": None,
            "image_score_max": None,
            "best_overall_quality": None,
            "best_condition": None,
            "best_modernness": None,
            "best_furnishing_quality": None,
            "kitchen_quality_weighted": None,
            "bathroom_quality_weighted": None,
        }

        if len(usable) == 0:
            rows.append(base_row)
            continue

        usable = usable.copy()
        usable["weight"] = (
            0.15 +
            0.55 * usable["confidence"].clip(0, 1) +
            0.20 * usable["is_overview"].astype(float) +
            0.10 * usable["overall_quality"].fillna(0).clip(0, 1)
        )

        usable_sorted = usable.sort_values(
            ["photo_score", "confidence", "overall_quality"],
            ascending=False
        )
        top2 = usable_sorted.head(2)

        kitchen_df = usable[
            usable["kitchen_visible"] & usable["kitchen_quality"].notna()
        ].copy()

        bathroom_df = usable[
            usable["bathroom_visible"] & usable["bathroom_quality"].notna()
        ].copy()

        base_row.update({
            "image_score_mean": usable["photo_score"].mean(),
            "image_score_weighted": weighted_mean_col(usable, "photo_score", "weight"),
            "image_score_top2_weighted": weighted_mean_col(top2, "photo_score", "weight"),
            "image_score_max": usable["photo_score"].max(),
            "best_overall_quality": usable["overall_quality"].max(),
            "best_condition": usable["condition"].max(),
            "best_modernness": usable["modernness"].max(),
            "best_furnishing_quality": usable["furnishing_quality"].max(),
            "kitchen_quality_weighted": weighted_mean_col(kitchen_df, "kitchen_quality", "weight"),
            "bathroom_quality_weighted": weighted_mean_col(bathroom_df, "bathroom_quality", "weight"),
        })

        rows.append(base_row)

    return pd.DataFrame(rows)


def atomic_write_csv(df, path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def save_progress(rows, reason="progress"):
    image_df = pd.DataFrame(rows)
    listing_df = build_listing_scores(image_df)
    atomic_write_csv(image_df, OUT_IMAGE_CSV)
    atomic_write_csv(listing_df, OUT_LISTING_CSV)
    print(
        f"\n[SAVE] reason={reason} | images={len(image_df)} | listings={len(listing_df)}",
        flush=True
    )
    return image_df, listing_df


def test_ollama():
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly this JSON: {\"ok\": true}"
            }
        ],
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "format": {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"}
            },
            "required": ["ok"]
        },
        "options": {
            "temperature": 0
        }
    }

    r = session.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    content = r.json()["message"]["content"]
    parsed = parse_json_text(content)
    return parsed.get("ok") is True


def list_image_files(listing_dir):
    return sorted([
        p for p in listing_dir.iterdir()
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ])


def estimate_total_pending_images(listing_dirs, done):
    total = 0
    for listing_dir in listing_dirs:
        for img_path in list_image_files(listing_dir):
            if (listing_dir.name, img_path.name) not in done:
                total += 1
    return total


def request_stop(signum, frame):
    global stop_requested
    stop_requested = True
    try:
        signame = signal.Signals(signum).name
    except Exception:
        signame = str(signum)
    print(f"\nStop requested by signal: {signame}. Will save progress and exit cleanly.", flush=True)


# Register stop handlers
signal.signal(signal.SIGINT, request_stop)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, request_stop)


# =========================
# MAIN
# =========================
print("Checking Ollama...", flush=True)
try:
    ok = test_ollama()
    print(f"Ollama test OK: {ok}", flush=True)
except Exception as e:
    print(f"Failed to talk to Ollama: {e}", flush=True)
    raise SystemExit(1)

if not IMAGES_ROOT.exists():
    print(f"Images folder does not exist: {IMAGES_ROOT}", flush=True)
    raise SystemExit(1)

rows = load_existing_rows(OUT_IMAGE_CSV)
done = make_done_set(rows)

listing_dirs = sorted([p for p in IMAGES_ROOT.iterdir() if p.is_dir()])
print(f"Found listing folders: {len(listing_dirs)}", flush=True)

total_pending_images = estimate_total_pending_images(listing_dirs, done)
print(f"Pending images to score: {total_pending_images}", flush=True)

processed_images = 0
new_images_this_run = 0

try:
    for i, listing_dir in enumerate(listing_dirs, start=1):
        if stop_requested:
            break

        listing_elapsed = time.perf_counter() - start_time
        avg_listing_time = listing_elapsed / i if i else 0
        listing_eta = avg_listing_time * (len(listing_dirs) - i)

        print(
            f"\n[{i}/{len(listing_dirs)}] Listing: {listing_dir.name} "
            f"| elapsed={format_seconds(listing_elapsed)} "
            f"| listing_eta~={format_seconds(listing_eta)}",
            flush=True
        )

        image_files = list_image_files(listing_dir)

        if not image_files:
            print("  -> no images found", flush=True)
            continue

        for img_path in image_files:
            if stop_requested:
                break

            key = (listing_dir.name, img_path.name)

            if key in done:
                print(f"  -> skipping already scored: {img_path.name}", flush=True)
                continue

            if TEST_IMAGE_LIMIT is not None and processed_images >= TEST_IMAGE_LIMIT:
                print("\nReached TEST_IMAGE_LIMIT. Saving and stopping.", flush=True)
                stop_requested = True
                break

            elapsed = time.perf_counter() - start_time
            done_so_far = max(1, new_images_this_run)
            remaining = max(0, total_pending_images - new_images_this_run)
            eta = (elapsed / done_so_far) * remaining if new_images_this_run > 0 else 0

            w, h = get_image_dimensions(img_path)
            print(
                f"  -> scoring {img_path.name} ({w}x{h}) "
                f"| image {new_images_this_run + 1}/{total_pending_images} "
                f"| elapsed={format_seconds(elapsed)} "
                f"| eta~={format_seconds(eta)}",
                flush=True
            )

            try:
                data = ask_model(img_path)
                scored = compute_photo_score(data)

                row = {
                    "listing_id": listing_dir.name,
                    "image_name": img_path.name,
                    "image_path": str(img_path),
                    "image_width": w,
                    "image_height": h,
                    **scored
                }

                rows.append(row)
                done.add(key)
                processed_images += 1
                new_images_this_run += 1

                save_progress(rows, reason="checkpoint_ok")

                print(
                    "     OK -> "
                    f"usable={row['usable_for_listing']} | "
                    f"score={row['photo_score']:.3f} | "
                    f"conf={row['confidence']:.3f} | "
                    f"reason={row['short_reason']}",
                    flush=True
                )

            except Exception as e:
                row = {
                    "listing_id": listing_dir.name,
                    "image_name": img_path.name,
                    "image_path": str(img_path),
                    "image_width": w,
                    "image_height": h,
                    "is_interior": False,
                    "is_relevant": False,
                    "is_overview": False,
                    "usable_for_listing": False,
                    "junk_type": None,
                    "room_type": None,
                    "modernness": None,
                    "condition": None,
                    "brightness": None,
                    "furnishing_quality": None,
                    "spacious_feel": None,
                    "overall_quality": None,
                    "kitchen_visible": False,
                    "bathroom_visible": False,
                    "kitchen_quality": None,
                    "bathroom_quality": None,
                    "confidence": None,
                    "short_reason": f"ERROR: {e}",
                    "photo_score_raw": None,
                    "photo_score": None
                }

                rows.append(row)
                done.add(key)
                processed_images += 1
                new_images_this_run += 1

                save_progress(rows, reason="checkpoint_error")
                print(f"     FAILED -> {e}", flush=True)

            time.sleep(0.2)

        if stop_requested:
            break

except KeyboardInterrupt:
    stop_requested = True
    print("\nKeyboardInterrupt received. Saving progress before exit...", flush=True)

finally:
    try:
        image_df, listing_df = save_progress(rows, reason="final")
        total_elapsed = time.perf_counter() - start_time

        print(f"\n--- {format_seconds(total_elapsed)} ---", flush=True)
        print("\nDone.", flush=True)
        print(f"New images processed this run: {new_images_this_run}", flush=True)
        print(f"Total image rows saved: {len(image_df)}", flush=True)
        print(f"Total listing rows saved: {len(listing_df)}", flush=True)
        print(f"Saved: {OUT_IMAGE_CSV}", flush=True)
        print(f"Saved: {OUT_LISTING_CSV}", flush=True)
    except Exception as save_error:
        print(f"\nFATAL: could not save final progress: {save_error}", flush=True)
        sys.exit(1)
