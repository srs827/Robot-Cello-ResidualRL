"""Download the two public cello datasets: Good-sounds (cello part only) and the Iowa cello arco notes."""
import os
import re
import struct
import sys
import threading
import time
import urllib.parse
import zlib
from concurrent.futures import ThreadPoolExecutor

import requests

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from common import RAW

GS_BASE = "https://zenodo.org/api/records/4588740/files/"
GS_ZIP = GS_BASE + "good-sounds.zip/content"
GS_DIR = os.path.join(RAW, "good_sounds")
GS_FOLDERS = ["cello_margarita_attack", "cello_margarita_dynamics_stability", "cello_margarita_open_strings",
              "cello_margarita_pitch_stability", "cello_margarita_reference", "cello_margarita_timbre_richness",
              "cello_margarita_timbre_stability", "cello_nico_improvement_recordings"]
IOWA_PAGE = "https://theremin.music.uiowa.edu/MIScello.html"
IOWA_BASE = "https://theremin.music.uiowa.edu/"
IOWA_DIR = os.path.join(RAW, "iowa_cello")
CHUNK = 48 * 1024 * 1024


def _get_range(start, end):
    """Download one byte range of the Good-sounds zip, retrying on rate limits."""
    for attempt in range(8):
        try:
            r = requests.get(GS_ZIP, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=300)
            if r.status_code == 429:
                time.sleep(min(int(r.headers.get("Retry-After", 20)), 120))
                continue
            r.raise_for_status()
            return r.content
        except Exception:
            time.sleep(5 + 5 * attempt)
    raise RuntimeError("download failed")


def good_sounds(workers=6):
    """Fetch the Good-sounds metadata json files and the cello audio folders (~1.9 GB)."""
    from remotezip import RemoteZip
    os.makedirs(GS_DIR, exist_ok=True)
    for n in ["sounds.json", "takes.json", "packs.json", "ratings.json"]:
        p = os.path.join(GS_DIR, n)
        if not os.path.exists(p):
            r = requests.get(GS_BASE + n + "/content", timeout=120)
            r.raise_for_status()
            open(p, "wb").write(r.content)
    out = os.path.join(GS_DIR, "audio") + "/"
    infos = sorted(RemoteZip(GS_ZIP).infolist(), key=lambda i: i.header_offset)
    ends = {a.filename: b.header_offset for a, b in zip(infos, infos[1:])}
    ends[infos[-1].filename] = infos[-1].header_offset + infos[-1].compress_size + 2000
    prefixes = tuple(f"good-sounds/sound_files/{p}" for p in GS_FOLDERS)
    sel = [i for i in infos if i.filename.startswith(prefixes) and not i.is_dir()
           and "/._" not in i.filename and not i.filename.endswith("/Icon")]
    chunks, cur, size = [], [], 0
    for i in sel:
        dst = out + i.filename[len("good-sounds/"):]
        if os.path.exists(dst) and os.path.getsize(dst) == i.file_size:
            continue
        span = ends[i.filename] - i.header_offset
        if cur and (size + span > CHUNK or i.header_offset - ends[cur[-1].filename] > 4_000_000):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += span
    if cur:
        chunks.append(cur)
    print(f"good-sounds: {len(sel)} files, {len(chunks)} chunks to download", flush=True)
    lock, done = threading.Lock(), [0]

    def fetch_chunk(entries):
        """Download one chunk of consecutive zip entries and unpack them."""
        start = entries[0].header_offset
        blob = _get_range(start, ends[entries[-1].filename])
        for i in entries:
            o = i.header_offset - start
            n, m = struct.unpack("<HH", blob[o + 26:o + 30])
            data = blob[o + 30 + n + m: o + 30 + n + m + i.compress_size]
            if i.compress_type == 8:
                data = zlib.decompress(data, -15)
            dst = out + i.filename[len("good-sounds/"):]
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst + ".part", "wb").write(data)
            os.replace(dst + ".part", dst)
        with lock:
            done[0] += 1
            print(f"  chunk {done[0]}/{len(chunks)}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(fetch_chunk, chunks))


def iowa():
    """Fetch the Iowa cello arco .aiff files (~200 MB)."""
    os.makedirs(IOWA_DIR, exist_ok=True)
    page = requests.get(IOWA_PAGE, timeout=60).text
    links = sorted(set(re.findall(r'href="([^"]+\.aiff?)"', page, flags=re.I)))
    arco = [l for l in links if "arco" in l.lower()]
    print(f"iowa: {len(arco)} arco files", flush=True)
    for l in arco:
        dst = os.path.join(IOWA_DIR, os.path.basename(l))
        if os.path.exists(dst) and os.path.getsize(dst) > 1e5:
            continue
        for attempt in range(3):
            try:
                r = requests.get(IOWA_BASE + urllib.parse.quote(l), timeout=180, stream=True)
                r.raise_for_status()
                with open(dst + ".part", "wb") as f:
                    for ch in r.iter_content(1 << 20):
                        f.write(ch)
                os.replace(dst + ".part", dst)
                break
            except Exception:
                time.sleep(3)


if __name__ == "__main__":
    iowa()
    good_sounds()
    print("done", flush=True)
