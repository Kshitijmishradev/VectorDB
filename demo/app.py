"""Local interactive text/image -> CIFAR-100 similarity search."""
import html
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from demo.clip_tools import image_to_data_url, load_manifest, load_ml, normalize_rows
from vectordb.ivf_pq import IVFPQIndex


ARTIFACT_DIR = Path(os.environ.get("CLIP_DEMO_DIR", "demo/artifacts"))
DATA_ROOT = os.environ.get("CIFAR100_DATA_ROOT", "data")
STATE = {}


@asynccontextmanager
async def lifespan(_app):
    if not (ARTIFACT_DIR / "manifest.json").exists():
        raise RuntimeError(
            "demo index missing; run `python demo/build_clip_demo.py build` first")
    torch, clip, CIFAR100, model, preprocess, device = load_ml(DATA_ROOT)
    STATE.update(
        torch=torch, clip=clip, model=model, preprocess=preprocess, device=device,
        dataset=CIFAR100(DATA_ROOT, train=True, download=False),
        index=IVFPQIndex.load(str(ARTIFACT_DIR / "index")),
        manifest=load_manifest(ARTIFACT_DIR))
    yield
    STATE["index"].close()
    STATE.clear()


app = FastAPI(title="VectorDB CLIP image search", lifespan=lifespan)


def _page(body=""):
    return f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>VectorDB · CLIP search</title><style>
body{{font:16px system-ui;background:#0b1020;color:#edf2ff;max-width:1100px;margin:40px auto;padding:0 20px}}
h1{{font-size:2.5rem;margin-bottom:8px}} .muted{{color:#a8b3cf}} form{{display:flex;gap:10px;margin:18px 0}}
input,button{{padding:12px;border-radius:10px;border:1px solid #35405e;background:#151d33;color:#fff}}
input[type=text]{{flex:1}} button{{background:#695cff;font-weight:700;cursor:pointer}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:16px;margin-top:24px}}
.card{{background:#151d33;border:1px solid #27314b;border-radius:14px;padding:12px}}
.card img{{width:100%;aspect-ratio:1;object-fit:cover;image-rendering:auto;border-radius:9px}}
.metric{{font-variant-numeric:tabular-nums;color:#a8b3cf;font-size:.9rem}}
</style></head><body><h1>VectorDB × CLIP</h1>
<p class='muted'>Search 50,000 CIFAR-100 images using text or an uploaded image.</p>
<form action='/search/text' method='post'><input type='text' name='prompt' placeholder='a red pickup truck' required><button>Search text</button></form>
<form action='/search/image' method='post' enctype='multipart/form-data'><input type='file' name='image' accept='image/*' required><button>Search image</button></form>
{body}</body></html>"""


def _render_results(ids, distances, latency_ms, title):
    cards = []
    dataset = STATE["dataset"]
    for vector_id, distance in zip(ids, distances):
        image, label = dataset[int(vector_id)]
        cards.append(
            "<div class='card'><img src='{}'><strong>{}</strong>"
            "<div class='metric'>id {} · distance {:.4f}</div></div>".format(
                image_to_data_url(image), html.escape(dataset.classes[label]),
                vector_id, distance))
    return HTMLResponse(_page(
        f"<h2>{html.escape(title)}</h2><div class='metric'>{latency_ms:.2f} ms · "
        f"residual IVF+PQ + exact reranking</div><div class='grid'>{''.join(cards)}</div>"))


def _search(vector, title):
    manifest = STATE["manifest"]
    start = time.perf_counter()
    ids, distances = STATE["index"].search(
        vector, k=10, max_candidates=manifest["candidate_budget"],
        rerank=manifest["rerank"])
    latency_ms = (time.perf_counter() - start) * 1000
    return _render_results(ids, distances, latency_ms, title)


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(_page())


@app.post("/search/text", response_class=HTMLResponse)
def search_text(prompt: str = Form(...)):
    torch, clip = STATE["torch"], STATE["clip"]
    with torch.no_grad():
        tokens = clip.tokenize([prompt]).to(STATE["device"])
        vector = STATE["model"].encode_text(tokens).float().cpu().numpy()
    return _search(normalize_rows(vector)[0], f'Text: “{prompt}”')


@app.post("/search/image", response_class=HTMLResponse)
async def search_image(image: UploadFile = File(...)):
    try:
        pil_image = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid image") from exc
    torch = STATE["torch"]
    with torch.no_grad():
        tensor = STATE["preprocess"](pil_image).unsqueeze(0).to(STATE["device"])
        vector = STATE["model"].encode_image(tensor).float().cpu().numpy()
    return _search(normalize_rows(vector)[0], f"Image: {image.filename or 'upload'}")
