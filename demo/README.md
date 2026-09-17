# CIFAR-100 + CLIP demo

This optional demo embeds CIFAR-100 with OpenAI CLIP ViT-B/32, builds a
residual IVF+PQ index, validates ANN recall against brute-force CLIP search,
and serves text-to-image and image-to-image retrieval through FastAPI.

```bash
pip install -r requirements-demo.txt

# Downloads CIFAR-100 and CLIP weights, then builds the 50k-vector index.
python3 demo/build_clip_demo.py build --routing-backend mlx

# Measures 500 test queries against brute-force CLIP ground truth.
python3 demo/build_clip_demo.py evaluate --eval-queries 500

uvicorn demo.app:app --reload
```

Open <http://127.0.0.1:8000>. Set `CLIP_DEMO_DIR` or
`CIFAR100_DATA_ROOT` to use non-default artifact/dataset locations.

Datasets, weights, embeddings, and generated indexes are ignored by Git. The
core database never imports PyTorch or CLIP; these dependencies are isolated
in `requirements-demo.txt`.
