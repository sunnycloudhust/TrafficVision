import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from trafficvision.data.sources.cloud_storage import CloudObjectStorageSource
from trafficvision.data.datasets.ua_detrac import build_dataset, collate_detection
from torch.utils.data import DataLoader


def test_http_stream_and_cache(prepared):
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(prepared)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        metadata = json.loads((prepared / "shards.json").read_text())
        for shard in metadata["shards"]:
            shard["url"] = f"http://127.0.0.1:{server.server_port}/{shard['url']}"
        path = prepared / "remote.json"
        path.write_text(json.dumps(metadata))
        for mode in ("stream", "cache"):
            rows = list(CloudObjectStorageSource(path, "train", mode=mode))
            assert len(rows) == 1 and rows[0][1]["video_id"] == "MVI_0"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_stream_workers_do_not_duplicate_samples(prepared):
    dataset = build_dataset(
        dict(source="cloud", manifest=str(prepared / "shards.json"), mode="stream"), "train"
    )
    loader = DataLoader(
        dataset, num_workers=2, batch_size=1, collate_fn=collate_detection, multiprocessing_context="spawn"
    )
    ids = [int(target[0]["image_id"]) for _, target in loader]
    assert len(ids) == len(set(ids)) == 1
