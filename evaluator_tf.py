import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    import evaluator
except Exception as exc:
    raise SystemExit(
        "Failed to import the TensorFlow evaluator. Use an environment with "
        "tensorflow.compat.v1 plus numpy<2/protobuf<4, or run the DLC scripts "
        "with AUTO_FIX_FID_ENV=1 to repair the FID environment before eval."
    ) from exc


DEFAULT_INCEPTION_GRAPH = (
    "/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/"
    "codes/REPA/classify_image_graph_def.pb"
)


if __name__ == "__main__":
    inception_graph = os.environ.get("INCEPTION_V3_PATH", DEFAULT_INCEPTION_GRAPH)
    if not os.path.isfile(inception_graph):
        raise FileNotFoundError(f"Missing Inception graph: {inception_graph}")
    evaluator.INCEPTION_V3_PATH = inception_graph
    evaluator.main()
