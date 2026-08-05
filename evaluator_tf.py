import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import evaluator


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
