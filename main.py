import time
import argparse
import logging
import numpy as np
import parsl

from config import gen_config
from apps import kmeans_fragment, reduce_and_update

# === PARÂMETROS ===
N_POINTS = 131_072_000      
DIMENSIONS = 100
K = 1000
N_FRAGMENTS = 1024
ITERATIONS = 10
SEED = 42

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )
    logging.getLogger("parsl").setLevel(logging.WARNING)
    logging.getLogger("parsl.dataflow").setLevel(logging.WARNING)
    logging.getLogger("parsl.executors").setLevel(logging.WARNING)
    logging.getLogger("parsl.providers").setLevel(logging.WARNING)
    logging.getLogger("parsl.jobs").setLevel(logging.WARNING)
    logging.getLogger("parsl").propagate = False
def main(args):
    setup_logging()

    logging.info("========== KMEANS ==========")
    logging.info(f"Pontos totais : {N_POINTS}")
    logging.info(f"Dimensões    : {DIMENSIONS}")
    logging.info(f"Clusters (K) : {K}")
    logging.info(f"Fragmentos   : {N_FRAGMENTS}")
    logging.info(f"Iterações    : {ITERATIONS}")
    logging.info("==============================================")

    points_per_fragment = N_POINTS // N_FRAGMENTS

    np.random.seed(SEED)
    centroids = np.random.random((K, DIMENSIONS)).astype(np.float64)

    start_total = time.time()

    for it in range(ITERATIONS):
        logging.info(f"--- ITERAÇÃO {it + 1}/{ITERATIONS} ---")
        iter_start = time.time()

        futures = []

        for frag_id in range(N_FRAGMENTS):
            futures.append(
                kmeans_fragment(
                    fragment_id=SEED + frag_id + it * 100_000,
                    points_per_fragment=points_per_fragment,
                    dimensions=DIMENSIONS,
                    centroids=centroids
                )
            )

        logging.info("[MAIN] Redução global e atualização dos centróides...")
        centroids = reduce_and_update(centroids, *futures).result()

        logging.info(
            f"[MAIN] Iteração {it + 1} concluída em "
            f"{time.time() - iter_start:.2f} s"
        )

    logging.info("========== FIM ==========")
    logging.info(
        f"Tempo total (10 iterações): "
        f"{time.time() - start_total:.2f} s"
    )
    
    parsl.dfk().cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onslurm", action="store_true")
    args = parser.parse_args()

    cfg = gen_config(slurm=args.onslurm, monitoring=False)
    parsl.load(cfg)

    main(args)
