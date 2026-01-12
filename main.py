import time
import argparse
import numpy as np
import parsl
import logging
from config import gen_config
from apps import kmeans_fragment, reduce_and_update

N_POINTS = 131_072_000
#N_POINTS = 2_000_000
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
    logging.info("\n========== KMEANS ==========")
    logging.info(f"Pontos totais : {N_POINTS}")
    logging.info(f"Dimensões    : {DIMENSIONS}")
    logging.info(f"Clusters     : {K}")
    logging.info(f"Fragmentos   : {N_FRAGMENTS}")
    logging.info(f"Iterações    : {ITERATIONS}")
    logging.info("==============================================\n")

    points_per_fragment = N_POINTS // N_FRAGMENTS

    np.random.seed(SEED)
    fragments = []

    logging.info("[MAIN] Gerando dataset fixo...")
    for i in range(N_FRAGMENTS):
        pts = np.random.random(
            (points_per_fragment, DIMENSIONS)
        ).astype(np.float64)
        fragments.append(pts)

    centroids = np.random.random((K, DIMENSIONS)).astype(np.float64)

    start_total = time.time()

    for it in range(ITERATIONS):
        logging.info(f"\n--- ITERAÇÃO {it+1}/{ITERATIONS} ---")
        iter_start = time.time()

        futures = []
        for i, pts in enumerate(fragments):
            futures.append(
                kmeans_fragment(pts, centroids)
            )

        centroids = reduce_and_update(centroids, *futures).result()

        logging.info(f"[MAIN] Iteração {it+1} finalizada em {time.time()-iter_start:.2f}s")

    logging.info("\n========== FIM ==========")
    logging.info(f"Tempo total: {time.time() - start_total:.2f} s")
    logging.info("========================\n")
  
    parsl.dfk().cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onslurm", action="store_true")
    args = parser.parse_args()

    cfg = gen_config(slurm=args.onslurm, monitoring=False)
    parsl.load(cfg)

    main(args)
