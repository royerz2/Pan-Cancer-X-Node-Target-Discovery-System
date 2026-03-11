#!/usr/bin/env python3
"""Build the LINCS L1000 signature index."""
import logging
import traceback
import sys
import os
import signal

# Force unbuffered
os.environ["PYTHONUNBUFFERED"] = "1"

# Log to file + stdout
log_file = "lincs_build.log"
handlers = [
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(log_file, mode="w"),
]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=handlers,
)
logger = logging.getLogger(__name__)

def handle_signal(signum, frame):
    logger.error(f"Received signal {signum}, exiting")
    sys.exit(1)

signal.signal(signal.SIGTERM, handle_signal)

try:
    from alin.lincs import LINCSSignatureDB

    pert_types = ["trt_xpr", "trt_sh", "trt_cp"]
    logger.info(
        "Starting LINCS index build (all 3 modalities: %s, landmark genes)...",
        ", ".join(pert_types),
    )
    db = LINCSSignatureDB(
        "lincs_data", pert_types=pert_types, landmark_only=True
    )
    db.build_index(force=True)

    summary = db.summary()
    logger.info("=== INDEX BUILD COMPLETE ===")
    logger.info("Targets: %d", db.n_targets)
    logger.info("Multi-modal targets: %d", summary.get("n_multi_modal", 0))
    logger.info("With compound evidence: %d", summary.get("n_with_compound_evidence", 0))
    logger.info("With genetic evidence: %d", summary.get("n_with_genetic_evidence", 0))
    logger.info("Modality counts: %s", summary.get("modality_counts", {}))

    print(f"=== INDEX BUILD COMPLETE ===")
    print(f"Targets: {db.n_targets}")
    print(f"Multi-modal targets: {summary.get('n_multi_modal', 0)}")
    print(f"Modality counts: {summary.get('modality_counts', {})}")
except Exception as e:
    traceback.print_exc()
    logger.error(f"FAILED: {e}")
    traceback.print_exc(file=open(log_file, "a"))
    sys.exit(1)
