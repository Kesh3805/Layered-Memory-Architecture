"""Wrapper to run comparison and save output to file."""
import sys
import logging

# Redirect logging to file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler("experiments/results/run_log.txt", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)

from experiments.compare import main
main()
