"""Unified evaluation harness for NetrAI experiments.

New experiments use this package so every number is computed one way, carries bootstrap
intervals at the cluster (patient/exam) level, and is stamped with its evidence tier.
Locked evaluations of rounds 2-4 keep using ``locked_evaluation.py`` unchanged.
"""
from . import bootstrap, grading, metrics, report  # noqa: F401
