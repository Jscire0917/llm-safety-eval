# src/ai_eval/utils/logging.py
"""
Simple logging setup with loguru.
"""

from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
logger.add("evaluation_results/logs.txt", rotation="1 MB", level="DEBUG")