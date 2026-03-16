# src/ai_eval/db/db_utils.py
"""
SQLite database utilities for saving and querying evaluation results.
Saves results for leaderboard ranking.
"""

import sqlite3
from typing import Dict, Any
from datetime import datetime

DB_PATH = "src/ai_eval/db/results.db"


def init_db():
    """Initialize SQLite database if not exists."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            provider TEXT,
            timestamp DATETIME,
            bias_score FLOAT,
            toxicity_score FLOAT,
            hallucination_score FLOAT,
            average_score FLOAT
        )
    """)
    conn.commit()
    conn.close()

# Call once at server startup (in api.py)
init_db()

def save_evaluation(model_name: str, provider: str, results: Dict[str, Any]):
    """Save evaluation results to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    bias_score = results.get("BiasEvaluator", {}).get("bias_score", 0.0)
    toxicity_score = results.get("ToxicityEvaluator", {}).get("toxicity_score", 0.0)
    hallucination_score = results.get("HallucinationEvaluator", {}).get("hallucination_score", 0.0)
    average_score = (bias_score + toxicity_score + hallucination_score) / 3

    cursor.execute("""
        INSERT INTO evaluations (model_name, provider, timestamp, bias_score, toxicity_score, hallucination_score, average_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (model_name, provider, datetime.now(), bias_score, toxicity_score, hallucination_score, average_score))
    conn.commit()
    conn.close()

def get_leaderboard():
    """Query and rank models by average score (low to high)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT model_name, provider, AVG(average_score) as avg_score
        FROM evaluations
        GROUP BY model_name, provider
        ORDER BY avg_score ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_comparison_table():
    """Return per-metric averages for comparison table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            model_name,
            provider,
            AVG(bias_score) as avg_bias,
            AVG(toxicity_score) as avg_toxicity,
            AVG(hallucination_score) as avg_hallucination
        FROM evaluations
        GROUP BY model_name, provider
        ORDER BY avg_bias ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows