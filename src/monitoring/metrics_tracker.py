import json
from pathlib import Path
from datetime import datetime


class MetricsTracker:
    """Track system metrics over time"""

    def __init__(self, metrics_dir="data/monitoring"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def log_retrieval(self, query, recall, precision, mrr, latency):
        """Log retrieval metrics"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "recall@5": recall,
            "precision@5": precision,
            "mrr": mrr,
            "latency_seconds": latency
        }

        metrics_file = self.metrics_dir / "retrieval_metrics.jsonl"
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def log_generation(self, question, keyword_coverage, sources_accurate):
        """Log generation metrics"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "keyword_coverage": keyword_coverage,
            "source_accuracy": sources_accurate
        }

        metrics_file = self.metrics_dir / "generation_metrics.jsonl"
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def log_quality_score(self, metric_name, score, target):
        """Log overall quality"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "metric": metric_name,
            "score": score,
            "target": target,
            "passing": score >= target
        }

        quality_file = self.metrics_dir / "quality_scores.csv"
        # Append to CSV