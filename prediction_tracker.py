import json
from datetime import datetime, timezone
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PredictionTracker:
    def __init__(self, data_dir=None, filename="predictions.json"):
        # Use current working directory instead of file location
        if data_dir is None:
            self.data_dir = Path.cwd() / "data"
        else:
            data_path = Path(data_dir)
            if not data_path.is_absolute():
                self.data_dir = Path.cwd() / data_path
            else:
                self.data_dir = data_path
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.filepath = self.data_dir / filename
        self._predictions = self._load()

    def _load(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, "r") as file:
                    return json.load(file)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading predictions: {e}")
                return {}
        else:
            return {}
        
    def _save(self):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, "w") as file:
                json.dump(self._predictions, file, indent=4)
        except IOError as e:
            print(f"Error saving predictions: {e}")
            logger.error(f"Error saving predictions: {e}")
    
    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    def update_prediction(self, question_id: int, value: list, timestamp=None):
        qid = str(question_id)
        if qid not in self._predictions:
            self._predictions[qid] = {
                "first": value,
                "latest": value,
                "timestamp_first": timestamp or self._now(),
                "timestamp_latest": timestamp or self._now(),
            }
        else:
            self._predictions[qid]["latest"] = value
            self._predictions[qid]["timestamp_latest"] = timestamp or self._now()

        self._save()

    def get_prediction(self, question_id: int):
        return self._predictions.get(str(question_id))
    
    def list_questions(self):
        return list(self._predictions.keys())
    