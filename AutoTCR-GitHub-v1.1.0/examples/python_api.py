"""Run from the repository root after installing runtime dependencies."""
from autotcr import AutoTCRPredictor

predictor = AutoTCRPredictor.from_pretrained("loveCloud/AutoTCR", device="cpu")
result = predictor.predict_repertoire("examples/repertoire.csv", sample_id="example")
result.save(summary_path="outputs/summary.csv", predictions_path="outputs/predictions.csv.gz")
print(result.summary)

