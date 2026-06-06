from src.evaluate import evaluate_predictions


def test_evaluate_returns_accuracy_and_macro_f1():
    y_true = ["政治", "體育", "政治", "體育"]
    y_pred = ["政治", "體育", "體育", "體育"]
    m = evaluate_predictions(y_true, y_pred)
    assert m["accuracy"] == 0.75
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert "report" in m
