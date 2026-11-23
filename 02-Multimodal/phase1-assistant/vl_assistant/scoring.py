"""ChartQA relaxed accuracy (Masry et al., 2022, §5.1)."""
from collections import defaultdict
from decimal import Decimal, InvalidOperation


def number(text):
    try:
        value = Decimal(text[:-1]) / 100 if text.endswith("%") else Decimal(text)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def relaxed_correct(prediction, references):
    candidate = prediction.strip()
    actual = number(candidate)
    for reference in references:
        reference = reference.strip()
        expected = number(reference)
        if actual is not None and expected is not None:
            if abs(actual - expected) <= abs(expected) * Decimal("0.05"):
                return True
        elif candidate.casefold() == reference.casefold():
            return True
    return False


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["subset"]].append(row)
    groups["overall"] = rows
    return {
        group: {
            "n": len(items),
            "correct": sum(row["correct"] for row in items),
            "failed": sum(row["status"] != "ok" for row in items),
            "relaxed_accuracy": sum(row["correct"] for row in items) / len(items),
        }
        for group, items in groups.items()
        if items
    }
