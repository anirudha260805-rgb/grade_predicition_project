"""Interactively collect student details and show score-model predictions."""

import os

from final_grade import predict_final_grade
from stu_per import MODEL_PATH, predict_score, train_model


FEATURE_INPUT_ORDER = [
    "age",
    "gender",
    "school_type",
    "parent_education",
    "study_hours",
    "attendance_percentage",
    "internet_access",
    "travel_time",
    "extra_activities",
    "study_method",
    "math_score",
    "science_score",
    "english_score",
]
"""Feature names, in the order prompted and accepted by the predictor."""

FEATURE_DETAILS = {
    "age": {"label": "Age", "unit": "years", "type": "numeric"},
    "gender": {"label": "Gender", "options": ["male", "female", "other"], "type": "categorical"},
    "school_type": {"label": "School type", "options": ["public", "private"], "type": "categorical"},
    "parent_education": {
        "label": "Parent education",
        "options": ["no formal", "high school", "diploma", "graduate", "post graduate", "phd"],
        "type": "categorical",
    },
    "study_hours": {"label": "Study hours", "unit": "hours per day", "type": "numeric"},
    "attendance_percentage": {"label": "Attendance", "unit": "percentage", "type": "numeric"},
    "internet_access": {"label": "Internet access", "options": ["yes", "no"], "type": "categorical"},
    "travel_time": {
        "label": "Travel time",
        "unit": "minutes",
        "type": "numeric",
    },
    "extra_activities": {"label": "Extra activities", "options": ["yes", "no"], "type": "categorical"},
    "study_method": {
        "label": "Study method",
        "options": ["notes", "textbook", "group study", "coaching", "mixed", "online videos"],
        "type": "categorical",
    },
    "math_score": {"label": "Math score", "unit": "out of 100", "type": "numeric"},
    "science_score": {"label": "Science score", "unit": "out of 100", "type": "numeric"},
    "english_score": {"label": "English score", "unit": "out of 100", "type": "numeric"},
}
"""Prompt labels, accepted categories, and units for each model input."""


def format_prompt(column):
    """Build a readable input prompt for one feature."""
    detail = FEATURE_DETAILS.get(column, {})
    label = detail.get("label", column.replace("_", " ").title())
    unit = detail.get("unit")
    options = detail.get("options")

    if options:
        return f"Enter {label} ({', '.join(options)}): "
    if unit:
        return f"Enter {label} ({unit}): "
    return f"Enter {label}: "


def parse_feature_value(column, raw_value):
    """Validate and convert one prompt response to its feature value."""
    detail = FEATURE_DETAILS.get(column, {})
    value = raw_value.strip()

    if not value:
        raise ValueError("Input cannot be empty.")

    if column == "travel_time":
        try:
            minutes = float(value)
        except ValueError as exc:
            raise ValueError("Please enter a valid number for Travel time.") from exc
        if not (-float("inf") < minutes < float("inf")) or minutes < 0:
            raise ValueError("Travel time must be a finite, non-negative number.")
        if minutes < 15:
            return "<15 min"
        if minutes <= 30:
            return "15-30 min"
        if minutes <= 60:
            return "30-60 min"
        return ">60 min"
    
    options = detail.get("options")
    if options:
        normalized = value.lower()
        for option in options:
            if normalized == option.lower():
                return option
        raise ValueError(f"Please choose one of: {', '.join(options)}")

    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"Please enter a valid number for {detail.get('label', column)}.") from exc

    if column == "age":
        if not 14 <= number <= 20:
            raise ValueError("Please enter a valid age (14 to 20).")
        return number

    if column == "attendance_percentage" and not 0 <= number <= 100:
        raise ValueError("Please enter a valid attendance (0 to 100).")

    if column in {"math_score", "science_score", "english_score"} and not 0 <= number <= 100:
        raise ValueError(f"{detail.get('label', column)} must be between 0 and 100.")

    return number


def get_feature_inputs():
    """Prompt for all model features and return them by feature name."""
    feature_columns = FEATURE_INPUT_ORDER
    raw_features = {}

    for column in feature_columns:
        while True:
            try:
                value = input(format_prompt(column))
                raw_features[column] = parse_feature_value(column, value)
                break
            except ValueError as exc:
                print(f"Invalid input: {exc}")

    return raw_features


def evaluate_student_performance(features=None):
    """Predict and display a student's overall score and validation metrics."""
    if features is None:
        print("\nEnter student details")
        features = get_feature_inputs()

    predicted_score = float(predict_score(features))
    predicted_grade = predict_final_grade(predicted_score)
    print("\n-------------------------")
    print(f"Predicted overall score: {predicted_score:.1f}")
    print(f"Predicted final grade: {predicted_grade}")
    print("Held-out error metrics are displayed when the model is trained.")
    print("-------------------------")
    return predicted_score


def main():
    """Run the interactive evaluator until the user declines another prediction."""
    if not os.path.exists(MODEL_PATH):
        print("No saved score model found; training it before starting evaluation.")
        train_model()

    while True:
        evaluate_student_performance()
        again = input("\nPredict another student? (yes/no): ").strip()
        if not again.lower().startswith("y"):
            break


if __name__ == "__main__":
    main()
