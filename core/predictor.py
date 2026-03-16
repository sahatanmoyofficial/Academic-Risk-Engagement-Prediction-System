from utils.model_loader import load_model
from utils.validators import validate_input
from core.feature_builder import build_features
from core.recommender import generate_recommendations

model = load_model()

feature_cols = ["total_click",
                "early_click",
                "early_active_days",
                "first_activity_day",
                "pre_course_engaged"
]

def predict_and_recommend(student_dict):

    validate_input(student_dict)

    input_df = build_features(student_dict)

    risk_probability = model.predict_proba(input_df)[0][1]

    if risk_probability >=0.7:
        risk_level = "High"
    elif risk_probability >=0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    observations, actions = generate_recommendations(risk_probability, student_dict)

    tree_explanation = explain_tree_decision(model, input_df, feature_cols)

    result = {
        "risk_probability" : round(float(risk_probability), 3),
        "risk_level" : risk_level,
        "observations" : observations,
        "recommendations" : actions,
        "model_explanation" : tree_explanation
    }

    return result

def explain_tree_decision(model, input_df, feature_names):
    node_indicators = model.decision_path(input_df) # contains index of nodes that got activated
    feature = model.tree_.feature # -2, 0
    threshold = model.tree_.threshold

    # print(node_indicators)

    # print(node_indicators.indices)

    # print(feature)

    # print(threshold)
    # Compressed Sparse Row sparse matrix of dtype 'int64'
    #     with 5 stored elements and shape (1, 31)>
    # Coords        Values
    # (0, 0)        1
    # (0, 1)        1
    # (0, 2)        1
    # (0, 6)        1
    # (0, 7)        1
    # [0 1 2 6 7]
    # [ 0  0  0  0 -2 -2  1 -2 -2  1  3 -2 -2  3 -2 -2  0  1  3 -2 -2  3 -2 -2
    # 0  1 -2 -2  0 -2 -2]
    # [ 3.7750e+02  1.3950e+02  7.1500e+01  2.0500e+01 -2.0000e+00 -2.0000e+00
    # 1.0500e+01 -2.0000e+00 -2.0000e+00  5.1500e+01 -1.0500e+01 -2.0000e+00
    # -2.0000e+00 -1.0500e+01 -2.0000e+00 -2.0000e+00  1.1655e+03  1.1850e+02
    # -1.0500e+01 -2.0000e+00 -2.0000e+00 -1.0500e+01 -2.0000e+00 -2.0000e+00
    # 2.1625e+03  2.5350e+02 -2.0000e+00 -2.0000e+00  4.2730e+03 -2.0000e+00
    # -2.0000e+00]

    


    explanation = []

    for node_id in node_indicators.indices:
        if feature[node_id] != -2:
            fname = feature_names[feature[node_id]]
            thresh = threshold[node_id]
            val = input_df.iloc[0][fname]

            if val <= thresh:
                explanation.append(
                    f"{fname.replace('_', ' ').title()} <= {int(thresh)}"
                )
            else:
                explanation.append(
                    f"{fname.replace('_', ' ').title()} > {int(thresh)}"
                )

    return explanation
