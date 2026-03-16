
REQUIRED_FIELDS = ["total_click",
"early_click",
"early_active_days",
"first_activity_day",
"pre_course_engaged"
]

def validate_input(student_dict):
    missing = [k for k in REQUIRED_FIELDS if k not in student_dict]
    if missing:
        raise ValueError(f"Missing required fields : {missing}")