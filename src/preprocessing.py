"""Loading and preprocessing for the UCI student dropout dataset.

Category codes follow the variable descriptions on the UCI dataset page:
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
"""
import numpy as np
import pandas as pd

COLUMN_NAMES = {
    "Marital status": "marital_status",
    "Application mode": "application_mode",
    "Application order": "application_order",
    "Course": "course",
    "Daytime/evening attendance": "daytime_attendance",  # 1 = daytime, 0 = evening
    "Previous qualification": "previous_qualification",
    "Previous qualification (grade)": "previous_qualification_grade",
    "Nacionality": "nationality",
    "Mother's qualification": "mother_qualification",
    "Father's qualification": "father_qualification",
    "Mother's occupation": "mother_occupation",
    "Father's occupation": "father_occupation",
    "Admission grade": "admission_grade",
    "Displaced": "displaced",
    "Educational special needs": "special_needs",
    "Debtor": "debtor",
    "Tuition fees up to date": "tuition_up_to_date",
    "Gender": "male",  # 1 = male, 0 = female
    "Scholarship holder": "scholarship",
    "Age at enrollment": "age_at_enrollment",
    "International": "international",
    "Curricular units 1st sem (credited)": "units_1st_credited",
    "Curricular units 1st sem (enrolled)": "units_1st_enrolled",
    "Curricular units 1st sem (evaluations)": "units_1st_evaluations",
    "Curricular units 1st sem (approved)": "units_1st_approved",
    "Curricular units 1st sem (grade)": "grade_1st",
    "Curricular units 1st sem (without evaluations)": "units_1st_no_evaluation",
    "Curricular units 2nd sem (credited)": "units_2nd_credited",
    "Curricular units 2nd sem (enrolled)": "units_2nd_enrolled",
    "Curricular units 2nd sem (evaluations)": "units_2nd_evaluations",
    "Curricular units 2nd sem (approved)": "units_2nd_approved",
    "Curricular units 2nd sem (grade)": "grade_2nd",
    "Curricular units 2nd sem (without evaluations)": "units_2nd_no_evaluation",
    "Unemployment rate": "unemployment_rate",
    "Inflation rate": "inflation_rate",
    "GDP": "gdp",
    "Target": "target",  # Dropout / Enrolled / Graduate
}

# Parents' qualification codes grouped into four levels.
QUALIFICATION_GROUPS = {
    "basic": [19, 35, 36, 37, 38],  # cannot read/write up to basic education 3rd cycle
    "secondary": [1, 9, 10, 11, 12, 13, 14, 20, 25, 26, 29, 30],  # 7th-12th year, complementary courses
    "higher": [2, 3, 4, 5, 6, 22, 39, 40, 41, 42, 43, 44],  # degrees, technical/professional higher courses
    "other": [18, 27, 31, 33, 34],  # commerce/administration courses, unknown
}

# Parents' occupation codes grouped by skill level.
OCCUPATION_GROUPS = {
    # Directors, specialists (science, health, teaching, ICT, finance)
    "high": [1, 2, 112, 114, 121, 122, 123, 124, 125],
    # Technicians, administrative staff, services, sales, skilled trades, operators
    "medium": [3, 4, 5, 6, 7, 8, 131, 132, 134, 135, 141, 143, 151, 152, 153, 154,
               161, 163, 171, 172, 173, 174, 181, 182, 183],
    # Unskilled workers, cleaning, food processing, meal assistants, street vendors
    "low": [9, 175, 191, 192, 193, 194, 195],
    # Student, armed forces, other situation, blank, other support staff
    "other": [0, 10, 90, 99, 101, 102, 103, 144],
}

ONE_HOT_COLUMNS = [
    "marital_status", "application_mode", "course", "previous_qualification",
    "nationality", "mother_qualification", "father_qualification", "mother_occupation", "father_occupation",
]


def _code_to_group(groups: dict) -> dict:
    return {code: group for group, codes in groups.items() for code in codes}


QUALIFICATION_BY_CODE = _code_to_group(QUALIFICATION_GROUPS)
OCCUPATION_BY_CODE = _code_to_group(OCCUPATION_GROUPS)


def load_data(path) -> pd.DataFrame:
    """Read the raw UCI file and rename the columns to snake_case."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df.rename(columns=COLUMN_NAMES)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run consistency checks, print any issue found and fix the ones that can be fixed."""
    df = df.copy()
    issues = []

    # More approved units than enrolled units?
    more_1st = df["units_1st_approved"] > df["units_1st_enrolled"]
    more_2nd = df["units_2nd_approved"] > df["units_2nd_enrolled"]
    if more_1st.any() or more_2nd.any():
        issues.append(f"students with more approved than enrolled units: {more_1st.sum() + more_2nd.sum()}")
        # cap approved units at the number of enrolled units
        df.loc[more_1st, "units_1st_approved"] = df.loc[more_1st, "units_1st_enrolled"]
        df.loc[more_2nd, "units_2nd_approved"] = df.loc[more_2nd, "units_2nd_enrolled"]

    for col in ("unemployment_rate", "inflation_rate"):
        if ((df[col] < -50) | (df[col] > 100)).any():
            issues.append(f"extreme values in {col}")

    # Semester grades are on a 0-20 scale.
    for col in ("grade_1st", "grade_2nd"):
        if (df[col] > 20).any():
            issues.append(f"grade outside 0-20 in {col}")

    if (df["age_at_enrollment"] < 14).any():
        issues.append("students younger than 14")

    for issue in issues:
        print(f"[validation] {issue}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features. Must run before encode_categorical."""
    df = df.copy()

    # Share of enrolled units that were approved, per semester (0 if nothing enrolled).
    df["pass_ratio_1st"] = df["units_1st_approved"] / df["units_1st_enrolled"].replace(0, np.nan)
    df["pass_ratio_2nd"] = df["units_2nd_approved"] / df["units_2nd_enrolled"].replace(0, np.nan)
    df[["pass_ratio_1st", "pass_ratio_2nd"]] = df[["pass_ratio_1st", "pass_ratio_2nd"]].fillna(0)

    # Change in average grade between semesters.
    df["grade_change"] = df["grade_2nd"] - df["grade_1st"]

    # Total academic load over the year.
    df["total_units_enrolled"] = df["units_1st_enrolled"] + df["units_2nd_enrolled"]

    # Both parents with higher education.
    mother = df["mother_qualification"].map(QUALIFICATION_BY_CODE)
    father = df["father_qualification"].map(QUALIFICATION_BY_CODE)
    df["both_parents_higher_ed"] = ((mother == "higher") & (father == "higher")).astype(int)
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Group parents' qualification/occupation, simplify nationality and one-hot encode."""
    df = df.copy()
    for col in ("mother_qualification", "father_qualification"):
        df[col] = df[col].map(QUALIFICATION_BY_CODE)
    for col in ("mother_occupation", "father_occupation"):
        df[col] = df[col].map(OCCUPATION_BY_CODE)
    unmapped = df[["mother_qualification", "father_qualification",
                   "mother_occupation", "father_occupation"]].isna().sum().sum()
    if unmapped:
        raise ValueError(f"{unmapped} qualification/occupation codes without a group")

    # Nationality: Portuguese (code 1) or other.
    df["nationality"] = df["nationality"].apply(lambda x: "portuguese" if x == 1 else "other")
    return pd.get_dummies(df, columns=ONE_HOT_COLUMNS, drop_first=True, dtype=int)


def fit_age_cap(train: pd.DataFrame, quantile: float = 0.99) -> float:
    """Upper limit for age at enrolment, estimated on the training set only."""
    return float(train["age_at_enrollment"].quantile(quantile))


def apply_age_cap(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Clip age at enrolment at the given upper limit."""
    df = df.copy()
    df["age_at_enrollment"] = df["age_at_enrollment"].clip(upper=cap)
    return df


def split_features_target(df: pd.DataFrame, target: str, exclude=None):
    """Return (X, y), dropping the target and any extra columns in `exclude`."""
    drop = [target] + list(exclude or [])
    return df.drop(columns=drop), df[target]
