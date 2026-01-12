import pandas as pd
from pathlib import Path

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

def load_and_aggregate():
    # Get project root directory
    BASE_DIR = Path(__file__).resolve().parent.parent
    excel_path = BASE_DIR / "data" / "student_schedule.xlsx"

    df = pd.read_excel(excel_path)

    daily = (
        df.groupby(["StudentName", "Day"])["Duration"]
        .sum()
        .unstack(fill_value=0)
    )

    daily = daily.reindex(columns=DAYS, fill_value=0)
    return daily

