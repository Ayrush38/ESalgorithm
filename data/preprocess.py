import pandas as pd

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

def load_and_aggregate(path="data/student_schedule.xlsx"):
    df = pd.read_excel(path)

    # Aggregate total study hours per student per day
    daily = (
        df.groupby(["StudentName", "Day"])["Duration"]
        .sum()
        .unstack(fill_value=0)
    )

    # Ensure weekday order
    daily = daily.reindex(columns=DAYS, fill_value=0)

    return daily
