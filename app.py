import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Study Schedule Optimisation (ES)",
    layout="centered"
)

st.title("📘 Study Schedule Optimisation Using Evolution Strategies")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MAX_DAILY_HOURS = 2.5

# ===============================
# LOAD & PREPROCESS DATA
# ===============================
def load_and_aggregate():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "student_schedule.csv"

    if not csv_path.exists():
        st.error(f"Dataset not found: {csv_path}")
        st.stop()

    df = pd.read_csv(csv_path)

    daily = (
        df.groupby(["StudentName", "Day"])["Duration"]
        .sum()
        .unstack(fill_value=0)
    )

    daily = daily.reindex(columns=DAYS, fill_value=0)
    return daily

# ===============================
# EVOLUTION STRATEGIES
# ===============================
def fitness(schedule):
    variance = np.var(schedule)
    overload_penalty = np.sum(np.maximum(0, schedule - MAX_DAILY_HOURS))
    return variance + overload_penalty

def optimise_es(original_schedule, generations=120, mu=10, lam=40):
    total_hours = np.sum(original_schedule)
    days = len(original_schedule)

    population = np.random.dirichlet(np.ones(days), mu) * total_hours
    sigma = 0.6

    for _ in range(generations):
        offspring = []

        for parent in population:
            for _ in range(lam // mu):
                child = parent + np.random.normal(0, sigma, days)
                child = np.clip(child, 0, None)

                # Preserve total weekly hours
                child *= total_hours / np.sum(child)
                offspring.append(child)

        offspring = np.array(offspring)
        scores = np.array([fitness(o) for o in offspring])

        population = offspring[np.argsort(scores)[:mu]]
        sigma *= 0.98  # self-adaptation

    return population[0]

# ===============================
# MAIN APP
# ===============================
daily_df = load_and_aggregate()

student = st.selectbox("Select Student", daily_df.index)

original = daily_df.loc[student].values
optimised = optimise_es(original)

st.subheader("Original Daily Study Hours")
st.table(pd.DataFrame(original, index=DAYS, columns=["Hours"]))

st.subheader("Optimised Daily Study Hours (Evolution Strategies)")
st.table(pd.DataFrame(optimised.round(2), index=DAYS, columns=["Hours"]))

st.subheader("Summary")
st.write(f"Total Weekly Hours: **{original.sum():.2f} → {optimised.sum():.2f}**")
st.write(f"Fitness Improvement: **{fitness(original):.4f} → {fitness(optimised):.4f}**")
