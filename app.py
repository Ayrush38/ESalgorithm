import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Study Schedule Optimisation (Evolution Strategies)",
    layout="centered"
)

st.title("📘 Study Schedule Optimisation Using Evolution Strategies")

# ==================================================
# CONSTANTS
# ==================================================
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MAX_DAILY_HOURS = 2.5

# ==================================================
# LOAD & PREPROCESS DATA
# ==================================================
def load_and_aggregate():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "student_schedule (1).csv"

    if not csv_path.exists():
        st.error("❌ student_schedule.csv not found in project folder")
        st.stop()

    df = pd.read_csv(csv_path)

    daily = (
        df.groupby(["StudentName", "Day"])["Duration"]
        .sum()
        .unstack(fill_value=0)
    )

    daily = daily.reindex(columns=DAYS, fill_value=0)
    return daily

# ==================================================
# FITNESS FUNCTION (MULTI-OBJECTIVE)
# ==================================================
def fitness(schedule):
    variance = np.var(schedule)
    overload_penalty = np.sum(np.maximum(0, schedule - MAX_DAILY_HOURS))
    return variance + overload_penalty

# ==================================================
# EVOLUTION STRATEGIES
# ==================================================
def optimise_es(original_schedule, generations=120, mu=10, lam=40):
    total_hours = np.sum(original_schedule)
    days = len(original_schedule)

    population = np.random.dirichlet(np.ones(days), mu) * total_hours
    sigma = 0.6
    fitness_history = []

    for _ in range(generations):
        offspring = []

        for parent in population:
            for _ in range(lam // mu):
                child = parent + np.random.normal(0, sigma, days)
                child = np.clip(child, 0, None)
                child *= total_hours / np.sum(child)
                offspring.append(child)

        offspring = np.array(offspring)
        scores = np.array([fitness(o) for o in offspring])

        best_idx = np.argsort(scores)[:mu]
        population = offspring[best_idx]

        fitness_history.append(scores[best_idx[0]])
        sigma *= 0.98  # self-adaptation

    return population[0], fitness_history

# ==================================================
# MAIN APP
# ==================================================
daily_df = load_and_aggregate()

student = st.selectbox("Select Student", daily_df.index)

original = daily_df.loc[student].values
optimised, fitness_curve = optimise_es(original)

# ==================================================
# DISPLAY TABLES
# ==================================================
st.subheader("Original Daily Study Hours")
st.table(pd.DataFrame(original, index=DAYS, columns=["Hours"]))

st.subheader("Optimised Daily Study Hours (Evolution Strategies)")
st.table(pd.DataFrame(optimised.round(2), index=DAYS, columns=["Hours"]))

# ==================================================
# PERFORMANCE METRICS
# ==================================================
st.subheader("Performance Analysis")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Weekly Hours (Before)", f"{original.sum():.2f}")
    st.metric("Fitness (Before)", f"{fitness(original):.4f}")

with col2:
    st.metric("Total Weekly Hours (After)", f"{optimised.sum():.2f}")
    st.metric("Fitness (After)", f"{fitness(optimised):.4f}")

# ==================================================
# VISUALISATION
# ==================================================
st.subheader("Workload Distribution Comparison")

fig, ax = plt.subplots()
ax.bar(DAYS, original, alpha=0.7, label="Original")
ax.bar(DAYS, optimised, alpha=0.7, label="Optimised")
ax.axhline(MAX_DAILY_HOURS, linestyle="--", label="Max Recommended Hours")
ax.set_ylabel("Study Hours")
ax.legend()

st.pyplot(fig)

# ==================================================
# FITNESS CONVERGENCE
# ==================================================
st.subheader("Fitness Convergence (Extended Analysis)")

fig2, ax2 = plt.subplots()
ax2.plot(fitness_curve)
ax2.set_xlabel("Generation")
ax2.set_ylabel("Fitness Value")
ax2.set_title("Evolution Strategies Convergence")

st.pyplot(fig2)

# ==================================================
# SUMMARY
# ==================================================
st.subheader("Summary")
st.write(
    f"""
    - **Variance reduced:** {np.var(original):.4f} → {np.var(optimised):.4f}  
    - **Overload reduced:** Days exceeding {MAX_DAILY_HOURS} hours were minimised  
    - **Total weekly hours preserved:** {original.sum():.2f} hours  
    - **Multi-objective optimisation:** Balance + overload reduction  
    """
)
