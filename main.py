import pandas as pd
import streamlit as st
import plotly.express as px

# --- Load Data ---
df = pd.read_csv("cleaned_ctg_uc_sites.csv")

# --- Clean and Prepare ---
df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
df['primary_completion_date'] = pd.to_datetime(df['primary_completion_date'], errors='coerce')

# Phase tagging
def map_phase(phase_str):
    if pd.isna(phase_str): return 'unknown'
    p = phase_str.lower()
    if 'early phase 1' in p or 'phase 1' in p: return 'phase_1'
    elif 'phase 2' in p: return 'phase_2'
    return 'other'

df['phase_tag'] = df['phases'].apply(map_phase)

# Keyword scoring
keywords = [
    'adc', 'antibody-drug conjugate', 'antibody drug conjugate', 'bioconjugate',
    'slitrk6', 'slitrk', 'topoisomerase', 'topo-i', 'exatecan',
    'camptothecin', 'irinotecan', 'sn-38', 'dxd', 'deruxtecan',
    'payload', 'cleavable linker', 'hydrophilic linker', 'sesutecan',
    'targeted', 'targeted therapy', 'biomarker', 'ihc'
]

def keyword_score(row):
    text = f"{row.get('interventions', '')} {row.get('study_title', '')}".lower()
    return sum(1 for kw in keywords if kw in text)

df['keyword_hits'] = df.apply(keyword_score, axis=1)

# Aggregate to site level
site_summary = (
    df.groupby('institution')
      .agg(
          n_trials=('nct_number', 'nunique'),
          phase_1_trials=('phase_tag', lambda x: (x == 'phase_1').sum()),
          phase_2_trials=('phase_tag', lambda x: (x == 'phase_2').sum()),
          last_start_date=('start_date', 'max'),
          last_complete_date=('primary_completion_date', 'max'),
          active_trials=('study_status', lambda x: x.str.lower().str.contains("recruiting|not yet|active").sum()),
          keyword_signal=('keyword_hits', 'sum')
      )
      .reset_index()
)

# Recency calculation
site_summary['years_since_last_start'] = (pd.Timestamp.now() - site_summary['last_start_date']).dt.days / 365.25
site_summary = site_summary.fillna({'years_since_last_start': site_summary['years_since_last_start'].max()})

# --- Streamlit UI ---
st.title("Urothelial Cancer Trial Site Scorer")
st.markdown("Use sliders to weight importance of each criterion.")

volume_wt = st.slider("Trial Volume Weight", 0.0, 5.0, 0.0, 0.1)
early_phase_wt = st.slider("Early Phase Experience Weight", 0.0, 5.0, 0.0, 0.1)
recency_penalty = st.slider("Recency Penalty Weight (older = worse)", 0.0, 5.0, 0.0, 0.1)
mechanism_wt = st.slider("Mechanism Keyword Match Weight", 0.0, 5.0, 0.0, 0.1)
competition_penalty = st.slider("Active Competing Trials Penalty", 0.0, 5.0, 0.0, 0.1)

# Scoring function
def calculate_score(df):
    return (
        df['n_trials'] * volume_wt +
        df['phase_1_trials'] * early_phase_wt +
        df['keyword_signal'] * mechanism_wt -
        df['years_since_last_start'] * recency_penalty -
        df['active_trials'] * competition_penalty
    )

site_summary['score'] = calculate_score(site_summary)

# --- 2x2 Plot ---
x_axis = st.selectbox("X Axis", ['n_trials', 'phase_1_trials', 'keyword_signal', 'years_since_last_start'])
y_axis = st.selectbox("Y Axis", ['score', 'keyword_signal', 'years_since_last_start', 'active_trials'])

fig = px.scatter(
    site_summary,
    x=x_axis,
    y=y_axis,
    size='phase_1_trials',
    color='score',
    hover_name='institution',
    title="Site Scoring Bubble Plot",
    height=700
)
st.plotly_chart(fig)

# Show table
st.dataframe(site_summary.sort_values('score', ascending=False))
