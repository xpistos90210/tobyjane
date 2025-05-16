import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Load Data ---
df = pd.read_csv("cleaned_ctg_uc_sites.csv")

# --- Clean and Prepare ---
df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
df['primary_completion_date'] = pd.to_datetime(df['primary_completion_date'], errors='coerce')

def map_phases(phase_str):
    if pd.isna(phase_str):
        return []
    return [p.strip().lower() for p in str(phase_str).split('|')]

df['phase_tags'] = df['phases'].apply(map_phases)
df['is_phase_1'] = df['phase_tags'].apply(lambda tags: any('phase1' in p for p in tags))
df['is_phase_2'] = df['phase_tags'].apply(lambda tags: any('phase2' in p for p in tags))

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

site_summary = (
    df.groupby('institution')
      .agg(
          n_trials=('nct_number', 'nunique'),
          last_start_date=('start_date', 'max'),
          last_complete_date=('primary_completion_date', 'max'),
          active_trials=('study_status', lambda x: x.str.lower().str.contains("recruiting|not yet|active").sum()),
          keyword_signal=('keyword_hits', 'sum')
      )
      .reset_index()
)

phase_counts = (
    df.groupby('institution')
      .agg(
          phase_1_trials=('is_phase_1', 'sum'),
          phase_2_trials=('is_phase_2', 'sum')
      )
      .reset_index()
)
site_summary = site_summary.merge(phase_counts, on='institution', how='left')

site_summary['years_since_last_start'] = (pd.Timestamp.now() - site_summary['last_start_date']).dt.days / 365.25
site_summary = site_summary.fillna({'years_since_last_start': site_summary['years_since_last_start'].max()})

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Urothelial Cancer Trial Site Scorer")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Adjust Weights")
    volume_wt = st.slider("Trial Volume Weight", 0.0, 2.0, 0.0, 0.1)
    early_phase_wt = st.slider("Early Phase Experience Weight", 0.0, 2.0, 0.0, 0.1)
    recency_penalty = st.slider("Recency Penalty Weight (older = worse)", 0.0, 2.0, 0.0, 0.1)
    mechanism_wt = st.slider("Mechanism Keyword Match Weight", 0.0, 2.0, 0.0, 0.1)
    competition_penalty = st.slider("Active Competing Trials Penalty", 0.0, 2.0, 0.0, 0.1)

    x_axis = st.selectbox("X Axis", ['n_trials', 'phase_1_trials', 'keyword_signal', 'years_since_last_start'])
    y_axis = st.selectbox("Y Axis", ['score', 'keyword_signal', 'years_since_last_start', 'active_trials'])

    st.markdown("""
    **Bubble size** represents the number of Phase 1 urothelial cancer trials conducted by the site.  
    **Bubble color** represents the overall weighted score based on the criteria you've selected above.  
    Quadrants are determined by the median values of the X and Y axes.  
    
    **Quadrant Labels:**  
    ⬆ High Potential – High Score & High Feature Strength  
    ↘ Monitor – High Score but Lower Feature Strength  
    ↖ Consider – Strong Feature but Lower Score  
    ⬇ Low Priority – Lower on Both Dimensions  
    """)

    st.download_button(
        label="Download Ranked Site Table as CSV",
        data=site_summary.sort_values('score', ascending=False).to_csv(index=False),
        file_name='ranked_sites.csv',
        mime='text/csv'
    )

with col2:
    def calculate_score(df):
        return (
            df['n_trials'] * volume_wt +
            df['phase_1_trials'] * early_phase_wt +
            df['keyword_signal'] * mechanism_wt -
            df['years_since_last_start'] * recency_penalty -
            df['active_trials'] * competition_penalty
        )

    site_summary['score'] = calculate_score(site_summary)

    x_vals = site_summary[x_axis]
    y_vals = site_summary[y_axis]
    x_median = x_vals.median()
    y_median = y_vals.median()

    fig = px.scatter(
        site_summary,
        x=x_axis,
        y=y_axis,
        size='phase_1_trials',
        color='score',
        hover_name='institution',
        title="Site Scoring Bubble Plot",
        height=700,
        color_continuous_scale='Viridis'
    )

    fig.add_shape(type='line', x0=x_median, x1=x_median, y0=y_vals.min(), y1=y_vals.max(),
                  line=dict(dash='dash', width=1, color='gray'))
    fig.add_shape(type='line', x0=x_vals.min(), x1=x_vals.max(), y0=y_median, y1=y_median,
                  line=dict(dash='dash', width=1, color='gray'))

    st.plotly_chart(fig, use_container_width=True)

# Show table
st.markdown("### Ranked Site Table")
st.dataframe(site_summary.sort_values('score', ascending=False), use_container_width=True)