"""
Streamlit Dashboard for MedGraph Arena Results.
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    QUERY_OUTPUTS_PATH,
    RAGAS_RESULTS_PATH,
    JUDGE_RESULTS_PATH,
    SYSTEM_DISPLAY_NAMES,
)

st.set_page_config(page_title="MedGraph Arena", page_icon="🧬", layout="wide")

def load_data():
    queries = []
    if QUERY_OUTPUTS_PATH.exists():
        with open(QUERY_OUTPUTS_PATH, 'r') as f:
            queries = json.load(f)
            
    ragas = {}
    if RAGAS_RESULTS_PATH.exists():
        with open(RAGAS_RESULTS_PATH, 'r') as f:
            ragas = json.load(f)
            
    judge = {"stats": {}, "matchups": []}
    if JUDGE_RESULTS_PATH.exists():
        with open(JUDGE_RESULTS_PATH, 'r') as f:
            judge = json.load(f)
            
    return queries, ragas, judge

def main():
    st.title("🧬 MedGraph Arena Benchmark")
    st.markdown("Comparing 4 RAG architectures on medical knowledge: **Naive RAG vs GraphRAG vs LightRAG vs NodeRAG**.")

    queries, ragas, judge = load_data()

    if not queries:
        st.warning("No data found! Please run the benchmark pipeline first (`python run_benchmark.py`).")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Leaderboard", 
        "📊 RAGAS Metrics", 
        "⚖️ LLM Judge", 
        "🔍 Question Explorer"
    ])

    with tab1:
        st.header("Overall Performance")
        
        # Display Judge Win Rates
        if judge["stats"]:
            stats_df = pd.DataFrame.from_dict(judge["stats"], orient='index')
            stats_df.index = stats_df.index.map(lambda x: SYSTEM_DISPLAY_NAMES.get(x, x))
            stats_df['Total Matches'] = stats_df['wins'] + stats_df['losses'] + stats_df['ties']
            stats_df['Win Rate %'] = (stats_df['wins'] / stats_df['Total Matches'] * 100).round(1)
            stats_df = stats_df.sort_values('Win Rate %', ascending=False)
            
            st.subheader("LLM-as-Judge Win Rate")
            fig = px.bar(stats_df, y=stats_df.index, x='Win Rate %', orientation='h', 
                         color='Win Rate %', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

        # Display Latency Average
        sys_latencies = {sys: [] for sys in SYSTEM_DISPLAY_NAMES.keys()}
        for q in queries:
            for sys_name, sys_data in q['systems'].items():
                if 'latency_ms' in sys_data:
                    sys_latencies[sys_name].append(sys_data['latency_ms'])
                    
        avg_lat = {SYSTEM_DISPLAY_NAMES.get(k, k): sum(v)/len(v) if v else 0 for k, v in sys_latencies.items()}
        lat_df = pd.DataFrame(list(avg_lat.items()), columns=['System', 'Avg Latency (ms)'])
        lat_df = lat_df.sort_values('Avg Latency (ms)')
        
        st.subheader("Average Query Latency")
        fig_lat = px.bar(lat_df, x='System', y='Avg Latency (ms)', color='Avg Latency (ms)')
        st.plotly_chart(fig_lat, use_container_width=True)

    with tab2:
        st.header("RAGAS Evaluation Metrics")
        if ragas:
            df_ragas = pd.DataFrame.from_dict(ragas, orient='index')
            df_ragas.index = df_ragas.index.map(lambda x: SYSTEM_DISPLAY_NAMES.get(x, x))
            
            st.dataframe(df_ragas.style.highlight_max(axis=0, color='lightgreen'))
            
            # Radar chart
            categories = ['faithfulness', 'context_precision', 'context_recall']
            fig_radar = go.Figure()
            
            for sys_name in df_ragas.index:
                fig_radar.add_trace(go.Scatterpolar(
                    r=[df_ragas.loc[sys_name, cat] for cat in categories],
                    theta=categories,
                    fill='toself',
                    name=sys_name
                ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("RAGAS results not available.")

    with tab3:
        st.header("LLM Judge Matchups")
        if judge["matchups"]:
            for match in judge["matchups"]:
                with st.expander(f"Question {match['question_id']}: {match['system_a']} vs {match['system_b']} (Winner: {match['winner']})"):
                    st.write(f"**Winner:** {SYSTEM_DISPLAY_NAMES.get(match['winner'], match['winner'])}")
                    st.write("**Reasoning 1:**", match['reasoning_1'])
                    st.write("**Reasoning 2 (Swapped):**", match['reasoning_2'])
        else:
            st.info("LLM Judge results not available.")

    with tab4:
        st.header("Question Explorer")
        q_options = {f"{q['id']}: {q['question'][:50]}...": q for q in queries}
        selected_q_label = st.selectbox("Select a question:", list(q_options.keys()))
        
        if selected_q_label:
            selected_q = q_options[selected_q_label]
            st.markdown(f"**Question:** {selected_q['question']}")
            st.markdown(f"**Ground Truth:** {selected_q['ground_truth']}")
            st.markdown(f"**Type:** {selected_q['question_type']} | **Sources:** {', '.join(selected_q['sources_needed'])}")
            
            cols = st.columns(4)
            for i, (sys_name, sys_disp) in enumerate(SYSTEM_DISPLAY_NAMES.items()):
                with cols[i]:
                    st.subheader(sys_disp)
                    sys_data = selected_q['systems'].get(sys_name, {})
                    ans = sys_data.get('answer', 'N/A')
                    lat = sys_data.get('latency_ms', 0)
                    st.caption(f"Latency: {lat}ms")
                    st.write(ans)

if __name__ == "__main__":
    main()
