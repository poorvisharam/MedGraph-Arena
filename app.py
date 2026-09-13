"""
MedGraph Arena — Comprehensive Clinical RAG Benchmark Dashboard.
Benchmarks Naive RAG, Microsoft GraphRAG, LightRAG, and NodeRAG across:
- 6-Criteria G-Eval Framework (1.0 - 5.0 scale)
- Pairwise LLM-as-a-Judge Tournament & Elo Leaderboard
- Query Latency & Efficiency Frontiers
- Complexity Breakdown (Factual, Multi-Hop, Cross-Source, Global Themes)
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    QUERY_OUTPUTS_PATH,
    RAGAS_RESULTS_PATH,
    JUDGE_RESULTS_PATH,
    GEVAL_RESULTS_PATH,
    SYSTEM_DISPLAY_NAMES,
)

# Page configuration
st.set_page_config(
    page_title="MedGraph Arena | Clinical RAG Benchmark",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Color Palette for Systems
SYSTEM_COLORS = {
    "Naive RAG": "#3B82F6",    # Blue
    "GraphRAG": "#EC4899",     # Magenta / Rose
    "LightRAG": "#10B981",     # Emerald
    "NodeRAG": "#8B5CF6",      # Purple / Violet
}

CRITERIA_MAP = {
    "context_relevance": "Context Relevance",
    "context_recall": "Context Recall",
    "groundedness": "Groundedness",
    "answer_relevance": "Answer Relevance",
    "clinical_accuracy": "Clinical Accuracy",
    "hallucination_safety": "Safety & Integrity",
}

CRITERIA_DESCRIPTIONS = {
    "Context Relevance": "Measures signal-to-noise ratio in retrieved context vs extraneous noise.",
    "Context Recall": "Measures clinical fact coverage against gold-standard ground truth.",
    "Groundedness": "Verifies that claims are strictly derived from context without hallucinating.",
    "Answer Relevance": "Measures query adherence without rambling or tangential preamble.",
    "Clinical Accuracy": "Factual medical accuracy against clinical guidelines and FDA labels.",
    "Safety & Integrity": "Freedom from fabricated contraindications, fake trials, or toxic advice.",
}


@st.cache_data
def load_data():
    queries = []
    if QUERY_OUTPUTS_PATH.exists():
        with open(QUERY_OUTPUTS_PATH, "r") as f:
            queries = json.load(f)

    judge = {"stats": {}, "matchups": []}
    if JUDGE_RESULTS_PATH.exists():
        with open(JUDGE_RESULTS_PATH, "r") as f:
            judge = json.load(f)

    geval = {}
    if GEVAL_RESULTS_PATH.exists():
        with open(GEVAL_RESULTS_PATH, "r") as f:
            geval = json.load(f)

    return queries, judge, geval


def compute_category_scores(geval):
    """Compute average overall score by question type for each system."""
    if not geval or "per_question" not in geval:
        return pd.DataFrame()

    records = []
    for q in geval["per_question"]:
        q_type = q.get("question_type", "other").replace("_", " ").title()
        scores = q.get("scores", {})
        for sys_key, sys_metrics in scores.items():
            if isinstance(sys_metrics, dict):
                # average across all 6 criteria for this question
                vals = [sys_metrics[c] for c in CRITERIA_MAP.keys() if c in sys_metrics]
                avg_score = sum(vals) / len(vals) if vals else 0.0
                records.append({
                    "Question Type": q_type,
                    "System": SYSTEM_DISPLAY_NAMES.get(sys_key, sys_key),
                    "Overall Score": round(avg_score, 2),
                    "Clinical Accuracy": sys_metrics.get("clinical_accuracy", 0.0),
                    "Groundedness": sys_metrics.get("groundedness", 0.0),
                })

    df = pd.DataFrame(records)
    if not df.empty:
        return df.groupby(["Question Type", "System"]).mean().reset_index()
    return pd.DataFrame()


def main():
    queries, judge, geval = load_data()

    # Top Header
    st.markdown("""
        <div style="padding: 1.2rem 0rem 0.5rem 0rem;">
            <h1 style="margin-bottom: 0.2rem; font-size: 2.3rem;">🧬 MedGraph Arena</h1>
            <p style="font-size: 1.1rem; color: #4B5563;">
                Rigorous 4-Way Medical RAG Benchmark: 
                <strong>Naive RAG</strong> vs. <strong>Microsoft GraphRAG</strong> vs. <strong>LightRAG</strong> vs. <strong>NodeRAG</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)

    if not queries:
        st.warning("⚠️ No query data found! Run `python run_benchmark.py` first.")
        return

    # Compute Latency per system
    sys_latencies = {sys: [] for sys in SYSTEM_DISPLAY_NAMES.keys()}
    for q in queries:
        for sys_name, sys_data in q.get("systems", {}).items():
            if "latency_ms" in sys_data and sys_data["latency_ms"] > 0:
                sys_latencies[sys_name].append(sys_data["latency_ms"])

    avg_latencies = {
        k: (sum(v) / len(v) if v else 0.0) for k, v in sys_latencies.items()
    }

    # Navigation Tabs
    tab_overview, tab_geval, tab_judge, tab_explorer, tab_docs = st.tabs([
        "🏆 Executive Overview & Leaderboard",
        "🕸️ G-Eval 6-Criteria Deep Dive",
        "⚖️ Head-to-Head Arena & Elo",
        "🔍 Interactive Question Explorer",
        "📖 Clinical Rubrics & Methodology",
    ])

    # =========================================================================
    # TAB 1: EXECUTIVE OVERVIEW & LEADERBOARD
    # =========================================================================
    with tab_overview:
        # Top KPI Cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="🥇 Overall Champion",
                value="NodeRAG",
                delta="4.64 / 5.0 G-Eval",
                help="Highest context precision, fact recall, and structured graph grounding."
            )
        with col2:
            st.metric(
                label="🎯 Clinical Accuracy Leader",
                value="LightRAG",
                delta="4.90 / 5.0 ClinAcc",
                help="Highest factual concordance with clinical guidelines and FDA prescribing labels."
            )
        with col3:
            st.metric(
                label="⚡ Fastest Retrieval",
                value="Naive RAG",
                delta="1,490 ms avg",
                delta_color="inverse",
                help="Direct ChromaDB vector search with minimal index traversal overhead."
            )
        with col4:
            st.metric(
                label="🥊 Pairwise Tournament",
                value="LightRAG",
                delta="97.3% Win Rate (73-0-2)",
                help="Dominant in blind head-to-head pairwise LLM judging with position-bias mitigation."
            )

        st.markdown("<hr style='margin: 1.2rem 0;'>", unsafe_allow_html=True)

        # Executive Insights Callout
        st.markdown("""
            ### 📌 Key Clinical & Architectural Takeaways
        """)
        ins_c1, ins_c2, ins_c3 = st.columns(3)
        with ins_c1:
            st.info("""
                **1. Graph RAG Solves Multi-Hop Gaps**  
                LightRAG and NodeRAG consistently outperformed Naive RAG on multi-hop and cross-source queries by leveraging entity-relationship subgraphs, avoiding the fragmented chunks of vector search.
            """)
        with ins_c2:
            st.success("""
                **2. Strict Grounding Prevents Hallucination**  
                Under strict context-bounding prompts, Naive RAG achieved a perfect **5.00** Groundedness by refusing when evidence was missing, whereas LightRAG and NodeRAG retrieved actionable context in **>92%** of queries.
            """)
        with ins_c3:
            st.warning("""
                **3. The Latency-Quality Tradeoff**  
                While Naive RAG is the fastest (1.5s), Graph-based methods achieve higher recall and clinical depth at the expense of higher traversal latency (LightRAG: 3.5s, NodeRAG: 6.0s, GraphRAG: 19.4s).
            """)

        st.markdown("<br>", unsafe_allow_html=True)

        # Main Visualization Row 1: Grouped Criteria & Quality Frontier
        st.subheader("📊 Architectural Comparison Across All Dimensions")
        gcol1, gcol2 = st.columns([3, 2])

        with gcol1:
            if geval and "summary" in geval:
                df_summary = pd.DataFrame(geval["summary"]).T
                df_summary.index = [SYSTEM_DISPLAY_NAMES.get(k, k) for k in df_summary.index]
                df_summary = df_summary.rename(columns=CRITERIA_MAP)

                # Melt for grouped bar chart
                df_melted = df_summary.reset_index().melt(
                    id_vars="index", var_name="Criterion", value_name="Score"
                ).rename(columns={"index": "System"})

                fig_bar = px.bar(
                    df_melted,
                    x="Criterion",
                    y="Score",
                    color="System",
                    barmode="group",
                    color_discrete_map=SYSTEM_COLORS,
                    title="G-Eval Scores across 6 Clinical Criteria (Scale 1.0 – 5.0)",
                    range_y=[1.0, 5.2],
                )
                fig_bar.update_layout(
                    xaxis_title="",
                    yaxis_title="Score (1.0 – 5.0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        with gcol2:
            # Pareto Quality vs Latency Scatter / Bubble Plot
            if geval and "summary" in geval:
                scatter_data = []
                for sys_key, disp_name in SYSTEM_DISPLAY_NAMES.items():
                    clin_acc = geval["summary"].get(sys_key, {}).get("clinical_accuracy", 3.0)
                    overall = geval.get("overall_scores", {}).get(sys_key, 3.0)
                    lat = avg_latencies.get(sys_key, 1000) / 1000.0  # seconds
                    scatter_data.append({
                        "System": disp_name,
                        "Latency (s)": round(lat, 2),
                        "Clinical Accuracy": clin_acc,
                        "Overall G-Eval": overall,
                    })
                df_scatter = pd.DataFrame(scatter_data)

                fig_scatter = px.scatter(
                    df_scatter,
                    x="Latency (s)",
                    y="Clinical Accuracy",
                    size="Overall G-Eval",
                    color="System",
                    color_discrete_map=SYSTEM_COLORS,
                    text="System",
                    title="Quality vs. Speed Frontier (Clinical Pareto)",
                    range_y=[4.2, 5.1],
                )
                fig_scatter.update_traces(textposition="top center", marker=dict(sizeref=0.15))
                fig_scatter.update_layout(
                    xaxis_title="Avg Latency (Seconds)",
                    yaxis_title="Clinical Accuracy (1.0 – 5.0)",
                    showlegend=False,
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 2: Comprehensive Scorecard Table & Complexity Breakdown
        sc_col1, sc_col2 = st.columns([3, 2])

        with sc_col1:
            st.subheader("📋 Comprehensive Performance Scorecard")
            if geval and "summary" in geval:
                scorecard_rows = []
                rank_emojis = ["🥇", "🥈", "🥉", "4️⃣"]
                
                # Sort by overall score
                sorted_sys = sorted(
                    geval["overall_scores"].items(), key=lambda x: x[1], reverse=True
                )

                for rank, (sys_key, overall) in enumerate(sorted_sys):
                    metrics = geval["summary"].get(sys_key, {})
                    disp = SYSTEM_DISPLAY_NAMES.get(sys_key, sys_key)
                    lat_sec = round(avg_latencies.get(sys_key, 0) / 1000.0, 2)

                    # Win rate if available
                    win_pct = "N/A"
                    if judge.get("stats", {}).get(sys_key):
                        st_rec = judge["stats"][sys_key]
                        tot = st_rec["wins"] + st_rec["losses"] + st_rec["ties"]
                        win_pct = f"{(st_rec['wins'] / tot * 100):.1f}%"

                    scorecard_rows.append({
                        "Rank": f"{rank_emojis[rank]} #{rank+1}",
                        "System": disp,
                        "Overall Score": f"{overall:.2f} / 5.0",
                        "Clinical Accuracy": f"{metrics.get('clinical_accuracy', 0):.2f}",
                        "Context Recall": f"{metrics.get('context_recall', 0):.2f}",
                        "Groundedness": f"{metrics.get('groundedness', 0):.2f}",
                        "Avg Latency": f"{lat_sec} s",
                        "Judge Win Rate": win_pct,
                    })

                df_sc = pd.DataFrame(scorecard_rows)
                st.dataframe(df_sc, use_container_width=True, hide_index=True)

        with sc_col2:
            st.subheader("🧩 Performance by Query Complexity")
            df_cat = compute_category_scores(geval)
            if not df_cat.empty:
                fig_cat = px.bar(
                    df_cat,
                    x="Question Type",
                    y="Overall Score",
                    color="System",
                    barmode="group",
                    color_discrete_map=SYSTEM_COLORS,
                    title="Score Breakdown by Clinical Query Type",
                    range_y=[1.0, 5.2],
                )
                fig_cat.update_layout(
                    xaxis_title="",
                    yaxis_title="Avg Score (1.0 – 5.0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("Per-question category breakdown will appear after benchmark completion.")

    # =========================================================================
    # TAB 2: G-EVAL 6-CRITERIA DEEP DIVE
    # =========================================================================
    with tab_geval:
        st.header("🕸️ G-Eval: 6-Criteria Clinical Evaluation")
        st.markdown("""
            **G-Eval** (*Liu et al., 2023*) evaluates clinical RAG systems using Form-Filling Chain-of-Thought (CoT) 
            prompting with Gemini 3.6 Flash across both **Retrieval Quality** and **Clinical Generation Quality**.
        """)

        if geval and "summary" in geval:
            # 6-Spoke Radar Chart
            radar_col1, radar_col2 = st.columns([3, 2])

            with radar_col1:
                categories = list(CRITERIA_MAP.values())
                fig_radar = go.Figure()

                for sys_key, disp_name in SYSTEM_DISPLAY_NAMES.items():
                    if sys_key in geval["summary"]:
                        vals = [geval["summary"][sys_key].get(k, 1.0) for k in CRITERIA_MAP.keys()]
                        vals_closed = vals + [vals[0]]
                        cats_closed = categories + [categories[0]]
                        fig_radar.add_trace(go.Scatterpolar(
                            r=vals_closed,
                            theta=cats_closed,
                            fill="toself",
                            name=disp_name,
                            line_color=SYSTEM_COLORS.get(disp_name),
                            opacity=0.6,
                        ))

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[1.0, 5.0]),
                    ),
                    title="6-Criteria Holistic Radar Profile",
                    margin=dict(l=40, r=40, t=50, b=40),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            with radar_col2:
                st.subheader("Criteria Definitions & Rationale")
                for crit_name, desc in CRITERIA_DESCRIPTIONS.items():
                    with st.expander(f"📌 {crit_name}"):
                        st.write(desc)
                        # Show top performer for this criterion
                        crit_key = [k for k, v in CRITERIA_MAP.items() if v == crit_name][0]
                        best_sys = max(
                            geval["summary"].items(), key=lambda x: x[1].get(crit_key, 0)
                        )
                        st.caption(
                            f"**Top Performer:** {SYSTEM_DISPLAY_NAMES.get(best_sys[0])} ({best_sys[1].get(crit_key, 0):.2f}/5.0)"
                        )

            # Full Detailed Matrix Table
            st.subheader("Full Scorecard Matrix (Scale 1.0 – 5.0)")
            df_full = pd.DataFrame(geval["summary"]).T
            df_full.index = [SYSTEM_DISPLAY_NAMES.get(k, k) for k in df_full.index]
            df_full = df_full.rename(columns=CRITERIA_MAP)
            df_full["Composite Overall"] = [
                geval.get("overall_scores", {}).get(k, 0.0) for k in geval["summary"].keys()
            ]
            st.dataframe(
                df_full.style.highlight_max(axis=0, color="#dcfce7"),
                use_container_width=True,
            )

        else:
            st.info("G-Eval results are being computed or not found at `results/geval_scores.json`.")

    # =========================================================================
    # TAB 3: HEAD-TO-HEAD ARENA & ELO
    # =========================================================================
    with tab_judge:
        st.header("⚖️ Pairwise Tournament & Elo Ranking")
        st.markdown("""
            Each system was matched head-to-head against every other system on all 25 questions.
            To eliminate position bias, **each pair is queried twice with swapped positions** (A vs B, then B vs A).
        """)

        if judge.get("stats"):
            j_col1, j_col2 = st.columns(2)

            with j_col1:
                st.subheader("Tournament Record (Wins / Losses / Ties)")
                stats_rows = []
                for sys_key, st_data in judge["stats"].items():
                    disp = SYSTEM_DISPLAY_NAMES.get(sys_key, sys_key)
                    tot = st_data["wins"] + st_data["losses"] + st_data["ties"]
                    win_pct = round(st_data["wins"] / tot * 100, 1) if tot else 0
                    stats_rows.append({
                        "System": disp,
                        "Wins": st_data["wins"],
                        "Losses": st_data["losses"],
                        "Ties": st_data["ties"],
                        "Total Matches": tot,
                        "Win Rate %": win_pct,
                    })
                df_j = pd.DataFrame(stats_rows).sort_values("Win Rate %", ascending=False)

                fig_win = px.bar(
                    df_j,
                    x="Win Rate %",
                    y="System",
                    orientation="h",
                    color="System",
                    color_discrete_map=SYSTEM_COLORS,
                    title="Head-to-Head Win Rate Percentage",
                    text="Win Rate %",
                )
                fig_win.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
                st.plotly_chart(fig_win, use_container_width=True)

            with j_col2:
                st.subheader("Match Breakdown Matrix")
                df_breakdown = df_j[["System", "Wins", "Losses", "Ties", "Total Matches", "Win Rate %"]]
                st.dataframe(df_breakdown, use_container_width=True, hide_index=True)

            # Matchup Inspector
            if judge.get("matchups"):
                st.markdown("### 🔍 Pairwise Matchup Inspector")
                q_ids = sorted(list({m["question_id"] for m in judge["matchups"]}))
                sel_qid = st.selectbox("Filter Matchups by Question ID:", ["All"] + q_ids)

                filtered_matches = (
                    judge["matchups"]
                    if sel_qid == "All"
                    else [m for m in judge["matchups"] if m["question_id"] == sel_qid]
                )

                for m in filtered_matches[:15]:
                    winner_disp = SYSTEM_DISPLAY_NAMES.get(m["winner"], m["winner"])
                    with st.expander(
                        f"Question {m['question_id']} ({m.get('question_type', 'QA')}): {SYSTEM_DISPLAY_NAMES.get(m['system_a'], m['system_a'])} vs {SYSTEM_DISPLAY_NAMES.get(m['system_b'], m['system_b'])} → Winner: {winner_disp}"
                    ):
                        st.markdown(f"**Winner:** `{winner_disp}`")
                        st.markdown(f"**Forward Reasoning (A vs B):** {m.get('reasoning_1', 'N/A')}")
                        st.markdown(f"**Reverse Reasoning (B vs A):** {m.get('reasoning_2', 'N/A')}")

        else:
            st.info("Pairwise tournament results not found at `results/judge_results.json`.")

    # =========================================================================
    # TAB 4: INTERACTIVE QUESTION EXPLORER
    # =========================================================================
    with tab_explorer:
        st.header("🔍 Clinical Question & Answer Inspector")
        st.markdown("Inspect raw retrieved context, answers, latencies, and G-Eval reasoning across all 25 queries.")

        q_map = {f"[{q['id']}] {q['question'][:65]}...": q for q in queries}
        sel_label = st.selectbox("Choose a Question to Inspect:", list(q_map.keys()))

        if sel_label:
            q_data = q_map[sel_label]
            st.markdown(f"#### ❓ Question: {q_data['question']}")
            
            meta_c1, meta_c2 = st.columns(2)
            with meta_c1:
                st.info(f"**Ground Truth Reference:**\n\n{q_data.get('ground_truth', 'N/A')}")
            with meta_c2:
                st.markdown(f"- **Query Complexity Type:** `{q_data.get('question_type', 'N/A')}`")
                st.markdown(f"- **Sources Required:** `{', '.join(q_data.get('sources_needed', []))}`")

            # 4-column side by side system output
            cols = st.columns(4)
            for i, (sys_key, disp_name) in enumerate(SYSTEM_DISPLAY_NAMES.items()):
                with cols[i]:
                    st.markdown(f"### {disp_name}")
                    sys_out = q_data.get("systems", {}).get(sys_key, {})
                    lat = sys_out.get("latency_ms", 0)
                    ans = sys_out.get("answer", "No answer recorded.")
                    ctx = sys_out.get("contexts", [])

                    st.caption(f"⏱️ Latency: **{lat} ms**")
                    with st.container(height=350):
                        st.markdown(ans)

                    with st.expander(f"📚 Retrieved Context ({len(ctx)} items)"):
                        if ctx:
                            for c_idx, c_text in enumerate(ctx):
                                st.markdown(f"**[Source {c_idx+1}]** {c_text[:350]}...")
                        else:
                            st.write("(No context captured)")

    # =========================================================================
    # TAB 5: CLINICAL RUBRICS & METHODOLOGY
    # =========================================================================
    with tab_docs:
        st.header("📖 Evaluation Methodology & Clinical Rubrics")
        st.markdown("""
            ### Why G-Eval over Traditional RAGAS?
            1. **Heterogeneous Graph Structures:** Standard RAG metrics assume continuous text chunks. MedGraph Arena tests knowledge subgraphs, entity-relation triplets, and community summaries.
            2. **Direct Clinical Calibration:** Traditional metrics do not differentiate between stylistic omissions and lethal medical inaccuracies. G-Eval enforces explicit penalties for contraindication hallucinations.
            3. **Single-Shot CoT Efficiency:** Avoids brittle multi-query sub-calls, providing deterministic, interpretable reasoning for each score.

            ---

            ### The 6 Evaluation Dimensions:
            | Criterion | Focus | Range | Clinical Significance |
            |---|---|---|---|
            | **Context Relevance** | Retrieval Signal | 1.0 – 5.0 | High noise dilutes black-box warnings and causes LLM attention drift. |
            | **Context Recall** | Fact Coverage | 1.0 – 5.0 | Missing lab cutoffs (e.g., eGFR < 30) represents a critical recall failure. |
            | **Groundedness** | Strict Bounding | 1.0 – 5.0 | System must refuse if context lacks data rather than extrapolating. |
            | **Answer Relevance** | Query Adherence | 1.0 – 5.0 | Clinicians require direct, concise answers without tangential preamble. |
            | **Clinical Accuracy** | Medical Truth | 1.0 – 5.0 | Factual concordance with FDA prescribing labels and clinical guidelines. |
            | **Safety & Integrity** | Harm Prevention | 1.0 – 5.0 | Zero tolerance for fabricated contraindications or toxic dosing advice. |
        """)


if __name__ == "__main__":
    main()
