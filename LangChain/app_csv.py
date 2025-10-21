import pandas as pd
import streamlit as st
import json
from datetime import datetime
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import re 
import os


os.environ["GOOGLE_API_KEY"] = ""

# ---------------------------
# APP CONFIG
# ---------------------------
st.set_page_config(page_title="Healthcare Call Transcript Analyzer", page_icon="📞", layout="wide")
st.title("📞 Healthcare Call Transcript Analyzer")

# ---------------------------
# LOAD DATA
# ---------------------------
uploaded_file = st.file_uploader("Upload call transcripts CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    required_cols = {"date", "primary_reason", "sub_reason", "reason_for_interaction", "transcript"}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must have columns: {', '.join(required_cols)}")
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ---------------------------
    # FILTERS
    # ---------------------------
    st.sidebar.header("🔎 Filters")

    min_date, max_date = df["date"].min(), df["date"].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

    primary_filter = st.sidebar.multiselect("Filter by Primary Reason", df["primary_reason"].unique())
    sub_filter = st.sidebar.multiselect("Filter by Sub Reason", df["sub_reason"].unique())
    keyword = st.sidebar.text_input("Search Keyword (optional)", "")

    # Apply filters
    mask = (df["date"] >= pd.to_datetime(date_range[0])) & (df["date"] <= pd.to_datetime(date_range[1]))
    if primary_filter:
        mask &= df["primary_reason"].isin(primary_filter)
    if sub_filter:
        mask &= df["sub_reason"].isin(sub_filter)
    filtered_df = df.loc[mask]
    if keyword:
        filtered_df = filtered_df[filtered_df["transcript"].str.contains(keyword, case=False, na=False)]

    st.write(f"### Showing {len(filtered_df)} filtered calls")
    st.dataframe(filtered_df.head(10))

    # ---------------------------
    # ANALYSIS SECTION
    # ---------------------------
    if len(filtered_df) > 0:
        st.subheader("📊 Choose Analysis Mode")
        mode = st.radio("Select Mode", ["Analyze Single Call", "Analyze All Filtered Calls"])

        if mode == "Analyze Single Call":
            selected_index = st.selectbox(
                "Select a call to analyze",
                filtered_df.index,
                format_func=lambda x: f"{filtered_df.loc[x, 'date'].date()} | {filtered_df.loc[x, 'primary_reason']} | {filtered_df.loc[x, 'transcript'][:60]}..."
            )
            selected_transcript = filtered_df.loc[selected_index, "transcript"]
            st.write(selected_transcript)

            if st.button("🔍 Analyze Selected Call"):
                with st.spinner("Analyzing call... ⏳"):
                    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
                    prompt = f"""
                    You are a healthcare insurance CX analyst.
                    Analyze this call transcript and extract:
                    1. Main topic (e.g., Claim denial, Billing issue, PCP update)
                    2. Customer's main pain point (1 line)
                    3. Sentiment (Positive, Neutral, Negative)
                    4. Suggested improvement (1 line)
                    Return a valid JSON only.

                    Transcript:
                    {selected_transcript}
                    """
                    response = model.invoke([HumanMessage(content=prompt)])
                    raw_output = response.content.strip()

                    # Clean and parse JSON
                    clean_output = re.sub(r"^```json|```$", "", raw_output, flags=re.MULTILINE).strip()
                    clean_output = "\n".join(
                        [line for line in clean_output.splitlines() if not line.strip().startswith("E0000")]
                    ).strip()
                    match = re.search(r"\{[\s\S]*\}", clean_output)
                    if not match:
                        st.error("⚠️ Could not find JSON in model output.")
                        st.text_area("Raw model output:", raw_output, height=300)
                        st.stop()

                    result = json.loads(match.group(0))
                    st.success("✅ Analysis Complete")

                    st.markdown("### 🧩 Topic")
                    st.write(result.get("main_topic", "N/A"))
                    st.markdown("### 💬 Pain Point")
                    st.write(result.get("customer_pain_point", "N/A"))
                    st.markdown("### 🙂 Sentiment")
                    st.write(result.get("sentiment", "N/A"))
                    st.markdown("### 💡 Suggested Improvement")
                    st.write(result.get("suggested_improvement", "N/A"))
                    with st.expander("🧾 Full JSON Output"):
                        st.json(result)

        elif mode == "Analyze All Filtered Calls":
            if st.button("🚀 Analyze All Calls"):
                all_results = []
                model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

                for idx, row in filtered_df.iterrows():
                    with st.spinner(f"Analyzing call on {row['date'].date()}..."):
                        prompt = f"""
                        You are a healthcare insurance CX analyst.
                        Analyze this call transcript and extract:
                        1. Main topic (e.g., Claim denial, Billing issue, PCP update)
                        2. Customer's main pain point (1 line)
                        3. Sentiment (Positive, Neutral, Negative)
                        4. Suggested improvement (1 line)
                        Return a valid JSON only.

                        Transcript:
                        {row['transcript']}
                        """
                        response = model.invoke([HumanMessage(content=prompt)])
                        raw_output = response.content.strip()
                        clean_output = re.sub(r"^```json|```$", "", raw_output, flags=re.MULTILINE).strip()
                        match = re.search(r"\{[\s\S]*\}", clean_output)
                        if match:
                            result = json.loads(match.group(0))
                            result["date"] = row["date"]
                            result["primary_reason"] = row["primary_reason"]
                            all_results.append(result)

                results_df = pd.DataFrame(all_results)
                st.success("✅ Batch Analysis Complete")
                st.dataframe(results_df)

                # Summary Dashboard
                st.subheader("📈 Summary Insights")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Top Call Topics**")
                    st.bar_chart(results_df["main_topic"].value_counts())

                with col2:
                    st.markdown("**Sentiment Distribution**")
                    st.bar_chart(results_df["sentiment"].value_counts())

                with st.expander("📋 View Full Results JSON"):
                    st.json(all_results)
