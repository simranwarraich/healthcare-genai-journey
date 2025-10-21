import snowflake.connector
import pandas as pd
import streamlit as st
import json
from datetime import datetime
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage

import os

os.environ["GOOGLE_API_KEY"] = ""

# ---------------------------
# APP CONFIG
# ---------------------------
st.set_page_config(page_title="Call Transcript Insights", page_icon="📞", layout="wide")
st.title("📞 Healthcare Call Transcript Analyzer")

# ---------------------------
# CONNECT TO SNOWFLAKE
# ---------------------------
conn = snowflake.connector.connect(
    user = "",
    password = "",
    account = "",
    warehouse = "",
    database = "",
    schema = "PUBLIC",
    role = "ACCOUNTADMIN"
)

query = """
SELECT CC_NAME, CC_REC_START_DATE, CC_MKT_DESC
FROM tpcds_10tb.tpcds_sf10tcl.call_center
"""

df = pd.read_sql(query, conn)
df["CC_REC_START_DATE"] = pd.to_datetime(df["CC_REC_START_DATE"])

st.sidebar.header("🔎 Filters")

# Time frame filter
min_date = df["CC_REC_START_DATE"].min()
max_date = df["CC_REC_START_DATE"].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

# Keyword filter
keyword = st.sidebar.text_input("Search Keyword (optional)", "")

# Apply filters
mask = (df["CC_REC_START_DATE"] >= pd.to_datetime(date_range[0])) & (df["CC_REC_START_DATE"] <= pd.to_datetime(date_range[1]))
filtered_df = df.loc[mask]

if keyword:
    filtered_df = filtered_df[filtered_df["CC_MKT_DESC"].str.contains(keyword, case=False, na=False)]

st.write(f"### Showing {len(filtered_df)} filtered calls")
st.dataframe(filtered_df[["CC_REC_START_DATE", "CC_MKT_DESC"]].head(10))

# ---------------------------
# ANALYZE SELECTED CALL
# ---------------------------
if len(filtered_df) > 0:
    selected_index = st.selectbox("Select a call to analyze", filtered_df.index, 
                                    format_func=lambda x: f"{filtered_df.loc[x, 'CC_REC_START_DATE'].date()} | {filtered_df.loc[x, 'CC_MKT_DESC'][:60]}...")
    
    selected_transcript = filtered_df.loc[selected_index, "CC_MKT_DESC"]
    
    st.subheader("📜 Selected Transcript")
    st.write(selected_transcript)
    
    if st.button("🔍 Analyze Selected Call"):
        with st.spinner("Analyzing call using Gemini... ⏳"):
            model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

            prompt = f"""
            You are an insurance customer experience analyst.
            Read the following call transcript and extract:
            1. Main topic (e.g., Claim delay, Billing issue, Policy query, Hospital issue)
            2. Customer's pain point (one line)
            3. Sentiment (Positive, Neutral, Negative)
            4. Suggested improvement (one line)
            Return your answer in **pure JSON only**, with keys:
            topic, pain_point, sentiment, suggestion

            Transcript:
            {selected_transcript}
            """
            try:
                response = model.invoke([HumanMessage(content=prompt)])
                print ("Raw model output:", response)

                raw_output = response.content.strip()
                
                # 🧹 Extract JSON block from model output (ignore markdown or text)
                match = re.search(r"\{.*\}", raw_output, re.DOTALL)
                if match:
                    clean_json = match.group(0)
                    result = json.loads(clean_json)
                else:
                    st.error("Model did not return valid JSON output.")
                    st.write("Raw output:", raw_output)
                    st.stop()

                st.success("✅ Analysis Complete")
                st.markdown(f"**🧩 Topic:** {result.get('topic', 'N/A')}")
                st.markdown(f"**💬 Pain Point:** {result.get('pain_point', 'N/A')}")
                st.markdown(f"**🙂 Sentiment:** {result.get('sentiment', 'N/A')}")
                st.markdown(f"**💡 Suggested Improvement:** {result.get('suggestion', 'N/A')}")

            except Exception as e:
                st.error(f"Error: {e}")