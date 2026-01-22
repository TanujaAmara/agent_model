import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from main1 import run_main_pipeline


st.set_page_config(page_title="Agent Orchestration", layout="wide")

st.title("🤖 Agent Orchestration App")
st.write("This app runs 3 agents:")

# User Input
user_input = st.text_input("Enter your topic / query:", placeholder="Example: Uses of AI in healthcare")

if st.button("Run Agents ✅"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a valid query.")
    else:
        with st.spinner("Running agents... Please wait ⏳"):
            result = run_main_pipeline(user_input)

        st.success("✅ Done! Here are the outputs:")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🔍 Research Output")
            st.write(result["Research Output"])

        with col2:
            st.subheader("📝 Summary Output")
            st.write(result["Summary Output"])

        with col3:
            st.subheader("📧 Email Output")
            st.write(result["Email Output"])
