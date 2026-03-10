import streamlit as st

st.markdown("""
<style>

/* Hide Streamlit default multipage navigation */
[data-testid="stSidebarNav"] {
    display: none;
}

/* Optional: hide the divider line above it */
[data-testid="stSidebarNavSeparator"] {
    display: none;
}

</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Teaching Demo",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.big-icon {
    font-size: 56px;
    text-align: center;
    line-height: 1.1;
    margin-bottom: 6px;
}
.big-label {
    text-align: center;
    font-weight: 700;
    font-size: 18px;
    margin-bottom: 12px;
}
.section-card {
    background: #ffffff;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
}
.footer-text {
    text-align: center;
    color: #6b7280;
    font-size: 14px;
    padding-top: 8px;
    padding-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# 🎓 Teaching Demo")

    st.markdown("### 👨‍🏫 Lecturer")
    st.write("**Bui Tien Duc**")
    st.write("📞 0769690731")

    st.divider()

    st.markdown("### 📚 Lesson Structure")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="big-icon">🖥️</div>', unsafe_allow_html=True)
        st.markdown('<div class="big-label">APP</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="big-icon">📊</div>', unsafe_allow_html=True)
        st.markdown('<div class="big-label">DATA MINING</div>', unsafe_allow_html=True)

    st.page_link("app.py", label="Computerization", icon="🖥️")
    st.page_link("pages/data_mining.py", label="Data Mining", icon="📊")

    st.divider()

    st.markdown("### 🎯 Teaching Goal")
    st.info(
        "Show how computerization supports daily work and how Data Mining discovers useful knowledge from accumulated data."
    )

st.title("🎓 Teaching Demo (30 Minutes)")
st.subheader("🖥️ Computerization → 📊 Data Mining")

st.divider()

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("🖥️ Computerization")

st.write("""
Computerization means applying computer systems to support work more efficiently.
Organizations can store data, process information quickly, reduce manual effort, and improve decision-making.
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🛒 Example: Laptop E-Commerce Store")

st.write("Every day, the system collects and stores data such as:")

st.markdown("""
- 👤 Customer information  
- 💻 Product information  
- 🧾 Orders and transactions  
- 🌐 Website visits  
- 💰 Sales values  
""")

st.write("""
As the business grows, the amount of data increases rapidly.
Manual management becomes inefficient.

Therefore, computerization is necessary to store and manage data effectively.
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("📊 From Computerization to Data Mining")

st.write("""
Once data has been accumulated in the system, we can apply Data Mining techniques to discover:
""")

st.markdown("""
- hidden patterns  
- customer behavior  
- useful trends  
- knowledge for decision support  
""")

st.info("➡️ Next step: click Data Mining in the sidebar.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer-text">🎓 Teaching Demo – Interview Lecture | Bui Tien Duc</div>',
    unsafe_allow_html=True
)