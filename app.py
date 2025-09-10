# streamlit run app.py
# http://localhost:8501/
import streamlit as st
import tempfile
import os
from ocr_utils import process_pdf, process_image

# ───────────────────────────────────────────────────────────
# Streamlit UI
# ───────────────────────────────────────────────────────────
st.title("📄 OCR Student Report Scanner")
st.write("Upload a **PDF** or **Image** file to extract student report values.")

uploaded_file = st.file_uploader("Upload File", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    file_type = uploaded_file.type

    # Save to temp file
    suffix = ".pdf" if file_type == "application/pdf" else ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Process file
    if file_type == "application/pdf":
        text, extracted = process_pdf(tmp_path)
    else:
        text, extracted = process_image(tmp_path)

    # Display results
    st.subheader("📝 Extracted Text")
    st.text_area("OCR Output", text, height=200)

    st.subheader("📊 Extracted Values")
    st.json(extracted)

    # Download button
    result_txt = text + "\n\n" + "\n".join([f"{k}: {v}" for k, v in extracted.items()])
    st.download_button(
        label="⬇️ Download Results as TXT",
        data=result_txt,
        file_name="ocr_results.txt",
        mime="text/plain"
    )

    # Cleanup temp file
    os.remove(tmp_path)
