from __future__ import annotations

import os

import httpx
import streamlit as st

st.set_page_config(
    page_title="CT Scan MLOps",
    page_icon="🩻",
    layout="wide",
)


def _api_url() -> str:
    return os.environ.get("CT_SCAN_API_URL", "http://127.0.0.1:8000")


def _check_health(api_base_url: str) -> dict | None:
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{api_base_url.rstrip('/')}/health")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError:
        return None


def _predict(api_base_url: str, uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> dict:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    with httpx.Client(timeout=30) as client:
        response = client.post(f"{api_base_url.rstrip('/')}/predict", files=files)
        response.raise_for_status()
        return response.json()


def _send_feedback(
    api_base_url: str,
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
    predicted_class: str,
    is_correct: bool,
    correct_class: str | None,
) -> dict:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    data = {
        "predicted_class": predicted_class,
        "is_correct": str(is_correct).lower(),
    }
    if not is_correct and correct_class is not None:
        data["correct_class"] = correct_class

    with httpx.Client(timeout=30) as client:
        response = client.post(f"{api_base_url.rstrip('/')}/feedback", files=files, data=data)
        response.raise_for_status()
        return response.json()


st.markdown(
    """
<div style="padding: 12px 0 8px 0;">
    <h1 style="margin-bottom: 6px;">CT Scan MLOps</h1>
    <p style="font-size: 1.05rem; color: #6b7280;">Upload a chest CT scan and get an instant model prediction.</p>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Upload & Predict")
    st.caption("Accepted formats: PNG, JPG, JPEG")

    uploaded_file = st.file_uploader(
        "Drag and drop a CT scan image",
        type=["png", "jpg", "jpeg"],
        help="For best results, use a single-slice CT scan image.",
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Preview", use_container_width=True)

    st.divider()

    api_base_url = st.text_input(
        "API base URL",
        value=_api_url(),
        help="Runs against the FastAPI app in api.py. Example: http://127.0.0.1:8000",
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        run_prediction = st.button("Run prediction", type="primary", use_container_width=True)
    with col_b:
        check_api = st.button("Check API", use_container_width=True)

    if check_api:
        health = _check_health(api_base_url)
        if health is None:
            st.error("API unreachable. Start the FastAPI server and try again.")
        else:
            st.success("API is healthy.")
            st.json(health)

    if run_prediction:
        if uploaded_file is None:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Sending image to the model..."):
                try:
                    result = _predict(api_base_url, uploaded_file)
                    pred_class = result.get("pred_class", "Unknown")
                    pred_index = result.get("pred_index", "-")
                    st.session_state["last_prediction"] = {
                        "pred_class": pred_class,
                        "pred_index": pred_index,
                    }
                    st.success("Prediction complete")
                    st.metric("Predicted class", pred_class)
                    st.caption(f"Class index: {pred_index}")
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text
                    st.error(f"Prediction failed: {detail}")
                except httpx.HTTPError as exc:
                    st.error(f"Request error: {exc}")

    if "last_prediction" in st.session_state and uploaded_file is not None:
        st.divider()
        st.subheader("Feedback")
        st.caption("Help us validate predictions by confirming the correct class.")

        feedback_choice = st.radio(
            "Is the prediction correct?",
            options=["👍 Correct", "👎 Incorrect"],
            horizontal=True,
        )
        is_correct = feedback_choice == "👍 Correct"

        correct_class = None
        if not is_correct:
            correct_class = st.selectbox(
                "Select the correct class",
                options=[
                    "adenocarcinoma",
                    "large_cell_carcinoma",
                    "normal",
                    "squamous_cell_carcinoma",
                ],
            )

        if st.button("Submit feedback", use_container_width=True):
            pred_class = st.session_state["last_prediction"]["pred_class"]
            with st.spinner("Sending feedback..."):
                try:
                    response = _send_feedback(
                        api_base_url=api_base_url,
                        uploaded_file=uploaded_file,
                        predicted_class=pred_class,
                        is_correct=is_correct,
                        correct_class=correct_class,
                    )
                    st.success("Thanks for your feedback!")
                    st.caption(f"Saved to: {response.get('saved_to', 'unknown')}")
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text
                    st.error(f"Feedback failed: {detail}")
                except httpx.HTTPError as exc:
                    st.error(f"Request error: {exc}")

with right:
    st.subheader("How it works")
    st.markdown(
        """
        1. Upload a CT scan image.
        2. The app sends it to the FastAPI `/predict` endpoint.
        3. The model returns a classification label.
        """
    )

    st.subheader("Model classes")
    st.markdown(
        """
        - adenocarcinoma
        - large_cell_carcinoma
        - normal
        - squamous_cell_carcinoma
        """
    )

    st.subheader("Tips")
    st.markdown(
        """
        - Use clear, single-slice scans.
        - Ensure the API server is running before predicting.
        """
    )
