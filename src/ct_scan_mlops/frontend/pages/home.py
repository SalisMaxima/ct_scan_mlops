from __future__ import annotations

import base64
import io
import os

import httpx
import pandas as pd
import streamlit as st
from PIL import Image


def render() -> None:
    st.set_page_config(
        page_title="CT Scan MLOps - Medical Imaging AI",
        page_icon="🩻",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Custom CSS for enhanced styling
    st.markdown(
        """
        <style>
        /* Main title styling */
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        /* Subtitle styling */
        .subtitle {
            font-size: 1.1rem;
            color: #6b7280;
            margin-bottom: 2rem;
        }

        /* Confidence indicator */
        .confidence-high {
            background: #10b981;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: 600;
            display: inline-block;
        }

        .confidence-medium {
            background: #f59e0b;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: 600;
            display: inline-block;
        }

        .confidence-low {
            background: #ef4444;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: 600;
            display: inline-block;
        }

        /* Info box styling */
        .info-box {
            background: #f3f4f6;
            border-left: 4px solid #667eea;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }

        /* Medical disclaimer */
        .disclaimer {
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            font-size: 0.9rem;
        }

        /* Prediction card */
        .prediction-card {
            background: white;
            border: 2px solid #e5e7eb;
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        /* Class badge */
        .class-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            background: #667eea;
            color: white;
            border-radius: 2rem;
            font-weight: 600;
            font-size: 1.1rem;
            margin: 0.5rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        """
        <div style="padding: 12px 0 8px 0;">
            <h1 class="main-title">🩻 CT Scan MLOps</h1>
            <p class="subtitle">Advanced AI-powered lung cancer classification from chest CT scans</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Medical disclaimer
    st.markdown(
        """
        <div class="disclaimer">
            <strong>⚠️ Medical Research Tool</strong><br>
            This is an experimental AI system for research and educational purposes only.
            All predictions must be validated by qualified medical professionals.
            Not approved for clinical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main layout
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        st.markdown("### 📤 Upload & Analyze")

        # API Configuration
        with st.expander("⚙️ API Configuration", expanded=False):
            api_base_url = st.text_input(
                "API Endpoint",
                value=_api_url(),
                help="FastAPI server URL. Default: http://127.0.0.1:8000",
            )

            col_check1, col_check2 = st.columns(2)
            with col_check1:
                if st.button("🔍 Check API Health", use_container_width=True):
                    with st.spinner("Checking API..."):
                        health = _check_health(api_base_url)
                        if health is None:
                            st.error("❌ API unreachable. Please start the FastAPI server.")
                        else:
                            st.success("✅ API is healthy and ready")
                            with st.expander("View API Details"):
                                st.json(health)

            with col_check2:
                if st.button("📊 View Feedback Stats", use_container_width=True):
                    with st.spinner("Loading feedback statistics..."):
                        stats = _get_feedback_stats(api_base_url)
                        if stats:
                            st.success("✅ Feedback statistics loaded")
                            with st.expander("View Statistics"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Feedback", stats.get("total_feedback", 0))
                                with col2:
                                    st.metric(
                                        "Accuracy",
                                        f"{stats.get('accuracy', 0):.1%}",
                                    )
                                with col3:
                                    st.metric("Correct", stats.get("correct_predictions", 0))

        st.divider()

        # File upload
        uploaded_file = st.file_uploader(
            "**Select CT Scan Image**",
            type=["png", "jpg", "jpeg"],
            help="Upload a chest CT scan for classification",
        )

        if uploaded_file is not None:
            # Display image preview
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded CT Scan", use_container_width=True)

            # Image info
            st.caption(f"Image: {uploaded_file.name} | Size: {image.size[0]}x{image.size[1]} | Mode: {image.mode}")

            st.divider()

            # Prediction buttons
            col_pred1, col_pred2 = st.columns(2)

            with col_pred1:
                run_prediction = st.button(
                    "🔬 Run Classification",
                    type="primary",
                    use_container_width=True,
                    help="Send image to AI model for analysis",
                )

            with col_pred2:
                run_explanation = st.button(
                    "🔍 Get Explanation",
                    use_container_width=True,
                    help="Generate visual explanation of model decision",
                    disabled="last_prediction" not in st.session_state,
                )

            # Run prediction
            if run_prediction:
                with st.spinner("🤖 Analyzing CT scan..."):
                    try:
                        result = _predict(api_base_url, uploaded_file)
                        prediction = result.get("prediction", {})
                        probabilities = result.get("probabilities", {})
                        metadata = result.get("metadata", {})

                        # Store in session state
                        st.session_state["last_prediction"] = prediction
                        st.session_state["last_probabilities"] = probabilities
                        st.session_state["last_metadata"] = metadata
                        st.session_state["last_file"] = uploaded_file

                        st.rerun()

                    except httpx.HTTPStatusError as exc:
                        st.error(f"❌ Prediction failed: {exc.response.text}")
                    except httpx.HTTPError as exc:
                        st.error(f"❌ Network error: {exc}")

            # Run explanation
            if run_explanation and "last_prediction" in st.session_state:
                with st.spinner("🧠 Generating AI explanation..."):
                    try:
                        explanation = _get_explanation(api_base_url, uploaded_file)
                        st.session_state["last_explanation"] = explanation
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Explanation failed: {e}")

            # Display results
            if "last_prediction" in st.session_state:
                st.divider()
                _display_results(
                    st.session_state["last_prediction"],
                    st.session_state.get("last_probabilities", {}),
                    st.session_state.get("last_metadata", {}),
                )

            # Display explanation
            if "last_explanation" in st.session_state:
                st.divider()
                _display_explanation(st.session_state["last_explanation"])

            # Feedback section
            if "last_prediction" in st.session_state and uploaded_file is not None:
                st.divider()
                _render_feedback_form(api_base_url, uploaded_file)

        else:
            # Placeholder when no file uploaded
            st.info("👆 Please upload a chest CT scan image to begin analysis")

    with right_col:
        st.markdown("### 📖 How It Works")
        st.markdown(
            """
            <div class="info-box">
            <strong>1. Upload Image</strong><br>
            Select a chest CT scan in PNG, JPG, or JPEG format<br><br>

            <strong>2. AI Analysis</strong><br>
            Deep learning model processes the scan using dual-pathway architecture<br><br>

            <strong>3. Classification</strong><br>
            Model predicts one of four classes with confidence scores<br><br>

            <strong>4. Review Results</strong><br>
            Examine predictions, probabilities, and optional explanations<br><br>

            <strong>5. Provide Feedback</strong><br>
            Help improve the model by confirming or correcting predictions
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🏥 Classification Categories")
        st.markdown(
            """
            - **Adenocarcinoma** - Glandular tissue cancer
            - **Large Cell Carcinoma** - Undifferentiated cancer
            - **Squamous Cell Carcinoma** - Epithelial cancer
            - **Normal** - No cancer detected
            """
        )

        st.markdown("### 💡 Best Practices")
        st.markdown(
            """
            - ✅ Use clear, single-slice CT scans
            - ✅ Ensure proper image quality and contrast
            - ✅ Verify API server is running before analysis
            - ✅ Review confidence scores critically
            - ✅ Provide feedback for model improvement
            - ⚠️ Always validate with medical professionals
            """
        )

        st.markdown("### 🔬 Model Information")
        if "last_metadata" in st.session_state:
            metadata = st.session_state["last_metadata"]
            st.markdown(
                f"""
                - **Architecture**: {metadata.get("model_type", "Unknown").replace("_", " ").title()}
                - **Features**: {"Enabled" if metadata.get("features_used") else "Disabled"}
                - **Device**: {metadata.get("device", "Unknown")}
                """
            )


def _display_results(prediction: dict, probabilities: dict, metadata: dict) -> None:
    """Display prediction results with enhanced visualization."""
    st.markdown("### 🎯 Analysis Results")

    pred_class = prediction.get("class", "Unknown")
    confidence = prediction.get("confidence", 0.0)

    # Main prediction card
    st.markdown(
        f"""
        <div class="prediction-card">
            <h4 style="margin-top: 0;">Predicted Diagnosis</h4>
            <span class="class-badge">{pred_class.replace("_", " ").title()}</span>
            <br><br>
            <strong>Confidence:</strong> {confidence:.1%}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Confidence indicator
    if confidence >= 0.80:
        st.markdown(
            '<div class="confidence-high">🟢 High Confidence Prediction</div>',
            unsafe_allow_html=True,
        )
        st.caption("Model shows strong certainty in this classification.")
    elif confidence >= 0.60:
        st.markdown(
            '<div class="confidence-medium">🟡 Moderate Confidence</div>',
            unsafe_allow_html=True,
        )
        st.warning("⚠️ Consider expert review for validation")
    else:
        st.markdown(
            '<div class="confidence-low">🔴 Low Confidence</div>',
            unsafe_allow_html=True,
        )
        st.error("⚠️ Expert review strongly recommended")

    # Probability distribution
    if probabilities:
        st.markdown("#### 📊 Probability Distribution")

        # Create DataFrame for visualization
        prob_df = pd.DataFrame(
            {
                "Class": [k.replace("_", " ").title() for k in probabilities],
                "Probability": [v * 100 for v in probabilities.values()],
            }
        ).sort_values("Probability", ascending=False)

        # Display as bar chart
        st.bar_chart(prob_df.set_index("Class"))

        # Display as table
        with st.expander("View Detailed Probabilities"):
            for class_name, prob in probabilities.items():
                display_name = class_name.replace("_", " ").title()
                st.progress(prob, text=f"{display_name}: {prob:.1%}")


def _display_explanation(explanation: dict) -> None:
    """Display GradCAM explanation visualization."""
    st.markdown("### 🧠 AI Explanation (GradCAM)")

    exp_data = explanation.get("explanation", {})
    heatmap_b64 = exp_data.get("heatmap")

    if heatmap_b64:
        # Decode and display heatmap
        heatmap_bytes = base64.b64decode(heatmap_b64)
        heatmap_img = Image.open(io.BytesIO(heatmap_bytes))

        col1, col2 = st.columns(2)
        with col1:
            st.image(
                st.session_state.get("last_file"),
                caption="Original CT Scan",
                use_container_width=True,
            )
        with col2:
            st.image(heatmap_img, caption="Attention Heatmap", use_container_width=True)

        st.info(f"ℹ️ {exp_data.get('description', '')}")

        st.caption(
            "**How to read**: Warmer colors (red/yellow) show regions the AI focused on. "
            "Cooler colors (blue/purple) indicate less important areas."
        )
    else:
        st.warning("No heatmap data available")


def _render_feedback_form(api_base_url: str, uploaded_file) -> None:
    """Render enhanced feedback form."""
    st.markdown("### 📋 Clinical Feedback")
    st.caption("Your feedback helps improve model accuracy for future diagnoses")

    with st.form("feedback_form", clear_on_submit=True):
        st.markdown("**Was the prediction accurate?**")

        feedback_choice = st.radio(
            "Prediction Accuracy",
            options=["✅ Correct", "❌ Incorrect", "🤔 Uncertain"],
            horizontal=True,
            label_visibility="collapsed",
        )

        is_correct = feedback_choice == "✅ Correct"
        correct_class = None

        if feedback_choice == "❌ Incorrect":
            correct_class = st.selectbox(
                "**What is the actual diagnosis?**",
                options=[
                    "adenocarcinoma",
                    "large_cell_carcinoma",
                    "normal",
                    "squamous_cell_carcinoma",
                ],
                format_func=lambda x: x.replace("_", " ").title(),
            )

        user_note = st.text_area(
            "**Additional observations (optional)**",
            placeholder="Image quality, unusual features, diagnostic notes, etc.",
            max_chars=500,
            help="Any relevant clinical observations",
        )

        confidence_rating = st.select_slider(
            "**Your confidence in this assessment**",
            options=["Very Low", "Low", "Medium", "High", "Very High"],
            value="Medium",
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            submitted = st.form_submit_button("📤 Submit Feedback", use_container_width=True, type="primary")
        with col2:
            st.form_submit_button("❌ Cancel", use_container_width=True)

        if submitted:
            pred = st.session_state.get("last_prediction", {})
            pred_class = pred.get("class", "")
            pred_confidence = pred.get("confidence", 0.0)

            with st.spinner("💾 Saving feedback..."):
                try:
                    note_with_rating = f"{user_note or ''} [User Confidence: {confidence_rating}]".strip()

                    response = _send_feedback(
                        api_base_url=api_base_url,
                        uploaded_file=uploaded_file,
                        predicted_class=pred_class,
                        predicted_confidence=pred_confidence,
                        is_correct=is_correct,
                        correct_class=correct_class,
                        user_note=note_with_rating,
                        confidence_rating=confidence_rating,
                    )

                    st.success("✅ Thank you! Your feedback has been recorded.")
                    st.balloons()

                    with st.expander("Feedback Details"):
                        st.json(response)

                except httpx.HTTPStatusError as exc:
                    st.error(f"❌ Failed to submit: {exc.response.text}")
                except Exception as exc:
                    st.error(f"❌ Error: {exc}")


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


def _predict(api_base_url: str, uploaded_file) -> dict:
    """Send prediction request to API."""
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


def _get_explanation(api_base_url: str, uploaded_file) -> dict:
    """Get GradCAM explanation from API."""
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    with httpx.Client(timeout=60) as client:
        response = client.post(f"{api_base_url.rstrip('/')}/explain", files=files)
        response.raise_for_status()
        return response.json()


def _send_feedback(
    api_base_url: str,
    uploaded_file,
    predicted_class: str,
    predicted_confidence: float,
    is_correct: bool,
    correct_class: str | None,
    user_note: str | None,
    confidence_rating: str | None,
) -> dict:
    """Send feedback to API with enhanced metadata."""
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }

    data = {
        "predicted_class": predicted_class,
        "predicted_confidence": str(predicted_confidence),
        "is_correct": str(is_correct).lower(),
    }

    if not is_correct and correct_class is not None:
        data["correct_class"] = correct_class

    if user_note:
        data["user_note"] = user_note

    if confidence_rating:
        data["confidence_rating"] = confidence_rating

    with httpx.Client(timeout=30) as client:
        response = client.post(f"{api_base_url.rstrip('/')}/feedback", files=files, data=data)
        response.raise_for_status()
        return response.json()


def _get_feedback_stats(api_base_url: str) -> dict | None:
    """Retrieve feedback statistics from API."""
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{api_base_url.rstrip('/')}/feedback/stats")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError:
        return None


if __name__ == "__main__":
    render()
