# Quick Start Guide - Enhanced Frontend & API

Get started with the new CT Scan MLOps frontend and API in 5 minutes!

---

## 🚀 Quick Start

### Step 1: Ensure Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate

# Install any new dependencies (matplotlib, scipy for GradCAM)
uv sync

# Verify installation
uv run python -c "import matplotlib, scipy; print('✅ Dependencies OK')"
```

---

### Step 2: Start the API Server

```bash
# Make sure you have a model checkpoint
ls outputs/checkpoints/model.pt  # or best_model.ckpt

# Start the FastAPI server
uvicorn ct_scan_mlops.api:app --reload --host 0.0.0.0 --port 8000

# You should see:
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Verify API is running:**
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "ok": true,
  "model_loaded": true,
  "model_type": "dual_pathway",
  ...
}
```

---

### Step 3: Launch the Frontend

**In a new terminal:**

```bash
# Activate environment
source .venv/bin/activate

# Start Streamlit
streamlit run src/ct_scan_mlops/frontend/pages/home.py

# Opens automatically at http://localhost:8501
```

---

### Step 4: Test the System

**In the Frontend:**

1. **Check API Connection**
   - Expand "⚙️ API Configuration"
   - Click "🔍 Check API Health"
   - Should see "✅ API is healthy and ready"

2. **Upload a CT Scan**
   - Click "Select CT Scan Image"
   - Choose a test image (PNG, JPG, or JPEG)
   - Preview appears automatically

3. **Run Classification**
   - Click "🔬 Run Classification"
   - Wait 1-2 seconds
   - View results with confidence scores

4. **Get Explanation (Optional)**
   - After classification, click "🔍 Get Explanation"
   - View GradCAM heatmap
   - See side-by-side comparison

5. **Submit Feedback**
   - Scroll to "📋 Clinical Feedback"
   - Select accuracy (Correct/Incorrect/Uncertain)
   - Add notes if desired
   - Click "📤 Submit Feedback"
   - See success confirmation! 🎈

---

## 🧪 Testing New API Endpoints

### Test Enhanced Prediction

```bash
# Single prediction with full response
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/ct_scan.png" \
  | jq '.'

# Look for:
# - prediction.confidence
# - probabilities (all 4 classes)
# - metadata.model_type
```

### Test Batch Processing

```bash
# Process multiple images at once
curl -X POST "http://localhost:8000/predict/batch" \
  -F "files=@scan1.png" \
  -F "files=@scan2.png" \
  -F "files=@scan3.png" \
  | jq '.batch_summary'

# Output:
# {
#   "total": 3,
#   "successful": 3,
#   "failed": 0
# }
```

### Test Explainability

```bash
# Get GradCAM heatmap
curl -X POST "http://localhost:8000/explain" \
  -F "file=@ct_scan.png" \
  | jq '.explanation.heatmap' -r \
  | base64 -d > heatmap.png

# View heatmap.png
```

### Test Feedback Stats

```bash
# View feedback analytics
curl http://localhost:8000/feedback/stats | jq '.'

# Output includes:
# - total_feedback
# - accuracy
# - class_distribution
# - recent_feedback
```

---

## 📊 Exploring the Frontend

### Main Features

**1. Upload Section (Left)**
- Drag & drop or browse for files
- Real-time image preview
- File metadata display

**2. Results Display**
- **Prediction Card**: Large, prominent diagnosis
- **Confidence Indicator**: Color-coded (Green/Yellow/Red)
- **Probability Chart**: Bar chart of all classes
- **Detailed View**: Expandable table with exact percentages

**3. Explanation (Optional)**
- **Original Image**: Your uploaded CT scan
- **Attention Heatmap**: What the AI focused on
- **Color Legend**: Red = high attention, Blue = low attention

**4. Feedback Form**
- Simple 3-option assessment
- Optional clinical notes
- Confidence rating
- Instant confirmation

**5. Info Panel (Right)**
- Step-by-step workflow
- Class descriptions
- Best practices
- Model metadata

---

## 🎨 UI Features

### Confidence Indicators

| Color | Range | Recommendation |
|-------|-------|----------------|
| 🟢 Green | ≥80% | High confidence - reliable prediction |
| 🟡 Yellow | 60-79% | Moderate - consider expert review |
| 🔴 Red | <60% | Low - expert review required |

### Visual Elements

- **Gradient Header**: Purple gradient title
- **Disclaimer Box**: Yellow warning banner
- **Prediction Cards**: Clean, shadowed containers
- **Progress Bars**: Visual probability representation
- **Smooth Animations**: Balloons on feedback success

---

## 🔧 Configuration

### Environment Variables

```bash
# API endpoint (default: http://127.0.0.1:8000)
export CT_SCAN_API_URL=http://localhost:8000

# Model path (optional override)
export MODEL_PATH=outputs/checkpoints/model.pt

# Config path (optional)
export CONFIG_PATH=configs/config.yaml

# Feedback storage
export FEEDBACK_DIR=feedback
export FEEDBACK_DB=feedback/feedback.db

# Drift monitoring
export DRIFT_CURRENT_PATH=data/drift/current.csv
```

### Model Requirements

**Minimum:**
- Model checkpoint: `outputs/checkpoints/model.pt` (or `.ckpt`)
- Config file: `configs/config.yaml`

**For Dual Pathway:**
- Feature metadata: `outputs/checkpoints/feature_metadata.json`

**For GradCAM:**
- matplotlib: `uv add matplotlib`
- scipy: `uv add scipy`

---

## 📱 Mobile/Tablet Support

The frontend is responsive and works on:
- ✅ Desktop (optimal)
- ✅ Tablet (good)
- ⚠️ Mobile (limited - small screen)

**Tip**: Use tablets or larger for best experience with medical imaging.

---

## 🐛 Troubleshooting

### Issue: API Connection Failed

**Solution:**
```bash
# 1. Check if API is running
curl http://localhost:8000/health

# 2. Verify environment variable
echo $CT_SCAN_API_URL

# 3. Check frontend config
# In Streamlit UI: Expand "⚙️ API Configuration"
# Verify URL matches where API is running
```

### Issue: Model Not Loaded

**Solution:**
```bash
# Check model file exists
ls outputs/checkpoints/model.pt

# Check API logs for errors
# Look for "Missing model weights" error

# If using .ckpt, convert to .pt:
uv run python -c "
from ct_scan_mlops.promote_model import convert_ckpt_to_pt
convert_ckpt_to_pt(
    'outputs/checkpoints/best_model.ckpt',
    'outputs/checkpoints/model.pt'
)
"
```

### Issue: GradCAM Not Working

**Solution:**
```bash
# Install required dependencies
uv add matplotlib scipy

# Verify installation
uv run python -c "import matplotlib.pyplot as plt; print('OK')"

# Check API logs for specific error
```

### Issue: Feedback Not Saving

**Solution:**
```bash
# Check feedback directory exists
mkdir -p feedback

# Check database
ls feedback/feedback.db

# Manually initialize if needed
sqlite3 feedback/feedback.db "CREATE TABLE IF NOT EXISTS feedback (...);"

# Check permissions
chmod 777 feedback  # Development only
```

---

## 📈 Performance Tips

### For Faster Predictions

1. **Use GPU:**
   ```bash
   # Check CUDA availability
   uv run python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Use .pt files:**
   - Faster loading than .ckpt
   - More secure (weights_only=True)

3. **Batch Processing:**
   - Use `/predict/batch` for multiple images
   - More efficient than sequential

### For Better UX

1. **Pre-check API:**
   - Click "Check API Health" before uploading
   - Ensures connection before work

2. **Use Feedback:**
   - Submit feedback for continuous improvement
   - View stats to see model performance

3. **Explain Sparingly:**
   - GradCAM takes extra time
   - Use when you need to understand decision

---

## 🎓 Next Steps

1. **Explore API Documentation:**
   - Visit http://localhost:8000/docs
   - Try interactive API explorer

2. **Check Metrics:**
   - Visit http://localhost:8000/metrics
   - Monitor system performance

3. **Review Feedback:**
   - Query feedback database
   - Analyze model performance

4. **Read Full Documentation:**
   - See `docs/FRONTEND_API_ENHANCEMENTS.md`
   - Complete feature list and examples

---

## 📚 Additional Resources

- **Main README**: `README.md`
- **Project Structure**: `docs/Structure.md`
- **API Details**: `docs/FRONTEND_API_ENHANCEMENTS.md`
- **Dual Pathway Guide**: `docs/MIGRATION_DUAL_PATHWAY.md`
- **FastAPI Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

---

## ✅ Success Checklist

After following this guide, you should have:

- [ ] ✅ API running on port 8000
- [ ] ✅ Frontend running on port 8501
- [ ] ✅ Health check passing
- [ ] ✅ Successfully uploaded an image
- [ ] ✅ Received prediction with confidence scores
- [ ] ✅ Viewed probability distribution
- [ ] ✅ Generated GradCAM explanation
- [ ] ✅ Submitted feedback successfully
- [ ] ✅ Viewed feedback statistics

**You're ready to use the enhanced CT Scan MLOps system! 🎉**

---

## 💬 Getting Help

If you encounter issues:

1. Check troubleshooting section above
2. Review API logs for errors
3. Verify all dependencies installed
4. Check file permissions
5. Ensure model checkpoint exists
6. Review `docs/FRONTEND_API_ENHANCEMENTS.md`

**Happy analyzing! 🩻✨**
