# How to Get Accurate Predictions (Not Just 50-60%)

## Why You're Getting 50-60% Scores

Currently, your system is using **dummy/placeholder predictions** because:
- ❌ **Image model is NOT trained** → Always returns 0.5 (neutral)
- ✅ **Text model IS loaded** (DistilBERT) → Works but uses simple keyword matching
- ❌ **Fusion model is NOT trained** → Uses simple weighted average

**Result**: Scores cluster around 50-60% because:
- Image contributes 0.5 (dummy)
- Text contributes 0.5-0.7 (keyword-based, not ML)
- Metadata contributes normalized rating
- Final = weighted average ≈ 0.5-0.6

## ✅ Solution: Train the Models

### Step 1: Prepare Your Dataset

Organize images in this structure:

```
dataset/
├── train/
│   ├── real/          # Authentic product images
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── fake/          # Counterfeit product images
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

**Minimum Requirements:**
- At least 50 images per class per split (300 total minimum)
- More is better! Aim for 200+ per class per split

**Where to Get Data:**
- Your own product images
- Public datasets (if available)
- Scraped e-commerce listings (with permission)
- Synthetic data generation

### Step 2: Train Image Model

```bash
# Make sure you're in the project root
cd D:\Downloads\Techno

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Train the image model
python ml_pipeline/scripts/train_image_model.py \
    --dataset_path dataset/ \
    --epochs 20 \
    --batch_size 32
```

**This will:**
- Load your images
- Train EfficientNetB0 on fake vs real classification
- Save model to `ml_pipeline/models/image_model.h5`
- Take 30-60 minutes depending on dataset size

### Step 3: Train Fusion Model

```bash
# After image model is trained
python ml_pipeline/scripts/train_multimodal.py \
    --dataset_path dataset/ \
    --image_model_path ml_pipeline/models/image_model.h5 \
    --epochs 20 \
    --batch_size 32
```

**This will:**
- Extract features from trained image model
- Extract text embeddings (DistilBERT)
- Combine with metadata
- Train fusion model
- Save to `ml_pipeline/models/fusion_model.h5`

### Step 4: Restart Backend

After training, **restart your backend server**:

1. Stop the current server (Ctrl+C)
2. Start it again:
   ```bash
   cd backend
   uvicorn app.main:app --reload --port 8000
   ```

The server will automatically load the new models!

### Step 5: Test Again

Now when you test:
- ✅ Image model will analyze actual images
- ✅ Text model already works (DistilBERT)
- ✅ Fusion model will combine everything intelligently
- ✅ **Scores will be accurate and varied!**

## 🎯 Expected Results After Training

**Before Training:**
- Scores: 50-60% (always similar)
- Image analysis: "Image model not available"
- Low accuracy

**After Training:**
- Scores: 0-100% (varies based on actual analysis)
- Image analysis: "Image shows characteristics consistent with authentic products"
- High accuracy (80-95% depending on dataset quality)

## 📊 Improving Accuracy Further

### 1. Better Dataset
- More diverse images
- Better labeling
- Balanced classes
- High-quality images

### 2. Fine-Tune Models
- Train for more epochs
- Adjust hyperparameters
- Use data augmentation
- Try different architectures

### 3. Add More Features
- Product category
- Price analysis
- Seller history
- Review sentiment

## 🚀 Quick Test Without Full Training

If you want to test with **better dummy predictions** right now:

I've already improved the code to:
- Give more weight to text analysis (50% instead of 30%)
- Give more weight to metadata (30% instead of 20%)
- Use better text scoring based on keywords

**Restart your server** and test again - you should see more variation in scores!

But for **real accuracy**, you MUST train the models with your dataset.

## 💡 Tips

1. **Start Small**: Train with 100-200 images first to test the pipeline
2. **Monitor Training**: Watch for overfitting (high train accuracy, low val accuracy)
3. **Use Validation Set**: Don't train on validation data!
4. **Iterate**: Improve dataset → Retrain → Test → Repeat

## 📝 Next Steps

1. **Collect/Gather Dataset**: Get images of real and fake products
2. **Organize Dataset**: Put images in the folder structure above
3. **Train Image Model**: Run training script
4. **Train Fusion Model**: Run fusion training script
5. **Test**: Restart server and test with real predictions!

---

**Remember**: Machine learning models need data to learn! Without training, you'll always get placeholder predictions.

