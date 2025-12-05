# Quick Testing Guide - Your Server is Running! 🎉

Your backend is running at: **http://127.0.0.1:8000**

## ✅ What's Working

- ✅ Backend server is running
- ✅ Text model (DistilBERT) loaded successfully
- ⚠️ Image & Fusion models not loaded (expected - will use dummy predictions)

## 🧪 Test Methods

### Method 1: Browser Test (Easiest)

1. **Open your browser** and go to:
   ```
   http://localhost:8000
   ```
   You should see:
   ```json
   {
     "status": "ok",
     "version": "1.0.0",
     "models_loaded": {
       "image_model": false,
       "text_model": true,
       "fusion_model": false
     }
   }
   ```

2. **Test Interactive API Docs**:
   ```
   http://localhost:8000/docs
   ```
   This opens Swagger UI where you can:
   - See all endpoints
   - Test them directly
   - See request/response examples

### Method 2: Test Script (Automated)

Open a **NEW terminal** in VS Code (click `+` in terminal panel):

```bash
# Make sure you're in the project root
cd D:\Downloads\Techno

# Run the test script
python backend/test_api.py
```

This will test:
- Health endpoint
- Predict endpoint (with dummy image)

### Method 3: REST Client (VS Code Extension)

1. **Install REST Client extension** (if not already):
   - Press `Ctrl + Shift + X`
   - Search: "REST Client"
   - Install by Huachao Mao

2. **Open** `backend/test.http` file

3. **Click "Send Request"** above any request

### Method 4: Test with cURL (Terminal)

Open a **NEW terminal**:

```powershell
# Health check
curl http://localhost:8000/

# Predict (you'll need an actual image file)
curl -X POST http://localhost:8000/api/predict `
  -F "image=@dataset/real/image.png" `
  -F "title=Test Product" `
  -F "description=This is a test product"
```

### Method 5: Test Frontend (Full Experience)

1. **Open a NEW terminal** in VS Code

2. **Install frontend dependencies** (first time only):
   ```bash
   cd frontend
   npm install
   ```

3. **Start frontend**:
   ```bash
   npm run dev
   ```

4. **Open browser**: http://localhost:3000

5. **Test the UI**:
   - Upload an image
   - Enter title and description
   - Click "Check Authenticity"
   - See results!

## 🎯 Quick Test Right Now

**Easiest way - just open in browser:**

1. Open: http://localhost:8000
2. Then open: http://localhost:8000/docs
3. Click on `GET /` endpoint
4. Click "Try it out" → "Execute"
5. See the response!

## 📝 Testing the Predict Endpoint

### Using API Docs (Easiest):

1. Go to: http://localhost:8000/docs
2. Find `POST /api/predict`
3. Click "Try it out"
4. Fill in:
   - `title`: "Nike Air Max 90"
   - `description`: "Authentic Nike shoes"
   - `seller_rating`: 4.5
   - `num_reviews`: 1000
   - `image`: Click "Choose File" and select an image
5. Click "Execute"
6. See the prediction result!

### Expected Response:

```json
{
  "authenticity_score": 0.5,
  "decision": "flag",
  "explanations": {
    "image_reason": "Image model not available (using placeholder)",
    "text_reason": "Text description appears authentic and professional",
    "metadata_reason": "High seller rating (4.5/5.0). Established seller with 1000 reviews.",
    "heatmap": null
  }
}
```

## 🔍 What to Expect

Since models aren't trained yet:
- ✅ API works perfectly
- ✅ Text analysis works (DistilBERT is loaded)
- ⚠️ Image analysis uses dummy predictions
- ⚠️ Final score uses weighted average (not trained fusion model)

**This is normal!** The system is designed to work even without trained models.

## 🚀 Next Steps

1. **Test the API** using any method above
2. **Test the frontend** (Method 5) for full experience
3. **Train models** later when you have a dataset

## 💡 Pro Tips

- Keep the backend terminal open (don't close it)
- Use multiple terminals for frontend/testing
- The API docs at `/docs` are your best friend!
- Check terminal output for any errors

---

**Your server is ready! Start testing! 🎉**

