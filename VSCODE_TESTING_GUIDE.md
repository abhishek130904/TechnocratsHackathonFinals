# Testing in VS Code - Step-by-Step Guide

This guide will help you test the Counterfeit Product Detection system directly in VS Code.

## Prerequisites

1. **Python Extension**: Install "Python" extension by Microsoft
2. **ES7+ React/Redux/React-Native snippets**: Optional but helpful
3. **REST Client Extension**: Optional, for testing API endpoints

## Step 1: Set Up Python Environment

### 1.1 Create Virtual Environment

1. Open VS Code terminal: `Ctrl + ~` (or `View > Terminal`)
2. Run:
   ```bash
   python -m venv venv
   ```

### 1.2 Activate Virtual Environment

**In VS Code Terminal:**
- **Windows PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows CMD:**
  ```cmd
  venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

**Note**: If you see an execution policy error in PowerShell, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 1.3 Select Python Interpreter

1. Press `Ctrl + Shift + P` (Command Palette)
2. Type: `Python: Select Interpreter`
3. Choose: `.\venv\Scripts\python.exe` (or `./venv/bin/python` on Mac/Linux)

### 1.4 Install Backend Dependencies

In the terminal (with venv activated):
```bash
cd backend
pip install -r requirements.txt
```

**This will take a few minutes** - it's installing TensorFlow and other large packages.

## Step 2: Set Up Frontend

### 2.1 Install Node Dependencies

1. Open a **new terminal** in VS Code: Click the `+` button in terminal panel
2. Run:
   ```bash
   cd frontend
   npm install
   ```

## Step 3: Test Backend (Terminal Method)

### 3.1 Start Backend Server

1. Open terminal (ensure venv is activated)
2. Navigate to backend:
   ```bash
   cd backend
   ```
3. Start server:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 3.2 Test Health Endpoint

1. Open browser: http://localhost:8000
2. Or use VS Code REST Client (see Step 6)

You should see:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "models_loaded": {
    "image_model": false,
    "text_model": false,
    "fusion_model": false
  }
}
```

### 3.3 Test API Documentation

Open: http://localhost:8000/docs

This is FastAPI's interactive Swagger UI where you can test endpoints directly!

## Step 4: Test Frontend

### 4.1 Start Frontend Dev Server

1. Open a **new terminal** in VS Code (click `+` in terminal panel)
2. Run:
   ```bash
   cd frontend
   npm run dev
   ```

You should see:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

### 4.2 Open Frontend in Browser

1. Click the link: http://localhost:3000
2. Or manually open: http://localhost:3000

### 4.3 Test the UI

1. **Upload an image**: Click "Click to upload image"
2. **Enter title**: e.g., "Nike Air Max 90"
3. **Enter description**: e.g., "Authentic Nike shoes"
4. **Click "Check Authenticity"**
5. **View results**: You'll see a prediction (even without trained models, it uses dummy predictions)

## Step 5: Use VS Code Debugger (Advanced)

### 5.1 Create Launch Configuration

1. Create `.vscode/launch.json`:
   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Python: FastAPI",
         "type": "python",
         "request": "launch",
         "module": "uvicorn",
         "args": [
           "app.main:app",
           "--reload",
           "--host",
           "0.0.0.0",
           "--port",
           "8000"
         ],
         "jinja": true,
         "justMyCode": true,
         "cwd": "${workspaceFolder}/backend"
       }
     ]
   }
   ```

2. Set breakpoints in your code (click left of line numbers)
3. Press `F5` to start debugging
4. The debugger will pause at breakpoints

### 5.2 Debug Frontend (Optional)

1. Install "Debugger for Chrome" extension
2. Add to `.vscode/launch.json`:
   ```json
   {
     "name": "Launch Chrome",
     "type": "chrome",
     "request": "launch",
     "url": "http://localhost:3000",
     "webRoot": "${workspaceFolder}/frontend"
   }
   ```

## Step 6: Test API with REST Client (Optional)

### 6.1 Install REST Client Extension

1. Install "REST Client" extension by Huachao Mao
2. Create `backend/test.http` file:

```http
### Health Check
GET http://localhost:8000/

### Predict (with form data)
POST http://localhost:8000/api/predict
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW

------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="title"

Nike Air Max 90 Running Shoes
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="description"

Authentic Nike Air Max 90 with original box and tags
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="seller_rating"

4.5
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="num_reviews"

1250
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="image"; filename="test_image.jpg"
Content-Type: image/jpeg

< ./test_image.jpg
------WebKitFormBoundary7MA4YWxkTrZu0gW--
```

3. Click "Send Request" above each request

## Step 7: Run Both Servers Simultaneously

### 7.1 Using VS Code Tasks

1. Create `.vscode/tasks.json`:
   ```json
   {
     "version": "2.0.0",
     "tasks": [
       {
         "label": "Start Backend",
         "type": "shell",
         "command": "uvicorn app.main:app --reload",
         "options": {
           "cwd": "${workspaceFolder}/backend"
         },
         "isBackground": true,
         "problemMatcher": []
       },
       {
         "label": "Start Frontend",
         "type": "shell",
         "command": "npm run dev",
         "options": {
           "cwd": "${workspaceFolder}/frontend"
         },
         "isBackground": true,
         "problemMatcher": []
       },
       {
         "label": "Start All",
         "dependsOn": ["Start Backend", "Start Frontend"],
         "problemMatcher": []
       }
     ]
   }
   ```

2. Press `Ctrl + Shift + P`
3. Type: `Tasks: Run Task`
4. Select: `Start All`

## Step 8: Test with Python Script

### 8.1 Create Test Script

Create `backend/test_api.py`:

```python
"""Simple test script for the API"""

import requests
import json

# Test health endpoint
print("Testing health endpoint...")
response = requests.get("http://localhost:8000/")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test predict endpoint (with dummy data)
print("\nTesting predict endpoint...")
files = {
    'image': ('test.jpg', b'fake_image_data', 'image/jpeg')  # Dummy image
}
data = {
    'title': 'Test Product',
    'description': 'This is a test product description',
    'seller_rating': 4.5,
    'num_reviews': 100
}

response = requests.post("http://localhost:8000/api/predict", files=files, data=data)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    print(f"Response: {json.dumps(response.json(), indent=2)}")
else:
    print(f"Error: {response.text}")
```

### 8.2 Run Test Script

In terminal (with backend running):
```bash
cd backend
python test_api.py
```

## Step 9: Check Logs

### 9.1 View Backend Logs

Backend logs appear in the terminal where you started the server.

### 9.2 View Log File

If configured, logs are saved to `logs/app.log`. Open it in VS Code to view.

## Step 10: Common Issues & Solutions

### Issue: "Module not found" errors

**Solution:**
1. Ensure venv is activated (you should see `(venv)` in terminal)
2. Reinstall: `pip install -r backend/requirements.txt`

### Issue: Port already in use

**Solution:**
1. Find process using port:
   ```bash
   # Windows
   netstat -ano | findstr :8000
   
   # Kill process (replace PID)
   taskkill /PID <PID> /F
   ```

### Issue: Frontend can't connect to backend

**Solution:**
1. Ensure backend is running on port 8000
2. Check CORS settings in `backend/app/config.py`
3. Check browser console for errors (F12)

### Issue: Models not loading

**This is normal!** The system works without trained models using dummy predictions.

**To fix:**
1. Train models using scripts in `ml_pipeline/scripts/`
2. Restart backend server

## Quick Test Checklist

- [ ] Virtual environment created and activated
- [ ] Backend dependencies installed
- [ ] Frontend dependencies installed
- [ ] Backend server running (http://localhost:8000)
- [ ] Frontend server running (http://localhost:3000)
- [ ] Health endpoint returns OK
- [ ] API docs accessible (http://localhost:8000/docs)
- [ ] Frontend UI loads
- [ ] Can upload image and get prediction

## Tips for VS Code

1. **Use Split Terminal**: Right-click terminal tab → "Split Terminal" to run both servers
2. **Use Integrated Terminal**: `Ctrl + ~` to toggle terminal
3. **Use Problems Panel**: `Ctrl + Shift + M` to see errors
4. **Use Command Palette**: `Ctrl + Shift + P` for all commands
5. **Use Go to Definition**: `F12` on any function/class to see its definition
6. **Use Peek Definition**: `Alt + F12` to see definition inline

## Next Steps

Once everything is running:
1. Read the code comments (they're extensive!)
2. Try modifying thresholds in `backend/app/config.py`
3. Train models using the scripts
4. Explore the API docs at http://localhost:8000/docs

---

**Happy Testing! 🚀**

