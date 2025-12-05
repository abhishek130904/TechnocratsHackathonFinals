# API Specification

Complete API documentation for the Counterfeit Product Detection API.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. For production, implement API keys or OAuth2.

## Endpoints

### 1. Health Check

**Endpoint:** `GET /`

**Description:** Check API status and model loading status.

**Response:**

```json
{
  "status": "ok",
  "version": "1.0.0",
  "models_loaded": {
    "image_model": true,
    "text_model": true,
    "fusion_model": true
  }
}
```

**Status Codes:**
- `200 OK`: API is running

**Example:**

```bash
curl http://localhost:8000/
```

---

### 2. Predict Authenticity

**Endpoint:** `POST /api/predict`

**Description:** Main prediction endpoint. Analyzes product image, text, and metadata to determine authenticity.

**Content-Type:** `multipart/form-data`

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | File | Yes | Product image (JPG, PNG, WEBP, max 10MB) |
| `title` | String | Yes | Product title (1-200 characters) |
| `description` | String | Yes | Product description (1-2000 characters) |
| `seller_rating` | Float | No | Seller rating (0.0 to 5.0) |
| `num_reviews` | Integer | No | Number of reviews (≥ 0) |

**Response:**

```json
{
  "authenticity_score": 0.83,
  "decision": "flag",
  "explanations": {
    "image_reason": "Image shows some inconsistencies that warrant review",
    "text_reason": "Description contains suspicious terms: replica, copy",
    "metadata_reason": "Low seller rating (2.1/5.0). Very few reviews (3).",
    "heatmap": "iVBORw0KGgoAAAANSUhEUgAA..." // Base64 encoded PNG or null
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `authenticity_score` | Float | Score from 0.0 (fake) to 1.0 (real) |
| `decision` | String | One of: "approve", "flag", "reject" |
| `explanations.image_reason` | String | Explanation based on image analysis |
| `explanations.text_reason` | String | Explanation based on text analysis |
| `explanations.metadata_reason` | String | Explanation based on metadata |
| `explanations.heatmap` | String\|null | Base64-encoded heatmap image (optional) |

**Status Codes:**
- `200 OK`: Prediction successful
- `400 Bad Request`: Invalid input (missing fields, invalid file format, etc.)
- `500 Internal Server Error`: Server error during processing

**Example (cURL):**

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "image=@product_image.jpg" \
  -F "title=Nike Air Max 90 Running Shoes" \
  -F "description=Authentic Nike Air Max 90 with original box and tags" \
  -F "seller_rating=4.5" \
  -F "num_reviews=1250"
```

**Example (JavaScript/Fetch):**

```javascript
const formData = new FormData();
formData.append('image', imageFile);
formData.append('title', 'Nike Air Max 90');
formData.append('description', 'Authentic product...');
formData.append('seller_rating', '4.5');
formData.append('num_reviews', '1250');

const response = await fetch('http://localhost:8000/api/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);
```

**Example (Python/Requests):**

```python
import requests

url = "http://localhost:8000/api/predict"

files = {"image": open("product_image.jpg", "rb")}
data = {
    "title": "Nike Air Max 90 Running Shoes",
    "description": "Authentic Nike Air Max 90 with original box",
    "seller_rating": 4.5,
    "num_reviews": 1250
}

response = requests.post(url, files=files, data=data)
result = response.json()
print(result)
```

---

### 3. Submit Feedback

**Endpoint:** `POST /api/feedback`

**Description:** Submit feedback on prediction accuracy. Used for improving the model.

**Content-Type:** `application/x-www-form-urlencoded` or `multipart/form-data`

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prediction_id` | String | No | ID of the prediction being reviewed |
| `was_correct` | Boolean | Yes | Whether the prediction was correct |
| `actual_label` | String | No | Actual label ("real" or "fake") if prediction was wrong |
| `comments` | String | No | Additional feedback comments |

**Response:**

```json
{
  "status": "success",
  "message": "Feedback received. Thank you for helping improve the model!",
  "prediction_id": "abc123"
}
```

**Status Codes:**
- `200 OK`: Feedback received
- `400 Bad Request`: Invalid input

**Example:**

```bash
curl -X POST http://localhost:8000/api/feedback \
  -F "prediction_id=abc123" \
  -F "was_correct=false" \
  -F "actual_label=real" \
  -F "comments=Model incorrectly flagged authentic product"
```

---

## Error Responses

All errors follow a consistent format:

```json
{
  "error": "Error message",
  "detail": "Additional error details (optional)"
}
```

### Common Error Codes

| Status Code | Description |
|-------------|-------------|
| `400 Bad Request` | Invalid request parameters |
| `404 Not Found` | Endpoint not found |
| `422 Unprocessable Entity` | Validation error (Pydantic) |
| `500 Internal Server Error` | Server error |

### Example Error Response

```json
{
  "error": "Invalid image format",
  "detail": "Only JPG, PNG, and WEBP formats are supported"
}
```

---

## Rate Limiting

Currently, no rate limiting is implemented. For production, consider:
- Per-IP rate limits
- Per-API-key rate limits
- Request throttling

---

## CORS

CORS is enabled for the following origins:
- `http://localhost:3000` (React dev server)
- `http://localhost:5173` (Vite dev server)

Configure additional origins in `backend/app/config.py`.

---

## Interactive API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide:
- Interactive API testing
- Request/response schemas
- Example requests

---

## WebSocket Support (Future)

For real-time predictions, WebSocket support could be added:

```
ws://localhost:8000/ws/predict
```

This would enable:
- Streaming predictions
- Real-time updates
- Progress notifications

---

## Versioning

Current API version: `1.0.0`

Future versions can be accessed via:
- URL versioning: `/api/v1/predict`
- Header versioning: `X-API-Version: 1.0.0`

---

## Best Practices

### Request Guidelines

1. **Image Files**:
   - Use JPG or PNG format
   - Recommended size: 224x224 to 1024x1024 pixels
   - Maximum file size: 10 MB

2. **Text Fields**:
   - Provide clear, complete descriptions
   - Avoid special characters that might cause encoding issues

3. **Metadata**:
   - Provide accurate seller ratings and review counts
   - Use consistent units (e.g., always 0.0-5.0 for ratings)

### Response Handling

1. **Always check status codes** before processing responses
2. **Handle errors gracefully** with user-friendly messages
3. **Cache predictions** when appropriate to reduce API calls
4. **Display loading states** during API calls

---

For implementation details, see:
- [Architecture Documentation](architecture.md)
- [Model Training Notes](model_notes.md)

