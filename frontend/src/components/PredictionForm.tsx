/**
 * Prediction Form Component
 * 
 * This component handles:
 * - Image file upload
 * - Text input fields (title, description)
 * - Metadata inputs (seller rating, number of reviews)
 * - Form submission to the API
 */

import React, { useState, useRef } from 'react'
import axios from 'axios'
import { PredictionResponse, PredictionRequest } from '../types'
import './PredictionForm.css'

interface PredictionFormProps {
  onPrediction: (result: PredictionResponse) => void
  onLoading: (loading: boolean) => void
  onError: (error: string) => void
  onClear: () => void
}

const PredictionForm: React.FC<PredictionFormProps> = ({
  onPrediction,
  onLoading,
  onError,
  onClear,
}) => {
  // Form state
  const [image, setImage] = useState<File | null>(null)
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [sellerRating, setSellerRating] = useState<number | undefined>(undefined)
  const [numReviews, setNumReviews] = useState<number | undefined>(undefined)
  const [imagePreview, setImagePreview] = useState<string | null>(null)

  // File input ref
  const fileInputRef = useRef<HTMLInputElement>(null)

  /**
   * Handle image file selection
   */
  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      // Validate file type
      const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
      if (!validTypes.includes(file.type)) {
        onError('Invalid image format. Please upload JPG, PNG, or WEBP files.')
        return
      }

      // Validate file size (10 MB max)
      if (file.size > 10 * 1024 * 1024) {
        onError('Image too large. Maximum size is 10 MB.')
        return
      }

      setImage(file)
      onClear() // Clear previous prediction

      // Create preview
      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  /**
   * Handle form submission
   */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    // Validation
    if (!image) {
      onError('Please upload an image')
      return
    }

    if (!title.trim()) {
      onError('Please enter a product title')
      return
    }

    if (!description.trim()) {
      onError('Please enter a product description')
      return
    }

    // Prepare form data
    const formData = new FormData()
    formData.append('image', image)
    formData.append('title', title.trim())
    formData.append('description', description.trim())

    if (sellerRating !== undefined) {
      formData.append('seller_rating', sellerRating.toString())
    }

    if (numReviews !== undefined) {
      formData.append('num_reviews', numReviews.toString())
    }

    // Call API
    onLoading(true)
    onError(null)

    try {
      const response = await axios.post<PredictionResponse>(
        'http://localhost:8000/api/predict',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      )

      onPrediction(response.data)
    } catch (err: any) {
      if (axios.isAxiosError(err)) {
        const errorMessage =
          err.response?.data?.detail || err.message || 'Failed to get prediction'
        onError(errorMessage)
      } else {
        onError('An unexpected error occurred')
      }
    } finally {
      onLoading(false)
    }
  }

  /**
   * Reset form
   */
  const handleReset = () => {
    setImage(null)
    setTitle('')
    setDescription('')
    setSellerRating(undefined)
    setNumReviews(undefined)
    setImagePreview(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
    onClear()
  }

  return (
    <div className="prediction-form">
      <form onSubmit={handleSubmit}>
        {/* Image Upload Section */}
        <div className="card">
          <h2>1. Upload Product Image</h2>
          <div className="image-upload-section">
            {imagePreview ? (
              <div className="image-preview-container">
                <img src={imagePreview} alt="Preview" className="image-preview" />
                <button
                  type="button"
                  onClick={() => {
                    setImage(null)
                    setImagePreview(null)
                    if (fileInputRef.current) {
                      fileInputRef.current.value = ''
                    }
                  }}
                  className="btn btn-secondary"
                >
                  Remove Image
                </button>
              </div>
            ) : (
              <div className="image-upload-placeholder">
                <label htmlFor="image-upload" className="upload-label">
                  <span>📷 Click to upload image</span>
                  <span className="upload-hint">JPG, PNG, or WEBP (max 10 MB)</span>
                </label>
                <input
                  id="image-upload"
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/jpg,image/png,image/webp"
                  onChange={handleImageChange}
                  className="file-input"
                />
              </div>
            )}
          </div>
        </div>

        {/* Text Input Section */}
        <div className="card">
          <h2>2. Product Information</h2>
          <div className="form-group">
            <label htmlFor="title" className="form-label">
              Product Title *
            </label>
            <input
              id="title"
              type="text"
              className="form-input"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g., Nike Air Max 90 Running Shoes"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="description" className="form-label">
              Product Description *
            </label>
            <textarea
              id="description"
              className="form-textarea"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Enter product description..."
              required
            />
          </div>
        </div>

        {/* Metadata Section */}
        <div className="card">
          <h2>3. Seller Information (Optional)</h2>
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="seller-rating" className="form-label">
                Seller Rating (0.0 - 5.0)
              </label>
              <input
                id="seller-rating"
                type="number"
                className="form-input"
                min="0"
                max="5"
                step="0.1"
                value={sellerRating || ''}
                onChange={(e) =>
                  setSellerRating(e.target.value ? parseFloat(e.target.value) : undefined)
                }
                placeholder="4.5"
              />
            </div>

            <div className="form-group">
              <label htmlFor="num-reviews" className="form-label">
                Number of Reviews
              </label>
              <input
                id="num-reviews"
                type="number"
                className="form-input"
                min="0"
                value={numReviews || ''}
                onChange={(e) =>
                  setNumReviews(e.target.value ? parseInt(e.target.value) : undefined)
                }
                placeholder="1250"
              />
            </div>
          </div>
        </div>

        {/* Submit Buttons */}
        <div className="form-actions">
          <button type="submit" className="btn btn-primary" disabled={!image || !title || !description}>
            🔍 Check Authenticity
          </button>
          <button type="button" onClick={handleReset} className="btn btn-secondary">
            Clear Form
          </button>
        </div>
      </form>
    </div>
  )
}

export default PredictionForm

