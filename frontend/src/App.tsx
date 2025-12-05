/**
 * Main App Component
 * 
 * This is the root component of the Counterfeit Product Detection application.
 * It contains:
 * - Image upload functionality
 * - Text input fields (title, description)
 * - Metadata inputs (seller rating, reviews)
 * - Prediction display with results
 */

import React, { useState } from 'react'
import './App.css'
import PredictionForm from './components/PredictionForm'
import PredictionResult from './components/PredictionResult'
import { PredictionResponse } from './types'

function App() {
  // State to store the prediction result
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  /**
   * Handle prediction submission
   * This function is called when the user submits the form
   */
  const handlePrediction = async (result: PredictionResponse) => {
    setPrediction(result)
    setError(null)
  }

  /**
   * Handle loading state changes
   */
  const handleLoading = (isLoading: boolean) => {
    setLoading(isLoading)
  }

  /**
   * Handle errors
   */
  const handleError = (errorMessage: string) => {
    setError(errorMessage)
    setPrediction(null)
  }

  /**
   * Clear the current prediction
   */
  const handleClear = () => {
    setPrediction(null)
    setError(null)
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>🔍 Counterfeit Product Detection</h1>
        <p className="subtitle">
          Multi-Modal AI System for Detecting Fake Products
        </p>
      </header>

      <main className="container">
        {/* Error message display */}
        {error && (
          <div className="alert alert-error">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Prediction form */}
        <PredictionForm
          onPrediction={handlePrediction}
          onLoading={handleLoading}
          onError={handleError}
          onClear={handleClear}
        />

        {/* Loading indicator */}
        {loading && (
          <div className="card">
            <div className="spinner"></div>
            <p style={{ textAlign: 'center', marginTop: '10px' }}>
              Analyzing product... This may take a few seconds.
            </p>
          </div>
        )}

        {/* Prediction results */}
        {prediction && !loading && (
          <PredictionResult prediction={prediction} />
        )}
      </main>

      <footer className="App-footer">
        <p>
          Built with React, FastAPI, TensorFlow, and HuggingFace Transformers
        </p>
      </footer>
    </div>
  )
}

export default App

