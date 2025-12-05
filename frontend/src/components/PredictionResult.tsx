/**
 * Prediction Result Component
 * 
 * This component displays the prediction results:
 * - Authenticity score (0-100%)
 * - Decision (approve/flag/reject)
 * - Explanations from each modality
 * - Optional heatmap visualization
 */

import React from 'react'
import { PredictionResponse } from '../types'
import './PredictionResult.css'

interface PredictionResultProps {
  prediction: PredictionResponse
}

const PredictionResult: React.FC<PredictionResultProps> = ({ prediction }) => {
  const { authenticity_score, decision, explanations } = prediction

  // Convert score to percentage
  const scorePercent = Math.round(authenticity_score * 100)

  // Determine score class for styling
  const getScoreClass = () => {
    if (scorePercent >= 75) return 'score-high'
    if (scorePercent >= 50) return 'score-medium'
    return 'score-low'
  }

  // Get decision badge class
  const getDecisionClass = () => {
    return `decision-badge decision-${decision}`
  }

  // Get decision label
  const getDecisionLabel = () => {
    switch (decision) {
      case 'approve':
        return '✅ APPROVE'
      case 'flag':
        return '⚠️ FLAG'
      case 'reject':
        return '❌ REJECT'
      default:
        return decision.toUpperCase()
    }
  }

  return (
    <div className="prediction-result">
      <div className="card">
        <h2>Prediction Results</h2>

        {/* Score Display */}
        <div className="score-section">
          <div className={`score-display ${getScoreClass()}`}>
            {scorePercent}%
          </div>
          <p className="score-label">Authenticity Score</p>
          <div className={getDecisionClass()}>{getDecisionLabel()}</div>
        </div>

        {/* Explanations */}
        <div className="explanations-section">
          <h3>Analysis Details</h3>

          {explanations.image_reason && (
            <div className="explanation-item">
              <div className="explanation-header">
                <span className="explanation-icon">🖼️</span>
                <strong>Image Analysis</strong>
              </div>
              <p className="explanation-text">{explanations.image_reason}</p>
            </div>
          )}

          {explanations.text_reason && (
            <div className="explanation-item">
              <div className="explanation-header">
                <span className="explanation-icon">📝</span>
                <strong>Text Analysis</strong>
              </div>
              <p className="explanation-text">{explanations.text_reason}</p>
            </div>
          )}

          {explanations.metadata_reason && (
            <div className="explanation-item">
              <div className="explanation-header">
                <span className="explanation-icon">📊</span>
                <strong>Metadata Analysis</strong>
              </div>
              <p className="explanation-text">{explanations.metadata_reason}</p>
            </div>
          )}

          {/* Heatmap Display */}
          {explanations.heatmap && (
            <div className="explanation-item">
              <div className="explanation-header">
                <span className="explanation-icon">🔥</span>
                <strong>Attention Heatmap</strong>
              </div>
              <p className="explanation-text">
                This heatmap shows which parts of the image the model focused on
                when making its prediction. Red areas indicate regions of high
                importance.
              </p>
              <div className="heatmap-container">
                <img
                  src={`data:image/png;base64,${explanations.heatmap}`}
                  alt="Attention heatmap"
                  className="heatmap-image"
                />
              </div>
            </div>
          )}
        </div>

        {/* Interpretation Guide */}
        <div className="interpretation-guide">
          <h4>How to Interpret Results</h4>
          <ul>
            <li>
              <strong>Approve (75-100%):</strong> Product appears authentic based
              on image, text, and metadata analysis.
            </li>
            <li>
              <strong>Flag (50-74%):</strong> Product shows some suspicious
              characteristics. Manual review recommended.
            </li>
            <li>
              <strong>Reject (0-49%):</strong> Product shows strong indicators
              of being counterfeit. Further investigation required.
            </li>
          </ul>
        </div>
      </div>
    </div>
  )
}

export default PredictionResult

