/**
 * TypeScript type definitions for the application
 * 
 * These types match the API response schemas from the backend.
 */

export type Decision = 'approve' | 'flag' | 'reject'

export interface ExplanationDetails {
  image_reason?: string | null
  text_reason?: string | null
  metadata_reason?: string | null
  heatmap?: string | null
}

export interface PredictionResponse {
  authenticity_score: number
  decision: Decision
  explanations: ExplanationDetails
}

export interface PredictionRequest {
  image: File
  title: string
  description: string
  seller_rating?: number
  num_reviews?: number
}

