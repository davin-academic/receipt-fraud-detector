import type { AnalysisResult } from '../types/receipt'

const API_BASE = 'http://localhost:8000/api/receipts'

export async function uploadReceipt(file: File): Promise<string> {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(API_BASE, { method: 'POST', body: formData })
  if (!response.ok) throw new Error('Upload failed.')
  const data = await response.json()
  return data.file_id
}

export function getDetectionsUrl(fileId: string): string {
  return `${API_BASE}/${fileId}/detections`
}

export function getResultUrl(fileId: string): string {
  return `${API_BASE}/${fileId}/result`
}

export function getZoomUrl(fileId: string): string {
  return `${API_BASE}/${fileId}/zoom`
}

const TEST_BASE = 'http://localhost:8000/api/test-receipts'

export async function listTestReceipts(): Promise<string[]> {
  const response = await fetch(TEST_BASE)
  if (!response.ok) throw new Error('Failed to load test receipts.')
  return response.json()
}

export async function loadTestReceipt(filename: string): Promise<string> {
  const response = await fetch(`${TEST_BASE}/${filename}/load`, { method: 'POST' })
  if (!response.ok) throw new Error('Failed to load test receipt.')
  const data = await response.json()
  return data.file_id
}

export function getTestReceiptUrl(filename: string): string {
  return `${TEST_BASE}/${filename}`
}

export async function analyzeReceipt(fileId: string): Promise<AnalysisResult> {
  const response = await fetch(`${API_BASE}/${fileId}/analyze`)
  if (!response.ok) throw new Error('Analysis failed.')
  return response.json()
}
