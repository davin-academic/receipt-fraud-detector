import { useState, useEffect } from 'react'
import Header from '../components/Header'
import UploadBox from '../components/UploadBox'
import ResultCard from '../components/ResultCard'
import {
  uploadReceipt,
  getDetectionsUrl,
  getResultUrl,
  getZoomUrl,
  analyzeReceipt,
  listTestReceipts,
  loadTestReceipt,
  getTestReceiptUrl,
} from '../api/receiptApi'
import type { AnalysisResult } from '../types/receipt'

type Phase = 'idle' | 'uploading' | 'detecting' | 'analyzing' | 'done'

function HomePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [zoomUrl, setZoomUrl] = useState<string | null>(null)
  const [phase, setPhase] = useState<Phase>('idle')
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [testFiles, setTestFiles] = useState<string[]>([])

  useEffect(() => {
    listTestReceipts().then(setTestFiles).catch(() => {})
  }, [])

  const handleFileSelect = (file: File | null) => {
    setSelectedFile(file)
    setResult(null)
    setZoomUrl(null)
    setError(null)
    setPhase('idle')
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(file ? URL.createObjectURL(file) : null)
  }

  const runPipeline = async (id: string) => {
    setPhase('detecting')
    setPreviewUrl(getDetectionsUrl(id))

    setPhase('analyzing')
    const data = await analyzeReceipt(id)
    setResult(data)
    setPreviewUrl(getResultUrl(id))
    if (data.verdict !== 'CLEAN') setZoomUrl(getZoomUrl(id))
    setPhase('done')
  }

  const handleAnalyze = async () => {
    if (!selectedFile) return
    setError(null)
    setResult(null)
    setZoomUrl(null)

    try {
      setPhase('uploading')
      const id = await uploadReceipt(selectedFile)
      await runPipeline(id)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
      setPhase('idle')
    }
  }

  const handleLoadTest = async (filename: string) => {
    setError(null)
    setResult(null)
    setZoomUrl(null)
    setSelectedFile(null)
    setPreviewUrl(getTestReceiptUrl(filename))
    setPhase('uploading')

    try {
      const id = await loadTestReceipt(filename)
      await runPipeline(id)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
      setPhase('idle')
    }
  }

  const handleReset = () => {
    if (previewUrl && selectedFile) URL.revokeObjectURL(previewUrl)
    setSelectedFile(null)
    setPreviewUrl(null)
    setZoomUrl(null)
    setResult(null)
    setError(null)
    setPhase('idle')
  }

  const phaseLabel: Record<Phase, string> = {
    idle: '',
    uploading: 'Uploading...',
    detecting: 'Detecting regions...',
    analyzing: 'Analyzing for manipulation...',
    done: 'Analysis complete',
  }

  const loading = phase !== 'idle' && phase !== 'done'

  return (
    <main className="app">
      <Header />

      <section className="layout">
        <div>
          <UploadBox
            selectedFile={selectedFile}
            loading={loading}
            onFileSelect={handleFileSelect}
            onAnalyze={handleAnalyze}
            onReset={handleReset}
          />

          {testFiles.length > 0 && (
            <div className="panel">
              <h2>Test images</h2>
              <div className="actions">
                {testFiles.map((filename) => (
                  <button key={filename} onClick={() => handleLoadTest(filename)} disabled={loading}>
                    {filename.replace(/\.[^.]+$/, '').replace(/_/g, ' ')}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="panel">
          <h2>Preview</h2>

          {previewUrl ? (
            <img src={previewUrl} alt="Receipt preview" className="preview-image" />
          ) : (
            <div className="empty-state">No image selected yet.</div>
          )}

          {phaseLabel[phase] && <p className="muted">{phaseLabel[phase]}</p>}
        </div>
      </section>

      {error && <p className="error">{error}</p>}

      <ResultCard result={result} loading={phase === 'analyzing'} zoomUrl={zoomUrl} />
    </main>
  )
}

export default HomePage
