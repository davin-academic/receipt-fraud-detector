import type { AnalysisResult } from '../types/receipt'

type ResultCardProps = {
  result: AnalysisResult | null
  loading: boolean
  zoomUrl: string | null
}

const verdictConfig = {
  MANIPULATED: { label: 'High probability of manipulation', className: 'high', message: 'This receipt shows signs of manipulation.' },
  SUSPICIOUS: { label: 'Medium probability of manipulation', className: 'medium', message: 'This receipt has suspicious characteristics.' },
  CLEAN: { label: 'Low probability of manipulation', className: 'low', message: 'No manipulation detected.' },
}

function ResultCard({ result, loading, zoomUrl }: ResultCardProps) {
  const config = result ? verdictConfig[result.verdict] : null

  return (
    <section className="panel result-panel">
      <h2>Results</h2>

      {!result && !loading && <p className="muted">No receipt analyzed yet.</p>}
      {loading && <p className="muted">Analysis in progress...</p>}

      {result && config && (
        <div className={`result-card ${config.className}`}>
          <span className="result-badge">{config.label}</span>

          <h3>{config.message}</h3>

          <div className="confidence-wrapper">
            <div className="confidence-header">
              <strong>Max manipulation probability: </strong>
              <span>{(result.max_prob * 100).toFixed(1)}%</span>
            </div>
            <div className="confidence-bar">
              <div
                className={`confidence-fill ${config.className}`}
                style={{ width: `${result.max_prob * 100}%` }}
              />
            </div>
          </div>

          <p><strong>Manipulated pixels:</strong> {result.pct_manipulated.toFixed(2)}%</p>

          {zoomUrl && (
            <div>
              <strong>Manipulation area (zoomed):</strong>
              <img src={zoomUrl} alt="Zoomed manipulation area" style={{ display: 'block', marginTop: '8px', width: '100%', borderRadius: '4px' }} />
            </div>
          )}
        </div>
      )}
    </section>
  )
}

export default ResultCard
