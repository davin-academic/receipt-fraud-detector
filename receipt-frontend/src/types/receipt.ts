export type Verdict = 'CLEAN' | 'SUSPICIOUS' | 'MANIPULATED'

export type AnalysisResult = {
  verdict: Verdict
  max_prob: number
  pct_manipulated: number
}
