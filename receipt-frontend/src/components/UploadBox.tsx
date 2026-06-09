import { useState } from 'react'

type UploadBoxProps = {
  selectedFile: File | null
  loading: boolean
  onFileSelect: (file: File | null) => void
  onAnalyze: () => void
  onReset: () => void
}

function UploadBox({
  selectedFile,
  loading,
  onFileSelect,
  onAnalyze,
  onReset,
}: UploadBoxProps) {
  const [isDragging, setIsDragging] = useState(false)

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    onFileSelect(event.target.files?.[0] || null)
  }

  const handleDrop = (event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault()
    setIsDragging(false)

    const file = event.dataTransfer.files?.[0] || null

    if (file && file.type.startsWith('image/')) {
      onFileSelect(file)
    }
  }

  return (
    <div className="panel">
      <h2>Upload receipt</h2>

      <label
        className={`upload-box ${isDragging ? 'dragging' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <input type="file" accept="image/*" onChange={handleInputChange} />

        <span>{selectedFile ? selectedFile.name : 'Choose or drop an image here'}</span>
        <small>PNG, JPG or JPEG</small>
      </label>

      <div className="actions">
        <button onClick={onAnalyze} disabled={!selectedFile || loading}>
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>

        <button className="secondary-button" onClick={onReset}>
          Reset
        </button>
      </div>
    </div>
  )
}

export default UploadBox