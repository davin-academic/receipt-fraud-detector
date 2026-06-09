# Receipt Fraud Detector

Detects manipulation in receipt images using a two-stage ML pipeline:
1. **YOLO OBB** - detects and crops the date field from the receipt
2. **U-Net** - runs manipulation detection on the cropped date field and generates a heatmap overlay

---

## Requirements

- Docker
- Node.js 18+

---

## Running the app

### 1. Start the backend

```bash
docker compose up --build
```

Backend runs at `http://localhost:8000`
API docs at `http://localhost:8000/docs`

### 2. Start the frontend

```bash
cd receipt-frontend
npm install
npm run dev
```

App runs at `http://localhost:5173`

---

## Usage
1. Train the models in backend/ml/notebooks by following the instructions inside the jupyter notebooks
2. Update paths if necessary inside backend/app/services/object_detection/detection.py and backend/app/services/manipulation_detection/predict_manipulation.py
2. Upload a receipt image, or put test receipts inside backend/app/uploads/test and click one of the **test images** to use a sample
3. The app detects the receipt and date field regions
4. The date field is analyzed for signs of manipulation
5. Results show the verdict, confidence scores, a heatmap overlay on the original receipt, and a zoomed view of the manipulation area if detected
