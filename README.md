# ChickenCare AI Backend

## Setup

```sh
pip install -r
```

## Run

```sh
uvicorn main:app --reload
```

- Health check: [http://localhost:8000/](http://localhost:8000/)
- Disease detection: POST an image to [http://localhost:8000/detect-disease/](http://localhost:8000/detect-disease/)

## Frontend Integration

To integrate the disease detection feature in the frontend using a file input and a button, you can use the following code:

```javascript
const handleCapture = async (blob: Blob) => {
  const formData = new FormData()
  formData.append("file", blob, "photo.jpg")

  const res = await fetch("http://localhost:8000/detect-disease", {
    method: "POST",
    body: formData,
  })
  const data = await res.json()
  alert(JSON.stringify(data))
}

// Usage:
<CameraCapture onCapture={handleCapture} />
```