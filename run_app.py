"""
Khởi động SmartCV Web UI
Chạy: python run_app.py
Sau đó mở trình duyệt: http://localhost:8000
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("App.api:app", host="0.0.0.0", port=8000, reload=True)
