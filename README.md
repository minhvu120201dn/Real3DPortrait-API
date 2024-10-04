# FastAPI Docker App

This project sets up a **FastAPI** application for video lip-syncing using **[Real3DPortrait](https://github.com/yerfor/Real3DPortrait)** model, running within a Docker container.

## Setup

1. **Clone the repo**:
    ```bash
    git clone https://github.com/minhvu120201dn/Real3DPortrait-API.git
    cd Real3DPortrait-API
    ```

2. **Build the Docker image**:
    ```bash
    sudo docker build -t real3dportrait-api .
    ```

3. **Run the container**:
    ```bash
    sudo docker run -d -p 8000:8000 real3dportrait-api
    ```

4. Access at: `http://localhost:8000`

## API

### POST `/lip-sync/`
- **Inputs**: `video` (mp4), `audio` (wav)
- **Output**: Generated video

Example:
```bash
curl -X POST 'http://localhost:8000/lip-sync/' -F 'video=@video.mp4' -F 'audio=@audio.wav'
