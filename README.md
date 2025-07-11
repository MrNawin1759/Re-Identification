# Soccer Player Tracking and Re-Identification System

This system tracks soccer players in video footage, maintaining consistent IDs even when players leave and re-enter the frame.

## Features
- Player detection using custom YOLOv11 model
- Player tracking with DeepSORT algorithm
- Re-identification when players re-enter frame
- Output video with visualized tracking results

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU (recommended) with CUDA support

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/soccer-player-tracking.git
   cd soccer-player-tracking

2. Create and activate a virtual environment:
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate    # Windows
3. Install dependencies:
    pip install -r requirements.txt
##Usage
Place your input video in the project root directory

Place your custom YOLO model (best.pt) in the project root

Run the tracking script:
python 15_sec.py --input 15sec_input_720p.mp4 --model best.pt
