import os
import json
from flask import Flask, request, render_template, send_from_directory, jsonify
from pathlib import Path
from main_pipeline import hybrid_hierarchical_chunking

os.environ['GOOGLE_API_KEY'] = 'AIzaSyAlQvAVQ90nji-Y0yh0nwAYWI0unxxKdGY'

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    output_path = os.path.join(OUTPUT_FOLDER, Path(file.filename).stem, "chunks.json")
    
    # Run the new hybrid pipeline
    hybrid_hierarchical_chunking(
        video_path=filepath,
        output_path=output_path,
        device="cpu"
    )
    
    return jsonify({
        'video_url': f'/static/uploads/{file.filename}',
        'chunks_url': f'/static/output/{Path(file.filename).stem}/chunks.json'
    })

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
