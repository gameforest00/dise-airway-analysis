"""
Flask Web Application for DISE Video Analysis (PyTorch)

실행: python app.py
접속: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import json
import sys

# step3_enhanced.py의 클래스 import
sys.path.insert(0, str(Path(__file__).parent))
from step3_enhanced import EnhancedAnalyzer, Config

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = Path(__file__).parent.parent / 'uploads_temp'
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# 허용 확장자
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Analyzer 초기화
config = Config()
analyzer = EnhancedAnalyzer(config)

def allowed_file(filename):
    """허용된 파일 확장자 확인"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """파일 업로드"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv'}), 400
    
    try:
        # 파일 저장
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(str(filepath))
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': str(filepath)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """비디오 분석"""
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    filepath = app.config['UPLOAD_FOLDER'] / filename
    
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # 분석 실행
        print(f"\n{'='*60}")
        print(f"Analyzing: {filename}")
        print(f"{'='*60}\n")
        
        output_dir = analyzer.analyze_video(filepath)
        
        # 결과 요약
        summary = extract_summary(output_dir)
        
        result_summary = {
            'success': True,
            'video_name': filepath.name,
            'report_url': f'/results/{output_dir.name}/report.html',
            'timeline_url': f'/results/{output_dir.name}/timeline.png',
            'summary': summary
        }
        
        return jsonify(result_summary)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def extract_summary(output_dir):
    """결과 요약 추출"""
    try:
        # HTML 파일 직접 파싱해서 통계 추출
        report_html = output_dir / 'report.html'
        
        if not report_html.exists():
            return {'analyzed': False}
        
        # Segment 정보는 frames 폴더에서 추출
        frames_dir = output_dir / 'frames'
        if not frames_dir.exists():
            return {'analyzed': False}
        
        # 프레임 파일명에서 상태 파악
        frame_files = list(frames_dir.glob('seg_*.jpg'))
        
        open_count = len([f for f in frame_files if '_Open.jpg' in f.name])
        partial_count = len([f for f in frame_files if '_Partial.jpg' in f.name])
        close_count = len([f for f in frame_files if '_Close.jpg' in f.name])
        
        total = open_count + partial_count + close_count
        
        # HTML에서 AI 원인 찾기 (영어 버전)
        try:
            with open(report_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Primary Cause 찾기
            import re
            primary_pattern = r'<div class="metric-label">Primary Cause</div>\s*<div class="metric-value">([^<]+)</div>'
            primary_match = re.search(primary_pattern, html_content)
            
            if primary_match:
                main_cause = primary_match.group(1).strip()
            else:
                main_cause = '-'
            
            # Predicted Cause에서 원인들 추출 (Velum, Oropharynx 등)
            # <strong>Velum</strong> <span class='conf'>(78%)</span> 형태
            cause_pattern = r'<strong>([A-Za-z]+)</strong>\s*<span class=[\'"]conf[\'"]>\((\d+)%\)</span>'
            matches = re.findall(cause_pattern, html_content)
            
            causes = [m[0] for m in matches]
            confidences = [int(m[1]) for m in matches]
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                cause_count = len(causes)
            else:
                avg_confidence = 0
                cause_count = 0
            
            # Obstruction Ratio 찾기
            obstruction_pattern = r'<div class="metric-label">Obstruction Ratio</div>\s*<div class="metric-value">([0-9.]+)%</div>'
            obstruction_match = re.search(obstruction_pattern, html_content)
            obstruction_ratio = float(obstruction_match.group(1)) if obstruction_match else 0
            
            # Diagnosis 찾기
            diagnosis_pattern = r'<div class="metric-label">Diagnosis</div>\s*<div class="metric-value"[^>]*>([^<]+)</div>'
            diagnosis_match = re.search(diagnosis_pattern, html_content)
            diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else '-'
            
        except Exception as e:
            print(f"HTML parsing error: {e}")
            main_cause = '-'
            cause_count = 0
            avg_confidence = 0
            obstruction_ratio = 0
            diagnosis = '-'
        
        # Close/Partial 이벤트 수
        problem_events = close_count + partial_count
        
        return {
            'analyzed': True,
            'diagnosis': diagnosis,
            'total_segments': total,
            'open': open_count,
            'open_percent': round(open_count / total * 100, 1) if total > 0 else 0,
            'partial': partial_count,
            'partial_percent': round(partial_count / total * 100, 1) if total > 0 else 0,
            'close': close_count,
            'close_percent': round(close_count / total * 100, 1) if total > 0 else 0,
            'main_cause': main_cause,
            'cause_count': cause_count,
            'confidence': round(avg_confidence, 1),
            'obstruction_ratio': round(obstruction_ratio, 1),
            'problem_events': problem_events,
            'has_timeline': (output_dir / 'timeline.png').exists(),
            'has_report': True
        }
    
    except Exception as e:
        print(f"Summary extraction error: {e}")
        import traceback
        traceback.print_exc()
        return {'analyzed': False}


@app.route('/results/<path:filename>')
def serve_results(filename):
    """결과 파일 서빙"""
    results_dir = config.RESULTS_DIR
    return send_from_directory(results_dir, filename)


@app.route('/health')
def health():
    """서버 상태 확인"""
    return jsonify({
        'status': 'ok',
        'device': str(config.DEVICE),
        'model_loaded': analyzer.ai_model is not None
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print(" DISE Analysis Web Server (PyTorch)")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    print(f"AI Model: {' Loaded' if analyzer.ai_model else '⚠️  Not loaded'}")
    print(f"\n Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f" Results folder: {config.RESULTS_DIR}")
    print(f"\n Open browser: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)