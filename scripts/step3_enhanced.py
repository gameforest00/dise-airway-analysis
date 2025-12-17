"""
Step 3: Enhanced Hybrid Analyzer (Final Fixed Graph)
수정사항:
1. Waveform 그래프에 50%(Partial), 75%(Severe) 기준선 복구
2. 리포트 UI 최적화 유지

실행: python app.py -> 웹에서 확인
"""

import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import webbrowser
import sys


class Config:
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    MODELS_DIR = PROJECT_ROOT / "models_full"
    RESULTS_DIR = PROJECT_ROOT / "results_full"
    VELUM_DIR = PROJECT_ROOT / "Velum"
    OTE_DIR = PROJECT_ROOT / "OTE"
    
    HSV_LOWER = np.array([0, 0, 0])
    HSV_UPPER = np.array([180, 100, 50])
    
    FPS_SAMPLE = 5
    MIN_EVENT_DURATION = 0.5
    
    OPEN_THRESHOLD = 0.5
    PARTIAL_THRESHOLD = 0.25
    
    CAUSE_LABELS = ['no', 'Velum', 'Oropharynx', 'Tongue', 'Epiglottis']
    PHASE_MAP = {'Velum': 0, 'OTE': 1}
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def imread_safe(path):
    try:
        with open(path, 'rb') as f: arr = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except: return None


class MultiInputEfficientNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=0)
        self.img_fc = nn.Sequential(nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.3))
        self.phase_fc = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(288, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes))
    
    def forward(self, img, phase):
        x1 = self.img_fc(self.backbone(img))
        x2 = self.phase_fc(phase)
        return self.classifier(torch.cat([x1, x2], dim=1))


class EnhancedAnalyzer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.ai_model = None
        self.load_ai_model()
        
    def load_ai_model(self):
        """AI 모델 로드 (최신 모델 우선)"""
        try:
            # 1순위: 방금 학습한 최종 모델 (final_model.pth)
            model_path = self.config.MODELS_DIR / "final_model.pth"
            
            # 2순위: 혹시 없으면 베스트 모델 (best_model.pth)
            if not model_path.exists():
                model_path = self.config.MODELS_DIR / "best_model.pth"
            
            # 모델 파일이 존재하면 로드
            if model_path.exists():
                print(f"\n{'='*40}")
                print(f" Loading Model: {model_path.name}")  # 여기서 파일명 확인
                print(f"{'='*40}\n")
                
                self.ai_model = MultiInputEfficientNet()
                
                # 가중치 로드 (안전 모드 경고 무시)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                if 'model_state_dict' in checkpoint:
                    self.ai_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.ai_model.load_state_dict(checkpoint)
                
                self.ai_model.to(self.device).eval()
                print(" Model loaded successfully.")
            else:
                print(f" Error: 모델 파일이 없습니다! ({self.config.MODELS_DIR})")
                print("   Step 2 학습이 정상적으로 완료되었는지 확인해주세요.")
                
        except Exception as e:
            print(f"Model load failed: {e}")
                
    
    def extract_roi(self, frame):
        if frame is None: return None, None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            margin = 0.075
            mx, my = int(w * margin), int(h * margin)
            x, y = max(0, x + mx), max(0, y + my)
            w = min(frame.shape[1] - x, w - 2 * mx)
            h = min(frame.shape[0] - y, h - 2 * my)
            if w > 0 and h > 0: return frame[y:y+h, x:x+w], (x, y, w, h)
        return frame, (0, 0, frame.shape[1], frame.shape[0])
    
    def detect_lumen_improved(self, roi):
        if roi is None or roi.size == 0: return 0, None, None
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.bitwise_or(cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,60])),
                                  cv2.threshold(cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),(7,7),0), 22, 255, cv2.THRESH_BINARY_INV)[1])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num > 1:
                min_area = roi.shape[0]*roi.shape[1]*0.005
                valid = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_area]
                if valid:
                    idx = max(valid, key=lambda i: stats[i, cv2.CC_STAT_AREA])
                    return stats[idx, cv2.CC_STAT_AREA], (labels==idx).astype('uint8')*255, (stats[idx, cv2.CC_STAT_LEFT], stats[idx, cv2.CC_STAT_TOP], stats[idx, cv2.CC_STAT_WIDTH], stats[idx, cv2.CC_STAT_HEIGHT])
            return 0, None, None
        except: return 0, None, None
    
    def predict_cause(self, roi, phase):
        if not self.ai_model or roi is None: return None
        try:
            img = cv2.resize(roi, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            img = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            phase_input = torch.FloatTensor([[phase]]).to(self.device)
            
            with torch.no_grad():
                outputs = self.ai_model(img, phase_input)
                probs = torch.softmax(outputs, dim=1)[0]
                
                top_idx = torch.argmax(probs).item()
                top_conf = probs[top_idx].item()
                
                anat_probs = probs[1:]
                sum_anat = torch.sum(anat_probs).item()
                if sum_anat > 0:
                    norm_probs = anat_probs / sum_anat
                    anat_idx_rel = torch.argmax(norm_probs).item()
                    anat_idx = anat_idx_rel + 1
                    anat_conf = norm_probs[anat_idx_rel].item()
                else:
                    anat_idx = 1; anat_conf = 0.0

            return {
                'top_cause': self.config.CAUSE_LABELS[top_idx],
                'top_conf': top_conf,
                'anat_cause': self.config.CAUSE_LABELS[anat_idx],
                'anat_conf': anat_conf
            }
        except: return None
    
    def analyze_video(self, video_path):
        video_path = Path(video_path)
        output_dir = self.config.RESULTS_DIR / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        gt = self.load_ground_truth(video_path)
        phase_input = self.config.PHASE_MAP.get(gt.get('phase', 'Velum'), 0)
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps == 0: fps = 30.0
        duration = total_frames / fps
        
        sample_interval = max(1, int(fps / self.config.FPS_SAMPLE))
        raw_results = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % sample_interval == 0:
                roi, roi_bbox = self.extract_roi(frame)
                lumen_area, lumen_mask, lumen_bbox = self.detect_lumen_improved(roi)
                ai_result = self.predict_cause(roi, phase_input)
                raw_results.append({
                    'frame': frame_idx, 'time': frame_idx/fps,
                    'lumen_area': lumen_area, 'lumen_mask': lumen_mask,
                    'lumen_bbox': lumen_bbox, 'roi_bbox': roi_bbox, 'ai': ai_result
                })
            frame_idx += 1
        cap.release()
        
        areas = [r['lumen_area'] for r in raw_results]
        max_area = max(areas) if areas else 1
        for r in raw_results:
            r['state'] = 'Open' if r['lumen_area'] >= max_area*self.config.OPEN_THRESHOLD else ('Partial' if r['lumen_area'] >= max_area*self.config.PARTIAL_THRESHOLD else 'Close')
            
        segments = self.build_segments(raw_results)
        self.save_visual_frames(video_path, segments, raw_results, output_dir)
        self.plot_timeline(raw_results, segments, max_area, output_dir)
        self.generate_html_report(video_path, segments, duration, output_dir)
        return output_dir
    
    def build_segments(self, raw_results):
        if not raw_results: return []
        segments = []
        curr_state = raw_results[0]['state']
        start_time = raw_results[0]['time']
        start_idx = 0
        preds = []
        if raw_results[0]['ai']: preds.append(raw_results[0]['ai'])
        
        for i in range(1, len(raw_results)):
            r = raw_results[i]
            if r['state'] != curr_state:
                seg = {'id': len(segments)+1, 'state': curr_state, 'start_time': start_time, 'end_time': r['time'], 'duration': r['time']-start_time, 'mid_idx': (start_idx+i)//2, 'img_file': ''}
                self.assign_cause(seg, preds)
                segments.append(seg)
                curr_state = r['state']; start_time = r['time']; start_idx = i; preds = []
            if r['ai']: preds.append(r['ai'])
            
        last = {'id': len(segments)+1, 'state': curr_state, 'start_time': start_time, 'end_time': raw_results[-1]['time'], 'duration': raw_results[-1]['time']-start_time, 'mid_idx': (start_idx+len(raw_results)-1)//2, 'img_file': ''}
        self.assign_cause(last, preds)
        segments.append(last)
        return segments

    def assign_cause(self, seg, preds):
        if not preds: seg['cause']='no'; seg['confidence']=0; return
        if seg['state'] == 'Open':
            seg['cause'] = 'no'
            seg['confidence'] = 0
        else:
            causes = [p['anat_cause'] for p in preds]
            confs = [p['anat_conf'] for p in preds]
            if causes:
                common = Counter(causes).most_common(1)[0][0]
                avg_conf = np.mean([c for i, c in enumerate(confs) if causes[i] == common])
                seg['cause'] = common
                seg['confidence'] = avg_conf
            else:
                seg['cause'] = 'Uncertain'

    def save_visual_frames(self, video_path, segments, raw_results, output_dir):
        frames_dir = output_dir / 'frames'; frames_dir.mkdir(exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        for seg in segments:
            data = raw_results[seg['mid_idx']]
            cap.set(cv2.CAP_PROP_POS_FRAMES, data['frame'])
            ret, frame = cap.read()
            if not ret: continue
            rx, ry, rw, rh = data['roi_bbox']
            roi = frame[ry:ry+rh, rx:rx+rw].copy()
            if data['lumen_mask'] is not None:
                roi = cv2.addWeighted(roi, 0.7, cv2.merge([np.zeros_like(data['lumen_mask']), data['lumen_mask'], data['lumen_mask']]), 0.3, 0)
            
            color = {'Open':(0,200,0), 'Partial':(0,165,255), 'Close':(0,0,255)}.get(seg['state'])
            cv2.rectangle(roi, (0,0), (roi.shape[1],30), color, -1)
            txt = f"#{seg['id']} {seg['state']}"
            if seg['state'] != 'Open': txt += f" - {seg['cause']} ({seg.get('confidence',0):.0%})"
            cv2.putText(roi, txt, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            fname = f"seg_{seg['id']:02d}_{seg['state']}.jpg"
            cv2.imwrite(str(frames_dir/fname), roi)
            seg['img_file'] = fname
        cap.release()

    def plot_timeline(self, raw_results, segments, max_area, output_dir):
        times = [r['time'] for r in raw_results]
        collapse = [(1 - r['lumen_area']/max_area)*100 if max_area>0 else 100 for r in raw_results]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Waveform Plot
        ax1.plot(times, collapse, color='#e74c3c', lw=2)
        ax1.fill_between(times, collapse, 100, color='#e74c3c', alpha=0.2)
        
        # [복구된 부분] 50%, 75% 기준선 추가
        ax1.axhline(50, color='orange', linestyle='--', label='Partial (50%)', alpha=0.8)
        ax1.axhline(75, color='red', linestyle='--', label='Collapse (75%)', alpha=0.8)
        
        ax1.set_ylim(0, 105)
        ax1.invert_yaxis()
        ax1.set_ylabel('Collapse (%)')
        ax1.legend(loc='lower right')
        ax1.grid(alpha=0.3)
        
        # Timeline Bar
        for s in segments:
            c = {'Open':'#2ecc71', 'Partial':'#f39c12', 'Close':'#e74c3c'}.get(s['state'])
            ax2.barh(0, s['duration'], left=s['start_time'], height=0.6, color=c, edgecolor='white')
            if s['state']!='Open' and s['duration']>1.0: 
                ax2.text(s['start_time']+s['duration']/2, 0, s['cause'], ha='center', va='center', color='white', fontsize=8, fontweight='bold')
        
        ax2.set_yticks([])
        ax2.set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(output_dir/'timeline.png')
        plt.close()

    def generate_html_report(self, video_path, segments, duration, output_dir):
        obs_events = [s for s in segments if s['state'] in ['Close', 'Partial'] and s['duration'] >= 0.5]
        
        if obs_events:
            max_duration = max(s['duration'] for s in obs_events)
            max_event = max(obs_events, key=lambda x: x['duration'])
            p_cause = max_event['cause']
        else:
            max_duration = 0.0
            p_cause = "-"

        total_obs_time = sum(s['duration'] for s in obs_events)
        obs_ratio = (total_obs_time / duration * 100) if duration > 0 else 0
        dur_color = "#e74c3c" if max_duration >= 10.0 else ("#e67e22" if max_duration >= 5.0 else "#2ecc71")
        
        rows = ""
        for s in segments:
            cls = s['state'].lower()
            bg = "#fef2f2" if cls=="close" else ("#fffbeb" if cls=="partial" else "#ffffff")
            st_col = "#dc2626" if cls=="close" else ("#d97706" if cls=="partial" else "#16a34a")
            
            if s['state'] == 'Open':
                c_html = "-"
            elif s['cause'] != 'no':
                c_html = f"<strong>{s['cause']}</strong> <small>({s.get('confidence',0):.0%})</small>"
            else:
                c_html = "<span style='color:gray'>Unspecified</span>"

            rows += f"<tr style='background:{bg}; border-bottom:1px solid #eee'><td style='padding:10px; text-align:center'>{s['id']}</td><td style='text-align:center'>{s['start_time']:.1f}-{s['end_time']:.1f}s</td><td style='text-align:center'><strong>{s['duration']:.2f}s</strong></td><td style='color:{st_col}; font-weight:bold'>{s['state'].upper()}</td><td>{c_html}</td><td style='text-align:center'><img src='frames/{s['img_file']}' class='thumb-img' onclick='window.open(this.src)'></td></tr>"
            
        html = f"""<!DOCTYPE html><html><head><link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet"><style>body{{font-family:sans-serif;background:#f8fafc}}.card{{background:white;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);padding:20px}}.thumb-img{{height:50px;border-radius:4px;cursor:pointer;transition:transform 0.2s;border:1px solid #ddd}}.thumb-img:hover{{transform:scale(2.5);z-index:10;position:relative}}</style></head><body class="p-8"><div class="max-w-6xl mx-auto"><div class="flex justify-between items-center mb-8 border-b pb-4"><h1 class="text-3xl font-bold text-gray-800">DISE AI Analysis Report</h1><div class="text-right text-gray-500"><div>File: {video_path.name}</div><div>Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div></div></div><div class="grid grid-cols-4 gap-4 mb-8"><div class="card text-center border-l-4 border-red-500"><div class="text-xs text-gray-500">MAX COLLAPSE DURATION</div><div class="text-2xl font-bold" style="color:{dur_color}">{max_duration:.1f}s</div></div><div class="card text-center border-l-4 border-indigo-500"><div class="text-xs text-gray-500">PRIMARY CAUSE (AT MAX)</div><div class="text-2xl font-bold">{p_cause}</div></div><div class="card text-center border-l-4 border-yellow-500"><div class="text-xs text-gray-500">TOTAL OBSTRUCTION</div><div class="text-2xl font-bold">{obs_ratio:.1f}%</div></div><div class="card text-center border-l-4 border-green-500"><div class="text-xs text-gray-500">VIDEO DURATION</div><div class="text-2xl font-bold">{duration:.1f}s</div></div></div><div class="card mb-8"><h2 class="text-xl font-bold mb-4">Waveform Analysis</h2><img src="timeline.png" class="w-full rounded"></div><div class="card"><h2 class="text-xl font-bold mb-4">Event Logs</h2><table class="w-full text-left border-collapse"><thead class="bg-gray-100 border-b"><tr><th class="p-3 text-center">#</th><th class="p-3 text-center">Time Range</th><th class="p-3 text-center">Duration</th><th class="p-3">Status</th><th class="p-3">Predicted Cause (AI)</th><th class="p-3 text-center">Frame</th></tr></thead><tbody>{rows}</tbody></table></div><div class="text-center mt-8 text-gray-400 text-sm">Generated by DISE AI Engine</div></div></body></html>"""
        with open(output_dir/'report.html', 'w', encoding='utf-8') as f: f.write(html)
    
    def load_ground_truth(self, video_path):
        try:
            with open(video_path.with_suffix('.json'), 'r', encoding='utf-8') as f: return json.load(f).get('metas', {})
        except: return {}
def main():
    config = Config()
    analyzer = EnhancedAnalyzer(config)
    
    video_path = None

    # 1. 터미널에서 경로를 직접 입력했을 때 (우선순위 1)
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
        if input_path.exists():
            video_path = input_path
        else:
            print(f"입력하신 파일을 찾을 수 없습니다: {input_path}")
            return

    # 2. 경로 입력이 없으면? -> 폴더 뒤져서 첫 번째 영상 자동 선택 (우선순위 2)
    else:
        print("\n 입력된 파일이 없어 테스트용 영상을 자동으로 찾습니다...")
        
        # Velum 폴더 먼저 보고, 없으면 OTE 폴더 봄
        search_dirs = [config.VELUM_DIR, config.OTE_DIR]
        
        for d in search_dirs:
            if d.exists():
                videos = sorted(list(d.glob("*.mp4")))
                if videos:
                    video_path = videos[0]  # 첫 번째 영상 선택
                    print(f" [Auto Select] 발견된 첫 번째 영상으로 테스트합니다.")
                    print(f" 파일: {video_path}")
                    break
    
    # 3. 분석 실행
    if video_path and video_path.exists():
        analyzer.analyze_video(video_path)
        
        # (선택사항) 분석 끝나면 리포트 바로 열어주기
        report_path = config.RESULTS_DIR / video_path.stem / 'report.html'
        if report_path.exists():
            print(f" 리포트를 브라우저에서 엽니다: {report_path}")
            webbrowser.open(str(report_path))
    else:
        print(" 분석할 비디오 파일(.mp4)을 찾을 수 없습니다.")
        print("   Velum 또는 OTE 폴더에 mp4 파일이 있는지 확인해주세요.")

if __name__ == "__main__":
    main()
