# ========================= realtime_prediction_system.py =========================
import os
import json
import time
import csv
import hashlib
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import deque

# (필요 시 활성화) GK2A 원본 다운로드/전처리를 쓰려면 사용하세요.
# from gk2a_preprocessor import GK2ACloudProcessor

# cloud_convlstm_model.py에 아래 별칭 중 하나가 있어야 합니다.
# - CloudMovementPredictor = ImprovedCloudMovementPredictor   (권장 별칭)
# - 또는 ImprovedCloudMovementPredictor 자체를 import해서 as로 별칭
from cloud_convlstm_model import ImprovedCloudMovementPredictor


# ----------------------------- 설정 -----------------------------
SEQ_LEN = 20          # 입력 프레임 수 (10분 간격 가정)
PRED_STEPS = 1        # 예측 스텝 수 (t+10m 한 스텝)
TARGET_HW = (256, 256)
THRESH = 0.5


# ----------------------------- 유틸 함수들 -----------------------------
def _ensure_hw256(mask: np.ndarray, target_size=TARGET_HW) -> np.ndarray:
    """(H,W) 실수 배열을 target_size로 중앙 크롭/패드."""
    mask = mask.astype(np.float32)
    H, W = target_size
    h, w = mask.shape
    if h >= H and w >= W:
        sh = (h - H) // 2
        sw = (w - W) // 2
        return mask[sh:sh+H, sw:sw+W]
    # pad
    ph = max(0, (H - h) // 2)
    pw = max(0, (W - w) // 2)
    phe = H - h - ph
    pwe = W - w - pw
    return np.pad(mask, ((ph, phe), (pw, pwe)), mode="constant")


def _load_mask_from_processed_pkl(pkl_path: str) -> np.ndarray:
    """
    gk2a_preprocessor.py가 만든 *_processed.pkl에서 마스크 로드.
    - 확률 채널이 있으면 우선 사용: ['prob_mask','cloud_prob','probability','prob']
    - 없으면 binary_mask 사용
    - 값 범위 정규화(0~1), 결측(-1)→0
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    m = None
    for key in ["prob_mask", "cloud_prob", "probability", "prob"]:
        if key in data:
            m = np.asarray(data[key]).astype(np.float32)
            break
    if m is None:
        m = np.asarray(data["binary_mask"]).astype(np.float32)
    m = np.where(m == -1, 0, m)
    if m.max() > 1:
        m = m / m.max()
    return _ensure_hw256(m, TARGET_HW)


def _cloud_coverage(mask_or_prob: np.ndarray, thresh: float = THRESH) -> float:
    """
    이진 마스크 또는 확률맵에서 구름 커버율(%) 반환.
    - 입력이 확률맵이면 thresh 기준 이진화.
    """
    arr = mask_or_prob
    if arr.dtype != np.bool_ and arr.max() <= 1.0 and arr.min() >= 0.0:
        arr = (arr > thresh).astype(np.float32)
    return float(arr.mean() * 100.0)


def _timestamp_from_filename(fname: str) -> datetime | None:
    """
    파일명에서 YYYYMMDDHHMM 추출.
    예: gk2a_ami_le2_cld_ko020lc_202508252150_processed.pkl
    """
    try:
        parts = os.path.basename(fname).split("_")
        for i, p in enumerate(parts):
            if len(p) == 12 and p.isdigit():
                return datetime.strptime(p, "%Y%m%d%H%M")
            if len(p) == 8 and p.isdigit() and i + 1 < len(parts):
                q = parts[i + 1]
                if len(q) == 4 and q.isdigit():
                    return datetime.strptime(p + q, "%Y%m%d%H%M")
    except Exception:
        pass
    return None


def _save_vis(input_seq: np.ndarray, pred_prob: np.ndarray, save_path: str, use_gray_r: bool = False):
    """
    시각화 저장:
    - 입력은 SEQ_LEN 중 '최근 4장'(t-30, -20, -10, -0)을 흑백으로 보여줌
    - 예측은 컬러바 포함
    input_seq: (S, H, W), pred_prob: (H, W)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    S = input_seq.shape[0]
    pick_ids = [max(0, S-4), max(0, S-3), max(0, S-2), max(0, S-1)]
    titles = ["t-30m", "t-20m", "t-10m", "t-0m"] if S >= 4 else [f"t-{(len(pick_ids)-i)*10}m" for i in range(len(pick_ids))]

    plt.figure(figsize=(3 * (len(pick_ids) + 1), 3))
    for i, idx in enumerate(pick_ids):
        plt.subplot(1, len(pick_ids) + 1, i + 1)
        plt.title(titles[i])
        cmap = "gray_r" if use_gray_r else "gray"
        plt.imshow(input_seq[idx], vmin=0, vmax=1, cmap=cmap)
        plt.axis("off")
    plt.subplot(1, len(pick_ids) + 1, len(pick_ids) + 1)
    plt.title("t+10m (prob)")
    im = plt.imshow(pred_prob, vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _save_error_map(gt: np.ndarray, pred_prob: np.ndarray, save_path: str):
    """관측(GT) vs 예측(≥0.5) 오차맵 저장."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pred_bin = (pred_prob >= THRESH).astype(np.float32)
    err = np.abs(gt - pred_bin)
    plt.figure(figsize=(4, 4))
    plt.title("Error map (GT vs Pred≥0.5)")
    plt.imshow(err, cmap="hot", vmin=0, vmax=1)
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _hash_mask(mask: np.ndarray) -> str:
    return hashlib.md5(mask.tobytes()).hexdigest()


# ----------------------- 실시간 예측 시스템 클래스 -----------------------
class RealtimeCloudPredictor:
    """
    실시간 구름 이동 예측 시스템.
    - 최근 SEQ_LEN 프레임(10분 간격 가정)을 버퍼에 쌓아 t+10m 1스텝 예측.
    - 입력은 GK2A 전처리 결과(*_processed.pkl) 또는 (H,W) numpy 배열.
    """

    def __init__(self,
                 model_path: str = "./models/best_cloud_model.pth",
                 data_folder: str = "./realtime_data",
                 device: torch.device | None = None):
        self.model_path = model_path
        self.data_folder = data_folder
        os.makedirs(self.data_folder, exist_ok=True)

        # CSV 로그 파일 경로
        self.log_csv = os.path.join(self.data_folder, "prediction_log.csv")
        if not os.path.exists(self.log_csv):
            with open(self.log_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "run_timestamp", "pred_time", "gt_stamp",
                    "IoU", "Acc", "Brier", "Coverage%",
                    "vis_path", "error_path", "seq_len"
                ])

        # 디바이스
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # 모델 로드
        self.model = self._load_model()

        # 최근 시퀀스 버퍼(프레임/타임스탬프/해시)
        self.data_buffer = deque(maxlen=SEQ_LEN)   # (H,W)
        self.ts_buffer = deque(maxlen=SEQ_LEN)     # datetime
        self.hash_buffer = deque(maxlen=SEQ_LEN)   # md5

        # 예측 기록
        self.predictions = []
        print(f"✅ 실시간 예측 시스템 초기화 완료 (device={self.device}, model={os.path.basename(self.model_path)})")

    def _load_model(self) -> torch.nn.Module:
        model = ImprovedCloudMovementPredictor(
            input_size=TARGET_HW,
            sequence_length=SEQ_LEN,
            prediction_steps=PRED_STEPS
        )
        loaded = False
        if os.path.exists(self.model_path):
            state = torch.load(self.model_path, map_location="cpu")

            cand = None
            if isinstance(state, dict):
                if "state_dict" in state and isinstance(state["state_dict"], dict):
                    cand = state["state_dict"]
                elif "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
                    cand = state["model_state_dict"]
                elif all(isinstance(k, str) for k in state.keys()):
                    cand = state  # 순수 state_dict로 추정

            if cand is not None:
                incompatible = model.load_state_dict(cand, strict=False)
                loaded = True
                miss = len(getattr(incompatible, "missing_keys", []))
                unexp = len(getattr(incompatible, "unexpected_keys", []))
                total_norm = 0.0
                with torch.no_grad():
                    for p in model.parameters():
                        total_norm += float((p.detach().float() ** 2).sum().cpu())
                print(f"✅ 모델 로드 완료: {self.model_path} | missing={miss}, unexpected={unexp}, param_norm={total_norm:.2e}")
            else:
                print("⚠️ 체크포인트 형식에서 state_dict를 찾지 못했습니다.")
        else:
            print(f"⚠️ 모델 파일이 없습니다: {self.model_path} (무게치 없음 상태)")

        if not loaded:
            print("⚠️ 가중치 로드 실패 → 랜덤 초기화 상태일 수 있습니다.")
        model = model.to(self.device).eval()
        return model

    # -------------------- 입력(프레임) 추가 API --------------------
    def add_frame_from_pkl(self, processed_pkl_path: str, timestamp: datetime | None = None, skip_duplicates: bool = True) -> bool:
        """
        *_processed.pkl 한 개를 읽어 버퍼에 추가.
        파일명에서 시간(YYYYMMDD_HHMM) 패턴을 자동 추출해서 timestamp로 사용.
        중복(직전 프레임과 완전 동일)이면 스킵 가능.
        """
        mask = _load_mask_from_processed_pkl(processed_pkl_path)  # (H,W)
        h = _hash_mask(mask)
        if skip_duplicates and len(self.hash_buffer) > 0 and self.hash_buffer[-1] == h:
            print(f"⚠️ 중복 프레임 감지 → 스킵: {os.path.basename(processed_pkl_path)}")
            return False

        ts = _timestamp_from_filename(processed_pkl_path) or timestamp or datetime.now()
        self.data_buffer.append(mask)
        self.ts_buffer.append(ts)
        self.hash_buffer.append(h)
        fname = os.path.basename(processed_pkl_path)
        print(f"➕ 프레임 추가: {fname} at {ts.isoformat()} "
              f"(buffer {len(self.data_buffer)}/{SEQ_LEN}, cover={_cloud_coverage(mask):.1f}%)")
        return True

    def add_frame_from_array(self, mask_2d: np.ndarray, timestamp: datetime | None = None, skip_duplicates: bool = True) -> bool:
        """
        (H,W) numpy 배열(0~1)을 입력으로 받아 버퍼에 추가.
        """
        mask = mask_2d.astype(np.float32)
        mask = np.where(mask == -1, 0, mask)
        if mask.max() > 1:
            mask = mask / mask.max()
        mask = _ensure_hw256(mask, TARGET_HW)
        h = _hash_mask(mask)
        if skip_duplicates and len(self.hash_buffer) > 0 and self.hash_buffer[-1] == h:
            print("⚠️ 중복 프레임(배열) 감지 → 스킵")
            return False

        ts = timestamp or datetime.now()
        self.data_buffer.append(mask)
        self.ts_buffer.append(ts)
        self.hash_buffer.append(h)
        print(f"➕ 프레임 추가(배열): {ts.isoformat()} (buffer {len(self.data_buffer)}/{SEQ_LEN}, cover={_cloud_coverage(mask):.1f}%)")
        return True

    # -------------------------- 예측 API --------------------------
    def make_prediction(self, save_dir: str = "./demo_outputs", use_gray_r: bool = False) -> dict | None:
        """
        최근 SEQ_LEN 프레임으로 t+10m 1스텝 예측.
        - return: {'timestamp', 'prediction_time', 'prediction'(np.ndarray), 'vis_path', 'coverage%'}
        """
        if len(self.data_buffer) < SEQ_LEN:
            print(f"⚠️ 예측 프레임 부족 (필요={SEQ_LEN}, 현재={len(self.data_buffer)})")
            return None

        try:
            seq = np.stack(list(self.data_buffer), axis=0)   # (S, H, W)
            seq_t = torch.from_numpy(seq).float().unsqueeze(0).unsqueeze(2).to(self.device)  # (1,S,1,H,W)

            with torch.no_grad():
                out = self.model(seq_t)                      # (1,1,H,W) 또는 (1,PRED_STEPS,1,H,W) 구조일 수도 있음
                if out.ndim == 5:  # (1,steps,1,H,W) → 한 스텝만 사용
                    out = out[:, 0]
                out_min = float(out.min().detach().cpu()); out_max = float(out.max().detach().cpu())
                if 0.0 <= out_min and out_max <= 1.0:
                    prob = out.cpu().numpy()[0, 0]
                    print(f"[Predict] raw_out in [0,1] → sigmoid 생략 | range=[{out_min:.4f}, {out_max:.4f}]")
                else:
                    prob = torch.sigmoid(out).cpu().numpy()[0, 0]
                    print(f"[Predict] raw_out logits → sigmoid 적용 | range=[{out_min:.4f}, {out_max:.4f}]")

            # 시각화 저장(최근 4장만 보여줌)
            os.makedirs(save_dir, exist_ok=True)
            vis_path = os.path.join(save_dir, f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            _save_vis(seq, prob, vis_path, use_gray_r=use_gray_r)

            pred_time = (self.ts_buffer[-1] if len(self.ts_buffer) else datetime.now()) + timedelta(minutes=10)
            result = {
                "timestamp": datetime.now().isoformat(),
                "prediction_time": pred_time.isoformat(),
                "prediction": prob,               # (H,W) 확률맵
                "vis_path": vis_path,
                "coverage%": _cloud_coverage(prob, thresh=THRESH)
            }
            self.predictions.append(result)
            print(f"🔮 예측 완료: t+10m={pred_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"coverage≈{result['coverage%']:.1f}% | saved → {vis_path}")

            # 라벨 극성(무엇이 1인지) 빠른 점검
            last_obs = self.data_buffer[-1]
            mean_on_ones = float(result["prediction"][last_obs == 1].mean()) if (last_obs == 1).any() else float('nan')
            mean_on_zeros = float(result["prediction"][last_obs == 0].mean()) if (last_obs == 0).any() else float('nan')
            print(f"[Polarity] mean(pred | obs=1)={mean_on_ones:.4f}, mean(pred | obs=0)={mean_on_zeros:.4f}")

            return result

        except Exception as e:
            print(f"❌ 예측 실패: {e}")
            return None

    # ---------------------- 평가/대시보드 ----------------------
    def evaluate_with_next_observation(self,
                                       data_folder: str,
                                       prefix: str,
                                       suffix: str,
                                       result: dict,
                                       save_dir: str = "./demo_outputs",
                                       invert_gt: bool = False) -> dict | None:
        """
        마지막 입력 시각 +10분의 *_processed.pkl을 찾아 정량 평가.
        invert_gt=True 이면 GT를 1-gt로 뒤집어 평가(라벨 극성 보정).
        또한 CSV 로그와 에러맵을 저장합니다.
        """
        if len(self.ts_buffer) == 0 or result is None:
            return None

        gt_ts = self.ts_buffer[-1] + timedelta(minutes=10)
        gt_stamp = gt_ts.strftime("%Y%m%d%H%M")
        gt_name = f"{prefix}{gt_stamp}{suffix}"
        gt_path = os.path.join(data_folder, gt_name)
        if not os.path.exists(gt_path):
            print(f"ℹ️ GT 없음: {gt_path}")
            return None

        gt = _load_mask_from_processed_pkl(gt_path).astype(np.float32)
        if invert_gt:
            gt = 1.0 - gt

        pred = result["prediction"].astype(np.float32)
        pred_bin = (pred >= THRESH).astype(np.float32)

        # IoU, Acc, Brier
        intersection = np.logical_and(pred_bin == 1, gt == 1).sum()
        union = np.logical_or(pred_bin == 1, gt == 1).sum()
        iou = float(intersection / max(1, union))
        acc = float((pred_bin == gt).mean())
        brier = float(((pred - gt) ** 2).mean())

        # 에러맵 저장
        err_path = os.path.join(save_dir, f"error_{gt_stamp}.png")
        _save_error_map(gt, pred, err_path)

        # CSV 로그 저장
        with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                result["timestamp"],
                result["prediction_time"],
                gt_stamp,
                f"{iou:.6f}",
                f"{acc:.6f}",
                f"{brier:.6f}",
                f"{result['coverage%']:.2f}",
                result["vis_path"],
                err_path,
                SEQ_LEN
            ])

        print(f"[Eval @ {gt_stamp}] IoU={iou:.3f}, Acc={acc:.3f}, Brier={brier:.4f} | error_map → {err_path} | invert_gt={invert_gt}")

        return {"IoU": iou, "Acc": acc, "Brier": brier, "gt_path": gt_path, "error_path": err_path}

    def create_prediction_dashboard(self,
                                    last_obs: np.ndarray | None,
                                    last_pred: np.ndarray | None,
                                    save_path: str = "prediction_dashboard.html",
                                    metrics: dict | None = None):
        """
        간단 HTML 대시보드 생성.
        - last_obs: 최근 관측(0~1 이진/확률), (H,W) or None
        - last_pred: 최근 예측 확률, (H,W) or None
        - metrics: {'IoU','Acc','Brier'} 등 평가 결과(있으면 표기)
        """
        obs_cov = _cloud_coverage(last_obs) if last_obs is not None else None
        pred_cov = _cloud_coverage(last_pred) if last_pred is not None else None
        now_kst = datetime.now().strftime("%Y-%m-%d %H:%M KST")
        pred_kst = (datetime.now() + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M KST")

        metrics_html = ""
        if metrics is not None:
            iou = metrics.get("IoU", float("nan"))
            acc = metrics.get("Acc", float("nan"))
            brier = metrics.get("Brier", float("nan"))
            metrics_html = f"""
            <h3>📊 최근 평가 지표</h3>
            <ul>
                <li>IoU: {iou:.3f}</li>
                <li>Acc: {acc:.3f}</li>
                <li>Brier: {brier:.4f}</li>
            </ul>
            """

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>구름 이동 예측 대시보드</title>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="600">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ text-align: center; color: #2c3e50; }}
        .prediction-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
        .prediction-box {{ border: 1px solid #ddd; padding: 15px; border-radius: 8px; }}
        .timestamp {{ color: #7f8c8d; font-size: 14px; }}
        .status {{ padding: 5px 10px; border-radius: 4px; color: white; background-color: #27ae60; display: inline-block; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌤️ 한국 구름 이동 예측 시스템</h1>
        <p>GK2A 위성 데이터 기반 실시간 구름 예측</p>
        <div class="status">운영 중</div>
    </div>

    <div class="prediction-grid">
        <div class="prediction-box">
            <h3>📡 최근 관측</h3>
            <p class="timestamp">업데이트: {now_kst}</p>
            <p>구름 커버: {obs_cov:.1f}%</p>
        </div>

        <div class="prediction-box">
            <h3>🔮 10분 후 예측</h3>
            <p class="timestamp">예측 시간: {pred_kst}</p>
            <p>예상 구름 커버: {pred_cov:.1f}%</p>
        </div>
    </div>

    {metrics_html}
</body>
</html>"""
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"✅ 대시보드 저장: {save_path}")


# --------------------------------- 데모 실행 ---------------------------------
if __name__ == "__main__":
    """
    간단 시현:
    1) 전처리된 GK2A 파일 20개를 시간 순서대로 버퍼에 추가(중복 회피)
    2) t+10m 예측을 수행
    3) 정답(+10분)으로 평가 및 로그/오차맵 저장
    4) 대시보드 생성
    """
    # 1) 모델 경로 자동 선택: improved 파일이 있으면 우선 사용
    default_model = "./models/best_cloud_model.pth"
    improved_model = "./best_cloud_model_improved.pth"  # 업로드/실험 파일 대응
    model_path = improved_model if os.path.exists(improved_model) else default_model

    predictor = RealtimeCloudPredictor(model_path=model_path, data_folder="./realtime_data")

    # 2) 전처리된 pkl 후보(명시적 목록이 있으면 우선 사용)
    explicit_targets = [
        # 필요 시 여기에 20개 경로를 직접 넣을 수 있습니다.
        # 예시(4개만 적어둠): 이후 자동 보충 로직이 나머지를 채움
        "./processed_cloud_data/gk2a_ami_le2_cld_ko020lc_202508252150_processed.pkl",
        "./processed_cloud_data/gk2a_ami_le2_cld_ko020lc_202508252200_processed.pkl",
        "./processed_cloud_data/gk2a_ami_le2_cld_ko020lc_202508252210_processed.pkl",
        "./processed_cloud_data/gk2a_ami_le2_cld_ko020lc_202508252220_processed.pkl",
    ]
    existing_targets = [p for p in explicit_targets if os.path.exists(p)]

    added = 0
    # 2-1) 명시 파일 먼저 넣기(중복은 내부에서 스킵될 수 있음)
    for p in existing_targets:
        ok = predictor.add_frame_from_pkl(p, skip_duplicates=True)
        if ok:
            added += 1
        if added >= SEQ_LEN:
            break

    # 2-2) 부족하면 폴더에서 보충(최신 → 과거 순서로 내려가며 unique하게 채움)
    if added < SEQ_LEN:
        print("ℹ️ 대체 프레임을 탐색합니다.")
        folder = "./processed_cloud_data"
        prefix = "gk2a_ami_le2_cld_ko020lc_"
        suffix = "_processed.pkl"

        candidates = []
        try:
            for fn in os.listdir(folder):
                if fn.startswith(prefix) and fn.endswith(suffix):
                    mid = fn[len(prefix):-len(suffix)]
                    if len(mid) == 12 and mid.isdigit():
                        try:
                            ts = datetime.strptime(mid, "%Y%m%d%H%M")
                            candidates.append((ts, os.path.join(folder, fn)))
                        except Exception:
                            pass
        except FileNotFoundError:
            print(f"⚠️ 폴더가 없습니다: {folder}")

        # 최신 → 과거
        candidates.sort(key=lambda x: x[0], reverse=True)
        # 이미 넣은 경로 제외
        used = set(os.path.abspath(p) for p in existing_targets)
        for ts, p in candidates:
            if os.path.abspath(p) in used:
                continue
            ok = predictor.add_frame_from_pkl(p, skip_duplicates=True)
            if ok:
                added += 1
                used.add(os.path.abspath(p))
            if added >= SEQ_LEN:
                break

    # 2-3) 그래도 부족하면 랜덤으로 채움(데모용)
    if added < SEQ_LEN:
        print(f"⚠️ 유효한 전처리 파일이 {SEQ_LEN}개 미만입니다. 데모용 랜덤 프레임으로 채웁니다.")
        rnd = (np.random.rand(*TARGET_HW) > 0.6).astype(np.float32)
        while added < SEQ_LEN:
            predictor.add_frame_from_array(rnd, skip_duplicates=False)
            added += 1

    if added < SEQ_LEN:
        print("🚫 실시간 구름 예측 시스템 시연 종료(프레임 부족)")
    else:
        # (선택) 스모크 테스트: 모델 출력 범위 확인
        x = torch.randn(1, SEQ_LEN, 1, *TARGET_HW, device=predictor.device)
        with torch.no_grad():
            y = predictor.model(x)
            y_min = float(y.min().detach().cpu())
            y_max = float(y.max().detach().cpu())
        print(f"[SMOKE] model raw_out range on random input: [{y_min:.4f}, {y_max:.4f}]")

        # 3) 예측 실행
        result = predictor.make_prediction(save_dir="./demo_outputs", use_gray_r=False)

        # 3-1) 정답(+10분)으로 평가 및 로그/오차맵 저장
        metrics = None
        if result is not None:
            metrics = predictor.evaluate_with_next_observation(
                data_folder="./processed_cloud_data",
                prefix="gk2a_ami_le2_cld_ko020lc_",
                suffix="_processed.pkl",
                result=result,
                save_dir="./demo_outputs",
                invert_gt=False  # 필요 시 True로 극성 보정
            )

        # 4) 대시보드 생성 (평가 결과 포함)
        if result is not None:
            last_obs = predictor.data_buffer[-1]           # 마지막 관측(형식상)
            last_pred = result["prediction"]               # 확률맵
            predictor.create_prediction_dashboard(last_obs, last_pred,
                                                  save_path="prediction_dashboard.html",
                                                  metrics=metrics)

        print("🚀 실시간 구름 예측 시스템 시연 완료")
# ==============================================================================
