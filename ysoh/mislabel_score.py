"""
mislabel_score 계산

전략:
1) (dup/ood로 의심되는 샘플을 제외하거나 가중치를 낮춘) 학습 데이터로
   k-fold cross-validation 분류기를 학습 (임베딩 -> Clean/Dusty)
2) out-of-fold(OOF) 예측 확률을 얻음 (각 샘플이 한 번도 학습에 쓰이지 않은 fold에서 예측됨)
3) 실제 라벨과 OOF 예측이 크게 불일치할수록 mislabel_score를 높게 부여
   score = |P(dusty) - label|  (label=0이면 P가 높을수록 의심, label=1이면 P가 낮을수록 의심)
4) (선택) 1~3을 반복: mislabel_score가 높은 샘플을 제외하고 재학습 -> self-training으로 정제
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def compute_mislabel_scores(embeddings, labels, exclude_mask=None, sample_weight=None,
                             n_splits=5, n_iterations=2, random_state=42, return_raw=False,
                             C=0.1):
    """
    embeddings: (N, D) 임베딩
    labels: (N,) 0/1 라벨 (신뢰 불가능한 원본 라벨)
    exclude_mask: (N,) bool, True인 샘플은 학습에서 완전히 제외(hard exclusion)
    sample_weight: (N,) float (0~1), 분류기 학습 시 각 샘플의 가중치.
                   예: 1 - ood_score 를 넘기면 ood성이 강할수록 학습 기여도가 줄어듦(soft exclusion).
                   None이면 모두 가중치 1.
    n_iterations: self-training 반복 횟수.
    return_raw: True면 (mislabel_score, oof_pred) 튜플 반환. oof_pred는 마지막 반복의
                out-of-fold 예측 확률(P(dusty)) 원본값 — 분류기 과확신 여부 진단용.
    C: LogisticRegression의 역정규화 강도. 기본 sklearn 값(1.0)은 512차원 임베딩 대비
       클래스 분리가 너무 쉬워 과확신(예측확률이 0/1 근처로 쏠림)을 유발할 수 있어
       기본값을 0.1로 낮춤(정규화 강화). 값을 낮출수록 더 보수적(덜 극단적)인 예측이 됨.
    """
    n = len(labels)
    if exclude_mask is None:
        exclude_mask = np.zeros(n, dtype=bool)
    if sample_weight is None:
        sample_weight = np.ones(n)

    current_exclude = exclude_mask.copy()
    mislabel_score = np.zeros(n)

    for it in range(n_iterations):
        oof_pred = np.zeros(n)
        train_idx_all = np.where(~current_exclude)[0]

        # 클래스별 최소 샘플 수보다 n_splits가 크면 자동으로 줄임 (소규모 테스트용 안전장치)
        min_class_count = np.bincount(labels[train_idx_all]).min()
        effective_splits = max(2, min(n_splits, min_class_count))

        skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
        for fold, (tr_local, va_local) in enumerate(skf.split(
                embeddings[train_idx_all], labels[train_idx_all])):
            tr_idx = train_idx_all[tr_local]
            va_idx_in_train = train_idx_all[va_local]

            scaler = StandardScaler().fit(embeddings[tr_idx])
            X_tr = scaler.transform(embeddings[tr_idx])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=C)
            clf.fit(X_tr, labels[tr_idx], sample_weight=sample_weight[tr_idx])

            # validation fold (학습 제외 대상이 아닌 샘플들) 예측
            X_va = scaler.transform(embeddings[va_idx_in_train])
            oof_pred[va_idx_in_train] = clf.predict_proba(X_va)[:, 1]

            # 학습에서 제외됐던 샘플들도 이 fold의 분류기로 예측해서 보강
            excluded_idx = np.where(current_exclude)[0]
            if len(excluded_idx) > 0:
                X_ex = scaler.transform(embeddings[excluded_idx])
                # 여러 fold 모델의 평균으로 누적
                oof_pred[excluded_idx] += clf.predict_proba(X_ex)[:, 1] / effective_splits

        mislabel_score = np.abs(oof_pred - labels)

        if it < n_iterations - 1:
            # 다음 반복을 위해 상위 5% 추가 제외 (라벨 노이즈로 강하게 의심되는 샘플)
            threshold = np.percentile(mislabel_score, 95)
            current_exclude = current_exclude | (mislabel_score > threshold)

    if return_raw:
        return mislabel_score, oof_pred
    return mislabel_score


def combine_with_dup_signal(mislabel_score, dup_groups_same_label_conflict):
    """
    같은 dup 그룹 내에서 라벨이 서로 다른 경우, 둘 중 하나는 명백히 틀렸을 가능성이 큼.
    이런 충돌 신호를 mislabel_score에 보강할 때 사용하는 보조 함수.
    dup_groups_same_label_conflict: (N,) bool, 그룹 내 라벨 불일치가 있는 샘플 표시
    """
    boosted = mislabel_score.copy()
    boosted[dup_groups_same_label_conflict] = np.maximum(
        boosted[dup_groups_same_label_conflict], 0.7
    )
    return boosted


if __name__ == "__main__":
    # 합성 데이터로 로직 검증 (실제 임베딩 없이 동작 확인)
    rng = np.random.RandomState(0)
    n = 200
    true_labels = rng.randint(0, 2, size=n)
    embeddings = rng.randn(n, 16) + true_labels[:, None] * 2.0  # 라벨에 따라 분리된 가짜 임베딩

    noisy_labels = true_labels.copy()
    flip_idx = rng.choice(n, size=20, replace=False)
    noisy_labels[flip_idx] = 1 - noisy_labels[flip_idx]  # 20개 라벨 오염

    scores = compute_mislabel_scores(embeddings, noisy_labels, n_iterations=2)

    # 오염시킨 샘플들의 평균 점수가 정상 샘플보다 높아야 함 (정상 동작 검증)
    print("오염 샘플 평균 mislabel_score:", scores[flip_idx].mean())
    print("정상 샘플 평균 mislabel_score:", scores[np.setdiff1d(np.arange(n), flip_idx)].mean())
    print("상위 20개 중 실제 오염 샘플 적중 수:", len(set(np.argsort(-scores)[:20]) & set(flip_idx)), "/ 20")