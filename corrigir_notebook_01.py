
# ============================================================
# GERADOR CORRIGIDO — NOTEBOOK 1
# ============================================================
# Não depende de número fixo de célula.
# Preserva as células anteriores do notebook.
# Substitui o treinamento por uma implementação completa.
# ============================================================

import ast
import json
from pathlib import Path

PASTA = Path(r"C:\Users\Matheus\Documents\Modelos")

CANDIDATOS = [
    PASTA / "01_SVM_Condicoes_Subcategorias_Idade_Sexo(3).ipynb",
    PASTA / "01_SVM_Condicoes_Subcategorias_Idade_Sexo.ipynb",
]

ORIGEM = next(
    (p for p in CANDIDATOS if p.is_file()),
    None
)

if ORIGEM is None:
    raise FileNotFoundError(
        "Notebook original não encontrado. "
        "Coloque o arquivo (3).ipynb na pasta Modelos."
    )

SAIDA = PASTA / (
    "01_SVM_Condicoes_Subcategorias_Idade_Sexo_CORRIGIDO.ipynb"
)

nb = json.loads(ORIGEM.read_text(encoding="utf-8"))

if nb.get("nbformat") != 4:
    raise RuntimeError("Formato de notebook não suportado.")

TREINAMENTO = r'''
# ============================================================
# NOTEBOOK 1 — TREINAMENTO COMPLETO CORRIGIDO
# CATEGORIAS + SUBCATEGORIAS SCP
# ============================================================

from pathlib import Path
import re
import json
import hashlib
import zipfile
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    brier_score_loss
)

# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

SEED = globals().get("SEED", 42)

DESTINO = Path(
    r"C:\Users\Matheus\Documents\Modelos\modelos_01"
)

PASTAS = {
    "modelos": DESTINO,
    "graficos": DESTINO / "graficos",
    "relatorios": DESTINO / "relatorios",
    "previsoes": DESTINO / "previsoes"
}

for pasta in PASTAS.values():
    pasta.mkdir(parents=True, exist_ok=True)

C_CANDIDATOS = [0.3, 1.0, 3.0]

MIN_POS_TREINAR = 2
MIN_NEG_TREINAR = 2

BOOTSTRAP_N = 100

MAX_TREINO_POR_CLASSE = globals().get(
    "MAX_TREINO_POR_CLASSE", 2500
)

FAIXAS_ETARIAS = {
    "pediatrico": [0, 16],
    "adulto_jovem": [17, 39],
    "adulto": [40, 59],
    "idoso": [60, None]
}

SEX_ENCODING = {
    "masculino": 0,
    "feminino": 1
}

GRUPOS = ["MI", "STTC", "CD", "HYP", "FORM"]

# ============================================================
# 2. VERIFICAÇÕES
# ============================================================

for nome in [
    "merged",
    "FEATURE_COLS",
    "LABELS",
    "LABEL_INFO"
]:
    if nome not in globals():
        raise RuntimeError(
            f"Execute as células anteriores: {nome} ausente."
        )

assert "age" in FEATURE_COLS
assert "sex_ptbxl" in FEATURE_COLS

# Confirmar indicadores das quatro faixas.
INDICADORES_ETARIOS = [
    "faixa_pediatrico_0_16",
    "faixa_adulto_jovem_17_39",
    "faixa_adulto_40_59",
    "faixa_idoso_60_mais"
]

faltando_faixas = [
    c for c in INDICADORES_ETARIOS
    if c not in FEATURE_COLS
]

if faltando_faixas:
    raise RuntimeError(
        "Os indicadores de faixas etárias não estão "
        f"presentes em FEATURE_COLS: {faltando_faixas}. "
        "Execute a célula de contextualização antes "
        "de iniciar o treinamento."
    )

X = merged[FEATURE_COLS].to_numpy(dtype=float)
folds = merged.strat_fold.to_numpy(dtype=int)

TRAIN = np.flatnonzero(
    (folds >= 1) & (folds <= 8)
)
VALID = np.flatnonzero(folds == 9)
TEST = np.flatnonzero(folds == 10)

assert len(TRAIN) > 0
assert len(VALID) > 0
assert len(TEST) > 0

if "patient_id" in merged.columns:
    def pacientes(indices):
        return set(
            merged.iloc[indices].patient_id.dropna()
        )

    p1 = pacientes(TRAIN)
    p2 = pacientes(VALID)
    p3 = pacientes(TEST)

    assert not (p1 & p2), "Vazamento treino/validação"
    assert not (p1 & p3), "Vazamento treino/teste"
    assert not (p2 & p3), "Vazamento validação/teste"

print("Total de candidatos:", len(LABELS))
print("Treino:", len(TRAIN))
print("Validação:", len(VALID))
print("Teste:", len(TEST))

# ============================================================
# 3. NOMES SEGUROS PARA WINDOWS
# ============================================================

def nome_seguro(label):
    nome = re.sub(
        r'[<>:"/\\|?*\x00-\x1f]',
        "_",
        str(label)
    )
    return nome.rstrip(" .") or "sem_nome"


NOMES_ARQUIVOS = {}
usados = set()

for label in LABELS:
    nome = nome_seguro(label)

    if nome.casefold() in usados:
        sufixo = hashlib.sha256(
            str(label).encode("utf-8")
        ).hexdigest()[:8]

        nome = f"{nome}_{sufixo}"

    if nome.casefold() in usados:
        raise RuntimeError("Colisão de nomes de arquivos.")

    usados.add(nome.casefold())
    NOMES_ARQUIVOS[label] = nome

# ============================================================
# 4. PIPELINE SVM
# ============================================================

def criar_pipeline(C):
    try:
        imputer = SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True
        )
    except TypeError:
        imputer = SimpleImputer(
            strategy="median",
            add_indicator=True
        )

    return Pipeline([
        ("imputer", imputer),
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=C,
            gamma="scale",
            class_weight="balanced",
            probability=False,
            random_state=SEED
        ))
    ])


def amostrar_treino(y):
    rng = np.random.default_rng(SEED)

    selecionados = []

    for classe in [0, 1]:
        ids = TRAIN[y[TRAIN] == classe]

        if (
            MAX_TREINO_POR_CLASSE > 0
            and len(ids) > MAX_TREINO_POR_CLASSE
        ):
            ids = rng.choice(
                ids,
                size=MAX_TREINO_POR_CLASSE,
                replace=False
            )

        selecionados.extend(ids.tolist())

    return np.asarray(
        sorted(selecionados),
        dtype=int
    )

# ============================================================
# 5. CALIBRAÇÃO
# ============================================================

def calibrar_platt(scores, y):
    calibrador = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=SEED
    )

    calibrador.fit(
        np.asarray(scores).reshape(-1, 1),
        y
    )

    return calibrador


def prob_platt(calibrador, scores):
    return calibrador.predict_proba(
        np.asarray(scores).reshape(-1, 1)
    )[:, 1]


def calibrar_cv(X_train, y_train, C=1.0):
    positivos = int((y_train == 1).sum())
    negativos = int((y_train == 0).sum())

    n_splits = min(
        3, positivos, negativos
    )

    if n_splits < 2:
        return None

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED
    )

    try:
        calibrated = CalibratedClassifierCV(
            estimator=criar_pipeline(C),
            method="sigmoid",
            cv=cv
        )
    except TypeError:
        calibrated = CalibratedClassifierCV(
            base_estimator=criar_pipeline(C),
            method="sigmoid",
            cv=cv
        )

    calibrated.fit(X_train, y_train)

    return calibrated

# ============================================================
# 6. SELEÇÃO DO LIMIAR
# ============================================================

def selecionar_limiar(y, prob):
    if len(np.unique(y)) < 2:
        return 0.5, np.nan

    p, r, thresholds = precision_recall_curve(
        y, prob
    )

    if len(thresholds) == 0:
        return 0.5, np.nan

    f1 = (
        2 * p[:-1] * r[:-1]
        / np.maximum(p[:-1] + r[:-1], 1e-12)
    )

    melhor = np.max(f1)
    candidatos = np.flatnonzero(
        np.isclose(f1, melhor)
    )

    idx = candidatos[
        np.argmax(p[:-1][candidatos])
    ]

    return float(thresholds[idx]), float(f1[idx])

# ============================================================
# 7. MÉTRICAS
# ============================================================

def calcular_ece(y, prob, bins=10):
    limites = np.linspace(0, 1, bins + 1)

    ids = np.digitize(
        prob, limites[1:-1], right=True
    )

    erro = 0.0

    for i in range(bins):
        mask = ids == i

        if not mask.any():
            continue

        erro += mask.mean() * abs(
            prob[mask].mean() - y[mask].mean()
        )

    return float(erro)


def calcular_metricas(y, prob, threshold):
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    duas_classes = len(np.unique(y)) == 2

    def divisao(num, den):
        return float(num / den) if den else np.nan

    result = {
        "n_teste": len(y),
        "positivos_teste": int(y.sum()),
        "negativos_teste": int((y == 0).sum()),

        "accuracy": float(accuracy_score(y, pred)),

        "balanced_accuracy": (
            float(balanced_accuracy_score(y, pred))
            if duas_classes else np.nan
        ),

        "precision": divisao(tp, tp + fp),
        "sensibilidade": divisao(tp, tp + fn),
        "recall": divisao(tp, tp + fn),
        "especificidade": divisao(tn, tn + fp),

        "f1": (
            float(f1_score(y, pred, zero_division=0))
            if (tp + fp + fn) else np.nan
        ),

        "auc_roc": (
            float(roc_auc_score(y, prob))
            if duas_classes else np.nan
        ),

        "auc_pr_ap": (
            float(average_precision_score(y, prob))
            if duas_classes else np.nan
        ),

        "brier_score": float(
            brier_score_loss(y, prob)
        ),
        "ece": calcular_ece(y, prob),

        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

    return result, pred


# ============================================================
# 8. INTERVALOS DE CONFIANÇA — BOOTSTRAP
# ============================================================

def bootstrap_ic(y, prob, threshold):
    if len(np.unique(y)) != 2:
        return {}

    rng = np.random.default_rng(SEED)

    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)

    nomes = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensibilidade",
        "especificidade",
        "f1",
        "auc_roc",
        "auc_pr_ap",
        "brier_score",
        "ece"
    ]

    series = {k: [] for k in nomes}

    for _ in range(BOOTSTRAP_N):
        ids = np.concatenate([
            rng.choice(idx0, len(idx0), replace=True),
            rng.choice(idx1, len(idx1), replace=True)
        ])

        m, _ = calcular_metricas(
            y[ids], prob[ids], threshold
        )

        for nome in nomes:
            series[nome].append(m[nome])

    result = {}

    for nome, values in series.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]

        if len(arr):
            result[f"{nome}_ic95_inf"] = float(
                np.percentile(arr, 2.5)
            )
            result[f"{nome}_ic95_sup"] = float(
                np.percentile(arr, 97.5)
            )

    return result

# ============================================================
# 9. GRÁFICOS
# ============================================================

def gerar_graficos(label, nome, y, prob, pred, metricas,
                   threshold):
    pasta = PASTAS["graficos"] / nome
    pasta.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(
        y, pred, labels=[0, 1]
    )

    pd.DataFrame(
        cm,
        index=["Real 0", "Real 1"],
        columns=["Previsto 0", "Previsto 1"]
    ).to_csv(
        pasta / "matriz_confusao.csv"
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center"
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusão — {label}")

    fig.tight_layout()
    fig.savefig(
        pasta / "matriz_confusao.png",
        dpi=180
    )
    plt.close(fig)

    # ROC / PR somente com duas classes no teste
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, prob)

        pd.DataFrame({
            "fpr": fpr,
            "tpr": tpr
        }).to_csv(
            pasta / "curva_roc.csv",
            index=False
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(
            fpr, tpr,
            label=f"AUC-ROC = {metricas['auc_roc']:.4f}"
        )
        ax.plot([0, 1], [0, 1], "--")
        ax.set_xlabel("Taxa de falsos positivos")
        ax.set_ylabel("Sensibilidade")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            pasta / "curva_roc.png", dpi=180
        )
        plt.close(fig)

        precision, recall, _ = (
            precision_recall_curve(y, prob)
        )

        pd.DataFrame({
            "recall": recall,
            "precision": precision
        }).to_csv(
            pasta / "curva_pr.csv",
            index=False
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(
            recall, precision,
            label=f"AP = {metricas['auc_pr_ap']:.4f}"
        )
        ax.axhline(
            y.mean(), linestyle="--",
            label="Prevalência"
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precisão")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            pasta / "curva_pr.png", dpi=180
        )
        plt.close(fig)

        fraction, mean_pred = calibration_curve(
            y, prob, n_bins=10,
            strategy="quantile"
        )

        pd.DataFrame({
            "probabilidade_media": mean_pred,
            "frequencia_observada": fraction
        }).to_csv(
            pasta / "calibracao.csv",
            index=False
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "--")
        ax.plot(mean_pred, fraction, "o-")
        ax.set_xlabel("Probabilidade estimada")
        ax.set_ylabel("Frequência observada")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            pasta / "calibracao.png", dpi=180
        )
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        prob[y == 0], bins=25,
        alpha=0.6, label="Ausente"
    )
    ax.hist(
        prob[y == 1], bins=25,
        alpha=0.6, label="Presente"
    )
    ax.axvline(
        threshold, linestyle="--",
        label=f"Limiar {threshold:.4f}"
    )
    ax.legend()
    ax.set_xlabel("Probabilidade")
    ax.set_ylabel("Quantidade")
    fig.tight_layout()
    fig.savefig(
        pasta / "distribuicao_probabilidades.png",
        dpi=180
    )
    plt.close(fig)

# ============================================================
# 10. AUDITORIA, RELATÓRIOS E EXPORTAÇÕES
# ============================================================

REPORT = []
IGNORADOS = []
AUDITORIA = []


def salvar_progresso():
    pd.DataFrame(REPORT).to_csv(
        DESTINO / "metricas_todos_modelos.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(IGNORADOS).to_csv(
        DESTINO / "modelos_nao_treinados.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(AUDITORIA).to_csv(
        DESTINO / "auditoria_todos_modelos.csv",
        index=False,
        encoding="utf-8-sig"
    )


def salvar_relatorio(item, nome):
    pasta = PASTAS["relatorios"] / nome
    pasta.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([item]).to_csv(
        pasta / "metricas.csv",
        index=False,
        encoding="utf-8-sig"
    )

    with open(
        pasta / "metricas.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            item, f,
            ensure_ascii=False,
            indent=2,
            allow_nan=True
        )

    linhas = [
        f"{k}: {v}"
        for k, v in item.items()
    ]

    (pasta / "relatorio.txt").write_text(
        "\n".join(linhas),
        encoding="utf-8"
    )

# ============================================================
# 11. TREINAMENTO
# ============================================================

for numero, (label, y) in enumerate(
    LABELS.items(), 1
):
    y = np.asarray(y, dtype=int)

    info = LABEL_INFO[label]

    n_pos = int(y[TRAIN].sum())
    n_neg = int((y[TRAIN] == 0).sum())

    n_valid = int(y[VALID].sum())
    n_valid_neg = int((y[VALID] == 0).sum())

    n_test = int(y[TEST].sum())
    n_test_neg = int((y[TEST] == 0).sum())

    nome = NOMES_ARQUIVOS[label]
    filename = f"{nome}.joblib"

    entrada = {
        "modelo": label,
        "categoria": info["categoria"],
        "tipo": info["tipo"],
        "codigo_scp": info["codigo_scp"],
        "positivos_treino": n_pos,
        "negativos_treino": n_neg,
        "positivos_validacao": n_valid,
        "negativos_validacao": n_valid_neg,
        "positivos_teste": n_test,
        "negativos_teste": n_test_neg,
        "arquivo": filename,
        "status": "pendente",
        "motivo": ""
    }

    AUDITORIA.append(entrada)

    print(
        f"[{numero}/{len(LABELS)}] {label} | "
        f"Treino +{n_pos} / -{n_neg}"
    )

    # Critério baseado somente no treinamento
    if (
        n_pos < MIN_POS_TREINAR
        or n_neg < MIN_NEG_TREINAR
    ):
        entrada["status"] = "nao_treinavel"
        entrada["motivo"] = (
            "Menos de dois exemplos positivos "
            "ou negativos no treinamento"
        )
        IGNORADOS.append(entrada.copy())
        salvar_progresso()
        continue

    indices_treino = amostrar_treino(y)

    usar_fold9 = (
        n_valid >= 9
        and n_valid_neg >= 9
    )

    if usar_fold9:
        try:
            sel, resto = train_test_split(
                VALID,
                test_size=0.60,
                stratify=y[VALID],
                random_state=SEED
            )

            cal, lim = train_test_split(
                resto,
                test_size=0.50,
                stratify=y[resto],
                random_state=SEED + 1
            )

            usar_fold9 = all(
                len(np.unique(y[idx])) == 2
                for idx in [sel, cal, lim]
            )

        except ValueError:
            usar_fold9 = False

    model = None
    calibrator = None
    threshold = 0.5
    f1_limiar = np.nan
    C_final = 1.0
    ap_selecao = np.nan

    if usar_fold9:
        # Seleção de C no fold 9
        best = None

        for C in C_CANDIDATOS:
            candidate = criar_pipeline(C)

            candidate.fit(
                X[indices_treino],
                y[indices_treino]
            )

            scores = candidate.decision_function(
                X[sel]
            )

            ap = average_precision_score(
                y[sel], scores
            )

            if best is None or ap > best["ap"]:
                best = {
                    "model": candidate,
                    "C": C,
                    "ap": float(ap)
                }

        model = best["model"]
        C_final = best["C"]
        ap_selecao = best["ap"]

        scores_cal = model.decision_function(X[cal])

        calibrator = calibrar_platt(
            scores_cal, y[cal]
        )

        scores_lim = model.decision_function(X[lim])

        p_lim = prob_platt(
            calibrator, scores_lim
        )

        threshold, f1_limiar = selecionar_limiar(
            y[lim], p_lim
        )

        estrategia = "platt_fold9"

        scores_test = model.decision_function(
            X[TEST]
        )

        prob_test = prob_platt(
            calibrator, scores_test
        )

    else:
        # Calibração cruzada no treinamento
        model = calibrar_cv(
            X[indices_treino],
            y[indices_treino]
        )

        if model is None:
            entrada["status"] = "nao_treinavel"
            entrada["motivo"] = (
                "Calibração cruzada inviável"
            )
            IGNORADOS.append(entrada.copy())
            salvar_progresso()
            continue

        estrategia = "calibracao_cv_treino"

        if len(np.unique(y[VALID])) == 2:
            p_valid = model.predict_proba(
                X[VALID]
            )[:, 1]

            threshold, f1_limiar = selecionar_limiar(
                y[VALID], p_valid
            )

        prob_test = model.predict_proba(
            X[TEST]
        )[:, 1]

    # ========================================================
    # 12. AVALIAÇÃO INDEPENDENTE
    # ========================================================

    metricas, pred_test = calcular_metricas(
        y[TEST],
        prob_test,
        threshold
    )

    ic95 = bootstrap_ic(
        y[TEST],
        prob_test,
        threshold
    )

    # ========================================================
    # 13. EXPORTAR MODELO
    # ========================================================

    artifact = {
        "model": model,
        "calibrator": calibrator,
        "features": list(FEATURE_COLS),

        "label": str(label),
        "task": "condicoes_e_subcategorias",
        "categoria": info["categoria"],
        "tipo": info["tipo"],
        "codigo_scp": info["codigo_scp"],

        "threshold": float(threshold),
        "C": C_final,
        "calibration": estrategia,

        "probability_output": True,
        "probability_method": (
            "model.predict_proba"
            if calibrator is None
            else "calibrator.predict_proba"
        ),

        "sex_encoding": SEX_ENCODING,
        "age_groups": FAIXAS_ETARIAS,
        "train_folds": list(range(1, 9)),
        "validation_fold": 9,
        "test_fold": 10,

        "arquivo_modelo": filename,
        "note": (
            "Probabilidade exploratória de rótulo SCP. "
            "Não substitui diagnóstico médico."
        )
    }

    joblib.dump(
        artifact,
        DESTINO / filename,
        compress=3
    )

    # ========================================================
    # 14. EXPORTAR PREVISÕES
    # ========================================================

    dados_teste = merged.iloc[TEST]

    previsoes = pd.DataFrame({
        "ecg_id": dados_teste.ecg_id.to_numpy(),
        "patient_id": dados_teste.patient_id.to_numpy(),
        "idade": dados_teste.age.to_numpy(),
        "sexo": dados_teste.sex_ptbxl.to_numpy(),
        "y_true": y[TEST],
        "y_pred": pred_test,
        "probabilidade": prob_test,
        "percentual": prob_test * 100,
        "threshold": threshold
    })

    previsoes.to_csv(
        PASTAS["previsoes"] / f"{nome}_previsoes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 15. RELATÓRIO E GRÁFICOS
    # ========================================================

    item = {
        "modelo": label,
        "categoria": info["categoria"],
        "tipo": info["tipo"],
        "codigo_scp": info["codigo_scp"],
        "arquivo": filename,
        "estrategia": estrategia,

        "C": C_final,
        "threshold": float(threshold),

        "train_pos": n_pos,
        "valid_pos": n_valid,
        "test_pos": n_test,

        "fold9_ap_selecao": ap_selecao,
        "fold9_f1_limiar": f1_limiar,

        **metricas,
        **ic95
    }

    gerar_graficos(
        label,
        nome,
        y[TEST],
        prob_test,
        pred_test,
        metricas,
        threshold
    )

    salvar_relatorio(item, nome)

    REPORT.append(item)

    entrada["status"] = (
        "treinado"
        if usar_fold9
        else "treinado_raro"
    )

    salvar_progresso()

    print(
        f"  {filename} | "
        f"F1={metricas['f1']} | "
        f"AUC={metricas['auc_roc']}"
    )

# ============================================================
# 16. INVENTÁRIO COMPLETO
# ============================================================

treinados = {
    item["modelo"]: item
    for item in REPORT
}

INVENTARIO = {
    "versao": "5.0",
    "tipo": "condicoes_e_subcategorias",
    "feature_cols": list(FEATURE_COLS),
    "faixas_etarias": FAIXAS_ETARIAS,
    "sexo": SEX_ENCODING,
    "modelos": {},
    "categorias": {}
}

for grupo in GRUPOS:
    INVENTARIO["categorias"][grupo] = {
        "modelo_geral": None,
        "subcategorias": []
    }

for item in AUDITORIA:
    label = item["modelo"]
    grupo = item["categoria"]

    registro = {
        "label": label,
        "categoria": grupo,
        "tipo": item["tipo"],
        "codigo_scp": item["codigo_scp"],
        "status": item["status"],
        "motivo": item["motivo"],
        "arquivo": (
            treinados[label]["arquivo"]
            if label in treinados else None
        )
    }

    INVENTARIO["modelos"][label] = registro

    if item["tipo"] == "categoria":
        INVENTARIO["categorias"][grupo][
            "modelo_geral"
        ] = registro
    else:
        INVENTARIO["categorias"][grupo][
            "subcategorias"
        ].append(registro)

with open(
    DESTINO / "inventario_modelos.json",
    "w", encoding="utf-8"
) as f:
    json.dump(
        INVENTARIO, f,
        indent=2, ensure_ascii=False
    )

salvar_progresso()

# ============================================================
# 17. RESUMO
# ============================================================

report_df = pd.DataFrame(REPORT)
ignored_df = pd.DataFrame(IGNORADOS)
auditoria_df = pd.DataFrame(AUDITORIA)

print("\nTREINAMENTO FINALIZADO")
print("Candidatos:", len(LABELS))
print("Treinados:", len(REPORT))
print("Não treináveis:", len(IGNORADOS))

display(
    auditoria_df.groupby(
        ["categoria", "status"]
    ).size().unstack(fill_value=0)
)

if not report_df.empty:
    display(
        report_df[
            [
                "modelo", "estrategia",
                "accuracy", "sensibilidade",
                "especificidade", "precision",
                "f1", "auc_roc", "auc_pr_ap",
                "brier_score", "ece"
            ]
        ]
    )

# ============================================================
# 18. ZIP DOS RESULTADOS
# ============================================================

zip_path = DESTINO / "resultados_completos_svm.zip"

with zipfile.ZipFile(
    zip_path,
    "w",
    compression=zipfile.ZIP_STORED
) as z:

    for arquivo in DESTINO.rglob("*"):
        if (
            arquivo.is_file()
            and arquivo.resolve() != zip_path.resolve()
        ):
            z.write(
                arquivo,
                arcname=arquivo.relative_to(DESTINO)
            )

print("Saída:", DESTINO.resolve())
print("ZIP:", zip_path.resolve())
'''

# ============================================================
# LOCALIZAR AUTOMATICAMENTE CÉLULA DE TREINAMENTO
# ============================================================

def conteudo(celula):
    src = celula.get("source", [])
    return "".join(src) if isinstance(src, list) else str(src)

candidatas = []

for indice, celula in enumerate(nb["cells"]):

    if celula.get("cell_type") != "code":
        continue

    src = conteudo(celula)

    # Identifica o treinamento pelos dados,
    # não pelo número ou por funções auxiliares.
    if (
        "LABELS.items()" in src
        and "joblib.dump" in src
        and (
            "REPORT.append" in src
            or "train_targets(" in src
        )
    ):
        candidatas.append(indice)

if not candidatas:
    raise RuntimeError(
        "Não foi encontrada uma célula que contenha "
        "o laço LABELS.items(), joblib.dump e REPORT.append. "
        "Confira se abriu o notebook correto."
    )

# Priorizar a última célula de treinamento encontrada.
alvo = candidatas[-1]

# Verificar sintaxe ANTES de modificar o notebook.
ast.parse(TREINAMENTO)

# Substituir apenas a célula de treinamento.
nb["cells"][alvo]["source"] = (
    TREINAMENTO.splitlines(keepends=True)
)
nb["cells"][alvo]["outputs"] = []
nb["cells"][alvo]["execution_count"] = None

# Limpar resultados antigos, sem mudar o código
# de nenhuma outra célula.
for celula in nb["cells"]:
    if celula.get("cell_type") == "code":
        celula["outputs"] = []
        celula["execution_count"] = None

# ============================================================
# SALVAR E VALIDAR NOTEBOOK
# ============================================================

SAIDA.write_text(
    json.dumps(
        nb,
        ensure_ascii=False,
        indent=1
    ),
    encoding="utf-8"
)

verificacao = json.loads(
    SAIDA.read_text(encoding="utf-8")
)

assert len(verificacao["cells"]) == len(nb["cells"])

assert (
    "MIN_POS_TREINAR = 2"
    in conteudo(verificacao["cells"][alvo])
)

print("=" * 60)
print("NOTEBOOK ATUALIZADO")
print("=" * 60)
print("Origem:", ORIGEM.resolve())
print("Destino:", SAIDA.resolve())
print("Célula modificada:", alvo + 1)
print("Total de células:", len(nb["cells"]))
print("Sintaxe do novo treinamento: OK")
