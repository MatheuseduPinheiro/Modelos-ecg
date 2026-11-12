
import json
from pathlib import Path

CELULAS = []

def markdown(texto):
    CELULAS.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": texto.splitlines(keepends=True)
    })

def codigo(texto):
    CELULAS.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": texto.splitlines(keepends=True)
    })


markdown(r"""
# Notebook 1 — SVM Hierárquico de Condições e Subcategorias SCP

## Objetivo

Treinar classificadores SVM independentes para:

- MI — Infarto do miocárdio
- STTC — Alterações ST/T
- CD — Distúrbios de condução
- HYP — Hipertrofias
- FORM — Alterações morfológicas

Além dos cinco classificadores gerais, cada código SCP
pertencente às categorias terá seu próprio classificador
binário quando houver dados suficientes.

## Características

- ECG bruto da PTB-XL, preferencialmente a 500 Hz
- 12 derivações preservadas individualmente
- Características estatísticas e espectrais
- Picos R e características RR estimadas
- Idade exata
- Sexo
- Quatro grupos etários

## Estratégia de classificação

Os códigos são multirrótulo. Um mesmo ECG pode conter
mais de um código SCP.

As probabilidades dos códigos são independentes e não
precisam somar 100%.

**Importante:** uma probabilidade estimada não equivale
a uma certeza diagnóstica.

## Estratégia de validação

- Folds 1–8: treinamento dos SVMs
- Fold 9: escolha de C e calibração
- Fold 10: avaliação independente

A seleção e a calibração no mesmo fold 9 podem produzir
estimativas otimistas nesse conjunto. O fold 10 permanece
intocado até a avaliação final.
""")

markdown(r"""
## 1. Instalação das dependências

Execute no terminal do VS Code:

```powershell
python -m pip install numpy pandas scipy scikit-learn wfdb joblib matplotlib
```

O notebook utiliza `scipy.integrate.trapezoid`, evitando
a dependência de `numpy.trapezoid`.

Configure a pasta BASE na célula seguinte.
""")

codigo(r"""
from pathlib import Path
import ast
import json
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy import signal
from scipy.integrate import trapezoid

import wfdb
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, roc_auc_score,
    average_precision_score, brier_score_loss,
    confusion_matrix
)

SEED = 42

BASE = Path(r"C:\Users\Matheus\Documents\PTB-XL")

METADATA_CSV = BASE / "ptbxl_database.csv"
SCP_CSV = BASE / "scp_statements.csv"

DESTINO = Path(
    r"C:\Users\Matheus\Documents\Modelos\modelos_01"
)

CACHE = BASE / "resultados_svm_reestruturados" / (
    "cache_condicoes_subcategorias_v1.pkl"
)

DESTINO.mkdir(parents=True, exist_ok=True)
CACHE.parent.mkdir(parents=True, exist_ok=True)

# 0 = processar todos os registros.
MAX_REGISTROS = 0

# Controle de custo computacional do SVM RBF.
# 0 = sem limite por classe.
MAX_TREINO_POR_CLASSE = 2500

MIN_POS_TREINO = 30
MIN_POS_CALIBRACAO = 15
MIN_POS_TESTE = 5

C_CANDIDATOS = [0.3, 1.0, 3.0]

LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

GRUPOS = ["MI", "STTC", "CD", "HYP", "FORM"]

assert METADATA_CSV.exists(), METADATA_CSV
assert SCP_CSV.exists(), SCP_CSV

print("BASE:", BASE)
print("DESTINO:", DESTINO)
print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
""")

markdown(r"""
## 2. Leitura e auditoria da PTB-XL

Os códigos individuais são extraídos de `scp_codes`.

O `scp_statements.csv` estabelece a relação entre
códigos e categorias.

- MI, STTC, CD, HYP: coluna `diagnostic_class`
- FORM: coluna `form`

Um código pode participar de mais de uma dimensão
descritiva. Não se força exclusividade artificial.

Registros com NORM e diagnóstico anormal simultâneo
serão identificados e excluídos conforme a regra
utilizada anteriormente no projeto.
""")

codigo(r"""
df = pd.read_csv(METADATA_CSV)
scp = pd.read_csv(SCP_CSV, index_col=0)

scp.index = scp.index.map(str)

def parse_codes(value):
    try:
        obj = ast.literal_eval(value) if isinstance(value, str) else value
        if isinstance(obj, dict):
            return set(str(k).strip() for k in obj)
    except (SyntaxError, ValueError, TypeError):
        pass
    return set()

df["code_set"] = df["scp_codes"].map(parse_codes)

MAPA = {g: set() for g in GRUPOS}
MAPA_CODIGO = {}

for code, row in scp.iterrows():
    code = str(code)
    membership = set()

    diagnostic_class = row.get("diagnostic_class")

    if pd.notna(diagnostic_class):
        group = str(diagnostic_class).strip()
        if group in GRUPOS:
            membership.add(group)

    form = row.get("form")
    if pd.notna(form) and float(form) == 1:
        membership.add("FORM")

    MAPA_CODIGO[code] = membership

    for group in membership:
        MAPA[group].add(code)

def categorias_registro(codes):
    result = set()
    for code in codes:
        result.update(MAPA_CODIGO.get(code, set()))
    return result

df["categories"] = df["code_set"].map(categorias_registro)

DIAGNOSTICOS = {"MI", "STTC", "CD", "HYP"}

df["conflito_norm"] = df.apply(
    lambda r:
        ("NORM" in r["code_set"])
        and bool(r["categories"] & DIAGNOSTICOS),
    axis=1
)

print("Total:", len(df))
print("Conflitos NORM/anormal:", int(df.conflito_norm.sum()))

for grupo in GRUPOS:
    print(grupo, "códigos:", len(MAPA[grupo]))

# Pacientes não devem cruzar os folds existentes.
cross = (
    df.groupby("patient_id", dropna=True)["strat_fold"]
      .nunique()
)

if (cross > 1).any():
    raise ValueError(
        "Existem pacientes em folds diferentes: "
        "verifique a divisão antes do treinamento."
    )

metadata = df.loc[
    ~df["conflito_norm"] & df["code_set"].map(bool)
].copy()

if MAX_REGISTROS:
    metadata = (
        metadata.groupby("strat_fold", group_keys=False)
        .apply(
            lambda x: x.sample(
                n=min(
                    len(x),
                    max(1, MAX_REGISTROS // 10)
                ),
                random_state=SEED
            )
        )
    )

display(
    pd.DataFrame([
        {
            "categoria": g,
            "quantidade_codigos": len(MAPA[g]),
            "codigos": ", ".join(sorted(MAPA[g]))
        }
        for g in GRUPOS
    ])
)
""")

markdown(r"""
## 3. Extração de características dos sinais

São extraídas características por derivação, preservando
a identificação anatômica das 12 derivações.

Inclui:

- Desvio-padrão
- RMS
- Amplitude pico a pico
- MAD
- Variabilidade da primeira diferença
- Energia espectral em 1–5, 5–15 e 15–35 Hz
- Indicadores de qualidade e disponibilidade

A filtragem abaixo é apropriada para características
exploratórias, mas **não é utilizada para medir desvio
clínico de ST**.

A detecção de picos R é heurística e não equivale a
delineação clínica validada.
""")

codigo(r"""
def clean_signal(x, fs):
    x = np.asarray(x, dtype=float)

    if len(x) < 3 * fs:
        return None

    good = np.isfinite(x)

    if good.mean() < 0.99:
        return None

    x = np.where(good, x, np.nanmedian(x[good]))

    hi = min(40.0, fs * 0.4)

    if hi <= 0.5:
        return None

    sos = signal.butter(
        3, [0.5, hi],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return signal.sosfiltfilt(sos, x)


METRICAS_LEAD = [
    "std", "rms", "ptp", "mad",
    "deriv_mad", "energy_1_5",
    "energy_5_15", "energy_15_35"
]

def lead_features(raw, fs, lead):
    x = clean_signal(raw, fs)

    if x is None:
        out = {
            f"{lead}_{k}": np.nan
            for k in METRICAS_LEAD
        }
        out[f"{lead}_valid"] = 0
        return out

    out = {
        f"{lead}_valid": 1,
        f"{lead}_std": float(np.std(x)),
        f"{lead}_rms": float(np.sqrt(np.mean(x*x))),
        f"{lead}_ptp": float(np.ptp(x)),
        f"{lead}_mad": float(
            np.median(np.abs(x - np.median(x)))
        ),
        f"{lead}_deriv_mad": float(
            np.median(np.abs(np.diff(x)))
        )
    }

    freq, psd = signal.welch(
        x, fs=fs,
        nperseg=min(len(x), int(2*fs))
    )

    for a, b, name in [
        (1, 5, "energy_1_5"),
        (5, 15, "energy_5_15"),
        (15, 35, "energy_15_35")
    ]:
        mask = (freq >= a) & (freq < b)

        out[f"{lead}_{name}"] = (
            float(trapezoid(psd[mask], x=freq[mask]))
            if mask.sum() > 1
            else np.nan
        )

    return out


def rhythm_summary(x, fs):
    names = [
        "n_peaks", "hr_bpm", "rr_mean_ms",
        "rr_sd_ms", "rr_cv", "rmssd_ms",
        "rr_p10_ms", "rr_p90_ms"
    ]

    out = {k: np.nan for k in names}

    x = clean_signal(x, fs)

    if x is None:
        return out

    sos = signal.butter(
        2, [5, min(20, fs * 0.4)],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    y = signal.sosfiltfilt(sos, x)

    grad = np.gradient(y)

    width = max(3, int(0.08 * fs))

    energy = np.convolve(
        grad**2,
        np.ones(width) / width,
        mode="same"
    )

    peaks, _ = signal.find_peaks(
        energy,
        distance=max(1, int(0.28 * fs)),
        prominence=max(
            np.percentile(energy, 85) * 0.3,
            1e-12
        )
    )

    refined = []

    for p in peaks:
        left = max(0, p - int(0.075*fs))
        right = min(len(y), p + int(0.075*fs) + 1)

        if right > left:
            refined.append(
                left + int(np.argmax(np.abs(y[left:right])))
            )

    refined = np.unique(refined)

    if len(refined) > 1:
        keep = np.r_[
            True,
            np.diff(refined) > int(0.24 * fs)
        ]
        refined = refined[keep]

    rr = np.diff(refined) / fs

    rr = rr[(rr >= 0.28) & (rr <= 2.5)]

    if len(rr) < 4:
        return out

    out.update({
        "n_peaks": float(len(refined)),
        "hr_bpm": float(60 / np.mean(rr)),
        "rr_mean_ms": float(1000 * np.mean(rr)),
        "rr_sd_ms": float(1000 * np.std(rr)),
        "rr_cv": float(np.std(rr) / np.mean(rr)),
        "rmssd_ms": float(
            1000 * np.sqrt(np.mean(np.diff(rr)**2))
        ),
        "rr_p10_ms": float(1000 * np.quantile(rr, .1)),
        "rr_p90_ms": float(1000 * np.quantile(rr, .9))
    })

    return out


def extract_record(record):
    x = np.asarray(record.p_signal, dtype=float)
    fs = float(record.fs)

    names = list(record.sig_name)

    if x.ndim != 2 or x.shape[0] < int(5 * fs):
        return None

    out = {}

    for lead in LEADS:
        if lead in names:
            raw = x[:, names.index(lead)]
        else:
            raw = np.array([])

        out.update(lead_features(raw, fs, lead))

    reference = (
        "II" if "II" in names
        else ("I" if "I" in names else None)
    )

    if reference:
        out.update(
            rhythm_summary(x[:, names.index(reference)], fs)
        )
    else:
        out.update(
            rhythm_summary(np.array([]), fs)
        )

    return out

print("Extrator definido.")
""")

markdown(r"""
## 4. Extração e cache

O cache guarda características derivadas do sinal,
sem modificar os arquivos WFDB originais.

O contexto de idade e sexo é aplicado posteriormente,
para permitir simulações com os mesmos sinais.

A primeira execução processará os sinais disponíveis.
As execuções seguintes poderão reutilizar o cache.
""")

codigo(r"""
def build_cache(meta, path):
    if path.exists():
        cached = pd.read_pickle(path)

        if (
            isinstance(cached, pd.DataFrame)
            and "ecg_id" in cached.columns
            and not cached.empty
        ):
            print("Cache recuperado:", len(cached))
            return cached

    rows = []
    failures = []

    for i, (_, row) in enumerate(meta.iterrows(), 1):
        record = None

        for field in ["filename_hr", "filename_lr"]:
            filename = row.get(field)

            if not isinstance(filename, str):
                continue

            prefix = BASE / filename

            if not prefix.with_suffix(".hea").exists():
                continue

            try:
                record = wfdb.rdrecord(str(prefix))
                break
            except Exception:
                continue

        if record is None:
            failures.append(
                (row.ecg_id, "WFDB indisponível")
            )
            continue

        try:
            feat = extract_record(record)
        except Exception as exc:
            failures.append(
                (row.ecg_id, str(exc))
            )
            continue

        if feat is None:
            failures.append(
                (row.ecg_id, "sinal inválido")
            )
            continue

        feat["ecg_id"] = int(row.ecg_id)
        rows.append(feat)

        if i % 500 == 0:
            print(
                f"{i}/{len(meta)}; válidos={len(rows)}",
                flush=True
            )

    pd.DataFrame(
        failures,
        columns=["ecg_id", "motivo"]
    ).to_csv(
        path.with_suffix(".falhas.csv"),
        index=False
    )

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError(
            "Nenhum sinal foi processado. "
            "Verifique records500/ e records100/."
        )

    result.to_pickle(path)

    print("Registros aproveitados:", len(result))

    return result


assert (
    (BASE / "records500").exists()
    or (BASE / "records100").exists()
), "Não foram encontradas pastas de sinais WFDB."

features_table = build_cache(metadata, CACHE)
""")

markdown(r"""
## 5. Idade, sexo e faixas etárias

### Grupos da interface

- Pediátrico: 0–16
- Adulto jovem: 17–39
- Adulto: 40–59
- Idoso: 60+

A idade exata também permanece como característica.

Na PTB-XL:

- `sex = 0`: masculino
- `sex = 1`: feminino

As regras exclusivas para adultos utilizam idade
mínima de 18 anos, mesmo quando a interface agrupa
a idade de 17 anos em adulto jovem.

O modelo poderá aprender interações demográficas,
mas não se força uma mudança artificial nas probabilidades.

### Regras clínicas

O motor clínico original descreve limites para
FC, PR, QRS, QTc e ST.

Este notebook só extrai diretamente características
exploratórias de sinal e RR. Não possui delineação
clínica validada de PR, QRS, QTc e ponto J.

Consequentemente, não inventará essas medidas nem
transformará ausência de medição em normalidade.

As regras clínicas completas devem permanecer
separadas do SVM até existirem medidas válidas.
""")

codigo(r"""
def age_group(age):
    if pd.isna(age) or age < 0 or age > 120:
        return "desconhecido"
    if age <= 16:
        return "pediatrico_0_16"
    if age <= 39:
        return "adulto_jovem_17_39"
    if age <= 59:
        return "adulto_40_59"
    return "idoso_60_mais"


AGE_GROUPS = [
    "pediatrico_0_16",
    "adulto_jovem_17_39",
    "adulto_40_59",
    "idoso_60_mais",
    "desconhecido"
]


def contextualize(frame):
    d = frame.copy()

    age = pd.to_numeric(
        d["age"], errors="coerce"
    )

    sex = pd.to_numeric(
        d["sex_ptbxl"], errors="coerce"
    )

    groups = age.map(age_group)

    d["age_missing"] = age.isna().astype(int)

    d["sex_masculino"] = (sex == 0).astype(int)
    d["sex_feminino"] = (sex == 1).astype(int)

    d["sex_desconhecido"] = (
        ~sex.isin([0, 1])
    ).astype(int)

    d["idade_maior_igual_18"] = (
        age >= 18
    ).astype(int)

    for group in AGE_GROUPS:
        d["faixa_" + group] = (
            groups == group
        ).astype(int)

    return d


merged = features_table.merge(
    metadata[
        [
            "ecg_id", "patient_id",
            "age", "sex", "strat_fold",
            "code_set", "categories"
        ]
    ],
    on="ecg_id",
    how="inner",
    validate="one_to_one"
)

merged = merged.rename(
    columns={"sex": "sex_ptbxl"}
)

merged = contextualize(merged)

IGNORAR = {
    "ecg_id", "patient_id", "strat_fold",
    "code_set", "categories"
}

FEATURE_COLS = [
    c for c in merged.columns
    if c not in IGNORAR
    and pd.api.types.is_numeric_dtype(merged[c])
]

print("ECGs:", len(merged))
print("Características:", len(FEATURE_COLS))
print("Folds:", merged.strat_fold.value_counts().sort_index())
""")

markdown(r"""
## 6. Alvos hierárquicos e multirrótulo

Esta etapa corrige a limitação do notebook anterior.

Serão construídos:

1. Cinco alvos gerais.
2. Alvos individuais para os códigos SCP de cada grupo.

O mapeamento real é obtido do `scp_statements.csv`.

Exemplo de nomenclatura:

- `MI`
- `MI__IMI`
- `MI__ASMI`
- `CD__CRBBB`

Os nomes representam tarefas distintas.

O mesmo código SCP pode pertencer a mais de uma
dimensão documentada. Essa informação é preservada.

Um modelo somente será treinado se houver exemplos
positivos e negativos suficientes nos folds exigidos.
""")

codigo(r"""
LABELS = {}
LABEL_INFO = {}

# Cinco classificadores gerais
for group in GRUPOS:
    LABELS[group] = merged["categories"].map(
        lambda groups: int(group in groups)
    ).to_numpy(dtype=int)

    LABEL_INFO[group] = {
        "tipo": "categoria",
        "categoria": group,
        "codigo_scp": None
    }

# Subcategorias individuais
for group in GRUPOS:
    for code in sorted(MAPA[group]):

        name = f"{group}__{code}"

        LABELS[name] = merged["code_set"].map(
            lambda codes, code=code:
                int(code in codes)
        ).to_numpy(dtype=int)

        LABEL_INFO[name] = {
            "tipo": "subcategoria",
            "categoria": group,
            "codigo_scp": code
        }

audit = []

for label, y in LABELS.items():
    item = LABEL_INFO[label]

    audit.append({
        "modelo": label,
        "categoria": item["categoria"],
        "tipo": item["tipo"],
        "codigo_scp": item["codigo_scp"],
        "positivos": int(y.sum()),
        "negativos": int((1-y).sum())
    })

audit = pd.DataFrame(audit)

audit.to_csv(
    DESTINO / "auditoria_alvos.csv",
    index=False
)

display(audit)
print("Total de tarefas candidatas:", len(LABELS))
""")

markdown(r"""
## 7. Treinamento SVM e calibração

### SVM

Cada alvo terá um classificador binário SVC com:

- Kernel RBF
- `class_weight="balanced"`
- Imputação mediana
- Padronização das variáveis
- Valores candidatos de C: 0,3; 1; 3

A pesquisa de C utilizará o fold 9.

### Probabilidades

O `decision_function()` do SVC não representa uma
probabilidade.

Para obter percentuais será utilizada calibração
sigmoidal de Platt:

1. Treinar SVM nos folds 1–8.
2. Obter escores no fold 9.
3. Ajustar uma regressão logística de calibração.
4. Aplicar SVM e calibrador no fold 10.

A calibração e a escolha de C compartilham o fold 9,
portanto o desempenho nesse fold é exploratório.

O desempenho definitivo será reportado no fold 10.

O calibrador aprende probabilidades relativas aos
rótulos anotados, não certezas clínicas.
""")

codigo(r"""
def pipeline_svm(C):
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True
            )
        ),
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=C,
                gamma="scale",
                class_weight="balanced",
                probability=False,
                random_state=SEED
            )
        )
    ])


def sample_train(y, indices, cap, seed):
    rng = np.random.default_rng(seed)
    selected = []

    for value in [0, 1]:
        ids = indices[y[indices] == value]

        if cap and len(ids) > cap:
            ids = rng.choice(
                ids, size=cap, replace=False
            )

        selected.extend(ids.tolist())

    return np.asarray(
        sorted(selected), dtype=int
    )


def metricas(y, probability, threshold=0.5):
    pred = (probability >= threshold).astype(int)

    cm = confusion_matrix(
        y, pred, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    result = {
        "n": int(len(y)),
        "positivos": int(y.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y, pred, zero_division=0)
        ),
        "specificity": (
            float(tn / (tn + fp))
            if tn + fp else np.nan
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(
            roc_auc_score(y, probability)
        )
        result["average_precision"] = float(
            average_precision_score(y, probability)
        )
        result["brier_score"] = float(
            brier_score_loss(y, probability)
        )
    else:
        result.update({
            "roc_auc": np.nan,
            "average_precision": np.nan,
            "brier_score": np.nan
        })

    return result


def platt_fit(scores, targets):
    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=SEED
    )

    calibrator.fit(
        np.asarray(scores).reshape(-1, 1),
        targets
    )

    return calibrator


def probability_from_scores(calibrator, scores):
    return calibrator.predict_proba(
        np.asarray(scores).reshape(-1, 1)
    )[:, 1]


print("Treinamento e calibração definidos.")
""")

markdown(r"""
## 8. Treinar os classificadores gerais e específicos

Para cada alvo:

1. Confirmar quantidade de amostras.
2. Selecionar hiperparâmetro usando fold 9.
3. Ajustar o calibrador.
4. Avaliar no fold 10.
5. Salvar o pipeline e o calibrador.
6. Salvar a lista de características.
7. Registrar a categoria e o código SCP.

Modelos ignorados serão documentados separadamente.

**Atenção ao custo:** SVM RBF com muitos alvos pode
exigir bastante memória e processamento. A configuração
de amostragem do treinamento pode ser ajustada.
""")

codigo(r"""
X = merged[FEATURE_COLS].to_numpy(dtype=float)
folds = merged["strat_fold"].to_numpy(dtype=int)

TRAIN = np.flatnonzero(folds <= 8)
VALID = np.flatnonzero(folds == 9)
TEST = np.flatnonzero(folds == 10)

assert all(len(x) for x in [TRAIN, VALID, TEST])

REPORT = []
IGNORADOS = []

for k, (label, y) in enumerate(LABELS.items(), 1):

    n_train = int(y[TRAIN].sum())
    n_valid = int(y[VALID].sum())
    n_test = int(y[TEST].sum())

    if (
        n_train < MIN_POS_TREINO
        or n_valid < MIN_POS_CALIBRACAO
        or n_test < MIN_POS_TESTE
        or len(np.unique(y[TRAIN])) < 2
        or len(np.unique(y[VALID])) < 2
    ):
        IGNORADOS.append({
            "modelo": label,
            "motivo": "amostras insuficientes",
            "train_pos": n_train,
            "valid_pos": n_valid,
            "test_pos": n_test
        })
        continue

    chosen_train = sample_train(
        y, TRAIN,
        MAX_TREINO_POR_CLASSE,
        SEED
    )

    best = None

    for C in C_CANDIDATOS:
        candidate = pipeline_svm(C)

        candidate.fit(
            X[chosen_train],
            y[chosen_train]
        )

        scores = candidate.decision_function(X[VALID])

        # Seleção de C pelo desempenho de discriminação
        ap = average_precision_score(
            y[VALID], scores
        )

        if best is None or ap > best["ap"]:
            best = {
                "model": candidate,
                "C": C,
                "ap": float(ap),
                "scores": scores
            }

    calibrator = platt_fit(
        best["scores"],
        y[VALID]
    )

    test_scores = best["model"].decision_function(
        X[TEST]
    )

    test_prob = probability_from_scores(
        calibrator, test_scores
    )

    metrics = metricas(y[TEST], test_prob)

    info = LABEL_INFO[label]

    artifact = {
        "model": best["model"],
        "calibrator": calibrator,
        "features": list(FEATURE_COLS),
        "label": label,
        "task": "condicoes_e_subcategorias",
        "categoria": info["categoria"],
        "tipo": info["tipo"],
        "codigo_scp": info["codigo_scp"],
        "threshold": 0.5,
        "C": best["C"],
        "calibration": "platt_logistic_fold9",
        "probability_output": True,
        "train_folds": list(range(1, 9)),
        "calibration_fold": 9,
        "test_fold": 10,
        "sex_encoding": {
            "masculino": 0,
            "feminino": 1
        },
        "age_groups": AGE_GROUPS,
        "note": (
            "Probabilidade calibrada de rótulo SCP. "
            "Não é probabilidade clínica validada."
        )
    }

    filename = f"{label}.joblib"

    joblib.dump(
        artifact,
        DESTINO / filename,
        compress=3
    )

    REPORT.append({
        "modelo": label,
        **info,
        "arquivo": filename,
        "C": best["C"],
        "fold9_ap_selecao": best["ap"],
        **metrics
    })

    print(
        f"[{k}/{len(LABELS)}] {label} "
        f"F1={metrics['f1']:.3f} "
        f"AP={metrics['average_precision']:.3f}",
        flush=True
    )

report_df = pd.DataFrame(REPORT)
ignored_df = pd.DataFrame(IGNORADOS)

report_df.to_csv(
    DESTINO / "metricas_todos_modelos.csv",
    index=False
)

ignored_df.to_csv(
    DESTINO / "modelos_nao_treinados.csv",
    index=False
)

print("Modelos treinados:", len(REPORT))
print("Modelos ignorados:", len(IGNORADOS))

display(report_df.head(30))
""")

markdown(r"""
## 9. Inventário de modelos para o visualizador

O inventário identifica:

- Categoria principal
- Subcategoria SCP
- Nome do arquivo `.joblib`
- Características esperadas
- Tipo de probabilidade
- Limiar de classificação

O visualizador deverá carregar esse inventário para
apresentar as subcategorias da condição selecionada.

Os ritmos continuam pertencendo a outro conjunto de
modelos e não são tratados como subcategorias de MI,
STTC, CD, HYP ou FORM.
""")

codigo(r"""
INVENTARIO = {
    "versao": "1.0",
    "tipo": "condicoes_hierarquicas_multirrotulo",
    "categorias": {},
    "feature_cols": list(FEATURE_COLS),
    "folds": {
        "treino": list(range(1, 9)),
        "calibracao": 9,
        "teste": 10
    },
    "probabilidades": (
        "Calibradas por regressão logística no fold 9"
    ),
    "observacao": (
        "Probabilidades de rótulos SCP; "
        "não representam certeza diagnóstica."
    )
}

trained = {
    item["modelo"]: item
    for item in REPORT
}

for group in GRUPOS:

    general = (
        f"{group}.joblib"
        if group in trained
        else None
    )

    children = []

    for code in sorted(MAPA[group]):
        label = f"{group}__{code}"

        if label not in trained:
            continue

        description = None

        if code in scp.index:
            description = scp.loc[code].get(
                "description"
            )

        children.append({
            "codigo": code,
            "descricao": (
                str(description)
                if pd.notna(description)
                else code
            ),
            "modelo": f"{label}.joblib"
        })

    INVENTARIO["categorias"][group] = {
        "modelo_geral": general,
        "subcategorias": children
    }

with open(
    DESTINO / "inventario_modelos.json",
    "w", encoding="utf-8"
) as f:
    json.dump(
        INVENTARIO, f,
        ensure_ascii=False, indent=2
    )

print("Inventário exportado.")
print(
    "Total de modelos disponíveis:",
    len(REPORT)
)
""")

markdown(r"""
## 10. Exemplo de inferência contextual

O mesmo ECG pode ser avaliado usando diferentes
idades e sexos.

A aplicação deve:

1. Preservar as características do sinal.
2. Atualizar a idade e o sexo selecionados.
3. Recalcular os indicadores de faixa etária.
4. Reorganizar o vetor na ordem de treinamento.
5. Executar o modelo e o calibrador.
6. Retornar a probabilidade estimada.

Uma nova inferência é obrigatória quando muda o
contexto, mas uma mudança numérica na probabilidade
não é garantida.

As probabilidades de cada subcategoria são independentes.
""")

codigo(r"""
def predict_artifact(artifact, frame):
    cols = artifact["features"]

    for col in cols:
        if col not in frame.columns:
            raise ValueError(
                f"Característica ausente: {col}"
            )

    x = frame[cols].to_numpy(dtype=float)

    scores = artifact["model"].decision_function(x)

    probabilities = probability_from_scores(
        artifact["calibrator"],
        scores
    )

    return probabilities


def simular_contexto(
    ecg_id,
    idade,
    sexo,
    categoria="MI"
):
    # sexo: 0 masculino, 1 feminino
    if sexo not in (0, 1):
        raise ValueError("Sexo deve ser 0 ou 1.")

    base_row = merged.loc[
        merged.ecg_id == ecg_id
    ].copy()

    if len(base_row) != 1:
        raise ValueError("ECG não encontrado ou duplicado.")

    # Mantém as características reais do ECG.
    base_row["age"] = idade
    base_row["sex_ptbxl"] = sexo

    # Recria indicadores contextuais.
    base_row = contextualize(base_row)

    results = []

    config = INVENTARIO["categorias"][categoria]

    models = []

    if config["modelo_geral"]:
        models.append(config["modelo_geral"])

    models.extend(
        item["modelo"]
        for item in config["subcategorias"]
    )

    for filename in models:
        artifact = joblib.load(
            DESTINO / filename
        )

        p = predict_artifact(
            artifact, base_row
        )[0]

        results.append({
            "modelo": artifact["label"],
            "categoria": artifact["categoria"],
            "codigo_scp": artifact["codigo_scp"],
            "idade": idade,
            "sexo": sexo,
            "probabilidade": float(p),
            "percentual": round(100 * float(p), 2)
        })

    return pd.DataFrame(results)


# Exemplo executável após o treinamento:
#
# ecg_exemplo = int(merged.iloc[0].ecg_id)
#
# display(simular_contexto(
#     ecg_exemplo, idade=25, sexo=0, categoria="MI"
# ))
#
# display(simular_contexto(
#     ecg_exemplo, idade=65, sexo=1, categoria="MI"
# ))
""")

markdown(r"""
## 11. Interpretação dos resultados e limitações

### O que o notebook entrega

- Um classificador geral por categoria elegível.
- Classificadores individuais por código SCP elegível.
- Probabilidades calibradas de rótulo.
- Características das 12 derivações.
- Características de RR estimadas.
- Idade e sexo.
- Indicadores das quatro faixas etárias.
- Avaliação independente no fold 10.
- Inventário para integração no visualizador.

### O que não deve ser afirmado

- Que cada probabilidade representa certeza clínica.
- Que uma subcategoria foi diagnosticada definitivamente.
- Que todas as subcategorias terão dados suficientes.
- Que mudar idade/sexo sempre mudará a pontuação.
- Que PR, QRS, QTc e ST foram medidos clinicamente
  quando o extrator não fornece essas medidas.

### Integração com o visualizador

Ao selecionar MI, apresentar apenas as probabilidades
dos códigos mapeados para MI.

O mesmo deve ocorrer com STTC, CD, HYP e FORM.

A seleção de uma categoria altera a organização da
exibição, não o sinal nem os rótulos originais.

Os modelos de ritmo devem continuar complementares
e separados.

### Próxima validação necessária

Comparar os modelos com e sem características
demográficas, verificar a calibração por faixa etária
e sexo e avaliar desempenho em pacientes independentes
e, idealmente, em bases externas.
""")

NOTEBOOK = {
    "cells": CELULAS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

OUTPUT = Path(
    "01_SVM_Condicoes_Subcategorias_Idade_Sexo.ipynb"
)

OUTPUT.write_text(
    json.dumps(
        NOTEBOOK,
        ensure_ascii=False,
        indent=1
    ),
    encoding="utf-8"
)

print("Notebook criado:", OUTPUT.resolve())
print("Quantidade de células:", len(CELULAS))
