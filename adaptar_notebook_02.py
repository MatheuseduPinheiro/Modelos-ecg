
from pathlib import Path
import json
import textwrap

ORIGEM = Path("02_SVM_Ritmos_Regras_Clinicas.ipynb")
DESTINO = Path("02_SVM_Ritmos_Percentuais_Idade_Sexo.ipynb")

if not ORIGEM.exists():
    raise FileNotFoundError(ORIGEM)

notebook = json.loads(ORIGEM.read_text(encoding="utf-8"))

def texto(cell):
    src = cell.get("source", [])
    return "".join(src) if isinstance(src, list) else src

def atualizar(cell, source):
    cell["source"] = source.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

def markdown(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True)
    }

def code(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True)
    }

# Compatibilidade: usar a função trapezoid do SciPy.
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue

    src = texto(cell)

    if "np.trapz(" in src or "np.trapezoid(" in src:
        src = (
            "from scipy.integrate import trapezoid\n"
            + src.replace("np.trapz(", "trapezoid(")
                 .replace("np.trapezoid(", "trapezoid(")
        )
        atualizar(cell, src)

# Substituir o treinamento anterior por uma versão
# com calibração e exportação individual dos ritmos.
NOVO_TREINAMENTO = r'''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import zipfile

FAIXAS_ETARIAS = {
    "pediatrico": (0, 16),
    "adulto_jovem": (17, 39),
    "adulto": (40, 59),
    "idoso": (60, 120),
}

CODIFICACAO_SEXO = {
    "masculino": 0,
    "feminino": 1,
}

def calibrar_probabilidade(scores_valid, y_valid):
    """Calibração sigmoidal usando somente fold 9."""
    calibrador = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=SEED
    )
    calibrador.fit(
        np.asarray(scores_valid).reshape(-1, 1),
        y_valid
    )
    return calibrador

def probabilidade_calibrada(calibrador, scores):
    return calibrador.predict_proba(
        np.asarray(scores).reshape(-1, 1)
    )[:, 1]

def train_targets(merged, feature_cols, labels, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    X = merged[feature_cols].to_numpy(dtype=float)
    fold = merged.strat_fold.to_numpy(dtype=int)

    tr = np.flatnonzero(fold <= 8)
    va = np.flatnonzero(fold == 9)
    te = np.flatnonzero(fold == 10)

    assert len(tr) and len(va) and len(te)

    # Verificar separação por paciente.
    patient_train = set(merged.iloc[tr].patient_id.dropna())
    patient_val = set(merged.iloc[va].patient_id.dropna())
    patient_test = set(merged.iloc[te].patient_id.dropna())

    assert not (patient_train & patient_val)
    assert not (patient_train & patient_test)
    assert not (patient_val & patient_test)

    reports = []
    skipped = []

    for index, (label, y) in enumerate(labels.items(), 1):
        y = np.asarray(y, dtype=int)

        positivos = {
            "treino": int(y[tr].sum()),
            "validacao": int(y[va].sum()),
            "teste": int(y[te].sum())
        }

        if (
            positivos["treino"] < MIN_POS_TREINO
            or positivos["validacao"] < max(10, MIN_POS_VALID)
            or positivos["teste"] < MIN_POS_TESTE
            or len(np.unique(y[tr])) < 2
            or len(np.unique(y[va])) < 2
            or len(np.unique(y[te])) < 2
        ):
            skipped.append({
                "ritmo": label,
                "motivo": "amostras insuficientes",
                **positivos
            })
            continue

        # Amostragem apenas no conjunto de treinamento.
        train_idx = sample_train_indices(
            y, tr, MAX_TREINO_POR_CLASSE, SEED
        )

        best = None

        for C in [0.3, 1.0, 3.0]:
            model = make_pipeline_svm(C=C, gamma="scale")

            model.fit(X[train_idx], y[train_idx])

            scores_valid = model.decision_function(X[va])

            ap = average_precision_score(
                y[va], scores_valid
            )

            if best is None or ap > best["ap"]:
                best = {
                    "model": model,
                    "C": C,
                    "ap": float(ap),
                    "scores_valid": scores_valid
                }

        calibrador = calibrar_probabilidade(
            best["scores_valid"], y[va]
        )

        scores_test = best["model"].decision_function(X[te])
        probs_test = probabilidade_calibrada(
            calibrador, scores_test
        )

        pred_test = (probs_test >= 0.5).astype(int)

        resultado = {
            "label": label,
            "C": best["C"],
            "train_pos": positivos["treino"],
            "validation": {
                "average_precision_selecao": best["ap"],
                "threshold": 0.5
            },
            "test": {
                "n": int(len(te)),
                "positivos": positivos["teste"],
                "accuracy": float(
                    accuracy_score(y[te], pred_test)
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y[te], pred_test)
                ),
                "precision_pos": float(
                    precision_score(
                        y[te], pred_test, zero_division=0
                    )
                ),
                "recall_pos": float(
                    recall_score(
                        y[te], pred_test, zero_division=0
                    )
                ),
                "f1_pos": float(
                    f1_score(y[te], pred_test, zero_division=0)
                ),
                "roc_auc": float(
                    roc_auc_score(y[te], probs_test)
                ),
                "pr_auc_ap": float(
                    average_precision_score(y[te], probs_test)
                ),
                "brier_score": float(
                    brier_score_loss(y[te], probs_test)
                )
            }
        }

        artifact = {
            "model": best["model"],
            "calibrator": calibrador,
            "features": list(feature_cols),
            "label": label,
            "task": "ritmos",
            "threshold": 0.5,
            "C": best["C"],
            "probability_output": True,
            "calibration": "platt_logistic_fold9",
            "sex_encoding": CODIFICACAO_SEXO,
            "age_groups": FAIXAS_ETARIAS,
            "train_folds": list(range(1, 9)),
            "validation_fold": 9,
            "test_fold": 10,
            "note": (
                "Probabilidade calibrada do rotulo SCP; "
                "nao equivale a certeza diagnostica."
            )
        }

        filename = f"{label}.joblib"

        joblib.dump(
            artifact,
            output_dir / filename,
            compress=3
        )

        resultado["arquivo_modelo"] = filename
        reports.append(resultado)

        print(
            f"[{index}/{len(labels)}] {label} "
            f"F1={resultado['test']['f1_pos']:.3f} "
            f"AP={resultado['test']['pr_auc_ap']:.3f}",
            flush=True
        )

    pd.json_normalize(reports).to_csv(
        output_dir / "metricas_por_ritmo.csv",
        index=False
    )

    pd.DataFrame(skipped).to_csv(
        output_dir / "ritmos_nao_treinados.csv",
        index=False
    )

    inventory = {
        "task": "ritmos_multirrotulo",
        "probabilidades_calibradas": True,
        "calibracao": "Platt",
        "feature_cols": list(feature_cols),
        "faixas_etarias": FAIXAS_ETARIAS,
        "codificacao_sexo": CODIFICACAO_SEXO,
        "ritmos": {
            item["label"]: item["arquivo_modelo"]
            for item in reports
        },
        "quantidade_treinados": len(reports),
        "quantidade_ignorados": len(skipped)
    }

    (output_dir / "inventario_modelos.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Compatibilidade com a célula de inventário original.
    inventory["arquivos_modelo"] = [
        item["arquivo_modelo"] for item in reports
    ]

    with open(
        output_dir / "inventario_modelos.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            inventory, f, ensure_ascii=False, indent=2
        )

    archive = output_dir / "modelos_ritmos_calibrados.zip"

    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED
    ) as z:
        for item in reports:
            filename = item["arquivo_modelo"]
            z.write(output_dir / filename, arcname=filename)

        for filename in [
            "inventario_modelos.json",
            "metricas_por_ritmo.csv",
            "ritmos_nao_treinados.csv"
        ]:
            z.write(output_dir / filename, arcname=filename)

    print("Modelos de ritmo treinados:", len(reports))
    print("Ritmos ignorados:", len(skipped))
    print("Exportação:", output_dir)

    return reports, skipped
'''

found = False

for cell in notebook["cells"]:
    src = texto(cell)

    if (
        cell["cell_type"] == "code"
        and "def train_targets(" in src
    ):
        # Preserva as funções anteriores e redefine o
        # treinamento com probabilidades calibradas.
        atualizar(
            cell,
            src + "\n\n" + textwrap.dedent(NOVO_TREINAMENTO)
        )
        found = True
        break

if not found:
    raise RuntimeError(
        "Não encontrei a função train_targets() "
        "no notebook original."
    )

# Ajustar inventário e instruções.
notebook["cells"].insert(
    0,
    markdown("""# Notebook 2 — Ritmos com Probabilidades Calibradas

Este notebook mantém os classificadores independentes
dos códigos SCP de ritmo e acrescenta probabilidades
calibradas por modelo.

**Idade:** 0–16, 17–39, 40–59 e 60+ anos.

**Sexo:** masculino (0) e feminino (1).

A idade exata e os indicadores de faixa etária são
utilizados como características de entrada.

A probabilidade de cada ritmo é independente.
Os percentuais não precisam somar 100%.

**Observação:** selecionar uma faixa etária na interface
não define automaticamente uma idade exata. O visualizador
precisa enviar uma idade específica para a inferência.

O fold 9 é utilizado na seleção de C e na calibração.
O fold 10 permanece independente para avaliação final.

Os ritmos não são categorias de doenças.
""")
)

# Acrescentar exemplo de inferência contextual.
notebook["cells"].append(
    markdown("""## Simulação de idade e sexo

Após o treinamento, use a função abaixo para avaliar
o mesmo sinal com diferentes idades e sexos.

É necessário reconstruir os indicadores contextuais
com a mesma função utilizada no treinamento.

Uma nova inferência deve ser executada a cada alteração,
mas os percentuais não são obrigados a mudar.
""")
)

notebook["cells"].append(
    code(r'''
def simular_ritmos(ecg_id, idade, sexo):
    if sexo not in (0, 1):
        raise ValueError(
            "Sexo deve ser 0 (masculino) ou 1 (feminino)"
        )

    # Recomeçar a partir das características extraídas
    # do sinal, sem alterar o registro original.
    row = features_table.loc[
        features_table.ecg_id == ecg_id
    ].copy()

    if len(row) != 1:
        raise ValueError("ECG não encontrado")

    row["age"] = idade
    row["sex_ptbxl"] = sexo

    # Reaplicar exatamente o mesmo motor contextual
    # presente no notebook original.
    row = contextualize_features(row)

    results = []

    for path in sorted(MODELOS_DIR.glob("*.joblib")):
        artifact = joblib.load(path)

        if not (
            isinstance(artifact, dict)
            and artifact.get("task") == "ritmos"
            and "calibrator" in artifact
        ):
            continue

        cols = artifact["features"]

        X = row.reindex(columns=cols).to_numpy(dtype=float)

        scores = artifact["model"].decision_function(X)

        probabilities = artifact["calibrator"].predict_proba(
            np.asarray(scores).reshape(-1, 1)
        )[:, 1]

        results.append({
            "ritmo": artifact["label"],
            "idade": idade,
            "sexo": sexo,
            "probabilidade": float(probabilities[0]),
            "percentual": round(
                100 * float(probabilities[0]), 2
            )
        })

    return pd.DataFrame(results).sort_values(
        "percentual", ascending=False
    ) if results else pd.DataFrame(
        columns=["ritmo", "idade", "sexo", "probabilidade", "percentual"]
    )


# Exemplo após treinamento:
#
# ecg_id = int(features_table.iloc[0].ecg_id)
#
# display(simular_ritmos(ecg_id, 25, 0))
# display(simular_ritmos(ecg_id, 65, 1))
''')
)

# Validar sintaxe de todas as células de código.
import ast

for i, cell in enumerate(notebook["cells"], 1):
    if cell["cell_type"] == "code":
        try:
            ast.parse(texto(cell))
        except SyntaxError as exc:
            raise SyntaxError(
                f"Erro de sintaxe na célula {i}: {exc}"
            )

DESTINO.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False),
    encoding="utf-8"
)

print("Notebook gerado:", DESTINO.resolve())
print("Células:", len(notebook["cells"]))
