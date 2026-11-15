import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sk
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

dadosPaises = pd.read_csv("arquivos/originais/country_gun_profile.csv")

alvo = "latest_homicide_rate"
numericas = ["latitude", "longitude", "urban_population_pct_2017",
             "civilian_firearms", "registered_firearms", "unregistered_firearms"]
contagens = ["civilian_firearms", "registered_firearms", "unregistered_firearms"]


def prepararDados():
    # remove países sem alvo e sem coordenadas (territórios sem localização)
    d = dadosPaises.dropna(subset=[alvo, "latitude", "longitude"]).copy()
    d["unregistered_firearms"] = (d["civilian_firearms"] - d["registered_firearms"]).clip(lower=0)
    d["registro_ausente"] = d["registered_firearms"].isna().astype(int)
    d["urbana_ausente"] = d["urban_population_pct_2017"].isna().astype(int)
    d[contagens] = np.log1p(d[contagens])  # contagens têm escala enorme
    d = d[["income_group", *numericas, "registro_ausente", "urbana_ausente", alvo]]
    d = pd.get_dummies(d, columns=["income_group"], dtype=int)
    return d


def separarDados(d):
    X = d.drop(columns=[alvo])
    y = np.log1p(d[alvo])
    XTr, XVa, yTr, yVa = sk.train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Treino: {len(XTr)} | Validação: {len(XVa)} | Total: {len(d)}")
    return XTr, XVa, yTr, yVa


def imputarNulos(XTr, XVa):
    # Preenche os nulos das colunas numéricas com a mediana do grupo de renda (income_group) do país,
    # calculada SÓ no treino. Se o grupo não tiver nenhum valor, usa a mediana geral do treino.
    colunasGrupo = [c for c in XTr.columns if c.startswith("income_group_")]
    grupoTr = XTr[colunasGrupo].idxmax(axis=1)  # recupera o grupo a partir das colunas 0/1
    grupoVa = XVa[colunasGrupo].idxmax(axis=1)
    medianasGrupo = XTr[numericas].groupby(grupoTr).median()
    medianasGeral = XTr[numericas].median()
    XTr = XTr.copy(); XVa = XVa.copy()
    for col in numericas:
        XTr[col] = XTr[col].fillna(grupoTr.map(medianasGrupo[col]))
        XVa[col] = XVa[col].fillna(grupoVa.map(medianasGrupo[col]))
    XTr[numericas] = XTr[numericas].fillna(medianasGeral)
    XVa[numericas] = XVa[numericas].fillna(medianasGeral)
    return XTr, XVa


def metricas(nome, modelo, XTr, yTr, XVa, yVa):
    pVa = modelo.predict(XVa)
    print(f"\n{nome}")
    print(f"  R² treino (log):     {r2_score(yTr, modelo.predict(XTr)):.3f}")
    print(f"  R² validação (log):  {r2_score(yVa, pVa):.3f}")
    print(f"  MAE validação:       {mean_absolute_error(np.expm1(yVa), np.expm1(pVa)):.2f} homicídios/100 mil")
    return pVa


ativacoes = ["relu", "tanh", "logistic"]
grade = {"mlp__learning_rate_init": [0.01, 0.001],
         "mlp__batch_size": [8, 16, 32]}


def treinarMLPs(XTr, yTr):
    resultados = {}
    for ativ in ativacoes:
        pipe = Pipeline([
            ("escala", StandardScaler()),  # padronização (média 0, desvio 1)
            ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32, 16, 8), activation=ativ,
                                 solver="adam", alpha=0.01, max_iter=1000,
                                 early_stopping=True, validation_fraction=0.15,
                                 n_iter_no_change=30, random_state=42)),
        ])
        busca = sk.GridSearchCV(pipe, grade, cv=5, scoring="r2", n_jobs=-1)
        busca.fit(XTr, yTr)
        print(f"\n[{ativ}] melhores parâmetros: {busca.best_params_} | R² CV: {busca.best_score_:.3f}")
        resultados[ativ] = busca.best_estimator_
    return resultados


def mostrarErros(resultados):
    for ativ, modelo in resultados.items():
        mlp = modelo.named_steps["mlp"]
        print(f"\nVetor de erros por época ({ativ}) - {len(mlp.loss_curve_)} épocas:")
        print(np.round(mlp.loss_curve_, 4))


def graficos(resultados, yVa, preds, melhor):
    # Gera 4 figuras (salvas em PNG e exibidas): perda por época, R² de validação interna por época,
    # comparação das métricas finais entre as funções de ativação e real x previsto de cada ativação.
    cores = {"relu": "tab:blue", "tanh": "tab:orange", "logistic": "tab:green"}
    real = np.expm1(yVa)

    # Figura 1: perda de treino por época (vetor de erros) de cada ativação
    plt.figure(figsize=(8, 5))
    for ativ, modelo in resultados.items():
        mlp = modelo.named_steps["mlp"]
        plt.plot(mlp.loss_curve_, label=f"{ativ} ({len(mlp.loss_curve_)} épocas)", color=cores[ativ])
        melhorEpoca = int(np.argmax(mlp.validation_scores_))  # época em que o early stopping guardou os pesos
        plt.scatter(melhorEpoca, mlp.loss_curve_[melhorEpoca], color=cores[ativ], zorder=3)
    plt.title("Perda de treino por época")
    plt.xlabel("Época")
    plt.ylabel("Perda (erro quadrático médio / 2, alvo em log)")
    plt.legend(title="Ativação (ponto = melhor época)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figura1_perda.png", dpi=150)

    # Figura 2: R² na validação interna (15% do treino) por época
    plt.figure(figsize=(8, 5))
    for ativ, modelo in resultados.items():
        mlp = modelo.named_steps["mlp"]
        plt.plot(mlp.validation_scores_, label=ativ, color=cores[ativ])
        melhorEpoca = int(np.argmax(mlp.validation_scores_))
        plt.scatter(melhorEpoca, mlp.validation_scores_[melhorEpoca], color=cores[ativ], zorder=3)
    plt.ylim(-1, 1)  # corta as primeiras épocas, muito negativas, para enxergar o restante
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.title("R² na validação interna por época")
    plt.xlabel("Época")
    plt.ylabel("R² (alvo em log)")
    plt.legend(title="Ativação (ponto = melhor época)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figura2_r2_validacao_interna.png", dpi=150)

    # Figura 3: comparação das métricas finais (conjunto de validação) entre as ativações
    nomes = list(preds.keys())
    r2s = [r2_score(yVa, preds[a]) for a in nomes]
    maes = [mean_absolute_error(real, np.expm1(preds[a])) for a in nomes]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    barras1 = ax1.bar(nomes, r2s, color=[cores[a] for a in nomes])
    ax1.bar_label(barras1, fmt="%.3f")
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_title("R² na validação")
    ax1.set_ylabel("R² (alvo em log)")
    barras2 = ax2.bar(nomes, maes, color=[cores[a] for a in nomes])
    ax2.bar_label(barras2, fmt="%.2f")
    ax2.set_title("MAE na validação")
    ax2.set_ylabel("Homicídios por 100 mil habitantes")
    for ax in (ax1, ax2):
        ax.set_xlabel("Função de ativação")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figura3_comparacao_ativacoes.png", dpi=150)

    # Figura 4: real x previsto de cada ativação (mesma escala nos três painéis)
    fig, eixos = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    lim = max(real.max(), max(np.expm1(p).max() for p in preds.values())) * 1.05
    for ax, ativ in zip(eixos, nomes):
        ax.scatter(real, np.expm1(preds[ativ]), alpha=0.7, color=cores[ativ])
        ax.plot([0, lim], [0, lim], "r--", label="previsão perfeita")
        titulo = f"{ativ} (R² = {r2_score(yVa, preds[ativ]):.3f})"
        ax.set_title(titulo + (" - melhor" if ativ == melhor else ""))
        ax.set_xlabel("Taxa real (homicídios/100 mil)")
        ax.grid(alpha=0.3)
    eixos[0].set_ylabel("Taxa prevista (homicídios/100 mil)")
    eixos[0].legend()
    plt.tight_layout()
    plt.savefig("figura4_real_x_previsto.png", dpi=150)

    plt.show()


def rodar():
    d = prepararDados()
    XTr, XVa, yTr, yVa = separarDados(d)
    XTr, XVa = imputarNulos(XTr, XVa)

    mlps = treinarMLPs(XTr, yTr)
    mostrarErros(mlps)
    preds = {}
    for ativ, m in mlps.items():
        preds[ativ] = metricas(f"MLP ({ativ})", m, XTr, yTr, XVa, yVa)
    melhor = max(preds, key=lambda a: r2_score(yVa, preds[a]))
    graficos(mlps, yVa, preds, melhor)


if __name__ == "__main__":
    rodar()