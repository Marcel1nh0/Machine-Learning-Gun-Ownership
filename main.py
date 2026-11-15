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
    medianas = XTr[numericas].median()
    XTr = XTr.copy(); XVa = XVa.copy()
    XTr[numericas] = XTr[numericas].fillna(medianas)
    XVa[numericas] = XVa[numericas].fillna(medianas)
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


def graficos(resultados, yVa, pVaMelhor, nomeMelhor):
    fig, eixos = plt.subplots(1, 3, figsize=(18, 5))
    for ativ, modelo in resultados.items():
        mlp = modelo.named_steps["mlp"]
        eixos[0].plot(mlp.loss_curve_, label=ativ)
        eixos[1].plot(mlp.validation_scores_, label=ativ)
    eixos[0].set_title("Perda de treino por época"); eixos[0].set_xlabel("Época"); eixos[0].set_ylabel("Perda"); eixos[0].legend()
    eixos[1].set_title("R² de validação interna por época"); eixos[1].set_xlabel("Época"); eixos[1].set_ylabel("R²"); eixos[1].legend()
    real, prev = np.expm1(yVa), np.expm1(pVaMelhor)
    eixos[2].scatter(real, prev, alpha=0.7)
    lim = max(real.max(), prev.max())
    eixos[2].plot([0, lim], [0, lim], "r--")
    eixos[2].set_title(f"Real x Previsto - MLP {nomeMelhor}"); eixos[2].set_xlabel("Real"); eixos[2].set_ylabel("Previsto")
    plt.tight_layout()
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
    graficos(mlps, yVa, preds[melhor], melhor)


if __name__ == "__main__":
    rodar()