# %% [markdown]
# # T2 - Análisis exploratorio de datos y comparación preliminar de algoritmos
# Proyecto: predicción de la resistencia a compresión del concreto (Yeh, 1998).
# Marco VDS (Yu y Barter, 2024): predictibilidad, computabilidad y estabilidad.

# %%
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

RAIZ = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path("..")
RAW = RAIZ / "data" / "raw" / "Concrete_Data.xls"
PROC = RAIZ / "data" / "processed"
FIGS = RAIZ / "docs" / "figuras_t2"
PROC.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
resultados = {}

# %% [markdown]
# ## 1. Carga y calidad de datos

# %%
df = pd.read_excel(RAW)
df.columns = ["cemento", "escoria", "ceniza", "agua", "superplast",
              "grueso", "fino", "edad", "fc"]
COMP = ["cemento", "escoria", "ceniza", "agua", "superplast", "grueso", "fino"]

resultados["filas"] = int(len(df))
resultados["nulos"] = int(df.isna().sum().sum())
resultados["duplicadas_exactas"] = int(df.duplicated().sum())
resultados["duplicadas_comp_edad"] = int(df.duplicated(subset=COMP + ["edad"]).sum())
print(resultados)

# Las 25 filas idénticas se eliminan (decisión de juicio documentada en el informe)
df = df.drop_duplicates().reset_index(drop=True)
resultados["filas_limpias"] = int(len(df))

# %% [markdown]
# ## 2. Identificación de mezclas (para evitar fuga de datos)
# Una misma dosificación (7 componentes) se ensaya a varias edades.

# %%
df["grupo"] = df.groupby(COMP).ngroup()
tam = df.groupby("grupo").size()
resultados["mezclas_distintas"] = int(df["grupo"].nunique())
resultados["mezclas_multiples_edades"] = int((tam > 1).sum())
resultados["filas_en_mezclas_multiples"] = int(tam[tam > 1].sum())
resultados["max_filas_por_mezcla"] = int(tam.max())
print({k: v for k, v in resultados.items() if "mezcla" in k})

# %% [markdown]
# ## 3. Variables derivadas (conocimiento del dominio)

# %%
df["aglomerante"] = df.cemento + df.escoria + df.ceniza
df["a_c"] = df.agua / df.cemento
df["a_aglom"] = df.agua / df.aglomerante
df["prop_adiciones"] = (df.escoria + df.ceniza) / df.aglomerante
df["pasta_agregados"] = (df.aglomerante + df.agua) / (df.grueso + df.fino)
df["log_edad"] = np.log(df.edad)

DERIV = ["a_c", "a_aglom", "prop_adiciones", "pasta_agregados", "log_edad"]
corr = df[COMP + ["edad"] + DERIV + ["fc"]].corr(method="spearman")["fc"].drop("fc")
resultados["spearman_fc"] = corr.round(3).to_dict()
print(corr.sort_values())

# %% [markdown]
# ## 4. Figuras

# %%
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(3, 3, figsize=(11, 8))
for ax, c in zip(axes.ravel(), COMP + ["edad", "fc"]):
    sns.histplot(df[c], bins=30, ax=ax, color="#2F5496")
    ax.set_xlabel(c)
    ax.set_ylabel("")
fig.suptitle("Distribución de las variables")
fig.tight_layout()
fig.savefig(FIGS / "fig1_distribuciones.png", dpi=150)
plt.close(fig)

cols = COMP + ["edad"] + DERIV + ["fc"]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[cols].corr(method="spearman"), annot=True, fmt=".2f",
            cmap="vlag", center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Correlación de Spearman")
fig.tight_layout()
fig.savefig(FIGS / "fig2_correlacion.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
sns.scatterplot(data=df, x="a_aglom", y="fc", hue="edad", palette="viridis",
                s=18, ax=axes[0], legend=False)
axes[0].set_xlabel("agua / aglomerante")
sns.scatterplot(data=df, x="a_c", y="fc", s=18, ax=axes[1], color="#C55A11")
axes[1].set_xlabel("agua / cemento")
orden = sorted(df.edad.unique())
sns.boxplot(data=df, x="edad", y="fc", order=orden, ax=axes[2], color="#9DC3E6")
axes[2].tick_params(axis="x", rotation=60)
fig.suptitle("Resistencia vs. relaciones de mezcla y edad")
fig.tight_layout()
fig.savefig(FIGS / "fig3_relaciones.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 5. Valores atípicos (criterio IQR, solo se reportan)

# %%
atip = {}
for c in COMP + ["edad", "fc"]:
    q1, q3 = df[c].quantile([0.25, 0.75])
    i = q3 - q1
    atip[c] = int(((df[c] < q1 - 1.5 * i) | (df[c] > q3 + 1.5 * i)).sum())
resultados["atipicos_iqr"] = atip
print(atip)

# %% [markdown]
# ## 6. Comparación preliminar: partición aleatoria vs. agrupada por mezcla
# Evidencia de fuga de datos y base para el plan de la predictibilidad.

# %%
BASE = COMP + ["edad"]
EXT = BASE + ["a_c", "a_aglom", "prop_adiciones", "pasta_agregados", "log_edad"]


def modelos():
    return {
        "Regresión lineal": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
    }


def evaluar(cols, splitter, grupos=None):
    out = {}
    X, y = df[cols].values, df["fc"].values
    for nombre, mod in modelos().items():
        r2s, rmses, maes = [], [], []
        for tr, te in splitter.split(X, y, grupos):
            mod.fit(X[tr], y[tr])
            p = mod.predict(X[te])
            r2s.append(r2_score(y[te], p))
            rmses.append(mean_squared_error(y[te], p) ** 0.5)
            maes.append(mean_absolute_error(y[te], p))
        out[nombre] = {"R2": float(np.mean(r2s)), "R2_sd": float(np.std(r2s)),
                       "RMSE": float(np.mean(rmses)), "MAE": float(np.mean(maes))}
    return out


kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
gkf = GroupKFold(n_splits=5)

tabla = {}
for etiqueta_cols, cols in [("base", BASE), ("derivadas", EXT)]:
    tabla[f"{etiqueta_cols}|aleatoria"] = evaluar(cols, kf)
    tabla[f"{etiqueta_cols}|agrupada"] = evaluar(cols, gkf, df["grupo"].values)
resultados["comparacion"] = tabla
for k, v in tabla.items():
    print(k)
    for m, r in v.items():
        print(f"  {m:18s} R2={r['R2']:.3f}±{r['R2_sd']:.3f} RMSE={r['RMSE']:.2f} MAE={r['MAE']:.2f}")

# %% [markdown]
# ## 7. Estabilidad del error por edad (partición agrupada)

# %%
edad_bins = {"1-7 d": (0, 7), "14-28 d": (8, 28), "56-100 d": (29, 100), ">100 d": (101, 400)}
est = {}
X, y = df[EXT].values, df["fc"].values
pred_oof = np.zeros(len(df))
for tr, te in gkf.split(X, y, df["grupo"].values):
    m = GradientBoostingRegressor(random_state=SEED).fit(X[tr], y[tr])
    pred_oof[te] = m.predict(X[te])
df["pred_gb"] = pred_oof
for nombre, (a, b) in edad_bins.items():
    s = df[(df.edad >= a) & (df.edad <= b)]
    est[nombre] = {"n": int(len(s)),
                   "RMSE": float(mean_squared_error(s.fc, s.pred_gb) ** 0.5),
                   "MAE": float(mean_absolute_error(s.fc, s.pred_gb))}
resultados["error_por_edad"] = est
print(est)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(df.fc, df.pred_gb, s=12, alpha=0.6, color="#2F5496")
lim = [0, df.fc.max() + 3]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("f'c real (MPa)")
ax.set_ylabel("f'c predicha (MPa)")
ax.set_title("Gradient Boosting, partición agrupada")
fig.tight_layout()
fig.savefig(FIGS / "fig4_real_vs_pred.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 8. Guardado de datos procesados y resultados

# %%
df.drop(columns=["pred_gb"]).to_csv(PROC / "concreto_limpio.csv", index=False)
with open(PROC / "resultados_t2.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)
print("listo")
