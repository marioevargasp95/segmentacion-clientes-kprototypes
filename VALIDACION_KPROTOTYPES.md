# Validación Técnica — K-Prototypes PAP
**Rol:** Senior ML Engineer — Clustering No Supervisado
**Fecha:** 2026-03-27
**Proyecto:** Segmentación de clientes PAP con variables mixtas

---

## 1. Escalamiento de Variables Numéricas

**Regla:** `StandardScaler` si distribución normal sin outliers. `RobustScaler` si hay outliers > 1% de los datos.

**Para este proyecto:**
- `Estrato`, `Cantidad_mascotas` → `StandardScaler` ✓
- `Valortotalplan` → evaluar `RobustScaler` (puede tener outliers por planes caros)

```python
from scipy.stats import shapiro

def elegir_scaler(df, num_cols, alpha=0.05):
    for col in num_cols:
        serie = df[col].dropna()
        muestra = serie.sample(min(500, len(serie)), random_state=42)
        stat, p = shapiro(muestra)
        tiene_outliers = (
            (serie > serie.mean() + 3 * serie.std()).sum() > len(serie) * 0.01
        )
        rec = "RobustScaler" if tiene_outliers else "StandardScaler"
        print(f"{col:25s} | normal={p>alpha} | outliers={tiene_outliers} → {rec}")
```

---

## 2. Cálculo de Gamma Óptimo

El gamma automático de `kmodes` usa `0.5 * mean(std(X_num))` — es punto de partida, no garantiza balance.

**Método correcto:** grid search sobre rango `[0.3x, 2.0x]` del gamma base, maximizando Silhouette Score.

**Rango aceptable:** gamma entre `0.2` y `0.5 * mean(std_num)`. Si gamma óptimo > 1.0 → numéricas mal escaladas o categóricas dominantes.

```python
def buscar_gamma_optimo(X_num, X_cat, k, gammas=None, n_init=5):
    if gammas is None:
        std_mean = np.mean(np.std(X_num, axis=0))
        base     = 0.5 * std_mean
        gammas   = np.round(np.linspace(base * 0.3, base * 2.0, 10), 4)

    X_full   = np.hstack([X_num, X_cat])
    cat_idx  = list(range(X_num.shape[1], X_full.shape[1]))
    resultados = []

    for g in gammas:
        kp = KPrototypes(n_clusters=k, init='Cao', n_init=n_init,
                         gamma=g, random_state=42)
        labels = kp.fit_predict(X_full, categorical=cat_idx)
        sil = silhouette_score(X_num, labels, metric='euclidean')
        resultados.append({"gamma": g, "silhouette": sil})

    df_res    = pd.DataFrame(resultados)
    gamma_opt = df_res.loc[df_res.silhouette.idxmax(), 'gamma']
    print(f"Gamma óptimo: {gamma_opt}  (Silhouette={df_res.silhouette.max():.4f})")
    return gamma_opt, df_res
```

---

## 3. Selección de K — Tres Métricas Combinadas

| Métrica | Buscar | Bueno | Aceptable |
|---|---|---|---|
| Silhouette | Máximo | > 0.50 | > 0.25 |
| Davies-Bouldin | Mínimo | < 1.0 | < 1.5 |
| Elbow (Costo) | Quiebre | — | — |

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

def validar_k_completo(X_full, X_num, cat_idx, k_range=range(2, 9), gamma=None, n_init=5):
    resultados = []
    for k in k_range:
        kp = KPrototypes(n_clusters=k, init='Cao', n_init=n_init,
                         gamma=gamma, random_state=42)
        labels = kp.fit_predict(X_full, categorical=cat_idx)
        resultados.append({
            "k":             k,
            "costo":         kp.cost_,
            "silhouette":    silhouette_score(X_num, labels),
            "davies_bouldin": davies_bouldin_score(X_num, labels),
        })

    df = pd.DataFrame(resultados)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    df.plot(x='k', y='costo',           ax=axes[0], marker='o', title='Elbow (Costo)')
    df.plot(x='k', y='silhouette',      ax=axes[1], marker='o', title='Silhouette ↑')
    df.plot(x='k', y='davies_bouldin',  ax=axes[2], marker='o', title='Davies-Bouldin ↓')
    plt.tight_layout()
    plt.savefig('validacion_k.png', dpi=150)
    return df
```

---

## 4. Validación Estadística de Clusters

Confirma que los clusters son distintos estadísticamente, no solo visualmente.

- **Numéricas:** Kruskal-Wallis (no asume normalidad)
- **Categóricas:** Chi-cuadrado

```python
from scipy.stats import kruskal, chi2_contingency

def validar_significancia(df, labels_col, num_cols, cat_cols, alpha=0.05):
    df = df.copy()
    df['_cluster'] = labels_col
    grupos = [g for _, g in df.groupby('_cluster')]

    print("VARIABLES NUMÉRICAS (Kruskal-Wallis)")
    sig_num = []
    for col in num_cols:
        muestras = [g[col].dropna().values for g in grupos]
        stat, p  = kruskal(*muestras)
        sig = "✅ SIGNIFICATIVA" if p < alpha else "❌ no significativa"
        print(f"  {col:25s} | p={p:.4f} | {sig}")
        if p < alpha: sig_num.append(col)

    print("\nVARIABLES CATEGÓRICAS (Chi-cuadrado)")
    sig_cat = []
    for col in cat_cols:
        tabla = pd.crosstab(df['_cluster'], df[col])
        chi2, p, dof, _ = chi2_contingency(tabla)
        sig = "✅ SIGNIFICATIVA" if p < alpha else "❌ no significativa"
        print(f"  {col:25s} | p={p:.4f} | {sig}")
        if p < alpha: sig_cat.append(col)

    return sig_num, sig_cat

# Variables del proyecto PAP
num_cols = ['Antigedad', 'Cantidad_mascotas', 'Cuotas', 'Estrato', 'Valortotalplan']
cat_cols = ['EstadoCivil', 'TienePadres', 'TieneEsposa', 'TieneHijos', 'REGION']
sig_num, sig_cat = validar_significancia(df_resultado, labels, num_cols, cat_cols)
```

**Meta:** al menos 70% de variables con p < 0.05. Cada cluster con mínimo 30 instancias.

---

## 5. Estabilidad de Centroides

Verifica que los clusters no dependan de la semilla aleatoria.

```python
from sklearn.metrics import adjusted_rand_score

def validar_estabilidad(X_full, cat_idx, k, gamma, n_runs=10):
    etiquetas_runs = []
    costos = []

    for seed in range(n_runs):
        kp = KPrototypes(n_clusters=k, init='Cao', n_init=3,
                         gamma=gamma, random_state=seed)
        labels = kp.fit_predict(X_full, categorical=cat_idx)
        etiquetas_runs.append(labels)
        costos.append(kp.cost_)

    aris = [
        adjusted_rand_score(etiquetas_runs[i], etiquetas_runs[j])
        for i in range(n_runs)
        for j in range(i + 1, n_runs)
    ]

    ari_mean = np.mean(aris)
    estable  = ari_mean > 0.85
    print(f"ARI promedio: {ari_mean:.4f} | {'✅ ESTABLE' if estable else '⚠️ INESTABLE — aumenta n_init'}")
    print(f"Variación de costo: {np.std(costos):.2f}")
    return ari_mean, estable
```

**Umbral:** ARI > 0.85 entre 10 runs = modelo estable. Si ARI < 0.85 → aumentar `n_init` a 20.

---

## Checklist de Validación Técnica

### Preprocesamiento
- [ ] Diagnóstico de outliers por variable numérica (IQR o 3-sigma)
- [ ] StandardScaler si distribución normal, RobustScaler si hay outliers
- [ ] Variables categóricas sin OHE — pasar como string/object a KPrototypes
- [ ] Nulos imputados (mediana para numéricas, moda para categóricas)

### Gamma
- [ ] Imprimir gamma automático calculado por la librería
- [ ] Grid search de gamma sobre rango [0.3x, 2.0x] del gamma base
- [ ] Gamma óptimo validado con Silhouette Score
- [ ] Gamma final entre 0.2 y 0.5 * mean(std_num)

### Selección de K
- [ ] Elbow method (costo vs K)
- [ ] Silhouette Score > 0.25 (aceptable) o > 0.50 (bueno)
- [ ] Davies-Bouldin Score < 1.5
- [ ] Los 3 criterios apuntan al mismo K (o rango de 1 valor)

### Significancia Estadística
- [ ] Kruskal-Wallis sobre todas las variables numéricas (p < 0.05)
- [ ] Chi-cuadrado sobre todas las variables categóricas (p < 0.05)
- [ ] Al menos 70% de variables con diferencias significativas entre clusters
- [ ] Cada cluster tiene n mínimo de 30 instancias

### Estabilidad
- [ ] n_init >= 10 ✅ (corregido en pipeline)
- [ ] ARI > 0.85 entre 10 runs con seeds distintos
- [ ] Variación de costo entre runs < 5%
- [ ] init='Cao' ✅ (corregido en pipeline)

### Interpretación
- [ ] Perfil de cada cluster documentado (media/moda por variable)
- [ ] Nombre semántico asignado a cada cluster
- [ ] PCA varianza PC1+PC2 > 50% (era 37.7% con OHE — pendiente verificar con OrdinalEncoder)
- [ ] Distribución de clusters: ningún cluster < 5% del total

---

## Puntos críticos pendientes en el pipeline PAP

1. **Gamma post-StandardScaler** — el fix de escalado de todas las numéricas cambia el gamma automático; verificar que siga en rango aceptable
2. **Validación estadística** — correr `validar_significancia()` en Notebook 03 con los clusters finales
3. **PCA varianza** — el 37.7% anterior era por usar OHE; con OrdinalEncoder debe superar 50%
4. **Distribución de clusters** — verificar que ningún cluster tenga < 5% del total de registros
