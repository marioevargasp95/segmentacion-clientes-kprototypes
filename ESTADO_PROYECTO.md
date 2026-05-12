# Estado del Proyecto — Modelo K-Prototypes Corporativo
> Última actualización: 2026-04-06

---

## Descripción general

Pipeline de segmentación de clientes corporativos mediante clustering con datos mixtos (variables numéricas y categóricas). El objetivo es agrupar clientes de un producto de seguros PAP en segmentos homogéneos para análisis comercial y toma de decisiones.

**Fuentes de datos principales:**
| Archivo | Descripción | Tamaño |
|---|---|---|
| `PAP_informacion_clientes.csv` | Clientes activos con características demográficas y de producto | ~104 MB |
| `ReporteAcum.csv` | Acumulado histórico de familias/contratos 2020-2025 | ~80 MB |
| `Segmentacion Base JUL25.xlsx` | Segmentación financiera por entidad (Gerencia Financiera) | ~840 KB |
| `Productos.xlsx` | Tabla maestra de homologación de planes | ~21 KB |

---

## Estructura de notebooks

### `01_Data_limpia_EDA_corporativo.ipynb` — 28 celdas
**Pipeline:** Carga → Limpieza/Fusión → Homologación → Geografía → Features → EDA → Modelo → Exportación

**Secciones:**
| # | Sección | Descripción |
|---|---|---|
| 0 | Imports | pandas, numpy, matplotlib, seaborn, re, unicodedata, sklearn |
| 1 | Carga de datos | 4 fuentes: clientes, productos, segmentación, acumulado |
| 2 | Limpieza y fusión | Normalización de columnas, merge con segmentación y acumulado |
| 3 | Homologación de productos | Merge con tabla maestra + `MAPEO_ADICIONAL` para nombres sin cobertura |
| 4 | Procesamiento geográfico | Ciudad → Departamento → Región (5 regiones + INSULAR + OTROS) |
| 5 | Ingeniería de features | Edad recalculada, Valortotalplan, Rango_afiliados, mascotas, Sexo, Estrato, TiposSeguros |
| 6 | EDA | Perfil estructural, estadísticas descriptivas, distribuciones categóricas (subplot 2×5), boxplots Valortotalplan |
| 7 | Preparación modelo | StandardScaler, codificación ordinal, df_kprototypes, pairplots |
| 8 | Exportación | `data_limpia.csv` |

**Output:** `data_limpia.csv` (~196 MB)

**Cambios aplicados (2026-04-06) — Refactorización completa:**
- Reducido de **109 → 28 celdas** (-75%)
- Eliminadas: 15 celdas huérfanas (`.shape`, `.columns`, `.head()` sin contexto)
- Unificada lógica de homologación duplicada (celdas 33+34 → una sola)
- Eliminados aliases innecesarios (`clientes_2020_2025 = df`, `df_nov_dic_reciente = df_unido`)
- Corregida columna `Rango_afiliados` duplicada en lista de selección
- Eliminada conversión Valortotalplan → string → float (round-trip innecesario)
- Unificadas funciones `normalizar()` y `normalizar_texto()` en una sola
- `df` reasignada con 3 significados distintos → renombrada consistentemente
- EDA consolidado: 8 `value_counts` individuales → subplot 2×5 unificado
- 3 boxplots separados → subplot 1×3
- `StandardScaler` importado 2 veces → una sola vez al inicio

---

### `02_Ingeniería_y_modelado_de_características.ipynb` — 34 celdas
**Pipeline:** Carga → Pre-filtro → Selección variables (3 capas) → K-Prototypes → Validación → DBSCAN+Gower → Exportación

**Secciones:**
| # | Sección | Descripción |
|---|---|---|
| 0 | Imports | Todos centralizados: phik, kmodes, gower, plotnine, sklearn, scipy |
| 1 | Carga | Lee `data_limpia.csv`, limpia nombres de columnas |
| 2 | Selección de variables | Pre-filtro (54 cols excluidas) + 3 capas matemáticas |
| 3 | Construcción df_kprototypes | Imputación, PowerTransformer Yeo-Johnson, PhiK correlation |
| 4 | Modelado K-Prototypes | Elbow (10k muestra) + Silhouette/Gower (5k muestra) + modelo final |
| 5 | Validación y análisis | Random Forest + Regresión Logística, importancia variables, PCA 2D + Scree, perfiles de clusters |
| 6 | Exportación K-Prototypes | `resultado_etiquetado.csv` + `modelo_clusters_rf.pkl` |
| 7 | DBSCAN + Gower | Muestra 10k estratificada, selección eps automática, propagación KNN |
| 8 | Resumen ejecutivo | Comparación K-Prototypes vs DBSCAN |

**Selección de variables — 3 capas:**
- **Pre-filtro:** Excluye identificadores, varianza cero, buckets, >50% NaN, redundantes comerciales
- **Capa 1A:** Coeficiente de variación CV < 0.10 → descarta numéricas con baja varianza
- **Capa 1B:** Entropía de Shannon H_rel < 0.30 → descarta categóricas muy concentradas
- **Capa 2A:** V de Cramér > 0.70 → detecta redundancia categórica vs categórica
- **Capa 2B:** Spearman |ρ| > 0.85 → detecta redundancia numérica vs numérica
- **Capa 2C:** Eta cuadrado η² → detecta asociación numérica vs categórica
- **Capa 3:** Silhouette Leave-One-Out con Gower + K-Prototypes (k=3, muestra 2000)

**Decisión de variables DBSCAN vs K-Prototypes:**
- K-Prototypes usa el subconjunto filtrado por Capa 3 (específico del algoritmo)
- DBSCAN usa las 11 candidatas de Capas 1+2 (Gower no depende de centroides)

**Cambios aplicados (2026-04-06) — Refactorización:**
- Reducido de **39 → 34 celdas**
- `!pip install` reemplazado por `try/import` con subprocess (no reinstala si ya existe)
- Imports unificados en bloque inicial (estaban dispersos en 6 celdas: 2, 10, 11, 22, 28, 33)
- Celdas 3+4+5+6 (carga) → una sola celda
- Celda 16 (`dfMatrix` recalculada innecesariamente) → eliminada
- Celdas 26+27 (dos títulos DBSCAN, uno incompleto) → un markdown consolidado
- Re-imports completos en sección DBSCAN → eliminados
- Celda 38 vacía → eliminada
- Boxplots numéricos por variable (loop individual) → subplot consolidado
- `eta_cuadrado()` refactorizada para recibir `df_sub` como parámetro (eliminó dependencia global implícita)

**Outputs generados:**
| Archivo | Descripción |
|---|---|
| `resultado_etiquetado.csv` | Dataset completo + columna `Cluster` (K-Prototypes) |
| `modelo_clusters_rf.pkl` | Modelo Random Forest para asignación de nuevos clientes |
| `k_distancia.png` | Curva k-distancia para selección de eps DBSCAN |
| `dbscan_gower_pca.png` | Proyección PCA 2D de clusters DBSCAN |
| `heatmap_distancias.png` | Distancias medias Gower entre clusters |
| `perfiles_clusters.csv` | Perfiles descriptivos por cluster DBSCAN |
| `clientes_segmentados.csv` | Dataset completo + `cluster_dbscan` (propagado por KNN) |

---

### `03_Análisis_del_segmento_de_clientes.ipynb` — 52 celdas
> **Estado:** Pendiente de revisión y refactorización

---

## Dependencias instaladas

```
pandas, numpy, matplotlib, seaborn          # base
scikit-learn                                 # ML pipeline
kmodes                                       # K-Prototypes
gower                                        # distancia Gower para datos mixtos
phik                                         # correlación PhiK (variables mixtas)
plotnine                                     # ggplot-style para Elbow plot
chardet                                      # detección de encoding CSV
joblib                                       # serialización de modelos
scipy                                        # chi2_contingency para V de Cramér
```

---

## Problemas resueltos durante el desarrollo

| Fecha | Problema | Solución |
|---|---|---|
| 2026-04-06 | `jupyter notebook` no encontrado | `pip install notebook` |
| 2026-04-06 | `No module named 'chardet'` | `pip install chardet` |
| 2026-04-06 | `SyntaxError: unterminated string literal` (celda 30 nb01) | `print("` con salto de línea literal → reemplazado por `\n` |
| 2026-04-06 | `No module named 'kmodes'` | `pip install kmodes` |

---

## Notas técnicas

- **Dataset completo:** ~345k registros tras limpieza y filtro de canales
- **Canales excluidos:** MASIVO - BOGOTA, PAP - BOGOTA, PAP - BOYACA, CONTACT CENTER
- **Regiones:** ANDINA, CARIBE, PACIFICO, ORINOQUIA, AMAZONIA, INSULAR, OTROS
- **Escala DBSCAN:** Matriz Gower sobre 303k filas requiere ~350 GB RAM → se trabaja con muestra 10k + propagación KNN
- **Modelo de producción:** `modelo_clusters_rf.pkl` permite asignar cluster a nuevos clientes sin reentrenar K-Prototypes

---

## Próximos pasos

- [ ] Revisar y refactorizar `03_Análisis_del_segmento_de_clientes.ipynb` (52 celdas)
- [ ] Ejecutar pipeline completo end-to-end tras refactorizaciones
- [ ] Validar cobertura geográfica (% registros sin departamento asignado)
- [ ] Definir K óptimo final basado en resultados Elbow + Silhouette
- [ ] Análisis de perfiles de clusters para interpretación de negocio
