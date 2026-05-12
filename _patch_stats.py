import os
import json, uuid, sys

PATH = r"C:\Users\mario_481\Analisis_ia\Modelo_dos_K_MEAS_corporativo\01_Data_limpia_EDA_corporativo.ipynb"
nb = json.load(open(PATH, encoding='utf-8'))
cells = nb['cells']

def code(src):
    return {"cell_type":"code","execution_count":None,"id":str(uuid.uuid4()),
            "metadata":{},"outputs":[],"source":src}
def md(src):
    return {"cell_type":"markdown","id":str(uuid.uuid4()),"metadata":{},"source":src}

# Localizar índices de las 3 celdas de código de sección 10
idx_funcs, idx_global, idx_seg = None, None, None
for i, c in enumerate(cells):
    src = ''.join(c.get('source',''))
    if 'def bloque_nits' in src:
        idx_funcs = i
    elif 'bloque_nits(df_modelo)' in src:
        idx_global = i
    elif 'bloque_nits(df_seg)' in src:
        idx_seg = i

print(f"Celdas: funcs={idx_funcs}, global={idx_global}, seg={idx_seg}")

# ── CELDA FUNCIONES ────────────────────────────────────────────────────────
FUNCS = """\
SEP  = '\u2500' * 60
SEP2 = '\u2550' * 60
CORP_PALETTE = ['#24743c', '#79a65c', '#f39c14', '#f3eb8b']

def _dist(serie, pct=False, n=None):
    vc = serie.value_counts(normalize=pct)
    if n: vc = vc.head(n)
    for k, v in vc.items():
        val = f'{v*100:6.1f}%' if pct else f'{int(v):>8,}'
        print(f'    {str(k):<36} {val}')

def bloque_nits(df):
    df_n = df.drop_duplicates(subset='Nit')
    act  = pd.to_numeric(df.drop_duplicates(subset='Contrato')['Act.valor'], errors='coerce')
    print(f'  NITs \u00fanicos:                {df_n["Nit"].nunique():>8,}')
    print(f'  Familias  (Act.valor):      {act.sum():>8,.0f}')
    print(f'  Personas registradas:       {len(df):>8,}')
    print('\\n  Zona (Canal):')
    _dist(df_n['Canal'])
    print('\\n  Departamento:')
    _dist(df_n['DEPARTAMENTO'])
    print('\\n  Rango de afiliados:')
    _dist(df_n['Rango_afiliados'])

def bloque_familias(df):
    df_c = df.drop_duplicates(subset='Contrato').copy()
    df_c['_act'] = pd.to_numeric(df_c['Act.valor'], errors='coerce').fillna(0)
    print('\\n  Familias por zona (Canal):')
    fz = df_c.groupby('Canal')['_act'].sum().sort_values(ascending=False)
    for k, v in fz.items(): print(f'    {k:<36} {v:>10,.0f}')
    print('\\n  Familias por departamento:')
    fg = df_c.groupby('DEPARTAMENTO')['_act'].sum().sort_values(ascending=False)
    for k, v in fg.items(): print(f'    {k:<36} {v:>10,.0f}')
    print('\\n  Sexo (% personas):')
    _dist(df['Sexo'], pct=True)
    print('\\n  Rango de edad:')
    _dist(df['Rango_edad'])
    print('\\n  Tipo de asistencia:')
    _dist(df['Tiposseguros_ajuste'])
    print('\\n  Producto:')
    _dist(df['Producto'])

def bloque_graficas(df, titulo=''):
    df_n = df.drop_duplicates(subset='Nit')
    df_c = df.drop_duplicates(subset='Contrato').copy()
    df_c['_act'] = pd.to_numeric(df_c['Act.valor'], errors='coerce').fillna(0)

    def _bar(ax, serie, title, max_n=10):
        data = serie.value_counts().head(max_n)
        n = len(data)
        colors = [CORP_PALETTE[i % 4] for i in range(n)]
        y = range(n)
        ax.barh(list(y), list(data.values[::-1]), color=colors)
        ax.set_yticks(list(y))
        ax.set_yticklabels([str(k)[:22] for k in data.index[::-1]], fontsize=8)
        mx = max(data.values) if len(data) else 1
        for j, v in enumerate(data.values[::-1]):
            ax.text(v + mx * 0.01, j, f'{int(v):,}', va='center', fontsize=7, color='#24743c')
        ax.set_title(title, fontsize=9, fontweight='bold', color='#24743c')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(left=False)
        ax.set_xlim(0, mx * 1.18)

    def _bar_sum(ax, df_c, gcol, vcol, title, max_n=10):
        data = df_c.groupby(gcol)[vcol].sum().sort_values(ascending=False).head(max_n)
        n = len(data)
        colors = [CORP_PALETTE[i % 4] for i in range(n)]
        y = range(n)
        ax.barh(list(y), list(data.values[::-1]), color=colors)
        ax.set_yticks(list(y))
        ax.set_yticklabels([str(k)[:22] for k in data.index[::-1]], fontsize=8)
        mx = max(data.values) if len(data) else 1
        for j, v in enumerate(data.values[::-1]):
            ax.text(v + mx * 0.01, j, f'{v:,.0f}', va='center', fontsize=7, color='#24743c')
        ax.set_title(title, fontsize=9, fontweight='bold', color='#24743c')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(left=False)
        ax.set_xlim(0, mx * 1.18)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.patch.set_facecolor('white')
    if titulo:
        fig.suptitle(titulo, fontsize=13, fontweight='bold', color='#24743c', y=1.01)

    # Fila 1 \u2014 Empresas
    _bar(axes[0, 0], df_n['Canal'],           'NITs por Canal')
    _bar(axes[0, 1], df_n['DEPARTAMENTO'],    'NITs por Departamento', 8)
    _bar(axes[0, 2], df_n['Rango_afiliados'], 'Rango de Afiliados')
    sexo = df['Sexo'].value_counts()
    axes[0, 3].pie(
        sexo.values, labels=sexo.index,
        colors=CORP_PALETTE[:len(sexo)],
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 9}
    )
    axes[0, 3].set_title('Sexo (personas)', fontsize=9, fontweight='bold', color='#24743c')

    # Fila 2 \u2014 Familias y demograf\u00eda
    _bar_sum(axes[1, 0], df_c, 'Canal',        '_act', 'Familias por Canal')
    _bar_sum(axes[1, 1], df_c, 'DEPARTAMENTO', '_act', 'Familias por Departamento', 8)
    _bar(axes[1, 2], df['Rango_edad'],         'Rango de Edad')
    _bar(axes[1, 3], df['Tiposseguros_ajuste'], 'Tipo de Asistencia')

    plt.tight_layout()
    plt.show()

print('Funciones OK')
"""

# ── CELDA GLOBAL ───────────────────────────────────────────────────────────
GLOBAL = """\
print(SEP2)
print('  CORPORATIVO \u2014 NIVEL GLOBAL')
print(SEP2)
bloque_nits(df_modelo)
print('\\n  Nota \u2014 NITs sin actualizaci\u00f3n hasta renovaci\u00f3n:')
print('    Revisar columna "Top_resultados" o "Estado" en datos fuente.')
print(f'\\n{SEP}')
bloque_familias(df_modelo)
bloque_graficas(df_modelo, 'Corporativo \u2014 Nivel Global')
"""

# ── CELDA SEGMENTOS ────────────────────────────────────────────────────────
SEGMENTOS = """\
SEGMENTOS = {
    'CUENTAS ESPECIALES': df_modelo['Canal'].str.upper().str.contains('CUENTA',     na=False),
    'MICROPYME'         : df_modelo['Canal'].str.upper().str.contains('MICROPYME',  na=False),
    'INDIVIDUALES'      : df_modelo['Canal'].str.upper().str.contains('INDIVIDUAL', na=False),
}

for nombre, mask in SEGMENTOS.items():
    df_seg = df_modelo[mask].copy()
    print(f'\\n{SEP2}')
    print(f'  SEGMENTO: {nombre}  ({len(df_seg):,} registros)')
    print(SEP2)
    if df_seg.empty:
        print('  Sin registros para este canal.')
        continue
    bloque_nits(df_seg)
    print(f'\\n{SEP}')
    bloque_familias(df_seg)
    bloque_graficas(df_seg, f'Segmento: {nombre}')
"""

cells[idx_funcs]['source'] = FUNCS
cells[idx_global]['source'] = GLOBAL
cells[idx_seg]['source']    = SEGMENTOS

nb['cells'] = cells
with open(PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"OK — {len(cells)} celdas")
