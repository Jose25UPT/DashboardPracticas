import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import requests
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import numpy as np
from collections import Counter

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Dashboard Inteligente - Prácticas UPT",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# ESTILOS CSS
# ==========================================
st.markdown("""
<style>
    .stApp {
        background: #f5f7fa !important;
    }
    
    .stApp * {
        color: #000000 !important;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff !important;
    }
    
    .main-header p {
        color: #ffffff !important;
        font-size: 1.1rem;
    }
    
    .section-title {
        color: #000000 !important;
        font-size: 1.6rem;
        font-weight: 700;
        border-left: 5px solid #667eea;
        padding: 1rem;
        margin: 2rem 0 1.5rem 0;
        background: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .sub-title {
        color: #000000 !important;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }
    
    .kpi-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #667eea !important;
        margin: 0.5rem 0;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: #000000 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #000000 !important;
        font-size: 0.85rem;
        margin-top: 3rem;
        border-top: 2px solid #667eea;
        background: #ffffff;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# PALETA DE COLORES
# ==========================================
PALETA_PRINCIPAL = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', 
                    '#43e97b', '#38f9d7', '#fa709a', '#fee140', '#30cfd0', '#330867']

# ==========================================
# FUNCIONES DE LIMPIEZA
# ==========================================

def normalizar_texto(valor):
    """Normaliza texto: quita tildes, convierte a mayúsculas"""
    if pd.isna(valor) or str(valor).strip() == '':
        return None
    
    texto = str(valor).strip().upper()
    tildes = {'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ñ': 'NN'}
    for tilde, normal in tildes.items():
        texto = texto.replace(tilde, normal)
    
    return texto

def eliminar_columnas_duplicadas(df):
    """
    ELIMINA columnas duplicadas (mantiene solo la primera)
    """
    columnas_vistas = set()
    columnas_a_mantener = []
    
    for col in df.columns:
        col_str = str(col).strip().upper()
        
        # Si ya hemos visto esta columna, omitirla
        if col_str in columnas_vistas:
            print(f"⚠️ Columna duplicada eliminada: {col}")
            continue
        else:
            columnas_vistas.add(col_str)
            columnas_a_mantener.append(col)
    
    # Mantener solo columnas únicas
    df = df[columnas_a_mantener].copy()
    return df

def detectar_y_corregir_encabezado(df_raw):
    """Detecta automáticamente la fila de encabezado"""
    if len(df_raw) == 0:
        return df_raw
    
    mejor_fila_header = 0
    max_puntuacion = 0
    
    for idx_fila in range(min(5, len(df_raw))):
        fila = df_raw.iloc[idx_fila]
        puntuacion = 0
        
        for valor in fila:
            if pd.notna(valor):
                texto = str(valor).strip()
                if any(c.isalpha() for c in texto) and len(texto) < 100:
                    puntuacion += 1
                headers_comunes = ['AÑO', 'NOMBRE', 'ESTADO', 'SECTOR', 'TIPO', 
                                  'DURACIÓN', 'EMPRESA', 'DOCENTE', 'FECHA', 
                                  'REGISTRO', 'CODIGO', 'DNI', 'EMAIL', 'RUBRO']
                if texto.upper() in headers_comunes or any(h in texto.upper() for h in headers_comunes):
                    puntuacion += 2
        
        if puntuacion > max_puntuacion:
            max_puntuacion = puntuacion
            mejor_fila_header = idx_fila
    
    if max_puntuacion > 0:
        headers = df_raw.iloc[mejor_fila_header].tolist()
        
        headers_limpios = []
        for i, h in enumerate(headers):
            if pd.notna(h) and str(h).strip() != '':
                header_limpio = str(h).strip().upper()
                headers_limpios.append(header_limpio)
            else:
                headers_limpios.append(f'COLUMNA_{i+1}')
        
        df = df_raw.iloc[mejor_fila_header + 1:].copy()
        df.columns = headers_limpios
        
        return df
    
    headers = [str(h).strip().upper() if pd.notna(h) and str(h).strip() != '' else f'COLUMNA_{i}' 
               for i, h in enumerate(df_raw.iloc[0])]
    df = df_raw.iloc[1:].copy()
    df.columns = headers
    
    return df

def limpiar_nombres_columnas(df):
    """Limpia nombres de columnas"""
    mapeo_nombres = {
        'AÑO': 'AÑO',
        'NRO REGISTRO': 'NRO_REGISTRO',
        'NRO DE CARTA': 'NRO_CARTA',
        'CODIGO': 'CODIGO',
        'DNI': 'DNI',
        'NOMBRE': 'NOMBRE',
        'CELULAR': 'CELULAR',
        'EMAIL': 'EMAIL',
        'DIRECCIÓN': 'DIRECCION',
        'FECHA DE INICIO': 'FECHA_INICIO',
        'FECHA DE FIN': 'FECHA_FIN',
        'DURACIÓN': 'DURACION',
        'EMPRESA': 'EMPRESA',
        'ENCARGADO': 'ENCARGADO',
        'RUBRO': 'RUBRO',
        'FUNCIONES /AREA': 'FUNCIONES_AREA',
        'DOCENTE REVISOR': 'DOCENTE_REVISOR',
        'CORREO ENVIADO': 'CORREO_ENVIADO',
        'TIEMPO (DIAS) DE REVISON DEL DOCENTE': 'TIEMPO_REVISION',
        'EGRESO/CICLO CURSADO': 'EGRESO_CICLO',
        'NOMBRE DEL INFORME': 'NOMBRE_INFORME',
        'SECTOR': 'SECTOR',
        'TIPO': 'TIPO',
        'FECHA DE INFORME DE CONFORMIDAD': 'FECHA_CONFORMIDAD',
        'ESTADO': 'ESTADO',
        'OBSERVACIÓN': 'OBSERVACION',
    }
    
    nuevas_columnas = []
    for col in df.columns:
        col_upper = col.upper().strip()
        if col_upper in mapeo_nombres:
            nuevas_columnas.append(mapeo_nombres[col_upper])
        else:
            nuevas_columnas.append(col_upper)
    
    df.columns = nuevas_columnas
    
    return df

# ==========================================
# FUNCIONES DE ANÁLISIS
# ==========================================

def detectar_tipo_columna(serie):
    """Detecta automáticamente el tipo de columna"""
    serie_limpia = serie.dropna()
    
    if len(serie_limpia) == 0:
        return 'vacia'
    
    if pd.api.types.is_numeric_dtype(serie_limpia):
        return 'numerica'
    
    try:
        pd.to_numeric(serie_limpia)
        return 'numerica'
    except:
        pass
    
    try:
        pd.to_datetime(serie_limpia)
        return 'fecha'
    except:
        pass
    
    unicidad = serie_limpia.nunique() / len(serie_limpia)
    if unicidad < 0.1 or serie_limpia.nunique() < 20:
        return 'categorica'
    
    return 'texto'

def limpiar_datos_categoricos(df, columna):
    """Limpia datos categóricos"""
    df_limpio = df.copy()
    valores = df_limpio[columna].copy()
    
    valores = valores[valores.astype(str).str.upper().str.strip() != columna.upper().strip()]
    
    headers_comunes = ['CANTIDAD', 'PORCENTAJE', 'TOTAL', 'COUNT', 'FRECUENCIA', 
                       'TIPO', 'ESTADO', 'SECTOR', 'NOMBRE', 'VALOR']
    
    mascara_validos = ~valores.astype(str).str.upper().str.strip().isin(headers_comunes)
    valores = valores[mascara_validos]
    
    valores = valores[~valores.astype(str).str.match(r'^\d+$')]
    
    valores_normalizados = valores.apply(normalizar_texto)
    
    mascara_no_nulos = valores_normalizados.notna()
    valores = valores[mascara_no_nulos]
    valores_normalizados = valores_normalizados[mascara_no_nulos]
    
    conteos = valores_normalizados.value_counts().reset_index()
    conteos.columns = [columna, 'Cantidad']
    
    total = conteos['Cantidad'].sum()
    if total > 0:
        conteos['Porcentaje'] = (conteos['Cantidad'] / total * 100).round(2)
    else:
        conteos['Porcentaje'] = 0
    
    return conteos

def generar_grafico_numerico(df, columna):
    """Genera gráficos para columnas numéricas"""
    figs = {}
    
    df_num = df.copy()
    df_num[columna] = pd.to_numeric(df_num[columna], errors='coerce')
    df_num = df_num.dropna(subset=[columna])
    
    if len(df_num) == 0:
        return None
    
    fig_hist = px.histogram(df_num, x=columna, 
                           title=f'📊 DISTRIBUCIÓN: {columna}',
                           color_discrete_sequence=['#667eea'],
                           opacity=0.9)
    fig_hist.update_layout(
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=13, color='#000000'),
        title_font=dict(size=18, color='#000000'),
        xaxis_title=columna,
        yaxis_title='Frecuencia',
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='#e2e8f0', 
                   tickfont=dict(color='#000000', size=12)),
        yaxis=dict(showgrid=True, gridcolor='#e2e8f0',
                   tickfont=dict(color='#000000', size=12))
    )
    figs['histograma'] = fig_hist
    
    fig_box = px.box(df_num, y=columna,
                    title=f'📦 DISTRIBUCIÓN ESTADÍSTICA: {columna}',
                    color_discrete_sequence=['#f5576c'],
                    points='all')
    fig_box.update_layout(
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=13, color='#000000'),
        title_font=dict(size=18, color='#000000'),
        yaxis_title=columna,
        yaxis=dict(showgrid=True, gridcolor='#e2e8f0',
                   tickfont=dict(color='#000000', size=12))
    )
    figs['boxplot'] = fig_box
    
    stats = df_num[columna].describe()
    figs['estadisticas'] = stats
    
    return figs

def generar_grafico_categorico(df, columna):
    """Genera gráficos para columnas categóricas"""
    figs = {}
    
    conteos = limpiar_datos_categoricos(df, columna)
    
    if len(conteos) == 0:
        return None
    
    fig_bar = px.bar(conteos, y=columna, x='Cantidad',
                    title=f'📊 DISTRIBUCIÓN: {columna}',
                    color='Cantidad',
                    color_continuous_scale='Viridis',
                    text='Cantidad',
                    orientation='h')
    fig_bar.update_traces(textposition='outside',
                         marker_line_color='#000000',
                         marker_line_width=1)
    fig_bar.update_layout(
        height=max(400, len(conteos) * 35),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12, color='#000000'),
        title_font=dict(size=18, color='#000000'),
        xaxis_title='Cantidad',
        yaxis_title=columna,
        showlegend=False,
        margin=dict(l=200, r=50, t=80, b=50),
        xaxis=dict(showgrid=True, gridcolor='#e2e8f0',
                   tickfont=dict(color='#000000', size=12)),
        yaxis=dict(showgrid=False,
                   tickfont=dict(color='#000000', size=12))
    )
    figs['barras'] = fig_bar
    
    if len(conteos) <= 15:
        fig_pie = px.pie(conteos, values='Cantidad', names=columna,
                        title=f'🥧 DISTRIBUCIÓN PORCENTUAL: {columna}',
                        hole=0.4,
                        color_discrete_sequence=PALETA_PRINCIPAL)
        fig_pie.update_traces(textposition='inside',
                             textinfo='percent+label+value',
                             textfont=dict(size=11, color='#ffffff'),
                             marker=dict(line=dict(color='#000000', width=2)))
        fig_pie.update_layout(
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12, color='#000000'),
            title_font=dict(size=18, color='#000000'),
            legend=dict(font=dict(size=11, color='#000000'), bgcolor='white')
        )
        figs['circular'] = fig_pie
    
    figs['tabla'] = conteos
    
    return figs

def generar_grafico_temporal(df, columna):
    """Genera gráficos para columnas de fecha"""
    figs = {}
    
    df_temp = df.copy()
    df_temp = df_temp[~df_temp[columna].astype(str).str.upper().str.strip().isin([columna.upper(), 'FECHA', 'DATE'])]
    
    df_temp[columna] = pd.to_datetime(df_temp[columna], errors='coerce')
    df_temp = df_temp.dropna(subset=[columna])
    
    if len(df_temp) == 0:
        return None
    
    df_temp['Año-Mes'] = df_temp[columna].dt.to_period('M').astype(str)
    
    evol = df_temp.groupby('Año-Mes').size().reset_index(name='Cantidad')
    
    fig_line = px.line(evol, x='Año-Mes', y='Cantidad',
                      title=f'📈 EVOLUCIÓN TEMPORAL: {columna}',
                      markers=True)
    fig_line.update_traces(line=dict(color='#667eea', width=3),
                          marker=dict(size=8, color='#f5576c'))
    fig_line.update_layout(
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=13, color='#000000'),
        title_font=dict(size=18, color='#000000'),
        xaxis_title='Período',
        yaxis_title='Cantidad',
        xaxis=dict(tickangle=-45, showgrid=True, gridcolor='#e2e8f0',
                   tickfont=dict(color='#000000', size=12)),
        yaxis=dict(showgrid=True, gridcolor='#e2e8f0',
                   tickfont=dict(color='#000000', size=12))
    )
    figs['evolucion'] = fig_line
    
    df_temp['Año'] = df_temp[columna].dt.year
    por_año = df_temp.groupby('Año').size().reset_index(name='Cantidad')
    
    fig_bar = px.bar(por_año, x='Año', y='Cantidad',
                    title=f'📊 DISTRIBUCIÓN POR AÑO: {columna}',
                    color='Cantidad',
                    color_continuous_scale='Plasma',
                    text='Cantidad')
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=13, color='#000000'),
        title_font=dict(size=18, color='#000000'),
        xaxis_title='Año',
        yaxis_title='Cantidad',
        xaxis=dict(showgrid=True, gridcolor='#e2e8f0',
                   tickfont=dict(color='#000000', size=12)),
        yaxis=dict(showgrid=True, gridcolor='#e2e8f0',
                   tickfont=dict(color='#000000', size=12))
    )
    figs['por_año'] = fig_bar
    
    return figs

def generar_grafico_automatico(df, columna, tipo):
    """Genera el gráfico apropiado según el tipo"""
    if tipo == 'numerica':
        return generar_grafico_numerico(df, columna)
    elif tipo == 'categorica':
        return generar_grafico_categorico(df, columna)
    elif tipo == 'fecha':
        return generar_grafico_temporal(df, columna)
    else:
        return None

# ==========================================
# CARGA DE DATOS
# ==========================================
URL_EXCEL = "https://uptpe-my.sharepoint.com/personal/sistemas_upt_pe/_layouts/15/download.aspx?share=IQAOIPpSBepgQKXqK5_pr0xZASDwWyblJ-22PdBsf-qLSfQ"

@st.cache_data(ttl=30)
def cargar_datos():
    try:
        response = requests.get(URL_EXCEL)
        if response.status_code == 200:
            df_raw = pd.read_excel(io.BytesIO(response.content), sheet_name='PRACTICAS PRE', header=None)
            df = detectar_y_corregir_encabezado(df_raw)
            df = limpiar_nombres_columnas(df)
            df = eliminar_columnas_duplicadas(df)  # ✅ ELIMINA duplicados
            df = df.dropna(how='all')
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al cargar: {e}")
        return pd.DataFrame()

st_autorefresh(interval=30000, limit=100, key="refresh")

# ==========================================
# CARGAR DATOS
# ==========================================
df = cargar_datos()

if df.empty:
    st.error("❌ No se pudieron cargar los datos.")
    st.stop()

# ==========================================
# ANÁLISIS DE COLUMNAS
# ==========================================
def inicializar_analisis_columnas():
    """Inicializa el análisis de columnas"""
    analisis = {}
    for col in df.columns:
        try:
            tipo = detectar_tipo_columna(df[col])
            analisis[col] = {
                'tipo': tipo,
                'unicos': int(df[col].nunique()),
                'nulos': int(df[col].isnull().sum()),
                'total': len(df)
            }
        except:
            analisis[col] = {
                'tipo': 'texto',
                'unicos': 0,
                'nulos': 0,
                'total': len(df)
            }
    return analisis

if 'analisis_columnas' not in st.session_state:
    st.session_state.analisis_columnas = inicializar_analisis_columnas()

# ==========================================
# NAVEGACIÓN
# ==========================================
if 'pagina_actual' not in st.session_state:
    st.session_state.pagina_actual = 'inicio'

def navegar_a(pagina):
    st.session_state.pagina_actual = pagina
    st.rerun()

# ==========================================
# SIDEBAR CON FILTROS DINÁMICOS
# ==========================================
with st.sidebar:
    st.markdown("## 📊 Dashboard Inteligente")
    st.markdown("---")
    
    if st.button("🏠 Inicio", use_container_width=True, key="nav1"):
        navegar_a('inicio')
    
    if st.button("📊 Análisis Automático", use_container_width=True, key="nav2"):
        navegar_a('auto')
    
    if st.button("🔍 Análisis por Columna", use_container_width=True, key="nav3"):
        navegar_a('columnas')
    
    if st.button("🎯 Análisis con Filtros", use_container_width=True, key="nav_filtros"):
        navegar_a('filtros')
    
    if st.button("📈 Indicadores ICACIT", use_container_width=True, key="nav4"):
        navegar_a('icacit')
    
    if st.button("📋 Datos Completos", use_container_width=True, key="nav5"):
        navegar_a('datos')
    
    st.markdown("---")
    st.markdown("### 🎛️ FILTROS GLOBALES")
    
    # Crear filtros dinámicos para cada columna categórica
    filtros_aplicados = {}
    
    for col in df.columns:
        info_col = st.session_state.analisis_columnas.get(col, {})
        tipo = info_col.get('tipo', 'texto')
        
        # Solo mostrar filtros para columnas categóricas con pocos valores únicos
        if tipo == 'categorica' and info_col.get('unicos', 0) <= 30:
            valores_unicos = sorted(df[col].dropna().unique().tolist())
            
            # Limpiar valores
            valores_limpios = [v for v in valores_unicos if pd.notna(v) and str(v).strip() != '']
            
            if len(valores_limpios) > 0 and len(valores_limpios) <= 20:
                seleccion = st.multiselect(
                    f"🔹 {col}",
                    options=valores_limpios,
                    default=valores_limpios,
                    key=f"filtro_{col}"
                )
                
                if seleccion and len(seleccion) < len(valores_limpios):
                    filtros_aplicados[col] = seleccion

# Aplicar filtros globales
df_filtrado = df.copy()
for col, valores in filtros_aplicados.items():
    df_filtrado = df_filtrado[df_filtrado[col].isin(valores)]

# ==========================================
# PÁGINA: INICIO
# ==========================================
if st.session_state.pagina_actual == 'inicio':
    st.markdown("""
    <div class="main-header">
        <h1>📊 Sistema Inteligente de Análisis de Prácticas</h1>
        <p>Universidad Privada de Tacna - Visualización Automática de Datos</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{len(df_filtrado)}</p>
            <p class="kpi-label">Total Registros</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{len(df.columns)}</p>
            <p class="kpi-label">Columnas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        categoricas = sum(1 for v in st.session_state.analisis_columnas.values() if v.get('tipo') == 'categorica')
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{categoricas}</p>
            <p class="kpi-label">Variables Categóricas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        numericas = sum(1 for v in st.session_state.analisis_columnas.values() if v.get('tipo') == 'numerica')
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{numericas}</p>
            <p class="kpi-label">Variables Numéricas</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<p class="section-title">📋 Columnas Detectadas</p>', unsafe_allow_html=True)
    
    for tipo_nombre, tipo_valor in [('Numéricas', 'numerica'), ('Categóricas', 'categorica'), ('Fechas', 'fecha')]:
        cols = [col for col, info in st.session_state.analisis_columnas.items() if info.get('tipo') == tipo_valor]
        if cols:
            st.markdown(f"### {tipo_nombre}")
            for col in cols:
                unicos = st.session_state.analisis_columnas[col].get('unicos', 0)
                st.markdown(f"✅ **{col}** ({unicos} valores únicos)")
            st.markdown("---")

# ==========================================
# PÁGINA: ANÁLISIS AUTOMÁTICO
# ==========================================
elif st.session_state.pagina_actual == 'auto':
    st.markdown("""
    <div class="main-header">
        <h1>📊 Análisis Automático Inteligente</h1>
        <p>Gráficos generados automáticamente</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="section-title">🎯 Gráficos por Columna</p>', unsafe_allow_html=True)
    
    for col in df_filtrado.columns:
        info_col = st.session_state.analisis_columnas.get(col, {})
        tipo = info_col.get('tipo', 'texto')
        unicos = info_col.get('unicos', 0)
        
        if tipo in ['numerica', 'categorica', 'fecha']:
            expander_title = f"📊 {col} - {tipo.upper()} ({unicos} valores únicos)"
            
            with st.expander(expander_title, expanded=False):
                st.markdown(f'<p class="sub-title">Análisis de: {col}</p>', unsafe_allow_html=True)
                
                graficos = generar_grafico_automatico(df_filtrado, col, tipo)
                
                if graficos:
                    if tipo == 'numerica':
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(graficos['histograma'], use_container_width=True)
                        with col2:
                            st.plotly_chart(graficos['boxplot'], use_container_width=True)
                        st.markdown("#### 📈 Estadísticas Descriptivas:")
                        st.dataframe(graficos['estadisticas'], use_container_width=True)
                    
                    elif tipo == 'categorica':
                        if 'barras' in graficos:
                            st.plotly_chart(graficos['barras'], use_container_width=True)
                        if 'circular' in graficos:
                            st.plotly_chart(graficos['circular'], use_container_width=True)
                        st.markdown("#### 📋 Tabla de Frecuencias:")
                        st.dataframe(graficos['tabla'], use_container_width=True, hide_index=True)
                    
                    elif tipo == 'fecha':
                        if 'evolucion' in graficos:
                            st.plotly_chart(graficos['evolucion'], use_container_width=True)
                        if 'por_año' in graficos:
                            st.plotly_chart(graficos['por_año'], use_container_width=True)

# ==========================================
# PÁGINA: ANÁLISIS POR COLUMNA
# ==========================================
elif st.session_state.pagina_actual == 'columnas':
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Análisis Detallado por Columna</h1>
        <p>Selecciona una columna específica</p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(df_filtrado.columns) == 0:
        st.warning("No hay columnas disponibles.")
        st.stop()
    
    def format_columna(x):
        try:
            info = st.session_state.analisis_columnas.get(x, {})
            tipo = info.get('tipo', 'desconocido')
            return f"{x} ({tipo})"
        except:
            return str(x)
    
    col_seleccionada = st.selectbox(
        "Selecciona una columna:",
        options=list(df_filtrado.columns),
        format_func=format_columna,
        key="selectbox_columna"
    )
    
    if col_seleccionada:
        info_col = st.session_state.analisis_columnas.get(col_seleccionada, {})
        tipo = info_col.get('tipo', 'texto')
        unicos = info_col.get('unicos', 0)
        nulos = info_col.get('nulos', 0)
        
        st.markdown(f'<p class="section-title">📊 Análisis de: {col_seleccionada}</p>', unsafe_allow_html=True)
        st.info(f"**Tipo:** {tipo.upper()} | **Valores únicos:** {unicos} | **Nulos:** {nulos}")
        
        graficos = generar_grafico_automatico(df_filtrado, col_seleccionada, tipo)
        
        if graficos:
            if tipo == 'numerica':
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(graficos['histograma'], use_container_width=True)
                with col2:
                    st.plotly_chart(graficos['boxplot'], use_container_width=True)
                st.dataframe(graficos['estadisticas'], use_container_width=True)
            
            elif tipo == 'categorica':
                if 'barras' in graficos:
                    st.plotly_chart(graficos['barras'], use_container_width=True)
                if 'circular' in graficos:
                    st.plotly_chart(graficos['circular'], use_container_width=True)
                st.dataframe(graficos['tabla'], use_container_width=True, hide_index=True)
            
            elif tipo == 'fecha':
                if 'evolucion' in graficos:
                    st.plotly_chart(graficos['evolucion'], use_container_width=True)
                if 'por_año' in graficos:
                    st.plotly_chart(graficos['por_año'], use_container_width=True)

# ==========================================
# PÁGINA: ANÁLISIS CON FILTROS (NUEVA)
# ==========================================
elif st.session_state.pagina_actual == 'filtros':
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Análisis con Filtros Dinámicos</h1>
        <p>Usa los filtros del sidebar para analizar datos específicos</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**📊 Registros filtrados:** {len(df_filtrado)} de {len(df)}")
    
    if len(filtros_aplicados) > 0:
        st.markdown("### 🔍 Filtros Aplicados:")
        for col, valores in filtros_aplicados.items():
            st.write(f"**{col}:** {', '.join(str(v) for v in valores[:5])}{'...' if len(valores) > 5 else ''}")
    
    st.markdown('<p class="section-title">📈 Gráficos Estadísticos con Filtros</p>', unsafe_allow_html=True)
    
    # Selector de columnas para graficar
    col_x = st.selectbox(
        "Selecciona columna para el eje X (categoría):",
        options=[col for col, info in st.session_state.analisis_columnas.items() 
                if info.get('tipo') in ['categorica', 'texto']],
        key="filtro_col_x"
    )
    
    col_y = st.selectbox(
        "Selecciona columna para el eje Y (numérica):",
        options=[col for col, info in st.session_state.analisis_columnas.items() 
                if info.get('tipo') == 'numerica'],
        key="filtro_col_y"
    )
    
    if col_x and col_y:
        # Preparar datos
        df_plot = df_filtrado.copy()
        df_plot[col_y] = pd.to_numeric(df_plot[col_y], errors='coerce')
        df_plot = df_plot.dropna(subset=[col_x, col_y])
        
        if len(df_plot) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de barras agrupadas
                df_agg = df_plot.groupby(col_x)[col_y].agg(['mean', 'sum', 'count']).reset_index()
                df_agg.columns = [col_x, 'Promedio', 'Suma', 'Cantidad']
                
                fig_bar = px.bar(df_agg, x=col_x, y='Suma',
                               title=f'📊 {col_y} por {col_x}',
                               color='Cantidad',
                               color_continuous_scale='Viridis',
                               text='Cantidad')
                fig_bar.update_traces(textposition='outside')
                fig_bar.update_layout(
                    height=500,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=12, color='#000000'),
                    title_font=dict(size=16, color='#000000'),
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Boxplot
                fig_box = px.box(df_plot, x=col_x, y=col_y,
                               title=f'📦 Distribución de {col_y} por {col_x}',
                               color=col_x,
                               color_discrete_sequence=PALETA_PRINCIPAL)
                fig_box.update_layout(
                    height=500,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=12, color='#000000'),
                    title_font=dict(size=16, color='#000000'),
                    xaxis_tickangle=-45,
                    showlegend=False
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Tabla de estadísticas
            st.markdown("### 📋 Estadísticas por Categoría")
            st.dataframe(df_agg, use_container_width=True, hide_index=True)

# ==========================================
# PÁGINA: INDICADORES ICACIT
# ==========================================
elif st.session_state.pagina_actual == 'icacit':
    st.markdown("""
    <div class="main-header">
        <h1>📈 Indicadores ICACIT</h1>
        <p>Métricas de gestión y acreditación</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_estado = next((col for col in df_filtrado.columns if 'ESTADO' in col.upper()), None)
    col_sector = next((col for col in df_filtrado.columns if 'SECTOR' in col.upper()), None)
    col_duracion = next((col for col in df_filtrado.columns if 'DURACION' in col.upper()), None)
    col_año = next((col for col in df_filtrado.columns if 'AÑO' in col.upper()), None)
    
    if col_estado and col_año:
        st.markdown('<p class="section-title">📊 Tasa de Culminación por Año</p>', unsafe_allow_html=True)
        
        tasas = []
        for año in sorted(df_filtrado[col_año].dropna().unique()):
            df_año = df_filtrado[df_filtrado[col_año] == año]
            total = len(df_año)
            terminados = len(df_año[df_año[col_estado].astype(str).str.upper().str.contains('TERMINADO', na=False)])
            tasa = (terminados / total * 100) if total > 0 else 0
            tasas.append({'Año': año, 'Total': total, 'Terminados': terminados, 'Tasa (%)': round(tasa, 1)})
        
        df_tasas = pd.DataFrame(tasas)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(df_tasas, x='Año', y='Tasa (%)',
                        title='TASA DE CULMINACIÓN (%)',
                        color='Tasa (%)',
                        color_continuous_scale='RdYlGn',
                        text='Tasa (%)')
            fig.update_traces(textposition='outside')
            fig.update_layout(yaxis_range=[0, 100], height=450,
                            plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(df_tasas, use_container_width=True, hide_index=True)
    
    if col_sector:
        st.markdown('<p class="section-title">🏢 Distribución por Sector</p>', unsafe_allow_html=True)
        
        sector_stats = df_filtrado[col_sector].value_counts().reset_index()
        sector_stats.columns = ['Sector', 'Cantidad']
        sector_stats['Porcentaje'] = (sector_stats['Cantidad'] / sector_stats['Cantidad'].sum() * 100).round(1)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(sector_stats, values='Cantidad', names='Sector',
                        title='DISTRIBUCIÓN POR SECTOR',
                        hole=0.4,
                        color_discrete_sequence=PALETA_PRINCIPAL)
            fig.update_traces(textposition='inside', textinfo='percent+label+value')
            fig.update_layout(height=450, plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(sector_stats, use_container_width=True, hide_index=True)
    
    if col_duracion:
        st.markdown('<p class="section-title">⏱️ Duración de Prácticas</p>', unsafe_allow_html=True)
        
        df_dur = df_filtrado.copy()
        df_dur['DURACION_NUM'] = pd.to_numeric(df_dur[col_duracion], errors='coerce')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Promedio", f"{df_dur['DURACION_NUM'].mean():.0f} días")
        with col2:
            st.metric("Mediana", f"{df_dur['DURACION_NUM'].median():.0f} días")
        with col3:
            st.metric("Máximo", f"{df_dur['DURACION_NUM'].max():.0f} días")
        with col4:
            st.metric("Mínimo", f"{df_dur['DURACION_NUM'].min():.0f} días")

# ==========================================
# PÁGINA: DATOS COMPLETOS
# ==========================================
elif st.session_state.pagina_actual == 'datos':
    st.markdown("""
    <div class="main-header">
        <h1>📋 Datos Completos</h1>
        <p>Visualización y exportación de datos</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**📊 Registros:** {len(df_filtrado)}")
    
    st.dataframe(df_filtrado, use_container_width=True, height=600)
    
    csv = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar CSV",
        data=csv,
        file_name=f"practicas_pre_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    <p><strong>🎓 Universidad Privada de Tacna</strong> - Escuela de Ingeniería de Sistemas</p>
    <p>📊 Dashboard Inteligente con Filtros Dinámicos | Acreditación ICACIT</p>
    <p>Actualizado: {fecha}</p>
</div>
""".format(fecha=datetime.now().strftime('%d/%m/%Y %H:%M:%S')), unsafe_allow_html=True)