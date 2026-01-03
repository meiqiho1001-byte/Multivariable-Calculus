import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
import re

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Calculus Intelligence Pro", layout="wide")

# ---------------------------
# STYLE
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap');

.stApp {
    background-color: #05070a;
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
}

.topic-card {
    background: rgba(23, 28, 40, 0.75);
    padding: 30px;
    border-radius: 20px;
    margin-bottom: 25px;
}

/* Topic color accents */
.geom { border: 2px solid #38bdf8; }      /* blue */
.grad { border: 2px solid #22c55e; }      /* green */
.partial { border: 2px solid #facc15; }   /* yellow */
.theory { border: 2px solid #a78bfa; }    /* purple */
.critical { border: 2px solid #f472b6; }  /* pink */

.topic-header {
    color: #e5e7eb;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 15px;
}

.explanation-text {
    color: #b0b8c1;
    font-size: 1.05rem;
    line-height: 1.6;
    margin-bottom: 18px;
}

/* Widgets */
label[data-testid="stWidgetLabel"],
.stSelectbox label,
.stRadio label,
.stTextInput label {
    color: #ffffff !important;
}

.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #020617 !important;
    color: white !important;
    border: 2px solid #38bdf8 !important;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# PARSER
# ---------------------------
def robust_math_parse(text):
    text = text.lower().replace(" ", "")
    text = re.sub(r'(\d)([a-z\(])', r'\1*\2', text)
    text = re.sub(r'([x-z])([a-z\(])', r'\1*\2', text)
    return text.replace("^", "**")

MATH_MAP = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "sqrt": sp.sqrt, "log": sp.log}

# ---------------------------
# SIDEBAR INTERVALS
# ---------------------------
with st.sidebar:
    st.header("📍 Domain Space")
    st.write("Enter exact intervals for calculation:")
    
    col_x1, col_x2 = st.columns(2)
    x_min = col_x1.number_input("X Min", value=-5.0)
    x_max = col_x2.number_input("X Max", value=5.0)
    
    col_y1, col_y2 = st.columns(2)
    y_min = col_y1.number_input("Y Min", value=-5.0)
    y_max = col_y2.number_input("Y Max", value=5.0)
    
    col_z1, col_z2 = st.columns(2)
    z_min = col_z1.number_input("Z Min", value=-5.0)
    z_max = col_z2.number_input("Z Max", value=5.0)

# ---------------------------
# FUNCTION INPUT
# ---------------------------
st.title("Multivariable Calculus Explorer")
user_raw = st.text_input("Define your function f(x, y, z):", value="sin(2*x) + 5*cos(y) - 8*z")

x, y, z = sp.symbols("x y z")
try:
    processed = robust_math_parse(user_raw)
    f_sym = sp.parse_expr(processed, local_dict=MATH_MAP)
    vars_present = f_sym.free_symbols
except:
    st.error("Syntax Error: Ensure standard function notation is used.")
    st.stop()

fx = sp.diff(f_sym, x)
fy = sp.diff(f_sym, y)
fz = sp.diff(f_sym, z)

# ---------------------------
# DASHBOARD LAYOUT
# ---------------------------
col_left, col_right = st.columns([3,2])

# ---------------------------
# TOPIC I: GEOMETRY
# ---------------------------
with col_left:
    st.markdown('<div class="topic-card geom">', unsafe_allow_html=True)
    st.markdown('<div class="topic-header">I. Geometric Meaning & Visualization</div>', unsafe_allow_html=True)

    if z in vars_present:
        st.markdown('<div class="explanation-text"><b>Function of Three Variables:</b> This represents a scalar field. We use a level surface (z-slice) to visualize the topography at a specific altitude.</div>', unsafe_allow_html=True)
        z_slice = st.select_slider("Adjust Z-Slice Position", options=np.round(np.linspace(z_min, z_max, 21), 2), value=0.0)
        f_plot = f_sym.subs(z, z_slice)
    else:
        st.markdown('<div class="explanation-text"><b>Function of Two Variables:</b> Represents a standard 3D surface mapping (x,y) to a vertical height.</div>', unsafe_allow_html=True)
        f_plot = f_sym

    # Grid
    xv = np.linspace(x_min, x_max, 50)
    yv = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(xv, yv)
    fn = sp.lambdify((x, y), f_plot, "numpy")
    Z_vals = fn(X,Y)
    if np.isscalar(Z_vals): Z_vals = np.full(X.shape, Z_vals)

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z_vals, colorscale="IceFire")])
    fig.update_layout(height=550, margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    z_low, z_high = np.nanmin(Z_vals), np.nanmax(Z_vals)
    st.write(f"**Domain:** $x \in [{x_min}, {x_max}], y \in [{y_min}, {y_max}]$")
    st.write(f"**Observed Range:** $f \in [{z_low:.2f}, {z_high:.2f}]$")

    if any(trig in user_raw.lower() for trig in ["sin","cos","tan"]):
        st.markdown("---")
        st.write("**Shape:** Periodic/Wave Surface. The function oscillates between high and low points.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# TOPIC II: GRADIENT
# ---------------------------
with col_right:
    st.markdown('<div class="topic-card grad">', unsafe_allow_html=True)
    st.markdown('<div class="topic-header">II. Gradient & Steepest Ascent</div>', unsafe_allow_html=True)
    st.markdown('<div class="explanation-text">The <b>Gradient</b> ($\\nabla f$) is a vector field that lives in the domain of the function. Its most critical property is that at any given point, the gradient vector points in the <b>Direction of Steepest Ascent</b>. If you were standing on this surface, following the gradient would be the most efficient path to go uphill. The magnitude of the gradient tells you exactly how steep that slope is.</div>', unsafe_allow_html=True)

    g_comps = [sp.latex(fx), sp.latex(fy)]
    if z in vars_present: g_comps.append(sp.latex(fz))
    st.latex(rf"\nabla f = \langle {', '.join(g_comps)} \rangle")

    st.write("**Geometric Meaning:**")
    st.markdown("- **Direction:** Points toward local maxima.")
    st.markdown("- **Orthogonality:** The gradient is always perpendicular to the level curves (contours).")
    st.markdown("- **Rate of Change:** $|\nabla f|$ is the maximum possible directional derivative.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# TOPIC III: PARTIAL DERIVATIVE
# ---------------------------
st.markdown('<div class="topic-card partial">', unsafe_allow_html=True)
st.markdown('<div class="topic-header">III. Partial Derivative Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="explanation-text">A <b>partial derivative</b> measures how the function changes as one variable varies, while others are held constant.</div>', unsafe_allow_html=True)

cols = st.columns(2)
with cols[0]:
    var_choice = st.selectbox("Differentiate with respect to:", [v for v in ['x','y','z'] if sp.Symbol(v) in vars_present])
with cols[1]:
    order = st.radio("Order of derivative:", [1,2], horizontal=True)

var = sp.Symbol(var_choice)
partial = sp.diff(f_sym, var, order)
st.write("**Resulting Partial Derivative:**")
st.latex(rf"\frac{{\partial^{order} f}}{{\partial {var_choice}^{order}}} = {sp.latex(partial)}")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# TOPIC IV: TOTAL DIFFERENTIAL
# ---------------------------
st.markdown('<div class="topic-card theory">', unsafe_allow_html=True)
st.markdown('<div class="topic-header">IV. Total Differential (df)</div>', unsafe_allow_html=True)
st.markdown('<div class="explanation-text">The total differential represents the principal part of the change in a function $f$ with respect to changes in the independent variables.</div>', unsafe_allow_html=True)
df_expr = rf"df = \left( {sp.latex(fx)} \right)dx + \left( {sp.latex(fy)} \right)dy"
if z in vars_present: df_expr += rf" + \left( {sp.latex(fz)} \right)dz"
st.latex(df_expr)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# TOPIC V: CONTINUITY & DIFFERENTIABILITY
# ---------------------------
st.markdown('<div class="topic-card theory">', unsafe_allow_html=True)
st.markdown('<div class="topic-header">V. Continuity & Differentiability</div>', unsafe_allow_html=True)
st.markdown('<div class="explanation-text">Continuity and differentiability are fundamental properties. A function differentiable at a point is always continuous there.</div>', unsafe_allow_html=True)

elementary_funcs = (sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.sqrt)
is_elementary = all(
    isinstance(node, (sp.Add, sp.Mul, sp.Pow, sp.Symbol, sp.Number)) or
    any(isinstance(node, f) for f in elementary_funcs)
    for node in sp.preorder_traversal(f_sym)
)
if is_elementary:
    st.write("✅ **Continuity:** Function is elementary → continuous everywhere.")
    st.write("✅ **Differentiability:** All partial derivatives exist → differentiable everywhere.")
else:
    st.write("⚠️ **General Case:** Differentiability not guaranteed. Further analysis needed.")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# TOPIC VI: CRITICAL POINTS
# ---------------------------
st.markdown('<div class="topic-card critical">', unsafe_allow_html=True)
st.markdown('<div class="topic-header">VI. Critical Points & Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="explanation-text">Critical points occur where $\\nabla f = 0$. We classify using $D = f_{xx}f_{yy}-f_{xy}^2$.</div>', unsafe_allow_html=True)

try:
    sols = sp.solve([fx, fy], (x,y), dict=True)
    if sols:
        for s in sols[:2]:
            fxx = sp.diff(f_sym, x,2).subs(s)
            fyy = sp.diff(f_sym, y,2).subs(s)
            fxy = sp.diff(f_sym, x,y).subs(s)
            D = fxx*fyy - fxy**2
            label = "Saddle Point" if D<0 else ("Local Minimum" if fxx>0 else "Local Maximum")
            st.write(f"📍 **Point ({s[x]}, {s[y]})** — This is a **{label}**.")
    else:
        st.write("No stationary points found for the given variables.")
except:
    st.write("Analytical solution unavailable for this complexity level.")

st.markdown('</div>', unsafe_allow_html=True)
