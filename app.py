"""
app.py - Front end for the clinical guideline assistant.

Run with:
    streamlit run app.py

A wide, three-column layout: knowledge base (left), the question and answer
(centre), and a context panel (right) that shows how it works before a question
and the source citations after. All three are fixed page columns - nothing is
collapsible. Independent educational prototype; does not use the NHS logo or
imply any official affiliation.
"""

import streamlit as st
from rag import answer, list_sources

st.set_page_config(
    page_title="Clinical Guideline Assistant",
    page_icon=":hospital:",
    layout="wide",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root{
  --nhs-blue:#005EB6; --nhs-blue-dark:#003d7a; --nhs-tint:#e8f1fb;
  --ink:#0b1f33; --muted:#5b6b7b; --line:#e3e8ee; --bg:#f6f8fa; --card:#ffffff;
}
[data-testid="stAppViewContainer"] *{
  font-family:'IBM Plex Sans',-apple-system,Arial,sans-serif;
}
[data-testid="stAppViewContainer"]{ background:var(--bg); }
[data-testid="stHeader"]{ background:transparent; }
[data-testid="stToolbar"], [data-testid="stDeployButton"]{ display:none; }
#MainMenu{ visibility:hidden; }
[data-testid="stMainBlockContainer"], .block-container{
  max-width:1320px; margin:0 auto; padding-top:2.2rem; padding-bottom:3rem;
}

/* Header */
.app-head{ display:flex; align-items:center; gap:14px; }
.app-logo{ width:46px; height:46px; border-radius:13px; background:var(--nhs-blue);
  display:flex; align-items:center; justify-content:center; flex:0 0 auto; }
.app-title{ font-size:1.6rem; font-weight:700; color:var(--ink); margin:0;
  line-height:1.1; letter-spacing:-0.02em; }
.app-sub{ color:var(--muted); font-size:.97rem; margin:.2rem 0 0; }
.head-rule{ border:0; border-top:1px solid var(--line); margin:1.2rem 0 1.8rem; }

/* Labels */
.section-label{ text-transform:uppercase; letter-spacing:.09em; font-size:.72rem;
  font-weight:600; color:var(--nhs-blue); margin:.1rem 0 .55rem; }
.chips-label{ color:var(--muted); font-size:.82rem; margin:.8rem 0 .4rem; }

/* Input */
[data-testid="stTextInput"] label{ font-weight:600; color:var(--ink); font-size:.95rem; }
[data-testid="stTextInput"] input{
  border:1px solid #cdd6df !important; border-radius:10px; padding:.82rem .95rem;
  font-size:1.02rem; background:var(--card); }
[data-testid="stTextInput"] input:focus{
  border-color:var(--nhs-blue) !important; outline:none !important;
  box-shadow:0 0 0 3px rgba(0,94,182,.16) !important; }

/* Primary button */
.stButton > button[kind="primary"]{
  background:var(--nhs-blue); color:#fff; border:1px solid var(--nhs-blue);
  border-radius:10px; font-weight:600; font-size:.97rem; padding:.62rem 2rem;
  transition:background .15s ease, transform .05s ease; }
.stButton > button[kind="primary"]:hover{
  background:var(--nhs-blue-dark); border-color:var(--nhs-blue-dark); color:#fff; }
.stButton > button[kind="primary"]:active{ transform:translateY(1px); }

/* Example chips */
.stButton > button[kind="secondary"]{
  background:var(--card); color:var(--nhs-blue-dark); border:1px solid #d3e1f2;
  border-radius:10px; font-weight:500; font-size:.85rem; padding:.55rem .7rem;
  transition:background .12s ease, border-color .12s ease; }
.stButton > button[kind="secondary"]:hover{
  background:var(--nhs-tint); border-color:var(--nhs-blue); color:var(--nhs-blue-dark); }

/* Answer card */
[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--card); border:1px solid var(--line) !important; border-radius:14px;
  box-shadow:0 1px 2px rgba(11,31,51,.05), 0 6px 20px rgba(11,31,51,.04);
  padding:.7rem 1.6rem 1.2rem; }
[data-testid="stVerticalBlockBorderWrapper"] p,
[data-testid="stVerticalBlockBorderWrapper"] li{ line-height:1.65; }

/* Knowledge base panel */
.kb-panel{ background:var(--card); border:1px solid var(--line); border-radius:14px;
  padding:1.1rem 1.1rem 1.2rem; }
.kb-head{ display:flex; align-items:center; flex-wrap:wrap; gap:.5rem; margin-bottom:.2rem; }
.kb-title{ font-weight:700; color:var(--ink); font-size:1.05rem; }
.kb-badge{ background:var(--nhs-tint); color:var(--nhs-blue-dark); font-size:.72rem;
  font-weight:600; padding:.12rem .5rem; border-radius:999px; }
.kb-note{ color:var(--muted); font-size:.8rem; margin:.1rem 0 .7rem; }
.kb-item{ display:flex; gap:.55rem; align-items:flex-start; padding:.45rem .4rem;
  border-radius:9px; font-size:.85rem; color:var(--ink); line-height:1.35; }
.kb-item:hover{ background:var(--bg); }
.kb-item svg{ flex:0 0 auto; margin-top:1px; color:var(--nhs-blue); }

/* Right-panel info cards */
.info-card{ background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:.85rem 1rem; margin-bottom:.7rem; }
.info-title{ font-weight:600; color:var(--ink); font-size:.92rem;
  display:flex; gap:.55rem; align-items:center; }
.info-title svg{ color:var(--nhs-blue); flex:0 0 auto; }
.info-desc{ color:var(--muted); font-size:.83rem; margin:.3rem 0 0; line-height:1.45; }

/* Source expanders */
[data-testid="stExpander"]{ border:1px solid var(--line); border-radius:10px;
  background:var(--card); margin-bottom:.45rem; }
[data-testid="stExpander"] summary{ font-weight:500; color:var(--nhs-blue-dark);
  font-size:.88rem; }
[data-testid="stExpander"] summary:hover{ color:var(--nhs-blue); }

/* Footer */
.app-footer{ margin-top:2.6rem; padding-top:1rem; border-top:1px solid var(--line);
  color:var(--muted); font-size:.8rem; line-height:1.55; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PULSE_SVG = ('<svg viewBox="0 0 24 24" width="25" height="25" fill="none" stroke="#fff" '
             'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
             '<path d="M3 12h4l2-5 4 10 2-5h6"/></svg>')
DOC_SVG = ('<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" '
           'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
           '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>'
           '<path d="M14 3v5h5M9 13h6M9 17h6"/></svg>')


def icon(path, size=16):
    return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" '
            f'stroke="currentColor" stroke-width="1.9" stroke-linecap="round" '
            f'stroke-linejoin="round">{path}</svg>')


HOW_IT_WORKS = [
    (icon('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>'
          '<path d="M14 3v5h5"/>'),
     "Grounded in guidelines",
     "Answers are composed only from the indexed documents, never from general knowledge."),
    (icon('<path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="9"/>'),
     "Every claim cited",
     "Each statement is tagged to the source passage, shown in the panel here."),
    (icon('<circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h.01"/>'),
     "Refuses when unsure",
     "If the answer is not in the guidelines, it says so rather than guessing."),
]

EXAMPLES = [
    "First-line management of STEMI",
    "How is asthma diagnosed in adults?",
    "Recognising neutropenic sepsis",
]


def pretty_name(filename):
    import re
    name = re.sub(r"\.pdf$", "", filename, flags=re.I)
    name = re.split(r"-pdf-|-\d{6,}", name)[0]
    name = name.replace("-", " ").replace("_", " ").strip()
    return name[:1].upper() + name[1:] if name else filename


st.session_state.setdefault("query", "")
st.session_state.setdefault("submitted", False)


def submit_example(text):
    st.session_state.query = text
    st.session_state.submitted = True


# Header (full width)
st.markdown(
    f'<div class="app-head"><div class="app-logo">{PULSE_SVG}</div>'
    f'<div><div class="app-title">Clinical Guideline Assistant</div>'
    f'<div class="app-sub">Guideline-grounded clinical answers, with citations to the source.</div>'
    f'</div></div><hr class="head-rule">',
    unsafe_allow_html=True,
)

# Three fixed page columns
kb_col, main_col, side_col = st.columns([1, 2, 1.1], gap="large")

# Left: knowledge base panel
with kb_col:
    try:
        sources = list_sources()
    except Exception:
        sources = []
    items = "".join(
        f'<div class="kb-item">{DOC_SVG}<span>{pretty_name(s)}</span></div>'
        for s in sources
    ) if sources else '<div class="kb-note">No guidelines indexed yet. Run python ingest.py first.</div>'
    st.markdown(
        f'<div class="kb-panel">'
        f'<div class="kb-head"><span class="kb-title">Knowledge base</span>'
        f'<span class="kb-badge">{len(sources)}</span></div>'
        f'<div class="kb-note">Answers are drawn only from these indexed guidelines.</div>'
        f'{items}</div>',
        unsafe_allow_html=True,
    )

# Centre: question
with main_col:
    st.text_input(
        "Ask a clinical question",
        key="query",
        placeholder="e.g. What is the first-line management of...",
    )
    st.markdown('<div class="chips-label">Try an example</div>', unsafe_allow_html=True)
    chip_cols = st.columns(len(EXAMPLES))
    for col, ex in zip(chip_cols, EXAMPLES):
        with col:
            st.button(ex, key=f"ex_{ex}", type="secondary",
                      on_click=submit_example, args=(ex,), use_container_width=True)
    ask = st.button("Ask", type="primary")

# Decide whether to run
run = ask or st.session_state.submitted
st.session_state.submitted = False
question = st.session_state.query.strip()

response = passages = err = None
if run and question:
    with st.spinner("Searching guidelines and composing an answer..."):
        try:
            response, passages = answer(question)
        except Exception as e:
            err = e

with main_col:
    if err:
        st.error(
            f"Something went wrong: {err}\n\n"
            "Checklist: have you run `python ingest.py`, added PDFs to "
            "guidelines/, and set ANTHROPIC_API_KEY in your .env file?"
        )
    elif response:
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
            st.markdown(response)

# Right: context panel
with side_col:
    if response and passages:
        st.markdown('<div class="section-label">Sources</div>', unsafe_allow_html=True)
        for i, (doc, meta) in enumerate(passages, start=1):
            with st.expander(f"[{i}]  {pretty_name(meta['source'])} - page {meta['page']}"):
                st.write(doc)
    elif not err:
        st.markdown('<div class="section-label">How it works</div>', unsafe_allow_html=True)
        for svg, title, desc in HOW_IT_WORKS:
            st.markdown(
                f'<div class="info-card"><div class="info-title">{svg}{title}</div>'
                f'<div class="info-desc">{desc}</div></div>',
                unsafe_allow_html=True,
            )

# Footer
st.markdown(
    '<div class="app-footer">'
    'Independent educational prototype. Not affiliated with, or endorsed by, '
    'the NHS or NICE. Not a clinical decision-making tool - always verify '
    'against the original guideline.'
    '</div>',
    unsafe_allow_html=True,
)