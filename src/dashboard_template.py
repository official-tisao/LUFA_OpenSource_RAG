#!/usr/bin/env python3
r"""
HTML template for the LUFA RAG Evaluation Dashboard.

Design:
  * The page renders a COMPLETE static report first: KPI cards, basic bar
    charts (pure CSS/SVG), and the full detailed table — all written by Python.
    Everything here is readable with JavaScript disabled.
  * `<noscript>` keeps the static blocks visible and hides the JS-only chrome.
  * When JS runs, it hides the static charts, reveals Chart.js canvases, a
    per-column table filter bar and a "Normalize Data" gear, and recomputes
    every KPI / chart / table row live from the embedded per-row data.

Placeholders filled by dashboard_generator.generate_dashboard:
  __DATA_PLACEHOLDER__  __GEN_AT__  __STATIC_KPIS__  __STATIC_GEN__
  __STATIC_RET__  __STATIC_JUDGE__  __STATIC_LANG__  __STATIC_DIFF__
  __STATIC_CAT__  __STATIC_TABLE__
"""

DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>LUFA RAG Evaluation Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#0f172a; --card:#1e293b; --border:#334155; --muted:#94a3b8;
    --accent:#93c5fd; --accent2:#3b82f6;
  }
  *{box-sizing:border-box;}
  body{ background:var(--bg); color:#ffffff; font-family:'Segoe UI',system-ui,sans-serif;
        margin:0; padding:24px; font-weight:700; }
  /* Requirement 6: all supervisor-facing text white + bold for readability */
  body, .card, td, th, p, span, div, label, button, h1, h2, h3 { color:#ffffff; }
  a{ color:var(--accent); }

  .topbar{ display:flex; align-items:flex-start; justify-content:space-between;
           gap:16px; flex-wrap:wrap; margin-bottom:24px; }
  .topbar h1{ font-size:1.9rem; font-weight:800; margin:0 0 4px; color:#ffffff; }
  .topbar .sub{ color:#cbd5e1; font-size:.85rem; font-weight:600; }

  .norm-wrap{ display:flex; align-items:center; gap:10px; background:var(--card);
              border:1px solid var(--border); border-radius:10px; padding:10px 14px; }
  .norm-wrap label{ font-weight:800; cursor:pointer; user-select:none; }
  .gear{ font-size:1.1rem; }
  /* native checkbox shown in no-JS; styled toggle shown with JS */
  .norm-cb{ width:18px; height:18px; accent-color:var(--accent2); cursor:pointer; }
  .norm-note{ color:#cbd5e1; font-size:.72rem; font-weight:600; max-width:230px; }

  .card{ background:var(--card); border-radius:12px; padding:20px;
         border:1px solid var(--border); }
  .section-title{ font-size:1.05rem; font-weight:800; color:var(--accent);
                  margin:0 0 14px; text-transform:uppercase; letter-spacing:.05em; }

  .kpi-grid{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px; margin-bottom:26px; }
  @media(min-width:640px){ .kpi-grid{ grid-template-columns:repeat(4,1fr);} }
  @media(min-width:1100px){ .kpi-grid{ grid-template-columns:repeat(7,1fr);} }
  .metric-card{ background:linear-gradient(135deg,#1e3a5f,#1e293b); border-radius:10px;
                padding:14px; border:1px solid #2563eb55; text-align:center; }
  .kpi-value{ font-size:1.5rem; font-weight:800; color:#ffffff; line-height:1.1; }
  .kpi-label{ font-size:.72rem; color:#cbd5e1; margin-top:6px; font-weight:700;
              text-transform:uppercase; letter-spacing:.03em; }

  .grid2{ display:grid; grid-template-columns:1fr; gap:22px; margin-bottom:22px; }
  @media(min-width:1024px){ .grid2{ grid-template-columns:1fr 1fr; } }

  canvas{ max-height:300px; }

  /* static (no-JS) bar charts */
  .sbar-group{ display:flex; flex-direction:column; gap:9px; }
  .sbar-row{ display:flex; align-items:center; gap:10px; font-size:.8rem; }
  .sbar-label{ width:92px; flex:none; color:#ffffff; font-weight:700;
               white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .sbar-track{ flex:1; height:16px; background:#0f172a; border-radius:6px;
               border:1px solid var(--border); overflow:hidden; }
  .sbar-fill{ display:block; height:100%; border-radius:6px; }
  .sbar-val{ width:48px; flex:none; text-align:right; color:#ffffff; font-weight:800; }
  .sbar-subhead{ margin:6px 0 2px; font-weight:800; color:var(--accent); font-size:.82rem; }
  .sbar-empty{ color:#cbd5e1; font-weight:700; }

  /* filter bar */
  .filter-bar{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:14px; }
  .filter-bar input, .filter-bar select{
      background:#0f172a; color:#ffffff; font-weight:700;
      border:1px solid var(--border); border-radius:8px; padding:7px 10px; font-size:.8rem; }
  .filter-bar input::placeholder{ color:#94a3b8; font-weight:600; }
  .chip{ background:#0f172a; border:1px solid var(--border); border-radius:8px;
         padding:6px 10px; font-size:.74rem; font-weight:800; color:#cbd5e1; }
  .clear-btn{ background:var(--accent2); border:none; color:#fff; font-weight:800;
              border-radius:8px; padding:7px 12px; cursor:pointer; font-size:.78rem; }
  .clear-btn:hover{ filter:brightness(1.1); }

  table{ width:100%; border-collapse:collapse; font-size:.78rem; }
  th{ background:#0f172a; color:#ffffff; padding:8px 10px; text-align:left;
      position:sticky; top:0; font-weight:800; border-bottom:2px solid var(--border);
      white-space:nowrap; }
  th.sortable{ cursor:pointer; }
  th.sortable:hover{ color:var(--accent); }
  td{ padding:7px 10px; border-bottom:1px solid #0f172a; color:#ffffff; font-weight:700; }
  tr:hover td{ background:#1e3a5f55; }
  .qcell{ max-width:280px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .muted{ color:#cbd5e1; }
  .center{ text-align:center; }
  .model{ color:var(--accent); font-size:.72rem; }
  .badge{ display:inline-block; padding:2px 9px; border-radius:9999px;
          font-size:.7rem; font-weight:800; }
  .badge-en{ background:#1d4ed8; color:#ffffff; }
  .badge-fr{ background:#7c3aed; color:#ffffff; }
  .badge-other{ background:#374151; color:#ffffff; }
  .score-high{ color:#4ade80; }
  .score-mid { color:#facc15; }
  .score-low { color:#f87171; }
  .ok{ color:#4ade80; font-weight:800; }
  .no{ color:#f87171; font-weight:800; }

  .table-wrap{ max-height:520px; overflow:auto; border-radius:10px; }
  .footer{ text-align:center; color:#cbd5e1; font-size:.75rem; padding:18px 0;
           font-weight:600; }
  ::-webkit-scrollbar{ width:8px; height:8px; }
  ::-webkit-scrollbar-track{ background:#0f172a; }
  ::-webkit-scrollbar-thumb{ background:var(--border); border-radius:4px; }

  /* progressive enhancement visibility:
     default = static visible / JS chrome hidden.
     <html class="js"> (set by inline script) reveals JS chrome (filters, gear note).
     Charts are gated separately on <html class="charts">, only added once
     Chart.js is confirmed loaded — so if the CDN is unreachable the static
     SVG/CSS bars stay visible instead of leaving blank boxes. */
  .js-only{ display:none; }
  html.js .js-only{ display:block; }
  /* canvases hidden until charts confirmed available */
  canvas.js-only{ display:none; }
  html.charts canvas.js-only{ display:block; }
  /* static bars stay until charts confirmed available */
  html.charts .chart-static{ display:none; }
</style>
<script>
  /* Set as early as possible so the static fallback never flashes when JS is on. */
  document.documentElement.className += " js";
</script>
</head>
<body>

<script id="dash-data" type="application/json">__DATA_PLACEHOLDER__</script>

<!-- ── Top banner + Normalize control ─────────────────────────────────────── -->
<div class="topbar">
  <div>
    <h1>&#127891; LUFA RAG Evaluation Dashboard</h1>
    <div class="sub">Agentic RAG for Cross-Lingual Retrieval of University Collective Agreements
      &nbsp;&middot;&nbsp; Generated __GEN_AT__</div>
  </div>

  <!-- No-JS: a real checkbox (req. 3). JS: a styled gear toggle. -->
  <div class="norm-wrap">
    <span class="gear">&#9881;&#65039;</span>
    <input class="norm-cb" type="checkbox" id="normalize-cb"/>
    <label for="normalize-cb">Normalize Data</label>
    <span class="norm-note js-only">Excludes rows with no generated answer
      (and no top-1 retrieval). Recomputes all stats.</span>
    <noscript><span class="norm-note" style="display:inline">
      Enable JavaScript to normalize interactively.</span></noscript>
  </div>
</div>

<!-- ── KPI cards (static + live share the same container) ─────────────────── -->
<div class="kpi-grid" id="kpi-row">__STATIC_KPIS__</div>

<!-- ── Charts ─────────────────────────────────────────────────────────────── -->
<div class="grid2">
  <div class="card">
    <div class="section-title">Generation Metrics by Model</div>
    <div class="chart-static">__STATIC_GEN__</div>
    <canvas class="js-only" id="genChart"></canvas>
  </div>
  <div class="card">
    <div class="section-title">Retrieval Metrics by Model</div>
    <div class="chart-static">__STATIC_RET__</div>
    <canvas class="js-only" id="retChart"></canvas>
  </div>
</div>

<div class="grid2">
  <div class="card">
    <div class="section-title">LLM-as-Judge Metrics (Radar)</div>
    <div class="chart-static">__STATIC_JUDGE__</div>
    <canvas class="js-only" id="radarChart"></canvas>
  </div>
  <div class="card">
    <div class="section-title">Performance by Language</div>
    <div class="chart-static">__STATIC_LANG__</div>
    <canvas class="js-only" id="langChart"></canvas>
  </div>
</div>

<div class="grid2">
  <div class="card">
    <div class="section-title">F1 Score by Difficulty</div>
    <div class="chart-static">__STATIC_DIFF__</div>
    <canvas class="js-only" id="diffChart"></canvas>
  </div>
  <div class="card">
    <div class="section-title">ROUGE-L by Category</div>
    <div class="chart-static">__STATIC_CAT__</div>
    <canvas class="js-only" id="catChart"></canvas>
  </div>
</div>

<!-- ── Detailed results ───────────────────────────────────────────────────── -->
<div class="card" style="margin-bottom:26px;">
  <div class="section-title">Detailed Results</div>

  <!-- Filter UI: only meaningful with JS -->
  <div class="filter-bar js-only" id="filter-bar"></div>
  <div class="js-only" style="margin-bottom:10px;">
    <span class="chip" id="row-count"></span>
    <button class="clear-btn" id="clear-filters">Clear filters</button>
  </div>

  <div class="table-wrap">
    <table id="results-table">
      <thead>
        <tr id="results-head">
          <th>#</th>
          <th class="sortable" data-col="question">Question</th>
          <th class="sortable" data-col="rag_base_model">Model</th>
          <th class="sortable" data-col="language">Lang</th>
          <th class="sortable" data-col="category">Category</th>
          <th class="sortable" data-col="difficulty">Difficulty</th>
          <th class="sortable" data-col="token_f1_score">F1</th>
          <th class="sortable" data-col="sentence_bleu_score">BLEU</th>
          <th class="sortable" data-col="rougeL">ROUGE-L</th>
          <th class="sortable" data-col="meteor">METEOR</th>
          <th class="sortable" data-col="mrr">MRR</th>
          <th class="sortable" data-col="recall_1">Recall@1</th>
          <th class="sortable" data-col="recall_3">Recall@3</th>
          <th class="sortable" data-col="recall_5">Recall@5</th>
          <th class="sortable" data-col="answer_relevance">Relevance</th>
          <th class="sortable" data-col="faithfulness">Faithful</th>
          <th class="sortable" data-col="context_precision">Precision</th>
          <th class="sortable" data-col="grounded">Grounded</th>
          <th class="sortable" data-col="attempts">Attempts</th>
        </tr>
      </thead>
      <!-- Static rows for no-JS; JS replaces tbody contents on first render. -->
      <tbody id="results-tbody">__STATIC_TABLE__</tbody>
    </table>
  </div>
</div>

<div class="footer">
  LUFA Agentic RAG Thesis &nbsp;&middot;&nbsp; Laurentian University
  &nbsp;&middot;&nbsp; Computational Sciences &nbsp;&middot;&nbsp; Generated __GEN_AT__
</div>

<!-- ───────────────────────────── INTERACTIVE LAYER ───────────────────────── -->
<script>
(function(){
  "use strict";
  var D;
  try { D = JSON.parse(document.getElementById('dash-data').textContent); }
  catch(e){ console.error('Dashboard data parse failed', e); return; }

  var META = D.metric_meta || {};
  var NUMERIC = META.numeric || [];
  var GEN = META.gen || [], RET = META.ret || [], JUDGE = META.judge || [];
  var KPIS = META.kpis || [];
  var ROW_COLS = META.row_columns || [];
  var ALL_ROWS = D.rows || [];

  var COLORS = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899','#06b6d4'];
  var LIGHT  = ['#93c5fd','#6ee7b7','#fcd34d','#fca5a5','#c4b5fd','#f9a8d4','#67e8f9'];

  function num(v){ var n=parseFloat(v); return isNaN(n)?0:n; }
  function fmt(v){ var n=parseFloat(v); return isNaN(n)?(v||''):n.toFixed(3); }
  function scoreClass(v){ v=num(v); return v>=0.7?'score-high':v>=0.4?'score-mid':'score-low'; }
  function esc(s){ s=(s==null?'':String(s));
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
  function isGrounded(v){ return ['true','1','1.0','yes'].indexOf(String(v).trim().toLowerCase())>=0; }
  function mean(arr){ if(!arr.length) return 0; var s=0; for(var i=0;i<arr.length;i++) s+=arr[i]; return s/arr.length; }

  // ── Filter state ──────────────────────────────────────────────────────────
  // Per-column text/select filters + the normalize toggle.  Both compose.
  var colFilters = {};                 // {col: "substring"}
  var normalize  = false;
  var sortCol = null, sortDir = 1;

  // Columns offered as filters (every detail column the user can see, req. 2/5).
  var FILTERABLE = ROW_COLS.filter(function(c){ return c!=='answer'; });
  // Choose dropdowns for low-cardinality categorical columns; text for the rest.
  var CATEGORICAL = ['rag_base_model','language','category','difficulty','grounded'];

  function distinctValues(col){
    var seen = {}, out = [];
    for(var i=0;i<ALL_ROWS.length;i++){
      var v = ALL_ROWS[i][col];
      if(col==='grounded') v = isGrounded(v) ? 'true' : 'false';
      v = (v==null?'':String(v)).trim();
      if(v!=='' && !seen[v]){ seen[v]=1; out.push(v); }
    }
    out.sort();
    return out;
  }

  function passesFilters(r){
    if(normalize){
      if(!r._has_answer) return false;      // drop unanswered questions
      // also drop rows with no top-1 retrieval (req. 3)
      if(r._has_retrieval === false) return false;
    }
    for(var col in colFilters){
      var want = colFilters[col];
      if(want==null || want==='') continue;
      var have = r[col];
      if(col==='grounded') have = isGrounded(have) ? 'true' : 'false';
      have = (have==null?'':String(have)).toLowerCase();
      if(CATEGORICAL.indexOf(col)>=0){
        if(have !== String(want).toLowerCase()) return false;   // exact match
      }else{
        if(have.indexOf(String(want).toLowerCase()) < 0) return false; // substring
      }
    }
    return true;
  }

  function currentRows(){
    var out = ALL_ROWS.filter(passesFilters);
    if(sortCol){
      out = out.slice().sort(function(a,b){
        var x=a[sortCol], y=b[sortCol];
        var nx=parseFloat(x), ny=parseFloat(y);
        if(!isNaN(nx) && !isNaN(ny)){ return (nx-ny)*sortDir; }
        x=(x==null?'':String(x)).toLowerCase(); y=(y==null?'':String(y)).toLowerCase();
        return (x<y?-1:x>y?1:0)*sortDir;
      });
    }
    return out;
  }

  // ── Aggregation (recomputed from filtered rows) ─────────────────────────────
  function overall(rows){
    var o = {};
    NUMERIC.forEach(function(m){ o[m] = rows.length ? mean(rows.map(function(r){return num(r[m]);})) : 0; });
    return o;
  }
  function groundedRate(rows){
    if(!rows.length) return 0;
    return mean(rows.map(function(r){ return isGrounded(r.grounded)?1:0; }));
  }
  function avgAttempts(rows){
    if(!rows.length) return 0;
    return mean(rows.map(function(r){ return num(r.attempts); }));
  }
  function avgByGroup(rows, groupCol, metric){
    var buckets = {};
    rows.forEach(function(r){
      var k = (r[groupCol]==null?'':String(r[groupCol])).trim();
      if(k==='' || k.toLowerCase()==='nan') return;
      (buckets[k] = buckets[k] || []).push(num(r[metric]));
    });
    var out = {};
    Object.keys(buckets).forEach(function(k){ out[k] = mean(buckets[k]); });
    return out;
  }

  // ── KPI render ──────────────────────────────────────────────────────────────
  function renderKPIs(rows){
    var ov = overall(rows);
    var html = '';
    KPIS.forEach(function(def){
      var key=def[0], label=def[1], kind=def[2], value;
      if(kind==='count') value = rows.length;
      else if(kind==='pct') value = (groundedRate(rows)*100).toFixed(1)+'%';
      else if(kind==='num') value = avgAttempts(rows).toFixed(2);
      else value = (ov[key]||0).toFixed(3);
      html += '<div class="metric-card"><div class="kpi-value">'+value+
              '</div><div class="kpi-label">'+esc(label)+'</div></div>';
    });
    document.getElementById('kpi-row').innerHTML = html;
  }

  // ── Charts ──────────────────────────────────────────────────────────────────
  var charts = {};
  function destroy(id){ if(charts[id]){ charts[id].destroy(); delete charts[id]; } }

  function barOpts(){
    return { responsive:true, maintainAspectRatio:true,
      plugins:{ legend:{ labels:{ color:'#ffffff', font:{ size:11, weight:'bold' } } } },
      scales:{
        x:{ ticks:{ color:'#e2e8f0', font:{weight:'bold'} }, grid:{ color:'#1e293b' } },
        y:{ ticks:{ color:'#e2e8f0', font:{weight:'bold'} }, grid:{ color:'#334155' },
            beginAtZero:true, max:1 } } };
  }

  function renderCharts(rows){
    var models = D.models && D.models.length ? D.models : ['default'];

    // Generation by model
    (function(){
      var keys=['token_f1_score','sentence_bleu_score','rouge1','rouge2','rougeL','meteor'];
      var labels=['F1','BLEU','ROUGE-1','ROUGE-2','ROUGE-L','METEOR'];
      var byModel={}; models.forEach(function(m){ byModel[m]={}; });
      keys.forEach(function(k){
        var g=avgByGroup(rows,'rag_base_model',k);
        models.forEach(function(m){ byModel[m][k]= g[m]||0; });
      });
      var ds=models.map(function(m,i){ return { label:m,
        data:keys.map(function(k){return byModel[m][k]||0;}),
        backgroundColor:COLORS[i%COLORS.length]+'99', borderColor:COLORS[i%COLORS.length], borderWidth:1 };});
      destroy('genChart');
      charts.genChart=new Chart(document.getElementById('genChart'),{type:'bar',data:{labels:labels,datasets:ds},options:barOpts()});
    })();

    // Retrieval by model
    (function(){
      var keys=['mrr','ndcg_at_k','recall_1','recall_3','recall_5'];
      var labels=['MRR','NDCG','Recall@1','Recall@3','Recall@5'];
      var byModel={}; models.forEach(function(m){ byModel[m]={}; });
      keys.forEach(function(k){
        var g=avgByGroup(rows,'rag_base_model',k);
        models.forEach(function(m){ byModel[m][k]= g[m]||0; });
      });
      var ds=models.map(function(m,i){ return { label:m,
        data:keys.map(function(k){return byModel[m][k]||0;}),
        backgroundColor:LIGHT[i%LIGHT.length]+'88', borderColor:LIGHT[i%LIGHT.length], borderWidth:1 };});
      destroy('retChart');
      charts.retChart=new Chart(document.getElementById('retChart'),{type:'bar',data:{labels:labels,datasets:ds},options:barOpts()});
    })();

    // Radar judge
    (function(){
      var keys=['answer_relevance','faithfulness','context_precision'];
      var labels=['Answer Relevance','Faithfulness','Context Precision'];
      var ds=models.map(function(m,i){
        var g={}; keys.forEach(function(k){ g[k]=avgByGroup(rows,'rag_base_model',k)[m]||0; });
        return { label:m, data:keys.map(function(k){return g[k];}),
          borderColor:COLORS[i%COLORS.length], backgroundColor:COLORS[i%COLORS.length]+'33',
          pointBackgroundColor:COLORS[i%COLORS.length], borderWidth:2 };
      });
      destroy('radarChart');
      charts.radarChart=new Chart(document.getElementById('radarChart'),{type:'radar',
        data:{labels:labels,datasets:ds},
        options:{ responsive:true, maintainAspectRatio:true,
          scales:{ r:{ min:0, max:1, ticks:{ stepSize:.2, color:'#cbd5e1', backdropColor:'transparent' },
            grid:{ color:'#334155' }, pointLabels:{ color:'#ffffff', font:{ size:11, weight:'bold' } },
            angleLines:{ color:'#334155' } } },
          plugins:{ legend:{ labels:{ color:'#ffffff', font:{weight:'bold'} } } } } });
    })();

    // Language: F1 / ROUGE-L / METEOR
    (function(){
      var f1=avgByGroup(rows,'language','token_f1_score');
      var rl=avgByGroup(rows,'language','rougeL');
      var mt=avgByGroup(rows,'language','meteor');
      var langs=Object.keys(f1);
      var ds=[
        {label:'F1',data:langs.map(function(l){return f1[l]||0;}),backgroundColor:COLORS[0]+'99',borderColor:COLORS[0],borderWidth:1},
        {label:'ROUGE-L',data:langs.map(function(l){return rl[l]||0;}),backgroundColor:COLORS[1]+'99',borderColor:COLORS[1],borderWidth:1},
        {label:'METEOR',data:langs.map(function(l){return mt[l]||0;}),backgroundColor:COLORS[2]+'99',borderColor:COLORS[2],borderWidth:1}
      ];
      destroy('langChart');
      charts.langChart=new Chart(document.getElementById('langChart'),{type:'bar',data:{labels:langs,datasets:ds},options:barOpts()});
    })();

    // Difficulty F1
    (function(){
      var g=avgByGroup(rows,'difficulty','token_f1_score');
      var diffs=Object.keys(g);
      destroy('diffChart');
      charts.diffChart=new Chart(document.getElementById('diffChart'),{type:'bar',
        data:{labels:diffs,datasets:[{label:'F1 by Difficulty',
          data:diffs.map(function(d){return g[d]||0;}),
          backgroundColor:diffs.map(function(_,i){return COLORS[i%COLORS.length]+'bb';}),
          borderColor:diffs.map(function(_,i){return COLORS[i%COLORS.length];}), borderWidth:1}]},
        options:barOpts()});
    })();

    // Category ROUGE-L
    (function(){
      var g=avgByGroup(rows,'category','rougeL');
      var cats=Object.keys(g);
      var labels=cats.map(function(c){return c.length>18?c.slice(0,16)+'\u2026':c;});
      destroy('catChart');
      charts.catChart=new Chart(document.getElementById('catChart'),{type:'bar',
        data:{labels:labels,datasets:[{label:'ROUGE-L by Category',
          data:cats.map(function(c){return g[c]||0;}),
          backgroundColor:cats.map(function(_,i){return LIGHT[i%LIGHT.length]+'99';}),
          borderColor:cats.map(function(_,i){return LIGHT[i%LIGHT.length];}), borderWidth:1}]},
        options:barOpts()});
    })();
  }

  // ── Table ─────────────────────────────────────────────────────────────────
  function renderTable(rows){
    var tbody=document.getElementById('results-tbody');
    var html=[];
    for(var i=0;i<rows.length;i++){
      var r=rows[i];
      var lang=String(r.language||'');
      var badge = lang.toLowerCase().indexOf('en')===0?'badge-en'
                : lang.toLowerCase().indexOf('fr')===0?'badge-fr':'badge-other';
      var grounded=isGrounded(r.grounded);
      html.push('<tr>'+
        '<td class="muted">'+(i+1)+'</td>'+
        '<td class="qcell" title="'+esc(r.question)+'">'+esc((r.question||'').slice(0,90))+'</td>'+
        '<td class="model">'+esc(r.rag_base_model)+'</td>'+
        '<td><span class="badge '+badge+'">'+esc(lang)+'</span></td>'+
        '<td>'+esc(r.category)+'</td>'+
        '<td>'+esc(r.difficulty)+'</td>'+
        '<td class="'+scoreClass(r.token_f1_score)+'">'+fmt(r.token_f1_score)+'</td>'+
        '<td class="'+scoreClass(r.sentence_bleu_score)+'">'+fmt(r.sentence_bleu_score)+'</td>'+
        '<td class="'+scoreClass(r.rougeL)+'">'+fmt(r.rougeL)+'</td>'+
        '<td class="'+scoreClass(r.meteor)+'">'+fmt(r.meteor)+'</td>'+
        '<td class="'+scoreClass(r.mrr)+'">'+fmt(r.mrr)+'</td>'+
        '<td class="'+scoreClass(r.recall_1)+'">'+fmt(r.recall_1)+'</td>'+
        '<td class="'+scoreClass(r.recall_3)+'">'+fmt(r.recall_3)+'</td>'+
        '<td class="'+scoreClass(r.recall_5)+'">'+fmt(r.recall_5)+'</td>'+
        '<td class="'+scoreClass(r.answer_relevance)+'">'+fmt(r.answer_relevance)+'</td>'+
        '<td class="'+scoreClass(r.faithfulness)+'">'+fmt(r.faithfulness)+'</td>'+
        '<td class="'+scoreClass(r.context_precision)+'">'+fmt(r.context_precision)+'</td>'+
        '<td class="center">'+(grounded?'<span class="ok">\u2713</span>':'<span class="no">\u2717</span>')+'</td>'+
        '<td class="center muted">'+esc(r.attempts)+'</td>'+
      '</tr>');
    }
    tbody.innerHTML = html.join('') ||
      '<tr><td colspan="19" class="center muted" style="padding:24px;">No rows match the current filters.</td></tr>';
    document.getElementById('row-count').textContent =
      rows.length + ' / ' + ALL_ROWS.length + ' rows' + (normalize?' (normalized)':'');
  }

  // ── Filter UI build ─────────────────────────────────────────────────────────
  function labelFor(col){
    var map={ rag_base_model:'Model', token_f1_score:'F1', sentence_bleu_score:'BLEU',
      rougeL:'ROUGE-L', rouge1:'ROUGE-1', rouge2:'ROUGE-2', meteor:'METEOR', mrr:'MRR',
      ndcg_at_k:'NDCG', recall_1:'Recall@1', recall_3:'Recall@3', recall_5:'Recall@5',
      answer_relevance:'Relevance', faithfulness:'Faithful', context_precision:'Precision',
      question_id:'ID', question:'Question', language:'Language', category:'Category',
      difficulty:'Difficulty', grounded:'Grounded', attempts:'Attempts' };
    return map[col]||col;
  }

  function buildFilterBar(){
    var bar=document.getElementById('filter-bar');
    bar.innerHTML='';
    FILTERABLE.forEach(function(col){
      if(CATEGORICAL.indexOf(col)>=0){
        var sel=document.createElement('select');
        sel.setAttribute('data-col',col);
        var opt0=document.createElement('option'); opt0.value=''; opt0.textContent=labelFor(col)+': all';
        sel.appendChild(opt0);
        distinctValues(col).forEach(function(v){
          var o=document.createElement('option'); o.value=v; o.textContent=v; sel.appendChild(o);
        });
        sel.addEventListener('change',function(){ colFilters[col]=this.value; render(); });
        bar.appendChild(sel);
      }else{
        var inp=document.createElement('input');
        inp.setAttribute('data-col',col);
        inp.placeholder=labelFor(col)+'\u2026';
        inp.addEventListener('input',function(){ colFilters[col]=this.value; render(); });
        bar.appendChild(inp);
      }
    });
  }

  function bindHead(){
    var ths=document.querySelectorAll('#results-head th.sortable');
    ths.forEach(function(th){
      th.addEventListener('click',function(){
        var col=th.getAttribute('data-col');
        if(sortCol===col){ sortDir=-sortDir; } else { sortCol=col; sortDir=1; }
        render();
      });
    });
  }

  // Chart.js may fail to load (e.g. offline / blocked CDN). Detect it once so
  // we can keep the static SVG/CSS bars visible instead of blank canvases.
  var CHARTS_OK = (typeof Chart !== 'undefined');
  if(CHARTS_OK){ document.documentElement.className += ' charts'; }

  // ── Master render ───────────────────────────────────────────────────────────
  function render(){
    var rows=currentRows();
    renderKPIs(rows);
    if(CHARTS_OK){ try { renderCharts(rows); } catch(e){ console.error('chart render', e); } }
    renderTable(rows);
  }

  // ── Wire up controls ─────────────────────────────────────────────────────────
  var cb=document.getElementById('normalize-cb');
  if(cb){ cb.addEventListener('change',function(){ normalize=this.checked; render(); }); }
  var clearBtn=document.getElementById('clear-filters');
  if(clearBtn){ clearBtn.addEventListener('click',function(){
    colFilters={}; sortCol=null; sortDir=1;
    document.querySelectorAll('#filter-bar input').forEach(function(i){ i.value=''; });
    document.querySelectorAll('#filter-bar select').forEach(function(s){ s.value=''; });
    render();
  }); }

  buildFilterBar();
  bindHead();
  render();   // first JS render replaces all static content with live content
})();
</script>
</body>
</html>"""
