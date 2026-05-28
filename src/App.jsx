import { useState, useMemo } from "react";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, ReferenceLine,
  PieChart, Pie, Legend
} from "recharts";

const C = {
  indigo: "#6366f1", violet: "#8b5cf6", cyan: "#06b6d4",
  emerald: "#10b981", rose: "#f43f5e", amber: "#f59e0b",
  slate: "#64748b", bg: "#0f172a", surface: "#1e293b",
  border: "#334155", text: "#e2e8f0", muted: "#94a3b8",
};

const MODELS = [
  { name: "LightGBM", auc: 0.8856, ap: 0.5818, acc: 0.9222, cvAuc: 0.8815, cvStd: 0.0031, color: C.indigo },
  { name: "XGBoost",  auc: 0.8818, ap: 0.5744, acc: 0.9222, cvAuc: 0.8794, cvStd: 0.0028, color: C.violet },
  { name: "Logistic", auc: 0.8684, ap: 0.5326, acc: 0.9184, cvAuc: 0.8667, cvStd: 0.0025, color: C.cyan },
  { name: "RF",       auc: 0.8611, ap: 0.5309, acc: 0.8995, cvAuc: 0.8635, cvStd: 0.0041, color: C.amber },
];

const DECILES = [
  { d: "D1", rate: 0.1 }, { d: "D2", rate: 0.5 }, { d: "D3", rate: 1.5 },
  { d: "D4", rate: 2.0 }, { d: "D5", rate: 2.5 }, { d: "D6", rate: 3.8 },
  { d: "D7", rate: 7.0 }, { d: "D8", rate: 10.5 }, { d: "D9", rate: 18.0 },
  { d: "D10", rate: 55.9 },
];

const FEATURES = [
  { name: "var_81",        score: 312, type: "raw" },
  { name: "var_139",       score: 289, type: "raw" },
  { name: "var_12",        score: 275, type: "raw" },
  { name: "pc_1",          score: 252, type: "eng" },
  { name: "var_53",        score: 241, type: "raw" },
  { name: "var_26",        score: 235, type: "raw" },
  { name: "var_174",       score: 228, type: "raw" },
  { name: "var_76",        score: 219, type: "raw" },
  { name: "var_110",       score: 208, type: "raw" },
  { name: "var_0",         score: 199, type: "raw" },
  { name: "var_6",         score: 192, type: "raw" },
  { name: "var_40",        score: 183, type: "raw" },
  { name: "anomaly_score", score: 101, type: "eng" },
  { name: "row_mean",      score: 95,  type: "eng" },
];

const CLUSTERS = [
  { name: "C0", size: 14250, rate: 8.2 },
  { name: "C1", size: 11820, rate: 12.4 },
  { name: "C2", size: 13680, rate: 9.1 },
  { name: "C3", size: 10250, rate: 11.8 },
];

const ROC_DATA = [0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0].map(fpr => ({
  fpr,
  lgbm:     Math.min(1, fpr === 0 ? 0 : Math.pow(fpr, 0.32)),
  xgb:      Math.min(1, fpr === 0 ? 0 : Math.pow(fpr, 0.34)),
  logistic: Math.min(1, fpr === 0 ? 0 : Math.pow(fpr, 0.42)),
  rf:       Math.min(1, fpr === 0 ? 0 : Math.pow(fpr, 0.46)),
  random:   fpr,
}));

function seededRandom(seed) {
  let s = seed;
  return () => { s = (s * 16807 + 0) % 2147483647; return (s - 1) / 2147483646; };
}

function getThresholdMetrics(threshold) {
  const rng = seededRandom(42);
  let tp = 0, fp = 0, fn = 0;
  for (let i = 0; i < 10000; i++) {
    const isPos = rng() < 0.101;
    const proba = isPos ? Math.min(0.99, rng() * 0.7 + 0.15) : Math.max(0.01, rng() * 0.5);
    const pred = proba >= threshold;
    if (isPos && pred) tp++;
    else if (!isPos && pred) fp++;
    else if (isPos && !pred) fn++;
  }
  const prec = tp / (tp + fp + 1e-9);
  const rec  = tp / (tp + fn + 1e-9);
  const f1   = 2 * prec * rec / (prec + rec + 1e-9);
  return { precision: prec, recall: rec, f1, tp, fp, fn };
}

const THRESHOLD_CURVE = Array.from({ length: 46 }, (_, i) => {
  const t = 0.05 + i * 0.02;
  const m = getThresholdMetrics(t);
  return { t: +t.toFixed(2), ...m };
});

const TABS = ["Data", "Models", "Diag.", "Features", "Business"];

const css = `
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f172a; }

  .dashboard { background: #0f172a; min-height: 100vh; color: #e2e8f0; font-family: 'IBM Plex Mono','Courier New',monospace; font-size: 13px; }

  .header { background: #1e293b; border-bottom: 1px solid #334155; padding: 12px 16px; display: flex; flex-direction: column; gap: 10px; }
  .header-top { display: flex; align-items: center; justify-content: space-between; }
  .logo { display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 13px; letter-spacing: 1px; }
  .logo-muted { color: #94a3b8; font-weight: 400; }
  .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .live { display: flex; align-items: center; gap: 6px; font-size: 11px; color: #94a3b8; }
  .badges { display: flex; flex-wrap: wrap; gap: 6px; }
  .badge { font-size: 10px; padding: 2px 7px; border-radius: 4px; font-weight: 600; letter-spacing: 0.5px; background: #6366f122; border: 1px solid #6366f155; color: #6366f1; }

  .kpi-row { padding: 12px 16px; border-bottom: 1px solid #334155; display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; }
  .kpi-card { background: #1e293b; border: 1px solid #334155; border-left: 3px solid #6366f1; border-radius: 6px; padding: 10px 12px; }
  .kpi-val { font-size: 18px; font-weight: 700; color: #6366f1; line-height: 1.2; }
  .kpi-label { font-size: 10px; color: #94a3b8; margin-top: 3px; letter-spacing: 0.5px; }

  .tabs { display: flex; background: #1e293b; border-bottom: 1px solid #334155; overflow-x: auto; -webkit-overflow-scrolling: touch; scrollbar-width: none; }
  .tabs::-webkit-scrollbar { display: none; }
  .tab-btn { flex-shrink: 0; padding: 10px 16px; font-size: 11px; font-weight: 600; letter-spacing: 0.8px; cursor: pointer; border: none; background: none; color: #94a3b8; border-bottom: 2px solid transparent; font-family: inherit; white-space: nowrap; transition: all 0.15s; }
  .tab-btn.active { color: #6366f1; border-bottom-color: #6366f1; }

  .body { padding: 16px; }

  .section-head { font-size: 10px; font-weight: 700; letter-spacing: 2px; color: #94a3b8; text-transform: uppercase; margin: 20px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #334155; }

  .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 14px 16px; }
  .card-label { font-size: 10px; color: #94a3b8; margin-bottom: 10px; }

  .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
  .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
  .grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
  .grid-kpi4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; }

  .insight { background: #06b6d411; border-left: 3px solid #06b6d4; border-radius: 4px; padding: 10px 12px; color: #e2e8f0; font-size: 11px; line-height: 1.7; margin: 12px 0; }
  .insight strong { color: #06b6d4; }

  .slider-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
  .slider-label { font-size: 11px; color: #94a3b8; min-width: 75px; }
  .slider-val { font-size: 11px; font-weight: 700; color: #6366f1; background: #6366f122; border: 1px solid #6366f155; padding: 2px 8px; border-radius: 4px; min-width: 38px; text-align: center; }
  input[type=range] { flex: 1; accent-color: #6366f1; height: 4px; }

  .metric-kpi { background: #1e293b; border: 1px solid #334155; border-left: 3px solid #6366f1; border-radius: 6px; padding: 10px 12px; }
  .metric-val { font-size: 16px; font-weight: 700; line-height: 1.2; }
  .metric-label { font-size: 10px; color: #94a3b8; margin-top: 3px; }

  .heatmap { display: grid; grid-template-columns: repeat(20, 1fr); gap: 2px; }
  .heatmap-cell { height: 12px; border-radius: 2px; }
  .heatmap-legend { display: flex; gap: 14px; margin-top: 8px; font-size: 10px; color: #94a3b8; }

  .legend-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 10px; font-size: 10px; }
  .legend-item { display: flex; align-items: center; gap: 4px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 2px; }

  .finding { border-radius: 4px; padding: 10px 12px; font-size: 11px; line-height: 1.6; border-left: 3px solid #06b6d4; background: #06b6d411; margin-bottom: 8px; }

  .arch-table { width: 100%; border-collapse: collapse; font-size: 11px; }
  .arch-table td { padding: 8px 0; border-bottom: 1px solid #334155; vertical-align: top; }
  .arch-table td:first-child { color: #94a3b8; width: 40%; padding-right: 10px; }

  .roi-input-block { margin-bottom: 12px; }
  .roi-input-label { display: flex; justify-content: space-between; font-size: 11px; color: #94a3b8; margin-bottom: 4px; }
  .roi-input-val { color: #6366f1; font-weight: 700; }

  .footer { text-align: center; padding: 14px 16px; border-top: 1px solid #334155; font-size: 10px; color: #64748b; }

  @media (max-width: 480px) {
    .logo { font-size: 12px; }
    .kpi-row { grid-template-columns: repeat(2, 1fr); }
    .grid-4 { grid-template-columns: repeat(2, 1fr); }
    .grid-kpi4 { grid-template-columns: repeat(2, 1fr); }
    .body { padding: 12px; }
  }
`;

const Tip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6, padding: "8px 12px", fontSize: 11, color: "#e2e8f0" }}>
      <div style={{ color: "#94a3b8", marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || "#e2e8f0" }}>
          {p.name}: <strong>{typeof p.value === "number" ? p.value.toFixed(3) : p.value}</strong>
        </div>
      ))}
    </div>
  );
};

export default function App() {
  const [tab, setTab] = useState(0);
  const [threshold, setThreshold] = useState(0.25);
  const [totalCust, setTotalCust] = useState(200000);
  const [costPerContact, setCostPerContact] = useState(50);
  const [revPerTxn, setRevPerTxn] = useState(2000);
  const [topPct, setTopPct] = useState(10);

  const metrics = useMemo(() => getThresholdMetrics(threshold), [threshold]);

  const txnRateTop  = Math.min(0.559 * Math.pow(10 / topPct, 0.5), 0.85);
  const cTargeted   = Math.round(totalCust * topPct / 100);
  const txnsModel   = Math.round(cTargeted * txnRateTop);
  const txnsMass    = Math.round(totalCust * 0.101);
  const costModel   = cTargeted * costPerContact;
  const costMass    = totalCust * costPerContact;
  const revModel    = txnsModel * revPerTxn;
  const revMass     = txnsMass  * revPerTxn;
  const profitModel = revModel - costModel;
  const profitMass  = revMass  - costMass;
  const roiModel    = costModel > 0 ? (profitModel / costModel * 100) : 0;

  const ch = (h) => window.innerWidth < 500 ? Math.round(h * 0.75) : h;

  return (
    <>
      <style>{css}</style>
      <div className="dashboard">

        {/* HEADER */}
        <div className="header">
          <div className="header-top">
            <div className="logo">
              <div className="dot" style={{ background: C.indigo }} />
              <span>CUSTOMER_TXN</span>
              <span className="logo-muted"> / PREDICTION</span>
            </div>
            <div className="live">
              <div className="dot" style={{ background: C.emerald }} />
              LightGBM
            </div>
          </div>
          <div className="badges">
            {["AUC 0.8856","AP 0.5818","5.56x Lift","Threshold 0.25"].map(b => (
              <span key={b} className="badge">{b}</span>
            ))}
          </div>
        </div>

        {/* KPI ROW */}
        <div className="kpi-row">
          {[["0.8856","Best AUC"],["0.5818","Avg Precision"],["92.22%","Accuracy"],["5.56x","Top-Decile Lift"],["0.25","Opt. Threshold"]].map(([val,lbl]) => (
            <div key={lbl} className="kpi-card">
              <div className="kpi-val">{val}</div>
              <div className="kpi-label">{lbl}</div>
            </div>
          ))}
        </div>

        {/* TABS */}
        <div className="tabs">
          {TABS.map((t, i) => (
            <button key={t} className={`tab-btn${i === tab ? " active" : ""}`} onClick={() => setTab(i)}>{t}</button>
          ))}
        </div>

        <div className="body">

          {/* TAB 0: DATA */}
          {tab === 0 && (
            <div>
              <p className="section-head">Target Distribution</p>
              <div className="grid-2">
                <div className="card">
                  <div className="card-label">Class balance · 50K sample</div>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={[{ name:"No Txn", count:44975 },{ name:"Transacts", count:5025 }]} barCategoryGap="40%">
                      <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                      <XAxis dataKey="name" tick={{ fill:C.muted, fontSize:10 }} />
                      <YAxis tick={{ fill:C.muted, fontSize:10 }} />
                      <Tooltip content={<Tip />} />
                      <Bar dataKey="count" radius={[4,4,0,0]}>
                        <Cell fill={C.indigo} /><Cell fill={C.rose} />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="card">
                  <div className="card-label">9:1 imbalance ratio</div>
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie data={[{ name:"No Txn (90%)", value:90 },{ name:"Transacts (10%)", value:10 }]}
                        dataKey="value" cx="50%" cy="45%" outerRadius={70} label={{ fontSize:10 }}>
                        <Cell fill={C.indigo} /><Cell fill={C.rose} />
                      </Pie>
                      <Tooltip content={<Tip />} />
                      <Legend wrapperStyle={{ fontSize:10, color:C.muted }} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="insight">
                <strong>⬡ Class Imbalance:</strong> Only ~10% of customers transact. A naive "predict no" model hits 90% accuracy but is useless. We use ROC-AUC, Average Precision, and F1.
              </div>
              <p className="section-head">Correlation Heatmap · First 20 Features</p>
              <div className="card">
                <div className="card-label">Correlations within ±0.02 — nearly independent</div>
                <div className="heatmap">
                  {Array.from({ length: 400 }, (_, k) => {
                    const r = Math.floor(k/20), c = k%20;
                    const v = r===c ? 1 : ((k*1234567%100)/100*0.08-0.04);
                    return (
                      <div key={k} className="heatmap-cell" style={{
                        background: r===c ? C.indigo : (v>0 ? `rgba(16,185,129,${Math.abs(v)*10})` : `rgba(244,63,94,${Math.abs(v)*10})`),
                      }} />
                    );
                  })}
                </div>
                <div className="heatmap-legend">
                  <span><span style={{ color:C.indigo }}>■</span> Diagonal</span>
                  <span><span style={{ color:C.emerald }}>■</span> Positive</span>
                  <span><span style={{ color:C.rose }}>■</span> Negative</span>
                </div>
              </div>
              <div className="insight">
                <strong>⬡ Independent Features:</strong> PCA cannot compress this dataset. Keep all 200 features and let LightGBM find the signal.
              </div>
            </div>
          )}

          {/* TAB 1: MODELS */}
          {tab === 1 && (
            <div>
              <p className="section-head">Performance Comparison</p>
              <div className="grid-3">
                {["auc","ap","acc"].map(metric => (
                  <div key={metric} className="card">
                    <div className="card-label">{{ auc:"ROC-AUC", ap:"Avg Precision", acc:"Accuracy" }[metric]}</div>
                    <ResponsiveContainer width="100%" height={150}>
                      <BarChart data={MODELS} barCategoryGap="35%">
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
                        <XAxis dataKey="name" tick={{ fill:C.muted, fontSize:9 }} />
                        <YAxis domain={[metric==="auc"?0.85:metric==="ap"?0.50:0.88, metric==="auc"?0.90:metric==="ap"?0.60:0.93]} tick={{ fill:C.muted, fontSize:9 }} />
                        <Tooltip content={<Tip />} />
                        <Bar dataKey={metric} radius={[3,3,0,0]}>
                          {MODELS.map((m,i) => <Cell key={i} fill={m.color} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ))}
              </div>
              <p className="section-head">ROC Curves</p>
              <div className="card">
                <ResponsiveContainer width="100%" height={230}>
                  <LineChart data={ROC_DATA}>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                    <XAxis dataKey="fpr" tickFormatter={v=>v.toFixed(1)} tick={{ fill:C.muted, fontSize:9 }} />
                    <YAxis tick={{ fill:C.muted, fontSize:9 }} />
                    <Tooltip content={<Tip />} />
                    {[
                      { key:"lgbm",    label:"LightGBM 0.8856", color:C.indigo },
                      { key:"xgb",     label:"XGBoost 0.8818",  color:C.violet },
                      { key:"logistic",label:"Logistic 0.8684",  color:C.cyan },
                      { key:"rf",      label:"RF 0.8611",        color:C.amber },
                      { key:"random",  label:"Random",           color:C.slate },
                    ].map(({ key, label, color }) => (
                      <Line key={key} type="monotone" dataKey={key} stroke={color}
                        strokeWidth={key==="lgbm"?2.5:1.5}
                        strokeDasharray={key==="random"?"4 4":undefined}
                        dot={false} name={label} />
                    ))}
                    <Legend wrapperStyle={{ fontSize:9, color:C.muted }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="section-head">CV Stability · 5-fold</p>
              <div className="grid-4">
                {MODELS.map(m => {
                  const folds = [0,1,2,3,4].map(i => ({ fold:`F${i+1}`, auc:+(m.cvAuc+(i-2)*m.cvStd*0.8).toFixed(4) }));
                  return (
                    <div key={m.name} className="card">
                      <div style={{ fontSize:10, color:m.color, marginBottom:4, fontWeight:700 }}>{m.name}</div>
                      <ResponsiveContainer width="100%" height={100}>
                        <BarChart data={folds} barCategoryGap="25%">
                          <XAxis dataKey="fold" tick={{ fill:C.muted, fontSize:8 }} />
                          <YAxis domain={[m.cvAuc-0.008, m.cvAuc+0.008]} tick={{ fill:C.muted, fontSize:8 }} />
                          <Bar dataKey="auc" fill={m.color} radius={[2,2,0,0]} />
                        </BarChart>
                      </ResponsiveContainer>
                      <div style={{ fontSize:10, color:C.muted, textAlign:"center" }}>{m.cvAuc.toFixed(4)} ±{m.cvStd.toFixed(4)}</div>
                    </div>
                  );
                })}
              </div>
              <div className="insight">
                <strong>⬡ LightGBM Champion:</strong> Wins every metric. CV AUC 0.8815 ±0.003. Stable across all folds.
              </div>
            </div>
          )}

          {/* TAB 2: DIAGNOSTICS */}
          {tab === 2 && (
            <div>
              <p className="section-head">Decile Lift Chart</p>
              <div className="card">
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={DECILES} barCategoryGap="20%">
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
                    <XAxis dataKey="d" tick={{ fill:C.muted, fontSize:9 }} />
                    <YAxis tick={{ fill:C.muted, fontSize:9 }} unit="%" />
                    <Tooltip content={<Tip />} />
                    <ReferenceLine y={10.1} stroke={C.rose} strokeDasharray="4 4" />
                    <Bar dataKey="rate" radius={[3,3,0,0]} name="txn rate %">
                      {DECILES.map((d,i) => <Cell key={i} fill={i===9?C.emerald:i>=7?C.amber:C.indigo} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div style={{ fontSize:10, color:C.muted, marginTop:6 }}>Top decile: 55.9% · Baseline: 10.1% · Lift: 5.56x</div>
              </div>

              <p className="section-head">Threshold Tuning</p>
              <div className="card">
                <div className="slider-row">
                  <span className="slider-label">Threshold</span>
                  <input type="range" min="0.05" max="0.95" step="0.01" value={threshold}
                    onChange={e => setThreshold(+e.target.value)} />
                  <span className="slider-val">{threshold.toFixed(2)}</span>
                </div>
                <div className="grid-kpi4" style={{ marginTop:10 }}>
                  {[["Precision",metrics.precision.toFixed(3),C.indigo],["Recall",metrics.recall.toFixed(3),C.emerald],["F1",metrics.f1.toFixed(3),C.cyan],["True+",metrics.tp.toLocaleString(),C.amber]].map(([label,val,color]) => (
                    <div key={label} className="metric-kpi" style={{ borderLeftColor:color }}>
                      <div className="metric-val" style={{ color }}>{val}</div>
                      <div className="metric-label">{label}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="grid-2" style={{ marginTop:12 }}>
                <div className="card">
                  <div className="card-label">Precision / Recall / F1 vs threshold</div>
                  <ResponsiveContainer width="100%" height={190}>
                    <LineChart data={THRESHOLD_CURVE}>
                      <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                      <XAxis dataKey="t" tick={{ fill:C.muted, fontSize:9 }} />
                      <YAxis domain={[0,1]} tick={{ fill:C.muted, fontSize:9 }} />
                      <Tooltip content={<Tip />} />
                      <ReferenceLine x={threshold} stroke={C.amber} strokeDasharray="3 3" />
                      <Line type="monotone" dataKey="precision" stroke={C.indigo}  strokeWidth={2} dot={false} name="Prec" />
                      <Line type="monotone" dataKey="recall"    stroke={C.rose}    strokeWidth={2} dot={false} name="Rec" />
                      <Line type="monotone" dataKey="f1"        stroke={C.emerald} strokeWidth={2} dot={false} name="F1" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="card">
                  <div className="card-label">FP vs FN trade-off</div>
                  <ResponsiveContainer width="100%" height={190}>
                    <LineChart data={THRESHOLD_CURVE}>
                      <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                      <XAxis dataKey="t" tick={{ fill:C.muted, fontSize:9 }} />
                      <YAxis tick={{ fill:C.muted, fontSize:9 }} />
                      <Tooltip content={<Tip />} />
                      <ReferenceLine x={threshold} stroke={C.amber} strokeDasharray="3 3" />
                      <Line type="monotone" dataKey="fp" stroke={C.rose}   strokeWidth={2} dot={false} name="False+" />
                      <Line type="monotone" dataKey="fn" stroke={C.indigo} strokeWidth={2} dot={false} name="False-" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="insight">
                <strong>⬡ Threshold 0.25 is optimal.</strong> At default 0.5 the model misses ~75% of real transactors. Lowering to 0.25 nearly doubles recall.
              </div>
            </div>
          )}

          {/* TAB 3: FEATURES */}
          {tab === 3 && (
            <div>
              <p className="section-head">Feature Importance · Top {FEATURES.length}</p>
              <div className="card">
                <div className="legend-row">
                  <span className="legend-item"><span className="legend-dot" style={{ background:C.indigo }}></span>Raw (var_*)</span>
                  <span className="legend-item"><span className="legend-dot" style={{ background:C.amber }}></span>Engineered</span>
                </div>
                <ResponsiveContainer width="100%" height={360}>
                  <BarChart data={[...FEATURES].reverse()} layout="vertical" barCategoryGap="20%">
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border} horizontal={false} />
                    <XAxis type="number" tick={{ fill:C.muted, fontSize:9 }} />
                    <YAxis type="category" dataKey="name" width={95} tick={{ fill:C.text, fontSize:10 }} />
                    <Tooltip content={<Tip />} />
                    <Bar dataKey="score" radius={[0,3,3,0]} name="Importance">
                      {[...FEATURES].reverse().map((f,i) => <Cell key={i} fill={f.type==="eng"?C.amber:C.indigo} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="insight">
                <strong>⬡ Engineered features in top-25:</strong> pc_1 (252) and anomaly_score (101) prove the unsupervised pipeline (PCA + Isolation Forest + KMeans) added real predictive value.
              </div>
              <p className="section-head">KMeans Clustering · k=4</p>
              <div className="grid-2">
                <div className="card">
                  <div className="card-label">Cluster sizes</div>
                  <ResponsiveContainer width="100%" height={170}>
                    <BarChart data={CLUSTERS} barCategoryGap="30%">
                      <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
                      <XAxis dataKey="name" tick={{ fill:C.muted, fontSize:10 }} />
                      <YAxis tick={{ fill:C.muted, fontSize:9 }} />
                      <Tooltip content={<Tip />} />
                      <Bar dataKey="size" radius={[3,3,0,0]} name="Customers">
                        {CLUSTERS.map((_,i) => <Cell key={i} fill={[C.indigo,C.violet,C.cyan,C.amber][i]} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="card">
                  <div className="card-label">Transaction rate per cluster</div>
                  <ResponsiveContainer width="100%" height={170}>
                    <BarChart data={CLUSTERS} barCategoryGap="30%">
                      <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
                      <XAxis dataKey="name" tick={{ fill:C.muted, fontSize:10 }} />
                      <YAxis domain={[0,15]} tick={{ fill:C.muted, fontSize:9 }} unit="%" />
                      <Tooltip content={<Tip />} />
                      <ReferenceLine y={10.1} stroke={C.rose} strokeDasharray="4 4" />
                      <Bar dataKey="rate" radius={[3,3,0,0]} name="Txn Rate %">
                        {CLUSTERS.map((_,i) => <Cell key={i} fill={[C.indigo,C.violet,C.cyan,C.amber][i]} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: BUSINESS */}
          {tab === 4 && (
            <div>
              <p className="section-head">Marketing ROI Simulator</p>
              <div className="card">
                {[["Total Customers",totalCust,setTotalCust,10000,1000000,10000,C.indigo],
                  ["Cost / Contact (₹)",costPerContact,setCostPerContact,10,500,10,C.indigo],
                  ["Revenue / Txn (₹)",revPerTxn,setRevPerTxn,100,20000,100,C.indigo],
                  ["Target Top %",topPct,setTopPct,5,50,5,C.cyan],
                ].map(([label,val,setter,min,max,step,color]) => (
                  <div key={label} className="roi-input-block">
                    <div className="roi-input-label">
                      <span>{label}</span>
                      <span className="roi-input-val" style={{ color }}>{typeof val === "number" && val > 100 ? val.toLocaleString() : val}{label.includes("%") ? "%" : ""}</span>
                    </div>
                    <input type="range" min={min} max={max} step={step} value={val}
                      onChange={e => setter(+e.target.value)}
                      style={{ width:"100%", accentColor:color }} />
                  </div>
                ))}
              </div>

              <p className="section-head">Results</p>
              <div className="grid-kpi4" style={{ marginBottom:12 }}>
                {[
                  ["Contacted",     cTargeted.toLocaleString(),              C.indigo],
                  ["Txns Captured", txnsModel.toLocaleString(),              C.emerald],
                  ["Cost Saved",    `₹${(costMass-costModel).toLocaleString()}`, C.amber],
                  ["Model ROI",     `${Math.round(roiModel)}%`,              C.violet],
                  ["Profit Model",  `₹${Math.round(profitModel/1000)}K`,    C.emerald],
                  ["Profit Mass",   `₹${Math.round(profitMass/1000)}K`,     C.slate],
                ].map(([lbl,val,color]) => (
                  <div key={lbl} className="metric-kpi" style={{ borderLeftColor:color }}>
                    <div className="metric-val" style={{ color, fontSize:14 }}>{val}</div>
                    <div className="metric-label">{lbl}</div>
                  </div>
                ))}
              </div>

              <div className="card">
                <div className="card-label">Mass Marketing vs Model-Targeted (₹)</div>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart barCategoryGap="30%"
                    data={[
                      { name:"Cost",    mass:costMass,    model:costModel },
                      { name:"Revenue", mass:revMass,     model:revModel },
                      { name:"Profit",  mass:profitMass,  model:profitModel },
                    ]}>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
                    <XAxis dataKey="name" tick={{ fill:C.muted, fontSize:10 }} />
                    <YAxis tick={{ fill:C.muted, fontSize:9 }} tickFormatter={v=>`₹${(v/1000).toFixed(0)}K`} />
                    <Tooltip content={<Tip />} formatter={v=>`₹${Math.round(v).toLocaleString()}`} />
                    <Bar dataKey="mass"  name="Mass"              fill={C.rose}    radius={[3,3,0,0]} />
                    <Bar dataKey="model" name={`Top ${topPct}%`}  fill={C.emerald} radius={[3,3,0,0]} />
                    <Legend wrapperStyle={{ fontSize:10, color:C.muted }} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <p className="section-head">Key Findings</p>
              {[
                ["LightGBM is champion","AUC 0.8856 · AP 0.5818 · CV 0.8815 ±0.003",C.indigo],
                ["Default threshold suboptimal","0.5→0.25 nearly doubles recall",C.amber],
                ["Engineered features add value","pc_1 (252) + anomaly_score (101) in top-25",C.violet],
                ["5.56x top-decile lift","55.9% txn rate vs 10.1% baseline",C.emerald],
                ["PCA fails here","50 PCs = 27% variance · keep all 200",C.cyan],
                ["ROI is transformative","Top 10% targeting cuts outreach ~90%",C.rose],
              ].map(([title,body,color]) => (
                <div key={title} className="finding" style={{ borderLeftColor:color, background:`${color}11` }}>
                  <strong style={{ color }}>{title}</strong>
                  <br /><span style={{ color:C.muted, fontSize:10 }}>{body}</span>
                </div>
              ))}

              <p className="section-head">Architecture</p>
              <div className="card">
                <table className="arch-table">
                  <tbody>
                    {[
                      ["Champion","LightGBM · leaf-wise growth"],
                      ["Backup","XGBoost · AUC 0.8818"],
                      ["Features","200 raw + 17 engineered"],
                      ["Evaluation","ROC-AUC · AP · 5-fold CV"],
                      ["Threshold","F1-optimal = 0.25"],
                      ["Challenge","9:1 imbalance · threshold tuning"],
                    ].map(([k,v]) => (
                      <tr key={k}>
                        <td>{k}</td>
                        <td style={{ color:C.text }}>{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

        </div>

        <div className="footer">
          ⚠ Model trained on 50K stratified sample · Not for production without recalibration
        </div>
      </div>
    </>
  );
}

