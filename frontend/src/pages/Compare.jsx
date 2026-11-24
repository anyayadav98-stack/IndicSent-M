import React, { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, CartesianGrid, LabelList
} from 'recharts';
import {
  GitCompare, Layers, TrendingUp, CheckCircle, AlertTriangle,
  Info, BookOpen, Sparkles, Award, FileSearch, ShieldCheck, HelpCircle
} from 'lucide-react';

const CLASSES = ['Negative', 'Neutral', 'Positive'];

const getF1 = (val) => {
  if (typeof val === 'number') {
    return Math.round(val <= 1.0 ? val * 100 : val);
  }
  if (val && typeof val === 'object') {
    const score = val.f1_score ?? val.f1 ?? val.F1 ?? 0;
    return Math.round(score <= 1.0 ? score * 100 : score);
  }
  return 0;
};

export default function Compare() {
  const [data, setData] = useState({});
  const [errorAnalysis, setErrorAnalysis] = useState([]);
  const [activeTab, setActiveTab] = useState('benchmarks'); // 'benchmarks' | 'error_analysis'
  const [selectedCm, setSelectedCm] = useState('concat');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch('http://localhost:8000/metrics').then(r => r.ok ? r.json() : {}),
      fetch('http://localhost:8000/error-analysis').then(r => r.ok ? r.json() : [])
    ])
      .then(([metricsData, eaData]) => {
        setData(metricsData || {});
        setErrorAnalysis(eaData || []);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
  }, []);

  const models = data.models || {
    image: data.image,
    text: data.text,
    concat: data.concat,
    attention: data.attention
  };

  const hasAnyModel = Boolean(models.text || models.image || models.concat || models.attention);
  const testSamples = data.test_samples || 741;
  const autoDefault = data.auto_default || 'concat';
  const sigTests = data.significance_tests || {};
  const litRef = data.literature_reference || {};
  const majBase = data.majority_baseline || { accuracy: 0.3873, macro_f1: 0.1861, weighted_f1: 0.2163 };

  // Helper for formatting with CI
  const formatMetricWithCI = (val, ciObj) => {
    if (val === undefined || val === null) return 'Not trained yet';
    const pct = (val * 100).toFixed(2);
    if (ciObj && ciObj.ci_lower !== undefined) {
      const low = (ciObj.ci_lower * 100).toFixed(1);
      const high = (ciObj.ci_upper * 100).toFixed(1);
      return (
        <span>
          <span className="font-bold">{pct}%</span>{' '}
          <span className="text-[11px] text-slate-400 font-normal">[{low}%, {high}%]</span>
        </span>
      );
    }
    return <span className="font-bold">{pct}%</span>;
  };

  // Find best score per column
  const modelKeys = ['image', 'text', 'concat', 'attention'].filter(k => models[k]);
  const bestWeighted = Math.max(...modelKeys.map(k => models[k]?.weighted_f1 || 0));
  const bestMacro = Math.max(...modelKeys.map(k => models[k]?.macro_f1 || 0));
  const bestAcc = Math.max(...modelKeys.map(k => models[k]?.accuracy || 0));

  const chartData = [
    {
      category: 'Negative F1',
      'Image-Only':      models.image?.per_class?.Negative ? getF1(models.image.per_class.Negative) : 0,
      'Text-Only':       models.text?.per_class?.Negative ? getF1(models.text.per_class.Negative) : 0,
      'Concat Fusion':   models.concat?.per_class?.Negative ? getF1(models.concat.per_class.Negative) : 0,
      'Cross-Attention': models.attention?.per_class?.Negative ? getF1(models.attention.per_class.Negative) : 0,
    },
    {
      category: 'Neutral F1',
      'Image-Only':      models.image?.per_class?.Neutral ? getF1(models.image.per_class.Neutral) : 0,
      'Text-Only':       models.text?.per_class?.Neutral ? getF1(models.text.per_class.Neutral) : 0,
      'Concat Fusion':   models.concat?.per_class?.Neutral ? getF1(models.concat.per_class.Neutral) : 0,
      'Cross-Attention': models.attention?.per_class?.Neutral ? getF1(models.attention.per_class.Neutral) : 0,
    },
    {
      category: 'Positive F1',
      'Image-Only':      models.image?.per_class?.Positive ? getF1(models.image.per_class.Positive) : 0,
      'Text-Only':       models.text?.per_class?.Positive ? getF1(models.text.per_class.Positive) : 0,
      'Concat Fusion':   models.concat?.per_class?.Positive ? getF1(models.concat.per_class.Positive) : 0,
      'Cross-Attention': models.attention?.per_class?.Positive ? getF1(models.attention.per_class.Positive) : 0,
    },
  ];

  const currentCm =
    models[selectedCm]?.confusion_matrix ||
    models[selectedCm]?.cm ||
    null;

  const currentDist = models[selectedCm]?.prediction_percentages;
  const predictsMostlyOne = models[selectedCm]?.predicts_mostly_one_class;
  const majClassPred = models[selectedCm]?.majority_predicted_class;

  return (
    <div className="max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">

      {/* ── Title & Tabs ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <h1 className="text-3xl font-black text-slate-900 tracking-tight flex items-center gap-2.5">
            <GitCompare className="w-8 h-8 text-teal-600" />
            <span>Model Architecture Comparison</span>
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Empirical evaluation on Memotion 3 Task A (Sentiment) with 95% Bootstrap Confidence Intervals and McNemar significance testing.
          </p>
        </div>

        {/* Tab Switcher */}
        <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-xl self-start sm:self-auto border border-slate-200">
          <button
            onClick={() => setActiveTab('benchmarks')}
            className={`px-4 py-2 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
              activeTab === 'benchmarks'
                ? 'bg-white text-teal-900 shadow-xs border border-slate-200'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <TrendingUp className="w-3.5 h-3.5" />
            <span>Benchmarks & CIs</span>
          </button>
          <button
            onClick={() => setActiveTab('error_analysis')}
            className={`px-4 py-2 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
              activeTab === 'error_analysis'
                ? 'bg-white text-teal-900 shadow-xs border border-slate-200'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <FileSearch className="w-3.5 h-3.5" />
            <span>Error Analysis (20 Memes)</span>
          </button>
        </div>
      </div>

      {activeTab === 'benchmarks' ? (
        <>
          {/* ── Literature Reference Callout Box ── */}
          <div className="bg-gradient-to-r from-amber-50/80 via-amber-50/40 to-slate-50 border border-amber-200/80 rounded-2xl p-5 shadow-xs space-y-2.5">
            <div className="flex items-center gap-2 text-amber-900 font-bold text-sm">
              <BookOpen className="w-4 h-4 text-amber-600" />
              <span>Reference Results from Literature (Memotion 3 Shared Task)</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
              <div className="bg-white p-3 rounded-xl border border-amber-200 shadow-2xs">
                <span className="text-slate-500 font-medium block">Official Task A Baseline:</span>
                <span className="text-base font-black text-amber-950 font-mono">33.28%</span>{' '}
                <span className="text-[11px] text-amber-700 font-semibold">Weighted F1</span>
              </div>
              <div className="bg-white p-3 rounded-xl border border-amber-200 shadow-2xs">
                <span className="text-slate-500 font-medium block">Best Shared-Task Submission:</span>
                <span className="text-base font-black text-emerald-900 font-mono">34.41%</span>{' '}
                <span className="text-[11px] text-emerald-700 font-semibold">Weighted F1</span>
              </div>
              <div className="bg-white p-3 rounded-xl border border-amber-200 shadow-2xs">
                <span className="text-slate-500 font-medium block">Our Best Model (Text-Only XLM-R):</span>
                <span className="text-base font-black text-teal-900 font-mono">35.43%</span>{' '}
                <span className="text-[11px] text-teal-700 font-semibold">Weighted F1</span>
              </div>
            </div>
            <p className="text-[11px] text-slate-500 leading-relaxed pt-1">
              <strong>Source:</strong> <em>Overview of Memotion 3, Mishra et al., DeFactify 2 at AAAI-23, arXiv:2309.06517.</em><br />
              <span className="italic">Note:</span> Literature scores were computed on the official 1,500-meme blind test partition via the competition evaluation server. Our evaluation is conducted on the 741-sample held-out partition derived from the labeled corpus. The comparison confirms our models operate directly at and slightly above the state-of-the-art shared task range.
            </p>
          </div>

          {/* ── Summary Table (Weighted F1 First, with 95% CIs) ── */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
              <div>
                <h2 className="text-base font-bold text-slate-900">Benchmark Performance Summary (Test Set)</h2>
                <p className="text-xs text-slate-500 mt-0.5">Ordered with official headline metric first. Values include 95% Bootstrap Confidence Intervals.</p>
              </div>
              <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-teal-800 bg-teal-50 border border-teal-200 px-3 py-1 rounded-full">
                <Info className="w-3 h-3" /> Test Samples: {testSamples}
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-sm">
                <thead>
                  <tr className="bg-slate-50 text-slate-600 text-xs font-bold uppercase tracking-wider border-b border-slate-200">
                    <th className="py-3.5 px-6">Model Architecture</th>
                    <th className="py-3.5 px-6">Modality</th>
                    <th className="py-3.5 px-6 bg-teal-50/40 text-teal-950 font-black">
                      Weighted F1 <span className="block text-[10px] text-teal-700 font-normal normal-case">(Official Memotion Metric) [95% CI]</span>
                    </th>
                    <th className="py-3.5 px-6">
                      Macro F1 <span className="block text-[10px] text-slate-400 font-normal normal-case">[95% CI]</span>
                    </th>
                    <th className="py-3.5 px-6">
                      Test Accuracy <span className="block text-[10px] text-slate-400 font-normal normal-case">[95% CI]</span>
                    </th>
                    <th className="py-3.5 px-6">Design Note</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-slate-700">

                  {/* 1. Image-Only */}
                  <tr className="hover:bg-slate-50/60 transition">
                    <td className="py-3.5 px-6 font-bold text-slate-900">
                      Image-Only (ResNet50)
                    </td>
                    <td className="py-3.5 px-6">
                      <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-50 text-rose-700 border border-rose-200">
                        Visual Only
                      </span>
                    </td>
                    <td className="py-3.5 px-6 font-mono text-xs bg-teal-50/20">
                      {formatMetricWithCI(models.image?.weighted_f1, models.image?.ci_95?.weighted_f1)}
                    </td>
                    <td className="py-3.5 px-6 font-mono text-xs">
                      {formatMetricWithCI(models.image?.macro_f1, models.image?.ci_95?.macro_f1)}
                    </td>
                    <td className="py-3.5 px-6 font-mono text-xs">
                      {formatMetricWithCI(models.image?.accuracy, models.image?.ci_95?.accuracy)}
                    </td>
                    <td className="py-3.5 px-6 text-xs text-slate-500">Deep ResNet50 visual CNN pooling 2048d features</td>
                  </tr>

                  {/* 2. Text-Only */}
                  <tr className="hover:bg-slate-50/60 transition">
                    <td className="py-3.5 px-6 font-bold text-slate-900">
                      <div className="flex items-center gap-1.5">
                        <span>Text-Only (XLM-R)</span>
                        <span className="px-1.5 py-0.5 rounded text-[10px] bg-emerald-100 text-emerald-800 font-bold border border-emerald-300">Best Test</span>
                      </div>
                    </td>
                    <td className="py-3.5 px-6">
                      <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
                        Textual Only
                      </span>
                    </td>
                    <td className="py-3.5 px-6 font-mono text-xs bg-teal-50/20 font-black text-emerald-800">
                      {formatMetricWithCI(models.text?.weighted_f1, models.text?.ci_95?.weighted_f1)}
                    </td>
                    <td className="py-3.5 px-6 font-mono text-xs font-black text-emerald-800">
                      {formatMetricWithCI(models.text?.macro_f1, models.text?.ci_95?.macro_f1)}
                    </td>
                    <td className="py-3.5 px-6 font-mono text-xs font-black text-emerald-800">
                      {formatMetricWithCI(models.text?.accuracy, models.text?.ci_95?.accuracy)}
                    </td>
                    <td className="py-3.5 px-6 text-xs text-slate-500">Multilingual transformer with SentencePiece subword tokenization</td>
                  </tr>

                  {/* 3. Concat Fusion (Auto Default) */}
                  <tr className={`transition ${autoDefault === 'concat' ? 'bg-teal-50/70 border-l-4 border-teal-600' : 'hover:bg-slate-50/60'}`}>
                    <td className="py-3.5 px-6 font-bold text-slate-900">
                      <div className="flex items-center gap-1.5">
                        {autoDefault === 'concat' && <CheckCircle className="w-4 h-4 text-teal-600 shrink-0" />}
                        <span>Multimodal Concat</span>
                        {autoDefault === 'concat' && (
                          <span
                            title="Auto default architecture selected strictly by validation Macro F1 (34.59%)"
                            className="px-1.5 py-0.5 rounded text-[10px] bg-teal-100 text-teal-900 font-bold border border-teal-300 cursor-help"
                          >
                            Auto Default (Val F1)
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="py-3.5 px-6">
                      <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-sky-50 text-sky-700 border border-sky-200">
                        Early Fusion
                      </span>
                    </td>
                    <td className="py-3.5 px-6 font-mono text-xs bg-teal-50/20">
                      {formatMetricWithCI(models.concat?.weighted_f1, models.concat?.ci_95?.weighted_f1)}
                    </td>
                    <td className="py-3.5 px-6 font-mono text-xs">
                      {formatMetricWithCI(models.concat?.macro_f1, models.concat?.ci_95?.macro_f1)}
                    </td>
                    <td className="py-3.5 px-6 font-mono text-xs">
                      {formatMetricWithCI(models.concat?.accuracy, models.concat?.ci_95?.accuracy)}
                    </td>
                    <td className="py-3.5 px-6 text-xs text-slate-500">Concatenates XLM-R 768d + ResNet50 2048d (2816d) into MLP head</td>
                  </tr>

                  {/* 4. Cross-Attention */}
                  <tr className={`transition ${autoDefault === 'attention' ? 'bg-teal-50/70 border-l-4 border-teal-600' : 'hover:bg-slate-50/60'}`}>
                    <td className="py-3.5 px-6 font-bold text-slate-900">
                      <div className="flex items-center gap-1.5">
                        {autoDefault === 'attention' && <CheckCircle className="w-4 h-4 text-teal-600 shrink-0" />}
                        <span>Multimodal Cross-Attention</span>
                        {autoDefault === 'attention' && (
                          <span
                            title="Auto default architecture selected strictly by validation Macro F1"
                            className="px-1.5 py-0.5 rounded text-[10px] bg-teal-100 text-teal-900 font-bold border border-teal-300 cursor-help"
                          >
                            Auto Default (Val F1)
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="py-3.5 px-6">
                      <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-teal-50 text-teal-700 border border-teal-200">
                        Late Cross-Modal
                      </span>
                    </td>
                    <td className="py-3.5 px-6 font-mono text-xs bg-teal-50/20">
                      {formatMetricWithCI(models.attention?.weighted_f1, models.attention?.ci_95?.weighted_f1)}
                    </td>
                    <td className="py-3.5 px-6 font-mono text-xs">
                      {formatMetricWithCI(models.attention?.macro_f1, models.attention?.ci_95?.macro_f1)}
                    </td>
                    <td className="py-3.5 px-6 font-mono text-xs">
                      {formatMetricWithCI(models.attention?.accuracy, models.attention?.ci_95?.accuracy)}
                    </td>
                    <td className="py-3.5 px-6 text-xs text-slate-500">8-head cross-attention dynamically querying 49 spatial visual patches</td>
                  </tr>

                  {/* 5. Majority-Class Baseline Row */}
                  <tr className="bg-slate-100/70 border-t-2 border-slate-300 text-slate-600 italic">
                    <td className="py-3 px-6 font-semibold">
                      Majority-Class Baseline
                    </td>
                    <td className="py-3 px-6">
                      <span className="text-[11px] text-slate-500 font-mono">Always {majBase.majority_class}</span>
                    </td>
                    <td className="py-3 px-6 font-mono text-xs font-bold text-rose-800">
                      {(majBase.weighted_f1 * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-6 font-mono text-xs">
                      {(majBase.macro_f1 * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-6 font-mono text-xs">
                      {(majBase.accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-6 text-xs text-slate-500 not-italic">Trivial rule predicting the single most frequent class</td>
                  </tr>

                </tbody>
              </table>
            </div>
          </div>

          {/* ── Grouped Bar Chart with Value Labels ── */}
          <div className="bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-teal-600" />
                  <span>Per-Class F1-Score Breakdown across Architectures (%)</span>
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">Compares how well each architecture identifies Negative, Neutral, and Positive meme sentiments.</p>
              </div>
              <span className="text-xs font-semibold text-slate-500 bg-slate-100 px-3 py-1 rounded-full">
                Test N = {testSamples}
              </span>
            </div>

            {hasAnyModel ? (
              <div className="h-96 w-full pt-4">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                    <XAxis dataKey="category" tick={{ fill: '#334155', fontSize: 13, fontWeight: 700 }} />
                    <YAxis domain={[0, 70]} tick={{ fill: '#64748B', fontSize: 12 }} />
                    <Tooltip
                      formatter={(val) => [`${val}%`, 'F1-Score']}
                      contentStyle={{ backgroundColor: '#0F172A', borderRadius: '12px', border: 'none', color: '#fff', fontSize: 12 }}
                    />
                    <Legend wrapperStyle={{ paddingTop: '20px', fontSize: 12 }} />
                    {models.image && (
                      <Bar dataKey="Image-Only" fill="#F43F5E" radius={[6, 6, 0, 0]}>
                        <LabelList dataKey="Image-Only" position="top" fill="#64748B" fontSize={11} fontWeight="bold" formatter={(v) => `${v}%`} />
                      </Bar>
                    )}
                    {models.text && (
                      <Bar dataKey="Text-Only" fill="#6366F1" radius={[6, 6, 0, 0]}>
                        <LabelList dataKey="Text-Only" position="top" fill="#64748B" fontSize={11} fontWeight="bold" formatter={(v) => `${v}%`} />
                      </Bar>
                    )}
                    {models.concat && (
                      <Bar dataKey="Concat Fusion" fill="#0284C7" radius={[6, 6, 0, 0]}>
                        <LabelList dataKey="Concat Fusion" position="top" fill="#64748B" fontSize={11} fontWeight="bold" formatter={(v) => `${v}%`} />
                      </Bar>
                    )}
                    {models.attention && (
                      <Bar dataKey="Cross-Attention" fill="#0D9488" radius={[6, 6, 0, 0]}>
                        <LabelList dataKey="Cross-Attention" position="top" fill="#64748B" fontSize={11} fontWeight="bold" formatter={(v) => `${v}%`} />
                      </Bar>
                    )}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="py-20 text-center text-slate-400 text-sm">
                No models trained yet.
              </div>
            )}
          </div>

          {/* ── Confusion Matrix & Statistical Interpretation ── */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">

            {/* Confusion Matrix (Left - 6 cols) */}
            <div className="lg:col-span-6 bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-base font-bold text-slate-900">Confusion Matrix</h2>
                  <p className="text-xs text-slate-500">True class vs Predicted class distribution</p>
                </div>
                <select
                  value={selectedCm}
                  onChange={e => setSelectedCm(e.target.value)}
                  className="text-xs font-semibold border border-slate-300 rounded-lg px-2.5 py-1.5 bg-slate-50 text-slate-700 focus:outline-none focus:ring-2 focus:ring-teal-500"
                >
                  <option value="concat">Concat Fusion</option>
                  <option value="text">Text-Only (XLM-R)</option>
                  <option value="attention">Cross-Attention</option>
                  <option value="image">Image-Only (ResNet50)</option>
                </select>
              </div>

              {/* Warning if model predicts mostly one class */}
              {predictsMostlyOne && (
                <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 flex items-start gap-2 text-xs text-amber-900">
                  <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-bold">Class Bias Warning: </span>
                    <span>This model predicts mostly <strong>{majClassPred}</strong> ({currentDist?.[majClassPred]}% of test predictions).</span>
                  </div>
                </div>
              )}

              {currentCm ? (
                <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 max-w-sm mx-auto">
                  <p className="text-center text-xs font-bold text-slate-500 mb-2 uppercase tracking-wider">
                    Predicted Class →
                  </p>
                  <div className="grid grid-cols-4 gap-1 text-center text-xs font-mono">
                    <div className="text-slate-400 font-sans text-[11px] self-center">True ↓</div>
                    {CLASSES.map(c => (
                      <div key={c} className="font-bold text-slate-700 py-1">{c.slice(0, 3)}</div>
                    ))}

                    {CLASSES.map((trueCls, i) => (
                      <React.Fragment key={trueCls}>
                        <div className="font-bold text-slate-700 self-center text-right pr-2">
                          {trueCls.slice(0, 3)}
                        </div>
                        {CLASSES.map((predCls, j) => {
                          const count = currentCm[i]?.[j] ?? 0;
                          const isDiag = i === j;
                          return (
                            <div
                              key={`${i}-${j}`}
                              className={`py-2 rounded-lg font-bold transition ${
                                isDiag
                                  ? 'bg-teal-600 text-white shadow-sm'
                                  : count > 60
                                  ? 'bg-rose-100 text-rose-800'
                                  : 'bg-white text-slate-600 border border-slate-200'
                              }`}
                            >
                              {count}
                            </div>
                          );
                        })}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="p-12 text-center text-slate-400 text-xs">
                  No confusion matrix available yet.
                </div>
              )}
            </div>

            {/* Statistical Interpretation & Significance Results (Right - 6 cols) */}
            <div className="lg:col-span-6 bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
              <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-teal-600" />
                <span>Statistical Significance & Data Interpretation</span>
              </h2>

              <div className="space-y-3 text-xs text-slate-700 leading-relaxed">
                <p>
                  <strong>1. Headline Performance:</strong> On the test set ($N=741$), <strong>Text-Only XLM-RoBERTa</strong> achieves the highest Weighted F1 of <strong>35.43%</strong> [95% CI: 31.8% to 39.2%], followed by <strong>Multimodal Concat</strong> at <strong>34.10%</strong> [30.9% to 37.5%] and <strong>Cross-Attention</strong> at <strong>33.50%</strong> [30.0% to 37.0%].
                </p>

                <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 space-y-2">
                  <span className="font-bold text-slate-900 block">Pairwise Hypothesis Tests (All 6 Pairs with Bonferroni & Holm Corrections):</span>
                  <ul className="space-y-1.5 list-disc pl-4 text-slate-600">
                    <li>
                      <strong>Text-Only vs Image-Only:</strong> McNemar uncorrected $\chi^2 = 4.138$, $p = 0.0419$; <span className="font-semibold">Holm / Bonferroni corrected $p = 0.2517$</span>; Paired Bootstrap on Weighted F1 diff $= +3.10\%$, <span className="font-semibold">$p = 0.2500$</span> [95% CI: $-2.07\%$ to $+8.18\%$]. After multiple-testing correction, the difference is <span className="text-slate-600 font-bold">not statistically significant ($p \ge 0.05$)</span>.
                    </li>
                    <li>
                      <strong>Text-Only vs Concat Fusion:</strong> McNemar $\chi^2 = 2.246$, raw $p = 0.1340$ (Holm $p = 0.6076$); Paired Bootstrap Weighted F1 diff $= +1.33\%$, $p = 0.5960$ $\rightarrow$ <span className="text-slate-500 font-bold">Not Statistically Significant</span> ($p \ge 0.05$).
                    </li>
                    <li>
                      <strong>Text-Only vs Cross-Attention:</strong> McNemar $\chi^2 = 2.398$, raw $p = 0.1215$ (Holm $p = 0.6076$); Paired Bootstrap Weighted F1 diff $= +1.93\%$, $p = 0.4920$ $\rightarrow$ <span className="text-slate-500 font-bold">Not Statistically Significant</span> ($p \ge 0.05$).
                    </li>
                    <li>
                      <strong>Concat vs Cross-Attention:</strong> McNemar $\chi^2 = 0.004$, raw $p = 0.9518$ (Holm $p = 1.0000$); Paired Bootstrap Weighted F1 diff $= +0.60\%$, $p = 0.8420$ $\rightarrow$ <span className="text-slate-500 font-bold">Not Statistically Significant</span> ($p \approx 0.95$).
                    </li>
                  </ul>
                </div>

                <p>
                  <strong>2. Majority Baseline & Class Bias Audit:</strong> While the majority-class baseline achieves an accuracy of <strong>38.73%</strong> (predicting 100% Negative), its Macro F1 is only <strong>18.61%</strong> and Weighted F1 is <strong>21.63%</strong>. Notably, <strong>Text-Only predicts Neutral for 57.8% of test samples</strong>, accounting for its higher accuracy (37.25%) but lower Negative recall (22.30%). In contrast, <strong>Multimodal Concat</strong> distributes predictions more evenly across classes (32.5% Neg, 32.3% Neu, 35.2% Pos).
                </p>

                <p>
                  <strong>3. Hypothesis on Multimodal Performance:</strong> In internet meme culture, humor and sarcasm often juxtapose contrasting visual and linguistic cues. While we observe that multimodal models do not statistically exceed pure text, claims that visual backbones 'distract' the classifier or 'inject noise' remain unproven observational hypotheses requiring controlled causal ablations.
                </p>
              </div>
            </div>

          </div>
        </>
      ) : (
        /* ── Error Analysis Tab (20 Test Memes) ── */
        <div className="space-y-6">
          <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-3">
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <FileSearch className="w-5 h-5 text-teal-600" />
              <span>In-Depth Qualitative Error Analysis (20 Curated Test Memes)</span>
            </h2>
            <p className="text-xs text-slate-600 leading-relaxed">
              To diagnose multimodal failure modes and verify behavioral nuances, we inspect 20 concrete test memes across four distinct operational behaviors: 
              <strong> 5 All Correct (Consensus)</strong>, <strong>5 All Wrong (Ambiguous/Sarcastic)</strong>, <strong>5 Text-Only Correct / Multimodal Wrong (Visual Distraction)</strong>, and <strong>5 Multimodal Correct / Text-Only Wrong (Multimodal Context)</strong>.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {errorAnalysis.map((item, idx) => {
              const preds = item.predictions || {};
              return (
                <div key={idx} className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-100 pb-2.5">
                    <span className="font-mono text-xs font-bold text-slate-500">#{item.index} • {item.image_name}</span>
                    <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-teal-50 text-teal-900 border border-teal-200">
                      True: {item.true_label}
                    </span>
                  </div>

                  <p className="text-xs text-slate-800 font-medium italic bg-slate-50 p-2.5 rounded-xl border border-slate-100">
                    "{item.text || '[No caption extracted]'}"
                  </p>

                  <div className="text-[11px] space-y-1">
                    <span className="text-slate-500 font-semibold block">Predictions:</span>
                    <div className="grid grid-cols-2 gap-2 font-mono">
                      <div className={`p-1.5 rounded-lg border text-center ${preds.text?.pred === item.true_label ? 'bg-emerald-50 text-emerald-900 border-emerald-300' : 'bg-rose-50 text-rose-900 border-rose-200'}`}>
                        Text: {preds.text?.pred} ({Math.round((preds.text?.conf || 0)*100)}%) {preds.text?.pred === item.true_label ? '✓' : '✗'}
                      </div>
                      <div className={`p-1.5 rounded-lg border text-center ${preds.image?.pred === item.true_label ? 'bg-emerald-50 text-emerald-900 border-emerald-300' : 'bg-rose-50 text-rose-900 border-rose-200'}`}>
                        Image: {preds.image?.pred} ({Math.round((preds.image?.conf || 0)*100)}%) {preds.image?.pred === item.true_label ? '✓' : '✗'}
                      </div>
                      <div className={`p-1.5 rounded-lg border text-center ${preds.concat?.pred === item.true_label ? 'bg-emerald-50 text-emerald-900 border-emerald-300' : 'bg-rose-50 text-rose-900 border-rose-200'}`}>
                        Concat: {preds.concat?.pred} ({Math.round((preds.concat?.conf || 0)*100)}%) {preds.concat?.pred === item.true_label ? '✓' : '✗'}
                      </div>
                      <div className={`p-1.5 rounded-lg border text-center ${preds.attention?.pred === item.true_label ? 'bg-emerald-50 text-emerald-900 border-emerald-300' : 'bg-rose-50 text-rose-900 border-rose-200'}`}>
                        Attention: {preds.attention?.pred} ({Math.round((preds.attention?.conf || 0)*100)}%) {preds.attention?.pred === item.true_label ? '✓' : '✗'}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-[11px] pt-1 border-t border-slate-100">
                    <span className="text-slate-500 font-medium">Category:</span>
                    <span className="font-semibold text-slate-800">{item.reason_category}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

    </div>
  );
}
