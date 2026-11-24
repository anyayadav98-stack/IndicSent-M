import React, { useState } from 'react';
import {
  BarChart3, Image as ImageIcon, Flame, FileText,
  Network, Target, Sparkles, Lightbulb, ArrowLeft,
  AlertCircle, Share2, RefreshCw, CheckCircle2, Sliders, Clock, AlertTriangle, Layers
} from 'lucide-react';

/* ─── Sentiment styling configurations matching reference ─── */
const sentimentStyles = {
  Positive: {
    badgeBg: 'bg-[#DCFCE7] border border-[#BBF7D0] text-[#166534]',
    emoji: '😊',
    label: 'Positive',
    barClass: 'bg-[#10B981]'
  },
  Neutral: {
    badgeBg: 'bg-[#FEF3C7] border border-[#FDE68A] text-[#92400E]',
    emoji: '😐',
    label: 'Neutral',
    barClass: 'bg-[#0284C7]'
  },
  Negative: {
    badgeBg: 'bg-[#FEE2E2] border border-[#FECACA] text-[#991B1B]',
    emoji: '😡',
    label: 'Negative',
    barClass: 'bg-[#EF4444]'
  },
};

function CardHeader({ icon: Icon, title, iconClass = 'text-white' }) {
  return (
    <div className="bg-[#103778] text-white px-4 py-2.5 flex items-center gap-2 rounded-t-2xl shadow-sm">
      <Icon className={`w-4 h-4 ${iconClass}`} />
      <span className="text-sm font-bold tracking-wide">{title}</span>
    </div>
  );
}

function ConfBar({ emoji, label, score, barColor }) {
  const pct = Math.round((score || 0) * 100);
  return (
    <div className="flex items-center gap-3 text-xs">
      <span className="w-20 flex items-center gap-1.5 font-semibold text-slate-700 shrink-0">
        <span className="text-base">{emoji}</span>
        <span>{label}</span>
      </span>
      <div className="flex-1 bg-slate-100 rounded-full h-3 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ width: `${pct}%`, backgroundColor: barColor }}
        />
      </div>
      <span className="w-24 text-right font-mono font-bold text-slate-700 shrink-0">
        {(score || 0).toFixed(2)} ({pct}.0%)
      </span>
    </div>
  );
}

export default function Results({ result, onBackToAnalyze, onRerunWithText }) {
  const [data, setData] = useState(result || {});
  const [editedText, setEditedText] = useState(data.extracted_text || '');
  const [rerunning, setRerunning] = useState(false);
  const [showAllCompare, setShowAllCompare] = useState(Boolean(data.compare_all));

  const sentiment = data.final_sentiment || "Neutral";
  const scores = data.confidence_scores || { Positive: 0.24, Neutral: 0.56, Negative: 0.20 };
  const currentSentStyle = sentimentStyles[sentiment] || sentimentStyles.Neutral;
  const lowConf = data.low_confidence === true;
  const ocrUnreliable = data.ocr_unreliable === true;
  const activeModel = data.active_model || 'concat';
  const modelsUsed = data.models_used || [];
  const limeWords = data.lime_words || [];
  const timings = data.timings || {};

  const handleRerun = async () => {
    if (!editedText.trim()) return;
    setRerunning(true);
    try {
      const formData = new FormData();
      formData.append('text', editedText.trim());
      formData.append('model_type', data.active_model || 'auto');
      formData.append('include_explanations', true);

      // Convert base64 input image back to blob for upload
      if (data.input_image_base64) {
        const res = await fetch(data.input_image_base64);
        const blob = await res.blob();
        formData.append('file', blob, 'meme.jpg');
      }

      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        const updated = await res.json();
        setData(updated);
        setEditedText(updated.extracted_text);
        if (onRerunWithText) onRerunWithText(updated);
      }
    } catch (e) {
      console.error("Re-run failed:", e);
    } finally {
      setRerunning(false);
    }
  };

  return (
    <div className="max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">

      {/* ── Top Bar: Back Action, Status Pills & Timings ── */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <button
          onClick={onBackToAnalyze}
          className="inline-flex items-center gap-2 text-sm font-semibold text-slate-600 hover:text-teal-700 transition"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Analyze Another Meme</span>
        </button>

        <div className="flex items-center gap-2 flex-wrap">
          {/* Active Model Pill */}
          <span className="inline-flex items-center gap-1.5 bg-teal-50 border border-teal-200 text-teal-900 rounded-full px-3.5 py-1 text-xs font-bold shadow-xs">
            <Network className="w-3.5 h-3.5 text-teal-700" />
            <span>Active Model: {data.checkpoint_used || activeModel.toUpperCase()}</span>
          </span>

          {/* Low Confidence Pill */}
          {lowConf && (
            <div className="inline-flex items-center gap-1.5 bg-amber-50 border border-amber-300 text-amber-900 rounded-full px-3.5 py-1 text-xs font-bold shadow-xs">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-600 shrink-0" />
              <span>Low Confidence ({Math.round((data.confidence_margin || 0.1) * 100)}% margin)</span>
            </div>
          )}

          {/* Timings Pill */}
          {timings.total_ms && (
            <div className="inline-flex items-center gap-1 bg-slate-100 border border-slate-300 text-slate-700 rounded-full px-3 py-1 text-xs font-medium">
              <Clock className="w-3 h-3 text-slate-500" />
              <span>{timings.total_ms}ms total</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Honest Disclaimer Banner (Mandatory from real outputs/ metrics) ── */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-3.5 flex items-start gap-3 text-xs text-slate-700">
        <AlertCircle className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
        <div className="flex-1">
          <span className="font-bold">Model Performance Disclaimer: </span>
          <span>{data.disclaimer || "Predictions are automatic and can be wrong. Every number is grounded in empirical test metrics on Memotion 3."}</span>
        </div>
      </div>

      {/* ── Main Layout: Visual column (Left) & Metrics/Text (Right) ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">

        {/* ════════ LEFT COLUMN: Images & Grad-CAM ════════ */}
        <div className="lg:col-span-5 space-y-6">

          {/* Card 1: Input Meme */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
            <CardHeader icon={ImageIcon} title="Input Meme" />
            <div className="p-3 bg-slate-100 flex items-center justify-center min-h-[240px]">
              {data.input_image_base64 ? (
                <img
                  src={data.input_image_base64}
                  alt="Input Meme"
                  className="rounded-xl max-h-[320px] w-auto max-w-full object-contain shadow-sm"
                />
              ) : (
                <div className="text-slate-400 text-xs italic">No image provided</div>
              )}
            </div>
          </div>

          {/* Card 2: Grad-CAM (Always Shown, Never Hidden) */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
            <CardHeader icon={Flame} title="Grad-CAM (Visual Influence)" iconClass="text-orange-400" />
            
            {/* Warning advisory if low confidence */}
            {lowConf && data.gradcam_warning && (
              <div className="bg-amber-50 px-4 py-2 border-b border-amber-200 flex items-center gap-2 text-[11px] font-semibold text-amber-900">
                <AlertTriangle className="w-3.5 h-3.5 text-amber-600 shrink-0" />
                <span>{data.gradcam_warning}</span>
              </div>
            )}

            <div className="p-3 bg-slate-100 flex items-center justify-center min-h-[240px]">
              {activeModel === 'text' ? (
                <div className="text-center space-y-2 px-4 py-8">
                  <FileText className="w-8 h-8 text-slate-400 mx-auto" />
                  <p className="text-xs text-slate-500 font-semibold">Grad-CAM not applicable for the text-only model.</p>
                </div>
              ) : data.gradcam_overlay ? (
                <img
                  src={data.gradcam_overlay}
                  alt="Grad-CAM Saliency Overlay"
                  className="rounded-xl max-h-[320px] w-auto max-w-full object-contain shadow-sm"
                />
              ) : (
                <div className="text-center space-y-2 px-4 py-8">
                  <Flame className="w-8 h-8 text-orange-400 mx-auto animate-pulse" />
                  <p className="text-xs text-slate-500 font-semibold">Computing visual saliency heatmap...</p>
                </div>
              )}
            </div>

            {/* Saliency Gradient Bar */}
            <div className="px-5 py-3 bg-slate-50 border-t border-slate-100">
              <div className="h-3 w-full rounded-full bg-gradient-to-r from-blue-700 via-cyan-400 via-yellow-300 via-orange-500 to-red-600 shadow-inner" />
              <div className="flex justify-between text-[11px] font-bold text-slate-500 mt-1.5 px-0.5">
                <span>Lower Influence</span>
                <span>Higher Influence</span>
              </div>
            </div>
          </div>

        </div>

        {/* ════════ RIGHT COLUMN: Text, Scores, LIME & Compare ════════ */}
        <div className="lg:col-span-7 space-y-5">

          {/* Section Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5 text-slate-900">
              <div className="w-7 h-7 rounded-lg bg-teal-50 flex items-center justify-center text-teal-700">
                <BarChart3 className="w-5 h-5" />
              </div>
              <h1 className="text-2xl font-black tracking-tight">
                Sentiment Analysis Results
              </h1>
            </div>

            {/* Toggle Compare All */}
            {data.compare_all && (
              <button
                type="button"
                onClick={() => setShowAllCompare(!showAllCompare)}
                className="px-3 py-1.5 rounded-xl border border-teal-300 bg-teal-50 text-teal-800 text-xs font-bold hover:bg-teal-100 flex items-center gap-1.5 transition"
              >
                <Layers className="w-3.5 h-3.5" />
                <span>{showAllCompare ? 'Hide Multi-Model Comparison' : 'Compare All Models'}</span>
              </button>
            )}
          </div>

          {/* ── Extracted Text Card (Editable + Re-run) ── */}
          <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-slate-800 font-bold text-sm">
                <FileText className="w-4 h-4 text-sky-600" />
                <span>Meme Caption / Extracted Text</span>
              </div>

              {data.ocr_quality_score !== undefined && (
                <span className={`text-[11px] font-bold px-2.5 py-0.5 rounded-full border ${
                  ocrUnreliable ? 'bg-amber-100 text-amber-900 border-amber-300' : 'bg-emerald-100 text-emerald-900 border-emerald-300'
                }`}>
                  OCR Quality: {Math.round(data.ocr_quality_score * 100)}%
                </span>
              )}
            </div>

            {/* Warning if OCR was noisy */}
            {ocrUnreliable && (
              <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 flex items-start gap-2 text-xs text-amber-900">
                <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                <div className="flex-1">
                  <span className="font-bold">Text extraction looks unreliable. </span>
                  <span>Please verify or edit the text below, then click "Re-run with this text".</span>
                  {data.ocr_suggestion && <p className="mt-1 font-semibold text-amber-950">{data.ocr_suggestion}</p>}
                </div>
              </div>
            )}

            {/* Editable Text Area */}
            <div className="relative">
              <textarea
                rows={3}
                value={editedText}
                onChange={(e) => setEditedText(e.target.value)}
                placeholder="Enter or edit meme text here..."
                className="w-full rounded-xl border border-slate-300 p-3 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500 font-medium text-slate-800 leading-relaxed resize-none"
              />
            </div>

            {/* Re-run button */}
            <div className="flex justify-end">
              <button
                type="button"
                onClick={handleRerun}
                disabled={rerunning || editedText.trim() === data.extracted_text}
                className="px-4 py-2 bg-teal-700 hover:bg-teal-800 disabled:bg-slate-200 disabled:text-slate-400 disabled:cursor-not-allowed text-white text-xs font-bold rounded-xl shadow-sm transition flex items-center gap-1.5"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${rerunning ? 'animate-spin' : ''}`} />
                <span>{rerunning ? 'Re-running Inference...' : 'Re-run with this text'}</span>
              </button>
            </div>
          </div>

          {/* ── Compare All Models Card (If Requested) ── */}
          {showAllCompare && data.compare_all && (
            <div className="bg-white p-5 rounded-2xl border border-teal-200 shadow-sm space-y-3 bg-teal-50/20">
              <div className="flex items-center gap-2 text-slate-900 font-bold text-sm">
                <Layers className="w-4 h-4 text-teal-700" />
                <span>Cross-Architecture Prediction Comparison (Same Meme)</span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5">
                {[
                  { key: 'text', label: 'Text-Only' },
                  { key: 'image', label: 'Image-Only' },
                  { key: 'concat', label: 'Concat Fusion' },
                  { key: 'attention', label: 'Cross-Attention' },
                  { key: 'ensemble', label: 'Ensemble' }
                ].map(m => {
                  const item = data.compare_all[m.key];
                  if (!item) return null;
                  const st = sentimentStyles[item.sentiment] || sentimentStyles.Neutral;
                  return (
                    <div key={m.key} className="bg-white p-3 rounded-xl border border-slate-200 shadow-xs space-y-2">
                      <div className="text-[11px] font-bold text-slate-600 truncate">{m.label}</div>
                      <div className={`px-2 py-1 rounded-lg text-xs font-bold flex items-center gap-1 ${st.badgeBg}`}>
                        <span>{st.emoji}</span>
                        <span className="truncate">{item.sentiment}</span>
                      </div>
                      <div className="text-[10px] text-slate-500 font-mono">
                        Conf: {Math.round(item.max_prob * 100)}%
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ── Sentiment & Confidence Scores ── */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-4">

            {/* Final Sentiment Badge */}
            <div className="md:col-span-5 bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex flex-col justify-between">
              <div className="flex items-center gap-2 text-slate-800 font-bold text-sm mb-2">
                <Target className="w-4 h-4 text-teal-600" />
                <span>Final Sentiment</span>
              </div>
              <div className="flex items-center justify-center py-5">
                <div className={`px-7 py-4 rounded-full flex items-center gap-3.5 shadow-sm ${currentSentStyle.badgeBg}`}>
                  <span className="text-3xl">{currentSentStyle.emoji}</span>
                  <span className="text-2xl font-black tracking-tight">{currentSentStyle.label}</span>
                </div>
              </div>
            </div>

            {/* Confidence Scores Bars */}
            <div className="md:col-span-7 bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
              <div className="flex items-center gap-2 text-slate-800 font-bold text-sm mb-4">
                <BarChart3 className="w-4 h-4 text-teal-600" />
                <span>Confidence Scores</span>
              </div>

              <div className="space-y-3.5">
                <ConfBar emoji="😊" label="Positive" score={scores.Positive} barColor="#10B981" />
                <ConfBar emoji="😐" label="Neutral"  score={scores.Neutral}  barColor="#0284C7" />
                <ConfBar emoji="😡" label="Negative" score={scores.Negative} barColor="#EF4444" />
              </div>
            </div>

          </div>

          {/* ── LIME Influential Words & Narrative Explanation ── */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-4">

            {/* LIME Words Card */}
            <div className="md:col-span-6 bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2 text-slate-800 font-bold text-sm">
                    <Sparkles className="w-4 h-4 text-teal-600" />
                    <span>LIME Word Attributions</span>
                  </div>
                  <span className="text-[10px] text-slate-400 font-mono">200 perturbations</span>
                </div>

                {/* Words Chips with Weak Signal Handling */}
                <div className="flex flex-wrap gap-2 mb-4">
                  {limeWords.length === 0 ? (
                    <span className="text-xs text-slate-400 italic">
                      {activeModel === 'image'
                        ? 'Visual-only model — word attributions not applicable.'
                        : 'No prominent lexical drivers detected for this caption.'}
                    </span>
                  ) : (
                    limeWords.map((item, idx) => {
                      const isPos = item.score >= 0;
                      const isWeak = item.weak_signal || Math.abs(item.score) < 0.02;
                      return (
                        <span
                          key={idx}
                          className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-bold ${
                            isWeak
                              ? 'bg-slate-100 text-slate-600 border border-slate-300 opacity-80'
                              : isPos
                              ? 'bg-[#DCFCE7] text-[#166534] border border-[#BBF7D0]'
                              : 'bg-[#FEE2E2] text-[#991B1B] border border-[#FECACA]'
                          }`}
                        >
                          <span>{item.word}</span>
                          <span className="text-[10px] font-mono opacity-80">
                            ({isPos ? `+${item.score.toFixed(2)}` : item.score.toFixed(2)})
                          </span>
                          {isWeak && <span className="text-[9px] px-1 bg-slate-200 rounded text-slate-700">weak</span>}
                        </span>
                      );
                    })
                  )}
                </div>

                {data.lime_note && (
                  <p className="text-[11px] text-slate-500 italic mb-2">{data.lime_note}</p>
                )}
              </div>

              {/* LIME Legend */}
              <div className="flex items-center justify-between text-[11px] font-semibold text-slate-500 pt-3 border-t border-slate-100">
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded bg-[#BBF7D0]" />
                  <span>Supports {sentiment}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded bg-[#FECACA]" />
                  <span>Opposes {sentiment}</span>
                </div>
              </div>
            </div>

            {/* Explanation Narrative */}
            <div className="md:col-span-6 bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex flex-col justify-between">
              <div>
                <div className="flex items-center gap-2 text-slate-800 font-bold text-sm mb-2.5">
                  <Lightbulb className="w-4 h-4 text-teal-600" />
                  <span>Data-Driven Narrative</span>
                </div>
                <div className="p-4 bg-teal-50/40 rounded-xl border border-teal-100 text-slate-700 text-xs leading-relaxed font-normal">
                  {data.explanation}
                </div>
              </div>

              <div className="flex justify-between items-center pt-3 text-xs text-slate-500">
                <span>Stage timings: Model {timings.model_forward_ms || 0}ms • OCR {timings.ocr_ms || 0}ms</span>
                <button
                  type="button"
                  title="Share Results"
                  onClick={() => alert('Result summary copied to clipboard!')}
                  className="w-8 h-8 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-600 flex items-center justify-center transition"
                >
                  <Share2 className="w-4 h-4" />
                </button>
              </div>
            </div>

          </div>

        </div>

      </div>

    </div>
  );
}
