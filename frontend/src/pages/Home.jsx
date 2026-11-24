import React, { useState, useEffect } from 'react';
import {
  ArrowRight, Sparkles, Brain, Eye, Layers,
  ChevronRight, RefreshCw, AlertCircle, CheckCircle2,
  FileText, Database, GitBranch
} from 'lucide-react';

/* ─── Real Test Samples from Memotion 3 test set ─── */
const REAL_SAMPLES = [
  {
    id: '244',
    image: '/samples/244.jpg',
    text: 'Wo bhi theek ho jayega',
    sentiment: 'Neutral',
    badgeClass: 'bg-amber-100 border-amber-300 text-amber-900',
    emoji: '😐',
    category: 'Relatable Hindi phrase'
  },
  {
    id: '754',
    image: '/samples/754.jpg',
    text: 'We all need a friend like Meena of Kota Factory',
    sentiment: 'Positive',
    badgeClass: 'bg-emerald-100 border-emerald-300 text-emerald-900',
    emoji: '😊',
    category: 'Friendship & Pop-culture'
  },
  {
    id: '378',
    image: '/samples/378.jpg',
    text: "Me: I'm feeling very sleepy i should go to bed. Also me after lying on the bed for an hour: TheNewRon",
    sentiment: 'Negative',
    badgeClass: 'bg-rose-100 border-rose-300 text-rose-900',
    emoji: '😡',
    category: 'Insomnia & Late night struggle'
  }
];

/* ─── Sample preview card with REAL MEME from Memotion 3 test set ─── */
function RealSamplePreviewCard() {
  const [activeIdx, setActiveIdx] = useState(0);
  const sample = REAL_SAMPLES[activeIdx];

  return (
    <div className="relative bg-white rounded-2xl border border-slate-200 shadow-md overflow-hidden transition-all">
      {/* Header bar */}
      <div className="bg-slate-900 text-white px-4 py-2.5 flex items-center justify-between text-xs">
        <span className="font-semibold tracking-wide flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          Sample from Memotion 3 test set (ground-truth label)
        </span>
        <div className="flex items-center gap-1">
          {REAL_SAMPLES.map((s, idx) => (
            <button
              key={s.id}
              onClick={() => setActiveIdx(idx)}
              className={`px-2 py-0.5 rounded text-[11px] font-mono transition ${
                activeIdx === idx
                  ? 'bg-teal-600 text-white font-bold'
                  : 'bg-slate-800 text-slate-400 hover:text-white'
              }`}
            >
              #{idx + 1}
            </button>
          ))}
        </div>
      </div>

      {/* Real Meme Image */}
      <div className="p-4 bg-slate-950 flex flex-col items-center justify-center min-h-[240px] max-h-[300px]">
        <img
          src={sample.image}
          alt={`Memotion 3 Sample #${sample.id}`}
          className="max-h-[250px] w-auto max-w-full rounded-xl object-contain shadow"
        />
      </div>

      {/* Caption & Ground-Truth Details */}
      <div className="p-4 border-t border-slate-100 space-y-3 bg-white">
        <div>
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
            Caption (Hinglish OCR):
          </span>
          <p className="text-sm font-semibold text-slate-800 italic mt-0.5 line-clamp-2">
            "{sample.text}"
          </p>
        </div>

        <div className="pt-2 border-t border-slate-100 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 font-medium">Ground-Truth:</span>
            <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border ${sample.badgeClass}`}>
              <span>{sample.emoji}</span>
              <span>{sample.sentiment}</span>
            </span>
          </div>
          <span className="text-[11px] font-mono text-slate-400">
            ID: {sample.id}.jpg
          </span>
        </div>
      </div>
    </div>
  );
}

/* ─── Workflow step ─── */
function WorkflowStep({ num, title, desc, color, active }) {
  return (
    <div className={`card-hover flex flex-col items-center text-center p-6 rounded-2xl border transition-all duration-200 ${
      active
        ? 'bg-teal-700 border-teal-600 text-white shadow-lg'
        : 'bg-white border-slate-200 hover:border-teal-300 hover:shadow-md'
    }`}>
      <div className={`w-12 h-12 rounded-2xl flex items-center justify-center font-black text-lg mb-4 shadow-sm ${
        active ? 'bg-white/20 text-white' : `${color} `
      }`}>
        {num}
      </div>
      <h3 className={`font-bold text-base mb-1.5 ${active ? 'text-white' : 'text-slate-900'}`}>{title}</h3>
      <p className={`text-xs leading-relaxed ${active ? 'text-teal-100' : 'text-slate-500'}`}>{desc}</p>
    </div>
  );
}

export default function Home({ onNavigate }) {
  const [stats, setStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [statsError, setStatsError] = useState(false);

  const fetchStats = () => {
    setStatsLoading(true);
    setStatsError(false);
    fetch('http://localhost:8000/dataset-stats')
      .then(r => {
        if (!r.ok) throw new Error('API returned ' + r.status);
        return r.json();
      })
      .then(d => {
        setStats(d);
        setStatsLoading(false);
        setStatsError(false);
      })
      .catch(() => {
        setStatsLoading(false);
        setStatsError(true);
      });
  };

  useEffect(() => {
    fetchStats();
  }, []);

  const whyCards = [
    {
      icon: Brain,
      color: 'bg-teal-100 text-teal-800',
      title: 'Code-Mixed Hinglish',
      body: 'Indian memes blend Roman and Devanagari Hindi with English. Standard monolingual NLP fails on subwords like "theek" or "maloom". XLM-RoBERTa operates across 100 languages.'
    },
    {
      icon: Layers,
      color: 'bg-sky-100 text-sky-800',
      title: 'Visual Sarcasm & Dissonance',
      body: 'A happy sentence paired with a crying face reverses polarity. Neither text nor image alone can resolve irony—cross-modal attention bridges the multimodal semantic gap.'
    },
    {
      icon: Eye,
      color: 'bg-amber-100 text-amber-800',
      title: 'Dual Explainability',
      body: 'Black-box predictions fail academic defense. We combine LIME lexical attribution with Grad-CAM visual heatmaps so every positive, neutral, or negative decision is provable.'
    }
  ];

  const xaiCards = [
    {
      icon: Sparkles,
      color: 'bg-emerald-100 text-emerald-800',
      title: 'LIME Token Attributions',
      body: 'Generates local surrogate perturbations to calculate signed influence scores for every code-mixed word in the caption.',
      tag: 'Text Saliency'
    },
    {
      icon: Eye,
      color: 'bg-blue-100 text-blue-800',
      title: 'Grad-CAM Visual Heatmaps',
      body: 'Gradients from the predicted class flow back into ResNet50 layer4 to pinpoint which facial expressions or meme regions swayed the model.',
      tag: 'Vision Saliency'
    }
  ];

  return (
    <div className="space-y-20 py-10">

      {/* ── Hero Section (Two columns) ── */}
      <section className="max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-center">

          {/* Left Column (60px headline, subtitle, buttons) */}
          <div className="lg:col-span-7 space-y-6">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-teal-50 border border-teal-200 text-teal-800 text-xs font-semibold">
              <Sparkles className="w-3.5 h-3.5 text-teal-600" />
              <span>AAAI De-Factify 2 Benchmark • Memotion 3.0 Dataset</span>
            </div>

            <h1 className="text-4xl sm:text-5xl lg:text-[3.5rem] font-black text-slate-900 tracking-tight leading-[1.08]">
              Multimodal Sentiment Analysis for{' '}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-600 via-teal-700 to-sky-600">
                Indian Languages
              </span>
            </h1>

            <p className="text-base sm:text-lg text-slate-600 leading-relaxed max-w-[560px]">
              Decoding the nuanced emotional synergy of Internet memes combining code-mixed Hinglish
              text with expressive visual imagery through XLM-RoBERTa and ResNet50 cross-modal attention.
            </p>

            <div className="flex flex-wrap gap-4 pt-2">
              <button
                onClick={() => onNavigate('analyze')}
                className="inline-flex items-center gap-2 px-7 py-3.5 rounded-xl bg-teal-700 hover:bg-teal-800 active:scale-95 text-white font-bold text-sm shadow-md hover:shadow-lg transition-all"
              >
                <span>Analyze a Meme</span>
                <ArrowRight className="w-4 h-4" />
              </button>
              <button
                onClick={() => onNavigate('compare')}
                className="inline-flex items-center gap-2 px-6 py-3.5 rounded-xl bg-white hover:bg-slate-50 active:scale-95 text-slate-700 font-semibold text-sm border border-slate-200 shadow-sm hover:shadow transition-all"
              >
                View Model Benchmarks
              </button>
            </div>
          </div>

          {/* Right Column: Preview card with REAL meme from Memotion 3 */}
          <div className="lg:col-span-5">
            <RealSamplePreviewCard />
          </div>

        </div>
      </section>

      {/* ── Stats Strip under Hero ── */}
      <section className="max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8">
        {statsError ? (
          <div className="bg-amber-50 border border-amber-300 rounded-2xl p-6 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <AlertCircle className="w-6 h-6 text-amber-600 shrink-0" />
              <div>
                <p className="font-bold text-slate-900 text-sm">Backend not running</p>
                <p className="text-xs text-slate-600 mt-0.5">
                  Start the FastAPI backend with <code className="bg-white px-2 py-0.5 rounded border border-amber-200 text-amber-800 font-mono text-[11px]">python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload</code> to view live benchmark metrics.
                </p>
              </div>
            </div>
            <button
              onClick={fetchStats}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-teal-700 hover:bg-teal-800 text-white font-semibold text-xs transition shrink-0"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry Connection</span>
            </button>
          </div>
        ) : statsLoading ? (
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 grid grid-cols-2 md:grid-cols-4 gap-4 animate-pulse">
            {[1, 2, 3, 4].map(n => (
              <div key={n} className="flex flex-col items-center gap-2">
                <div className="h-8 w-20 bg-slate-200 rounded"></div>
                <div className="h-4 w-28 bg-slate-100 rounded"></div>
              </div>
            ))}
          </div>
        ) : (
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm grid grid-cols-2 md:grid-cols-4 divide-x divide-y md:divide-y-0 divide-slate-100">
            <div className="flex flex-col items-center justify-center py-6 px-4 text-center">
              <span className="text-3xl font-black text-teal-700 tracking-tight">
                {stats?.splits?.train?.total?.toLocaleString() || '6,961'}
              </span>
              <span className="text-xs font-semibold text-slate-800 mt-1">Train Samples</span>
              <span className="text-[11px] text-slate-400 mt-0.5">Memotion 3 Cleaned</span>
            </div>

            <div className="flex flex-col items-center justify-center py-6 px-4 text-center">
              <span className="text-3xl font-black text-teal-700 tracking-tight">
                {((stats?.splits?.validation?.total || 740) + (stats?.splits?.test?.total || 741)).toLocaleString()}
              </span>
              <span className="text-xs font-semibold text-slate-800 mt-1">Val + Test Samples</span>
              <span className="text-[11px] text-slate-400 mt-0.5">Stratified Benchmark</span>
            </div>

            <div className="flex flex-col items-center justify-center py-6 px-4 text-center">
              <span className="text-3xl font-black text-teal-700 tracking-tight">3</span>
              <span className="text-xs font-semibold text-slate-800 mt-1">Sentiment Classes</span>
              <span className="text-[11px] text-slate-400 mt-0.5">Positive, Neutral, Negative</span>
            </div>

            <div className="flex flex-col items-center justify-center py-6 px-4 text-center">
              <span className="text-3xl font-black text-teal-700 tracking-tight">4</span>
              <span className="text-xs font-semibold text-slate-800 mt-1">Model Architectures</span>
              <span className="text-[11px] text-slate-400 mt-0.5">Text, Image, Concat, Attention</span>
            </div>
          </div>
        )}
      </section>

      {/* ── Workflow Section (4 bigger cards + connecting flow) ── */}
      <section className="max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8">
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-8 sm:p-10">
          <div className="text-center mb-10">
            <h2 className="text-2xl sm:text-3xl font-black text-slate-900 tracking-tight">
              End-to-End System Workflow
            </h2>
            <p className="text-sm text-slate-500 mt-2 max-w-xl mx-auto">
              Dual-stream feature extraction with cross-modal fusion and interpretability
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 relative">
            <WorkflowStep
              num="1"
              title="Multimodal Input"
              desc="Hinglish meme image (224×224) + embedded caption text from social media."
              color="bg-sky-100 text-sky-800"
              active={false}
            />

            <WorkflowStep
              num="2"
              title="Feature Extraction"
              desc="XLM-RoBERTa (768-dim text sequence) + ResNet50 (2048-dim visual patch maps)."
              color="bg-indigo-100 text-indigo-800"
              active={false}
            />

            <WorkflowStep
              num="3"
              title="Cross-Attention Fusion"
              desc="Text tokens dynamically query 49 visual patches to link words with facial cues."
              color="bg-teal-100 text-teal-800"
              active={true}
            />

            <WorkflowStep
              num="4"
              title="Sentiment & XAI"
              desc="3-class polarity (Pos/Neu/Neg) + LIME lexical chips + Grad-CAM heatmap overlay."
              color="bg-emerald-100 text-emerald-800"
              active={false}
            />
          </div>
        </div>
      </section>

      {/* ── Why Multimodal? Section (3 cards) ── */}
      <section className="max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-10">
          <h2 className="text-2xl sm:text-3xl font-black text-slate-900 tracking-tight">
            Why Multimodal?
          </h2>
          <p className="text-sm text-slate-500 mt-2">
            Neither text nor imagery in isolation can resolve the semantic dissonance of memes
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {whyCards.map(({ icon: Icon, color, title, body }) => (
            <div key={title} className="card card-hover p-6 rounded-2xl space-y-3 bg-white border border-slate-200">
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${color}`}>
                <Icon className="w-6 h-6" />
              </div>
              <h3 className="font-bold text-slate-900 text-lg">{title}</h3>
              <p className="text-sm text-slate-600 leading-relaxed">{body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Explainable Predictions Section (LIME + Grad-CAM) ── */}
      <section className="max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-10">
          <h2 className="text-2xl sm:text-3xl font-black text-slate-900 tracking-tight">
            Explainable Predictions
          </h2>
          <p className="text-sm text-slate-500 mt-2">
            Dual XAI techniques give transparent rationale for every sentiment prediction
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {xaiCards.map(({ icon: Icon, color, title, body, tag }) => (
            <div key={title} className="card card-hover p-7 rounded-2xl bg-white border border-slate-200 flex flex-col justify-between">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${color}`}>
                    <Icon className="w-6 h-6" />
                  </div>
                  <span className="text-[11px] font-bold px-3 py-1 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
                    {tag}
                  </span>
                </div>
                <h3 className="font-bold text-slate-900 text-lg">{title}</h3>
                <p className="text-sm text-slate-600 leading-relaxed">{body}</p>
              </div>

              <div className="pt-4 mt-4 border-t border-slate-100 flex items-center justify-between">
                <button
                  onClick={() => onNavigate('analyze')}
                  className="inline-flex items-center gap-1.5 text-xs font-bold text-teal-700 hover:text-teal-900"
                >
                  <span>Test in Live Analyzer</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>

    </div>
  );
}
