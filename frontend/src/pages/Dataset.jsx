import React, { useState, useEffect } from 'react';
import { Database, AlertTriangle, Loader2 } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend
} from 'recharts';

/* Skeleton block */
function Skeleton({ className = '' }) {
  return <div className={`animate-pulse bg-slate-200 rounded-xl ${className}`} />;
}

export default function Dataset() {
  const [stats,   setStats]   = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(false);

  useEffect(() => {
    fetch('http://localhost:8000/dataset-stats')
      .then(r => r.json())
      .then(d => { setStats(d); setLoading(false); })
      .catch(() => {
        // Use known correct numbers from preprocessing output
        setStats({
          total_labelled: 8442,
          splits: {
            train:      { total: 6961, distribution: { Positive: 2268, Neutral: 2955, Negative: 1738 } },
            validation: { total: 740,  distribution: { Positive: 168,  Neutral: 286,  Negative: 286  } },
            test:       { total: 741,  distribution: { Positive: 167,  Neutral: 287,  Negative: 287  } },
            blind_test: { total: 1484 },
          },
        });
        setLoading(false);
        setError(true);
      });
  }, []);

  const statCards = stats ? [
    { label: 'Train Set (Cleaned)',   value: stats.splits?.train?.total?.toLocaleString()      || '6,961', sub: '82.5% of labelled data' },
    { label: 'Validation Set',        value: stats.splits?.validation?.total?.toLocaleString() || '740',   sub: 'Stratified 50% split'  },
    { label: 'Test Set',              value: stats.splits?.test?.total?.toLocaleString()       || '741',   sub: 'Stratified 50% split'  },
    { label: 'Blind Challenge Test',  value: stats.splits?.blind_test?.total?.toLocaleString() || '1,484', sub: 'Held-out official set'  },
  ] : null;

  const chartData = stats ? [
    { split: 'Train (6,961)',  Positive: stats.splits.train?.distribution?.Positive,      Neutral: stats.splits.train?.distribution?.Neutral,      Negative: stats.splits.train?.distribution?.Negative      },
    { split: 'Val (740)',      Positive: stats.splits.validation?.distribution?.Positive, Neutral: stats.splits.validation?.distribution?.Neutral, Negative: stats.splits.validation?.distribution?.Negative },
    { split: 'Test (741)',     Positive: stats.splits.test?.distribution?.Positive,       Neutral: stats.splits.test?.distribution?.Neutral,       Negative: stats.splits.test?.distribution?.Negative       },
  ] : null;

  return (
    <div className="max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">

      {/* Title */}
      <div>
        <h1 className="text-3xl font-black text-slate-900 tracking-tight flex items-center gap-2">
          <Database className="w-7 h-7 text-teal-600" />
          Memotion 3.0 Dataset Explorer
        </h1>
        <p className="text-sm text-slate-500 mt-1">
          Benchmark statistics, 3-class sentiment distributions, and dataset characteristics.
        </p>
      </div>

      {/* Backend unreachable banner */}
      {error && (
        <div className="flex items-center gap-3 bg-amber-50 border border-amber-200 text-amber-900 rounded-xl px-4 py-3 text-sm">
          <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
          Backend offline — showing preprocessed counts from <code className="text-xs bg-amber-100 px-1 rounded">data/processed/</code>.
        </div>
      )}

      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {loading ? (
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-2">
              <Skeleton className="h-3 w-28" />
              <Skeleton className="h-8 w-16" />
              <Skeleton className="h-3 w-20" />
            </div>
          ))
        ) : statCards.map(({ label, value, sub }) => (
          <div key={label} className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm card-hover">
            <p className="text-xs font-semibold text-slate-500">{label}</p>
            <p className="text-3xl font-black text-teal-700 mt-1">{value}</p>
            <p className="text-[11px] text-slate-400 font-medium mt-1">{sub}</p>
          </div>
        ))}
      </div>

      {/* Distribution chart */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-bold text-slate-900">Class Distribution Across Splits</h2>
        </div>
        {loading ? (
          <Skeleton className="h-64 w-full" />
        ) : (
          <div className="h-72 w-full pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                <XAxis dataKey="split" tick={{ fill: '#475569', fontSize: 12, fontWeight: 600 }} />
                <YAxis tick={{ fill: '#64748B', fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: '#1E293B', borderRadius: '12px', border: 'none', color: '#fff', fontSize: 12 }} />
                <Legend wrapperStyle={{ paddingTop: 12, fontSize: 12 }} />
                <Bar dataKey="Positive" fill="#10B981" radius={[4,4,0,0]} />
                <Bar dataKey="Neutral"  fill="#0284C7" radius={[4,4,0,0]} />
                <Bar dataKey="Negative" fill="#EF4444" radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Challenges */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
        <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-amber-500" />
          Multimodal NLP Challenges in Memotion 3
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            {
              title: 'Code-Mixed Hinglish',
              body:  'Captions blend English vocabulary with Hindi grammar in Latin and Devanagari scripts, generating OOV challenges for monolingual English BERT models.',
            },
            {
              title: 'Class Imbalance',
              body:  'Neutral and Negative classes together comprise ~77% of test memes. Addressed through Class-Weighted Cross-Entropy Loss during training.',
            },
            {
              title: 'Visual Sarcasm & Dissonance',
              body:  'Text captions frequently contradict visual expressions, making unimodal classification unreliable and necessitating cross-modal attention fusion.',
            },
          ].map(({ title, body }) => (
            <div key={title} className="p-4 rounded-xl bg-slate-50 border border-slate-200 card-hover">
              <h3 className="font-bold text-slate-800 text-sm mb-1">{title}</h3>
              <p className="text-xs text-slate-600 leading-relaxed">{body}</p>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
