import React, { useState, useEffect } from 'react';
import { Upload, Sparkles, Loader2, History, AlertCircle, X, ArrowRight, HelpCircle, Eye, Sliders, Layers } from 'lucide-react';

/* ─── 3 Clean Real Test Memes from Memotion 3 ─── */
const SAMPLE_MEMES = [
  {
    id: '244',
    title: 'Neutral Relatable Phrase',
    image: '/samples/244.jpg',
    text: 'Wo bhi theek ho jayega',
    sentiment: 'Neutral',
    badgeClass: 'bg-amber-100 text-amber-900 border-amber-300',
    emoji: '😐'
  },
  {
    id: '754',
    title: 'Positive Friendship Meme',
    image: '/samples/754.jpg',
    text: 'We all need a friend like Meena of Kota Factory',
    sentiment: 'Positive',
    badgeClass: 'bg-emerald-100 text-emerald-900 border-emerald-300',
    emoji: '😊'
  },
  {
    id: '378',
    title: 'Negative Sleep Struggle',
    image: '/samples/378.jpg',
    text: "Me: I'm feeling very sleepy i should go to bed. Also me after lying on the bed for an hour: TheNewRon",
    sentiment: 'Negative',
    badgeClass: 'bg-rose-100 text-rose-900 border-rose-300',
    emoji: '😡'
  }
];

export default function Analyze({ onAnalysisComplete }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [caption, setCaption] = useState('');
  const [loading, setLoading] = useState(false);
  const [historyItems, setHistoryItems] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState(null);
  const [modelType, setModelType] = useState('auto');
  const [useAiOcr, setUseAiOcr] = useState(false);

  useEffect(() => {
    fetch('http://localhost:8000/history')
      .then(r => r.json())
      .then(d => setHistoryItems(d))
      .catch(() => {
        setHistoryItems([]);
      });
  }, []);

  const selectSampleMeme = async (sample) => {
    setCaption(sample.text);
    setPreviewUrl(sample.image);
    setError(null);

    try {
      const res = await fetch(sample.image);
      const blob = await res.blob();
      const file = new File([blob], `${sample.id}.jpg`, { type: 'image/jpeg' });
      setSelectedFile(file);
    } catch {
      setSelectedFile(null);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    if (!caption.trim() && !selectedFile) {
      setError('Please upload a meme image or enter code-mixed caption text.');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    if (selectedFile) formData.append('file', selectedFile);
    if (caption.trim()) formData.append('text', caption.trim());
    formData.append('model_type', modelType);
    formData.append('use_ai_ocr', useAiOcr);
    formData.append('include_explanations', true);

    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Prediction service returned status ${res.status}`);
      }

      const result = await res.json();
      onAnalysisComplete(result);
    } catch (err) {
      console.error('Backend predict failed:', err);
      setError('Backend is not running, run start_all.bat');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">

      {/* Title */}
      <div className="text-center max-w-2xl mx-auto space-y-2">
        <h1 className="text-3xl sm:text-4xl font-black text-slate-900 tracking-tight">
          Analyze Meme Sentiment
        </h1>
        <p className="text-sm text-slate-600">
          Upload an Indian internet meme and optional caption to run dual-stream multimodal inference with
          XLM-RoBERTa, ResNet50, LIME, and Grad-CAM.
        </p>
      </div>

      {/* ── Try a Sample Meme Gallery ── */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-teal-600" />
            <span className="text-sm font-bold text-slate-900">
              Try a Sample Meme from Memotion 3 Test Set
            </span>
          </div>
          <span className="text-xs text-slate-400 font-medium">Click any sample to load</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {SAMPLE_MEMES.map((sample) => (
            <div
              key={sample.id}
              onClick={() => selectSampleMeme(sample)}
              className="p-3.5 rounded-2xl border border-slate-200 bg-slate-50 hover:bg-teal-50/50 hover:border-teal-300 transition-all cursor-pointer flex flex-col justify-between group shadow-sm hover:shadow"
            >
              <div className="space-y-2.5">
                <div className="relative rounded-xl overflow-hidden bg-slate-900 h-44 flex items-center justify-center">
                  <img
                    src={sample.image}
                    alt={sample.title}
                    className="max-h-full w-auto object-contain group-hover:scale-105 transition duration-300"
                  />
                  <div className="absolute top-2 right-2">
                    <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-bold border shadow-sm ${sample.badgeClass}`}>
                      <span>{sample.emoji}</span>
                      <span>{sample.sentiment}</span>
                    </span>
                  </div>
                </div>

                <div>
                  <p className="text-xs font-bold text-slate-800 group-hover:text-teal-900">
                    {sample.title}
                  </p>
                  <p className="text-[11px] text-slate-500 italic mt-0.5 line-clamp-2">
                    "{sample.text}"
                  </p>
                </div>
              </div>

              <div className="pt-3 mt-2 border-t border-slate-200/60 flex items-center justify-between text-[11px] font-semibold text-teal-700">
                <span>Load into Analyzer</span>
                <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition" />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Main Analysis Form ── */}
      <form onSubmit={handleSubmit} className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-6">

        {error && (
          <div className="flex flex-wrap items-center justify-between gap-3 bg-rose-50 border border-rose-200 text-rose-800 text-sm rounded-xl px-4 py-3">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-rose-600 shrink-0" />
              <span className="font-medium">{error}</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleSubmit}
                className="px-3 py-1 bg-rose-600 text-white text-xs font-semibold rounded-lg hover:bg-rose-700 transition"
              >
                Retry
              </button>
              <button
                type="button"
                onClick={() => setError(null)}
                className="text-rose-500 hover:text-rose-800"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

          {/* Upload Dropzone */}
          <div className="space-y-2">
            <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
              1. Meme Image (Drag & Drop or Browse)
            </label>

            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => document.getElementById('meme-file-input').click()}
              className={`border-2 border-dashed rounded-2xl p-6 text-center transition cursor-pointer flex flex-col items-center justify-center min-h-[220px] ${
                dragActive
                  ? 'border-teal-500 bg-teal-50/50'
                  : 'border-slate-300 hover:border-teal-400 bg-slate-50/50'
              }`}
            >
              <input
                id="meme-file-input"
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileChange}
              />

              {previewUrl ? (
                <div className="space-y-2 w-full flex flex-col items-center">
                  <img
                    src={previewUrl}
                    alt="Meme Upload Preview"
                    className="max-h-48 rounded-xl object-contain shadow-sm border border-slate-200"
                  />
                  <p className="text-xs text-teal-700 font-semibold">Click or drop another image to replace</p>
                </div>
              ) : (
                <div className="space-y-2 py-4">
                  <div className="w-14 h-14 rounded-full bg-teal-50 text-teal-600 flex items-center justify-center mx-auto shadow-sm">
                    <Upload className="w-6 h-6" />
                  </div>
                  <p className="text-sm font-bold text-slate-800">Drop your meme image here, or browse</p>
                  <p className="text-xs text-slate-400">PNG, JPG, WEBP • Auto-resized to 224×224 for ResNet50</p>
                </div>
              )}
            </div>
          </div>

          {/* Caption Input */}
          <div className="space-y-2 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
                  2. Meme Caption / OCR Text
                </label>
                <div className="flex items-center gap-1.5 text-[11px] text-slate-500">
                  <Sliders className="w-3.5 h-3.5 text-teal-600" />
                  <span>Leave blank for automatic multi-pass OCR</span>
                </div>
              </div>
              <textarea
                rows={6}
                value={caption}
                onChange={(e) => setCaption(e.target.value)}
                placeholder="e.g. When you complete all your assignments but still have more due tomorrow&#10;or: Wo bhi theek ho jayega (Leave blank to run auto OCR)"
                className="w-full rounded-2xl border border-slate-300 p-4 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent transition resize-none leading-relaxed"
              />
            </div>

            {/* Optional Vision LLM OCR Toggle */}
            <div className="flex items-center justify-between pt-2 border-t border-slate-100 text-xs">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useAiOcr}
                  onChange={(e) => setUseAiOcr(e.target.checked)}
                  className="rounded text-teal-600 focus:ring-teal-500"
                />
                <span className="font-medium text-slate-700">AI Vision OCR (Gemini API)</span>
              </label>
              <span className="text-[10px] text-slate-400 italic">
                * Privacy note: image is sent to external API if key is set
              </span>
            </div>
          </div>

        </div>

        {/* ── Architecture Selector ── */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
              <span>3. Model Architecture</span>
              <span className="group relative cursor-pointer">
                <HelpCircle className="w-3.5 h-3.5 text-slate-400" />
                <span className="invisible group-hover:visible absolute z-10 w-72 p-2.5 mt-1 text-xs text-white bg-slate-800 rounded-xl shadow-lg -left-10">
                  Auto dynamically selects the architecture with the highest validation Macro F1 score on Memotion 3 (Concat Fusion).
                </span>
              </span>
            </label>
            <span className="text-xs text-teal-700 font-semibold">
              Default: Auto (best by validation Macro F1)
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-2">
            {[
              { value: 'auto', label: 'Auto (Best Val)', desc: 'Concat Fusion (34.59% F1)' },
              { value: 'concat', label: 'Concat Fusion', desc: 'XLM-R + ResNet50 MLP' },
              { value: 'attention', label: 'Cross-Attention', desc: 'Spatial Patch Attention' },
              { value: 'text', label: 'Text-Only', desc: 'XLM-RoBERTa Base' },
              { value: 'image', label: 'Image-Only', desc: 'Deep ResNet50 GAP' },
              { value: 'ensemble', label: 'Ensemble', desc: '4-Model Probability Avg' },
              { value: 'compare_all', label: 'Compare All', desc: 'All 4 Models Side-by-Side' },
            ].map(opt => (
              <button
                key={opt.value}
                type="button"
                onClick={() => setModelType(opt.value)}
                className={`p-2.5 rounded-xl border text-left transition-all ${
                  modelType === opt.value
                    ? 'bg-teal-700 text-white border-teal-700 shadow-sm'
                    : 'bg-white text-slate-700 border-slate-300 hover:border-teal-400 hover:bg-teal-50/50'
                }`}
              >
                <div className="font-bold text-xs truncate">{opt.label}</div>
                <div className={`text-[10px] mt-0.5 truncate ${modelType === opt.value ? 'text-teal-100' : 'text-slate-400'}`}>
                  {opt.desc}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading}
          className="w-full py-4 px-6 rounded-2xl bg-teal-700 hover:bg-teal-800 disabled:bg-slate-300 disabled:cursor-not-allowed active:scale-[0.99] text-white font-bold text-base shadow-md hover:shadow-lg transition-all flex items-center justify-center gap-2.5"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Running Neural Inference, OCR & Saliency Explainers...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5" />
              <span>
                {modelType === 'compare_all' ? 'Run All Models & Compare Side-by-Side' : `Analyze Sentiment (${modelType.toUpperCase()})`}
              </span>
            </>
          )}
        </button>

      </form>

      {/* ── Recent Analysis History ── */}
      {historyItems.length > 0 && (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-4">
          <div className="flex items-center gap-2 text-sm font-bold text-slate-800">
            <History className="w-4 h-4 text-teal-600" />
            <span>Recent Analysis History</span>
          </div>

          <div className="divide-y divide-slate-100">
            {historyItems.slice(0, 5).map((item, idx) => (
              <div
                key={idx}
                onClick={() => onAnalysisComplete(item)}
                className="py-3 px-2 flex items-center justify-between hover:bg-slate-50 rounded-xl cursor-pointer transition gap-4"
              >
                <div className="flex-1 pr-4">
                  <p className="text-xs font-medium text-slate-800 italic line-clamp-1">
                    "{item.extracted_text}"
                  </p>
                </div>
                <span className={`text-[11px] font-bold px-3 py-1 rounded-full border ${
                  item.final_sentiment === 'Positive' ? 'bg-emerald-100 text-emerald-800 border-emerald-300' :
                  item.final_sentiment === 'Negative' ? 'bg-rose-100 text-rose-800 border-rose-300' :
                  'bg-amber-100 text-amber-800 border-amber-300'
                }`}>
                  {item.final_sentiment}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  );
}
