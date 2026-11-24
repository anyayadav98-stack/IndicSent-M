import React from 'react';
import { Info, UserCheck, Users, Code, BookOpen, ExternalLink } from 'lucide-react';

const TEAM_MEMBERS = [
  { name: 'Anya Yadav',       role: 'NLP & Multimodal Architecture'                },
  { name: 'Pragya Singh',     role: 'Computer Vision & ResNet50 Feature Pipeline'  },
  { name: 'Priyanshu Thakur', role: 'Cross-Attention Fusion & PyTorch Trainer'     },
  { name: 'Ronit Jaiswal',    role: 'Explainable AI (LIME & Grad-CAM) & Backend'   },
  { name: 'Sparshman Gupta',  role: 'Frontend UI/UX & Experimental Evaluation'     },
];

const TECH_STACK = [
  { label: 'Text Encoder',    value: 'XLM-RoBERTa (Base)'      },
  { label: 'Vision Encoder',  value: 'ResNet50 (Pretrained)'   },
  { label: 'Fusion Methods',  value: 'Concat & Cross-Attention' },
  { label: 'Explainable AI',  value: 'LIME + Grad-CAM'         },
  { label: 'Backend',         value: 'FastAPI + Uvicorn'        },
  { label: 'Frontend',        value: 'React + Vite + Tailwind'  },
  { label: 'Visualizations',  value: 'Recharts + Matplotlib'    },
  { label: 'Target Classes',  value: 'Positive / Neutral / Negative' },
];

const avatarColors = [
  'bg-teal-100 text-teal-700',
  'bg-sky-100 text-sky-700',
  'bg-indigo-100 text-indigo-700',
  'bg-amber-100 text-amber-700',
  'bg-rose-100 text-rose-700',
];

export default function About() {
  return (
    <div className="max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">

      {/* Title */}
      <div>
        <h1 className="text-3xl font-black text-slate-900 tracking-tight flex items-center gap-2">
          <Info className="w-7 h-7 text-teal-600" />
          About IndicSent-M
        </h1>
        <p className="text-sm text-slate-500 mt-1">
          Academic Research Project · CSET 346: Natural Language Processing
        </p>
      </div>

      {/* Abstract */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-3">
        <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-teal-600" />
          Project Abstract & Objectives
        </h2>
        <p className="text-sm text-slate-600 leading-relaxed">
          <strong>IndicSent-M</strong> is an end-to-end multimodal deep learning platform engineered to address
          the linguistic and cultural complexities of internet memes in Indian languages
          (Hinglish: Hindi in Roman/Devanagari script mixed with English).
          By bridging multilingual transformer representations from <strong>XLM-RoBERTa</strong> with
          convolutional visual feature maps from <strong>ResNet50</strong> via cross-modal attention,
          the system classifies memes into Positive, Neutral, and Negative sentiment categories
          while explaining every decision via LIME lexical attributions and Grad-CAM visual heatmaps.
        </p>
        <div className="flex flex-wrap gap-2 pt-1">
          {['XLM-RoBERTa', 'ResNet50', 'Cross-Attention', 'LIME', 'Grad-CAM', 'Memotion 3.0'].map(tag => (
            <span key={tag} className="inline-block text-xs font-semibold px-2.5 py-1 rounded-full bg-teal-50 border border-teal-200 text-teal-800">
              {tag}
            </span>
          ))}
        </div>
      </div>

      {/* Faculty + Team */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-6">

        {/* Faculty */}
        <div className="md:col-span-5 bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2 mb-4">
            <UserCheck className="w-4 h-4 text-teal-600" />
            Project Supervision
          </h2>
          <div className="p-4 rounded-2xl bg-gradient-to-br from-teal-50 to-sky-50 border border-teal-100 space-y-1.5">
            <div className="w-12 h-12 rounded-full bg-teal-100 text-teal-700 flex items-center justify-center font-black text-lg mb-2">
              SC
            </div>
            <p className="text-base font-extrabold text-slate-900">Dr. Shubham Choudhary</p>
            <p className="text-xs font-semibold text-teal-700">Faculty Guide & Course Instructor</p>
            <p className="text-xs text-slate-500">Department of Computer Science & Engineering</p>
            <p className="text-xs text-slate-500">CSET 346: Natural Language Processing</p>
          </div>
        </div>

        {/* Team */}
        <div className="md:col-span-7 bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2 mb-4">
            <Users className="w-4 h-4 text-teal-600" />
            Student Research Team
          </h2>
          <div className="space-y-2">
            {TEAM_MEMBERS.map((m, i) => (
              <div key={m.name} className="flex items-center gap-3 p-3 rounded-xl bg-slate-50 border border-slate-100 card-hover">
                <div className={`w-9 h-9 rounded-full flex items-center justify-center font-bold text-sm shrink-0 ${avatarColors[i % avatarColors.length]}`}>
                  {m.name.split(' ').map(n => n[0]).join('')}
                </div>
                <div>
                  <p className="text-sm font-bold text-slate-900">{m.name}</p>
                  <p className="text-[11px] text-teal-700 font-medium">{m.role}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>

      {/* Tech stack */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
        <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
          <Code className="w-4 h-4 text-teal-600" />
          Approved Technical Stack
        </h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          {TECH_STACK.map(({ label, value }) => (
            <div key={label} className="p-3 rounded-xl bg-slate-50 border border-slate-200 card-hover">
              <p className="font-bold text-slate-800">{label}</p>
              <p className="text-slate-500 mt-0.5">{value}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Dataset credit */}
      <div className="text-center text-xs text-slate-400 pb-4">
        Dataset: Memotion 3.0 — AAAI De-Factify 2 Benchmark · Used for academic research only.
        No dataset credentials or passwords are stored in this repository.
      </div>

    </div>
  );
}
