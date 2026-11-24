import React, { useState, useEffect } from 'react';
import {
  Home as HomeIcon, Upload, BarChart3, GitCompare,
  Database, Info, Menu, X
} from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 4);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    let isMounted = true;
    const checkHealth = () => {
      fetch('http://localhost:8000/health', { method: 'GET' })
        .then((res) => {
          if (isMounted) setBackendOnline(res.ok);
        })
        .catch(() => {
          if (isMounted) setBackendOnline(false);
        });
    };
    checkHealth();
    const interval = setInterval(checkHealth, 4000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const navItems = [
    { id: 'home',    label: 'Home',    icon: HomeIcon   },
    { id: 'analyze', label: 'Analyze', icon: Upload     },
    { id: 'results', label: 'Results', icon: BarChart3  },
    { id: 'compare', label: 'Compare', icon: GitCompare },
    { id: 'dataset', label: 'Dataset', icon: Database   },
    { id: 'about',   label: 'About',   icon: Info       },
  ];

  const goto = (id) => { setActiveTab(id); setMobileOpen(false); };

  return (
    <header className={`bg-white sticky top-0 z-50 border-b border-slate-200 transition-all ${scrolled ? 'shadow-md' : 'shadow-sm'}`}>
      <div className="max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">

        {/* ── Logo & Subtitle matching Reference Design ── */}
        <div className="flex items-center gap-3 select-none shrink-0">
          <button
            onClick={() => goto('home')}
            className="flex items-center text-left"
          >
            <span className="text-2xl sm:text-[1.75rem] font-black tracking-tight text-slate-900 font-sans">
              IndicSent-<span className="text-[#0D9488]">M</span>
            </span>
          </button>
          <span className="hidden sm:inline-block text-slate-300 text-lg font-light">|</span>
          <span className="hidden sm:inline-block text-xs md:text-sm font-medium text-slate-600 truncate">
            Multimodal Sentiment Analysis for Indian Languages
          </span>
        </div>

        {/* ── Desktop Navigation matching Reference Teal Pill ── */}
        <div className="hidden md:flex items-center gap-4">
          <div
            className="flex items-center gap-2 px-2.5 py-1 rounded-full text-xs font-semibold border transition-all"
            style={{
              backgroundColor: backendOnline ? '#F0FDF4' : '#FEF2F2',
              borderColor: backendOnline ? '#BBF7D0' : '#FECACA',
              color: backendOnline ? '#15803D' : '#B91C1C'
            }}
            title={backendOnline ? "FastAPI backend running on port 8000" : "Backend unreachable (run start_all.bat)"}
          >
            <span
              className={`w-2 h-2 rounded-full ${backendOnline ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`}
            />
            <span className="text-[11px] font-mono">
              {backendOnline ? 'API Online' : 'API Offline'}
            </span>
          </div>

          <nav className="flex items-center gap-1.5 lg:gap-2">
            {navItems.map(({ id, label, icon: Icon }) => {
              const active = activeTab === id;
              return (
                <button
                  key={id}
                  onClick={() => goto(id)}
                  className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs lg:text-sm font-medium transition-all ${
                    active
                      ? 'bg-[#0D7A73] text-white shadow-sm font-semibold'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${active ? 'text-white' : 'text-slate-500'}`} />
                  <span>{label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* ── Mobile Hamburger ── */}
        <button
          className="md:hidden p-2 rounded-lg text-slate-600 hover:bg-slate-100 transition"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle navigation menu"
        >
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {/* ── Mobile Menu ── */}
      {mobileOpen && (
        <div className="md:hidden border-t border-slate-100 bg-white py-2 px-4 shadow-lg">
          {navItems.map(({ id, label, icon: Icon }) => {
            const active = activeTab === id;
            return (
              <button
                key={id}
                onClick={() => goto(id)}
                className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition mb-1 ${
                  active
                    ? 'bg-teal-50 text-teal-800 font-bold border border-teal-200'
                    : 'text-slate-700 hover:bg-slate-50'
                }`}
              >
                <Icon className={`w-4 h-4 ${active ? 'text-teal-600' : 'text-slate-400'}`} />
                <span>{label}</span>
              </button>
            );
          })}
        </div>
      )}
    </header>
  );
}
