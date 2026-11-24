import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Home    from './pages/Home';
import Analyze from './pages/Analyze';
import Results from './pages/Results';
import Compare from './pages/Compare';
import Dataset from './pages/Dataset';
import About   from './pages/About';

function App() {
  const getInitialTab = () => {
    try {
      const hash = window.location.hash.replace('#', '').toLowerCase();
      if (['home', 'analyze', 'results', 'compare', 'dataset', 'about'].includes(hash)) {
        return hash;
      }
      const params = new URLSearchParams(window.location.search);
      const tab = params.get('tab')?.toLowerCase();
      if (tab && ['home', 'analyze', 'results', 'compare', 'dataset', 'about'].includes(tab)) {
        return tab;
      }
    } catch (e) {
      // fallback
    }
    return 'home';
  };

  const [activeTab, setActiveTab] = useState(getInitialTab);
  const [analysisResult, setAnalysisResult] = useState(null);

  const changeTab = (tab) => {
    setActiveTab(tab);
    try {
      window.location.hash = tab;
    } catch (e) {}
  };

  const handleAnalysisComplete = (result) => {
    setAnalysisResult(result);
    changeTab('results');
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans">
      <Navbar activeTab={activeTab} setActiveTab={changeTab} />

      <main className="flex-1">
        {activeTab === 'home'    && <Home    onNavigate={changeTab} />}
        {activeTab === 'analyze' && <Analyze onAnalysisComplete={handleAnalysisComplete} />}
        {activeTab === 'results' && (
          <Results
            result={analysisResult}
            onBackToAnalyze={() => changeTab('analyze')}
          />
        )}
        {activeTab === 'compare' && <Compare />}
        {activeTab === 'dataset' && <Dataset />}
        {activeTab === 'about'   && <About   />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-8 mt-10">
        <div className="max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="text-center md:text-left">
              <p className="font-bold text-slate-800 text-sm">
                IndicSent-<span className="text-teal-600">M</span>
              </p>
              <p className="text-xs text-slate-400 mt-0.5">
                Multimodal Sentiment Analysis for Indian Languages
              </p>
            </div>
            <div className="text-center text-xs text-slate-400 space-y-0.5">
              <p className="font-semibold text-slate-600">Team: Anya Yadav · Pragya Singh · Priyanshu Thakur · Ronit Jaiswal · Sparshman Gupta</p>
              <p>Instructor: Dr. Shubham Choudhary · CSET 346: NLP · Department of Computer Science & Engineering</p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
