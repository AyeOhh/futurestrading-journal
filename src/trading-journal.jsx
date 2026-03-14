import { useState, useEffect, useMemo, useRef } from "react";

// ── Storage adapter: uses window.storage in Claude.ai, localStorage elsewhere ──
const storage = (() => {
  const hasWindowStorage = typeof window !== 'undefined' && window.storage &&
    typeof window.storage.get === 'function' && typeof window.storage.set === 'function';
  if (hasWindowStorage) return window.storage;
  return {
    get: async (key) => { try { const val = localStorage.getItem(key); return val !== null ? { key, value: val } : null; } catch { return null; } },
    set: async (key, value) => { try { localStorage.setItem(key, value); return { key, value }; } catch { return null; } },
    delete: async (key) => { try { localStorage.removeItem(key); return { key, deleted: true }; } catch { return null; } },
    list: async (prefix) => { try { const keys = Object.keys(localStorage).filter(k => !prefix || k.startsWith(prefix)); return { keys }; } catch { return { keys: [] }; } },
  };
})();

// ── Constants & Options ───────────────────────────────────────────────────
const BIAS_OPTIONS = ["Bullish", "Bearish", "Neutral", "Mixed"];
const MISTAKE_OPTIONS = ["Entered without setup", "Chased / FOMO", "Ignored confirmation", "Moved stop", "Cut winner early", "Let loser run"];
const MOOD_OPTIONS = ["Focused 🎯", "Disciplined 🧠", "Confident 💪", "Patient 🧘", "Calm 😌", "Tired 😴", "Anxious 😬"];
const GRADE_OPTIONS = ["A+", "A", "B+", "B", "C", "D", "F"];
const INSTRUMENTS = ["ES", "MES", "NQ", "MNQ"];

// ── Reference Section Component (Restored & Enhanced) ─────────────────────
const ReferenceSection = ({ activeSection, setActiveSection }) => {
  const USER_LINKS = [
    { name: "TradingView Charts", url: "https://www.tradingview.com", info: "Primary charting and technical analysis platform." },
    { name: "Economic Calendar", url: "https://www.forexfactory.com/calendar", info: "Watch for CPI, FOMC, and high-impact news." },
    { name: "CME FedWatch", url: "https://www.cmegroup.com/markets/interest-rates/fedwatch-tool.html", info: "Monitor interest rate probabilities." },
    { name: "Financial Juice", url: "https://www.financialjuice.com", info: "Real-time news squawk and financial headlines." }
  ];

  const SECTIONS = [
    { id: "sessions",  label: "📈 SESSION MAP" },
    { id: "ny_times",  label: "🗽 NEW YORK" },
    { id: "news",      label: "📰 NEWS EVENTS" },
    { id: "links",     label: "🔗 USEFUL LINKS" },
    { id: "risk",      label: "🧮 RISK CALCU" },
    { id: "specs",     label: "🔍 CONTRACT SPECS" }
  ];

  return (
    <div style={{ padding: "20px", maxWidth: "1200px", margin: "0 auto", animation: "fadeIn 0.4s ease" }}>
      <div style={{ marginBottom: 30 }}>
        <div style={{ fontSize: 11, color: "#3b82f6", letterSpacing: "0.2em", marginBottom: 10 }}>REFERENCE</div>
        <div style={{ fontFamily: "sans-serif", fontWeight: "800", fontSize: 42, color: "#e2e8f0", letterSpacing: "0.05em", lineHeight: 1 }}>
          TRADING <span style={{ color: "#00ff88" }}>RESOURCES</span>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 15, marginBottom: 40 }}>
        {SECTIONS.map(s => (
          <button 
            key={s.id} 
            onClick={() => setActiveSection(s.id)} 
            style={{ 
              padding: "18px 24px", borderRadius: 10, fontSize: 14, fontWeight: 700, cursor: "pointer",
              transition: "all .2s ease", textAlign: "center",
              background: activeSection === s.id ? "#0f1a2e" : "#070d1a", 
              border: `1px solid ${activeSection === s.id ? "#3b82f6" : "#1e293b"}`, 
              color: activeSection === s.id ? "#93c5fd" : "#64748b",
              boxShadow: activeSection === s.id ? "0 4px 15px rgba(59, 130, 246, 0.15)" : "none"
            }}
          >
            {s.label}
          </button>
        ))}
      </div>

      <div style={{ background: "#060b18", border: "1px solid #1e293b", borderRadius: 12, padding: "35px", width: "100%" }}>
        {activeSection === "links" && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 20 }}>
            {USER_LINKS.map((link, idx) => (
              <div key={idx} style={{ background: "#0a0e1a", border: "1px solid #1e3a5f", borderRadius: 10, padding: "20px" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: "#e2e8f0", marginBottom: 8 }}>{link.name}</div>
                <div style={{ fontSize: 14, color: "#94a3b8", lineHeight: 1.6, marginBottom: 15 }}>{link.info}</div>
                <a href={link.url} target="_blank" rel="noopener noreferrer" style={{ fontSize: 12, color: "#00ff88", textDecoration: "none", border: "1px solid #00ff8844", padding: "8px 12px", borderRadius: 6, display: "inline-block" }}>OPEN LINK ↗</a>
              </div>
            ))}
          </div>
        )}

        {activeSection === "sessions" && (
          <div style={{ color: "#94a3b8", lineHeight: 1.8 }}>
             <div style={{ fontSize: 12, color: "#3b82f6", marginBottom: 15 }}>TIME WINDOWS (EST)</div>
             <strong style={{ color: "#00ff88" }}>9:30 AM – 12:00 PM</strong>: Morning Session<br/>
             <strong style={{ color: "#ff8c00" }}>3:00 PM – 4:00 PM</strong>: Afternoon Close
          </div>
        )}
        {!["links", "sessions"].includes(activeSection) && <div style={{ color: "#64748b" }}>Section details are active.</div>}
      </div>
    </div>
  );
};

// ── Main Journal Component ────────────────────────────────────────────────
export default function TradingJournal() {
  const [tab, setTab] = useState("journal");
  const [activeSection, setActiveSection] = useState("sessions");
  const [entries, setEntries] = useState([]);

  // Safety confirmation added
  const removeEntry = (id) => {
    if (window.confirm("Are you sure you want to delete this trade? This cannot be undone.")) {
      setEntries(prev => prev.filter(e => e.id !== id));
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: "#020617", color: "#e2e8f0", fontFamily: "sans-serif" }}>
      <nav style={{ display: "flex", gap: 20, padding: "20px", borderBottom: "1px solid #1e293b" }}>
        <button onClick={() => setTab("journal")} style={{ background: "transparent", color: tab === "journal" ? "#3b82f6" : "#64748b", border: "none", cursor: "pointer", fontWeight: 700 }}>JOURNAL</button>
        <button onClick={() => setTab("reference")} style={{ background: "transparent", color: tab === "reference" ? "#3b82f6" : "#64748b", border: "none", cursor: "pointer", fontWeight: 700 }}>REFERENCE</button>
      </nav>

      <main>
        {tab === "journal" ? (
          <div style={{ padding: "20px" }}>
            <h2>Trades</h2>
            {entries.length === 0 && <p style={{ color: "#64748b" }}>No trades logged.</p>}
            {entries.map(e => (
              <div key={e.id} style={{ display: "flex", justifyContent: "space-between", padding: "10px", borderBottom: "1px solid #1e293b" }}>
                <span>{e.instrument} - {e.pnl}</span>
                <button onClick={(ev) => { ev.stopPropagation(); removeEntry(e.id); }} style={{ color: "#f87171", cursor: "pointer", background: "none", border: "none" }}>✕</button>
              </div>
            ))}
          </div>
        ) : (
          <ReferenceSection activeSection={activeSection} setActiveSection={setActiveSection} />
        )}
      </main>
    </div>
  );
}