import { useState, useEffect, useMemo, useRef } from "react";

// ── Storage adapter ─────────────────────────────────────────────────────────
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

// ── Constants ───────────────────────────────────────────────────────────────
const BIAS_OPTIONS = ["Bullish", "Bearish", "Neutral", "Mixed"];
const MISTAKE_OPTIONS = ["Entered without setup", "Chased / FOMO", "Ignored confirmation", "Moved stop", "Cut winner early", "Let loser run"];
const MOOD_OPTIONS = ["Focused 🎯", "Disciplined 🧠", "Confident 💪", "Patient 🧘", "Calm 😌", "Tired 😴", "Anxious 😬"];
const GRADE_OPTIONS = ["A+", "A", "B+", "B", "C", "D", "F"];
const INSTRUMENTS = ["ES", "MES", "NQ", "MNQ"];

// ── Reference Section (The Fixed Component) ────────────────────────────────
const ReferenceSection = ({ activeSection, setActiveSection }) => {
  const USER_LINKS = [
    { name: "TradingView Charts", url: "https://www.tradingview.com", info: "Primary charting and technical analysis platform." },
    { name: "Economic Calendar", url: "https://www.forexfactory.com/calendar", info: "Watch for CPI, FOMC, and high-impact news." },
    { name: "CME FedWatch", url: "https://www.cmegroup.com/markets/interest-rates/fedwatch-tool.html", info: "Monitor interest rate probabilities." }
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
    <div style={{ padding: "20px", maxWidth: "1200px", margin: "0 auto" }}>
      <div style={{ marginBottom: 30 }}>
        <div style={{ fontSize: 11, color: "#3b82f6", letterSpacing: "0.2em", marginBottom: 10 }}>REFERENCE</div>
        <div style={{ fontFamily: "sans-serif", fontWeight: "bold", fontSize: 36, color: "#e2e8f0", letterSpacing: "0.1em" }}>
          TRADING <span style={{ color: "#00ff88" }}>RESOURCES</span>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12, marginBottom: 40 }}>
        {SECTIONS.map(s => (
          <button 
            key={s.id} 
            onClick={() => setActiveSection(s.id)} 
            style={{ 
              padding: "16px 20px", borderRadius: 8, fontSize: 13, fontWeight: 600, cursor: "pointer",
              transition: "all .2s ease", textAlign: "center",
              background: activeSection === s.id ? "#0f1a2e" : "#070d1a", 
              border: `1px solid ${activeSection === s.id ? "#3b82f6" : "#1e293b"}`, 
              color: activeSection === s.id ? "#93c5fd" : "#64748b"
            }}
          >
            {s.label}
          </button>
        ))}
      </div>

      <div style={{ background: "#060b18", border: "1px solid #1e293b", borderRadius: 10, padding: "30px", width: "100%" }}>
        {activeSection === "links" && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 20 }}>
            {USER_LINKS.map((link, idx) => (
              <div key={idx} style={{ background: "#0a0e1a", border: "1px solid #1e3a5f", borderRadius: 8, padding: "18px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                  <div style={{ fontSize: 16, fontWeight: 700, color: "#e2e8f0" }}>{link.name}</div>
                  <a href={link.url} target="_blank" rel="noopener noreferrer" style={{ fontSize: 11, color: "#00ff88", textDecoration: "none", border: "1px solid #00ff8844", padding: "4px 10px", borderRadius: 4 }}>OPEN ↗</a>
                </div>
                <div style={{ fontSize: 13, color: "#94a3b8", lineHeight: 1.6 }}>{link.info}</div>
              </div>
            ))}
          </div>
        )}

        {activeSection === "sessions" && (
          <div style={{ color: "#94a3b8", lineHeight: 1.8 }}>
             <div style={{ fontSize: 12, color: "#3b82f6", marginBottom: 15 }}>TIME WINDOWS</div>
             Primary window: <strong style={{ color: "#00ff88" }}>9:30 AM–12:00 PM EST</strong><br/>
             Secondary window: <strong style={{ color: "#ff8c00" }}>3:00–4:00 PM EST</strong>
          </div>
        )}
        
        {/* Placeholder for other views to prevent crash */}
        {!["links", "sessions"].includes(activeSection) && <div style={{ color: "#64748b" }}>Section content coming soon...</div>}
      </div>
    </div>
  );
};

// ── Main Component ─────────────────────────────────────────────────────────
export default function TradingJournal() {
  const [entries, setEntries] = useState([]);
  const [activeSection, setActiveSection] = useState("sessions");
  const [tab, setTab] = useState("journal");

  const removeEntry = (id) => {
    if (window.confirm("Are you sure you want to delete this trade?")) {
      setEntries(entries.filter(e => e.id !== id));
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: "#020617", color: "#e2e8f0", fontFamily: "sans-serif" }}>
      {/* Tab Navigation */}
      <div style={{ display: "flex", gap: 20, padding: "20px", borderBottom: "1px solid #1e293b" }}>
        <button onClick={() => setTab("journal")} style={{ background: "transparent", color: tab === "journal" ? "#3b82f6" : "#64748b", border: "none", cursor: "pointer", fontWeight: 600 }}>JOURNAL</button>
        <button onClick={() => setTab("reference")} style={{ background: "transparent", color: tab === "reference" ? "#3b82f6" : "#64748b", border: "none", cursor: "pointer", fontWeight: 600 }}>REFERENCE</button>
      </div>

      {tab === "journal" && (
        <div style={{ padding: "20px" }}>
          <h2>Trading Journal</h2>
          {entries.length === 0 && <p style={{ color: "#64748b" }}>No trades logged yet.</p>}
          {entries.map(e => (
            <div key={e.id} style={{ padding: "10px", borderBottom: "1px solid #1e293b", display: "flex", justifyContent: "space-between" }}>
               <span>{e.date} - {e.instrument}</span>
               <button onClick={() => removeEntry(e.id)} style={{ color: "#f87171", background: "transparent", border: "none", cursor: "pointer" }}>✕</button>
            </div>
          ))}
        </div>
      )}

      {tab === "reference" && (
        <ReferenceSection activeSection={activeSection} setActiveSection={setActiveSection} />
      )}
    </div>
  );
}