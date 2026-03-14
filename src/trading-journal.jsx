import { useState, useEffect, useMemo, useRef } from "react";

// ... [Storage adapter and constant definitions remain unchanged] ...

// ── REFERENCE SECTION COMPONENT ──
const ReferenceSection = ({ activeSection, setActiveSection }) => {
  const SECTIONS = [
    { id: "sessions", label: "📈 SESSIONS" },
    { id: "rules", label: "📜 TRADING RULES" },
    { id: "checklist", label: "✅ CHECKLIST" },
    { id: "specs", label: "🔍 CONTRACT SPECS" }
  ];

  return (
    <div style={{ padding: "20px", maxWidth: "1200px", margin: "0 auto" }}>
      <div style={{ marginBottom: 30 }}>
        <div style={{ fontSize: 11, color: "#3b82f6", letterSpacing: "0.2em", marginBottom: 10 }}>REFERENCE</div>
        <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 32, color: "#e2e8f0", letterSpacing: "0.1em", lineHeight: 1 }}>
          TRADING <span style={{ color: "#00ff88" }}>SESSIONS</span>
        </div>
      </div>

      {/* ── SECTION TABS (Sizing Fixed) ── */}
      <div style={{ display: "flex", gap: 10, marginBottom: 30, flexWrap: "wrap" }}>
        {SECTIONS.map(s => (
          <button 
            key={s.id} 
            className="ref-sec-btn" 
            onClick={() => setActiveSection(s.id)} 
            style={{ 
              flex: "1 1 auto", // Allows buttons to grow and fill space
              minWidth: "140px",
              padding: "12px 24px", // Increased padding to match main page
              borderRadius: 6, 
              fontFamily: "inherit", 
              fontSize: 12, // Increased font size for legibility
              cursor: "pointer", 
              letterSpacing: "0.1em", 
              transition: "all .15s", 
              background: activeSection === s.id ? "#0a1628" : "transparent", 
              border: `1px solid ${activeSection === s.id ? "#1e3a5f" : "#1e293b"}`, 
              color: activeSection === s.id ? "#93c5fd" : "#64748b",
              fontWeight: activeSection === s.id ? 600 : 400
            }}
          >
            {s.label}
          </button>
        ))}
      </div>

      {/* ── TAB CONTENT: SESSIONS MAP (Sizing Fixed) ── */}
      {activeSection === "sessions" && (
        <div style={{ animation: "refFadeIn .3s ease", width: "100%" }}>
          <div style={{ 
            background: "#060b18", 
            border: "1px solid #1e293b", 
            borderRadius: 8, 
            padding: "25px", // Increased inner padding
            width: "100%",
            boxSizing: "border-box" 
          }}>
            <div style={{ fontSize: 11, color: "#64748b", letterSpacing: "0.15em", marginBottom: 20 }}>TIME WINDOWS</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 20 }}>
              <div style={{ fontSize: 14, color: "#94a3b8", lineHeight: 1.8 }}>
                Primary window: <strong style={{ color: "#00ff88" }}>9:30 AM–12:00 PM EST</strong><br/>
                Secondary window: <strong style={{ color: "#ff8c00" }}>3:00–4:00 PM EST</strong><br/>
                <span style={{ color: "#f87171" }}>All Lucid Flex positions must close by 4:45 PM EST.</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ... [Additional tab content logic] ... */}
    </div>
  );
};

export default function TradingJournal() {
  // ... [Main component logic] ...
}