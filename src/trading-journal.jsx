// ── REFERENCE SECTION COMPONENT ──
const ReferenceSection = ({ activeSection, setActiveSection }) => {
  
  // ── USER EDITABLE LINKS ──
  const USER_LINKS = [
    { 
      name: "TradingView Charts", 
      url: "https://www.tradingview.com", 
      info: "Primary charting platform for technical analysis and backtesting." 
    },
    { 
      name: "Economic Calendar", 
      url: "https://www.forexfactory.com/calendar", 
      info: "Monitor high-impact news events (CPI, FOMC, NFP) before trading." 
    },
    { 
      name: "CME Group FedWatch", 
      url: "https://www.cmegroup.com/markets/interest-rates/fedwatch-tool.html", 
      info: "Probability of Fed interest rate changes based on 30-Day Fed Fund futures." 
    }
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
      {/* ── HEADER ── */}
      <div style={{ marginBottom: 30 }}>
        <div style={{ fontSize: 11, color: "#3b82f6", letterSpacing: "0.2em", marginBottom: 10 }}>REFERENCE</div>
        <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 36, color: "#e2e8f0", letterSpacing: "0.1em", lineHeight: 1 }}>
          TRADING <span style={{ color: "#00ff88" }}>RESOURCES</span>
        </div>
      </div>

      {/* ── NAVIGATION GRID ── */}
      <div style={{ 
        display: "grid", 
        gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", 
        gap: 12, 
        marginBottom: 40 
      }}>
        {SECTIONS.map(s => (
          <button 
            key={s.id} 
            onClick={() => setActiveSection(s.id)} 
            style={{ 
              padding: "16px 20px",
              borderRadius: 8, 
              fontFamily: "inherit", 
              fontSize: 13, 
              fontWeight: 600,
              cursor: "pointer", 
              letterSpacing: "0.08em", 
              transition: "all .2s ease", 
              textAlign: "center",
              background: activeSection === s.id ? "#0f1a2e" : "#070d1a", 
              border: `1px solid ${activeSection === s.id ? "#3b82f6" : "#1e293b"}`, 
              color: activeSection === s.id ? "#93c5fd" : "#64748b",
              boxShadow: activeSection === s.id ? "0 4px 12px rgba(59, 130, 246, 0.15)" : "none"
            }}
          >
            {s.label}
          </button>
        ))}
      </div>

      {/* ── CONTENT AREA ── */}
      <div style={{ 
        background: "#060b18", 
        border: "1px solid #1e293b", 
        borderRadius: 10, 
        padding: "30px", 
        width: "100%",
        boxSizing: "border-box" 
      }}>
        
        {/* LINKS TAB */}
        {activeSection === "links" && (
          <div style={{ animation: "refFadeIn .3s ease" }}>
            <div style={{ fontSize: 12, color: "#3b82f6", letterSpacing: "0.15em", marginBottom: 25 }}>SAVED RESOURCES</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 20 }}>
              {USER_LINKS.map((link, idx) => (
                <div key={idx} style={{ background: "#0a0e1a", border: "1px solid #1e3a5f", borderRadius: 8, padding: "1