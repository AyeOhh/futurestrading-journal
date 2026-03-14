import { useState, useEffect, useMemo, useRef } from "react";

// ── Storage adapter: uses window.storage in Claude.ai, localStorage elsewhere ──
const storage = (() => {
  const hasWindowStorage = typeof window !== 'undefined' && window.storage &&
    typeof window.storage.get === 'function' && typeof window.storage.set === 'function';

  if (hasWindowStorage) return window.storage;

  return {
    get: async (key) => {
      try {
        const val = localStorage.getItem(key);
        return val !== null ? { key, value: val } : null;
      } catch { return null; }
    },
    set: async (key, value) => {
      try { localStorage.setItem(key, value); return { key, value }; } catch { return null; }
    },
    delete: async (key) => {
      try { localStorage.removeItem(key); return { key, deleted: true }; } catch { return null; }
    },
    list: async (prefix) => {
      try {
        const keys = Object.keys(localStorage).filter(k => !prefix || k.startsWith(prefix));
        return { keys };
      } catch { return { keys: [] }; }
    },
  };
})();

// ... [Existing Options and Registry Code] ...

const TradingJournal = () => {
  // ... [Existing State Logic] ...

  // ── RENDER INDIVIDUAL ENTRY IN LIST ──
  const RenderEntry = ({ e, idx }) => {
    const a = entryAnalytics[e.id];
    const isActive = form.id === e.id;
    
    return (
      <div 
        onClick={() => { setForm(e); setTab("session"); }}
        style={{ 
          padding: "12px 16px", 
          background: isActive ? "#0f172a" : (idx % 2 === 0 ? "#070d1a" : "transparent"),
          borderBottom: "1px solid #0f1729",
          cursor: "pointer",
          position: "relative",
          transition: "background .15s"
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          {/* ... [Entry details like date, P&L, etc] ... */}

          {/* ── UPDATED DELETE BUTTON WITH CONFIRMATION ── */}
          <button 
            onClick={(event) => {
              event.stopPropagation(); // Stop row click from opening the editor
              if (window.confirm("Are you sure you want to delete this trade entry? This cannot be undone.")) {
                removeEntry(e.id);
              }
            }} 
            style={{
              background: "transparent",
              border: "none",
              color: "#475569", 
              fontSize: 14,
              cursor: "pointer",
              padding: "4px 8px",
              transition: "color .15s",
              fontFamily: "inherit",
              zIndex: 10
            }}
            onMouseEnter={el => el.currentTarget.style.color = "#f87171"} // Warning color on hover
            onMouseLeave={el => el.currentTarget.style.color = "#475569"}
            title="Delete Entry"
          >
            ✕
          </button>
        </div>
      </div>
    );
  };

  // ... [Remainder of existing file content] ...
};

export default TradingJournal;