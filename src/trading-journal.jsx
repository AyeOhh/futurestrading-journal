import { useState, useEffect, useMemo, useRef } from "react";

// ── Storage adapter: localStorage FIRST (survives artifact updates in same browser)
// window.storage is artifact-instance specific — switching to a new artifact wipes it.
// localStorage is tied to the browser, so data persists across every code update.
const storage = (() => {
  const hasLocalStorage = typeof window !== 'undefined' && (() => {
    try { localStorage.setItem('__tj_test__', '1'); localStorage.removeItem('__tj_test__'); return true; } catch { return false; }
  })();

  if (hasLocalStorage) {
    // Primary: localStorage — data survives new artifact pastes in the same browser
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
  }

  // Fallback: window.storage (Claude.ai artifact API) — only used if localStorage unavailable
  if (typeof window !== 'undefined' && window.storage &&
      typeof window.storage.get === 'function' && typeof window.storage.set === 'function') {
    return window.storage;
  }

  // Last resort: in-memory (session only, no persistence)
  const mem = {};
  return {
    get: async (key) => mem[key] ? { key, value: mem[key] } : null,
    set: async (key, value) => { mem[key] = value; return { key, value }; },
    delete: async (key) => { delete mem[key]; return { key, deleted: true }; },
    list: async (prefix) => ({ keys: Object.keys(mem).filter(k => !prefix || k.startsWith(prefix)) }),
  };
})();


const BIAS_OPTIONS = ["Bullish", "Bearish", "Neutral", "Mixed"];
const MISTAKE_OPTIONS = [
  "Entered without my setup",
  "Chased / FOMO",
  "Ignored confirmation",
  "Moved stop farther",
  "Cut winner early",
  "Let loser run",
  "No pre-trade plan",
  "Traded outside time window",
  "Didn't stop at daily max loss",
  "Overtraded after a win",
  "Traded during news without plan",
  "Traded tired/distracted",
];
const MOOD_OPTIONS = ["Focused 🎯", "Disciplined 🧠", "Confident 💪", "Patient 🧘", "Calm 😌", "Locked In 🔒", "Motivated 🚀", "Clear-Headed 💎", "Recharged 🔋", "Excited ⚡", "Tired 😴", "Anxious 😬", "Fearful 😰", "Distracted 😵", "Frustrated 😤", "Overconfident 😎", "Greedy 🤑", "Revenge mode 😡"];
const GRADE_OPTIONS = ["A+", "A", "B+", "B", "C", "D", "F"];
const INSTRUMENTS = ["ES", "MES", "NQ", "MNQ"];

const JOURNAL_TYPES = { PERSONAL: "personal", PROP: "prop" };

// ── AI Provider Registry — Adapter Pattern ─────────────────────────────────
// Each entry is a self-contained adapter. Add a new provider by adding an entry here.
const AI_PROVIDER_REGISTRY = [
  {
    id: 'anthropic',
    label: 'Anthropic (Claude)',
    keyPlaceholder: 'sk-ant-api03-…',
    keyHint: 'console.anthropic.com',
    defaultModel: 'claude-sonnet-4-20250514',
    models: [
      { id: 'claude-opus-4-6',             label: 'Claude Opus 4.6'   },
      { id: 'claude-sonnet-4-20250514',    label: 'Claude Sonnet 4.5' },
      { id: 'claude-haiku-4-5-20251001',   label: 'Claude Haiku 4.5'  },
    ],
    async request(ai, { messages, max_tokens = 600, model, timeoutMs = 22000 }) {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), timeoutMs);
      try {
        const res = await fetch('https://api.anthropic.com/v1/messages', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': ai.apiKey,
            'anthropic-version': '2023-06-01',
            'anthropic-dangerous-direct-browser-access': 'true',
          },
          signal: ctrl.signal,
          body: JSON.stringify({ model: model || ai.model, max_tokens, messages }),
        });
        if (!res.ok) throw new Error(`API error ${res.status}`);
        const data = await res.json();
        if (data?.error) throw new Error(data.error.message || 'API returned an error');
        const text = data.content?.map(b => b.text || '').join('\n') || '';
        const clean = text.trim();
        if (!clean) throw new Error('Empty response from API');
        return clean;
      } finally { clearTimeout(t); }
    },
  },
  {
    id: 'openai',
    label: 'OpenAI (GPT)',
    keyPlaceholder: 'sk-…',
    keyHint: 'platform.openai.com',
    defaultModel: 'gpt-4o',
    models: [
      { id: 'gpt-4o',          label: 'GPT-4o'      },
      { id: 'gpt-4o-mini',     label: 'GPT-4o Mini' },
      { id: 'o3',              label: 'o3'           },
      { id: 'o4-mini',         label: 'o4-mini'      },
    ],
    async request(ai, { messages, max_tokens = 600, model, timeoutMs = 22000 }) {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), timeoutMs);
      try {
        const res = await fetch('https://api.openai.com/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${ai.apiKey}` },
          signal: ctrl.signal,
          body: JSON.stringify({ model: model || ai.model, max_tokens, messages }),
        });
        if (!res.ok) throw new Error(`API error ${res.status}`);
        const data = await res.json();
        if (data?.error) throw new Error(data.error.message || 'API returned an error');
        const text = data.choices?.[0]?.message?.content || '';
        const clean = text.trim();
        if (!clean) throw new Error('Empty response from API');
        return clean;
      } finally { clearTimeout(t); }
    },
  },
  {
    id: 'gemini',
    label: 'Google (Gemini)',
    keyPlaceholder: 'AIzaSy…',
    keyHint: 'aistudio.google.com',
    defaultModel: 'gemini-2.5-flash',
    models: [
      { id: 'gemini-2.5-pro',   label: 'Gemini 2.5 Pro (best quality)'     },
      { id: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash (recommended ✓)' },
      { id: 'gemini-2.0-flash-lite', label: 'Gemini 2.0 Flash Lite (fastest free tier)' },
    ],
    async request(ai, { messages, max_tokens = 600, model, timeoutMs = 120000, thinkingBudget }) {
      const contents = messages.map(m => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }],
      }));
      const mdl = model || ai.model;
      // thinkingBudget: 0 disables internal reasoning tokens so the full
      // maxOutputTokens budget goes to the actual response (critical for long recaps)
      const genConfig = { maxOutputTokens: max_tokens, temperature: 0.7 };
      if (thinkingBudget != null) genConfig.thinkingConfig = { thinkingBudget };
      const body = JSON.stringify({
        contents,
        generationConfig: genConfig,
        safetySettings: [
          { category: 'HARM_CATEGORY_HARASSMENT',        threshold: 'BLOCK_NONE' },
          { category: 'HARM_CATEGORY_HATE_SPEECH',       threshold: 'BLOCK_NONE' },
          { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_NONE' },
          { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_NONE' },
        ],
      });

      // Use streamGenerateContent (returns chunked JSON array).
      // Each chunk arrives as it's generated so the connection never stalls.
      // We wrap with a hard timeout using Promise.race.
      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Gemini request timed out after 2 minutes')), timeoutMs)
      );

      const fetchPromise = (async () => {
        const ctrl = new AbortController();
        const res = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/${mdl}:streamGenerateContent?key=${ai.apiKey}`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: ctrl.signal,
            body,
          }
        );
        if (!res.ok) {
          const errBody = await res.text().catch(() => '');
          throw new Error(`API error ${res.status}: ${errBody.slice(0, 300)}`);
        }
        // streamGenerateContent returns a JSON array: [chunk, chunk, ...]
        // Read the full body then parse — each chunk has candidates[0].content.parts
        const text = await res.text();
        // The response is a JSON array of GenerateContentResponse objects
        // Strip outer brackets and split on '},\r\n{' to parse chunks
        let accumulated = '';
        try {
          const chunks = JSON.parse(text);
          if (Array.isArray(chunks)) {
            for (const chunk of chunks) {
              const t = chunk.candidates?.[0]?.content?.parts?.map(p => p.text || '').join('') || '';
              accumulated += t;
            }
          } else {
            // Single response object fallback
            accumulated = chunks.candidates?.[0]?.content?.parts?.map(p => p.text || '').join('') || '';
          }
        } catch {
          // If JSON parse fails, try extracting text directly
          const matches = text.matchAll(/"text":\s*"((?:[^"\\]|\\.)*)"/g);
          for (const m of matches) {
            try { accumulated += JSON.parse(`"${m[1]}"`); } catch { accumulated += m[1]; }
          }
        }
        const clean = accumulated.trim();
        if (!clean) {
          throw new Error(`Empty response from Gemini — check model name and API key`);
        }
        return clean;
      })();

      return Promise.race([fetchPromise, timeoutPromise]);
    },
  },
  {
    id: 'grok',
    label: 'xAI (Grok)',
    keyPlaceholder: 'xai-…',
    keyHint: 'console.x.ai',
    defaultModel: 'grok-3',
    models: [
      { id: 'grok-3',      label: 'Grok 3'      },
      { id: 'grok-3-mini', label: 'Grok 3 Mini' },
    ],
    async request(ai, { messages, max_tokens = 600, model, timeoutMs = 22000 }) {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), timeoutMs);
      try {
        const res = await fetch('https://api.x.ai/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${ai.apiKey}` },
          signal: ctrl.signal,
          body: JSON.stringify({ model: model || ai.model, max_tokens, messages }),
        });
        if (!res.ok) throw new Error(`API error ${res.status}`);
        const data = await res.json();
        if (data?.error) throw new Error(data.error.message || 'API returned an error');
        const text = data.choices?.[0]?.message?.content || '';
        const clean = text.trim();
        if (!clean) throw new Error('Empty response from API');
        return clean;
      } finally { clearTimeout(t); }
    },
  },
];

// Helper: look up a provider adapter by id (falls back to anthropic)
const getProviderAdapter = (id) =>
  AI_PROVIDER_REGISTRY.find(p => p.id === id) || AI_PROVIDER_REGISTRY[0];

// ── AI Settings (local only, not included in exports) ─────────────────────────
const AI_SETTINGS_KEY = 'tj-ai-settings-v1';
const DEFAULT_AI_SETTINGS = {
  enabled: true,
  provider: 'anthropic',
  model: 'claude-sonnet-4-20250514',
  apiKey: '',
  brokerPreset: 'none',
  autoBackup: false,
  tzLock: true,
};

// ── AI Taglines — user picks one, shown in recap header and daily analysis ──────
const AI_TAGLINES = [
  { id: 'analyse', text: 'ANALYSE · REFLECT · IMPROVE' },
  { id: 'edge',    text: 'Edge is earned in the debrief' },
  { id: 'review',  text: 'Review the trade, not just the result' },
  { id: 'noise',   text: 'Data without reflection is just noise' },
  { id: 'journal', text: 'The journal is the edge' },
];
const AI_TAGLINE_KEY = 'tj-ai-tagline-v1';
const loadTagline = () => { try { return localStorage.getItem(AI_TAGLINE_KEY) || 'analyse'; } catch { return 'analyse'; } };
const saveTagline = (id) => { try { localStorage.setItem(AI_TAGLINE_KEY, id); } catch {} };
const getTaglineText = (id) => AI_TAGLINES.find(t => t.id === id)?.text || AI_TAGLINES[0].text;
// ── Shared notes hash — identical algorithm used by WeeklyPerformance and AIRecapView ──
const calcNotesHash = (periodEntries) => {
  const str = periodEntries.map(e =>
    [e.lessonsLearned, e.mistakes, e.improvements, e.rules, e.reinforceRule,
     e.tomorrow, e.marketNotes, e.bestTrade, e.worstTrade,
     (e.parsedTrades||[]).length].join('|')
  ).join('||');
  let h = 0;
  for (let i = 0; i < str.length; i++) { h = ((h << 5) - h) + str.charCodeAt(i); h |= 0; }
  return String(h);
};

// ── Note sanitiser — strips invisible Unicode, normalises quotes/dashes ──────
// Called before saving any freeform text field to prevent AI prompt corruption.

const BROKER_PRESETS = {
  none:        { label: 'Generic / Other',   hint: '' },
  tradovate:   { label: 'Tradovate',         hint: 'Data is from Tradovate. Columns are typically: Account, Contract, B/S, Qty, Price, Commission, P&L, Fill Time.' },
  ibkr:        { label: 'IBKR (TWS/Flex)',   hint: 'Data is from Interactive Brokers. Date/time in YYYYMMDD;HHMMSS format. Symbol includes expiry (e.g. MESM6). P&L in "Realized P&L" column.' },
  questrade:   { label: 'Questrade',         hint: 'Data is from Questrade. Columns include Symbol, Action (Buy/Sell), Quantity, Price, Gross Amount. Time is Eastern.' },
  td:          { label: 'TD Ameritrade/TOS', hint: 'Data is from TD Ameritrade / thinkorswim. Date/time as MM/DD/YYYY HH:MM:SS. Side column uses BUY_TO_OPEN/SELL_TO_CLOSE etc.' },
  ninjatrader: { label: 'NinjaTrader',       hint: 'Data is from NinjaTrader. Trade history export includes Entry time, Exit time, Instrument, Market pos., Quantity, Entry price, Exit price, Profit.' },
};



// ── AI cache (local only; avoids repeat calls for same input) ───────────────
const AI_CACHE_KEY = 'tj-ai-cache-v1';
const AI_CACHE_MAX = 250; // keep it small; this is just a speed/cost cache

const loadAiCache = () => {
  try {
    const raw = localStorage.getItem(AI_CACHE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return (parsed && typeof parsed === 'object') ? parsed : {};
  } catch { return {}; }
};

const saveAiCache = (cache) => {
  try { localStorage.setItem(AI_CACHE_KEY, JSON.stringify(cache)); } catch {}
};

const pruneAiCache = (cache) => {
  try {
    const keys = Object.keys(cache);
    if (keys.length <= AI_CACHE_MAX) return cache;
    keys.sort((a, b) => (cache[a]?.ts || 0) - (cache[b]?.ts || 0));
    const toDelete = keys.length - AI_CACHE_MAX;
    for (let i = 0; i < toDelete; i++) delete cache[keys[i]];
    return cache;
  } catch { return cache; }
};

const sha256 = async (str) => {
  const enc = new TextEncoder();
  const buf = await crypto.subtle.digest('SHA-256', enc.encode(str));
  return Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2,'0')).join('');
};

const getCachedAiText = (hash) => {
  const c = loadAiCache();
  const hit = c?.[hash];
  return hit?.text || '';
};

const setCachedAiText = (hash, text) => {
  const c = loadAiCache();
  c[hash] = { text, ts: Date.now() };
  saveAiCache(pruneAiCache(c));
};

// ── AI request hardening: timeout + retry + consistent messages ─────────────
const friendlyAiError = (err) => {
  const msg = (err?.message || '').toString();
  const name = (err?.name || '').toString();

  if (name === 'AbortError') return { code: 'timeout', message: 'AI request timed out. Try again.' };
  if (/Missing Anthropic API key/i.test(msg)) return { code: 'no_key', message: 'Missing API key. Add it in Settings (⚙).' };
  if (/AI features are disabled/i.test(msg)) return { code: 'disabled', message: 'AI features are disabled in Settings (⚙).' };
  if (/API error\s+401/.test(msg) || /status\s+401/.test(msg)) return { code: 'unauthorized', message: 'AI API key was rejected (401). Double-check your key.' };
  if (/API error\s+403/.test(msg) || /status\s+403/.test(msg)) return { code: 'forbidden', message: 'AI request blocked (403). Check your account / permissions.' };
  if (/API error\s+429/.test(msg) || /status\s+429/.test(msg)) return { code: 'rate_limit', message: 'Rate limited by the AI provider (429). Wait a moment and retry.' };
  if (/API error\s+5\d\d/.test(msg) || /status\s+5\d\d/.test(msg)) return { code: 'provider_down', message: 'AI provider error (5xx). Retry in a moment.' };
  if (/Failed to fetch/i.test(msg) || /NetworkError/i.test(msg)) return { code: 'network', message: 'Network error calling AI. Check your connection and retry.' };

  return { code: 'unknown', message: msg || 'AI request failed. Please try again.' };
};

const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const loadAiSettings = () => {
  try {
    const raw = localStorage.getItem(AI_SETTINGS_KEY);
    if (!raw) return { ...DEFAULT_AI_SETTINGS };
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_AI_SETTINGS, ...parsed };
  } catch {
    return { ...DEFAULT_AI_SETTINGS };
  }
};

const saveAiSettings = (s) => {
  try { localStorage.setItem(AI_SETTINGS_KEY, JSON.stringify(s)); } catch {}
};

const requireAiReady = (ai) => {
  if (!ai?.enabled) throw new Error('AI features are disabled in Settings.');
  if (!ai?.apiKey) throw new Error('Missing API key. Add it in AI Settings.');
};

const parseMaybeDate = (str) => {
  if (!str) return NaN;
  const s = String(str).trim();
  const t = Date.parse(s);
  if (!Number.isNaN(t)) return t;
  // Try MM/DD/YYYY HH:MM:SS (optionally with AM/PM)
  const m = s.match(/^(\d{1,2})\/(\d{1,2})\/(\d{2,4})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM)?$/i);
  if (m) {
    let mm = Number(m[1]), dd = Number(m[2]), yy = Number(m[3]);
    if (yy < 100) yy += 2000;
    let hh = Number(m[4]), mi = Number(m[5]), ss = Number(m[6] || 0);
    const ap = (m[7] || '').toUpperCase();
    if (ap === 'PM' && hh !== 12) hh += 12;
    if (ap === 'AM' && hh === 12) hh = 0;
    const d = new Date(yy, mm - 1, dd, hh, mi, ss);
    const ts = d.getTime();
    return Number.isNaN(ts) ? NaN : ts;
  }
  return NaN;
};

// Adapter-pattern dispatcher — routes to the correct provider adapter
const aiRequestText = async (ai, opts) => {
  requireAiReady(ai);
  const adapter = getProviderAdapter(ai?.provider);
  const maxAttempts = 3;
  let lastErr = null;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await adapter.request(ai, opts);
    } catch (err) {
      lastErr = err;
      const f = friendlyAiError(err);
      // Hard failures — don't retry
      if (['no_key','disabled','unauthorized','forbidden'].includes(f.code)) throw err;
      if (attempt < maxAttempts) {
        // 429 rate limit: wait 35s first retry, 60s second retry
        // Other errors: short exponential backoff
        const delay = f.code === 'rate_limit' ? (attempt === 1 ? 35000 : 60000) : 450 * attempt;
        await sleep(delay);
      }
    }
  }
  throw lastErr || new Error('AI request failed');
};

// LucidFlex rule templates — both phases per account size
const LUCID_FLEX_RULES = {
  50000: {
    label: "50K Flex",
    maxSize: "4 Mini / 40 Micro",
    eval: {
      profitTarget: 3000, maxLossLimit: 2000, dailyLossLimit: 0,
      consistencyRule: 50, minTradingDays: 0, drawdownType: "EOD",
    },
    funded: {
      maxLossLimit: 2000, dailyLossLimit: 0, consistencyRule: 0,
      minProfitDays: 5, minDailyProfit: 150, daysToPayout: 5, payoutsToLive: 5,
    },
  },
  100000: {
    label: "100K Flex",
    maxSize: "6 Mini / 60 Micro",
    eval: {
      profitTarget: 6000, maxLossLimit: 3000, dailyLossLimit: 0,
      consistencyRule: 50, minTradingDays: 0, drawdownType: "EOD",
    },
    funded: {
      maxLossLimit: 3000, dailyLossLimit: 0, consistencyRule: 0,
      minProfitDays: 5, minDailyProfit: 200, daysToPayout: 5, payoutsToLive: 5,
    },
  },
  150000: {
    label: "150K Flex",
    maxSize: "10 Mini / 100 Micro",
    eval: {
      profitTarget: 9000, maxLossLimit: 4500, dailyLossLimit: 0,
      consistencyRule: 50, minTradingDays: 0, drawdownType: "EOD",
    },
    funded: {
      maxLossLimit: 4500, dailyLossLimit: 0, consistencyRule: 0,
      minProfitDays: 5, minDailyProfit: 250, daysToPayout: 5, payoutsToLive: 5,
    },
  },
};
const PROP_ACCOUNT_SIZES = [50000, 100000, 150000];

// Default prop journal config — single lifecycle journal (eval → funded)
const defaultPropConfig = (accountSize = 50000) => {
  const tmpl = LUCID_FLEX_RULES[accountSize] || LUCID_FLEX_RULES[50000];
  return {
    firmName: "Lucid Trading",
    product: "LucidFlex",
    accountSize,
    maxSize: tmpl.maxSize,
    // Eval phase rules
    eval: { ...tmpl.eval },
    // Funded phase rules
    funded: { ...tmpl.funded },
    // Lifecycle state
    phase: "eval",           // "eval" | "funded" | "archived"
    passedDate: null,        // ISO date string when passed eval
    fundedStartBalance: accountSize, // fresh balance at funded start
    // Payouts (funded withdrawals)
    payouts: [],             // [{ id, date, amount, fee, note }]
    // Business tracking
    evalCost: 0,             // cost of the challenge/evaluation fee
    // Archive / breach
    isArchived: false,
    breachedDate: null,      // ISO date if account was breached
    breachType: null,        // "maxLoss" | "dailyLoss" | "slow_bleed" | "manual"
    postMortem: "",          // trader's own analysis of the failure
  };
};

const defaultPersonalConfig = (startingBalance = 10000) => ({ startingBalance });

// Compute prop firm status across full lifecycle (eval → funded)
const calcPropStatus = (entries, config) => {
  if (!config) return null;
  const sorted = [...entries].sort((a, b) => a.date.localeCompare(b.date));
  const passedDate = config.passedDate || null;
  const evalRules = config.eval || {};
  const fundedRules = config.funded || {};

  // Split entries by phase
  const evalEntries = passedDate ? sorted.filter(e => e.date < passedDate) : sorted;
  const fundedEntries = passedDate ? sorted.filter(e => e.date >= passedDate) : [];

  const calcPhase = (phaseEntries, startBalance) => {
    let cumPnl = 0, peakBalance = startBalance, maxDrawdownHit = 0, maxSingleDayPnl = 0;
    for (const e of phaseEntries) {
      const net = (parseFloat(e.pnl) || 0) - (parseFloat(e.commissions) || 0);
      cumPnl += net;
      const bal = startBalance + cumPnl;
      if (bal > peakBalance) peakBalance = bal;
      const dd = peakBalance - bal;
      if (dd > maxDrawdownHit) maxDrawdownHit = dd;
      if (net > maxSingleDayPnl) maxSingleDayPnl = net;
    }
    const daysTraded = phaseEntries.filter(e =>
      (e.parsedTrades?.length > 0) || Math.abs(parseFloat(e.pnl) || 0) > 0
    ).length;
    const currentBalance = startBalance + cumPnl;
    const trailingDrawdown = peakBalance - currentBalance;
    const consistencyPct = cumPnl > 0 ? (maxSingleDayPnl / cumPnl) * 100 : 0;
    return { cumPnl, currentBalance, peakBalance, trailingDrawdown, maxDrawdownHit, maxSingleDayPnl, daysTraded, consistencyPct };
  };

  // Payouts — sorted by date, only during funded phase
  const payouts = (config.payouts || []).sort((a, b) => a.date.localeCompare(b.date));
  const totalWithdrawn = payouts.reduce((s, p) => s + (parseFloat(p.amount) || 0), 0);
  const totalPayoutFees = payouts.reduce((s, p) => s + (parseFloat(p.fee) || 0), 0);

  // Payout-aware phase calculator: equity curve accounts for withdrawals reducing balance
  const calcPhaseWithPayouts = (phaseEntries, startBalance, phasePayouts) => {
    const base = calcPhase(phaseEntries, startBalance);
    // Build equity curve with payout drops injected
    let running = startBalance;
    let peak = startBalance;
    const curve = []; // { date, balance, isPayout, payoutAmt }
    // Merge entries + payouts into a unified timeline
    const timeline = [
      ...phaseEntries.map(e => ({ type: "entry", date: e.date, pnl: (parseFloat(e.pnl) || 0) - (parseFloat(e.commissions) || 0) })),
      ...(phasePayouts || []).map(p => ({ type: "payout", date: p.date, amount: parseFloat(p.amount) || 0, fee: parseFloat(p.fee) || 0 })),
    ].sort((a, b) => a.date.localeCompare(b.date));
    for (const item of timeline) {
      if (item.type === "entry") {
        running += item.pnl;
        if (running > peak) peak = running;
        curve.push({ date: item.date, balance: running, isPayout: false });
      } else {
        // Payout: balance drops, but the trailing drawdown floor does NOT drop
        running -= (item.amount + item.fee);
        curve.push({ date: item.date, balance: running, isPayout: true, payoutAmt: item.amount });
      }
    }
    return { ...base, payoutCurve: curve };
  };

  const evalStats = calcPhase(evalEntries, config.accountSize);
  const fundedStats = passedDate
    ? calcPhaseWithPayouts(fundedEntries, config.fundedStartBalance || config.accountSize, payouts)
    : null;

  // Detect breach: trailing DD >= maxLossLimit
  const activePhaseForBreach = passedDate ? fundedStats : evalStats;
  const activeLimitForBreach = passedDate ? (config.funded?.maxLossLimit || 0) : (config.eval?.maxLossLimit || 0);
  const autoIsBreached = config.isArchived || (activePhaseForBreach ? activePhaseForBreach.trailingDrawdown >= activeLimitForBreach : false);

  // Today's entry for daily loss check
  const active = passedDate ? fundedEntries : evalEntries;
  const todayEntry = active[active.length - 1];
  const todayPnl = todayEntry ? (parseFloat(todayEntry.pnl) || 0) - (parseFloat(todayEntry.commissions) || 0) : 0;

  // Funded: count qualifying profit days (days >= minDailyProfit)
  const qualifyingProfitDays = fundedEntries.filter(e => {
    const net = (parseFloat(e.pnl) || 0) - (parseFloat(e.commissions) || 0);
    return net >= (fundedRules.minDailyProfit || 0);
  }).length;

  // Active phase rules
  const activePhase = config.phase || "eval";
  const activeRules = activePhase === "funded" ? fundedRules : evalRules;
  const activeStats = activePhase === "funded" ? fundedStats : evalStats;

  const rules = activePhase === "eval" ? {
    maxLossLimit: activeStats ? activeStats.trailingDrawdown < evalRules.maxLossLimit : true,
    profitTarget: evalStats.cumPnl >= evalRules.profitTarget,
    consistency: evalRules.consistencyRule === 0 || evalStats.consistencyPct <= evalRules.consistencyRule || evalStats.cumPnl <= 0,
  } : {
    maxLossLimit: fundedStats ? fundedStats.trailingDrawdown < fundedRules.maxLossLimit : true,
    minProfitDays: qualifyingProfitDays >= (fundedRules.minProfitDays || 0),
  };

  return {
    phase: activePhase,
    passedDate,
    evalStats, fundedStats,
    evalRules, fundedRules,
    activeStats, activeRules,
    todayPnl,
    qualifyingProfitDays,
    rules,
    // Eval progress
    evalProfitTargetPct: Math.min(100, Math.max(0, (evalStats.cumPnl / (evalRules.profitTarget || 1)) * 100)),
    // Combined totals
    totalCumPnl: evalStats.cumPnl + (fundedStats?.cumPnl || 0),
    totalDaysTraded: evalStats.daysTraded + (fundedStats?.daysTraded || 0),
    // Payouts
    payouts, totalWithdrawn, totalPayoutFees,
    // Breach
    isBreached: autoIsBreached,
  };
};

// Compute personal account balance from entries + deposits
const calcPersonalBalance = (entries, config) => {
  if (!config) return null;
  const sorted = [...entries].sort((a, b) => a.date.localeCompare(b.date));
  let balance = config.startingBalance || 0;
  const curve = [];
  for (const e of sorted) {
    const net = (parseFloat(e.pnl) || 0) - (parseFloat(e.commissions) || 0);
    const deposit = parseFloat(e.cashDeposit) || 0;
    balance += net + deposit;
    curve.push({ date: e.date, balance, net, deposit });
  }
  return { currentBalance: balance, curve, startingBalance: config.startingBalance };
};

const parsePnl = (str) => {
  if (!str) return 0;
  const clean = str.toString().replace(/[$,\s]/g, "");
  const isNeg = clean.includes("(");
  const n = parseFloat(clean.replace(/[()]/g, ""));
  return isNeg ? -Math.abs(n) : n;
};

// Futures contract multipliers — $-value per 1 point move per contract
const FUTURES_MULTIPLIERS = {
  MES: 5, ES: 50, MNQ: 2, NQ: 20, MYM: 0.5, YM: 5,
  MBT: 5, BTC: 5, MGC: 10, GC: 100, SIL: 1000, SI: 5000,
  MCL: 100, CL: 1000, RTY: 10, M2K: 10,
};
const getMultiplier = (symbol) => {
  const base = symbol?.replace(/[HMUZ]\d+$/, "").toUpperCase();
  return FUTURES_MULTIPLIERS[base] || 1;
};

// Legacy fallback parser (original tab-separated format)
const parseTrades = (raw) => {
  if (!raw || !raw.trim()) return [];
  const lines = raw.trim().split("\n").filter(l => l.trim());
  const result = [];
  for (const line of lines) {
    const cols = line.split("\t").map(c => c.trim());
    if (cols.length < 8) continue;
    if (cols[0] === "Symbol" || cols[0] === "") continue;
    const pnl = parsePnl(cols[7]);
    if (isNaN(pnl) && cols[7] !== "$0.00" && cols[7] !== "$(0.00)") continue;
    result.push({
      symbol: cols[0], qty: parseInt(cols[1]) || 0,
      buyPrice: parseFloat(cols[2]) || 0, buyTime: cols[3],
      duration: cols[4], sellTime: cols[5],
      sellPrice: parseFloat(cols[6]) || 0, pnl,
    });
  }
  return result;
};

// AI-powered universal broker parser
// ── Deterministic IBKR Flex Query CSV parser ─────────────────────────────────
// Handles the exact IBKR format: FifoPnlRealized + AssetClass + TradePrice + Buy/Sell
// Supports overnight traders: closing fills with no matching entry in this file
// are treated as carry-forwards (opened in prior session) — P&L is preserved,
// entry price/time is left empty, notes="overnight-carry" is set.
// Returns array of trades or null if format not recognised
const parseIbkrCsv = (raw) => {
  try {
    const lines = raw.trim().split(/\r?\n/).filter(l => l.trim());
    if (lines.length < 2) return null;

    // Parse CSV header — handle quoted fields
    const parseRow = (line) => {
      const result = []; let cur = ''; let inQ = false;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"') { inQ = !inQ; }
        else if (ch === ',' && !inQ) { result.push(cur.trim()); cur = ''; }
        else { cur += ch; }
      }
      result.push(cur.trim());
      return result;
    };

    const headers = parseRow(lines[0]).map(h => h.replace(/"/g,'').trim());

    // Check this is IBKR format
    const hasIbkr = headers.includes('FifoPnlRealized') && headers.includes('AssetClass') && headers.includes('Buy/Sell');
    if (!hasIbkr) return null;

    const idx = (name) => headers.indexOf(name);
    const iAsset  = idx('AssetClass');
    const iSym    = idx('Symbol');
    const iQty    = idx('Quantity');
    const iPnl    = idx('FifoPnlRealized');
    const iComm   = idx('IBCommission');
    const iDt     = idx('DateTime');
    const iOt     = idx('OrderType');
    const iPrice  = idx('TradePrice');
    const iMult   = idx('Multiplier');

    // Parse all futures rows
    const futures = [];
    for (let i = 1; i < lines.length; i++) {
      const cols = parseRow(lines[i]).map(c => c.replace(/"/g,'').trim());
      if (cols[iAsset] !== 'FUT') continue;
      futures.push({
        sym:   cols[iSym]   || '',
        qty:   parseInt(cols[iQty])   || 0,
        pnl:   parseFloat(cols[iPnl]) || 0,
        comm:  parseFloat(cols[iComm])|| 0,
        dt:    cols[iDt]    || '',
        ot:    cols[iOt]    || 'MKT',
        price: parseFloat(cols[iPrice]) || 0,
        mult:  parseFloat(cols[iMult])  || 5,
      });
    }

    if (futures.length === 0) return null;

    // FIFO round-trip matching
    // BUY  qty>0 pnl=0  → long entry
    // SELL qty<0 pnl≠0  → long exit  → completed long trade
    // SELL qty<0 pnl=0  → short entry
    // BUY  qty>0 pnl≠0  → short exit → completed short trade
    const openLongs = [], openShorts = [], trades = [];

    const getDuration = (dtA, dtB) => {
      try {
        // Parse "YYYYMMDD HHMMSS" format
        const p = (s) => {
          const clean = s.replace(/[^0-9]/g, '');
          if (clean.length < 8) return NaN;
          const yr=+clean.slice(0,4), mo=+clean.slice(4,6)-1, dy=+clean.slice(6,8);
          const hr=+clean.slice(8,10)||0, mn=+clean.slice(10,12)||0, sc=+clean.slice(12,14)||0;
          return Date.UTC(yr, mo, dy, hr, mn, sc);
        };
        const secs = Math.abs(Math.round((p(dtB) - p(dtA)) / 1000));
        if (!secs || isNaN(secs)) return { secs: 0, str: '0s' };
        const m = Math.floor(secs/60), s = secs%60;
        return { secs, str: m > 0 ? m+'m'+s+'s' : s+'s' };
      } catch { return { secs: 0, str: '0s' }; }
    };

    for (const r of futures) {
      if (r.qty > 0 && r.pnl === 0) {
        openLongs.push(r);
      } else if (r.qty < 0 && r.pnl !== 0) {
        const entry = openLongs.shift();
        // When no matching entry exists, this is an overnight carry-forward (opened in prior session's CSV).
        // PNL from FifoPnlRealized is still accurate; entry price/time are unknown for this file.
        const isCarry = !entry;
        const dur = isCarry ? { str: 'prior-session', secs: 86400 } : getDuration(entry.dt, r.dt);
        // FIX: Store GROSS p&l so calcNetPnl (gross - commission) yields the correct net.
        // FifoPnlRealized is already net — using it directly caused double-deduction.
        // For carry-forwards (no entry data) we must use FifoPnlRealized as-is and zero commission.
        const grossPnlL = isCarry ? r.pnl : Math.round((r.price - entry.price) * r.mult * Math.abs(r.qty) * 100) / 100;
        const commL     = isCarry ? 0     : Math.round((Math.abs(r.comm) + Math.abs(entry.comm)) * 100) / 100;
        trades.push({
          symbol: r.sym, qty: Math.abs(r.qty), direction: 'long',
          buyPrice: entry ? entry.price : 0,
          buyTime: entry ? entry.dt : '',          // empty = entry was in prior session's CSV
          sellPrice: r.price, sellTime: r.dt,
          pnl: grossPnlL, commission: commL,
          orderType: r.ot === 'LMT' || r.ot === 'LIMIT' ? 'LMT' : r.ot === 'STP' || r.ot === 'STOP' ? 'STP' : 'MKT',
          multiplier: r.mult,
          notes: isCarry ? 'overnight-carry' : '',
          duration: dur.str, durationSecs: dur.secs,
        });
      } else if (r.qty < 0 && r.pnl === 0) {
        openShorts.push(r);
      } else if (r.qty > 0 && r.pnl !== 0) {
        const entry = openShorts.shift();
        // When no matching entry exists, this is an overnight carry-forward (opened in prior session's CSV).
        // PNL from FifoPnlRealized is still accurate; entry price/time are unknown for this file.
        const isCarry = !entry;
        const dur = isCarry ? { str: 'prior-session', secs: 86400 } : getDuration(entry.dt, r.dt);
        // FIX: Store GROSS p&l so calcNetPnl (gross - commission) yields the correct net.
        const grossPnlS = isCarry ? r.pnl : Math.round(((entry ? entry.price : r.price) - r.price) * r.mult * Math.abs(r.qty) * 100) / 100;
        const commS     = isCarry ? 0     : Math.round((Math.abs(r.comm) + Math.abs(entry.comm)) * 100) / 100;
        trades.push({
          symbol: r.sym, qty: Math.abs(r.qty), direction: 'short',
          buyPrice: r.price, buyTime: r.dt,
          sellPrice: entry ? entry.price : 0,
          sellTime: entry ? entry.dt : '',     // empty = entry was in prior session's CSV
          pnl: grossPnlS, commission: commS,
          orderType: r.ot === 'LMT' || r.ot === 'LIMIT' ? 'LMT' : r.ot === 'STP' || r.ot === 'STOP' ? 'STP' : 'MKT',
          multiplier: r.mult,
          notes: isCarry ? 'overnight-carry' : '',
          duration: dur.str, durationSecs: dur.secs,
        });
      }
      // qty>0 pnl=0 already handled (long entry); qty<0 pnl=0 = short entry queued for FIFO match.
      // rows with qty<0 pnl=0 and no close in this file = still-open position → correctly left in openShorts and not emitted.
    }

    return trades.length > 0 ? trades : null;
  } catch { return null; }
};

const parseTradesWithAI = async (raw, ai) => {
  const brokerHint = BROKER_PRESETS[ai?.brokerPreset || 'none']?.hint || '';
  const prompt = `You are a futures trade parser. Given raw broker export data, extract all COMPLETED round-trip trades.
${brokerHint ? `
BROKER CONTEXT: ${brokerHint}
` : ''}
IBKR FLEX QUERY SPECIFIC RULES (apply when data has FifoPnlRealized column):
The key insight for IBKR data: FifoPnlRealized is ONLY non-zero on the CLOSING fill of a round-trip.
- A row with Quantity > 0 (BUY) and FifoPnlRealized = 0 → LONG ENTRY (opening a long position)
- A row with Quantity < 0 (SELL) and FifoPnlRealized ≠ 0 → LONG EXIT (closing a long, completing a long trade)
- A row with Quantity < 0 (SELL) and FifoPnlRealized = 0 → SHORT ENTRY (opening a short position)
- A row with Quantity > 0 (BUY) and FifoPnlRealized ≠ 0 → SHORT EXIT (closing a short, completing a short trade)
Match each CLOSING fill to its most recent ENTRY fill of the same symbol using FIFO order.
Skip ALL rows where AssetClass is not FUT — ignore CASH, forex hedges (e.g. USD.CAD), STK, OPT rows entirely.
Skip rows where FifoPnlRealized = 0 AND there is no matching close in this file — these are still-open positions.

OVERNIGHT CARRY-FORWARD RULE (critical for overnight traders):
A CLOSING fill (FifoPnlRealized ≠ 0) with NO matching ENTRY fill in this file = position was opened in a PRIOR session's CSV.
DO NOT skip these — include them as completed trades with:
  - buyTime or sellTime = empty string "" (entry timestamp is unknown/in prior file)
  - buyPrice or sellPrice = 0 (entry price is unknown/in prior file), whichever is the entry leg
  - Use FifoPnlRealized as pnl — it is correct regardless of missing entry data
  - Set notes = "overnight-carry" to flag this trade

GENERAL RULES:
1. Include completed round-trip trades where BOTH entry and exit are present in this file
2. Also include overnight-carry trades: closing fills (FifoPnlRealized ≠ 0) with no matching entry in this file (see OVERNIGHT CARRY-FORWARD RULE above)
3. Use FifoPnlRealized directly as pnl when available — do not recalculate
4. TradePrice on the entry row = buyPrice for longs OR sellPrice for shorts
5. TradePrice on the exit row = sellPrice for longs OR buyPrice for shorts
   Example long: entry BUY TradePrice=6778.5 → buyPrice=6778.5, exit SELL TradePrice=6780.5 → sellPrice=6780.5
   Example short: entry SELL TradePrice=6755.0 → sellPrice=6755.0, exit BUY TradePrice=6753.5 → buyPrice=6753.5
6. If TradePrice is missing or 0, use 0 for that price field — the P&L from FifoPnlRealized is still correct
7. SKIP only rows where FifoPnlRealized = 0 AND no matching close exists in this file — these are still-open positions carried to the next session
8. IBCommission is negative in IBKR data — use absolute value, sum both legs for commission field (use 0 if entry leg unknown for carry-forward)
9. Duration = seconds between entry DateTime and exit DateTime. For overnight-carry trades with no entry time, use 86400 (24h placeholder) and duration string "prior-session"
10. Use these multipliers: MES=$5/pt, ES=$50/pt, MNQ=$2/pt, NQ=$20/pt, MYM=$0.50/pt, YM=$5/pt, MGC=$10/pt, GC=$100/pt, MCL=$100/pt, CL=$1000/pt, RTY=$10/pt, M2K=$10/pt
11. For partial fills on the same order, combine into one trade

CRITICAL OUTPUT RULE: Return ONLY a raw JSON array. No markdown. No code fences. No backticks of any kind. No explanation before or after. No trailing commas. All property names and string values must use double quotes only. Your response must begin with [ and end with ]. Anything other than a valid JSON array will break the parser. Each trade object must have exactly these fields:
{
  "symbol": "MESH6",
  "qty": 1,
  "direction": "long",
  "buyPrice": 6778.50,
  "buyTime": "20260311 132635",
  "sellTime": "20260311 132808",
  "sellPrice": 6780.50,
  "pnl": 8.76,
  "duration": "1m33s",
  "durationSecs": 93,
  "orderType": "LMT",
  "commission": 1.24,
  "multiplier": 5,
  "notes": ""
}

direction must be "long" (bought first, sold to close) or "short" (sold first, bought to close).
durationSecs is the number of seconds between entry and exit as an integer.
If a trade is a short (sold first, bought to close), buyPrice is the exit, sellPrice is the entry, and pnl should still be correctly signed (positive = profit).
orderType: "MKT", "LMT", or "STP" — copy the Order Type field exactly. LMT=limit order, STP=stop order, MKT=market order (use MKT if field is missing).
commission: total commission/fee for this round-trip as a positive number (e.g. 1.24). Sum both legs. Use IB Commission or Comm/Fee column if present.
multiplier: contract point value (e.g. 5 for MES, 50 for ES, 2 for MNQ, 20 for NQ). Use Multiplier field if present, otherwise infer from symbol.
notes: any Notes/Codes flags from the broker (e.g. "P" for partial, "M" for manual). Empty string if none.

RAW BROKER DATA:
${raw.slice(0, 12000)}`;

  const clean = await aiRequestText(ai, {
    max_tokens: 8000,
    timeoutMs: 60000,
    messages: [{ role: 'user', content: prompt }],
  });
  // ── Robust JSON repair pipeline — handles all known AI output failure modes ──
  const robustParseJson = (text) => {
    let s = text.trim();
    // 1. Strip markdown fences
    s = s.replace(/^```(?:json)?\s*/i, '').replace(/\s*```\s*$/, '').trim();
    // 2. Find array start — required. End optional (truncation recovery).
    let aStart = s.indexOf('[');
    if (aStart === -1) { const ob = s.indexOf('{'); if (ob === -1) throw new Error("No JSON content in AI response"); s = '[' + s.slice(ob); aStart = 0; }
    else { s = s.slice(aStart); }
    const aEnd = s.lastIndexOf(']');
    if (aEnd !== -1) s = s.slice(0, aEnd + 1);
    // 3. Direct parse — fastest path
    try { const r = JSON.parse(s); if (Array.isArray(r)) return r; } catch {}
    // 4. Clean control chars — string-aware (never touch chars inside quoted strings)
    const cleanCtrl = (t) => {
      let out = '', inStr = false, esc = false;
      for (const ch of t) {
        if (esc) { out += ch; esc = false; continue; }
        if (inStr && ch === '\\') { out += ch; esc = true; continue; }
        if (ch === '"') { inStr = !inStr; out += ch; continue; }
        if (inStr && ch.charCodeAt(0) < 32) { out += (ch === '\t' ? ' ' : ch === '\r' ? '' : ' '); continue; }
        out += ch;
      }
      return out;
    };
    s = cleanCtrl(s);
    // 5. Fix trailing commas before } or ]
    s = s.replace(/,([\s\n\r]*[}\]])/g, '$1');
    // 6. Try again after cleaning
    try { const r = JSON.parse(s); if (Array.isArray(r)) return r; } catch {}
    // 7. Object-by-object extraction (last resort — recovers partial truncated responses)
    const objects = [];
    let depth = 0, si = null;
    for (let i = 0; i < s.length; i++) {
      if (s[i] === '{') { if (depth === 0) si = i; depth++; }
      else if (s[i] === '}') {
        depth--;
        if (depth === 0 && si !== null) {
          const obj = s.slice(si, i + 1);
          for (const attempt of [obj, obj.replace(/,([\s\n\r]*})/g, '$1')]) {
            try { const o = JSON.parse(attempt); if (o && typeof o === 'object' && o.symbol) { objects.push(o); break; } } catch {}
          }
          si = null;
        }
      }
    }
    if (objects.length > 0) return objects;
    throw new Error("Could not parse trade data from AI response — try again");
  };
  const trades = robustParseJson(clean);
  if (!Array.isArray(trades)) throw new Error("Expected array");
  return trades.map(t => ({
    symbol: t.symbol || "",
    qty: Number(t.qty) || 0,
    direction: t.direction === "short" ? "short" : "long",
    buyPrice: Number(t.buyPrice) || 0,
    buyTime: t.buyTime || "",
    duration: t.duration || "",
    durationSecs: Number(t.durationSecs) || 0,
    sellTime: t.sellTime || "",
    sellPrice: Number(t.sellPrice) || 0,
    pnl: Number(t.pnl) || 0,
    orderType: (() => { const v = String(t.orderType||'').toUpperCase().trim(); if (['LMT','LIMIT'].includes(v)) return 'LMT'; if (['STP','STOP','STP LMT','STOP LIMIT'].includes(v)) return 'STP'; return 'MKT'; })(),
    commission: Number(t.commission) || 0,
    multiplier: Number(t.multiplier) || 0,
    notes: String(t.notes || '').trim(),
  }));
};

// Returns local→EST offset in minutes (positive = local is ahead of EST)
const getEstOffsetMinutes = () => {
  try {
    const now = new Date();
    const estStr = now.toLocaleString('en-US', { timeZone: 'America/New_York' });
    const locStr = now.toLocaleString('en-US');
    return Math.round((new Date(locStr) - new Date(estStr)) / 60000);
  } catch { return 0; }
};

const getSession = (timeStr, tzLock = false) => {
  if (!timeStr) return "Unknown";
  const str = timeStr.trim();

  let h, mn;

  // Method 1: YYYYMMDD HHMMSS or YYYYMMDD;HHMMSS (IBKR style) e.g. "20260311 132635"
  const ibkrMatch = str.match(/^\d{8}[;\s](\d{2})(\d{2})\d{2}$/);
  if (ibkrMatch) {
    h = parseInt(ibkrMatch[1]); mn = parseInt(ibkrMatch[2]);
  }

  // Method 2: ISO with T separator e.g. "2026-03-11T09:30:00" or "2026-03-11T09:30:00Z"
  if (h === undefined) {
    const isoMatch = str.match(/T(\d{1,2}):(\d{2})(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?$/i);
    if (isoMatch) { h = parseInt(isoMatch[1]); mn = parseInt(isoMatch[2]); }
  }

  // Method 3: HH:MM or H:MM with optional :SS and optional AM/PM anywhere in string
  if (h === undefined) {
    const hhmm = str.match(/\b(\d{1,2}):(\d{2})(?::\d{2})?\s*(AM|PM)?/i);
    if (hhmm) {
      h = parseInt(hhmm[1]); mn = parseInt(hhmm[2]);
      const ampm = (hhmm[3] || "").toLowerCase();
      if (ampm === "pm" && h !== 12) h += 12;
      else if (ampm === "am" && h === 12) h = 0;
    }
  }

  // Method 4: bare HHMMSS or HHMM string
  if (h === undefined) {
    const digits = str.replace(/\D/g, "");
    if (digits.length >= 6) { h = parseInt(digits.slice(0,2)); mn = parseInt(digits.slice(2,4)); }
    else if (digits.length === 4) { h = parseInt(digits.slice(0,2)); mn = parseInt(digits.slice(2,4)); }
  }

  if (h === undefined || isNaN(h) || isNaN(mn) || h > 23 || mn > 59) return "Unknown";

  let t = h * 60 + mn;
  if (tzLock) t = ((t - getEstOffsetMinutes()) % 1440 + 1440) % 1440;
  if (t >= 1080) return "Asian Session (6PM–12AM)";
  if (t < 570)   return "London Session (12AM–9:30AM)";
  if (t < 720)   return "NY Open (9:30AM–12PM)";
  if (t < 900)   return "Afternoon Deadzone (12–3PM)";
  if (t < 960)   return "Power Hour (3–4PM)";
  return "After Hours (4–6PM)";
};

// Strip futures contract month+year suffix so MESH6, MESH26, MESM5 all group as MES
const normalizeSymbol = (sym) => {
  if (!sym) return sym;
  return sym.replace(/^([A-Z]+)\s*[FGHJKMNQUVXZ]\d{1,2}$/, '$1').trim() || sym;
};

const calcAnalytics = (trades, tzLock = false) => {
  if (!trades || trades.length === 0) return null;
  // FIX: tNet computes net P&L per trade (gross minus per-trade commission).
  // All win/loss classification, equity curve, and breakdowns use net so that
  // a trade costing more in fees than it made is correctly classified as a loser.
  const tNet = (t) => t.pnl - (t.commission || 0);
  const winners = trades.filter(t => tNet(t) > 0);
  const losers  = trades.filter(t => tNet(t) < 0);
  const totalPnL    = trades.reduce((s, t) => s + t.pnl, 0);          // gross — for GROSS P&L display tile
  const totalNetPnL = trades.reduce((s, t) => s + tNet(t), 0);        // net  — for NET P&L display tile
  const avgWin  = winners.length ? winners.reduce((s, t) => s + tNet(t), 0) / winners.length : 0;
  const avgLoss = losers.length  ? losers.reduce((s, t)  => s + tNet(t), 0) / losers.length  : 0;
  const winRate = trades.length ? (winners.length / trades.length) * 100 : 0;
  const grossWin  = winners.reduce((s, t) => s + tNet(t), 0);
  const grossLoss = Math.abs(losers.reduce((s, t) => s + tNet(t), 0));
  const profitFactor = grossLoss > 0 ? grossWin / grossLoss : grossWin > 0 ? Infinity : null;
  const avgQty = trades.reduce((s, t) => s + t.qty, 0) / trades.length;

  let running = 0;
  // FIX: equity curve uses net P&L so the chart matches what the trader actually kept
  const equityCurve = trades.map((t) => { running += tNet(t); return { pnl: running, trade: t }; });

  // maxDD: peak-to-trough drawdown on the equity curve.
  // Initialise runPeak to the first curve point so sessions that open negative
  // are handled correctly (not anchored to 0 as a false peak).
  let runPeak = equityCurve.length ? equityCurve[0].pnl : 0;
  let maxDD = 0;
  for (const pt of equityCurve) {
    if (pt.pnl > runPeak) runPeak = pt.pnl;
    const dd = runPeak - pt.pnl;
    if (dd > maxDD) maxDD = dd;
  }

  const bySymbol = {};
  for (const t of trades) {
    const sym = normalizeSymbol(t.symbol);
    if (!bySymbol[sym]) bySymbol[sym] = { trades: 0, pnl: 0, wins: 0 };
    bySymbol[sym].trades++;
    bySymbol[sym].pnl += tNet(t);
    if (tNet(t) > 0) bySymbol[sym].wins++;
  }

  const bySession = {};
  for (const t of trades) {
    // Use exit time for session attribution so all trades are bucketed by WHEN they closed.
    // For longs: exit = sellTime. For shorts: exit = buyTime (buy-to-cover).
    // Overnight-carry trades may have an empty entry timestamp — fall back to whichever is populated.
    const exitTime = t.direction === 'short'
      ? (t.buyTime  || t.sellTime)
      : (t.sellTime || t.buyTime);
    const sess = getSession(exitTime, tzLock);
    if (!bySession[sess]) bySession[sess] = { trades: 0, pnl: 0, wins: 0 };
    bySession[sess].trades++;
    bySession[sess].pnl += tNet(t);
    if (tNet(t) > 0) bySession[sess].wins++;
  }

  let maxConsecWins = 0, maxConsecLoss = 0, curW = 0, curL = 0;
  for (const t of trades) {
    if (tNet(t) > 0) { curW++; curL = 0; maxConsecWins = Math.max(maxConsecWins, curW); }
    else if (tNet(t) < 0) { curL++; curW = 0; maxConsecLoss = Math.max(maxConsecLoss, curL); }
    else { curW = 0; curL = 0; }
  }

  // Long vs Short breakdown
  const byDirection = { long: { trades: 0, pnl: 0, wins: 0, losses: 0, grossWin: 0, grossLoss: 0 }, short: { trades: 0, pnl: 0, wins: 0, losses: 0, grossWin: 0, grossLoss: 0 } };
  for (const t of trades) {
    const dir = t.direction === "short" ? "short" : "long";
    const nt = tNet(t);
    byDirection[dir].trades++;
    byDirection[dir].pnl += nt;
    if (nt > 0) { byDirection[dir].wins++; byDirection[dir].grossWin += nt; }
    else if (nt < 0) { byDirection[dir].losses++; byDirection[dir].grossLoss += Math.abs(nt); }
  }

  // Duration analysis — bucket trades into time ranges
  const parseDurationSecs = (t) => {
    if (t.durationSecs > 0) return t.durationSecs;
    // Try to parse from duration string e.g. "21s", "3m", "1h 2m"
    const d = (t.duration || "").toLowerCase();
    const h = (d.match(/(\d+)h/) || [])[1] || 0;
    const m = (d.match(/(\d+)m/) || [])[1] || 0;
    const s = (d.match(/(\d+)s/) || [])[1] || 0;
    return Number(h)*3600 + Number(m)*60 + Number(s);
  };
  const DURATION_BUCKETS = [
    { key: "scalp",    label: "Scalp", subtitle: "< 1 min",    test: s => s > 0 && s < 60 },
    { key: "quick",    label: "Quick", subtitle: "1–5 min",    test: s => s >= 60 && s < 300 },
    { key: "intraday", label: "Swing", subtitle: "5–30 min",   test: s => s >= 300 && s < 1800 },
    { key: "hold",     label: "Hold",  subtitle: "30 min–2 hr",test: s => s >= 1800 && s < 7200 },
    { key: "extended", label: "Ext.",  subtitle: "> 2 hr",     test: s => s >= 7200 },
  ];
  const byDuration = {};
  for (const b of DURATION_BUCKETS) byDuration[b.key] = { trades: 0, pnl: 0, wins: 0, losses: 0, grossWin: 0, grossLoss: 0, avgSecs: 0, totalSecs: 0 };
  for (const t of trades) {
    const secs = parseDurationSecs(t);
    const bucket = DURATION_BUCKETS.find(b => b.test(secs));
    if (!bucket) continue;
    const b = byDuration[bucket.key];
    const nt = tNet(t);
    b.trades++; b.pnl += nt; b.totalSecs += secs;
    if (nt > 0) { b.wins++; b.grossWin += nt; }
    else if (nt < 0) { b.losses++; b.grossLoss += Math.abs(nt); }
  }
  for (const b of DURATION_BUCKETS) {
    if (byDuration[b.key].trades > 0) byDuration[b.key].avgSecs = byDuration[b.key].totalSecs / byDuration[b.key].trades;
  }

  // Winner vs loser average duration
  const winnerSecs = winners.map(parseDurationSecs).filter(s => s > 0);
  const loserSecs = losers.map(parseDurationSecs).filter(s => s > 0);
  const avgWinDuration = winnerSecs.length ? winnerSecs.reduce((a,b)=>a+b,0)/winnerSecs.length : 0;
  const avgLossDuration = loserSecs.length ? loserSecs.reduce((a,b)=>a+b,0)/loserSecs.length : 0;
  // Behavioral edge checks
  const ordered = [...trades].map((t, i) => ({ t, i, ts: parseMaybeDate(t.sellTime) || parseMaybeDate(t.buyTime) }))
    .sort((a, b) => {
      const at = Number.isNaN(a.ts) ? 1 : 0;
      const bt = Number.isNaN(b.ts) ? 1 : 0;
      if (at !== bt) return at - bt;
      if (!Number.isNaN(a.ts) && !Number.isNaN(b.ts) && a.ts !== b.ts) return a.ts - b.ts;
      return a.i - b.i;
    })
    .map(x => x.t);

  const sliceStats = (arr) => {
    const total = arr.length;
    const wins = arr.filter(t => tNet(t) > 0);
    const pnl = arr.reduce((s, t) => s + tNet(t), 0);
    const avgPnl = total ? pnl / total : 0;
    const avgQty = total ? arr.reduce((s, t) => s + (Number(t.qty) || 0), 0) / total : 0;
    const winRate = total ? (wins.length / total) * 100 : 0;
    return { total, wins: wins.length, pnl, avgPnl, avgQty, winRate };
  };

  const afterLossTrades = ordered.filter((t, i) => i > 0 && tNet(ordered[i - 1]) < 0);
  const afterWinTrades  = ordered.filter((t, i) => i > 0 && tNet(ordered[i - 1]) > 0);
  const first3Trades = ordered.slice(0, 3);
  const restTrades = ordered.slice(3);

  const afterLoss = sliceStats(afterLossTrades);
  const afterWin = sliceStats(afterWinTrades);
  const first3 = sliceStats(first3Trades);
  const rest = sliceStats(restTrades);

  // Order type breakdown
  const byOrderType = { MKT: { trades:0, pnl:0, wins:0, grossWin:0, grossLoss:0 }, LMT: { trades:0, pnl:0, wins:0, grossWin:0, grossLoss:0 }, STP: { trades:0, pnl:0, wins:0, grossWin:0, grossLoss:0 } };
  for (const t of trades) {
    const ot = (t.orderType === 'LMT' || t.orderType === 'STP') ? t.orderType : 'MKT';
    const nt = tNet(t);
    byOrderType[ot].trades++;
    byOrderType[ot].pnl += nt;
    if (nt > 0) { byOrderType[ot].wins++; byOrderType[ot].grossWin += nt; }
    else if (nt < 0) { byOrderType[ot].grossLoss += Math.abs(nt); }
  }

  // Commission totals
  const totalCommission = trades.reduce((s, t) => s + (t.commission || 0), 0);

  return {
    total: trades.length, winners: winners.length, losers: losers.length, breakeven: trades.filter(t => tNet(t) === 0).length,
    totalPnL, totalNetPnL, avgWin, avgLoss, winRate, profitFactor,
    largestWin:  winners.length ? Math.max(...winners.map(t => tNet(t))) : 0,
    largestLoss: losers.length  ? Math.min(...losers.map(t  => tNet(t))) : 0,
    equityCurve, maxDD, bySymbol, bySession,
    maxConsecWins, maxConsecLoss, avgQty,
    byDirection, byDuration, DURATION_BUCKETS,
    avgWinDuration, avgLossDuration,
    afterLoss, afterWin, first3, rest,
    byOrderType, totalCommission,
    // R-based metrics — use avg loss as 1R baseline (all net-based)
    avgWinR:  (avgLoss !== 0 && winners.length) ? avgWin / Math.abs(avgLoss) : null,
    avgLossR: (avgLoss !== 0 && losers.length)  ? avgLoss / Math.abs(avgLoss) : null,
    netR: avgLoss !== 0 ? totalNetPnL / Math.abs(avgLoss) : null,
    expectancy: avgLoss !== 0 ? ((winners.length / trades.length) * (avgWin / Math.abs(avgLoss))) + ((losers.length / trades.length) * -1) : null,
  };
};

const ENTRY_SCHEMA_VERSION = 4;

const emptyEntry = () => ({
  schemaVersion: ENTRY_SCHEMA_VERSION,
  id: Date.now(), date: new Date().toISOString().split("T")[0],
  instruments: [], bias: "", moods: [], grade: "", pnl: "",
  commissions: "", chartScreenshots: [],
  marketNotes: "", lessonsLearned: "", mistakes: "", improvements: "",
  bestTrade: "", worstTrade: "", rules: "", tomorrow: "", reinforceRule: "", sessionMistakes: [],
  rawTradeData: "", parsedTrades: [], cashDeposit: "", rawCsvFile: null,
  // AI persisted fields (included in backups)
  aiRewrites: {}, // { fieldKey: rewrittenText }
  aiRewritesMeta: {}, // { fieldKey: { srcHash, ts } }
  aiNoteSummary: "", // consolidated narrative
  aiNoteSummaryHash: "",
  // v4: Dual score system (Feature 3)
  executionScore: null,  // 1–10 or null
  decisionScore: null,   // 1–10 or null
  // v4: Mistake cost accounting (Feature 4)
  mistakeCosts: {},      // { mistakeTag: number | null }
  mistakeCostNotes: "",  // optional freetext
  // v4: Parse validation log (Feature 2)
  parseValidationLog: [], // [{ ts, hardRejected, softFlagged, userRejected, accepted }]
});

// ── Data health / migrations ───────────────────────────────────────────────
const tradeHash = (t) => [t.symbol, t.buyTime, t.sellTime, t.qty, t.buyPrice, t.sellPrice, t.pnl, t.direction].join('|');

const normalizeTrades = (trades) => {
  const changes = [];
  if (!Array.isArray(trades)) return { trades: [], changes: ['parsedTrades was not an array (reset to empty).'], changed: true };

  const out = [];
  const seen = new Set();
  let removedInvalid = 0, removedDupes = 0, fixed = 0;

  for (const raw of trades) {
    if (!raw || typeof raw !== 'object') { removedInvalid++; continue; }
    const t = { ...raw };

    t.symbol = (t.symbol || '').toString().trim();
    t.direction = t.direction === 'short' ? 'short' : 'long';
    t.qty = Number(t.qty) || 0;
    t.buyPrice = Number(t.buyPrice) || 0;
    t.sellPrice = Number(t.sellPrice) || 0;
    t.pnl = Number(t.pnl) || 0;
    t.buyTime = (t.buyTime || '').toString();
    t.sellTime = (t.sellTime || '').toString();
    t.duration = (t.duration || '').toString();
    t.durationSecs = Number(t.durationSecs) || 0;

    const invalid = !t.symbol || t.qty <= 0 || (!t.buyTime && !t.sellTime);
    if (invalid) { removedInvalid++; continue; }

    // Normalize durationSecs if missing but duration string exists
    if (!t.durationSecs && t.duration) {
      const d = t.duration.toLowerCase();
      const h = Number((d.match(/(\d+)h/) || [])[1] || 0);
      const m = Number((d.match(/(\d+)m/) || [])[1] || 0);
      const s = Number((d.match(/(\d+)s/) || [])[1] || 0);
      const secs = h * 3600 + m * 60 + s;
      if (secs > 0) { t.durationSecs = secs; fixed++; }
    }

    // De-dupe
    const hsh = tradeHash(t);
    if (seen.has(hsh)) { removedDupes++; continue; }
    seen.add(hsh);

    out.push(t);
  }

  if (removedInvalid) changes.push(`${removedInvalid} invalid trade${removedInvalid === 1 ? '' : 's'} removed`);
  if (removedDupes) changes.push(`${removedDupes} duplicate trade${removedDupes === 1 ? '' : 's'} removed`);
  if (fixed) changes.push(`${fixed} trade field${fixed === 1 ? '' : 's'} normalized`);

  return { trades: out, changes, changed: changes.length > 0 };
};

const normalizeEntry = (entry) => {
  const changes = [];
  const e = { ...entry };

  if (!e.schemaVersion) { e.schemaVersion = ENTRY_SCHEMA_VERSION; changes.push('schemaVersion added'); }
  if (e.schemaVersion < ENTRY_SCHEMA_VERSION) {
    e.schemaVersion = ENTRY_SCHEMA_VERSION;
    changes.push(`entry upgraded to v${ENTRY_SCHEMA_VERSION}`);
  }

  // Ensure arrays
  if (!Array.isArray(e.instruments)) { e.instruments = e.instruments ? [e.instruments] : []; changes.push('instruments normalized'); }
  if (!Array.isArray(e.moods)) { e.moods = e.moods ? [e.moods] : []; changes.push('moods normalized'); }
  if (!Array.isArray(e.sessionMistakes)) { e.sessionMistakes = e.sessionMistakes ? [e.sessionMistakes] : []; changes.push('sessionMistakes normalized'); }
  if (!Array.isArray(e.chartScreenshots)) { e.chartScreenshots = []; changes.push('chartScreenshots normalized'); }

  // AI persisted fields
  if (!e.aiRewrites || typeof e.aiRewrites !== 'object' || Array.isArray(e.aiRewrites)) { e.aiRewrites = {}; changes.push('aiRewrites normalized'); }
  if (!e.aiRewritesMeta || typeof e.aiRewritesMeta !== 'object' || Array.isArray(e.aiRewritesMeta)) { e.aiRewritesMeta = {}; changes.push('aiRewritesMeta normalized'); }
  if (typeof e.aiNoteSummary !== 'string') { e.aiNoteSummary = e.aiNoteSummary ? String(e.aiNoteSummary) : ''; changes.push('aiNoteSummary normalized'); }
  if (typeof e.aiNoteSummaryHash !== 'string') { e.aiNoteSummaryHash = e.aiNoteSummaryHash ? String(e.aiNoteSummaryHash) : ''; changes.push('aiNoteSummaryHash normalized'); }

  // v4: Dual scores (Feature 3)
  const normScore = (val, name) => {
    if (val === null || val === undefined) return null;
    const n = Number(val);
    if (!Number.isFinite(n) || n < 1 || n > 10) { changes.push(`${name} out of range, reset`); return null; }
    return Math.round(n);
  };
  e.executionScore = normScore(e.executionScore, 'executionScore');
  e.decisionScore  = normScore(e.decisionScore,  'decisionScore');

  // v4: Mistake costs (Feature 4)
  if (!e.mistakeCosts || typeof e.mistakeCosts !== 'object' || Array.isArray(e.mistakeCosts)) {
    e.mistakeCosts = {}; changes.push('mistakeCosts normalized');
  } else {
    const cleaned = {};
    for (const [k, v] of Object.entries(e.mistakeCosts)) {
      if (!k) continue; // drop empty-string keys
      if (v === null || v === undefined) { cleaned[k] = null; continue; }
      const n = Number(v);
      if (!Number.isFinite(n) || n < 0) { changes.push(`mistakeCosts[${k}] invalid, reset`); cleaned[k] = null; }
      else cleaned[k] = n;
    }
    // Prune keys for tags no longer in sessionMistakes
    for (const k of Object.keys(cleaned)) {
      if (!e.sessionMistakes.includes(k)) delete cleaned[k];
    }
    e.mistakeCosts = cleaned;
  }
  if (typeof e.mistakeCostNotes !== 'string') {
    e.mistakeCostNotes = e.mistakeCostNotes ? String(e.mistakeCostNotes) : ''; changes.push('mistakeCostNotes normalized');
  }

  // v4: Parse validation log (Feature 2)
  if (!Array.isArray(e.parseValidationLog)) { e.parseValidationLog = []; changes.push('parseValidationLog normalized'); }

  // Trades
  const nt = normalizeTrades(e.parsedTrades);
  if (nt.changed) { e.parsedTrades = nt.trades; changes.push(...nt.changes.map(c => `Trades: ${c}`)); }

  // Ensure strings for numeric fields (safe)
  if (typeof e.pnl === 'number') { e.pnl = e.pnl.toFixed(2); changes.push('pnl normalized'); }
  if (typeof e.commissions === 'number') { e.commissions = e.commissions.toFixed(2); changes.push('commissions normalized'); }
  if (typeof e.cashDeposit === 'number') { e.cashDeposit = e.cashDeposit.toFixed(2); changes.push('cashDeposit normalized'); }

  return { entry: e, changes, changed: changes.length > 0 };
};

const normalizeEntries = (entries) => {
  if (!Array.isArray(entries)) return { entries: [], report: { changed: true, summary: 'Entries data was not an array (reset).', details: ['Entries root was not an array.'] } };
  const out = [];
  const details = [];
  let changed = false;
  let upgraded = 0;

  for (const e of entries) {
    const r = normalizeEntry(e || {});
    out.push(r.entry);
    if (r.changed) {
      changed = true;
      upgraded++;
      details.push(`${r.entry.date || r.entry.id}: ${r.changes.join(', ')}`);
    }
  }

  const summary = changed ? `Data health update: ${upgraded} entr${upgraded === 1 ? 'y' : 'ies'} normalized.` : '';
  return { entries: out, report: { changed, summary, details } };
};

const calcNetPnl = (entry) => {
  const gross = parseFloat(entry.pnl) || 0;
  const comm = parseFloat(entry.commissions) || 0;
  return gross - comm;
};
// Alias kept for any inline usage
const netPnl = calcNetPnl;

// Profit factor display helpers — handles Infinity (perfect day: 0 losses)
const fmtPF = (pf) => { if (pf == null) return "—"; if (!isFinite(pf)) return "∞"; return pf.toFixed(2); };
const pfColor = (pf) => { if (pf == null) return "#64748b"; if (!isFinite(pf)) return "#4ade80"; return pf >= 1 ? "#4ade80" : "#f87171"; };

const MiniCurve = ({ curve, w = 100, h = 32 }) => {
  if (!curve || curve.length < 2) return null;
  const vals = curve.map(p => p.pnl);
  const min = Math.min(0, ...vals), max = Math.max(0, ...vals);
  const range = max - min || 1;
  const color = vals[vals.length - 1] >= 0 ? "#4ade80" : "#f87171";
  const pts = vals.map((v, i) => `${(i / (vals.length - 1)) * w},${h - ((v - min) / range) * h}`).join(" ");
  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
};

// Shared equity curve with Y-axis dollar labels
const EquityCurveChart = ({ values, dots = false, height = 90, gradientId = "ecFill", dangerLine = null, payoutMarkers = null }) => {
  // payoutMarkers: array of index positions in `values` where a payout drop occurred
  if (!values || values.length < 2) return null;
  const W = 800, PAD_L = 62, PAD_R = 8, PAD_T = 8, PAD_B = 4;
  const H = height;
  const chartW = W - PAD_L - PAD_R;
  const chartH = H - PAD_T - PAD_B;
  // Extend min to include dangerLine if it falls below existing range
  const rawMin = Math.min(0, ...values);
  const min = dangerLine !== null ? Math.min(rawMin, dangerLine) : rawMin;
  const max = Math.max(0, ...values);
  const range = max - min || 1;
  const toY  = v => PAD_T + chartH - ((v - min) / range) * chartH;
  const toX  = i => PAD_L + (values.length === 1 ? chartW / 2 : (i / (values.length - 1)) * chartW);
  const zeroY = toY(0);
  const lastVal = values[values.length - 1];
  // Danger zone: is the equity within 25% of the danger line from above?
  const inDangerZone = dangerLine !== null && lastVal <= dangerLine + Math.abs(dangerLine) * 0.25;
  const lineColor = lastVal >= 0 ? (inDangerZone ? "#fbbf24" : "#4ade80") : "#f87171";
  const pts = values.map((v, i) => `${toX(i)},${toY(v)}`).join(" ");
  const fillPath = `M${toX(0)},${zeroY} ` + values.map((v, i) => `L${toX(i)},${toY(v)}`).join(" ") + ` L${toX(values.length - 1)},${zeroY} Z`;
  const dangerY = dangerLine !== null ? toY(dangerLine) : null;

  // Smart Y-axis ticks — 4 levels, nicely rounded
  const fmtK = v => {
    const abs = Math.abs(v);
    if (abs >= 1000) return `${v < 0 ? "-" : ""}$${(abs / 1000).toFixed(abs % 1000 === 0 ? 0 : 1)}k`;
    return `${v < 0 ? "-" : ""}$${abs.toFixed(0)}`;
  };
  const niceStep = raw => {
    const mag = Math.pow(10, Math.floor(Math.log10(Math.abs(raw) || 1)));
    const norm = raw / mag;
    const nice = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
    return nice * mag;
  };
  const step = niceStep(range / 4);
  const tickMin = Math.ceil(min / step) * step;
  const ticks = [];
  for (let t = tickMin; t <= max + step * 0.01; t += step) {
    const y = toY(t);
    if (y >= PAD_T - 2 && y <= H - PAD_B + 2) ticks.push({ v: t, y });
  }
  // Always include zero
  if (!ticks.find(t => t.v === 0)) ticks.push({ v: 0, y: zeroY });

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ display: "block", overflow: "visible" }}>
      <defs>
        {/* Profit: journal gradient stroke */}
        <linearGradient id={`${gradientId}-stroke-profit`} x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#38bdf8" />
          <stop offset="50%" stopColor="#818cf8" />
          <stop offset="100%" stopColor="#c084fc" />
        </linearGradient>
        {/* Loss: red-violet-red stroke */}
        <linearGradient id={`${gradientId}-stroke-loss`} x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#f87171" />
          <stop offset="50%" stopColor="#c084fc" />
          <stop offset="100%" stopColor="#f87171" />
        </linearGradient>
        {/* Profit fill: journal gradient tint */}
        <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.12" />
          <stop offset="50%" stopColor="#818cf8" stopOpacity="0.1" />
          <stop offset="100%" stopColor="#c084fc" stopOpacity="0.05" />
        </linearGradient>
        {/* Loss fill: red tint */}
        <linearGradient id={`${gradientId}-loss`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#f87171" stopOpacity="0.04" />
          <stop offset="100%" stopColor="#f87171" stopOpacity="0.14" />
        </linearGradient>
        {dangerLine !== null && (
          <linearGradient id={`${gradientId}-danger`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f87171" stopOpacity="0.08" />
            <stop offset="100%" stopColor="#f87171" stopOpacity="0.18" />
          </linearGradient>
        )}
      </defs>
      {/* Y-axis grid lines + labels */}
      {ticks.map(({ v, y }) => (
        <g key={v}>
          <line x1={PAD_L} y1={y} x2={W - PAD_R} y2={y}
            stroke={v === 0 ? "#1e3a5f" : "#0f1a2a"}
            strokeWidth={v === 0 ? 1 : 0.75}
            strokeDasharray={v === 0 ? "4,4" : "2,4"} />
          <text x={PAD_L - 5} y={y + 3.5} textAnchor="end"
            style={{ fontSize: 9, fill: v === 0 ? "#475569" : "#334155", fontFamily: "DM Mono, monospace" }}>
            {fmtK(v)}
          </text>
        </g>
      ))}
      {/* Danger zone: red fill below the limit line */}
      {dangerLine !== null && dangerY !== null && (
        <rect x={PAD_L} y={dangerY} width={chartW} height={Math.max(0, H - PAD_B - dangerY)}
          fill={`url(#${gradientId}-danger)`} />
      )}

      {/* Danger line */}
      {dangerLine !== null && dangerY !== null && (
        <g>
          <line x1={PAD_L} y1={dangerY} x2={W - PAD_R} y2={dangerY}
            stroke="#f87171" strokeWidth="1.5" strokeDasharray="6,3" opacity="0.85" />
          <rect x={W - PAD_R - 68} y={dangerY - 9} width={66} height={13} rx="2" fill="#1a0505" opacity="0.9" />
          <text x={W - PAD_R - 35} y={dangerY + 1} textAnchor="middle"
            style={{ fontSize: 8, fill: "#f87171", fontFamily: "DM Mono, monospace", fontWeight: 600, letterSpacing: "0.05em" }}>
            {fmtK(dangerLine)} LIMIT
          </text>
        </g>
      )}
      {/* Payout drop markers — blue vertical dashed line + "$X" label */}
      {payoutMarkers && payoutMarkers.map((m, idx) => {
        const x = toX(m.index);
        const yTop = toY(m.balanceBefore);
        const yBot = toY(m.balanceAfter);
        return (
          <g key={idx}>
            <line x1={x} y1={yTop} x2={x} y2={yBot}
              stroke="#3b82f6" strokeWidth="2" strokeDasharray="3,2" opacity="0.9" />
            <circle cx={x} cy={yTop} r="3" fill="#3b82f6" opacity="0.85" />
            <circle cx={x} cy={yBot} r="3" fill="#60a5fa" opacity="0.85" />
            <rect x={x + 5} y={yBot - 8} width={48} height={12} rx="2" fill="#0f1a2e" opacity="0.92" />
            <text x={x + 9} y={yBot + 1} style={{ fontSize: 8, fill: "#60a5fa", fontFamily: "DM Mono, monospace", fontWeight: 600 }}>
              {`-$${Math.abs(m.amount).toLocaleString()}`}
            </text>
          </g>
        );
      })}
      {/* Area fill — journal gradient for profit, red tint for loss */}
      <path d={fillPath} fill={lastVal >= 0 ? `url(#${gradientId})` : `url(#${gradientId}-loss)`} />
      {/* Line — 1.5px gradient stroke */}
      <polyline points={pts} fill="none"
        stroke={lastVal >= 0 ? (inDangerZone ? "#fbbf24" : `url(#${gradientId}-stroke-profit)`) : `url(#${gradientId}-stroke-loss)`}
        strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
      {/* End dot — marks latest value */}
      {(() => {
        const lx = toX(values.length - 1);
        const ly = toY(lastVal);
        const dotColor = lastVal >= 0 ? (inDangerZone ? "#fbbf24" : "#c084fc") : "#f87171";
        return (
          <g>
            <circle cx={lx} cy={ly} r="6" fill="none" stroke={dotColor} strokeWidth="0.75" opacity="0.4" />
            <circle cx={lx} cy={ly} r="3.5" fill={dotColor} stroke="#0f1729" strokeWidth="2" />
          </g>
        );
      })()}
      {/* Mid-point dots (optional — small) */}
      {dots && values.slice(0, -1).map((v, i) => (
        <circle key={i} cx={toX(i)} cy={toY(v)} r="2.5"
          fill={v >= 0 ? "#818cf8" : "#f87171"} stroke="#0f1729" strokeWidth="1.5" opacity="0.7" />
      ))}
    </svg>
  );
};

// Renders AI markdown text as clean styled JSX — no raw ** or ### ever shown
function RenderAI({ text }) {
  if (!text) return null;
  const lines = text.split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    // Heading: **TEXT** on its own line, or ### TEXT, or ## TEXT
    const boldHeading = line.match(/^\*\*(.+?)\*\*\s*$/);
    const hashHeading = line.match(/^#{1,3}\s+(.+)$/);
    if (boldHeading || hashHeading) {
      const label = (boldHeading?.[1] || hashHeading?.[1] || "").trim();
      out.push(
        <div key={i} style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", fontWeight: 700, marginTop: out.length ? 18 : 0, marginBottom: 8 }}>
          {label}
        </div>
      );
    } else if (line.match(/^[-•]\s+/)) {
      // Bullet
      const content = line.replace(/^[-•]\s+/, "").replace(/\*\*(.+?)\*\*/g, "$1");
      out.push(
        <div key={i} style={{ display: "flex", gap: 8, marginBottom: 4 }}>
          <span style={{ color: "#3b82f6", flexShrink: 0, marginTop: 2 }}>·</span>
          <span style={{ fontSize: 13, color: "#cbd5e1", lineHeight: 1.7 }}>{content}</span>
        </div>
      );
    } else if (line.trim() === "") {
      if (out.length) out.push(<div key={i} style={{ height: 4 }} />);
    } else {
      // Normal paragraph — strip inline ** bold markers
      const cleaned = line.replace(/\*\*(.+?)\*\*/g, (_, m) => m);
      out.push(
        <div key={i} style={{ fontSize: 13, color: "#cbd5e1", lineHeight: 1.7, marginBottom: 4 }}>{cleaned}</div>
      );
    }
    i++;
  }
  return <div>{out}</div>;
}


function ChartScreenshotZone({ screenshots = [], onChange }) {
  const [dragging, setDragging] = useState(false);
  const [lightbox, setLightbox] = useState(null);
  const zoneRef = useRef(null);

  const addFromDataUrl = (dataUrl) => {
    onChange([...screenshots, { id: Date.now() + Math.random(), src: dataUrl }]);
  };

  const handlePaste = (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        const reader = new FileReader();
        reader.onload = ev => addFromDataUrl(ev.target.result);
        reader.readAsDataURL(file);
      }
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const files = [...(e.dataTransfer?.files || [])].filter(f => f.type.startsWith("image/"));
    files.forEach(file => {
      const reader = new FileReader();
      reader.onload = ev => addFromDataUrl(ev.target.result);
      reader.readAsDataURL(file);
    });
  };

  const remove = (id) => onChange(screenshots.filter(s => s.id !== id));

  return (
    <div style={{ marginTop: 24 }}>
      <div style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 10 }}>CHART SCREENSHOTS</div>

      {/* Paste zone */}
      {/* Paste zone — compact when images already added */}
      <div ref={zoneRef} tabIndex={0} onPaste={handlePaste}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        style={{ border: `1.5px dashed ${dragging ? "#3b82f6" : "#1e3a5f"}`, borderRadius: 6, padding: screenshots.length ? "10px 16px" : "20px", textAlign: "center", cursor: "text", outline: "none", background: dragging ? "#0a1628" : "transparent", transition: "all .15s", marginBottom: screenshots.length ? 12 : 0, display: "flex", alignItems: "center", justifyContent: "center", gap: 10 }}
        onClick={() => zoneRef.current?.focus()}>
        <span style={{ fontSize: screenshots.length ? 14 : 22, opacity: 0.4 }}>📋</span>
        <div>
          <div style={{ fontSize: screenshots.length ? 11 : 12, color: "#64748b", lineHeight: 1.5 }}>
            Click here, then <span style={{ color: "#3b82f6" }}>Ctrl+V</span> / <span style={{ color: "#3b82f6" }}>⌘V</span> to paste{screenshots.length ? " another" : " a screenshot"}
          </div>
          {!screenshots.length && <div style={{ fontSize: 10, color: "#64748b", marginTop: 3 }}>Or drag & drop an image file</div>}
        </div>
      </div>

      {/* Full-width images stacked */}
      {screenshots.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {screenshots.map((s, i) => (
            <div key={s.id} style={{ position: "relative", borderRadius: 5, overflow: "hidden", border: "1px solid #1e3a5f", cursor: "zoom-in" }}
              onClick={() => setLightbox(s.src)}
              onMouseEnter={e => e.currentTarget.style.borderColor = "#3b82f6"}
              onMouseLeave={e => e.currentTarget.style.borderColor = "#1e3a5f"}>
              <img src={s.src} alt={`Chart ${i + 1}`} style={{ display: "block", width: "100%", height: "auto", maxHeight: 420, objectFit: "contain", background: "#060810" }} />
              <button onClick={e => { e.stopPropagation(); remove(s.id); }}
                style={{ position: "absolute", top: 8, right: 8, background: "rgba(0,0,0,0.75)", color: "#f87171", border: "1px solid rgba(248,113,113,0.3)", borderRadius: 3, padding: "3px 8px", fontSize: 11, cursor: "pointer", fontFamily: "inherit" }}>✕ remove</button>
              <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, padding: "6px 10px", background: "linear-gradient(transparent, rgba(0,0,0,0.5))", fontSize: 9, color: "rgba(255,255,255,0.35)", letterSpacing: "0.06em" }}>CHART #{i + 1}</div>
            </div>
          ))}
        </div>
      )}

      {/* Lightbox */}
      {lightbox && (
        <div onClick={() => setLightbox(null)}
          style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.92)", zIndex: 9999, display: "flex", alignItems: "center", justifyContent: "center", cursor: "zoom-out" }}>
          <img src={lightbox} alt="Chart" style={{ maxWidth: "92vw", maxHeight: "90vh", borderRadius: 6, boxShadow: "0 0 60px rgba(0,0,0,0.8)" }} />
          <div style={{ position: "absolute", top: 20, right: 24, color: "#94a3b8", fontSize: 13, opacity: 0.6 }}>click anywhere to close</div>
        </div>
      )}
    </div>
  );
}

function AnalyticsPanel({ a, trades, pnlColor, fmtPnl, analyticsTab, setAnalyticsTab, totalFees = 0, dangerLine = null, rawCsvFile = null }) {
  const ATABS = ["overview", "by session", "trade log"];
  const [expandedTrade, setExpandedTrade] = useState(null);
  const [ecCollapsed, setEcCollapsed] = useState(false);
  // FIX: Use a.totalNetPnL (computed inside calcAnalytics from per-trade net) as the single
  // source of truth. Fall back to a.totalPnL - a.totalCommission for backward compatibility
  // with any entries saved before this fix.
  const netTotal = a.totalNetPnL != null ? a.totalNetPnL : (a.totalPnL - (a.totalCommission || 0));
  const feesDisplay = a.totalCommission > 0 ? a.totalCommission : totalFees;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        {ATABS.map(t => (
          <button key={t} onClick={() => setAnalyticsTab(t)}
            style={{ padding: "6px 12px", borderRadius: 3, fontSize: 11, cursor: "pointer", fontFamily: "inherit", letterSpacing: "0.05em", transition: "all .15s", background: analyticsTab === t ? "#1e3a5f" : "transparent", border: `1px solid ${analyticsTab === t ? "#3b82f6" : "#1e293b"}`, color: analyticsTab === t ? "#93c5fd" : "#94a3b8" }}>
            {t.toUpperCase()}
          </button>
        ))}
      </div>

      {analyticsTab === "overview" && (
        <>
          {/* P&L Summary */}
          <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 6 }}>P&L SUMMARY</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
            {[
              { l: "GROSS P&L", v: fmtPnl(a.totalPnL), c: pnlColor(a.totalPnL) },
              { l: "TOTAL FEES", v: feesDisplay > 0 ? `-$${feesDisplay.toFixed(2)}` : "—", c: "#475569", small: true },
              { l: "NET P&L", v: fmtPnl(netTotal), c: pnlColor(netTotal), highlight: true },
              { l: "LARGEST WIN", v: a.largestWin ? fmtPnl(a.largestWin) : "—", c: "#4ade80" },
              { l: "LARGEST LOSS", v: a.largestLoss ? fmtPnl(a.largestLoss) : "—", c: "#f87171" },
              { l: "MAX DRAWDOWN", v: a.maxDD > 0 ? `-$${a.maxDD.toFixed(2)}` : "—", c: "#f87171" },
            ].map(s => (
              <div key={s.l} style={{ background: s.highlight ? "#0f1a2e" : "#0f1729", border: `1px solid ${s.highlight ? "#1e3a5f" : "#1e293b"}`, borderRadius: 4, padding: "10px 12px" }}>
                <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 4 }}>{s.l}</div>
                <div style={{ fontSize: 16, color: s.c, fontWeight: 500 }}>{s.v}</div>
              </div>
            ))}
          </div>

          {/* Trade Stats */}
          <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 6, marginTop: 6 }}>TRADE STATISTICS</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
            {[
              { l: "TOTAL TRADES", v: a.total, c: "#e2e8f0" },
              { l: "WINNERS", v: a.winners, c: "#4ade80" },
              { l: "LOSERS", v: a.losers, c: "#f87171" },
              { l: "WIN RATE", v: `${a.winRate.toFixed(1)}%`, c: a.winRate >= 50 ? "#4ade80" : "#f87171" },
              { l: "PROFIT FACTOR", v: fmtPF(a.profitFactor), c: pfColor(a.profitFactor) },
              { l: "AVG CONTRACT SIZE", v: `${a.avgQty.toFixed(1)} cts`, c: "#e2e8f0" },
              { l: "AVG WIN", v: fmtPnl(a.avgWin), c: "#4ade80" },
              { l: "AVG LOSS", v: fmtPnl(a.avgLoss), c: "#f87171" },
              { l: "MAX WIN STREAK", v: `${a.maxConsecWins} trades`, c: "#4ade80" },
              { l: "MAX LOSS STREAK", v: `${a.maxConsecLoss} trades`, c: "#f87171" },
              { l: "MAX DRAWDOWN", v: a.maxDD > 0 ? `-$${a.maxDD.toFixed(2)}` : "—", c: "#f87171" },
            ].map(s => (
              <div key={s.l} style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "10px 12px" }}>
                <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 4 }}>{s.l}</div>
                <div style={{ fontSize: 16, color: s.c, fontWeight: 500 }}>{s.v}</div>
              </div>
            ))}
          </div>

          {/* Behavioral Edge Checks */}
          <div style={{ background: '#0f1729', border: '1px solid #1e293b', borderRadius: 4, padding: '14px 16px' }}>
            <div style={{ fontSize: 11, color: '#93c5fd', letterSpacing: '0.1em', marginBottom: 14 }}>BEHAVIORAL EDGE CHECKS</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10 }}>
              {[
                { label: 'AFTER A LOSS', data: a.afterLoss },
                { label: 'AFTER A WIN', data: a.afterWin },
                { label: 'FIRST 3 TRADES', data: a.first3 },
                { label: 'REST OF SESSION', data: a.rest },
              ].map(card => (
                <div key={card.label} style={{ background: '#0a0e1a', border: '1px solid #1e293b', borderRadius: 4, padding: '10px 12px' }}>
                  <div style={{ fontSize: 9, color: '#94a3b8', letterSpacing: '0.1em', marginBottom: 6 }}>{card.label}</div>
                  {card.data?.total ? (
                    <>
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, alignItems: 'baseline' }}>
                        <div style={{ fontSize: 13, color: '#e2e8f0' }}>{card.data.winRate.toFixed(0)}% WR</div>
                        <div style={{ fontSize: 13, color: card.label.includes('TRADES') || card.label.includes('SESSION') ? (card.data.pnl >= 0 ? '#4ade80' : '#f87171') : (card.data.avgPnl >= 0 ? '#4ade80' : '#f87171'), fontWeight: 600 }}>
                          {card.label === 'AFTER A LOSS' || card.label === 'AFTER A WIN'
                            ? `${card.data.avgPnl >= 0 ? '+' : '-'}$${Math.abs(card.data.avgPnl).toFixed(2)}/trade`
                            : `${card.data.pnl >= 0 ? '+' : '-'}$${Math.abs(card.data.pnl).toFixed(2)}`}
                        </div>
                      </div>
                      <div style={{ fontSize: 10, color: '#64748b', marginTop: 6 }}>n={card.data.total} · avg size {card.data.avgQty.toFixed(1)} cts</div>
                    </>
                  ) : (
                    <div style={{ fontSize: 12, color: '#64748b' }}>Not enough data</div>
                  )}
                </div>
              ))}
            </div>

            {feesDisplay > 0 && (
              <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid #1e293b', display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12 }}>
                <div>
                  <div style={{ fontSize: 9, color: '#94a3b8', letterSpacing: '0.1em', marginBottom: 4 }}>FEES IMPACT</div>
                  <div style={{ fontSize: 12, color: '#e2e8f0' }}>${(feesDisplay / Math.max(a.total || 1, 1)).toFixed(2)} per trade</div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: 12, color: netTotal >= 0 ? '#4ade80' : '#f87171', fontWeight: 600 }}>{fmtPnl(netTotal)} net</div>
                  <div style={{ fontSize: 10, color: '#64748b' }}>{a.totalPnL !== 0 ? `${Math.min(100, Math.abs(feesDisplay / a.totalPnL) * 100).toFixed(1)}% of gross P&L` : ''}</div>
                </div>
              </div>
            )}
          </div>

          {/* Performance by Symbol */}
          {Object.keys(a.bySymbol).length > 0 && (
            <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px" }}>
              <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 14 }}>PERFORMANCE BY SYMBOL</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {Object.entries(a.bySymbol).map(([sym, s]) => {
                  const maxPnl = Math.max(...Object.values(a.bySymbol).map(x => Math.abs(x.pnl)), 1);
                  const wr = Math.round(s.wins / s.trades * 100);
                  const barW = Math.abs(s.pnl) / maxPnl * 100;
                  return (
                    <div key={sym}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 5 }}>
                        <span style={{ fontSize: 11, color: "#93c5fd", fontWeight: 500 }}>{sym}</span>
                        <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
                          <span style={{ fontSize: 10, color: wr >= 50 ? "#4ade80" : "#f87171" }}>{wr}% WR</span>
                          <span style={{ fontSize: 10, letterSpacing: 0 }}><span style={{ letterSpacing: 0 }}><span style={{ color: "#4ade80" }}>{s.wins}</span><span style={{ color: "#94a3b8" }}>/</span><span style={{ color: "#f87171" }}>{s.trades - s.wins}</span></span></span>
                          <span style={{ fontSize: 12, fontWeight: 600, color: s.pnl >= 0 ? "#4ade80" : "#f87171", minWidth: 80, textAlign: "right" }}>{s.pnl >= 0 ? "+" : "-"}${Math.abs(s.pnl).toFixed(2)}</span>
                        </div>
                      </div>
                      <div style={{ background: "#0a0e1a", borderRadius: 2, height: 5, overflow: "hidden" }}>
                        <div style={{ width: `${barW}%`, height: "100%", background: s.pnl >= 0 ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Performance by Direction (Long vs Short) */}
          {a.byDirection && (a.byDirection.long.trades > 0 || a.byDirection.short.trades > 0) && (
            <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px" }}>
              <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 14 }}>LONG vs SHORT</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                {[
                  { key: "long",  label: "Long",  emoji: "📈" },
                  { key: "short", label: "Short", emoji: "📉" },
                ].map(({ key, label, emoji }) => {
                  const d = a.byDirection[key];
                  if (!d || d.trades === 0) return null;
                  const wr = Math.round(d.wins / d.trades * 100);
                  const avgWin  = d.wins   > 0 ? (d.grossWin  / d.wins).toFixed(0)  : "0";
                  const avgLoss = d.losses > 0 ? (d.grossLoss / d.losses).toFixed(0) : "0";
                  const pf = d.grossLoss > 0 ? d.grossWin / d.grossLoss : d.grossWin > 0 ? Infinity : null;
                  // Win/loss bar: split bar showing gross wins (green) vs gross losses (red)
                  const total = d.grossWin + d.grossLoss;
                  const winBarPct  = total > 0 ? (d.grossWin  / total * 100).toFixed(1) : "50";
                  const lossBarPct = total > 0 ? (d.grossLoss / total * 100).toFixed(1) : "50";
                  return (
                    <div key={key} style={{ background: "#0a0e1a", borderRadius: 4, padding: "10px 12px" }}>
                      {/* Header */}
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                        <span style={{ fontSize: 12, color: "#e2e8f0", fontWeight: 600 }}>{emoji} {label}</span>
                        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                          <span style={{ fontSize: 10, color: wr >= 50 ? "#4ade80" : "#f87171" }}>{wr}% WR</span>
                          <span style={{ fontSize: 10 }}>
                            <span style={{ color: "#4ade80" }}>{d.wins}W</span>
                            <span style={{ color: "#475569" }}>/</span>
                            <span style={{ color: "#f87171" }}>{d.losses}L</span>
                          </span>
                          <span style={{ fontSize: 10, color: "#475569" }}>{d.trades} trades</span>
                          <span style={{ fontSize: 13, fontWeight: 700, color: d.pnl >= 0 ? "#4ade80" : "#f87171" }}>
                            {d.pnl >= 0 ? "+" : "-"}${Math.abs(d.pnl).toFixed(0)} net
                          </span>
                        </div>
                      </div>
                      {/* Win/Loss split bar */}
                      <div style={{ marginBottom: 10 }}>
                        <div style={{ display: "flex", height: 8, borderRadius: 4, overflow: "hidden", gap: 1 }}>
                          <div style={{ width: `${winBarPct}%`, background: "rgba(74,222,128,0.7)", borderRadius: "3px 0 0 3px" }} title={`Gross wins: +$${d.grossWin.toFixed(0)}`} />
                          <div style={{ width: `${lossBarPct}%`, background: "rgba(248,113,113,0.7)", borderRadius: "0 3px 3px 0" }} title={`Gross losses: -$${d.grossLoss.toFixed(0)}`} />
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
                          <span style={{ fontSize: 9, color: "#4ade80" }}>+${d.grossWin.toFixed(0)} gross wins ({winBarPct}%)</span>
                          <span style={{ fontSize: 9, color: "#f87171" }}>-${d.grossLoss.toFixed(0)} gross losses ({lossBarPct}%)</span>
                        </div>
                      </div>
                      {/* Avg stats row */}
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6 }}>
                        <div style={{ background: "#0f1729", borderRadius: 3, padding: "6px 8px", textAlign: "center" }}>
                          <div style={{ fontSize: 8, color: "#475569", letterSpacing: "0.08em", marginBottom: 2 }}>AVG WIN</div>
                          <div style={{ fontSize: 12, color: "#4ade80", fontWeight: 600 }}>+${avgWin}</div>
                        </div>
                        <div style={{ background: "#0f1729", borderRadius: 3, padding: "6px 8px", textAlign: "center" }}>
                          <div style={{ fontSize: 8, color: "#475569", letterSpacing: "0.08em", marginBottom: 2 }}>AVG LOSS</div>
                          <div style={{ fontSize: 12, color: "#f87171", fontWeight: 600 }}>{d.losses > 0 ? `-$${avgLoss}` : "—"}</div>
                        </div>
                        <div style={{ background: "#0f1729", borderRadius: 3, padding: "6px 8px", textAlign: "center" }}>
                          <div style={{ fontSize: 8, color: "#475569", letterSpacing: "0.08em", marginBottom: 2 }}>PROF. FACTOR</div>
                          <div style={{ fontSize: 12, color: pfColor(pf), fontWeight: 600 }}>{fmtPF(pf)}</div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Performance by Holding Duration */}
          {a.byDuration && Object.values(a.byDuration).some(b => b.trades > 0) && (
            <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 14 }}>
                <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em" }}>PERFORMANCE BY HOLDING TIME</div>
                {a.avgWinDuration > 0 && a.avgLossDuration > 0 && (
                  <div style={{ fontSize: 9, color: a.avgLossDuration > a.avgWinDuration ? "#f87171" : "#475569" }}>
                    {a.avgLossDuration > a.avgWinDuration ? "⚠ holding losers longer than winners" : "✓ cutting losers faster than winners"}
          </div>
          )}
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {(a.DURATION_BUCKETS || []).map(bucket => {
                  const b = a.byDuration[bucket.key];
                  if (!b || b.trades === 0) return null;
                  const wr = Math.round(b.wins / b.trades * 100);
                  const maxPnl = Math.max(...(a.DURATION_BUCKETS || []).map(bk => Math.abs(a.byDuration[bk.key]?.pnl || 0)), 1);
                  const barW = Math.abs(b.pnl) / maxPnl * 100;
                  const avg = b.pnl / b.trades;
                  const fmtSecs = (s) => !s ? "—" : s < 60 ? `${Math.round(s)}s` : s < 3600 ? `${Math.floor(s/60)}m ${Math.round(s%60)}s` : `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`;
                  return (
                    <div key={bucket.key}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 5 }}>
                        <div>
                          <span style={{ fontSize: 11, color: "#e2e8f0", fontWeight: 500 }}>{bucket.label}</span>
                          <span style={{ fontSize: 9, color: "#64748b", marginLeft: 8 }}>{bucket.subtitle} · avg hold {fmtSecs(b.avgSecs)}</span>
                        </div>
                        <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
                          <span style={{ fontSize: 10, color: wr >= 50 ? "#4ade80" : "#f87171" }}>{wr}% WR</span>
                          <span style={{ fontSize: 10 }}><span style={{ color: "#4ade80" }}>{b.wins}</span><span style={{ color: "#94a3b8" }}>/</span><span style={{ color: "#f87171" }}>{b.losses}</span></span>
                          <span style={{ fontSize: 12, fontWeight: 600, color: b.pnl >= 0 ? "#4ade80" : "#f87171", minWidth: 80, textAlign: "right" }}>{b.pnl >= 0 ? "+" : "-"}${Math.abs(b.pnl).toFixed(2)}</span>
                          <span style={{ fontSize: 9, color: "#64748b", minWidth: 60, textAlign: "right" }}>avg {avg >= 0 ? "+" : "-"}${Math.abs(avg).toFixed(2)}</span>
                        </div>
                      </div>
                      <div style={{ background: "#0a0e1a", borderRadius: 2, height: 5, overflow: "hidden" }}>
                        <div style={{ width: `${barW}%`, height: "100%", background: b.pnl >= 0 ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} />
                      </div>
                    </div>
                  );
                })}
              </div>
              {/* Winner vs loser avg hold time */}
              {a.avgWinDuration > 0 && a.avgLossDuration > 0 && (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 14, paddingTop: 14, borderTop: "1px solid #1e293b" }}>
                  {[
                    { l: "AVG WINNER HELD", v: (() => { const s = a.avgWinDuration; return s < 60 ? `${Math.round(s)}s` : s < 3600 ? `${Math.floor(s/60)}m ${Math.round(s%60)}s` : `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`; })(), c: "#4ade80" },
                    { l: "AVG LOSER HELD",  v: (() => { const s = a.avgLossDuration; return s < 60 ? `${Math.round(s)}s` : s < 3600 ? `${Math.floor(s/60)}m ${Math.round(s%60)}s` : `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`; })(), c: "#f87171" },
                  ].map(s => (
                    <div key={s.l} style={{ background: "#0a0e1a", border: "1px solid #1e293b", borderRadius: 4, padding: "8px 12px" }}>
                      <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 4 }}>{s.l}</div>
                      <div style={{ fontSize: 16, color: s.c }}>{s.v}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Order Type Breakdown */}
          {a.byOrderType && (a.byOrderType.LMT.trades > 0 || a.byOrderType.MKT.trades > 0 || a.byOrderType.STP.trades > 0) && (() => {
            const activeOTs = [a.byOrderType.LMT, a.byOrderType.STP, a.byOrderType.MKT].filter(x => x.trades > 0);
            const hasBoth = activeOTs.length > 1;
            return (
              <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px" }}>
                <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 12 }}>ORDER TYPE BREAKDOWN</div>
                <div style={{ display: "grid", gridTemplateColumns: activeOTs.length === 3 ? "1fr 1fr 1fr" : activeOTs.length === 2 ? "1fr 1fr" : "1fr", gap: 10 }}>
                  {[
                    { key: "LMT", label: "LIMIT"  },
                    { key: "STP", label: "STOP"   },
                    { key: "MKT", label: "MARKET" },
                  ].filter(({ key }) => (a.byOrderType[key] || {}).trades > 0).map(({ key, label }) => {
                    const d = a.byOrderType[key];
                    const profitable = d.pnl >= 0;
                    const color  = profitable ? "#4ade80" : "#f87171";
                    const bg     = profitable ? "rgba(74,222,128,0.07)" : "rgba(248,113,113,0.07)";
                    const border = profitable ? "rgba(74,222,128,0.2)"  : "rgba(248,113,113,0.2)";
                    const wr = d.trades ? Math.round(d.wins / d.trades * 100) : 0;
                    const pf = fmtPF(d.grossLoss > 0 ? d.grossWin / d.grossLoss : d.grossWin > 0 ? Infinity : null);
                    const avgPnl = d.trades ? d.pnl / d.trades : 0;
                    return (
                      <div key={key} style={{ background: bg, border: `1px solid ${border}`, borderRadius: 6, padding: "12px 14px" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                          <span style={{ fontSize: 10, color, letterSpacing: "0.1em", fontWeight: 600 }}>{label}</span>
                          <span style={{ fontSize: 9, color: "#475569" }}>{d.trades} trade{d.trades !== 1 ? "s" : ""}</span>
                        </div>
                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                          {[
                            { l: "WIN RATE", v: `${wr}%`, c: wr >= 50 ? "#4ade80" : "#f87171" },
                            { l: "NET P&L", v: `${d.pnl >= 0 ? "+" : "-"}$${d.pnl >= 0 ? d.pnl.toFixed(0) : (-d.pnl).toFixed(0)}`, c: d.pnl >= 0 ? "#4ade80" : "#f87171" },
                            { l: "AVG/TRADE", v: `${avgPnl >= 0 ? "+" : "-"}$${avgPnl >= 0 ? avgPnl.toFixed(0) : (-avgPnl).toFixed(0)}`, c: avgPnl >= 0 ? "#4ade80" : "#f87171" },
                            { l: "PROF FACTOR", v: pf, c: "#e2e8f0" },
                          ].map(s => (
                            <div key={s.l} style={{ background: "#0a0e1a", borderRadius: 4, padding: "6px 8px" }}>
                              <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 3 }}>{s.l}</div>
                              <div style={{ fontSize: 13, color: s.c, fontWeight: 500 }}>{s.v}</div>
                            </div>
                          ))}
                        </div>
                        {hasBoth && (
                          <div style={{ marginTop: 8, fontSize: 9, color: "#334155", borderTop: "1px solid #1e293b", paddingTop: 8 }}>
                            {key === "LMT"
                              ? (a.byOrderType.LMT.pnl >= 0 ? "✓ Limit orders profitable" : "⚠ Limit entries losing — review placement")
                              : key === "STP"
                              ? (a.byOrderType.STP.pnl >= 0 ? "✓ Stops capturing profit" : "⚠ Stops closing at a loss — review placement")
                              : (a.byOrderType.MKT.pnl >= 0 ? "✓ Market orders profitable" : "⚠ Consider replacing MKT with LMT entries")
                            }
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
                {a.totalCommission > 0 && (
                  <div style={{ marginTop: 10, padding: "8px 12px", background: "#0a0e1a", borderRadius: 4, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em" }}>TOTAL COMMISSIONS</span>
                    <span style={{ fontSize: 12, color: "#f87171" }}>-${a.totalCommission.toFixed(2)}</span>
          </div>
          )}
              </div>
            );
          })()}

          {/* Equity Curve — collapsible */}
          <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, overflow: "hidden" }}>
            <div onClick={() => setEcCollapsed(p => !p)}
              style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 14px", cursor: "pointer", userSelect: "none" }}>
              <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em" }}>EQUITY CURVE</div>
              <span style={{ fontSize: 10, color: "#475569", transition: "transform .2s", display: "inline-block", transform: ecCollapsed ? "rotate(-90deg)" : "rotate(0deg)" }}>▾</span>
            </div>
            {!ecCollapsed && (
              <div style={{ padding: "0 14px 10px" }}>
                {(() => {
                  const vals = a.equityCurve.map(p => p.pnl);
                  const chartVals = vals.length === 1 ? [0, vals[0]] : vals;
                  return <EquityCurveChart values={chartVals} height={90} gradientId="ec1" dangerLine={dangerLine} />;
                })()}
              </div>
            )}
          </div>
        </>
      )}


      {analyticsTab === "by session" && (
        <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "16px" }}>
          <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 14 }}>P&L BY TIME OF DAY</div>
          {Object.entries(a.bySession).sort((a, b) => b[1].pnl - a[1].pnl).map(([sess, s]) => {
            const maxPnl = Math.max(...Object.values(a.bySession).map(x => Math.abs(x.pnl)));
            return (
              <div key={sess} style={{ marginBottom: 14 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5, fontSize: 12 }}>
                  <span style={{ color: "#e2e8f0" }}>{sess}</span>
                  <span style={{ letterSpacing: 0 }}><span style={{ color: "#4ade80" }}>{s.wins}</span><span style={{ color: "#94a3b8" }}>/</span><span style={{ color: "#f87171" }}>{s.trades - s.wins}</span></span><span style={{ color: "#94a3b8", marginLeft: 6 }}>{Math.round(s.wins / s.trades * 100)}% WR</span>
                  <span style={{ color: s.pnl >= 0 ? "#4ade80" : "#f87171" }}>{fmtPnl(s.pnl)}</span>
                </div>
                <div style={{ background: "#0a0e1a", borderRadius: 2, height: 4, overflow: "hidden" }}>
                  <div style={{ width: `${Math.min(Math.abs(s.pnl) / maxPnl * 100, 100)}%`, height: "100%", background: s.pnl >= 0 ? "#4ade80" : "#f87171", borderRadius: 2 }} />
                </div>
              </div>
            );
          })}
          {(() => {
            const sorted = Object.entries(a.bySession).sort((a, b) => b[1].pnl - a[1].pnl);
            const best = sorted[0];
            const worst = sorted[sorted.length - 1];
            return (
              <div style={{ marginTop: 14, background: "#0a0e1a", border: "1px solid #1e3a5f", borderRadius: 4, padding: "10px 14px", fontSize: 11, color: "#94a3b8", lineHeight: 1.7 }}>
                <span style={{ color: "#3b82f6" }}>💡 INSIGHT: </span>
                Best window is <span style={{ color: "#4ade80" }}>{best[0]}</span> ({fmtPnl(best[1].pnl)}).
                {worst[1].pnl < 0 && <> Consider reducing size or avoiding <span style={{ color: "#f87171" }}>{worst[0]}</span> ({fmtPnl(worst[1].pnl)}).</>}
              </div>
            );
          })()}
        </div>
      )}
      {analyticsTab === "trade log" && (() => {
        const tNetL      = (t) => Number.isFinite(t.pnl) ? (t.pnl - (t.commission||0)) : 0;
        const wins       = trades.filter(t => tNetL(t) > 0);
        const losses     = trades.filter(t => tNetL(t) < 0);
        const totalNet   = trades.reduce((s,t) => s + tNetL(t), 0);
        const avgWin     = wins.length   ? wins.reduce((s,t)=>s+tNetL(t),0)/wins.length   : 0;
        const avgLoss    = losses.length ? losses.reduce((s,t)=>s+tNetL(t),0)/losses.length: 0;
        const bestTrade  = wins.length   ? Math.max(...wins.map(t=>tNetL(t)))   : 0;
        const worstTrade = losses.length ? Math.min(...losses.map(t=>tNetL(t))) : 0;
        const winRate    = trades.length ? Math.round(wins.length/trades.length*100) : 0;
        return (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

            {/* ── SUMMARY HEADER ── */}
            <div style={{ background: "#0a0e1a", border: "1px solid #1e293b", borderRadius: 6, padding: "14px 18px" }}>
              <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.15em", marginBottom: 12 }}>SESSION SUMMARY</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10, marginBottom: 12 }}>
                {[
                  { label: "TRADES",    value: trades.length,                                           color: "#94a3b8" },
                  { label: "WIN RATE",  value: `${winRate}%`,                                           color: winRate >= 50 ? "#4ade80" : "#f87171" },
                  { label: "NET P&L",   value: `${totalNet >= 0 ? "+" : ""}$${Math.abs(totalNet).toFixed(2)}`, color: totalNet > 0 ? "#4ade80" : totalNet < 0 ? "#f87171" : "#94a3b8" },
                  { label: "W / L",     value: `${wins.length} / ${losses.length}`,                    color: "#94a3b8" },
                ].map(s => (
                  <div key={s.label} style={{ background: "#0f1729", borderRadius: 4, padding: "8px 10px" }}>
                    <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.12em", marginBottom: 4 }}>{s.label}</div>
                    <div style={{ fontSize: 14, color: s.color, fontWeight: 600 }}>{s.value}</div>
                  </div>
                ))}
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10 }}>
                {[
                  { label: "AVG WIN",    value: avgWin  ? `+$${avgWin.toFixed(2)}`   : "—", color: "#4ade80" },
                  { label: "AVG LOSS",   value: avgLoss ? `-$${Math.abs(avgLoss).toFixed(2)}` : "—", color: "#f87171" },
                  { label: "BEST TRADE", value: bestTrade  ? `+$${bestTrade.toFixed(2)}`        : "—", color: "#4ade80" },
                  { label: "WORST TRADE",value: worstTrade ? `-$${Math.abs(worstTrade).toFixed(2)}` : "—", color: "#f87171" },
                ].map(s => (
                  <div key={s.label} style={{ background: "#0f1729", borderRadius: 4, padding: "8px 10px" }}>
                    <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.12em", marginBottom: 4 }}>{s.label}</div>
                    <div style={{ fontSize: 13, color: s.color, fontWeight: 600 }}>{s.value}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* ── TRADE LIST ── */}
            <div style={{ background: "#0a0e1a", border: "1px solid #1e293b", borderRadius: 6, overflow: "hidden" }}>

              {/* Table header + export buttons */}
              <div style={{ display: "flex", alignItems: "center", padding: "8px 14px", borderBottom: "1px solid #1e293b", gap: 8 }}>
                <div style={{ display: "grid", gridTemplateColumns: "28px 36px 80px 50px 100px 100px 60px 88px 88px", flex: 1, fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em" }}>
                  <div></div>
                  <div>#</div>
                  <div>SYMBOL</div>
                  <div style={{ textAlign: "center" }}>DIR</div>
                  <div style={{ textAlign: "right" }}>ENTRY</div>
                  <div style={{ textAlign: "right" }}>EXIT</div>
                  <div style={{ textAlign: "center" }}>TYPE</div>
                  <div style={{ textAlign: "right" }}>DURATION</div>
                  <div style={{ textAlign: "right" }}>NET P&L</div>
                </div>
                <button onClick={() => {
                  const header = "Symbol,Qty,Buy Price,Buy Time,Duration,Sell Time,Sell Price,Gross P&L,Commission,Net P&L,Order Type,Direction,Notes\n";
                  const rows = trades.map(t =>
                    `${t.symbol||""},${t.qty||""},${Number.isFinite(t.buyPrice)?t.buyPrice:""},${t.buyTime||""},${t.duration||""},${t.sellTime||""},${Number.isFinite(t.sellPrice)?t.sellPrice:""},${Number.isFinite(t.pnl)?t.pnl.toFixed(2):""},${Number.isFinite(t.commission)?(t.commission||0).toFixed(2):"0"},${Number.isFinite(t.pnl)?((t.pnl-(t.commission||0)).toFixed(2)):""},${t.orderType||"MKT"},${t.direction||""},${t.notes||""}`
                  ).join("\n");
                  const blob = new Blob([header+rows],{type:"text/csv"});
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a"); a.href=url; a.download="trades-export.csv"; a.click(); URL.revokeObjectURL(url);
                }}
                  style={{ background:"transparent", border:"1px solid #1e293b", color:"#3b82f6", padding:"3px 10px", borderRadius:3, fontSize:9, cursor:"pointer", fontFamily:"inherit", letterSpacing:"0.06em", whiteSpace:"nowrap", flexShrink:0, transition:"all .15s" }}
                  onMouseEnter={e=>{e.currentTarget.style.borderColor="#3b82f6";e.currentTarget.style.color="#93c5fd";}}
                  onMouseLeave={e=>{e.currentTarget.style.borderColor="#1e293b";e.currentTarget.style.color="#475569";}}>↓ CSV</button>
                {rawCsvFile?.content && (
                  <button onClick={() => {
                    const blob = new Blob([rawCsvFile.content],{type:"text/csv"});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a"); a.href=url; a.download=rawCsvFile.name||"trades-original.csv"; a.click(); URL.revokeObjectURL(url);
                  }}
                    style={{ background:"transparent", border:"1px solid #1e293b", color:"#3b82f6", padding:"3px 10px", borderRadius:3, fontSize:9, cursor:"pointer", fontFamily:"inherit", letterSpacing:"0.06em", whiteSpace:"nowrap", flexShrink:0, transition:"all .15s" }}
                    onMouseEnter={e=>{e.currentTarget.style.borderColor="#4ade80";e.currentTarget.style.color="#4ade80";}}
                    onMouseLeave={e=>{e.currentTarget.style.borderColor="#1e293b";e.currentTarget.style.color="#475569";}}>↓ ORIGINAL</button>
                )}
              </div>

              {/* Expandable rows */}
              <div style={{ maxHeight: 440, overflowY: "auto" }}>
                {(() => {
                  let cumPnl = 0;
                  return trades.map((t, i) => {
                    const net    = Number.isFinite(t.pnl) ? (t.pnl - (t.commission||0)) : null;
                    const isWin  = net !== null && net > 0;
                    const isLoss = net !== null && net < 0;
                    if (net !== null) cumPnl += net;
                    const expanded = expandedTrade === i;
                    const dir = t.direction || (t.qty > 0 ? "LONG" : "SHORT") || "—";
                    const entryTime = t.buyTime?.split(" ")[1]  || t.buyTime  || "—";
                    const exitTime  = t.sellTime?.split(" ")[1] || t.sellTime || "—";
                    const rowBg = expanded ? "#0f1e38" : i % 2 === 0 ? "#0f1729" : "#0d1525";
                    const accentColor = isWin ? "#4ade80" : isLoss ? "#f87171" : "#94a3b8";
                    return (
                      <div key={i}>
                        {/* Collapsed row */}
                        <div onClick={() => setExpandedTrade(expanded ? null : i)}
                          style={{ display: "grid", gridTemplateColumns: "28px 36px 80px 50px 100px 100px 60px 88px 88px", padding: "9px 14px", borderBottom: expanded ? "none" : "1px solid #0a0e1a", fontSize: 12, background: rowBg, cursor: "pointer", transition: "background .12s" }}
                          onMouseEnter={e => { if (!expanded) e.currentTarget.style.background = "#131e35"; }}
                          onMouseLeave={e => { if (!expanded) e.currentTarget.style.background = rowBg; }}>
                          {/* Expand arrow */}
                          <div style={{ color: "#334155", fontSize: 9, display: "flex", alignItems: "center" }}>{expanded ? "▼" : "▶"}</div>
                          {/* Trade # */}
                          <div style={{ color: "#334155", fontSize: 11 }}>{i+1}</div>
                          {/* Symbol + win/loss dot */}
                          <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                            <div style={{ width: 6, height: 6, borderRadius: "50%", background: accentColor, flexShrink: 0 }} />
                            <span style={{ color: "#93c5fd", fontWeight: 600 }}>{t.symbol || "—"}</span>
                          </div>
                          {/* Direction badge */}
                          <div style={{ textAlign: "center" }}>
                            <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 3, letterSpacing: "0.06em",
                              background: dir === "LONG" ? "rgba(74,222,128,0.1)" : dir === "SHORT" ? "rgba(248,113,113,0.1)" : "rgba(148,163,184,0.08)",
                              color:      dir === "LONG" ? "#4ade80"              : dir === "SHORT" ? "#f87171"              : "#64748b" }}>{dir}</span>
                          </div>
                          {/* Entry price @ time */}
                          <div style={{ textAlign: "right" }}>
                            <div style={{ color: "#e2e8f0", fontSize: 12 }}>{Number.isFinite(t.buyPrice) ? `$${t.buyPrice.toFixed(2)}` : "—"}</div>
                            <div style={{ color: "#475569", fontSize: 9 }}>{entryTime}</div>
                          </div>
                          {/* Exit price @ time */}
                          <div style={{ textAlign: "right" }}>
                            <div style={{ color: "#e2e8f0", fontSize: 12 }}>{Number.isFinite(t.sellPrice) ? `$${t.sellPrice.toFixed(2)}` : "—"}</div>
                            <div style={{ color: "#475569", fontSize: 9 }}>{exitTime}</div>
                          </div>
                          {/* Order type */}
                          <div style={{ textAlign: "center" }}>
                            <span style={{ fontSize: 9, padding: "2px 5px", borderRadius: 3, letterSpacing: "0.05em",
                              background: t.orderType === "LMT" ? "rgba(147,197,253,0.12)" : t.orderType === "STP" ? "rgba(251,146,60,0.12)" : "rgba(148,163,184,0.08)",
                              color:      t.orderType === "LMT" ? "#93c5fd"                : t.orderType === "STP" ? "#fb923c"                : "#64748b" }}>{t.orderType || "MKT"}</span>
                          </div>
                          {/* Duration */}
                          <div style={{ textAlign: "right", color: "#64748b", fontSize: 11 }}>{t.duration || "—"}</div>
                          {/* Net P&L */}
                          <div style={{ textAlign: "right", color: accentColor, fontWeight: 600, fontSize: 13 }}>
                            {net !== null ? fmtPnl(net) : "—"}
                          </div>
                        </div>

                        {/* Expanded detail panel */}
                        {expanded && (
                          <div style={{ background: "#0f1e38", borderBottom: "1px solid #0a0e1a", padding: "12px 14px 14px 52px" }}>
                            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10, marginBottom: 10 }}>
                              {[
                                { label: "GROSS P&L",  value: Number.isFinite(t.pnl) ? fmtPnl(t.pnl) : "—",             color: t.pnl > 0 ? "#4ade80" : t.pnl < 0 ? "#f87171" : "#94a3b8" },
                                { label: "COMMISSION", value: (t.commission > 0 && Number.isFinite(t.commission)) ? `-$${t.commission.toFixed(2)}` : "—", color: "#f59e0b" },
                                { label: "NET P&L",    value: net !== null ? fmtPnl(net) : "—",                          color: accentColor },
                                { label: "RUNNING P&L",value: `${cumPnl >= 0 ? "+" : ""}$${Math.abs(cumPnl).toFixed(2)}`, color: cumPnl > 0 ? "#4ade80" : cumPnl < 0 ? "#f87171" : "#94a3b8" },
                              ].map(d => (
                                <div key={d.label} style={{ background: "#0a1628", borderRadius: 4, padding: "7px 10px" }}>
                                  <div style={{ fontSize: 8, color: "#334155", letterSpacing: "0.12em", marginBottom: 3 }}>{d.label}</div>
                                  <div style={{ fontSize: 13, color: d.color, fontWeight: 600 }}>{d.value}</div>
                                </div>
                              ))}
                            </div>
                            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10 }}>
                              {[
                                { label: "QTY / SIZE",   value: t.qty ?? "—",                                                          color: "#e2e8f0" },
                                { label: "ENTRY TIME",   value: entryTime,                                                             color: "#94a3b8" },
                                { label: "EXIT TIME",    value: exitTime,                                                              color: "#94a3b8" },
                                { label: "DURATION",     value: t.duration || "—",                                                    color: "#94a3b8" },
                              ].map(d => (
                                <div key={d.label} style={{ background: "#0a1628", borderRadius: 4, padding: "7px 10px" }}>
                                  <div style={{ fontSize: 8, color: "#334155", letterSpacing: "0.12em", marginBottom: 3 }}>{d.label}</div>
                                  <div style={{ fontSize: 12, color: d.color }}>{d.value}</div>
                                </div>
                              ))}
                            </div>
                            {t.notes && (
                              <div style={{ marginTop: 8, padding: "7px 10px", background: "#0a1628", borderRadius: 4, borderLeft: "2px solid #f59e0b" }}>
                                <div style={{ fontSize: 8, color: "#334155", letterSpacing: "0.12em", marginBottom: 3 }}>NOTES</div>
                                <div style={{ fontSize: 11, color: "#fde68a", lineHeight: 1.5 }}>{t.notes}</div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  });
                })()}
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}

function TopFindings({ entry, a, ai }) {
  const [findings, setFindings] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!entry) return;
    generate();
  }, [entry.date]);

  const generate = async () => {
    setLoading(true);
    setFindings("");
    setError("");
    const trades = entry.parsedTrades || [];
    const tNTF = (t) => t.pnl - (t.commission||0);
    const winners = trades.filter(t => tNTF(t) > 0);
    const losers = trades.filter(t => tNTF(t) < 0);
    const avgWin = winners.length ? (winners.reduce((s,t) => s+tNTF(t), 0)/winners.length).toFixed(2) : 0;
    const avgLoss = losers.length ? Math.abs(losers.reduce((s,t) => s+tNTF(t), 0)/losers.length).toFixed(2) : 0;
    const biggestWin = winners.length ? Math.max(...winners.map(t=>tNTF(t))).toFixed(2) : 0;
    const biggestLoss = losers.length ? Math.min(...losers.map(t=>tNTF(t))).toFixed(2) : 0;
    // Use AI-polished rewrites where available, fall back to raw original
    const rw = entry.aiRewrites || {};
    const note = (key) => (rw[key]?.trim() || entry[key] || "none");
    const noteSummary = entry.aiNoteSummary || "";
    const tradeLog = trades.map((t,i) => {
      const netPnlT = (t.pnl - (t.commission||0)).toFixed(2);
      const notesStr = t.notes ? ` [${t.notes}]` : "";
      return `Trade ${i+1}: ${t.symbol} qty:${t.qty} ${t.direction||""} | entry:$${t.buyPrice}@${t.buyTime?.split(" ")[1]||"?"} exit:$${t.sellPrice}@${t.sellTime?.split(" ")[1]||"?"} | hold:${t.duration||"?"} | type:${t.orderType||"MKT"} | gross:$${t.pnl?.toFixed(2)} comm:-$${(t.commission||0).toFixed(2)} net:$${netPnlT}${notesStr}`;
    }).join("\n");
    const sessionBreakdown = a?.bySession ? Object.entries(a.bySession).map(([s,d]) =>
      `${s}: ${d.trades} trades, ${d.wins} wins, $${d.pnl?.toFixed(2)}`
    ).join(" | ") : "none";
    const bySymbol = a?.bySymbol ? Object.entries(a.bySymbol).map(([sym,d]) =>
      `${sym}: ${d.trades} trades, $${d.pnl?.toFixed(2)}, ${((d.wins/d.trades)*100).toFixed(0)}% WR`
    ).join(" | ") : "none";
    const orderTypeBreakdown = a?.byOrderType ? Object.entries(a.byOrderType).filter(([,d])=>d.trades>0).map(([ot,d]) =>
      `${ot}: ${d.trades} trades, ${d.trades ? Math.round(d.wins/d.trades*100) : 0}% WR, gross $${d.pnl.toFixed(2)}`
    ).join(" | ") : "none";
    const totalComm = trades.reduce((s,t) => s+(t.commission||0), 0);
    const commDragPct = Math.abs(parseFloat(entry.pnl||0)) > 0 ? (totalComm/Math.abs(parseFloat(entry.pnl||0))*100).toFixed(1) : "N/A";
    const flaggedNotes = trades.filter(t=>t.notes && t.notes !== "overnight-carry").map(t=>`Trade ${trades.indexOf(t)+1}: ${t.notes}`).join(", ") || "none";
    const carryTrades  = trades.filter(t=>t.notes==="overnight-carry");
    const durationBreakdown = a?.DURATION_BUCKETS ? a.DURATION_BUCKETS.filter(b=>a.byDuration[b.key]?.trades>0).map(b=>{
      const bd = a.byDuration[b.key]; return `${b.key}: ${bd.trades}t ${Math.round(bd.wins/bd.trades*100)}%WR $${bd.pnl.toFixed(0)}`;
    }).join(" | ") : "none";
    // Direction breakdown — key edge signal for TopFindings
    const dirBreakdownTF = a?.byDirection ? ["long","short"].filter(k=>a.byDirection[k].trades>0).map(k => {
      const d = a.byDirection[k];
      const wr = d.trades ? Math.round(d.wins/d.trades*100) : 0;
      const commEst = trades.filter(t2=>(t2.direction||"long")===k).reduce((s,t2)=>s+(t2.commission||0),0);
      const net = (d.pnl - commEst).toFixed(2);
      const pf = d.grossLoss > 0 ? (d.grossWin/d.grossLoss).toFixed(2) : d.grossWin > 0 ? "inf" : "0";
      return `${k.toUpperCase()}: ${d.trades}t ${wr}%WR gross $${d.pnl.toFixed(2)} net ~$${net} PF ${pf}`;
    }).join(" | ") : "none";

    const prompt = `You are an expert futures trading analyst. Analyze this trader's full day of data and surface exactly 3 findings — the most insightful, non-obvious patterns that will directly improve their profitability. Each finding must be grounded in trade data or the trader's own written words. No fluff, no generic advice.

DATE: ${entry.date} (${new Date(entry.date+"T12:00:00").toLocaleDateString("en-US",{weekday:"long"})})
INSTRUMENTS: ${(entry.instruments?.length ? entry.instruments : [entry.instrument||"?"]).join(", ")}
GRADE: ${entry.grade||"?"} | BIAS: ${entry.bias||"?"} | MOOD: ${(entry.moods||[entry.mood]).filter(Boolean).join(", ")||"?"}
MISTAKES TAGGED (light reference only — use these to corroborate patterns in the written notes, not as the primary focus): ${(entry.sessionMistakes||[]).filter(m=>m!=="No Mistakes — Executed the Plan ✓").join(", ")||"none"}
GROSS P&L: $${entry.pnl||0} | NET P&L: $${(parseFloat(entry.pnl||0)-parseFloat(entry.commissions||0)).toFixed(2)} | FEES: $${entry.commissions||0} (${commDragPct}% of gross)
TOTAL TRADES: ${trades.length} | WINNERS: ${winners.length} | LOSERS: ${losers.length}${carryTrades.length > 0 ? ` | CARRY-FORWARD: ${carryTrades.length}` : ""}
WIN RATE: ${trades.length ? ((winners.length/trades.length)*100).toFixed(1) : 0}%
AVG WIN: $${avgWin} | AVG LOSS: $${avgLoss} | BIGGEST WIN: $${biggestWin} | BIGGEST LOSS: $${biggestLoss}
PROFIT FACTOR: ${fmtPF(a?.profitFactor)} | EXPECTANCY: ${a?.expectancy?.toFixed(2)||"N/A"}R
BY DIRECTION: ${dirBreakdownTF}
BY SESSION (exit time): ${sessionBreakdown}
BY SYMBOL: ${bySymbol}
BY ORDER TYPE: ${orderTypeBreakdown}
BY DURATION BUCKET: ${durationBreakdown}
COMMISSIONS: total $${totalComm.toFixed(2)} (${commDragPct}% of gross P&L)
BROKER FLAGS: ${flaggedNotes}
GRADE: ${entry.grade||"none"} | EXECUTION SCORE: ${entry.executionScore != null ? entry.executionScore+"/10" : "not logged"} | DECISION SCORE: ${entry.decisionScore != null ? entry.decisionScore+"/10" : "not logged"}
MOOD: ${(entry.moods?.length ? entry.moods : entry.mood ? [entry.mood] : []).join(", ")||"not logged"}
BIAS: ${entry.bias||"none"}
${noteSummary ? `JOURNAL NARRATIVE (AI-consolidated from all notes):\n${noteSummary}\n` : ""}MARKET NOTES: ${note("marketNotes")}
RULES FOLLOWED/BROKEN: ${note("rules")}
LESSONS LEARNED: ${note("lessonsLearned")}
MISTAKES (freeform, own words): ${note("mistakes")}
AREAS FOR IMPROVEMENT: ${note("improvements")}
BEST TRADE (own words): ${note("bestTrade")}
WORST TRADE (own words): ${note("worstTrade")}
RULE TO REINFORCE: ${note("reinforceRule")}
PLAN FOR TOMORROW: ${note("tomorrow")}
${(() => { const costs = entry.mistakeCosts||{}; const rows = Object.entries(costs).filter(([,v])=>v!=null&&Number(v)>0); if(!rows.length) return ""; const total = rows.reduce((s,[,v])=>s+Number(v),0); return `MISTAKE COST BREAKDOWN: ${rows.map(([tag,v])=>`${tag}: $${Number(v).toFixed(0)}`).join(", ")} — total $${total.toFixed(0)}${entry.mistakeCostNotes?" · Notes: "+entry.mistakeCostNotes:""}`;})()}

FULL TRADE LOG:
${tradeLog||"No trade data imported"}

PRIORITY ORDER for finding sources:
1. TRADE DATA — sequence patterns, direction/session edge, commission drag, hold-time, sizing
2. WRITTEN NOTES — what the trader actually said in market notes, lessons, rules, best/worst trade descriptions
3. TAGGED MISTAKES — only use as light corroboration of something already visible in #1 or #2, not as the primary finding

Before writing: (1) Is one direction net-negative while the other is profitable? (2) Is commission drag above 25%? (3) Which session had the worst net P&L? (4) What pattern in the WRITTEN NOTES contradicts or confirms what the trade log shows?

Return EXACTLY this format, nothing else:

**FINDING 1: [short punchy title]**
[2-3 sentences. Specific data point or own written words → what it means → one concrete action to fix or reinforce it]

**FINDING 2: [short punchy title]**
[2-3 sentences. Specific data point or own written words → what it means → one concrete action to fix or reinforce it]

**FINDING 3: [short punchy title]**
[2-3 sentences. Specific data point or own written words → what it means → one concrete action to fix or reinforce it]

**FINDING 3: [short punchy title]**
[2-3 sentences. Specific data point → what it means → one concrete action to fix or reinforce it]`;

    try {
      const txt = await aiRequestText(ai, {
        max_tokens: 600,
        timeoutMs: 120000,
        messages: [{ role: 'user', content: prompt }],
      });
      setFindings(txt);
    } catch (err) {
      setError(err.message || "Failed to generate findings. Please try again.");
    }
    setLoading(false);
  };

  const parsed = findings ? findings.split(/\*\*FINDING \d+:/).filter(Boolean).map(s => {
    const titleEnd = s.indexOf("**");
    const title = titleEnd > -1 ? s.slice(0, titleEnd).trim() : s.split("\n")[0].trim();
    const body = titleEnd > -1 ? s.slice(titleEnd+2).trim() : s.slice(s.indexOf("\n")).trim();
    return { title, body };
  }) : [];

  const colors = [
    { border: "#1e3a5f", title: "#93c5fd", num: "#1d4ed8", bg: "#0a1628" },
    { border: "#166534", title: "#4ade80", num: "#16a34a", bg: "#061f0f" },
    { border: "#7c3aed44", title: "#a78bfa", num: "#7c3aed", bg: "#0d0a1f" },
  ];

  return (
    <div style={{ marginBottom: 24 }}>
      <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 14, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span>🔍 TOP 3 FINDINGS</span>
        {!loading && findings && (
          <button onClick={generate} style={{ background: "transparent", border: "1px solid #1e293b", color: "#3b82f6", padding: "2px 8px", borderRadius: 3, fontSize: 9, cursor: "pointer", fontFamily: "inherit", letterSpacing: "0.06em" }}>↺ regenerate</button>
        )}
      </div>

      {loading && (
        <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 6, padding: "20px 16px", display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#3b82f6", animation: "pulse 1s infinite" }} />
          <div style={{ fontSize: 12, color: "#64748b", letterSpacing: "0.06em" }}>Analyzing trade data…</div>
        </div>
      )}

      {!loading && parsed.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {parsed.map((f, i) => (
            <div key={i} style={{ background: colors[i]?.bg || "#0f1729", border: `1px solid ${colors[i]?.border || "#1e293b"}`, borderRadius: 6, padding: "14px 16px", display: "flex", gap: 14 }}>
              <div style={{ flexShrink: 0, width: 24, height: 24, borderRadius: "50%", background: colors[i]?.num || "#1d4ed8", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, color: "white", fontWeight: 700, marginTop: 1 }}>{i+1}</div>
              <div>
                <div style={{ fontSize: 11, color: colors[i]?.title || "#93c5fd", letterSpacing: "0.08em", marginBottom: 6, fontWeight: 600 }}>{f.title}</div>
                <RenderAI text={f.body} />
              </div>
            </div>
          ))}
        </div>
      )}

      {!loading && !findings && (
        <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 6, padding: "14px 16px", fontSize: 12, color: "#64748b" }}>No data available to analyze.</div>
      )}

      {!loading && error && (
        <div style={{ background: "rgba(63,16,16,0.5)", border: "1px solid #7f1d1d", borderRadius: 6, padding: "12px 16px", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
          <span style={{ fontSize: 12, color: "#f87171" }}>⚠ {error}</span>
          <button onClick={generate} style={{ background: "transparent", border: "1px solid #7f1d1d", color: "#f87171", padding: "3px 10px", borderRadius: 3, fontSize: 10, cursor: "pointer", fontFamily: "inherit", letterSpacing: "0.06em", whiteSpace: "nowrap" }}>↺ retry</button>
        </div>
      )}
    </div>
  );
}

function DailyAIAnalysis({ entry, a, ai, priorPlan = null }) {
  const [analysis, setAnalysis] = useState("");
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);

  const buildPrompt = () => {
    const trades = entry.parsedTrades || [];
    const tNDA = (t) => t.pnl - (t.commission||0);
    const winners = trades.filter(t => tNDA(t) > 0);
    const losers = trades.filter(t => tNDA(t) < 0);
    const avgWin = winners.length ? winners.reduce((s, t) => s + tNDA(t), 0) / winners.length : 0;
    const avgLoss = losers.length ? Math.abs(losers.reduce((s, t) => s + tNDA(t), 0) / losers.length) : 0;
    const rrRatio = avgLoss > 0 ? (avgWin / avgLoss).toFixed(2) : "N/A";
    const winRate = trades.length ? ((winners.length / trades.length) * 100).toFixed(1) : 0;
    const profitFactor = fmtPF(a?.profitFactor);
    const expectancy = a?.expectancy !== null && a?.expectancy !== undefined ? a.expectancy.toFixed(2) + "R" : "N/A";
    // Use AI-polished rewrites where available, fall back to raw original
    const rw = entry.aiRewrites || {};
    const note = (key) => (rw[key]?.trim() || entry[key] || "None");
    const noteSummary = entry.aiNoteSummary || "";

    // Session breakdown — already net-based from calcAnalytics fix
    const sessionBreakdown = a?.bySession ? Object.entries(a.bySession).map(([s, d]) => {
      const wr = d.trades ? Math.round(d.wins/d.trades*100) : 0;
      return `  ${s}: ${d.trades} trades, ${wr}% WR, net $${d.pnl.toFixed(2)}`;
    }).join("\n") : "No session data";

    // Direction breakdown — already net-based from calcAnalytics fix
    const dirBreakdown = a?.byDirection ? ["long","short"].filter(k=>a.byDirection[k].trades>0).map(k => {
      const d = a.byDirection[k];
      const wr = d.trades ? Math.round(d.wins/d.trades*100) : 0;
      const pf = d.grossLoss > 0 ? (d.grossWin/d.grossLoss).toFixed(2) : d.grossWin > 0 ? "∞" : "0.00";
      return `  ${k.toUpperCase()}: ${d.trades} trades | ${wr}% WR | net $${d.pnl.toFixed(2)} | PF ${pf}`;
    }).join("\n") : "No direction data";

    const tradeLog = trades.map((t, i) => {
      const netT = (t.pnl - (t.commission||0)).toFixed(2);
      const notesStr = t.notes ? ` [broker flag: ${t.notes}]` : "";
      return `  Trade ${i+1}: ${t.symbol} | qty ${t.qty} | ${t.direction||"long"} | entry $${t.buyPrice} @ ${t.buyTime?.split(" ")[1]||"?"} exit $${t.sellPrice} @ ${t.sellTime?.split(" ")[1]||"?"} | hold ${t.duration||"?"} | ${t.orderType||"MKT"} order | gross $${t.pnl.toFixed(2)} comm -$${(t.commission||0).toFixed(2)} net $${netT}${notesStr}`;
    }).join("\n");
    const totalComm = trades.reduce((s,t) => s+(t.commission||0), 0);
    const commDragPct = Math.abs(parseFloat(entry.pnl||0)) > 0 ? (totalComm/Math.abs(parseFloat(entry.pnl||0))*100).toFixed(1) : "N/A";
    const orderTypeStats = a?.byOrderType ? Object.entries(a.byOrderType).filter(([,d])=>d.trades>0).map(([ot,d]) =>
      `    ${ot}: ${d.trades} trades | ${d.trades ? Math.round(d.wins/d.trades*100) : 0}% WR | gross $${d.pnl.toFixed(2)} | avg $${(d.pnl/d.trades).toFixed(2)}/trade`
    ).join("\n") : "  none";
    const durationStats = a?.DURATION_BUCKETS ? a.DURATION_BUCKETS.filter(b=>a.byDuration[b.key]?.trades>0).map(b=>{
      const bd = a.byDuration[b.key];
      return `    ${b.key}: ${bd.trades} trades | ${Math.round(bd.wins/bd.trades*100)}% WR | $${bd.pnl.toFixed(2)} total | avg hold ${bd.avgSecs<60?Math.round(bd.avgSecs)+"s":Math.floor(bd.avgSecs/60)+"m"}`;
    }).join("\n") : "  none";
    const flaggedTrades = trades.filter(t=>t.notes).map((t,i)=>`    Trade ${trades.indexOf(t)+1} (${t.symbol}): ${t.notes}`).join("\n") || "  none";

    return `You are an expert futures trading coach providing a deep data-driven daily performance analysis. Be direct, specific, and genuinely useful — not generic. Speak like a seasoned trading mentor.

DATE: ${entry.date}
INSTRUMENT: ${(entry.instruments?.length ? entry.instruments : entry.instrument ? [entry.instrument] : ["Unknown"]).join(", ")}
GRADE: ${entry.grade || "Not logged"}${entry.executionScore != null || entry.decisionScore != null ? ` (Execution: ${entry.executionScore != null ? entry.executionScore + "/10" : "—"}, Decision: ${entry.decisionScore != null ? entry.decisionScore + "/10" : "—"})` : ""}
BIAS: ${entry.bias || "Not logged"} | MOOD: ${(entry.moods?.length ? entry.moods : entry.mood ? [entry.mood] : ["Not logged"]).join(", ")}
MISTAKES TAGGED (light context — corroborate written notes, not primary source): ${(entry.sessionMistakes||[]).filter(m=>m!=="No Mistakes — Executed the Plan ✓").join(", ")||"none"}
${(() => {
  const costs = entry.mistakeCosts || {};
  const rows = Object.entries(costs).filter(([, v]) => v != null && v > 0);
  if (rows.length === 0) return "";
  const total = rows.reduce((s, [, v]) => s + parseFloat(v), 0);
  const netPnlVal = parseFloat(entry.pnl || 0) - parseFloat(entry.commissions || 0);
  return `MISTAKE COST ATTRIBUTION: ${rows.map(([tag, v]) => `${tag}: $${parseFloat(v).toFixed(0)}`).join(", ")} — Total attributed: $${total.toFixed(0)} (vs net P&L of ${netPnlVal >= 0 ? "+" : "-"}$${netPnlVal >= 0 ? netPnlVal.toFixed(0) : (-netPnlVal).toFixed(0)})${entry.mistakeCostNotes ? ` · Notes: ${entry.mistakeCostNotes}` : ""}\n`;
})()}
TRADE STATISTICS:
  Total trades: ${trades.length}
  Winners: ${winners.length} | Losers: ${losers.length} | Breakeven: ${trades.filter(t => t.pnl === 0).length}
  Win rate: ${winRate}%
  Avg winner: $${avgWin.toFixed(2)} | Avg loser: -$${avgLoss.toFixed(2)}
  Reward:Risk ratio: ${rrRatio}x
  R-multiple avg win: ${a?.avgWinR !== null && a?.avgWinR !== undefined ? a.avgWinR.toFixed(2) + "R" : "N/A"}
  Profit factor: ${profitFactor}
  Expectancy: ${expectancy}
  Gross P&L: $${parseFloat(entry.pnl || 0).toFixed(2)}
  Net P&L (after fees): $${(parseFloat(entry.pnl || 0) - parseFloat(entry.commissions || 0)).toFixed(2)}
  Commissions/Fees: $${parseFloat(entry.commissions || 0).toFixed(2)} (${commDragPct}% commission drag on gross)
  Intraday max drawdown: $${a?.maxDD?.toFixed(2) || "N/A"}
  Max consec. wins: ${a?.maxConsecWins || 0}
  Max consec. losses: ${a?.maxConsecLoss || 0}
  Avg contract size: ${a?.avgQty?.toFixed(1) || "N/A"}
  Largest single win: $${a?.largestWin?.toFixed(2) || "N/A"}
  Largest single loss: $${a?.largestLoss?.toFixed(2) || "N/A"}

SESSION BREAKDOWN (P&L by time window — uses exit time):
${sessionBreakdown}

DIRECTION BREAKDOWN (long vs short edge):
${dirBreakdown}

ORDER TYPE BREAKDOWN (LMT/STP/MKT performance):
${orderTypeStats}

HOLD TIME BREAKDOWN (duration buckets):
${durationStats}

BROKER FLAGS (partial fills, manual orders — trades to scrutinize):
${flaggedTrades}

FULL TRADE LOG (chronological — use this to detect sequencing, sizing, and revenge patterns):
${tradeLog}

TRADER'S WRITTEN NOTES (primary source — mine these for patterns, themes, self-assessments):
${noteSummary ? `  JOURNAL NARRATIVE (AI-consolidated):\n  ${noteSummary}\n` : ""}  Market notes: ${note("marketNotes")}
  Rules followed/broken: ${note("rules")}
  Lessons learned: ${note("lessonsLearned")}
  Mistakes (own words): ${note("mistakes")}
  Improvements: ${note("improvements")}
  Best trade: ${note("bestTrade")}
  Worst trade: ${note("worstTrade")}
  Rule to reinforce: ${note("reinforceRule")}
  Plan for tomorrow: ${note("tomorrow")}
  Mistake cost notes: ${entry.mistakeCostNotes || "None"}
${priorPlan && (priorPlan.plan || priorPlan.reinforceRule) ? `
PREVIOUS SESSION'S PLAN (written ${priorPlan.date}, for today):
  Plan for today: ${priorPlan.plan || "None written"}
  Rule to reinforce: ${priorPlan.reinforceRule || "None written"}

CRITICAL CROSS-REFERENCE REQUIRED: Compare the previous plan above against today's actual trades and notes. Were the stated intentions followed? Look for direct contradictions between what was planned and what was executed. If the plan said "no trades after 11am" and trades occurred at 1pm, that is a violation. If the plan said "size down" and contract sizes increased, name it.` : ""}

Provide a deep, data-driven daily analysis with exactly these sections. Use specific numbers from the data. Be a balanced coaching mentor — honest about weaknesses, genuine about strengths. Primary sources: trade data and written notes. Mistake tags are light context only.

**🔬 DATA SNAPSHOT**
3-4 sentences on what the raw numbers reveal. Mandatory checks: (1) Is one direction (long/short) net-negative while the other is profitable? Name it with exact figures. (2) Which session window drove the most P&L — and which drained it? (3) Commission drag: if fees exceed 20% of gross, name the exact % and dollar impact. (4) Lead with the single most important stat (profit factor, expectancy, or a direction/session edge). Be specific — cite numbers, not generalities.

**✅ WHAT WORKED TODAY**
2-3 specific things backed by the data. Reference actual trades by number, time window, or price. If a session window dominated, name it. If the trader wrote something in their notes that aligns with what the data shows — reinforce it explicitly and genuinely. A plan followed deserves recognition as much as a mistake deserves correction. No generic praise, only concrete specifics.

**⚠️ WHAT NEEDS WORK**
2-3 specific weaknesses sourced primarily from TRADE DATA and the trader's OWN WRITTEN NOTES — not the mistake dropdown tags. Look at: trade sequence patterns (sizing changes, re-entry timing after losses, session timing drift), what the trader wrote in lessons/market notes vs. what the trades actually show. Tagged mistakes can corroborate but should not be the headline. Quantify the gap where possible (e.g. avg loss after 3pm = 2× avg loss before 3pm).

**🧠 BEHAVIORAL INSIGHTS**
Two-part analysis: (A) TRADE SEQUENCE — scan the log for patterns in sequence and timing: losses followed by faster re-entry, trades well outside normal windows, escalating size after wins or losses, size inconsistency. Name each pattern with specific trade numbers as evidence. (B) NOTES VS. REALITY — read every word the trader wrote today (market notes, lessons, rules, best/worst trade). Where their written words contradict what the trades show, name it precisely. Where their written words accurately capture what the data confirms, reinforce it — self-awareness that matches reality is a real edge.
${priorPlan && (priorPlan.plan || priorPlan.reinforceRule) ? `
**🚩 RED FLAGS — PLAN VS. EXECUTION**
Cross-reference today's trades directly against the previous session's stated plan. For each contradiction: (1) quote what was planned, (2) describe what actually happened with specific trade evidence, (3) label the violation type. If no violations were found, state that explicitly and note which parts of the plan were honored — a clean execution day deserves acknowledgment.
` : ""}
**📈 PATTERN WATCH**
Look at the trade log and written notes together. Identify 1-2 patterns the trader may not have fully named — something visible in the sequence or timing of trades that their notes hint at but don't explicitly state. If a positive pattern is present (e.g. morning trades consistently profitable, limit orders outperform market orders), name it with data. Use specific trade numbers, times, and P&L figures.

**🎯 TOMORROW'S EDGE**
3 concrete, measurable action points rooted in what the TRADE DATA and WRITTEN NOTES revealed today. At least one should reinforce something the trader is already doing well. Make them specific and actionable: "Max 2 contracts until first profitable trade confirmed" not "size down."

Keep each section tight. No filler. Maximum value per word. Total response should be 400-600 words.`;
  };

  const generate = async () => {
    setLoading(true);
    setAnalysis("");
    setDone(false);
    try {
      const txt = await aiRequestText(ai, {
        max_tokens: 1200,
        timeoutMs: 120000,
        messages: [{ role: 'user', content: prompt }],
      });
      setAnalysis(txt || 'ERROR');
      setDone(true);
    } catch (err) {
      const f = friendlyAiError(err);
      setAnalysis(`ERROR:${f.message}`);
      setDone(true);
    }
    setLoading(false);
  };

  if (!(entry.parsedTrades?.length > 0)) return null;

  return (
    <div style={{ marginTop: 20, background: "#0a0e1a", border: "1px solid #1e3a5f", borderRadius: 6, overflow: "hidden" }}>
      {/* Header */}
      <div style={{ padding: "14px 20px", background: "#0a1628", borderBottom: "1px solid #1e293b", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ fontSize: 12, color: "#93c5fd", letterSpacing: "0.1em", fontWeight: 600 }}>✦ AI DAILY TRADE ANALYSIS</div>
          <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>{getTaglineText(loadTagline())} · Based on your full trade data & notes</div>
        </div>
        <button onClick={generate} disabled={loading}
          style={{ background: loading ? "transparent" : "#1d4ed8", color: loading ? "#475569" : "white", border: loading ? "1px solid #1e293b" : "none", padding: "8px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: loading ? "not-allowed" : "pointer", letterSpacing: "0.06em", transition: "all .15s" }}>
          {loading ? "ANALYSING..." : done ? "↺ REGENERATE" : "ANALYSE DAY →"}
        </button>
      </div>

      {/* Loading state */}
      {loading && (
        <div style={{ padding: "40px 20px", display: "flex", flexDirection: "column", alignItems: "center", gap: 12 }}>
          <style>{`@keyframes aiPulse { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
          <div style={{ fontSize: 11, color: "#3b82f6", letterSpacing: "0.15em", animation: "aiPulse 1.8s infinite" }}>✦ ANALYSING YOUR TRADES...</div>
          <div style={{ fontSize: 10, color: "#1e3a5f" }}>Reviewing trade log, session data, and behavioral patterns</div>
        </div>
      )}

      {/* Analysis content */}
      {!loading && analysis && !analysis.startsWith("ERROR") && (
        <div style={{ padding: "20px 24px" }}>
          <RenderAI text={analysis} />
        </div>
      )}

      {/* Error */}
      {!loading && analysis?.startsWith("ERROR") && (
        <div style={{ padding: "20px 24px" }}>
          <div style={{ color: "#f87171", fontSize: 12, fontWeight: 600, marginBottom: 6 }}>⚠ Analysis failed</div>
          <div style={{ color: "#94a3b8", fontSize: 11, lineHeight: 1.7 }}>
            {analysis.slice(6) || "Unknown error — check your API key in Settings (⚙) and try again."}
          </div>
        </div>
      )}

      {/* Empty prompt */}
      {!loading && !analysis && (
        <div style={{ padding: "28px 24px", display: "flex", gap: 20, alignItems: "flex-start" }}>
          {[
            { icon: "🔬", label: "Data Snapshot", desc: "Key stats distilled from your trade log" },
            { icon: "✅", label: "What Worked", desc: "Specific wins backed by numbers" },
            { icon: "⚠️", label: "What Needs Work", desc: "Honest assessment of weak spots" },
            { icon: "🧠", label: "Behavioral Insights", desc: "Psychology patterns in your trades" },
            { icon: "🎯", label: "Tomorrow's Edge", desc: "3 actionable focus points" },
          ].map(s => (
            <div key={s.label} style={{ flex: 1, textAlign: "center" }}>
              <div style={{ fontSize: 18, marginBottom: 6 }}>{s.icon}</div>
              <div style={{ fontSize: 10, color: "#3b82f6", fontWeight: 600, marginBottom: 3, letterSpacing: "0.05em" }}>{s.label}</div>
              <div style={{ fontSize: 9, color: "#1e3a5f", lineHeight: 1.5 }}>{s.desc}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ChartScreenshotDetailView({ screenshots }) {
  const [lightbox, setLightbox] = useState(null);
  return (
    <div style={{ marginTop: 20, background: "#0f1729", border: "1px solid #1e293b", borderRadius: 6, padding: "14px 16px" }}>
      <div style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 12 }}>CHART SCREENSHOTS <span style={{ color: "#64748b", fontWeight: 400 }}>· {screenshots.length}</span></div>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {screenshots.map((s, i) => (
          <div key={s.id || i} onClick={() => setLightbox(s.src)}
            style={{ position: "relative", borderRadius: 5, overflow: "hidden", border: "1px solid #1e3a5f", cursor: "zoom-in", transition: "border-color .15s" }}
            onMouseEnter={e => e.currentTarget.style.borderColor = "#3b82f6"}
            onMouseLeave={e => e.currentTarget.style.borderColor = "#1e3a5f"}>
            <img src={s.src} alt={`Chart ${i + 1}`} style={{ display: "block", width: "100%", height: "auto", maxHeight: 420, objectFit: "contain", background: "#060810" }} />
            <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, padding: "6px 10px", background: "linear-gradient(transparent, rgba(0,0,0,0.5))", fontSize: 9, color: "rgba(255,255,255,0.35)", letterSpacing: "0.06em" }}>CHART #{i + 1}</div>
          </div>
        ))}
      </div>
      {lightbox && (
        <div onClick={() => setLightbox(null)}
          style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.93)", zIndex: 9999, display: "flex", alignItems: "center", justifyContent: "center", cursor: "zoom-out" }}>
          <img src={lightbox} alt="Chart" style={{ maxWidth: "93vw", maxHeight: "91vh", borderRadius: 6, boxShadow: "0 0 80px rgba(0,0,0,0.9)" }} />
          <div style={{ position: "absolute", top: 20, right: 24, fontSize: 12, color: "#64748b" }}>click anywhere to close</div>
        </div>
      )}
    </div>
  );
}

function DetailView({ entry, a, pnlColor, fmtPnl, gradeColor, onRecap, ai, onUpdateEntry, dangerLine = null, priorPlan = null }) {
  const [atab, setAtab] = useState("overview");
  const [rewrites, setRewrites] = useState(() => entry.aiRewrites || {});
  const [rewriting, setRewriting] = useState({});
  const [showOriginal, setShowOriginal] = useState({});
  const [staleFields, setStaleFields] = useState({}); // Feature 1: { fieldKey: true } if stale
  const [noteSummary, setNoteSummary] = useState(() => entry.aiNoteSummary || "");
  const [noteSummaryLoading, setNoteSummaryLoading] = useState(false);
  const [showOriginalNotes, setShowOriginalNotes] = useState(false);
  const hasNotes = entry.lessonsLearned || entry.mistakes || entry.improvements || entry.marketNotes || entry.bestTrade || entry.worstTrade || entry.tomorrow || entry.rules || entry.reinforceRule || entry.sessionMistakes?.length;

  // Keep local UI state in sync when switching entries
  useEffect(() => {
    setRewrites(entry.aiRewrites || {});
    setNoteSummary(entry.aiNoteSummary || "");
    setShowOriginal({});
    setShowOriginalNotes(false);
    setStaleFields({});
  }, [entry.id]);

  // Feature 1: Compute staleness for all rewritten fields on load
  useEffect(() => {
    (async () => {
      const fields = {
        marketNotes: entry.marketNotes, rules: entry.rules, bestTrade: entry.bestTrade,
        worstTrade: entry.worstTrade, lessonsLearned: entry.lessonsLearned, mistakes: entry.mistakes,
        improvements: entry.improvements, reinforceRule: entry.reinforceRule, tomorrow: entry.tomorrow,
      };
      const sf = {};
      for (const [key, txt] of Object.entries(fields)) {
        const meta = entry.aiRewritesMeta?.[key];
        if (!meta?.srcHash || !entry.aiRewrites?.[key]) continue;
        if (!txt?.trim()) continue;
        const currentHash = await sha256(`v1|${key}|${txt}`);
        sf[key] = currentHash !== meta.srcHash;
      }
      setStaleFields(sf);
    })();
  }, [entry.id]);

  const rewrite = async (key, text) => {
    if (!text?.trim()) return;

    try {
      const srcHash = await sha256(`v1|${key}|${text}`);
      const existing = entry?.aiRewrites?.[key];
      const existingMeta = entry?.aiRewritesMeta?.[key];
      if (existing && existingMeta?.srcHash === srcHash) {
        setRewrites(p => ({ ...p, [key]: existing }));
        return;
      }

      const cacheKey = await sha256(`${ai?.provider || "anthropic"}|${ai?.model || ''}|rewrite|${key}|${text}`);
      const cached = getCachedAiText(cacheKey);
      if (cached) {
        setRewrites(p => ({ ...p, [key]: cached }));
        if (onUpdateEntry) {
          onUpdateEntry(entry.id, {
            aiRewrites: { ...(entry.aiRewrites || {}), [key]: cached },
            aiRewritesMeta: { ...(entry.aiRewritesMeta || {}), [key]: { srcHash, ts: Date.now() } },
          });
        }
        return;
      }

      setRewriting(p => ({ ...p, [key]: true }));
      const result = await aiRequestText(ai, {
        max_tokens: 400,
        timeoutMs: 22000,
        messages: [{ role: 'user', content: `You are helping a futures trader rewrite their quick journal notes into clearer, more reflective writing. Keep the exact same meaning, sentiment, and specifics — just make it more articulate and thoughtful. Write in first person. No bullet points. No added fluff or analysis. Just rewrite what they wrote, cleaner.

Original note:
"${text}"

Rewritten version (same meaning, cleaner and more reflective):` }],
      });

      if (result) {
        setCachedAiText(cacheKey, result);
        setRewrites(p => ({ ...p, [key]: result }));
        if (onUpdateEntry) {
          onUpdateEntry(entry.id, {
            aiRewrites: { ...(entry.aiRewrites || {}), [key]: result },
            aiRewritesMeta: { ...(entry.aiRewritesMeta || {}), [key]: { srcHash, ts: Date.now() } },
          });
        }
      }
    } catch (err) {
      const f = friendlyAiError(err);
      console.warn('Rewrite failed for', key, f.code, f.message);
    } finally {
      setRewriting(p => ({ ...p, [key]: false }));
    }
  };

  // Auto-rewrite all fields on load (and persist)
  useEffect(() => {
    let cancelled = false;
    if (!ai?.enabled || !ai?.apiKey) { return; }

    const fields = {
      marketNotes: entry.marketNotes,
      rules: entry.rules,
      bestTrade: entry.bestTrade,
      worstTrade: entry.worstTrade,
      lessonsLearned: entry.lessonsLearned,
      mistakes: entry.mistakes,
      improvements: entry.improvements,
      reinforceRule: entry.reinforceRule,
      tomorrow: entry.tomorrow,
    };

    (async () => {
      for (const [key, txt] of Object.entries(fields)) {
        if (!txt?.trim()) continue;
        if (cancelled) return;
        await rewrite(key, txt);
      }

      const notesParts = [
        entry.marketNotes && `MARKET / TAPE NOTES:
${entry.marketNotes}`,
        entry.rules && `RULES FOLLOWED / BROKEN:
${entry.rules}`,
        entry.lessonsLearned && `LESSONS LEARNED:
${entry.lessonsLearned}`,
        entry.mistakes && `MISTAKES TO AVOID:
${entry.mistakes}`,
        entry.improvements && `AREAS FOR IMPROVEMENT:
${entry.improvements}`,
        entry.bestTrade && `BEST TRADE:
${entry.bestTrade}`,
        entry.worstTrade && `WORST TRADE / MISTAKE:
${entry.worstTrade}`,
        entry.reinforceRule && `RULE TO REINFORCE:
${entry.reinforceRule}`,
        entry.tomorrow && `PLAN FOR TOMORROW:
${entry.tomorrow}`,
      ].filter(Boolean);

      if (notesParts.length === 0) return;

      const src = notesParts.join('\n\n');
      const srcHash = await sha256(`v1|notesSummary|${src}`);

      if (entry.aiNoteSummary && entry.aiNoteSummaryHash === srcHash) {
        if (!cancelled) setNoteSummary(entry.aiNoteSummary);
        return;
      }

      const cacheKey = await sha256(`${ai?.provider || "anthropic"}|${ai?.model || ''}|dailyRecap|${src}`);
      const cached = getCachedAiText(cacheKey);
      if (cached) {
        if (!cancelled) setNoteSummary(cached);
        if (onUpdateEntry) onUpdateEntry(entry.id, { aiNoteSummary: cached, aiNoteSummaryHash: srcHash });
        return;
      }

      setNoteSummaryLoading(true);
      try {
        const result = await aiRequestText(ai, {
          max_tokens: 800,
          timeoutMs: 120000,
          messages: [{ role: 'user', content: `You are helping a futures trader consolidate their daily journal notes into one clear, well-written summary. Rewrite all of the notes below into a single cohesive narrative in first person. Preserve every specific detail, insight, and intention — just make it flow naturally as one unified journal entry. Organize it logically: market context first, then trade execution, then lessons and tomorrow's plan. No bullet points. No headers. No added analysis or fluff — only what the trader actually wrote, rewritten clearly.

Notes:
${notesParts.join('\n\n')}

Rewritten summary:` }],
        });
        if (!cancelled && result) {
          setCachedAiText(cacheKey, result);
          setNoteSummary(result);
          if (onUpdateEntry) onUpdateEntry(entry.id, { aiNoteSummary: result, aiNoteSummaryHash: srcHash });
        }
      } catch (err) {
        const f = friendlyAiError(err);
        console.warn('Notes summary failed:', f.code, f.message);
      } finally {
        if (!cancelled) setNoteSummaryLoading(false);
      }
    })();

    return () => { cancelled = true; };
  }, [entry.id]);

  // Feature 1: Compute staleness — rewrite is stale if the stored srcHash no longer matches current text
  const isRewriteStale = (fieldKey, text) => {
    const meta = entry.aiRewritesMeta?.[fieldKey];
    if (!meta?.srcHash || !rewrites[fieldKey]) return false;
    // We can't do async sha256 inline, so we track staleFields in state (populated on load)
    return staleFields[fieldKey] === true;
  };

  const resetRewrite = (fieldKey) => {
    setRewrites(p => { const n = { ...p }; delete n[fieldKey]; return n; });
    setShowOriginal(p => { const n = { ...p }; delete n[fieldKey]; return n; });
    if (onUpdateEntry) {
      const newRewrites  = { ...(entry.aiRewrites  || {}) }; delete newRewrites[fieldKey];
      const newMeta      = { ...(entry.aiRewritesMeta || {}) }; delete newMeta[fieldKey];
      onUpdateEntry(entry.id, { aiRewrites: newRewrites, aiRewritesMeta: newMeta });
    }
  };

  const RewriteBtn = ({ fieldKey, text }) => {
    const isRewriting = rewriting[fieldKey];
    const hasRewrite  = !!rewrites[fieldKey];
    const isStale     = isRewriteStale(fieldKey, text);
    const isOrig      = showOriginal[fieldKey];
    return (
      <div style={{ display: "flex", gap: 5, alignItems: "center" }}>
        {hasRewrite && (
          <button
            onClick={() => { if (!isRewriting) rewrite(fieldKey, text); }}
            title="Re-run AI rewrite"
            style={{ background: "transparent", border: `1px solid ${isStale ? "#92400e" : "#1e293b"}`, color: isStale ? "#f59e0b" : "#334155", padding: "2px 7px", borderRadius: 3, fontSize: 9, cursor: isRewriting ? "wait" : "pointer", fontFamily: "inherit", letterSpacing: "0.06em", whiteSpace: "nowrap", transition: "all .15s" }}>
            {isStale ? "⚠ RE-RUN" : "↺ re-run"}
          </button>
        )}
        <button
          onClick={() => { if (!isRewriting && hasRewrite) setShowOriginal(p => ({ ...p, [fieldKey]: !p[fieldKey] })); }}
          style={{ background: "transparent", border: `1px solid ${isOrig ? "#1e3a5f" : "#475569"}`, color: isRewriting ? "#475569" : isOrig ? "#3b82f6" : "#475569", padding: "2px 8px", borderRadius: 3, fontSize: 9, cursor: isRewriting || !hasRewrite ? "default" : "pointer", fontFamily: "inherit", letterSpacing: "0.06em", transition: "all .15s", whiteSpace: "nowrap" }}>
          {isRewriting ? "✦ rewriting…" : isOrig ? "✦ rewritten" : "↺ original"}
        </button>
        {hasRewrite && !isRewriting && (
          <button
            onClick={() => resetRewrite(fieldKey)}
            title="Remove this AI rewrite"
            style={{ background: "transparent", border: "none", color: "#3b82f6", fontSize: 10, cursor: "pointer", padding: "2px 4px", fontFamily: "inherit", letterSpacing: "0.04em", lineHeight: 1, transition: "color .15s" }}
            onMouseEnter={e => e.currentTarget.style.color = "#f87171"}
            onMouseLeave={e => e.currentTarget.style.color = "#334155"}>
            ✕
          </button>
        )}
      </div>
    );
  };
  return (
    <div>
      {/* ── Day page header ── */}
      <div style={{ marginBottom: 24 }}>
        {/* Gradient rule + title */}
        <div style={{ height: 2, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc,transparent)", borderRadius: 1, marginBottom: 12 }} />
        <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 16 }}>
          {/* Left: weekday + date in journal gradient style */}
          <div>
            <div style={{ fontSize: 9, color: "#818cf8", letterSpacing: "0.25em", marginBottom: 4 }}>
              {new Date(entry.date + "T12:00:00").toLocaleDateString("en-US", { weekday: "long" }).toUpperCase()}
            </div>
            <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 36, letterSpacing: "0.12em", background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text", lineHeight: 1 }}>
              {(() => { const d = new Date(entry.date + "T12:00:00"); return d.toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" }).toUpperCase(); })()}
            </div>
            <div className="helper-text" style={{ marginTop: 4 }}>DAILY REVIEW · CAPTURE · ANALYSE · GROW</div>
          </div>
          {/* Right: NET P&L card with gradient border */}
          {entry.pnl !== "" && (
            <div style={{ position: "relative", padding: 1, borderRadius: 7, background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", flexShrink: 0 }}>
              <div style={{ background: "#070d1a", borderRadius: 6, padding: "12px 20px", textAlign: "right", minWidth: 140 }}>
                <div style={{ fontSize: 9, color: "#818cf8", letterSpacing: "0.15em", marginBottom: 4 }}>NET P&L</div>
                <div style={{ fontSize: 34, color: pnlColor(netPnl(entry)), fontWeight: 700, lineHeight: 1, letterSpacing: "0.02em" }}>{fmtPnl(netPnl(entry))}</div>
                {parseFloat(entry.commissions) > 0 && (
                  <div style={{ fontSize: 9, color: "#334155", marginTop: 4, lineHeight: 1.6 }}>
                    gross {fmtPnl(entry.pnl)}
                    <span style={{ color: "#475569" }}> − ${parseFloat(entry.commissions).toFixed(2)} fees</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 20 }}>
        {(entry.instruments?.length ? entry.instruments : entry.instrument ? [entry.instrument] : []).map(sym => (
          <span key={sym} style={{ display: "inline-block", padding: "3px 10px", borderRadius: 3, fontSize: 11, background: "#1e3a5f", color: "#93c5fd" }}>{sym}</span>
        ))}
        {entry.bias && <span style={{ display: "inline-block", padding: "3px 10px", borderRadius: 3, fontSize: 11, background: entry.bias === "Bullish" ? "#052e16" : entry.bias === "Bearish" ? "#450a0a" : "#1e1b4b", color: entry.bias === "Bullish" ? "#4ade80" : entry.bias === "Bearish" ? "#f87171" : "#a5b4fc" }}>{entry.bias.toUpperCase()}</span>}
        {(entry.moods?.length ? entry.moods : entry.mood ? [entry.mood] : []).map(m => (
          <span key={m} style={{ fontSize: 13 }}>{m}</span>
        ))}
        {entry.grade && <span style={{ display: "inline-block", padding: "3px 10px", borderRadius: 3, fontSize: 12, background: "#0f172a", border: `1px solid ${gradeColor(entry.grade)}`, color: gradeColor(entry.grade) }}>Grade: {entry.grade}</span>}
        {(entry.executionScore || entry.decisionScore) && (
          <span style={{ display: "inline-block", padding: "3px 10px", borderRadius: 3, fontSize: 11, background: "#0a0e1a", border: "1px solid #1e3a5f", color: "#94a3b8" }}>
            {entry.executionScore != null && <span>EXEC <span style={{ color: "#3b82f6", fontWeight: 600 }}>{entry.executionScore}/10</span></span>}
            {entry.executionScore != null && entry.decisionScore != null && <span style={{ color: "#475569" }}> · </span>}
            {entry.decisionScore != null && <span>DEC <span style={{ color: "#a855f7", fontWeight: 600 }}>{entry.decisionScore}/10</span></span>}
          </span>
        )}
      </div>
      {a && (
        <div style={{ marginBottom: 20 }}>
          <AnalyticsPanel a={a} trades={entry.parsedTrades || []} pnlColor={pnlColor} fmtPnl={fmtPnl} analyticsTab={atab} setAnalyticsTab={setAtab} totalFees={parseFloat(entry.commissions) || 0} dangerLine={dangerLine} rawCsvFile={entry.rawCsvFile || null} />
        </div>
      )}
      {/* Journal Recap */}
      {(entry.marketNotes || entry.rules || entry.sessionMistakes?.length || entry.moods?.length || entry.mood || entry.bestTrade || entry.worstTrade || entry.lessonsLearned || entry.mistakes || entry.improvements || entry.reinforceRule || entry.tomorrow) && (
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 14, display: "flex", alignItems: "center", gap: 8 }}>
            <span>📓</span> JOURNAL RECAP
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>

            {/* Session context row */}
            {(entry.sessionMistakes?.length > 0 || entry.moods?.length || entry.mood) && (
              <div style={{ display: "grid", gridTemplateColumns: entry.sessionMistakes?.length && (entry.moods?.length || entry.mood) ? "1fr 1fr" : "1fr", gap: 10 }}>
                {(entry.moods?.length || entry.mood) && (
                  <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 6, padding: "14px 16px" }}>
                    <div style={{ fontSize: 10, color: "#93c5fd", letterSpacing: "0.12em", marginBottom: 10 }}>🧠 MENTAL STATE</div>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      {(entry.moods?.length ? entry.moods : [entry.mood]).map(m => (
                        <span key={m} style={{ padding: "4px 10px", borderRadius: 20, fontSize: 12, background: "#0a1628", border: "1px solid #1e3a5f", color: "#e2e8f0" }}>{m}</span>
                      ))}
                    </div>
          </div>
          )}
                {entry.sessionMistakes?.length > 0 && (
                  <div style={{ background: "#0f1729", border: "1px solid #450a0a", borderRadius: 6, padding: "14px 16px" }}>
                    <div style={{ fontSize: 10, color: "#f87171", letterSpacing: "0.12em", marginBottom: 10 }}>⚠️ SESSION MISTAKES FLAGGED</div>
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                      {entry.sessionMistakes.map(m => (
                        <span key={m} style={{ padding: "4px 10px", borderRadius: 20, fontSize: 11, background: "rgba(63,16,16,0.5)", border: "1px solid #7f1d1d", color: "#f87171" }}>{m}</span>
                      ))}
                    </div>
                    {/* Feature 4: Mistake cost attribution display */}
                    {entry.mistakeCosts && Object.keys(entry.mistakeCosts).length > 0 && (() => {
                      const costs = entry.mistakeCosts;
                      const rows = entry.sessionMistakes.filter(m => costs[m] != null && costs[m] > 0);
                      if (rows.length === 0) return null;
                      const total = rows.reduce((s, m) => s + (parseFloat(costs[m]) || 0), 0);
                      return (
                        <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px solid #450a0a" }}>
                          <div style={{ fontSize: 9, color: "#7f1d1d", letterSpacing: "0.1em", marginBottom: 6 }}>💸 ATTRIBUTED COSTS</div>
                          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                            {rows.map(m => (
                              <div key={m} style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
                                <span style={{ color: "#94a3b8" }}>{m}</span>
                                <span style={{ color: "#fb923c", fontWeight: 500 }}>${parseFloat(costs[m]).toFixed(0)}</span>
                              </div>
                            ))}
                            {rows.length > 1 && (
                              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, borderTop: "1px solid #450a0a", paddingTop: 4, marginTop: 2 }}>
                                <span style={{ color: "#7f1d1d", letterSpacing: "0.08em" }}>TOTAL</span>
                                <span style={{ color: "#fb923c", fontWeight: 700 }}>${total.toFixed(0)}</span>
                              </div>
                            )}
                          </div>
                          {entry.mistakeCostNotes && (
                            <div style={{ marginTop: 6, fontSize: 10, color: "#64748b", fontStyle: "italic" }}>{entry.mistakeCostNotes}</div>
                          )}
                        </div>
                      );
                    })()}
          </div>
          )}
              </div>
            )}

            {/* Market & rules */}
            {entry.marketNotes && (
              <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 6, padding: "14px 16px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <div style={{ fontSize: 10, color: "#93c5fd", letterSpacing: "0.12em" }}>📈 MARKET / TAPE NOTES</div>
                </div>
                <div style={{ fontSize: 13, color: "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap", fontStyle: "normal" }}>{entry.marketNotes}</div>
              </div>
            )}
            {entry.rules && (
              <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 6, padding: "14px 16px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <div style={{ fontSize: 10, color: "#93c5fd", letterSpacing: "0.12em" }}>📋 RULES FOLLOWED / BROKEN</div>
                </div>
                <div style={{ fontSize: 13, color: "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap", fontStyle: "normal" }}>{entry.rules}</div>
              </div>
            )}

            {/* Trades */}
            {(entry.bestTrade || entry.worstTrade) && (
              <div style={{ display: "grid", gridTemplateColumns: entry.bestTrade && entry.worstTrade ? "1fr 1fr" : "1fr", gap: 10 }}>
                {entry.bestTrade && (
                  <div style={{ background: "rgba(16,63,33,0.55)", border: "1px solid #166534", borderRadius: 6, padding: "14px 16px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                      <div style={{ fontSize: 10, color: "#4ade80", letterSpacing: "0.12em" }}>✅ BEST TRADE</div>
                    </div>
                    <div style={{ fontSize: 13, color: "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap", fontStyle: "normal" }}>{entry.bestTrade}</div>
          </div>
          )}
                {entry.worstTrade && (
                  <div style={{ background: "rgba(63,16,16,0.5)", border: "1px solid #7f1d1d", borderRadius: 6, padding: "14px 16px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                      <div style={{ fontSize: 10, color: "#f87171", letterSpacing: "0.12em" }}>❌ WORST TRADE / MISTAKE</div>
                    </div>
                    <div style={{ fontSize: 13, color: "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap", fontStyle: "normal" }}>{entry.worstTrade}</div>
          </div>
          )}
              </div>
            )}

            {/* Lessons */}
            {(entry.lessonsLearned || entry.mistakes || entry.improvements) && (
              <div style={{ background: "#0a1628", border: "1px solid #1e3a5f", borderRadius: 6, padding: "16px 18px", display: "flex", flexDirection: "column", gap: 14 }}>
                <div style={{ fontSize: 10, color: "#93c5fd", letterSpacing: "0.12em" }}>🔑 LESSONS & REVIEW</div>
                {entry.lessonsLearned && (
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                      <div style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em" }}>LESSONS LEARNED</div>
                    </div>
                    <div style={{ fontSize: 13, color: "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap", fontStyle: "normal" }}>{entry.lessonsLearned}</div>
          </div>
          )}
                {entry.mistakes && (
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                      <div style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em" }}>MISTAKES TO AVOID</div>
                    </div>
                    <div style={{ fontSize: 13, color: "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap", fontStyle: "normal" }}>{entry.mistakes}</div>
          </div>
          )}
                {entry.improvements && (
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                      <div style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em" }}>AREAS FOR IMPROVEMENT</div>
                    </div>
                    <div style={{ fontSize: 13, color: "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap", fontStyle: "normal" }}>{entry.improvements}</div>
          </div>
          )}
              </div>
            )}

            {/* Rule to reinforce + tomorrow — side by side */}
            {(entry.reinforceRule || entry.tomorrow) && (
              <div style={{ display: "grid", gridTemplateColumns: entry.reinforceRule && entry.tomorrow ? "1fr 1fr" : "1fr", gap: 10 }}>
                {entry.reinforceRule && (
                  <div style={{ background: "#0f1729", border: "1px solid #1e3a5f", borderRadius: 6, padding: "14px 16px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                      <div style={{ fontSize: 10, color: "#93c5fd", letterSpacing: "0.12em" }}>🔒 ONE RULE TO REINFORCE TOMORROW</div>
                    </div>
                    <div style={{ fontSize: 13, color: "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap", fontStyle: "normal" }}>{entry.reinforceRule}</div>
          </div>
          )}
                {entry.tomorrow && (
                  <div style={{ background: "#0f1729", border: "1px solid #1e3a5f", borderRadius: 6, padding: "14px 16px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                      <div style={{ fontSize: 10, color: "#93c5fd", letterSpacing: "0.12em" }}>🗓 PLAN FOR TOMORROW</div>
                    </div>
                    <div style={{ fontSize: 13, color: "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap", fontStyle: "normal" }}>{entry.tomorrow}</div>
          </div>
          )}
              </div>
            )}

          </div>

            {/* Top 3 Findings */}
            <div style={{ marginTop: 10 }}>
              <TopFindings entry={entry} a={a} ai={ai} />
            </div>

            {/* Notes Summary */}
            {hasNotes && (
              <div style={{ marginTop: 10, background: "#0a1628", border: "1px solid #1e3a5f", borderRadius: 6, padding: "16px 18px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                  <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", fontWeight: 600 }}>📝 NOTES SUMMARY</div>
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    {noteSummary && (
                      <button onClick={() => setShowOriginalNotes(p => !p)}
                        style={{ background: "transparent", border: `1px solid ${showOriginalNotes ? "#1e3a5f" : "#475569"}`, color: showOriginalNotes ? "#3b82f6" : "#475569", padding: "3px 10px", borderRadius: 3, fontSize: 9, cursor: "pointer", fontFamily: "inherit", letterSpacing: "0.06em", transition: "all .15s" }}>
                        {showOriginalNotes ? "✦ rewritten" : "↺ original notes"}
                      </button>
                    )}
                  </div>
                </div>
                {noteSummaryLoading ? (
                  <div style={{ fontSize: 12, color: "#64748b", letterSpacing: "0.08em", animation: "pulse 1.5s infinite" }}>✦ generating summary…</div>
                ) : showOriginalNotes ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    {[
                      { label: "MARKET / TAPE NOTES", value: entry.marketNotes },
                      { label: "RULES FOLLOWED / BROKEN", value: entry.rules },
                      { label: "LESSONS LEARNED", value: entry.lessonsLearned },
                      { label: "MISTAKES TO AVOID", value: entry.mistakes },
                      { label: "AREAS FOR IMPROVEMENT", value: entry.improvements },
                      { label: "BEST TRADE", value: entry.bestTrade },
                      { label: "WORST TRADE", value: entry.worstTrade },
                      { label: "RULE TO REINFORCE", value: entry.reinforceRule },
                      { label: "PLAN FOR TOMORROW", value: entry.tomorrow },
                    ].filter(f => f.value?.trim()).map(f => (
                      <div key={f.label}>
                        <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 4 }}>{f.label}</div>
                        <div style={{ fontSize: 13, color: "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap" }}>{f.value}</div>
                      </div>
                    ))}
                  </div>
                ) : noteSummary ? (
                  <div style={{ fontSize: 13, color: "#93c5fd", lineHeight: 1.8, whiteSpace: "pre-wrap", fontStyle: "italic" }}>{noteSummary}</div>
                ) : (
                  <div style={{ fontSize: 12, color: "#64748b" }}>No notes to summarise.</div>
                )}
              </div>
            )}

        </div>
      )}

      {/* Chart Screenshots */}
      {entry.chartScreenshots?.length > 0 && (
        <ChartScreenshotDetailView screenshots={entry.chartScreenshots} />
      )}

      {/* AI Daily Analysis */}
      <DailyAIAnalysis entry={entry} a={a} ai={ai} priorPlan={priorPlan} />

      {hasNotes && onRecap && (
        <div style={{ marginTop: 20, background: "#0a0e1a", border: "1px solid #1e3a5f", borderRadius: 6, overflow: "hidden", position: "relative" }}>
          <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc)" }} />
          <div style={{ padding: "14px 20px", background: "#0a1628", borderBottom: "1px solid #1e293b", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: "0.1em", background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>✦ AI WEEKLY & MONTHLY RECAP</div>
              <div style={{ fontSize: 9, color: "#475569", marginTop: 3, letterSpacing: "0.1em" }}>ANALYSE · REFLECT · IMPROVE</div>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={() => onRecap("weekly")}
                style={{ background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", color: "#070d1a", border: "none", padding: "8px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.06em", fontWeight: 700, transition: "all .15s" }}>
                ANALYSE WEEK →
              </button>
              <button onClick={() => onRecap("monthly")}
                style={{ background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", color: "#070d1a", border: "none", padding: "8px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.06em", fontWeight: 700, transition: "all .15s" }}>
                ANALYSE MONTH →
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function CalendarView({ month, entries, onDayClick, onNewDay, pnlColor, fmtPnl, netPnl: calcNetPnlProp, gradeColor, calcAnalytics }) {

  // ── Calendar notes: state lives HERE, not in parent ─────────────────────
  // This prevents full-app re-renders on every keystroke.
  const [calendarNotes, setCalendarNotes] = useState(() => {
    try { return JSON.parse(localStorage.getItem("tj-cal-notes-v1") || "{}"); } catch { return {}; }
  });
  const saveCalNote = (dateStr, text) => {
    setCalendarNotes(prev => {
      const updated = { ...prev, [dateStr]: text };
      try { localStorage.setItem("tj-cal-notes-v1", JSON.stringify(updated)); } catch {}
      return updated;
    });
  };
  const calcNetPnl = calcNetPnlProp;
  const [year, mon] = month.split("-").map(Number);
  const [collapsed, setCollapsed] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("tj-collapse-cal-v1") || "{}");
      return { pnl: false, trades: false, time: false, symbol: false, dow: false, mistakes: false, mistakecost: false, mood: false, timeofday: false, mistakeimpact: false, cumChart: false, direction: false, behavioral: false, holdingtime: false, ...saved };
    } catch { return { pnl: false, trades: false, time: false, symbol: false, dow: false, mistakes: false, mistakecost: false, mood: false, timeofday: false, mistakeimpact: false, cumChart: false, direction: false, behavioral: false, holdingtime: false }; }
  });
  const toggleSection = (key) => setCollapsed(prev => {
    const next = { ...prev, [key]: !prev[key] };
    try { localStorage.setItem("tj-collapse-cal-v1", JSON.stringify(next)); } catch {}
    return next;
  });
  const SectionHeader = ({ label, skey, summary }) => (
    <div onClick={() => toggleSection(skey)}
      style={{ display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer", userSelect: "none", padding: "7px 12px", borderRadius: 4, marginBottom: collapsed[skey] ? 0 : 8, background: collapsed[skey] ? "#0f1729" : "transparent", border: `1px solid ${collapsed[skey] ? "#1e3a5f" : "transparent"}`, transition: "all .15s" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ fontSize: 11, color: collapsed[skey] ? "#93c5fd" : "#93c5fd", letterSpacing: "0.1em", transition: "color .15s" }}>{label}</div>
        {collapsed[skey] && summary && <div style={{ fontSize: 11, color: "#94a3b8" }}>{summary}</div>}
      </div>
      <span style={{ fontSize: 10, color: collapsed[skey] ? "#3b82f6" : "#475569", transition: "all .2s", display: "inline-block", transform: collapsed[skey] ? "rotate(-90deg)" : "rotate(0deg)" }}>▾</span>
    </div>
  );
  const firstDay = new Date(year, mon - 1, 1).getDay(); // 0=Sun
  const daysInMonth = new Date(year, mon, 0).getDate();
  const today = (() => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; })();

  // Map entries by date
  const byDate = {};
  for (const e of entries) {
    if (e.date?.startsWith(month)) byDate[e.date] = e;
  }

  // Month summary
  const monthEntries = Object.values(byDate);
  const monthPnL = monthEntries.reduce((s, e) => s + netPnl(e), 0);
  const monthWins = monthEntries.filter(e => netPnl(e) > 0).length;
  const monthLoss = monthEntries.filter(e => netPnl(e) < 0).length;
  const allMonthTrades = monthEntries.flatMap(e => e.parsedTrades || []);
  const allMonthWinningTrades = allMonthTrades.filter(t => (t.pnl - (t.commission||0)) > 0);
  const tradeWinRate = allMonthTrades.length ? Math.round(allMonthWinningTrades.length / allMonthTrades.length * 100) : null;

  // Full monthly analytics
  const monthAnalytics = allMonthTrades.length > 0 ? calcAnalytics(allMonthTrades, true) : null;
  const monthGross = monthEntries.reduce((s, e) => s + (parseFloat(e.pnl) || 0), 0);
  const monthFees = monthEntries.reduce((s, e) => s + (parseFloat(e.commissions) || 0), 0);
  // Feature 4: Aggregate mistake costs for the month
  const monthMistakeCostTotal = monthEntries.reduce((s, e) => {
    if (!e.mistakeCosts) return s;
    return s + Object.values(e.mistakeCosts).reduce((ss, v) => ss + (v != null && v > 0 ? parseFloat(v) : 0), 0);
  }, 0);

  const DAYS = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"];

  // Build grid cells
  const cells = [];
  for (let i = 0; i < firstDay; i++) cells.push(null);
  for (let d = 1; d <= daysInMonth; d++) cells.push(d);

  return (
    <div>
      {/* Full Monthly Stats Panel */}
      {monthEntries.length > 0 && (
        <div style={{ marginBottom: 20, background: "#0a0e1a", border: "1px solid #1e293b", borderRadius: 6, overflow: "hidden" }}>
          {/* Panel header */}
          <div style={{ padding: "12px 18px", background: "#0a1628", borderBottom: "1px solid #1e293b", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", fontWeight: 600 }}>
              {(() => { const [y, m] = month.split("-").map(Number); return new Date(y, m - 1, 1).toLocaleString("default", { month: "long", year: "numeric" }).toUpperCase(); })()} · MONTHLY OVERVIEW
            </div>
            <div style={{ display: "flex", gap: 8, fontSize: 10, color: "#94a3b8", alignItems: "center" }}>
              <span>{monthEntries.length} trading days</span>
              <span style={{ color: "#94a3b8" }}>·</span>
              <span style={{ letterSpacing: 0 }}><span style={{ color: "#4ade80" }}>{monthWins}</span><span style={{ color: "#94a3b8" }}>/</span><span style={{ color: "#f87171" }}>{monthLoss}</span></span>
              {monthMistakeCostTotal > 0 && (
                <><span style={{ color: "#94a3b8" }}>·</span><span style={{ color: "#fb923c" }}>💸 ${monthMistakeCostTotal.toFixed(0)}</span></>
              )}
            </div>
          </div>

          <div style={{ padding: "16px 18px", display: "flex", flexDirection: "column", gap: 16 }}>

            {/* P&L + Trade Statistics — combined */}
            <div>
              <SectionHeader label="P&L SUMMARY & TRADE STATISTICS" skey="pnl"
                summary={<span style={{ color: pnlColor(monthPnL), fontWeight: 600 }}>{fmtPnl(monthPnL)} net{monthAnalytics ? ` · ${monthAnalytics.total} trades` : ""}</span>} />
              {!collapsed.pnl && (
                <div style={{ display: "flex", flexDirection: "column", gap: 1, background: "#0a0e1a", border: "1px solid #1e293b", borderRadius: 6, overflow: "hidden" }}>

                  {/* Row 1: Money — the month's financial result */}
                  <div style={{ padding: "12px 14px 10px", borderBottom: "1px solid #0f1729" }}>
                    <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.15em", marginBottom: 8 }}>FINANCIALS</div>
                    <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1fr", gap: 8 }}>
                      {/* Net P&L — hero stat */}
                      <div style={{ background: "linear-gradient(135deg,rgba(56,189,248,0.1),rgba(129,140,248,0.12),rgba(192,132,252,0.08))", border: "1px solid rgba(129,140,248,0.25)", borderRadius: 5, padding: "12px 14px", position: "relative", overflow: "hidden" }}>
                        <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc)" }} />
                        <div style={{ fontSize: 8, color: "#818cf8", letterSpacing: "0.12em", marginBottom: 4 }}>NET P&L</div>
                        <div style={{ fontSize: 24, color: pnlColor(monthPnL), fontWeight: 700, lineHeight: 1 }}>{fmtPnl(monthPnL)}</div>
                        {monthFees > 0 && <div style={{ fontSize: 9, color: "#475569", marginTop: 3 }}>gross {fmtPnl(monthGross)}</div>}
                      </div>
                      <div style={{ background: "#0f1729", borderRadius: 5, padding: "10px 12px" }}>
                        <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 4 }}>GROSS P&L</div>
                        <div style={{ fontSize: 15, color: pnlColor(monthGross), fontWeight: 600 }}>{fmtPnl(monthGross)}</div>
                      </div>
                      <div style={{ background: "#0f1729", borderRadius: 5, padding: "10px 12px" }}>
                        <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 4 }}>FEES</div>
                        <div style={{ fontSize: 15, color: "#475569", fontWeight: 400 }}>{monthFees > 0 ? `-$${monthFees.toFixed(2)}` : "—"}</div>
                      </div>
                      {monthAnalytics ? (
                        <div style={{ background: "#0f1729", borderRadius: 5, padding: "10px 12px" }}>
                          <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 4 }}>MAX DRAWDOWN</div>
                          <div style={{ fontSize: 15, color: "#f87171", fontWeight: 600 }}>-${monthAnalytics.maxDD.toFixed(0)}</div>
                        </div>
                      ) : null}
                    </div>
                  </div>

                  {/* Row 2: Quality — how well you traded */}
                  <div style={{ padding: "10px 14px 10px", borderBottom: "1px solid #0f1729" }}>
                    <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.15em", marginBottom: 8 }}>PERFORMANCE QUALITY</div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 8 }}>
                      <div style={{ background: "#0f1729", borderRadius: 5, padding: "10px 12px" }}>
                        <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 4 }}>DAY WIN RATE</div>
                        <div style={{ fontSize: 15, color: monthEntries.length && monthWins / monthEntries.length >= 0.5 ? "#4ade80" : "#f87171", fontWeight: 600 }}>
                          {monthEntries.length ? `${Math.round(monthWins / monthEntries.length * 100)}%` : "—"}
                        </div>
                        <div style={{ fontSize: 9, color: "#64748b", marginTop: 2 }}>
                          <span style={{ color: "#4ade80" }}>{monthWins}W</span>
                          <span style={{ color: "#475569" }}> · </span>
                          <span style={{ color: "#f87171" }}>{monthLoss}L</span>
                        </div>
                      </div>
                      {monthAnalytics && <>
                        <div style={{ background: "#0f1729", borderRadius: 5, padding: "10px 12px" }}>
                          <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 4 }}>TRADE WIN RATE</div>
                          <div style={{ fontSize: 15, color: monthAnalytics.winRate >= 50 ? "#4ade80" : "#f87171", fontWeight: 600 }}>{monthAnalytics.winRate.toFixed(0)}%</div>
                          <div style={{ fontSize: 9, color: "#64748b", marginTop: 2 }}>
                            <span style={{ color: "#4ade80" }}>{monthAnalytics.winners}W</span>
                            <span style={{ color: "#475569" }}> · </span>
                            <span style={{ color: "#f87171" }}>{monthAnalytics.losers}L</span>
                          </div>
                        </div>
                        <div style={{ background: "#0f1729", borderRadius: 5, padding: "10px 12px" }}>
                          <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 4 }}>PROFIT FACTOR</div>
                          {monthAnalytics.profitFactor >= 1 ? (
                            <div style={{ fontSize: 15, fontWeight: 700, background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>
                              {fmtPF(monthAnalytics.profitFactor)}
                            </div>
                          ) : (
                            <div style={{ fontSize: 15, color: "#f87171", fontWeight: 600 }}>{fmtPF(monthAnalytics.profitFactor)}</div>
                          )}
                        </div>
                        <div style={{ background: "#0f1729", borderRadius: 5, padding: "10px 12px" }}>
                          <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 4 }}>AVG WIN / LOSS</div>
                          <div style={{ fontSize: 12, fontWeight: 600 }}>
                            <span style={{ color: "#4ade80" }}>{fmtPnl(monthAnalytics.avgWin)}</span>
                            <span style={{ color: "#475569" }}> / </span>
                            <span style={{ color: "#f87171" }}>{fmtPnl(monthAnalytics.avgLoss)}</span>
                          </div>
                        </div>
                        <div style={{ background: "#0f1729", borderRadius: 5, padding: "10px 12px" }}>
                          <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 4 }}>TOTAL TRADES</div>
                          <div style={{ fontSize: 15, color: "#e2e8f0", fontWeight: 600 }}>{monthAnalytics.total}</div>
                          <div style={{ fontSize: 9, color: "#64748b", marginTop: 2 }}>{monthAnalytics.avgQty.toFixed(1)} avg cts</div>
                        </div>
                      </>}
                    </div>
                  </div>

                  {/* Row 3: Trade detail — the underlying numbers */}
                  {monthAnalytics && (
                    <div style={{ padding: "10px 14px 12px" }}>
                      <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.15em", marginBottom: 8 }}>TRADE DETAIL</div>
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 8 }}>
                        {[
                          { l: "LARGEST WIN", v: fmtPnl(monthAnalytics.largestWin), c: "#4ade80" },
                          { l: "LARGEST LOSS", v: fmtPnl(monthAnalytics.largestLoss), c: "#f87171" },
                          { l: "BEST WIN DAY", v: fmtPnl(Math.max(...monthEntries.map(e => netPnl(e)))), c: "#4ade80" },
                          { l: "WORST LOSS DAY", v: fmtPnl(Math.min(...monthEntries.map(e => netPnl(e)))), c: "#f87171" },
                          { l: "WIN STREAK", v: `${monthAnalytics.maxConsecWins} trades`, c: "#4ade80" },
                          { l: "LOSS STREAK", v: `${monthAnalytics.maxConsecLoss} trades`, c: "#f87171" },
                        ].map(s => (
                          <div key={s.l} style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "8px 10px" }}>
                            <div style={{ fontSize: 8, color: "#94a3b8", letterSpacing: "0.08em", marginBottom: 3 }}>{s.l}</div>
                            <div style={{ fontSize: 13, color: s.c, fontWeight: 600 }}>{s.v}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

      {/* Calendar grid header */}
      <div style={{ marginBottom: 14, position: "relative" }}>
        {/* Accent top line */}
        <div style={{ height: 1, background: "linear-gradient(90deg, #38bdf8, #818cf8, #c084fc, transparent)", marginBottom: 10, opacity: 0.5 }} />
        <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
          <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 38, letterSpacing: "0.1em", lineHeight: 1, background: "linear-gradient(135deg,#38bdf8 0%,#818cf8 55%,#c084fc 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>
            {(() => { const [y, m] = month.split("-").map(Number); return new Date(y, m - 1, 1).toLocaleString("default", { month: "long" }).toUpperCase(); })()}
          </div>
          <div style={{ fontFamily: "'DM Mono',monospace", fontSize: 14, color: "#475569", letterSpacing: "0.18em", paddingBottom: 3 }}>
            {(() => { const [y] = month.split("-").map(Number); return y; })()}
          </div>
        </div>
        <div style={{ height: 1, background: "linear-gradient(90deg, transparent, #818cf8, #c084fc, transparent)", marginTop: 8, opacity: 0.25 }} />
      </div>

      {/* Day headers */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 3, marginBottom: 3 }}>
        {DAYS.map(d => (
          <div key={d} style={{ textAlign: "center", fontSize: 12, letterSpacing: "0.08em", padding: "6px 0", fontWeight: 600,
            ...(d === "SAT" || d === "SUN"
              ? { color: "#334155" }
              : { background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }
            ) }}>{d}</div>
        ))}
      </div>

      {/* Calendar grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 3 }}>
        {cells.map((day, idx) => {
          if (!day) return <div key={`empty-${idx}`} />;
          const dateStr = `${month}-${String(day).padStart(2, "0")}`;
          const entry = byDate[dateStr];
          const isToday = dateStr === today;
          const dow = (firstDay + day - 1) % 7;
          const isWeekend = dow === 0 || dow === 6;
          const n = entry ? netPnl(entry) : null;
          const hasEntry = !!entry;
          const isPast = dateStr < today;

          const isFuture = dateStr > today;

          let bgColor = "#0a0e1a";
          let borderColor = "#1e293b";
          if (hasEntry) {
            bgColor = n > 0 ? "rgba(16,63,33,0.55)" : n < 0 ? "rgba(63,16,16,0.5)" : "#0f1729";
            borderColor = n > 0 ? "#166534" : n < 0 ? "#7f1d1d" : "#1e293b";
          } else if (isWeekend) {
            bgColor = "#060810";
            borderColor = "#0d1018";
          }
          // Future weekday cells use the same default styling as any empty current-month cell
          // isToday gets gradient outline instead of blue border — applied inline below

          return (
            <div key={dateStr}
              onClick={() => hasEntry ? onDayClick(entry) : !isWeekend ? onNewDay(dateStr) : null}
              style={{
                background: bgColor, border: `1px solid ${borderColor}`, borderRadius: 4,
                padding: "12px 12px 10px", minHeight: 172, position: "relative",
                display: "flex", flexDirection: "column",
                cursor: hasEntry ? "pointer" : !isWeekend ? "pointer" : "default",
                transition: "all .15s", opacity: (isWeekend && !hasEntry && !calendarNotes[dateStr]) ? 0.3 : 1,
                ...(isToday ? { outline: "2px solid transparent", boxShadow: "0 0 0 2px #818cf8, 0 0 0 3px rgba(56,189,248,0.4), 0 0 0 4px rgba(192,132,252,0.3)" } : {}),
              }}
              onMouseEnter={e => { if (hasEntry || !isWeekend) e.currentTarget.style.borderColor = hasEntry ? (n > 0 ? "#22c55e" : n < 0 ? "#ef4444" : "#3b82f6") : "#475569"; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = borderColor; }}
            >
              {/* Row 1: day number + emoji + grade */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 7 }}>
                <span style={{ fontSize: 19, fontWeight: 700, ...(isToday
                    ? { background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }
                    : { color: hasEntry ? "#e2e8f0" : isWeekend ? "#475569" : "#94a3b8" }
                  ) }}>{day}</span>
                {entry?.grade && (
                  <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
                    {(entry.moods?.length ? entry.moods : entry.mood ? [entry.mood] : []).map(m => (
                      <span key={m} style={{ fontSize: 10 }}>{m.split(" ").pop()}</span>
                    ))}
                    <span style={{ fontSize: 9, fontWeight: 700, padding: "2px 5px", borderRadius: 3, background: gradeColor(entry.grade) + "22", border: `1px solid ${gradeColor(entry.grade)}`, color: gradeColor(entry.grade), letterSpacing: "0.06em" }}>{entry.grade}</span>
          </div>
          )}
              </div>

              {hasEntry ? (
                <>
                  {/* P&L — most important, largest */}
                  <div style={{ fontSize: 20, fontWeight: 700, color: n > 0 ? "#4ade80" : n < 0 ? "#f87171" : "#e2e8f0", lineHeight: 1, marginBottom: 7 }}>
                    {n > 0 ? "+" : n < 0 ? "-" : ""}{n !== null ? `$${Math.abs(n).toLocaleString("en-US", { minimumFractionDigits: n % 1 === 0 ? 0 : 2, maximumFractionDigits: 2 })}` : "—"}
                  </div>
                  {/* Trades + Win Rate */}
                  {entry.parsedTrades?.length > 0 && (() => {
                    const tN = (t) => t.pnl - (t.commission||0);
                    const wins = entry.parsedTrades.filter(t => tN(t) > 0).length;
                    const losses = entry.parsedTrades.filter(t => tN(t) < 0).length;
                    const gross = entry.parsedTrades.reduce((s,t) => tN(t)>0?s+tN(t):s, 0);
                    const grossLoss = entry.parsedTrades.reduce((s,t) => tN(t)<0?s+Math.abs(tN(t)):s, 0);
                    const pf = grossLoss > 0 ? (gross/grossLoss) : gross > 0 ? Infinity : null;
                    return (
                      <>
                        <div style={{ fontSize: 11, lineHeight: 1.5 }}>
                          <span style={{ color: "#4ade80" }}>W{wins}</span>
                          <span style={{ color: "#475569" }}> / </span>
                          <span style={{ color: "#f87171" }}>L{losses}</span>
                          <span style={{ color: "#475569" }}> · {entry.parsedTrades.length}T</span>
                        </div>
                        {pf !== null && (
                          pf >= 1 ? (
                            <div style={{ fontSize: 11, fontWeight: 700, background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text", lineHeight: 1.4 }}>PF {pf.toFixed(2)}</div>
                          ) : (
                            <div style={{ fontSize: 11, color: "#f87171", lineHeight: 1.4 }}>PF {pf.toFixed(2)}</div>
                          )
                        )}
                      </>
                    );
                  })()}
                  {/* Instruments — in flow, bottom of entry content */}
                  <div style={{ marginTop: 5, display: "flex", gap: 3, flexWrap: "wrap" }}>
                    {(entry.instruments?.length ? entry.instruments : entry.instrument ? [entry.instrument] : []).map(sym => (
                      <span key={sym} style={{ fontSize: 10, padding: "1px 6px", borderRadius: 2, background: "#1e3a5f33", color: "#3b82f6", border: "1px solid #1e3a5f55", letterSpacing: "0.04em" }}>{sym}</span>
                    ))}
                    {!entry.grade && (entry.moods?.length ? entry.moods : entry.mood ? [entry.mood] : []).map(m => (
                      <span key={m} style={{ fontSize: 11 }}>{m.split(" ").pop()}</span>
                    ))}
                  </div>
                </>
              ) : !isWeekend ? (
                <div style={{ fontSize: 10, color: isFuture ? "#1e3a5f" : "#334155", marginTop: 6, textAlign: "center", letterSpacing: "0.08em" }}>{isFuture ? "＋ plan" : "+ add"}</div>
              ) : null}
              {/* Note area — weekend plans / quick notes */}
              <div style={{ marginTop: "auto", paddingTop: 8, marginTop: 8, borderTop: calendarNotes[dateStr] ? "1px solid #1e293b" : "1px solid transparent" }}
                onClick={e => e.stopPropagation()}>
                <textarea
                  value={calendarNotes[dateStr] || ""}
                  onChange={e => {
                    e.stopPropagation();
                    saveCalNote(dateStr, e.target.value);
                    // Height managed purely by the field-sizing CSS — no imperative DOM resize
                    e.target.parentNode.style.borderTopColor = e.target.value ? "#1e293b" : "transparent";
                  }}
                  onClick={e => e.stopPropagation()}
                  placeholder={isWeekend ? "weekend plan…" : "notes…"}
                  style={{ width: "100%", fontSize: 12, background: "transparent", border: "none", resize: "none",
                    fontFamily: "DM Mono,monospace", outline: "none", padding: "3px 0 0 0", lineHeight: 1.5,
                    display: "block", overflow: "hidden", minHeight: "20px",
                    // Don't control height via React style — let field-sizing: content handle it.
                    // This prevents the collapse-then-expand flicker on every re-render.
                    textAlign: "center" }}
                  className="cal-note"
                  onFocus={e => {
                    e.currentTarget.style.setProperty('color', '#fde68a', 'important');
                    e.currentTarget.parentNode.style.borderTopColor = "rgba(251,191,36,0.4)";
                  }}
                  onBlur={e => {
                    e.currentTarget.style.setProperty('color', '#fbbf24', 'important');
                    const val = e.currentTarget.value;
                    e.currentTarget.parentNode.style.borderTopColor = val ? "#1e293b" : "transparent";
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div style={{ display: "flex", gap: 16, marginTop: 12, fontSize: 10, color: "#64748b" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}><div style={{ width: 8, height: 8, borderRadius: 2, background: "rgba(16,63,33,0.55)", border: "1px solid #166534" }} /> Win day</div>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}><div style={{ width: 8, height: 8, borderRadius: 2, background: "rgba(63,16,16,0.5)", border: "1px solid #7f1d1d" }} /> Loss day</div>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}><div style={{ width: 8, height: 8, borderRadius: 2, border: "1px solid #3b82f6" }} /> Today</div>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}><div style={{ width: 8, height: 8, borderRadius: 2, background: "#0a0e1a", border: "1px solid #334155", opacity: 0.4 }} /> Weekend / no trade</div>
      </div>

      {/* Cumulative P&L Chart — below calendar */}
      {monthEntries.length > 0 && (() => {
        const sorted = [...monthEntries].sort((a, b) => a.date.localeCompare(b.date));
        let running = 0;
        const points = sorted.map(e => { running += netPnl(e); return { date: e.date, cum: running, daily: netPnl(e) }; });
        const lineColor = points[points.length - 1].cum >= 0 ? "#4ade80" : "#f87171";
        const chartVals = points.length === 1 ? [0, points[0].cum] : points.map(p => p.cum);
        return (
          <div style={{ marginTop: 16 }}>
            <SectionHeader label="CUMULATIVE P&L" skey="cumChart"
              summary={<span style={{ color: lineColor, fontWeight: 600 }}>{fmtPnl(monthPnL)}</span>} />
            {!collapsed.cumChart && (
              <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 6, padding: "16px 18px" }}>
                <EquityCurveChart values={chartVals} dots={points.length > 1} height={110} gradientId="ec2" />
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8 }}>
                  {points.map((p, i) => {
                    const show = points.length <= 8 || i === 0 || i === points.length - 1 || i % Math.ceil(points.length / 6) === 0;
                    return (
                      <div key={i} style={{ fontSize: 9, color: show ? "#475569" : "transparent", textAlign: "center", flex: 1 }}>
                        {new Date(p.date + "T12:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        );
      })()}


            {/* Performance by Day of Week */}
            {monthEntries.length > 0 && (() => {
              const DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"];
              const dowStats = {};
              for (const day of DOW) dowStats[day] = { pnl: 0, trades: 0, tradeWins: 0, tradeLosses: 0, wins: 0, days: 0 };
              for (const e of monthEntries) {
                const d = new Date(e.date + "T12:00:00");
                const day = DOW[d.getDay() - 1];
                if (!day) continue;
                const np = netPnl(e);
                dowStats[day].pnl += np;
                dowStats[day].days++;
                if (np > 0) dowStats[day].wins++;
                const trades = e.parsedTrades || [];
                dowStats[day].trades += trades.length;
                dowStats[day].tradeWins += trades.filter(t => (t.pnl-(t.commission||0)) > 0).length;
                dowStats[day].tradeLosses += trades.filter(t => (t.pnl-(t.commission||0)) < 0).length;
              }
              const activeDays = DOW.filter(d => dowStats[d].days > 0);
              if (!activeDays.length) return null;
              const maxAbsPnl = Math.max(...activeDays.map(d => Math.abs(dowStats[d].pnl)), 1);
              const bestDay = activeDays.reduce((a, b) => dowStats[a].pnl > dowStats[b].pnl ? a : b);
              const worstDay = activeDays.reduce((a, b) => dowStats[a].pnl < dowStats[b].pnl ? a : b);
              return (
                <div>
                  <SectionHeader label="PERFORMANCE BY DAY OF WEEK" skey="dow" summary={<span style={{ color: "#64748b" }}>Best: {bestDay.slice(0,3)}</span>} />
                  {!collapsed.dow && (
                    <div>
                      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 8, fontSize: 10, color: "#64748b" }}>
                        Best: <span style={{ color: "#4ade80", marginLeft: 4 }}>{bestDay}</span>
                        {dowStats[worstDay].pnl < 0 && <> · Avoid: <span style={{ color: "#f87171", marginLeft: 4 }}>{worstDay}</span></>}
                      </div>
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 8 }}>
                        {DOW.map(day => {
                          const s = dowStats[day];
                          const hasData = s.days > 0;
                          const wr = hasData ? Math.round(s.wins / s.days * 100) : 0;
                          const barH = hasData ? Math.abs(s.pnl) / maxAbsPnl * 60 : 0;
                          const isPos = s.pnl >= 0;
                          const isBest = day === bestDay && hasData;
                          const isWorst = day === worstDay && hasData && s.pnl < 0;
                          return (
                            <div key={day} style={{ background: isBest ? "#061f0f" : isWorst ? "#1f0606" : "#0a0e1a", border: `1px solid ${isBest ? "#166534" : isWorst ? "#7f1d1d" : "#1e293b"}`, borderRadius: 5, padding: "12px 10px", textAlign: "center", opacity: hasData ? 1 : 0.3 }}>
                              <div style={{ fontSize: 10, color: isBest ? "#4ade80" : isWorst ? "#f87171" : "#64748b", letterSpacing: "0.08em", marginBottom: 8 }}>{day.slice(0,3).toUpperCase()}</div>
                              <div style={{ height: 64, display: "flex", alignItems: "flex-end", justifyContent: "center", marginBottom: 8 }}>
                                {hasData ? (
                                  <div style={{ width: 28, borderRadius: "3px 3px 0 0", background: isPos ? "#4ade80" : "#f87171", height: `${Math.max(barH, 4)}px`, opacity: 0.8, transition: "height .3s" }} />
                                ) : (
                                  <div style={{ width: 28, height: 4, background: "#1e293b", borderRadius: 2 }} />
                                )}
                              </div>
                              {hasData ? (
                                <>
                                  <div style={{ fontSize: 13, fontWeight: 600, color: isPos ? "#4ade80" : "#f87171", marginBottom: 4 }}>
                                    {s.pnl >= 0 ? "+" : "-"}${Math.abs(s.pnl).toFixed(0)}
                                  </div>
                                  <div style={{ fontSize: 10, color: wr >= 50 ? "#4ade80" : "#f87171", marginBottom: 3 }}>{wr}% WR</div>
                                  <div style={{ fontSize: 9, color: "#64748b" }}>
                                    {s.days} day{s.days !== 1 ? "s" : ""} · <span style={{ color: "#4ade80" }}>{s.tradeWins}</span><span style={{ color: "#475569" }}>/</span><span style={{ color: "#f87171" }}>{s.tradeLosses}</span>
                                  </div>
                                </>
                              ) : (
                                <div style={{ fontSize: 9, color: "#1e3a5f" }}>no data</div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              );
            })()}

            {/* Performance by Symbol */}
            {monthAnalytics && Object.keys(monthAnalytics.bySymbol).length > 0 && (
              <div>
                <SectionHeader label="PERFORMANCE BY SYMBOL" skey="symbol" summary={<span style={{ color: "#64748b" }}>{Object.keys(monthAnalytics.bySymbol).join(", ")}</span>} />
                {!collapsed.symbol && (
                <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px", display: "flex", flexDirection: "column", gap: 12 }}>
                  {Object.entries(monthAnalytics.bySymbol).map(([sym, s]) => {
                    const maxPnl = Math.max(...Object.values(monthAnalytics.bySymbol).map(x => Math.abs(x.pnl)), 1);
                    const wr = Math.round(s.wins / s.trades * 100);
                    const barW = Math.abs(s.pnl) / maxPnl * 100;
                    return (
                      <div key={sym}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 5 }}>
                          <span style={{ fontSize: 11, color: "#93c5fd", fontWeight: 500 }}>{sym}</span>
                          <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
                            <span style={{ fontSize: 10, color: wr >= 50 ? "#4ade80" : "#f87171" }}>{wr}% WR</span>
                            <span style={{ fontSize: 10, letterSpacing: 0 }}><span style={{ letterSpacing: 0 }}><span style={{ color: "#4ade80" }}>{s.wins}</span><span style={{ color: "#94a3b8" }}>/</span><span style={{ color: "#f87171" }}>{s.trades - s.wins}</span></span></span>
                            <span style={{ fontSize: 12, fontWeight: 600, color: s.pnl >= 0 ? "#4ade80" : "#f87171", minWidth: 80, textAlign: "right" }}>{s.pnl >= 0 ? "+" : "-"}${Math.abs(s.pnl).toFixed(2)}</span>
                          </div>
                        </div>
                        <div style={{ background: "#0a0e1a", borderRadius: 2, height: 5, overflow: "hidden" }}>
                          <div style={{ width: `${barW}%`, height: "100%", background: s.pnl >= 0 ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
                )}
              </div>
            )}

            {/* Behavioral Edge Checks */}
            {monthAnalytics && (monthAnalytics.afterLoss?.total > 0 || monthAnalytics.afterWin?.total > 0 || monthAnalytics.first3?.total > 0) && (
              <div>
                <SectionHeader label="BEHAVIORAL EDGE CHECKS" skey="behavioral"
                  summary={<span style={{ color: "#64748b" }}>after loss · after win · first 3</span>} />
                {!collapsed.behavioral && (
                  <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px" }}>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 10 }}>
                      {[
                        { label: "AFTER A LOSS",   data: monthAnalytics.afterLoss },
                        { label: "AFTER A WIN",    data: monthAnalytics.afterWin  },
                        { label: "FIRST 3 TRADES", data: monthAnalytics.first3    },
                        { label: "REST OF SESSION", data: monthAnalytics.rest     },
                      ].map(card => (
                        <div key={card.label} style={{ background: "#0a0e1a", border: "1px solid #1e293b", borderRadius: 4, padding: "10px 12px" }}>
                          <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 6 }}>{card.label}</div>
                          {card.data?.total ? (
                            <>
                              <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
                                <div style={{ fontSize: 13, color: "#e2e8f0" }}>{card.data.winRate.toFixed(0)}% WR</div>
                                <div style={{ fontSize: 13, fontWeight: 600, color: (card.label === "AFTER A LOSS" || card.label === "AFTER A WIN") ? (card.data.avgPnl >= 0 ? "#4ade80" : "#f87171") : (card.data.pnl >= 0 ? "#4ade80" : "#f87171") }}>
                                  {card.label === "AFTER A LOSS" || card.label === "AFTER A WIN"
                                    ? `${card.data.avgPnl >= 0 ? "+" : "-"}$${Math.abs(card.data.avgPnl).toFixed(0)}/trade`
                                    : `${card.data.pnl >= 0 ? "+" : "-"}$${Math.abs(card.data.pnl).toFixed(0)}`}
                                </div>
                              </div>
                              <div style={{ fontSize: 10, color: "#64748b", marginTop: 6 }}>n={card.data.total} · avg {card.data.avgQty.toFixed(1)} cts</div>
                            </>
                          ) : <div style={{ fontSize: 12, color: "#64748b" }}>Not enough data</div>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Performance by Holding Time */}
            {monthAnalytics?.byDuration && Object.values(monthAnalytics.byDuration).some(b => b.trades > 0) && (
              <div>
                <SectionHeader label="PERFORMANCE BY HOLDING TIME" skey="holdingtime"
                  summary={<span style={{ color: "#64748b" }}>{Object.values(monthAnalytics.byDuration).filter(b => b.trades > 0).length} buckets</span>} />
                {!collapsed.holdingtime && (
                  <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px", display: "flex", flexDirection: "column", gap: 12 }}>
                    {(monthAnalytics.DURATION_BUCKETS || []).map(bucket => {
                      const b = monthAnalytics.byDuration[bucket.key];
                      if (!b || b.trades === 0) return null;
                      const wr = Math.round(b.wins / b.trades * 100);
                      const maxPnl = Math.max(...(monthAnalytics.DURATION_BUCKETS || []).map(bk => Math.abs(monthAnalytics.byDuration[bk.key]?.pnl || 0)), 1);
                      const barW = Math.abs(b.pnl) / maxPnl * 100;
                      const avg = b.pnl / b.trades;
                      const fmtSecs = s => !s ? "—" : s < 60 ? `${Math.round(s)}s` : s < 3600 ? `${Math.floor(s/60)}m ${Math.round(s%60)}s` : `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`;
                      return (
                        <div key={bucket.key}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 5 }}>
                            <div>
                              <span style={{ fontSize: 11, color: "#e2e8f0", fontWeight: 500 }}>{bucket.label}</span>
                              <span style={{ fontSize: 9, color: "#64748b", marginLeft: 8 }}>avg {fmtSecs(b.avgSecs)}</span>
                            </div>
                            <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
                              <span style={{ fontSize: 10, color: wr >= 50 ? "#4ade80" : "#f87171" }}>{wr}% WR</span>
                              <span style={{ fontSize: 10 }}><span style={{ color: "#4ade80" }}>{b.wins}</span><span style={{ color: "#94a3b8" }}>/</span><span style={{ color: "#f87171" }}>{b.losses}</span></span>
                              <span style={{ fontSize: 12, fontWeight: 600, color: b.pnl >= 0 ? "#4ade80" : "#f87171", minWidth: 80, textAlign: "right" }}>{avg >= 0 ? "+" : "-"}${Math.abs(avg).toFixed(0)} avg</span>
                            </div>
                          </div>
                          <div style={{ background: "#0a0e1a", borderRadius: 2, height: 5, overflow: "hidden" }}>
                            <div style={{ width: `${barW}%`, height: "100%", background: b.pnl >= 0 ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {/* Long vs Short */}
            {monthAnalytics?.byDirection && (monthAnalytics.byDirection.long.trades > 0 || monthAnalytics.byDirection.short.trades > 0) && (() => {
              const dirs = [
                { key: "long",  label: "Long",  emoji: "📈", accentColor: "#4ade80", dimColor: "#166534", bgColor: "rgba(16,63,33,0.35)" },
                { key: "short", label: "Short", emoji: "📉", accentColor: "#f87171", dimColor: "#7f1d1d", bgColor: "rgba(63,16,16,0.35)" },
              ].filter(d => monthAnalytics.byDirection[d.key].trades > 0);
              const maxPnl = Math.max(Math.abs(monthAnalytics.byDirection.long.pnl), Math.abs(monthAnalytics.byDirection.short.pnl), 1);
              const totalTrades = (monthAnalytics.byDirection.long.trades || 0) + (monthAnalytics.byDirection.short.trades || 0);
              return (
                <div>
                  <SectionHeader label="LONG vs SHORT" skey="direction"
                    summary={<span style={{ color: "#64748b" }}>{totalTrades} trades · {dirs.length === 2 ? `${Math.round(monthAnalytics.byDirection.long.trades / totalTrades * 100)}% long` : dirs[0]?.label + " only"}</span>} />
                  {!collapsed.direction && (
                    <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px" }}>
                      {/* Side-by-side cards */}
                      <div style={{ display: "grid", gridTemplateColumns: dirs.length === 2 ? "1fr 1fr" : "1fr", gap: 10, marginBottom: 14 }}>
                        {dirs.map(({ key, label, emoji, accentColor, dimColor, bgColor }) => {
                          const d = monthAnalytics.byDirection[key];
                          const wr = Math.round(d.wins / d.trades * 100);
                          const avg = d.pnl / d.trades;
                          const pf = d.grossLoss > 0 ? (d.grossWin / d.grossLoss) : d.grossWin > 0 ? Infinity : null;
                          const sharePct = Math.round(d.trades / totalTrades * 100);
                          return (
                            <div key={key} style={{ background: bgColor, border: `1px solid ${dimColor}`, borderRadius: 5, padding: "12px 14px", position: "relative", overflow: "hidden" }}>
                              <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: accentColor, opacity: 0.5 }} />
                              {/* Header row */}
                              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                  <span style={{ fontSize: 14 }}>{emoji}</span>
                                  <span style={{ fontSize: 12, color: accentColor, fontWeight: 600, letterSpacing: "0.08em" }}>{label.toUpperCase()}</span>
                                </div>
                                <span style={{ fontSize: 10, color: "#475569" }}>{sharePct}% of trades</span>
                              </div>
                              {/* P&L hero */}
                              <div style={{ fontSize: 22, fontWeight: 700, color: d.pnl >= 0 ? "#4ade80" : "#f87171", lineHeight: 1, marginBottom: 8 }}>
                                {d.pnl >= 0 ? "+" : "-"}${Math.abs(d.pnl).toFixed(0)}
                              </div>
                              {/* Stats row */}
                              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6 }}>
                                <div style={{ background: "#0a0e1a", borderRadius: 3, padding: "6px 8px" }}>
                                  <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 2 }}>WIN RATE</div>
                                  <div style={{ fontSize: 13, color: wr >= 50 ? "#4ade80" : "#f87171", fontWeight: 600 }}>{wr}%</div>
                                  <div style={{ fontSize: 9, color: "#334155", marginTop: 1 }}><span style={{ color: "#4ade80" }}>{d.wins}</span><span style={{ color: "#334155" }}>W</span> <span style={{ color: "#f87171" }}>{d.losses}</span><span style={{ color: "#334155" }}>L</span></div>
                                </div>
                                <div style={{ background: "#0a0e1a", borderRadius: 3, padding: "6px 8px" }}>
                                  <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 2 }}>AVG/TRADE</div>
                                  <div style={{ fontSize: 13, color: avg >= 0 ? "#4ade80" : "#f87171", fontWeight: 600 }}>{avg >= 0 ? "+" : "-"}${Math.abs(avg).toFixed(0)}</div>
                                  <div style={{ fontSize: 9, color: "#334155", marginTop: 1 }}>{d.trades} trades</div>
                                </div>
                                <div style={{ background: "#0a0e1a", borderRadius: 3, padding: "6px 8px" }}>
                                  <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 2 }}>PROF. FACTOR</div>
                                  <div style={{ fontSize: 13, color: pfColor(pf), fontWeight: 600 }}>{fmtPF(pf)}</div>
                                  <div style={{ fontSize: 9, color: "#334155", marginTop: 1 }}>+${d.grossWin.toFixed(0)} / -${d.grossLoss.toFixed(0)}</div>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      {/* Bar comparison */}
                      {dirs.length === 2 && (() => {
                        const l = monthAnalytics.byDirection.long;
                        const s = monthAnalytics.byDirection.short;
                        const lBarW = Math.abs(l.pnl) / maxPnl * 100;
                        const sBarW = Math.abs(s.pnl) / maxPnl * 100;
                        const bestDir = l.pnl >= s.pnl ? "Long" : "Short";
                        return (
                          <div style={{ background: "#0a0e1a", borderRadius: 4, padding: "10px 12px" }}>
                            <div style={{ marginBottom: 8 }}>
                              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3, fontSize: 9, color: "#475569" }}>
                                <span>📈 Long</span><span style={{ color: l.pnl >= 0 ? "#4ade80" : "#f87171" }}>{l.pnl >= 0 ? "+" : "-"}${Math.abs(l.pnl).toFixed(0)}</span>
                              </div>
                              <div style={{ background: "#0f1729", borderRadius: 2, height: 5, overflow: "hidden" }}>
                                <div style={{ width: `${lBarW}%`, height: "100%", background: l.pnl >= 0 ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} />
                              </div>
                            </div>
                            <div>
                              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3, fontSize: 9, color: "#475569" }}>
                                <span>📉 Short</span><span style={{ color: s.pnl >= 0 ? "#4ade80" : "#f87171" }}>{s.pnl >= 0 ? "+" : "-"}${Math.abs(s.pnl).toFixed(0)}</span>
                              </div>
                              <div style={{ background: "#0f1729", borderRadius: 2, height: 5, overflow: "hidden" }}>
                                <div style={{ width: `${sBarW}%`, height: "100%", background: s.pnl >= 0 ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} />
                              </div>
                            </div>
                            <div style={{ marginTop: 8, fontSize: 10, color: "#64748b", lineHeight: 1.6 }}>
                              <span style={{ color: "#3b82f6" }}>💡 </span>
                              Stronger edge on <span style={{ color: bestDir === "Long" ? "#4ade80" : "#f87171" }}>{bestDir}</span> trades this month
                              {" · "}{Math.round(l.trades / totalTrades * 100)}% long / {Math.round(s.trades / totalTrades * 100)}% short split
                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  )}
                </div>
              );
            })()}

            {/* Mistake Frequency + Mood vs Performance — side by side */}
            {monthEntries.length > 0 && (() => {
              // ── Mistake Frequency data ──
              const mistakeCounts = {};
              let totalSessions = 0;
              for (const e of monthEntries) {
                const mistakes = (e.sessionMistakes || []).filter(m => m !== "No Mistakes — Executed the Plan ✓");
                if (mistakes.length > 0) totalSessions++;
                for (const m of mistakes) mistakeCounts[m] = (mistakeCounts[m] || 0) + 1;
              }
              const mistakeSorted = Object.entries(mistakeCounts).sort((a, b) => b[1] - a[1]);
              const cleanSessions = monthEntries.filter(e => e.sessionMistakes?.includes("No Mistakes — Executed the Plan ✓")).length;
              const maxCount = mistakeSorted[0]?.[1] || 1;
              const hasMistakes = mistakeSorted.length > 0 || cleanSessions > 0;

              // Compute P&L impact per mistake for combined display
              const mistakePerf = {};
              for (const e of monthEntries) {
                const ms = (e.sessionMistakes || []).filter(m => m !== "No Mistakes — Executed the Plan ✓");
                if (!ms.length) continue;
                const pnl = netPnl(e);
                for (const m of ms) {
                  if (!mistakePerf[m]) mistakePerf[m] = { sessions: 0, totalPnl: 0, wins: 0 };
                  mistakePerf[m].sessions++;
                  mistakePerf[m].totalPnl += pnl;
                  if (pnl > 0) mistakePerf[m].wins++;
                }
              }
              const cleanDaySessions = monthEntries.filter(e => e.sessionMistakes?.includes("No Mistakes — Executed the Plan ✓"));
              const cleanAvg = cleanDaySessions.length ? cleanDaySessions.reduce((s,e) => s + netPnl(e), 0) / cleanDaySessions.length : null;

              // ── Mood vs Performance data ──
              const moodStats = {};
              for (const e of monthEntries) {
                const moods = e.moods?.length ? e.moods : e.mood ? [e.mood] : [];
                const ep = netPnl(e);
                for (const m of moods) {
                  if (!moodStats[m]) moodStats[m] = { pnl: 0, sessions: 0, wins: 0 };
                  moodStats[m].pnl += ep;
                  moodStats[m].sessions++;
                  if (ep > 0) moodStats[m].wins++;
                }
              }
              const moodEntries = Object.entries(moodStats).filter(([, s]) => s.sessions >= 1).sort((a, b) => (b[1].pnl / b[1].sessions) - (a[1].pnl / a[1].sessions));
              const maxAvg = Math.max(...moodEntries.map(([, s]) => Math.abs(s.pnl / s.sessions)), 1);
              const hasMoods = moodEntries.length > 0;

              if (!hasMistakes && !hasMoods) return null;

              return (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, alignItems: "stretch" }}>

                  {/* LEFT — Mistake Frequency */}
                  <div style={{ minWidth: 0, display: "flex", flexDirection: "column" }}>
                    <SectionHeader label="MISTAKE FREQUENCY" skey="mistakes"
                      summary={<span style={{ color: "#64748b" }}>{mistakeSorted.length} types · {cleanSessions} clean day{cleanSessions !== 1 ? "s" : ""}</span>} />
                    {!collapsed.mistakes && (
                      <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px", flex: 1, display: "flex", flexDirection: "column" }}>
                        {cleanSessions > 0 && (
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: mistakeSorted.length ? 10 : 0, padding: "6px 10px", background: "rgba(16,63,33,0.55)", border: "1px solid #166534", borderRadius: 4 }}>
                            <span style={{ fontSize: 11, color: "#4ade80" }}>✓ Executed the Plan</span>
                            <span style={{ fontSize: 12, fontWeight: 600, color: "#4ade80" }}>{cleanSessions} day{cleanSessions !== 1 ? "s" : ""}</span>
                          </div>
                        )}
                        {hasMistakes ? mistakeSorted.map(([mistake, count]) => {
                          const barW = (count / maxCount) * 100;
                          const freq = Math.round((count / monthEntries.length) * 100);
                          const perf = mistakePerf[mistake];
                          const avg = perf ? perf.totalPnl / perf.sessions : null;
                          const delta = avg !== null && cleanAvg !== null ? avg - cleanAvg : null;
                          return (
                            <div key={mistake} style={{ marginBottom: 10 }}>
                              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
                                <span style={{ fontSize: 11, color: "#e2e8f0" }}>{mistake}</span>
                                <div style={{ display: "flex", gap: 8, alignItems: "center", flexShrink: 0, marginLeft: 8 }}>
                                  {avg !== null && <span style={{ fontSize: 11, fontWeight: 600, color: avg >= 0 ? "#4ade80" : "#f87171" }}>{avg >= 0 ? "+" : "-"}${Math.abs(avg).toFixed(0)} avg</span>}
                                  {delta !== null && <span style={{ fontSize: 9, color: delta < -30 ? "#f87171" : delta < 0 ? "#f59e0b" : "#4ade80" }}>{delta >= 0 ? "+" : ""}${delta.toFixed(0)} vs✓</span>}
                                  <span style={{ fontSize: 9, color: "#94a3b8" }}>{freq}%</span>
                                  <span style={{ fontSize: 12, fontWeight: 600, color: "#f87171" }}>{count}×</span>
                                </div>
                              </div>
                              <div style={{ background: "#0a0e1a", borderRadius: 2, height: 4, overflow: "hidden" }}>
                                <div style={{ width: `${barW}%`, height: "100%", background: "#f87171", borderRadius: 2, opacity: 0.7 }} />
                              </div>
                            </div>
                          );
                        }) : <div style={{ fontSize: 11, color: "#475569" }}>No mistakes flagged.</div>}
                        {cleanAvg !== null && mistakeSorted.length > 0 && (
                          <div style={{ marginTop: 8, fontSize: 10, color: "#64748b", lineHeight: 1.6 }}>
                            <span style={{ color: "#3b82f6" }}>💡 </span>
                            Clean day avg: <span style={{ color: "#4ade80", fontWeight: 600 }}>{cleanAvg >= 0 ? "+" : "-"}${Math.abs(cleanAvg).toFixed(0)}</span> · "vs✓" shows drag vs your clean sessions
                          </div>
                        )}
                        {/* ── Trend Recap insights — moved from removed Trend Recap section ── */}
                        {(() => {
                          const topM = Object.entries((() => {
                            const c = {};
                            for (const e of monthEntries) {
                              for (const m of (e.sessionMistakes||[]).filter(x=>x!=="No Mistakes — Executed the Plan ✓")) c[m]=(c[m]||0)+1;
                            }
                            return c;
                          })()).sort((a,b)=>b[1]-a[1])[0];
                          const netTrend = (() => {
                            const half = Math.floor(monthEntries.length/2);
                            const sorted = [...monthEntries].sort((a,b)=>a.date.localeCompare(b.date));
                            const f = sorted.slice(0,half).reduce((s,e)=>s+netPnl(e),0);
                            const s = sorted.slice(half).reduce((s,e)=>s+netPnl(e),0);
                            return s>f?"improving":s<f?"declining":"flat";
                          })();
                          const trendColor = netTrend==="improving"?"#4ade80":netTrend==="declining"?"#f87171":"#94a3b8";
                          const pf = monthAnalytics?.profitFactor;
                          if (!topM && !netTrend) return null;
                          return (
                            <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 6, paddingTop: 10, borderTop: "1px solid #1e293b" }}>
                              {topM && (
                                <div style={{ background: "rgba(63,16,16,0.4)", border: "1px solid #7f1d1d", borderRadius: 4, padding: "7px 10px", fontSize: 10, color: "#f87171" }}>
                                  ⚠ Most frequent mistake: <strong>{topM[0]}</strong> ({topM[1]}× · {Math.round(topM[1]/monthEntries.length*100)}% of sessions)
                                </div>
                              )}
                              <div style={{ fontSize: 10, color: "#64748b", lineHeight: 1.7 }}>
                                <span style={{ color: "#3b82f6" }}>💡 </span>
                                <span style={{ color: trendColor }}>{netTrend==="improving"?"Second half outperformed first — momentum building.":netTrend==="declining"?"Performance declined through the month — review fatigue or rule drift.":"Consistent performance across the month."}</span>
                                {pf!=null && isFinite(pf) && pf<1 && <span style={{ color: "#f59e0b" }}> Profit factor below 1 — exits need review.</span>}
                              </div>
                            </div>
                          );
                        })()}
                      </div>
                    )}
                  </div>

                  {/* RIGHT — Mood vs Performance */}
                  <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
                    <SectionHeader label="MOOD VS PERFORMANCE" skey="mood"
                      summary={<span style={{ color: "#64748b" }}>{moodEntries.length} mood{moodEntries.length !== 1 ? "s" : ""} tracked</span>} />
                    {!collapsed.mood && (
                      <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px", display: "flex", flexDirection: "column", gap: 10, flex: 1 }}>
                        {hasMoods ? moodEntries.map(([mood, s]) => {
                          const avg = s.pnl / s.sessions;
                          const wr = Math.round((s.wins / s.sessions) * 100);
                          const barW = Math.abs(avg) / maxAvg * 100;
                          const isPos = avg >= 0;
                          return (
                            <div key={mood}>
                              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
                                <span style={{ fontSize: 11, color: "#e2e8f0" }}>{mood}</span>
                                <div style={{ display: "flex", gap: 10, alignItems: "center", flexShrink: 0, marginLeft: 8 }}>
                                  <span style={{ fontSize: 9, color: "#94a3b8" }}>{s.sessions}× · {wr}% WR</span>
                                  <span style={{ fontSize: 11, fontWeight: 600, color: isPos ? "#4ade80" : "#f87171", minWidth: 62, textAlign: "right" }}>{avg >= 0 ? "+" : "-"}${Math.abs(avg).toFixed(0)} avg</span>
                                </div>
                              </div>
                              <div style={{ background: "#0a0e1a", borderRadius: 2, height: 4, overflow: "hidden" }}>
                                <div style={{ width: `${barW}%`, height: "100%", background: isPos ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} />
                              </div>
                            </div>
                          );
                        }) : <div style={{ fontSize: 11, color: "#475569" }}>No mood data logged.</div>}
                        {hasMoods && moodEntries.length >= 2 && (() => {
                          const best = moodEntries[0];
                          const worst = moodEntries[moodEntries.length - 1];
                          return (
                            <div style={{ marginTop: 4, padding: "8px 10px", background: "#0a1628", border: "1px solid #1e3a5f", borderRadius: 4, fontSize: 10, color: "#94a3b8", lineHeight: 1.7 }}>
                              <span style={{ color: "#3b82f6" }}>💡 </span>
                              Best mindset: <span style={{ color: "#4ade80" }}>{best[0]}</span> (+${(best[1].pnl / best[1].sessions).toFixed(0)} avg).
                              {(worst[1].pnl / worst[1].sessions) < 0 && <> Avoid trading in <span style={{ color: "#f87171" }}>{worst[0]}</span> state.</>}
                            </div>
                          );
                        })()}
                      </div>
                    )}
                  </div>

                </div>
              );
            })()}

            {/* Mistake Cost Accounting — full width row */}
            {monthEntries.length > 0 && (() => {
              const costByTag = {};
              for (const e of monthEntries) {
                const mc = e.mistakeCosts || {};
                for (const [tag, val] of Object.entries(mc)) {
                  if (val != null && val > 0 && tag && tag !== "No Mistakes — Executed the Plan ✓") {
                    costByTag[tag] = (costByTag[tag] || 0) + parseFloat(val);
                  }
                }
              }
              const sorted = Object.entries(costByTag).sort((a, b) => b[1] - a[1]);
              const totalCost = sorted.reduce((s, [, v]) => s + v, 0);
              if (sorted.length === 0 || totalCost === 0) return null;
              const maxCost = sorted[0][1] || 1;
              const topMistake = sorted[0][0];
              return (
                <div>
                  <SectionHeader label="MISTAKE COST ACCOUNTING" skey="mistakecost"
                    summary={<span style={{ color: "#fb923c" }}>💸 ${totalCost.toFixed(0)} attributed</span>} />
                  {!collapsed.mistakecost && (
                    <div style={{ background: "#0f1729", border: "1px solid #451a03", borderRadius: 4, padding: "14px 16px" }}>
                      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                        {sorted.map(([tag, cost]) => {
                          const pct = Math.round((cost / totalCost) * 100);
                          const barW = (cost / maxCost) * 100;
                          return (
                            <div key={tag}>
                              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
                                <span style={{ fontSize: 11, color: "#e2e8f0" }}>{tag}</span>
                                <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                                  <span style={{ fontSize: 9, color: "#7c3d12" }}>{pct}% of total</span>
                                  <span style={{ fontSize: 12, fontWeight: 600, color: "#fb923c" }}>${cost.toFixed(0)}</span>
                                </div>
                              </div>
                              <div style={{ background: "#0a0e1a", borderRadius: 2, height: 4, overflow: "hidden" }}>
                                <div style={{ width: `${barW}%`, height: "100%", background: "#fb923c", borderRadius: 2, opacity: 0.7 }} />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      <div style={{ marginTop: 12, paddingTop: 10, borderTop: "1px solid #451a03", fontSize: 10, color: "#7c3d12" }}>
                        #1 costliest mistake: <span style={{ color: "#fb923c", fontWeight: 600 }}>{topMistake}</span> — ${sorted[0][1].toFixed(0)} ({Math.round((sorted[0][1] / totalCost) * 100)}% of total)
                      </div>
                    </div>
                  )}
                </div>
              );
            })()}

                            {/* TIME OF DAY — session-based analysis (uses same getSession logic as analytics for consistency) */}
              {allMonthTrades.length > 0 && monthAnalytics?.bySession && (() => {
                // Re-use the bySession data already computed by calcAnalytics (with tzLock=true)
                // This is consistent with how the rest of the analytics work
                const SESSION_ORDER = [
                  "Asian Session (6PM–12AM)",
                  "London Session (12AM–9:30AM)",
                  "NY Open (9:30AM–12PM)",
                  "Afternoon Deadzone (12–3PM)",
                  "Power Hour (3–4PM)",
                  "After Hours (4–6PM)",
                ];
                const SESSION_META = {
                  "Asian Session (6PM–12AM)":       { short: "Asian",     time: "6PM – 12AM ET",   emoji: "🌙" },
                  "London Session (12AM–9:30AM)":   { short: "London",    time: "12AM – 9:30AM ET", emoji: "🇬🇧" },
                  "NY Open (9:30AM–12PM)":          { short: "NY Open",   time: "9:30AM – 12PM ET", emoji: "🔥" },
                  "Afternoon Deadzone (12–3PM)":    { short: "Midday",    time: "12PM – 3PM ET",    emoji: "😴" },
                  "Power Hour (3–4PM)":             { short: "Power Hr",  time: "3PM – 4PM ET",     emoji: "⚡" },
                  "After Hours (4–6PM)":            { short: "After Hrs", time: "4PM – 6PM ET",     emoji: "🌆" },
                };
                const sessData = SESSION_ORDER
                  .map(s => ({ name: s, ...(monthAnalytics.bySession[s] || { trades: 0, pnl: 0, wins: 0 }) }))
                  .filter(s => s.trades > 0);
                if (sessData.length === 0) return null;
                const maxAbs = Math.max(...sessData.map(s => Math.abs(s.pnl)), 1);
                const bestSess = sessData.reduce((a, b) => a.pnl > b.pnl ? a : b);
                const worstSess = sessData.reduce((a, b) => a.pnl < b.pnl ? a : b);
                const totalTrades = sessData.reduce((s, x) => s + x.trades, 0);
                return (
                  <div>
                    <SectionHeader label="TIME OF DAY ANALYSIS" skey="timeofday"
                      summary={<span style={{ color: "#64748b" }}>{sessData.length} session{sessData.length !== 1 ? "s" : ""} · {totalTrades} trades</span>} />
                    {!collapsed.timeofday && (
                      <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px" }}>
                        {/* Session rows with bar chart */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 11, marginBottom: 14 }}>
                          {sessData.map(s => {
                            const meta = SESSION_META[s.name] || { short: s.name, time: "", emoji: "📊" };
                            const wr = s.trades ? Math.round(s.wins / s.trades * 100) : 0;
                            const barW = Math.abs(s.pnl) / maxAbs * 100;
                            const isPos = s.pnl >= 0;
                            const isBest = s.name === bestSess.name;
                            const isWorst = s.pnl < 0 && sessData.length > 1 && s.name === worstSess.name;
                            const sharePct = Math.round(s.trades / totalTrades * 100);
                            return (
                              <div key={s.name}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 5 }}>
                                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                    <span style={{ fontSize: 12 }}>{meta.emoji}</span>
                                    <span style={{ fontSize: 11, color: isBest ? "#4ade80" : isWorst ? "#f87171" : "#e2e8f0", fontWeight: isBest || isWorst ? 600 : 400 }}>{meta.short}</span>
                                    <span style={{ fontSize: 9, color: "#94a3b8" }}>{meta.time}</span>
                                    {isBest && <span style={{ fontSize: 8, color: "#4ade80", background: "rgba(16,63,33,0.5)", border: "1px solid #166534", padding: "1px 5px", borderRadius: 2, letterSpacing: "0.05em" }}>BEST</span>}
                                    {isWorst && <span style={{ fontSize: 8, color: "#f87171", background: "rgba(63,16,16,0.5)", border: "1px solid #7f1d1d", padding: "1px 5px", borderRadius: 2, letterSpacing: "0.05em" }}>AVOID</span>}
                                  </div>
                                  <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                                    <span style={{ fontSize: 9, color: "#94a3b8" }}>{s.trades}T ({sharePct}%)</span>
                                    <span style={{ fontSize: 10, color: wr >= 50 ? "#4ade80" : "#f87171", minWidth: 42, textAlign: "right" }}>{wr}% WR</span>
                                    <span style={{ fontSize: 12, fontWeight: 600, color: isPos ? "#4ade80" : "#f87171", minWidth: 78, textAlign: "right" }}>{isPos ? "+" : "-"}${Math.abs(s.pnl).toFixed(0)}</span>
                                  </div>
                                </div>
                                {/* Bar */}
                                <div style={{ background: "#0a0e1a", borderRadius: 2, height: 6, overflow: "hidden" }}>
                                  <div style={{ width: `${barW}%`, height: "100%", background: isPos ? "#4ade80" : "#f87171", borderRadius: 2, opacity: isBest || isWorst ? 0.9 : 0.55, transition: "width .3s" }} />
                                </div>
                                {/* Mini per-trade avg beneath */}
                                <div style={{ marginTop: 3, fontSize: 9, color: "#64748b" }}>
                                  avg/trade: <span style={{ color: isPos ? "#4ade80" : "#f87171" }}>{isPos ? "+" : "-"}${Math.abs(s.pnl / s.trades).toFixed(0)}</span>
                                  {" · "}wins: <span style={{ color: "#4ade80" }}>{s.wins}</span> · losses: <span style={{ color: "#f87171" }}>{s.trades - s.wins}</span>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                        {/* Insight bar */}
                        <div style={{ fontSize: 10, color: "#64748b", lineHeight: 1.7, background: "#0a0e1a", borderRadius: 4, padding: "8px 12px" }}>
                          <span style={{ color: "#3b82f6" }}>💡 </span>
                          Best: <span style={{ color: "#4ade80" }}>{SESSION_META[bestSess.name]?.short || bestSess.name}</span> — {bestSess.trades} trades · {Math.round(bestSess.wins / bestSess.trades * 100)}% WR · <span style={{ color: "#4ade80" }}>+${bestSess.pnl.toFixed(0)}</span>.
                          {worstSess.pnl < 0 && worstSess.name !== bestSess.name && <> Weakest: <span style={{ color: "#f87171" }}>{SESSION_META[worstSess.name]?.short || worstSess.name}</span> (<span style={{ color: "#f87171" }}>-${Math.abs(worstSess.pnl).toFixed(0)}</span>).</>}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}



          </div>
        </div>
      )}


    </div>
  );
}

function WeeklyPerformance({ entries, netPnl: calcNetPnlProp, fmtPnl, pnlColor, calcAnalytics, ai, activeJournal }) {
  const calcNetPnl = calcNetPnlProp;
  const [selectedYear, setSelectedYear] = useState(() => new Date().getFullYear());
  const [selectedWeek, setSelectedWeek] = useState(null);

  // ── Unified recap storage — same key as AIRecapView, same window.storage ──
  // Schema: { texts: { [periodKey]: string }, meta: { [periodKey]: ISO }, hashes: { [periodKey]: string } }
  const journalId  = activeJournal?.id || 'default';
  const RECAP_KEY  = `ai-recaps-v2-${journalId}`;

  const [weeklyRecaps, setWeeklyRecaps] = useState({});  // { [weekKey]: { text, generatedAt (ISO), notesHash } }
  const [recapStoreReady, setRecapStoreReady] = useState(false);

  // Load on mount (and whenever journal switches)
  useEffect(() => {
    setRecapStoreReady(false);
    setWeeklyRecaps({});
    (async () => {
      try {
        const r = await storage.get(RECAP_KEY);
        if (r?.value) {
          const parsed = JSON.parse(r.value);
          // Rebuild weeklyRecaps from the shared store's texts + meta + hashes fields
          const texts   = parsed.texts  || {};
          const meta    = parsed.meta   || {};
          const hashes  = parsed.hashes || {};
          const rebuilt = {};
          for (const k of Object.keys(texts)) {
            rebuilt[k] = { text: texts[k], generatedAt: meta[k] || null, notesHash: hashes[k] || null };
          }
          setWeeklyRecaps(rebuilt);
        }
      } catch {}
      setRecapStoreReady(true);
    })();
  }, [journalId]);

  // Persist to the shared store whenever weeklyRecaps changes (after initial load)
  useEffect(() => {
    if (!recapStoreReady) return;
    if (Object.keys(weeklyRecaps).length === 0) return;
    // Merge into shared schema: texts + meta + hashes (monthly recaps already stored stay intact)
    (async () => {
      try {
        const r = await storage.get(RECAP_KEY);
        const existing = r?.value ? JSON.parse(r.value) : {};
        const existingTexts  = existing.texts  || {};
        const existingMeta   = existing.meta   || {};
        const existingHashes = existing.hashes || {};
        // Write only keys that belong to this component (week keys start with YYYY-W)
        for (const [k, v] of Object.entries(weeklyRecaps)) {
          existingTexts[k]  = v.text;
          existingMeta[k]   = v.generatedAt;
          existingHashes[k] = v.notesHash;
        }
        await storage.set(RECAP_KEY, JSON.stringify({ texts: existingTexts, meta: existingMeta, hashes: existingHashes }));
      } catch {}
    })();
  }, [weeklyRecaps, recapStoreReady]);

  const saveWeeklyRecap = (weekKey, text, hash) => {
    const now = new Date().toISOString();
    setWeeklyRecaps(prev => ({ ...prev, [weekKey]: { text, generatedAt: now, notesHash: hash } }));
  };

  const deleteWeeklyRecap = async (weekKey) => {
    setWeeklyRecaps(prev => { const n = {...prev}; delete n[weekKey]; return n; });
    // Also remove from shared store
    try {
      const r = await storage.get(RECAP_KEY);
      if (r?.value) {
        const parsed = JSON.parse(r.value);
        delete (parsed.texts  || {})[weekKey];
        delete (parsed.meta   || {})[weekKey];
        delete (parsed.hashes || {})[weekKey];
        await storage.set(RECAP_KEY, JSON.stringify(parsed));
      }
    } catch {}
  };
  const [recapLoading, setRecapLoading] = useState(false);
  const [recapError, setRecapError] = useState('');

  // notesHash moved to module scope as calcNotesHash — shared with AIRecapView
  const [collapsedNotes, setCollapsedNotes] = useState({}); // key: fieldKey, val: bool
  const [dayByDayCollapsed, setDayByDayCollapsed] = useState({}); // key: entry.id
  const [compiledCollapsed, setCompiledCollapsed] = useState(true); // whole compiled section collapsed by default
  const toggleNote = (key) => setCollapsedNotes(p => ({ ...p, [key]: !p[key] }));
  const toggleDay = (id) => setDayByDayCollapsed(p => ({ ...p, [id]: !p[id] }));

  const getISOWeek = (dateStr) => {
    const d = new Date(dateStr + "T12:00:00");
    const jan4 = new Date(d.getFullYear(), 0, 4);
    const startOfWeek1 = new Date(jan4);
    startOfWeek1.setDate(jan4.getDate() - ((jan4.getDay() + 6) % 7));
    const diff = d - startOfWeek1;
    const week = Math.floor(diff / (7 * 24 * 60 * 60 * 1000)) + 1;
    return `${d.getFullYear()}-W${String(week).padStart(2, "0")}`;
  };

  const getWeekRange = (weekKey) => {
    const [yr, wStr] = weekKey.split("-W");
    const year = parseInt(yr), week = parseInt(wStr);
    const jan4 = new Date(year, 0, 4);
    const startOfWeek1 = new Date(jan4);
    startOfWeek1.setDate(jan4.getDate() - ((jan4.getDay() + 6) % 7));
    const monday = new Date(startOfWeek1);
    monday.setDate(startOfWeek1.getDate() + (week - 1) * 7);
    const friday = new Date(monday); friday.setDate(monday.getDate() + 4);
    const fmt = d => d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    return { label: `${fmt(monday)} – ${fmt(friday)}, ${year}`, monday, friday };
  };

  const years = [...new Set(entries.map(e => e.date?.slice(0, 4)).filter(Boolean))].map(Number).sort((a, b) => b - a);
  if (!years.includes(new Date().getFullYear())) years.unshift(new Date().getFullYear());

  const yearEntries = entries.filter(e => e.date?.startsWith(String(selectedYear)));
  const byWeek = {};
  for (const e of yearEntries) {
    const wk = getISOWeek(e.date);
    if (!byWeek[wk]) byWeek[wk] = [];
    byWeek[wk].push(e);
  }
  const weeks = Object.keys(byWeek).sort();

  const SESSION_META = {
    "Asian Session (6PM\u201312AM)":       { short: "Asian",     emoji: "\u{1F319}" },
    "London Session (12AM\u20139:30AM)":   { short: "London",    emoji: "\u{1F1EC}\u{1F1E7}" },
    "NY Open (9:30AM\u201312PM)":          { short: "NY Open",   emoji: "\u{1F525}" },
    "Afternoon Deadzone (12\u20133PM)":    { short: "Midday",    emoji: "\u{1F634}" },
    "Power Hour (3\u20134PM)":             { short: "Power Hr",  emoji: "\u26A1" },
    "After Hours (4\u20136PM)":            { short: "After Hrs", emoji: "\u{1F306}" },
  };

  // ── WEEK DETAIL VIEW ──────────────────────────────────────────────────────
  if (selectedWeek && byWeek[selectedWeek]) {
    const wEntries = (byWeek[selectedWeek] || []).slice().sort((a, b) => a.date.localeCompare(b.date));
    const { label: weekLabel } = getWeekRange(selectedWeek);
    const weekNum = selectedWeek.split("-W")[1];
    const wNet   = wEntries.reduce((s, e) => s + calcNetPnl(e), 0);
    const wGross = wEntries.reduce((s, e) => s + (parseFloat(e.pnl) || 0), 0);
    const wFees  = wEntries.reduce((s, e) => s + (parseFloat(e.commissions) || 0), 0);
    const wWinDays = wEntries.filter(e => calcNetPnl(e) > 0).length;
    const wLossDays = wEntries.filter(e => calcNetPnl(e) < 0).length;
    const wTrades = wEntries.flatMap(e => e.parsedTrades || []);
    const wA = calcAnalytics(wTrades, true);
    const maxDayAbs = Math.max(...wEntries.map(e => Math.abs(calcNetPnl(e))), 1);

    // Mistake frequency
    const mistakeCounts = {};
    let cleanDays = 0;
    for (const e of wEntries) {
      if (e.sessionMistakes?.includes("No Mistakes — Executed the Plan ✓")) cleanDays++;
      for (const m of (e.sessionMistakes || [])) {
        if (m === "No Mistakes — Executed the Plan ✓") continue;
        mistakeCounts[m] = (mistakeCounts[m] || 0) + 1;
      }
    }
    const mistakeSorted = Object.entries(mistakeCounts).sort((a, b) => b[1] - a[1]);
    const maxMistake = mistakeSorted[0]?.[1] || 1;

    // Mood
    const moodStats = {};
    for (const e of wEntries) {
      const moods = e.moods?.length ? e.moods : e.mood ? [e.mood] : [];
      const ep = calcNetPnl(e);
      for (const m of moods) {
        if (!moodStats[m]) moodStats[m] = { pnl: 0, sessions: 0, wins: 0 };
        moodStats[m].pnl += ep; moodStats[m].sessions++; if (ep > 0) moodStats[m].wins++;
      }
    }
    const moodSorted = Object.entries(moodStats).sort((a, b) => (b[1].pnl / b[1].sessions) - (a[1].pnl / a[1].sessions));
    const maxMoodAvg = Math.max(...moodSorted.map(([, s]) => Math.abs(s.pnl / s.sessions)), 1);

    // Best/worst trade
    const allWeekTrades = wTrades;
    const bestTrade  = allWeekTrades.length ? allWeekTrades.reduce((a, b) => (a.pnl||0) > (b.pnl||0) ? a : b) : null;
    const worstTrade = allWeekTrades.length ? allWeekTrades.reduce((a, b) => (a.pnl||0) < (b.pnl||0) ? a : b) : null;

    const noteFields = [
      { key: "lessonsLearned", label: "LESSONS LEARNED",    icon: "📚", color: "#93c5fd" },
      { key: "mistakes",       label: "MISTAKES TO AVOID",  icon: "⚠️", color: "#f87171" },
      { key: "improvements",   label: "IMPROVEMENTS",       icon: "📈", color: "#4ade80" },
      { key: "rules",          label: "RULES",              icon: "📋", color: "#a78bfa" },
      { key: "reinforceRule",  label: "RULES TO REINFORCE", icon: "🔒", color: "#fbbf24" },
      { key: "tomorrow",       label: "PLANS WRITTEN",      icon: "🎯", color: "#38bdf8" },
      { key: "marketNotes",    label: "MARKET NOTES",       icon: "📊", color: "#64748b" },
    ];
    const getNote = (e, key) => (e[key] || "").trim();
    const fmtSecs = (s) => !s ? "—" : s < 60 ? `${Math.round(s)}s` : s < 3600 ? `${Math.floor(s/60)}m ${Math.round(s%60)}s` : `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`;

    return (
      <div>
        {/* Back */}
        <button onClick={() => setSelectedWeek(null)}
          style={{ background: "transparent", border: "1px solid rgba(129,140,248,0.3)", color: "#818cf8", padding: "7px 16px", borderRadius: 5, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.08em", marginBottom: 20 }}
          onMouseEnter={e => e.currentTarget.style.borderColor="#818cf8"}
          onMouseLeave={e => e.currentTarget.style.borderColor="rgba(129,140,248,0.3)"}>
          ← BACK TO WEEKS
        </button>

        {/* Header */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ height: 2, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc,transparent)", borderRadius: 1, marginBottom: 14 }} />
          <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
            <div>
              <div style={{ fontSize: 9, color: "#818cf8", letterSpacing: "0.28em", marginBottom: 4 }}>WEEKLY REVIEW</div>
              <div style={{ fontFamily: "\'Bebas Neue\',sans-serif", fontSize: 38, letterSpacing: "0.1em", lineHeight: 1, background: "linear-gradient(135deg,#38bdf8 0%,#818cf8 55%,#c084fc 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>WEEK {weekNum}</div>
              <div style={{ fontFamily: "\'DM Mono\',monospace", fontSize: 13, color: "#64748b", letterSpacing: "0.15em", marginTop: 4 }}>{weekLabel}</div>
            </div>
            <div style={{ position: "relative", padding: 1, borderRadius: 7, background: wNet >= 0 ? "linear-gradient(135deg,#38bdf8,#4ade80)" : "linear-gradient(135deg,#f87171,#818cf8)", flexShrink: 0 }}>
              <div style={{ background: "#070d1a", borderRadius: 6, padding: "12px 20px", textAlign: "right", minWidth: 140 }}>
                <div style={{ fontSize: 9, color: "#818cf8", letterSpacing: "0.15em", marginBottom: 4 }}>WEEK NET P&L</div>
                <div style={{ fontSize: 34, color: pnlColor(wNet), fontWeight: 700, lineHeight: 1 }}>{fmtPnl(wNet)}</div>
                {wFees > 0 && <div style={{ fontSize: 9, color: "#475569", marginTop: 4 }}>gross {fmtPnl(wGross)} · fees -${wFees.toFixed(0)}</div>}
              </div>
            </div>
          </div>
        </div>

        {/* Key Stats */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))", gap: 8, marginBottom: 20 }}>
          {[
            { l: "TRADING DAYS",   v: `${wWinDays}W / ${wLossDays}L`, c: "#e2e8f0" },
            { l: "TOTAL TRADES",   v: wTrades.length, c: "#e2e8f0" },
            { l: "TRADE WIN RATE", v: wA ? `${wA.winRate.toFixed(0)}%` : "—", c: wA && wA.winRate >= 50 ? "#4ade80" : "#f87171" },
            { l: "PROFIT FACTOR",  v: wA ? fmtPF(wA.profitFactor) : "—", c: wA ? pfColor(wA.profitFactor) : "#64748b" },
            { l: "AVG WIN",  v: wA && wA.avgWin  ? `+$${wA.avgWin.toFixed(0)}`  : "—", c: "#4ade80" },
            { l: "AVG LOSS", v: wA && wA.avgLoss ? `-$${Math.abs(wA.avgLoss).toFixed(0)}` : "—", c: "#f87171" },
          ].map(({ l, v, c }) => (
            <div key={l} style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 5, padding: "10px 12px" }}>
              <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 4 }}>{l}</div>
              <div style={{ fontSize: 14, color: c, fontWeight: 700 }}>{v}</div>
            </div>
          ))}
        </div>

        {/* Daily P&L Bar Chart */}
        <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 5, padding: "14px 16px", marginBottom: 20 }}>
          <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 14 }}>DAILY P&L BREAKDOWN</div>
          <div style={{ display: "flex", gap: 8, alignItems: "flex-end", height: 80 }}>
            {wEntries.map(e => {
              const dn = calcNetPnl(e);
              const barH = Math.max(4, Math.abs(dn) / maxDayAbs * 72);
              const dayLabel = new Date(e.date + "T12:00:00").toLocaleDateString("en-US", { weekday: "short" });
              return (
                <div key={e.id} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-end", gap: 4, height: "100%" }}>
                  <div style={{ fontSize: 9, color: pnlColor(dn), fontWeight: 600 }}>{fmtPnl(dn)}</div>
                  <div style={{ width: "100%", height: barH, background: dn >= 0 ? "rgba(74,222,128,0.7)" : "rgba(248,113,113,0.7)", borderRadius: 2, border: `1px solid ${dn >= 0 ? "#166534" : "#7f1d1d"}` }} />
                  <div style={{ fontSize: 9, color: "#64748b" }}>{dayLabel}</div>
                  {e.grade && <div style={{ fontSize: 8, color: "#475569" }}>{e.grade}</div>}
                </div>
              );
            })}
          </div>
        </div>

        {/* Session + Holding Time */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
          {wA?.bySession && Object.values(wA.bySession).some(s => s.trades > 0) && (() => {
            const sessOrder = ["Asian Session (6PM\u201312AM)","London Session (12AM\u20139:30AM)","NY Open (9:30AM\u201312PM)","Afternoon Deadzone (12\u20133PM)","Power Hour (3\u20134PM)","After Hours (4\u20136PM)"];
            const sessData = sessOrder.map(s => ({ name: s, ...(wA.bySession[s] || { trades: 0, pnl: 0, wins: 0 }) })).filter(s => s.trades > 0);
            const maxAbs = Math.max(...sessData.map(s => Math.abs(s.pnl)), 1);
            const best = sessData.reduce((a, b) => a.pnl > b.pnl ? a : b);
            return (
              <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 5, padding: "14px 16px" }}>
                <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 12 }}>TIME OF DAY</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {sessData.map(s => {
                    const meta = SESSION_META[s.name] || { short: s.name, emoji: "📊" };
                    const wr = Math.round(s.wins / s.trades * 100);
                    const barW = Math.abs(s.pnl) / maxAbs * 100;
                    const isBest = s.name === best.name && s.pnl > 0;
                    return (
                      <div key={s.name}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                          <span style={{ fontSize: 11, color: "#e2e8f0" }}>{meta.emoji} {meta.short}{isBest && <span style={{ marginLeft: 6, fontSize: 8, color: "#4ade80", background: "rgba(16,63,33,0.6)", padding: "1px 5px", borderRadius: 2 }}>BEST</span>}</span>
                          <div style={{ display: "flex", gap: 10 }}>
                            <span style={{ fontSize: 10, color: wr >= 50 ? "#4ade80" : "#f87171" }}>{wr}% WR</span>
                            <span style={{ fontSize: 11, fontWeight: 600, color: s.pnl >= 0 ? "#4ade80" : "#f87171" }}>{s.pnl >= 0 ? "+" : "-"}${Math.abs(s.pnl).toFixed(0)}</span>
                          </div>
                        </div>
                        <div style={{ background: "#0a0e1a", borderRadius: 2, height: 4 }}>
                          <div style={{ width: `${barW}%`, height: "100%", background: s.pnl >= 0 ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })()}
          {wA?.byDuration && Object.values(wA.byDuration).some(b => b.trades > 0) && (
            <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 5, padding: "14px 16px" }}>
              <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 12 }}>HOLDING TIME</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {(wA.DURATION_BUCKETS || []).map(bucket => {
                  const b = wA.byDuration[bucket.key];
                  if (!b || b.trades === 0) return null;
                  const wr = Math.round(b.wins / b.trades * 100);
                  const maxPnl = Math.max(...(wA.DURATION_BUCKETS || []).map(bk => Math.abs(wA.byDuration[bk.key]?.pnl || 0)), 1);
                  const barW = Math.abs(b.pnl) / maxPnl * 100;
                  const avg = b.pnl / b.trades;
                  return (
                    <div key={bucket.key}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                        <span style={{ fontSize: 11, color: "#e2e8f0" }}>{bucket.label}<span style={{ fontSize: 9, color: "#64748b", marginLeft: 6 }}>avg {fmtSecs(b.avgSecs)}</span></span>
                        <div style={{ display: "flex", gap: 10 }}>
                          <span style={{ fontSize: 10, color: wr >= 50 ? "#4ade80" : "#f87171" }}>{wr}% WR</span>
                          <span style={{ fontSize: 11, fontWeight: 600, color: avg >= 0 ? "#4ade80" : "#f87171" }}>{avg >= 0 ? "+" : "-"}${Math.abs(avg).toFixed(0)}/t</span>
                        </div>
                      </div>
                      <div style={{ background: "#0a0e1a", borderRadius: 2, height: 4 }}>
                        <div style={{ width: `${barW}%`, height: "100%", background: b.pnl >= 0 ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Behavioral Edge */}
        {wA && (wA.afterLoss?.total > 0 || wA.afterWin?.total > 0 || wA.first3?.total > 0) && (
          <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 5, padding: "14px 16px", marginBottom: 20 }}>
            <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 12 }}>BEHAVIORAL EDGE CHECKS</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
              {[{label:"AFTER A LOSS",data:wA.afterLoss},{label:"AFTER A WIN",data:wA.afterWin},{label:"FIRST 3 TRADES",data:wA.first3},{label:"REST OF SESSION",data:wA.rest}].map(card => (
                <div key={card.label} style={{ background: "#0a0e1a", border: "1px solid #1e293b", borderRadius: 4, padding: "10px 12px" }}>
                  <div style={{ fontSize: 8, color: "#94a3b8", letterSpacing: "0.08em", marginBottom: 6 }}>{card.label}</div>
                  {card.data?.total ? (
                    <>
                      <div style={{ fontSize: 14, color: "#e2e8f0", fontWeight: 700 }}>{card.data.winRate.toFixed(0)}% WR</div>
                      <div style={{ fontSize: 11, color: card.data.avgPnl >= 0 ? "#4ade80" : "#f87171", marginTop: 2 }}>{card.data.avgPnl >= 0 ? "+" : "-"}${Math.abs(card.data.avgPnl).toFixed(0)}/trade</div>
                      <div style={{ fontSize: 9, color: "#475569", marginTop: 4 }}>n={card.data.total}</div>
                    </>
                  ) : <div style={{ fontSize: 10, color: "#475569" }}>No data</div>}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Mistake + Mood side by side */}
        {(mistakeSorted.length > 0 || cleanDays > 0 || moodSorted.length > 0) && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20, alignItems: "stretch" }}>
            <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 5, padding: "14px 16px", display: "flex", flexDirection: "column" }}>
              <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 12 }}>MISTAKE FREQUENCY</div>
              {cleanDays > 0 && <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 8px", background: "rgba(16,63,33,0.55)", border: "1px solid #166534", borderRadius: 3, marginBottom: mistakeSorted.length ? 10 : 0 }}><span style={{ fontSize: 10, color: "#4ade80" }}>✓ Clean days</span><span style={{ fontSize: 10, fontWeight: 600, color: "#4ade80" }}>{cleanDays}</span></div>}
              {mistakeSorted.length === 0 && cleanDays === 0 && <div style={{ fontSize: 11, color: "#475569" }}>No mistakes logged.</div>}
              {mistakeSorted.map(([mistake, count]) => (
                <div key={mistake} style={{ marginBottom: 8 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}><span style={{ fontSize: 10, color: "#e2e8f0" }}>{mistake}</span><span style={{ fontSize: 10, fontWeight: 600, color: "#f87171" }}>{count}×</span></div>
                  <div style={{ background: "#0a0e1a", borderRadius: 2, height: 3 }}><div style={{ width: `${(count / maxMistake) * 100}%`, height: "100%", background: "#f87171", borderRadius: 2, opacity: 0.7 }} /></div>
                </div>
              ))}
            </div>
            <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 5, padding: "14px 16px", display: "flex", flexDirection: "column" }}>
              <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 12 }}>MOOD VS PERFORMANCE</div>
              {moodSorted.length === 0 && <div style={{ fontSize: 11, color: "#475569" }}>No mood data logged.</div>}
              {moodSorted.map(([mood, s]) => {
                const avg = s.pnl / s.sessions; const isPos = avg >= 0; const wr = Math.round((s.wins/s.sessions)*100);
                return (
                  <div key={mood} style={{ marginBottom: 8 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                      <span style={{ fontSize: 10, color: "#e2e8f0" }}>{mood}</span>
                      <div style={{ display: "flex", gap: 8 }}><span style={{ fontSize: 9, color: "#94a3b8" }}>{wr}% WR</span><span style={{ fontSize: 10, fontWeight: 600, color: isPos ? "#4ade80" : "#f87171" }}>{avg >= 0 ? "+" : "-"}${Math.abs(avg).toFixed(0)}</span></div>
                    </div>
                    <div style={{ background: "#0a0e1a", borderRadius: 2, height: 3 }}><div style={{ width: `${Math.abs(avg) / maxMoodAvg * 100}%`, height: "100%", background: isPos ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} /></div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Best + Worst Trade */}
        {(bestTrade || worstTrade) && bestTrade?.pnl !== worstTrade?.pnl && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
            {bestTrade && bestTrade.pnl > 0 && (
              <div style={{ background: "rgba(16,63,33,0.4)", border: "1px solid #166534", borderRadius: 5, padding: "14px 16px" }}>
                <div style={{ fontSize: 10, color: "#4ade80", letterSpacing: "0.1em", marginBottom: 8 }}>✅ BEST TRADE OF THE WEEK</div>
                <div style={{ display: "flex", gap: 12, alignItems: "baseline", marginBottom: 4 }}>
                  <span style={{ fontSize: 22, color: "#4ade80", fontWeight: 700 }}>+${bestTrade.pnl.toFixed(0)}</span>
                  <span style={{ fontSize: 11, color: "#94a3b8" }}>{bestTrade.symbol} · {(bestTrade.direction||"").toUpperCase()} · {bestTrade.qty}ct</span>
                </div>
                {wEntries.find(e => e.bestTrade) && <div style={{ fontSize: 11, color: "#4ade80", fontStyle: "italic", lineHeight: 1.6 }}>{wEntries.find(e => e.bestTrade)?.bestTrade}</div>}
              </div>
            )}
            {worstTrade && worstTrade.pnl < 0 && (
              <div style={{ background: "rgba(63,16,16,0.4)", border: "1px solid #7f1d1d", borderRadius: 5, padding: "14px 16px" }}>
                <div style={{ fontSize: 10, color: "#f87171", letterSpacing: "0.1em", marginBottom: 8 }}>❌ WORST TRADE OF THE WEEK</div>
                <div style={{ display: "flex", gap: 12, alignItems: "baseline", marginBottom: 4 }}>
                  <span style={{ fontSize: 22, color: "#f87171", fontWeight: 700 }}>-${Math.abs(worstTrade.pnl).toFixed(0)}</span>
                  <span style={{ fontSize: 11, color: "#94a3b8" }}>{worstTrade.symbol} · {(worstTrade.direction||"").toUpperCase()} · {worstTrade.qty}ct</span>
                </div>
                {wEntries.find(e => e.worstTrade) && <div style={{ fontSize: 11, color: "#f87171", fontStyle: "italic", lineHeight: 1.6 }}>{wEntries.find(e => e.worstTrade)?.worstTrade}</div>}
              </div>
            )}
          </div>
        )}

        {/* ── AI RECAP OF THE WEEK ── */}
        {(() => {
          const currentHash = calcNotesHash(wEntries);
          const saved = weeklyRecaps[selectedWeek];
          const isStale = saved && saved.notesHash !== currentHash;
          const generatedDate = saved ? new Date(saved.generatedAt).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }) : null;

          const generateRecap = async () => {
            if (!ai?.enabled || !ai?.apiKey) {
              setRecapError('AI not configured. Add your API key in Settings (⚙).');
              return;
            }
            setRecapLoading(true);
            setRecapError('');
            try {
              // Build prompt from week entries — reuse same structure as AIRecapView
              const sorted = wEntries;
              const totalPnl = sorted.reduce((s,e) => s + calcNetPnl(e), 0);
              const winDays  = sorted.filter(e => calcNetPnl(e) > 0).length;
              const lossDays = sorted.filter(e => calcNetPnl(e) < 0).length;
              const allT = sorted.flatMap(e => e.parsedTrades || []);
              const tN2  = (t) => t.pnl - (t.commission||0);
              const allW = allT.filter(t => tN2(t) > 0);
              const allL = allT.filter(t => tN2(t) < 0);
              const wr   = allT.length ? ((allW.length/allT.length)*100).toFixed(1) : "0";
              const pfv  = allL.length ? allW.reduce((s,t)=>s+tN2(t),0)/Math.abs(allL.reduce((s,t)=>s+tN2(t),0)) : allW.length > 0 ? Infinity : null;
              const fees = allT.reduce((s,t)=>s+(t.commission||0),0);
              const avgW = allW.length ? (allW.reduce((s,t)=>s+tN2(t),0)/allW.length).toFixed(0) : "0";
              const avgL = allL.length ? Math.abs(allL.reduce((s,t)=>s+tN2(t),0)/allL.length).toFixed(0) : "0";
              const mCounts = {};
              let cDays = 0;
              for (const e of sorted) {
                if (e.sessionMistakes?.includes("No Mistakes — Executed the Plan ✓")) cDays++;
                for (const m of (e.sessionMistakes||[])) { if (m !== "No Mistakes — Executed the Plan ✓") mCounts[m]=(mCounts[m]||0)+1; }
              }
              const mistakeLine = Object.entries(mCounts).sort((a,b)=>b[1]-a[1]).map(([m,n])=>`"${m}":${n}x`).join(", ")||"none";
              const grades = sorted.filter(e=>e.grade).map(e=>`${e.date.slice(5)}:${e.grade}`).join(", ")||"none";
              const planLines = [];
              for (let i=1;i<sorted.length;i++) {
                const prev=sorted[i-1], curr=sorted[i];
                if (prev.tomorrow?.trim()) {
                  const currClean = curr.sessionMistakes?.includes("No Mistakes — Executed the Plan ✓");
                  const nextNotes = [curr.marketNotes, curr.lessonsLearned, curr.mistakes, curr.rules, curr.bestTrade, curr.worstTrade].filter(Boolean).map(n=>n.trim()).join(' | ').slice(0,300);
                  const tags = (curr.sessionMistakes||[]).filter(m=>m!=="No Mistakes — Executed the Plan ✓").join(", ")||"none";
                  planLines.push(
                    `  ${prev.date} plan: "${prev.tomorrow.slice(0,200)}"\n` +
                    `  ${curr.date} outcome: grade ${curr.grade||"?"} | net $${calcNetPnl(curr).toFixed(0)}${currClean?" | CLEAN EXECUTION DAY":""}\n` +
                    `  ${curr.date} notes: ${nextNotes||"(none written)"}\n` +
                    `  ${curr.date} tagged: ${tags}`
                  );
                }
              }
              // Direction breakdown
              const wByDir = { long:{t:0,w:0,pnl:0,comm:0}, short:{t:0,w:0,pnl:0,comm:0} };
              for (const t of allT) {
                const d = wByDir[t.direction==="short"?"short":"long"];
                const nt = tN2(t);
                d.t++; d.pnl+=nt; d.comm+=(t.commission||0); if(nt>0)d.w++;
              }
              const wDirLine = ["long","short"].filter(k=>wByDir[k].t>0).map(k=>{
                const d=wByDir[k]; const wr2=d.t?Math.round(d.w/d.t*100):0;
                const sign=d.pnl>=0?"+":"";
                return `${k.toUpperCase()}: ${d.t}t ${wr2}%WR net ${sign}$${d.pnl.toFixed(0)}`;
              }).join(" | ")||"none";
              // Session breakdown with ET buckets
              const wSessMap={};
              for (const t of allT) {
                const ts = t.direction==="short"?(t.buyTime||t.sellTime):(t.sellTime||t.buyTime);
                const m2 = ts?.match(/^\d{8}\s(\d{2})(\d{2})/);
                const h = m2 ? parseInt(m2[1])+parseInt(m2[2])/60 : -1;
                const k = h>=20?"Asian":h<4?"London Overnight":h<9.5?"Pre-Market":h<12?"NY Morning":h<16?"NY Afternoon":h>=16?"After Hours":"Unknown";
                if(!wSessMap[k])wSessMap[k]={pnl:0,t:0,w:0,comm:0};
                const nt = tN2(t);
                wSessMap[k].pnl+=nt;wSessMap[k].t++;wSessMap[k].comm+=(t.commission||0);if(nt>0)wSessMap[k].w++;
              }
              const sessOrderW=["Asian","London Overnight","Pre-Market","NY Morning","NY Afternoon","After Hours","Unknown"];
              const wSessLine=sessOrderW.filter(k=>wSessMap[k]).map(k=>{
                const d=wSessMap[k]; const wr2=Math.round(d.w/d.t*100);
                const sign=d.pnl>=0?"+":"";
                return `${k}: ${d.t}t ${wr2}%WR net ${sign}$${d.pnl.toFixed(0)}`;
              }).join(" | ")||"none";
              const wSessEntries=Object.entries(wSessMap).filter(([,d])=>d.t>0);
              const wBestSess=wSessEntries.length?wSessEntries.reduce((a,b)=>(b[1].pnl-b[1].comm)>(a[1].pnl-a[1].comm)?b:a):null;
              const wWorstSess=wSessEntries.length?wSessEntries.reduce((a,b)=>(b[1].pnl-b[1].comm)<(a[1].pnl-a[1].comm)?b:a):null;
              const grossPnlW = allT.reduce((s,t)=>s+t.pnl,0);
              const commDragW = Math.abs(grossPnlW)>0?(fees/Math.abs(grossPnlW)*100).toFixed(1):"0";
              const carryW = allT.filter(t=>t.notes==="overnight-carry").length;

              const dayBlocks = sorted.map(e => {
                const t = e.parsedTrades || [];
                const dGross = t.reduce((s,tr)=>s+tr.pnl,0);
                const dComm  = t.reduce((s,tr)=>s+(tr.commission||0),0);
                const tradeLines = t.map((tr,i)=>{
                  const net=(tr.pnl-(tr.commission||0)).toFixed(2);
                  const exitT=tr.direction==="short"?(tr.buyTime||tr.sellTime):(tr.sellTime||tr.buyTime);
                  const hhmm=exitT?exitT.replace(/^\d{8}\s/,"").slice(0,4):"?";
                  const carry=tr.notes==="overnight-carry"?" [carry]":"";
                  return `    T${i+1} ${(tr.direction||"long").toUpperCase()} @${hhmm} gross $${tr.pnl.toFixed(2)} net $${net}${carry}`;
                }).join("\n");
                const lines = [`[${e.date}] Net:$${calcNetPnl(e).toFixed(0)} Gross:$${dGross.toFixed(0)} Fees:-$${dComm.toFixed(0)} | ${t.filter(x=>x.pnl>0).length}W/${t.filter(x=>x.pnl<0).length}L | Grade:${e.grade||"?"} | Mood:${(e.moods?.length?e.moods:e.mood?[e.mood]:[]).join(",")||"?"}`];
                if (tradeLines) lines.push(tradeLines);
                if (e.sessionMistakes?.filter(m=>m!=="No Mistakes — Executed the Plan ✓").length) lines.push(`  Mistakes: ${e.sessionMistakes.filter(m=>m!=="No Mistakes — Executed the Plan ✓").join(" | ")}`);
                const nf=[["Market notes",e.marketNotes],["Rules",e.rules],["Lessons",e.lessonsLearned],["Mistakes note",e.mistakes],["Improvements",e.improvements],["Best trade",e.bestTrade],["Worst trade",e.worstTrade],["Reinforce",e.reinforceRule],["Tomorrow plan",e.tomorrow]];
                for (const [lbl,val] of nf) if (val?.trim()) lines.push(`  ${lbl}: ${val.trim()}`);
                return lines.join("\n");
              }).join("\n\n");

              const prompt = `You are a professional futures trading coach. A trader has shared their complete journal for: ${weekLabel}

Be direct and specific — cite exact dates, dollar amounts, and quote their own words. Sharp mentor, not a cheerleader.

PERIOD OVERVIEW
Days: ${sorted.length} | ${winDays}W / ${lossDays}L | Gross: $${grossPnlW.toFixed(0)} | Fees: -$${fees.toFixed(0)} | Net: $${totalPnl.toFixed(0)} | Avg/day: $${(totalPnl/sorted.length).toFixed(0)}
Trades: ${allT.length} | WR: ${wr}% | PF: ${fmtPF(pfv)} | Avg win: +$${avgW} | Avg loss: -$${avgL}
Commission drag: $${fees.toFixed(0)} = ${commDragW}% of gross${parseFloat(commDragW)>30?" ⚠ HIGH":""}
${carryW>0?`Overnight carry-forwards: ${carryW} trade(s)\n`:""}DIRECTION: ${wDirLine}
SESSION (exit time ET): ${wSessLine}
${wBestSess?`Best session: ${wBestSess[0]} net $${(wBestSess[1].pnl-wBestSess[1].comm).toFixed(0)}`:""}${wWorstSess&&wWorstSess[0]!==wBestSess?.[0]?` | Worst: ${wWorstSess[0]} net $${(wWorstSess[1].pnl-wWorstSess[1].comm).toFixed(0)}`:""}
Grades: ${grades}
Mistakes tagged (light context): ${mistakeLine}${cDays>0?` | ${cDays} clean days`:""}
${planLines.length?`\nPLAN vs ACTUAL\n${planLines.join("\n")}`:""}

FULL JOURNAL — per-trade detail + every written note, day by day:
${dayBlocks}

---

BEFORE WRITING: Check the DIRECTION and SESSION data. If one direction is net-negative while the other is profitable, that is the headline. If commission drag > 30%, name it.

**📊 HEADLINE INSIGHTS**
3-4 punchy one-liners from the actual data, format: [Pattern]: [numbers] → [what it means]

**📓 NOTES ANALYSIS**
Read every written word:
• Recurring themes (quote each with date)
• Contradictions between written intentions and what trades show
• Unrealized plans that never appeared in subsequent behavior
• One observation worth reinforcing

**📈 PERFORMANCE PATTERNS**
3-4 bullets: direction/session edge vs drag, mistakes on specific day types, commission drag if material, any hold-time pattern.

**🚩 PLAN vs REALITY**
For every Tomorrow Plan written, use this format:
• **[Date] Plan:** Quote the specific intention (exact words, under 30 words)
• **What happened:** Actual P&L, grade, and key trades — specific, not just a label
• **Verdict:** "Plan held — [what they did right]" OR "Plan slipped — [the specific rule that broke, with evidence from trades or notes]"
Never use ✓ or ✗ as the lead word. Cite trade times, dollar figures, or quotes from notes as evidence. Skip if no plans exist.

**💡 STRENGTHS**
2-3 specific strengths with dates and dollar figures.

**🎯 ACTION PLAN FOR NEXT WEEK**
Exactly 3 rules. Format: [Root cause from this review] → [Specific measurable rule with threshold]`;

              const txt = await aiRequestText(ai, {
                max_tokens: 8192,
                timeoutMs: 120000,
                thinkingBudget: 0,
                messages: [{ role: 'user', content: prompt }],
              });
              saveWeeklyRecap(selectedWeek, txt, currentHash);
            } catch (err) {
              const f = friendlyAiError(err);
              setRecapError(f.message || 'Failed to generate recap. Try again.');
            }
            setRecapLoading(false);
          };

          return (
            <div style={{ marginBottom: 20 }}>
              <div style={{ height: 2, background: "linear-gradient(90deg,#7c3aed,#818cf8,#38bdf8,transparent)", borderRadius: 1, marginBottom: 16 }} />
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14, flexWrap: "wrap", gap: 8 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ fontSize: 22 }}>🤖</span>
                  <div>
                    <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 18, letterSpacing: "0.1em", background: "linear-gradient(135deg,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>AI RECAP OF THE WEEK</div>
                    {generatedDate && <div style={{ fontSize: 9, color: "#475569", letterSpacing: "0.1em", marginTop: 1 }}>Generated {generatedDate}</div>}
                  </div>
                </div>
                <button onClick={generateRecap} disabled={recapLoading}
                  style={{ background: saved ? "transparent" : "linear-gradient(135deg,#7c3aed,#818cf8)", color: saved ? "#818cf8" : "white", border: saved ? "1px solid rgba(129,140,248,0.4)" : "none", padding: "8px 18px", borderRadius: 5, fontFamily: "inherit", fontSize: 11, cursor: recapLoading ? "not-allowed" : "pointer", letterSpacing: "0.06em", fontWeight: 600, opacity: recapLoading ? 0.6 : 1, transition: "all .15s" }}>
                  {recapLoading ? "⏳ GENERATING..." : saved ? "↺ RE-RUN" : "✦ GENERATE RECAP"}
                </button>
              </div>

              {isStale && (
                <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", background: "rgba(251,191,36,0.08)", border: "1px solid rgba(251,191,36,0.3)", borderRadius: 4, marginBottom: 12, fontSize: 11, color: "#fbbf24" }}>
                  ⚠ Your notes were updated after this recap was generated. Hit ↺ RE-RUN to refresh.
                </div>
              )}

              {recapError && (
                <div style={{ padding: "10px 14px", background: "rgba(127,29,29,0.2)", border: "1px solid #7f1d1d", borderRadius: 4, marginBottom: 12, fontSize: 11, color: "#f87171" }}>
                  {recapError}
                </div>
              )}

              {recapLoading && (
                <div style={{ padding: "40px 20px", display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
                  <style>{`@keyframes aiPulse{0%,100%{opacity:1}50%{opacity:0.3}}`}</style>
                  <div style={{ fontSize: 11, color: "#818cf8", letterSpacing: "0.15em", animation: "aiPulse 1.8s infinite" }}>✦ ANALYSING YOUR WEEK...</div>
                  <div style={{ fontSize: 10, color: "#334155" }}>Reading all notes, trades, and behavioral patterns</div>
                </div>
              )}

              {!recapLoading && saved?.text && (
                <div style={{ background: "linear-gradient(135deg,rgba(124,58,237,0.06),rgba(129,140,248,0.08),rgba(56,189,248,0.04))", border: "1px solid rgba(129,140,248,0.2)", borderRadius: 6, padding: "20px 24px" }}>
                  <RenderAI text={saved.text} />
                </div>
              )}

              {!recapLoading && !saved && !recapError && (
                <div style={{ padding: "32px 20px", textAlign: "center", background: "#0a0e1a", border: "1px solid #1e293b", borderRadius: 6 }}>
                  <div style={{ fontSize: 22, marginBottom: 10 }}>🤖</div>
                  <div style={{ fontSize: 12, color: "#475569", lineHeight: 1.7 }}>No recap yet for this week.<br />Hit <span style={{ color: "#818cf8" }}>✦ GENERATE RECAP</span> to get your AI coaching review.</div>
                </div>
              )}
            </div>
          );
        })()}

        {/* Notes Section */}
        {wEntries.some(e => noteFields.some(({ key }) => getNote(e, key))) && (
          <div>
            <div style={{ height: 1, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc,transparent)", marginBottom: 16, opacity: 0.4 }} />
            <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 16 }}>📓 WEEKLY NOTES REVIEW</div>

            {/* Compiled by field — each section individually collapsible, all collapsed by default */}
            <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 20 }}>
              {noteFields.map(({ key, label, icon, color }) => {
                const entries_with_note = wEntries.filter(e => getNote(e, key));
                if (!entries_with_note.length) return null;
                const isOpen = collapsedNotes[key] === true; // default collapsed (undefined = closed)
                return (
                  <div key={key} style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 5, overflow: "hidden" }}>
                    <button
                      onClick={() => toggleNote(key)}
                      style={{ width: "100%", background: "transparent", border: "none", padding: "12px 16px", display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer", fontFamily: "inherit" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ fontSize: 13 }}>{icon}</span>
                        <span style={{ fontSize: 10, color, letterSpacing: "0.1em" }}>{label}</span>
                        <span style={{ fontSize: 9, color: "#475569", background: "#0a0e1a", padding: "1px 6px", borderRadius: 2 }}>{entries_with_note.length} day{entries_with_note.length !== 1 ? "s" : ""}</span>
                      </div>
                      <span style={{ fontSize: 10, color: "#475569", transition: "transform .15s", display: "inline-block", transform: isOpen ? "rotate(180deg)" : "rotate(0deg)" }}>▾</span>
                    </button>
                    {isOpen && (
                      <div style={{ padding: "0 16px 14px", display: "flex", flexDirection: "column", gap: 10 }}>
                        {entries_with_note.map((e, i) => {
                          const dow = new Date(e.date + "T12:00:00").toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
                          return (
                            <div key={i} style={{ paddingLeft: 10, borderLeft: `2px solid ${color}44` }}>
                              <div style={{ fontSize: 9, color: "#475569", letterSpacing: "0.08em", marginBottom: 3 }}>{dow}</div>
                              <div style={{ fontSize: 12, color: "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap" }}>{getNote(e, key)}</div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Day by day — each day individually collapsible, all collapsed by default */}
            <div style={{ fontSize: 10, color: "#475569", letterSpacing: "0.12em", marginBottom: 10 }}>DAY BY DAY</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {wEntries.map(e => {
                if (!noteFields.some(({ key }) => getNote(e, key))) return null;
                const dow = new Date(e.date + "T12:00:00").toLocaleDateString("en-US", { weekday: "long", month: "short", day: "numeric" });
                const isOpen = dayByDayCollapsed[e.id] === true;
                return (
                  <div key={e.id} style={{ background: "#0a0e1a", border: "1px solid #1e293b", borderRadius: 5, overflow: "hidden" }}>
                    <button
                      onClick={() => toggleDay(e.id)}
                      style={{ width: "100%", background: "transparent", border: "none", padding: "12px 16px", display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer", fontFamily: "inherit" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <span style={{ fontSize: 12, color: "#e2e8f0", fontWeight: 600 }}>{dow}</span>
                        {e.grade && <span style={{ fontSize: 10, padding: "1px 7px", borderRadius: 3, background: "#0f172a", border: `1px solid ${pnlColor(calcNetPnl(e))}44`, color: pnlColor(calcNetPnl(e)) }}>{e.grade}</span>}
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <span style={{ fontSize: 12, fontWeight: 700, color: pnlColor(calcNetPnl(e)) }}>{fmtPnl(calcNetPnl(e))}</span>
                        <span style={{ fontSize: 10, color: "#475569", transition: "transform .15s", display: "inline-block", transform: isOpen ? "rotate(180deg)" : "rotate(0deg)" }}>▾</span>
                      </div>
                    </button>
                    {isOpen && (
                      <div style={{ padding: "0 16px 14px", display: "flex", flexDirection: "column", gap: 10 }}>
                        {noteFields.map(({ key, label, icon, color }) => {
                          const val = getNote(e, key); if (!val) return null;
                          return (
                            <div key={key}>
                              <div style={{ fontSize: 9, color, letterSpacing: "0.08em", marginBottom: 3 }}>{icon} {label}</div>
                              <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.7, whiteSpace: "pre-wrap" }}>{val}</div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    );
  }

  // ── WEEK LIST ──────────────────────────────────────────────────────────────
  if (weeks.length === 0) return <div style={{ textAlign: "center", padding: "60px 0", color: "#64748b", fontSize: 12 }}>No entries for {selectedYear}.</div>;

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, flexWrap: "wrap", gap: 10 }}>
        <div style={{ display: "flex", gap: 6 }}>
          {years.map(y => (
            <button key={y} onClick={() => { setSelectedYear(y); setSelectedWeek(null); }}
              style={{ padding: "5px 12px", borderRadius: 3, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.05em", transition: "all .15s", background: selectedYear === y ? "#1e3a5f" : "transparent", border: `1px solid ${selectedYear === y ? "#3b82f6" : "#1e293b"}`, color: selectedYear === y ? "#93c5fd" : "#94a3b8" }}>
              {y}
            </button>
          ))}
        </div>
        <div style={{ fontSize: 10, color: "#64748b" }}>{weeks.length} weeks · {yearEntries.length} trading days</div>
      </div>
      <div style={{ marginBottom: 18 }}>
        <div style={{ height: 1, background: "linear-gradient(90deg, #38bdf8, #818cf8, #c084fc, transparent)", marginBottom: 10, opacity: 0.5 }} />
        <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
          <div style={{ fontFamily: "\'Bebas Neue\',sans-serif", fontSize: 38, letterSpacing: "0.1em", lineHeight: 1, background: "linear-gradient(135deg,#38bdf8 0%,#818cf8 55%,#c084fc 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>WEEKLY BREAKDOWN</div>
          <div style={{ fontFamily: "\'DM Mono\',monospace", fontSize: 14, color: "#475569", letterSpacing: "0.18em", paddingBottom: 3 }}>{selectedYear}</div>
        </div>
        <div style={{ height: 1, background: "linear-gradient(90deg, transparent, #818cf8, #c084fc, transparent)", marginTop: 8, opacity: 0.25 }} />
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {weeks.slice().reverse().map((wk) => {
          const wEntries = byWeek[wk] || [];
          const wNet = wEntries.reduce((s, e) => s + calcNetPnl(e), 0);
          const wGross = wEntries.reduce((s, e) => s + (parseFloat(e.pnl) || 0), 0);
          const wFees = wEntries.reduce((s, e) => s + (parseFloat(e.commissions) || 0), 0);
          const wWinDays = wEntries.filter(e => calcNetPnl(e) > 0).length;
          const wLossDays = wEntries.filter(e => calcNetPnl(e) < 0).length;
          const wTrades = wEntries.flatMap(e => e.parsedTrades || []);
          const wA = calcAnalytics(wTrades, true);
          const isPos = wNet >= 0;
          const weekNum = wk.split("-W")[1];
          const { label: wLabel } = getWeekRange(wk);
          const hasNotes = wEntries.some(e => e.lessonsLearned || e.mistakes || e.improvements || e.tomorrow || e.rules);
          const hasMistakes = wEntries.some(e => (e.sessionMistakes||[]).some(m => m !== "No Mistakes — Executed the Plan ✓"));
          return (
            <div key={wk} onClick={() => setSelectedWeek(wk)}
              style={{ background: "#0a0e1a", border: `1px solid ${isPos ? "#166534" : wNet < 0 ? "#7f1d1d" : "#1e293b"}`, borderRadius: 6, overflow: "hidden", cursor: "pointer", transition: "border-color .15s, background .15s" }}
              onMouseEnter={e => { e.currentTarget.style.borderColor="#3b82f6"; e.currentTarget.style.background="#0f1729"; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor=isPos?"#166534":wNet<0?"#7f1d1d":"#1e293b"; e.currentTarget.style.background="#0a0e1a"; }}>
              <div style={{ padding: "14px 18px", background: isPos ? "#061f0f" : wNet < 0 ? "#1f0606" : "#0f1729", display: "grid", gridTemplateColumns: "1fr auto", gap: 16, alignItems: "center", borderLeft: "3px solid transparent", borderImage: "linear-gradient(180deg,#38bdf8,#818cf8,#c084fc) 1" }}>
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6, flexWrap: "wrap" }}>
                    <span style={{ fontSize: 11, color: isPos ? "#4ade80" : wNet < 0 ? "#f87171" : "#93c5fd", letterSpacing: "0.1em", fontWeight: 600 }}>WEEK {weekNum}</span>
                    <span style={{ fontSize: 11, color: "#94a3b8" }}>{wLabel}</span>
                    {hasNotes && <span style={{ fontSize: 8, color: "#818cf8", background: "rgba(129,140,248,0.1)", padding: "1px 6px", borderRadius: 2 }}>📓 NOTES</span>}
                    {hasMistakes && <span style={{ fontSize: 8, color: "#f87171", background: "rgba(248,113,113,0.1)", padding: "1px 6px", borderRadius: 2 }}>⚠</span>}
                    <span style={{ fontSize: 8, color: "#334155", marginLeft: 4 }}>CLICK TO REVIEW →</span>
                  </div>
                  <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
                    <div><div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.08em", marginBottom: 2 }}>DAYS</div><div style={{ fontSize: 12, color: "#e2e8f0" }}><span style={{ color: "#4ade80" }}>{wWinDays}W</span><span style={{ color: "#475569" }}>/</span><span style={{ color: "#f87171" }}>{wLossDays}L</span></div></div>
                    {wA && <>
                      <div><div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.08em", marginBottom: 2 }}>TRADES</div><div style={{ fontSize: 12 }}><span style={{ color: "#4ade80" }}>{wA.winners}W</span><span style={{ color: "#475569" }}>/</span><span style={{ color: "#f87171" }}>{wA.losers}L</span></div></div>
                      <div><div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.08em", marginBottom: 2 }}>WIN RATE</div><div style={{ fontSize: 12, color: wA.winRate >= 50 ? "#4ade80" : "#f87171", fontWeight: 600 }}>{wA.winRate.toFixed(0)}%</div></div>
                      {wA.profitFactor != null && <div><div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.08em", marginBottom: 2 }}>PF</div><div style={{ fontSize: 12, color: pfColor(wA.profitFactor), fontWeight: 600 }}>{fmtPF(wA.profitFactor)}</div></div>}
                    </>}
                  </div>
                </div>
                <div style={{ textAlign: "right" }}>
                  {wFees > 0 && <div style={{ fontSize: 10, color: "#475569", marginBottom: 4 }}>gross {fmtPnl(wGross)}</div>}
                  <div style={{ fontSize: 22, color: pnlColor(wNet), fontWeight: 700, lineHeight: 1 }}>{fmtPnl(wNet)}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PerformanceOverview({ entries, netPnl: calcNetPnlProp, fmtPnl, pnlColor }) {
  const calcNetPnl = calcNetPnlProp;
  const today = new Date();
  const currentYear = today.getFullYear();
  const [selectedYear, setSelectedYear] = useState(currentYear);
  const [period, setPeriod] = useState("year"); // "q1"|"q2"|"q3"|"q4"|"h1"|"h2"|"year"
  const [expandedMonth, setExpandedMonth] = useState(null);
  const [collapsedSections, setCollapsedSections] = useState({});

  const toggleSection = (monthKey, skey) => {
    const k = `${monthKey}-${skey}`;
    setCollapsedSections(prev => ({ ...prev, [k]: !prev[k] }));
  };
  const isCollapsed = (monthKey, skey) => !!collapsedSections[`${monthKey}-${skey}`];

  // Available years from entries
  const years = [...new Set(entries.map(e => e.date?.slice(0, 4)).filter(Boolean))].map(Number).sort((a, b) => b - a);
  if (!years.includes(currentYear)) years.unshift(currentYear);

  const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  const periodMonths = {
    q1: [0,1,2], q2: [3,4,5], q3: [6,7,8], q4: [9,10,11],
    h1: [0,1,2,3,4,5], h2: [6,7,8,9,10,11],
    year: [0,1,2,3,4,5,6,7,8,9,10,11],
  };

  const activeMonths = periodMonths[period];

  // Build per-month data
  const monthData = activeMonths.map(mi => {
    const monthStr = `${selectedYear}-${String(mi + 1).padStart(2, "0")}`;
    const monthEntries = entries.filter(e => e.date?.startsWith(monthStr));
    const trades = monthEntries.flatMap(e => e.parsedTrades || []);
    const a = calcAnalytics(trades, true);
    const grossPnl = monthEntries.reduce((s, e) => s + (parseFloat(e.pnl) || 0), 0);
    const fees = monthEntries.reduce((s, e) => s + (parseFloat(e.commissions) || 0), 0);
    const netTotal = monthEntries.reduce((s, e) => s + netPnl(e), 0);
    const winDays = monthEntries.filter(e => netPnl(e) > 0).length;
    const lossDays = monthEntries.filter(e => netPnl(e) < 0).length;
    return { mi, monthStr, monthEntries, trades, a, grossPnl, fees, netTotal, winDays, lossDays };
  });

  // Period totals
  const periodEntries = entries.filter(e => {
    const yr = parseInt(e.date?.slice(0, 4));
    const mo = parseInt(e.date?.slice(5, 7)) - 1;
    return yr === selectedYear && activeMonths.includes(mo);
  });
  const periodTrades = periodEntries.flatMap(e => e.parsedTrades || []);
  const periodA = calcAnalytics(periodTrades, true);
  const periodNet = periodEntries.reduce((s, e) => s + netPnl(e), 0);
  const periodGross = periodEntries.reduce((s, e) => s + (parseFloat(e.pnl) || 0), 0);
  const periodFees = periodEntries.reduce((s, e) => s + (parseFloat(e.commissions) || 0), 0);
  const periodWinDays = periodEntries.filter(e => netPnl(e) > 0).length;
  const periodLossDays = periodEntries.filter(e => netPnl(e) < 0).length;

  // Cumulative equity across period
  const sortedPeriodEntries = [...periodEntries].sort((a, b) => a.date.localeCompare(b.date));
  let running = 0;
  const equityPoints = sortedPeriodEntries.map(e => { running += netPnl(e); return running; });

  const PERIOD_LABELS = { q1: "Q1 · JAN–MAR", q2: "Q2 · APR–JUN", q3: "Q3 · JUL–SEP", q4: "Q4 · OCT–DEC", h1: "H1 · JAN–JUN", h2: "H2 · JUL–DEC", year: "FULL YEAR" };

  const MonthSectionHeader = ({ label, monthKey, skey, summary }) => {
    const c = isCollapsed(monthKey, skey);
    return (
      <div onClick={() => toggleSection(monthKey, skey)}
        style={{ display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer", userSelect: "none", padding: "6px 10px", borderRadius: 4, marginBottom: c ? 0 : 8, background: c ? "#0f1729" : "transparent", border: `1px solid ${c ? "#1e3a5f" : "transparent"}`, transition: "all .15s" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ fontSize: 11, color: c ? "#93c5fd" : "#93c5fd", letterSpacing: "0.1em" }}>{label}</div>
          {c && summary && <div style={{ fontSize: 11, color: "#94a3b8" }}>{summary}</div>}
        </div>
        <span style={{ fontSize: 9, color: c ? "#3b82f6" : "#475569", display: "inline-block", transform: c ? "rotate(-90deg)" : "rotate(0deg)", transition: "all .2s" }}>▾</span>
      </div>
    );
  };

  return (
    <div>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20, flexWrap: "wrap", gap: 10 }}>
        <div>
          <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 22, letterSpacing: "0.1em", background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>PERFORMANCE OVERVIEW</div>
          <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>Cumulative results by month · click any month to expand full breakdown</div>
        </div>
        {/* Year selector */}
        <div style={{ display: "flex", gap: 6 }}>
          {years.map(y => (
            selectedYear === y ? (
              <button key={y} onClick={() => { setSelectedYear(y); setExpandedMonth(null); }}
                style={{ padding: "5px 12px", borderRadius: 3, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.05em", transition: "all .15s", background: "#1e3a5f", border: "1px solid #3b82f6", color: "#93c5fd", position: "relative", overflow: "hidden" }}>
                <span style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: 1, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc)" }} />
                {y}
              </button>
            ) : (
              <span key={y} style={{ display: "inline-block", padding: 1, borderRadius: 4, background: "linear-gradient(135deg,rgba(56,189,248,0.45),rgba(129,140,248,0.45),rgba(192,132,252,0.45))" }}>
                <button onClick={() => { setSelectedYear(y); setExpandedMonth(null); }}
                  style={{ display: "block", background: "#070d1a", color: "#64748b", border: "none", padding: "4px 11px", borderRadius: 2, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.05em" }}>
                  {y}
                </button>
              </span>
            )
          ))}
        </div>
      </div>

      {/* Period selector */}
      <div style={{ display: "flex", gap: 6, marginBottom: 20, flexWrap: "wrap" }}>
        {Object.entries(PERIOD_LABELS).map(([key, label]) => (
          period === key ? (
            <button key={key} onClick={() => { setPeriod(key); setExpandedMonth(null); }}
              style={{ padding: "6px 14px", borderRadius: 3, fontFamily: "inherit", fontSize: 10, cursor: "pointer", letterSpacing: "0.06em", transition: "all .15s", background: "#1e3a5f", border: "1px solid #3b82f6", color: "#93c5fd", position: "relative", overflow: "hidden" }}>
              <span style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: 1, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc)" }} />
              {label}
            </button>
          ) : (
            <span key={key} style={{ display: "inline-block", padding: 1, borderRadius: 4, background: "linear-gradient(135deg,rgba(56,189,248,0.45),rgba(129,140,248,0.45),rgba(192,132,252,0.45))" }}>
              <button onClick={() => { setPeriod(key); setExpandedMonth(null); }}
                style={{ display: "block", background: "#070d1a", color: "#64748b", border: "none", padding: "5px 13px", borderRadius: 2, fontFamily: "inherit", fontSize: 10, cursor: "pointer", letterSpacing: "0.06em" }}>
                {label}
              </button>
            </span>
          )
        ))}
      </div>

      {/* Period summary bar */}
      {periodEntries.length > 0 && (
        <div style={{ background: "#0a1628", border: "1px solid #1e3a5f", borderRadius: 6, padding: "14px 18px", marginBottom: 20 }}>
          <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 12 }}>{PERIOD_LABELS[period]} SUMMARY · {selectedYear}</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 14 }}>
            {[
              { l: "NET P&L", v: fmtPnl(periodNet), c: pnlColor(periodNet), hi: true },
              { l: "GROSS P&L", v: fmtPnl(periodGross), c: pnlColor(periodGross) },
              { l: "TOTAL FEES", v: periodFees > 0 ? `-$${periodFees.toFixed(0)}` : "—", c: "#475569", small: true },
              { l: "WIN DAYS", v: `${periodWinDays}/${periodLossDays}`, c: "#e2e8f0" },
              { l: "TOTAL TRADES", v: periodTrades.length, c: "#e2e8f0" },
              { l: "WIN RATE", v: periodA ? `${periodA.winRate.toFixed(1)}%` : "—", c: periodA?.winRate >= 50 ? "#4ade80" : "#f87171" },
            ].map(s => (
              <div key={s.l}>
                <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.08em", marginBottom: 4 }}>{s.l}</div>
                <div style={{ fontSize: 15, color: s.c, fontWeight: s.hi ? 700 : 500 }}>{s.v}</div>
              </div>
            ))}
          </div>
          {/* Period equity curve */}
          {equityPoints.length >= 1 && (
            <div style={{ marginTop: 14 }}>
              <EquityCurveChart values={equityPoints.length === 1 ? [0, equityPoints[0]] : equityPoints} height={80} gradientId="ec3" />
            </div>
          )}
        </div>
      )}

      {/* ── MONTH GRID — immediately after Full Year Summary ── */}
      <div style={{ marginBottom: 14, position: "relative" }}>
        <div style={{ height: 1, background: "linear-gradient(90deg, #38bdf8, #818cf8, #c084fc, transparent)", marginBottom: 10, opacity: 0.5 }} />
        <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
          <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 38, letterSpacing: "0.1em", lineHeight: 1, background: "linear-gradient(135deg,#38bdf8 0%,#818cf8 55%,#c084fc 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>
            {PERIOD_LABELS[period]}
          </div>
          <div style={{ fontFamily: "'DM Mono',monospace", fontSize: 14, color: "#475569", letterSpacing: "0.18em", paddingBottom: 3 }}>
            {selectedYear}
          </div>
        </div>
        <div style={{ height: 1, background: "linear-gradient(90deg, transparent, #818cf8, #c084fc, transparent)", marginTop: 8, opacity: 0.25 }} />
      </div>
      {/* Month grid */}
      <div style={{ display: "grid", gridTemplateColumns: period === "year" ? "repeat(4, 1fr)" : period === "h1" || period === "h2" ? "repeat(3, 1fr)" : "repeat(3, 1fr)", gap: 10, marginBottom: 16 }}>
        {monthData.map(({ mi, monthStr, monthEntries, trades, a, netTotal, winDays, lossDays }) => {
          const hasData = monthEntries.length > 0;
          const isExpanded = expandedMonth === monthStr;
          const isFuture = monthStr > `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}`;
          const isCurrentMonth = monthStr === `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}`;

          return (
            <div key={monthStr}
              onClick={() => hasData && setExpandedMonth(isExpanded ? null : monthStr)}
              style={{ background: hasData ? (netTotal > 0 ? "rgba(16,63,33,0.55)" : netTotal < 0 ? "rgba(63,16,16,0.5)" : "#0f1729") : isFuture ? "#060810" : "#0a0e1a", border: `1px solid ${isExpanded ? "#3b82f6" : hasData ? (netTotal > 0 ? "#166534" : netTotal < 0 ? "#7f1d1d" : "#1e293b") : "#0f1729"}`, borderRadius: 6, padding: "16px 18px", minHeight: hasData ? 130 : undefined, cursor: hasData ? "pointer" : "default", transition: "all .15s", opacity: isFuture ? 0.4 : 1, position: "relative", overflow: "hidden",
                ...(isCurrentMonth ? { boxShadow: "0 0 0 2px #818cf8, 0 0 0 3px rgba(56,189,248,0.4), 0 0 0 4px rgba(192,132,252,0.3)" } : {}) }}
              onMouseEnter={e => { if (hasData) e.currentTarget.style.borderColor = netTotal > 0 ? "#22c55e" : netTotal < 0 ? "#ef4444" : "#3b82f6"; }}
              onMouseLeave={e => { if (hasData) e.currentTarget.style.borderColor = isExpanded ? "#3b82f6" : netTotal > 0 ? "#166534" : netTotal < 0 ? "#7f1d1d" : "#1e293b"; }}>
              <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc)" }} />
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: hasData ? 10 : 0 }}>
                {hasData ? (
                <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 18, letterSpacing: "0.1em", background: netTotal > 0 ? "linear-gradient(135deg,#4ade80,#38bdf8)" : netTotal < 0 ? "linear-gradient(135deg,#f87171,#818cf8)" : "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>{MONTH_NAMES[mi]}</div>
              ) : (
                <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 18, color: "#1e3a5f", letterSpacing: "0.1em" }}>{MONTH_NAMES[mi]}</div>
              )}
                {isExpanded && <span style={{ fontSize: 9, color: "#3b82f6" }}>▴</span>}
                {!isExpanded && hasData && <span style={{ fontSize: 9, color: "#64748b" }}>▾</span>}
              </div>
              {hasData ? (
                <>
                  <div style={{ fontSize: 20, fontWeight: 700, color: pnlColor(netTotal), lineHeight: 1, marginBottom: 8 }}>
                    {fmtPnl(netTotal)}
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                    <div style={{ fontSize: 10, color: "#94a3b8" }}>
                      <span style={{ letterSpacing: 0 }}><span style={{ color: "#4ade80" }}>{winDays}</span><span style={{ color: "#475569" }}>/</span><span style={{ color: "#f87171" }}>{lossDays}</span></span>
                      <span style={{ color: "#64748b" }}> days</span>
                    </div>
                    {a && <div style={{ fontSize: 10, color: "#94a3b8" }}>{trades.length} trades · <span style={{ letterSpacing: 0 }}><span style={{ color: "#4ade80" }}>{a.winners}</span><span style={{ color: "#475569" }}>/</span><span style={{ color: "#f87171" }}>{a.losers}</span></span> · <span style={{ color: a.winRate >= 50 ? "#4ade80" : "#f87171" }}>{a.winRate.toFixed(0)}% WR</span></div>}
                    {a && <div style={{ fontSize: 10, color: "#64748b" }}>PF {fmtPF(a.profitFactor)}</div>}
                  </div>
                </>
              ) : (
                <div style={{ fontSize: 10, color: "#1e3a5f" }}>{isFuture ? "upcoming" : "no trades"}</div>
              )}
            </div>
          );
        })}
      </div>

      {/* Expanded month detail panel */}
      {expandedMonth && (() => {
        const { monthEntries: me, trades, a, netTotal: mNet, grossPnl: mGross, fees: mFees, winDays: mWins, lossDays: mLoss } = monthData.find(d => d.monthStr === expandedMonth);
        const mi = parseInt(expandedMonth.slice(5, 7)) - 1;
        const monthLabel = new Date(expandedMonth + "-01T12:00:00").toLocaleString("default", { month: "long", year: "numeric" }).toUpperCase();

        if (!me.length) return null;
        return (
          <div style={{ background: "#0a0e1a", border: "1px solid #1e3a5f", borderRadius: 6, overflow: "hidden", marginBottom: 16 }}>
            <div style={{ padding: "12px 18px", background: "#0a1628", borderBottom: "1px solid #1e293b", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", fontWeight: 600 }}>{monthLabel} · BREAKDOWN</div>
              <div style={{ display: "flex", gap: 12, fontSize: 10, color: "#94a3b8", alignItems: "center" }}>
                <span>{me.length} trading days</span>
                <span style={{ letterSpacing: 0 }}><span style={{ color: "#4ade80" }}>{mWins}</span><span style={{ color: "#94a3b8" }}>/</span><span style={{ color: "#f87171" }}>{mLoss}</span></span>
                <button onClick={() => setExpandedMonth(null)} style={{ background: "transparent", border: "1px solid #1e293b", color: "#64748b", padding: "3px 10px", borderRadius: 3, fontFamily: "inherit", fontSize: 10, cursor: "pointer" }}>✕ close</button>
              </div>
            </div>

            <div style={{ padding: "16px 18px", display: "flex", flexDirection: "column", gap: 16 }}>


      {/* Day of Week Performance */}
      {periodEntries.length > 0 && (() => {
        const DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"];
        const dowStats = {};
        for (const day of DOW) dowStats[day] = { pnl: 0, trades: 0, wins: 0, days: 0, tradeWins: 0, tradeLosses: 0 };
        for (const e of sortedPeriodEntries) {
          const d = new Date(e.date + "T12:00:00");
          const day = DOW[d.getDay() - 1];
          if (!day) continue;
          const np = netPnl(e);
          dowStats[day].pnl += np;
          dowStats[day].days++;
          if (np > 0) dowStats[day].wins++;
          const trades = e.parsedTrades || [];
          dowStats[day].trades += trades.length;
          dowStats[day].tradeWins += trades.filter(t => (t.pnl-(t.commission||0)) > 0).length;
          dowStats[day].tradeLosses += trades.filter(t => (t.pnl-(t.commission||0)) < 0).length;
        }
        const activeDays = DOW.filter(d => dowStats[d].days > 0);
        if (activeDays.length === 0) return null;
        const maxAbsPnl = Math.max(...activeDays.map(d => Math.abs(dowStats[d].pnl)), 1);
        const bestDay = activeDays.reduce((a, b) => dowStats[a].pnl > dowStats[b].pnl ? a : b);
        const worstDay = activeDays.reduce((a, b) => dowStats[a].pnl < dowStats[b].pnl ? a : b);
        return (
          <div style={{ background: "#0a0e1a", border: "1px solid #1e293b", borderRadius: 6, padding: "16px 18px", marginBottom: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 16 }}>
              <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em" }}>📅 PERFORMANCE BY DAY OF WEEK</div>
              <div style={{ fontSize: 10, color: "#64748b" }}>
                Best: <span style={{ color: "#4ade80" }}>{bestDay}</span>
                {dowStats[worstDay].pnl < 0 && <> · Avoid: <span style={{ color: "#f87171" }}>{worstDay}</span></>}
              </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 8 }}>
              {DOW.map(day => {
                const s = dowStats[day];
                const hasData = s.days > 0;
                const wr = hasData ? Math.round(s.wins / s.days * 100) : 0;
                const barH = hasData ? Math.abs(s.pnl) / maxAbsPnl * 60 : 0;
                const isPos = s.pnl >= 0;
                const isBest = day === bestDay && hasData;
                const isWorst = day === worstDay && hasData && s.pnl < 0;
                return (
                  <div key={day} style={{ background: isBest ? "#061f0f" : isWorst ? "#1f0606" : "#0f1729", border: `1px solid ${isBest ? "#166534" : isWorst ? "#7f1d1d" : "#1e293b"}`, borderRadius: 5, padding: "12px 10px", textAlign: "center", opacity: hasData ? 1 : 0.3 }}>
                    <div style={{ fontSize: 10, color: isBest ? "#4ade80" : isWorst ? "#f87171" : "#64748b", letterSpacing: "0.08em", marginBottom: 8 }}>{day.slice(0,3).toUpperCase()}</div>
                    {/* Bar chart */}
                    <div style={{ height: 64, display: "flex", alignItems: "flex-end", justifyContent: "center", marginBottom: 8 }}>
                      {hasData ? (
                        <div style={{ width: 28, borderRadius: "3px 3px 0 0", background: isPos ? "#4ade80" : "#f87171", height: `${Math.max(barH, 4)}px`, opacity: 0.8, transition: "height .3s" }} />
                      ) : (
                        <div style={{ width: 28, height: 4, background: "#1e293b", borderRadius: 2 }} />
                      )}
                    </div>
                    {hasData ? (
                      <>
                        <div style={{ fontSize: 13, fontWeight: 600, color: isPos ? "#4ade80" : "#f87171", marginBottom: 4 }}>
                          {s.pnl >= 0 ? "+" : ""}${Math.abs(s.pnl).toFixed(0)}
                        </div>
                        <div style={{ fontSize: 10, color: wr >= 50 ? "#4ade80" : "#f87171", marginBottom: 3 }}>{wr}% WR</div>
                        <div style={{ fontSize: 9, color: "#64748b" }}>
                          {s.days} day{s.days !== 1 ? "s" : ""} · {s.trades > 0 ? <><span style={{ color: "#4ade80" }}>{s.tradeWins}</span><span style={{ color: "#475569" }}>/</span><span style={{ color: "#f87171" }}>{s.tradeLosses}</span></> : "0"} trades
                        </div>
                      </>
                    ) : (
                      <div style={{ fontSize: 9, color: "#1e3a5f" }}>no data</div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })()}

              {/* P&L Summary */}
              <div>
                <MonthSectionHeader label="P&L SUMMARY" monthKey={expandedMonth} skey="pnl" summary={<span style={{ color: pnlColor(mNet), fontWeight: 600 }}>{fmtPnl(mNet)} net</span>} />
                {!isCollapsed(expandedMonth, "pnl") && (
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
                    {[
                      { l: "GROSS P&L", v: fmtPnl(mGross), c: pnlColor(mGross) },
                      { l: "TOTAL FEES", v: mFees > 0 ? `-$${mFees.toFixed(2)}` : "—", c: "#475569", small: true },
                      { l: "NET P&L", v: fmtPnl(mNet), c: pnlColor(mNet), hi: true },
                      { l: "DAY WIN RATE", v: me.length ? `${Math.round(mWins / me.length * 100)}%` : "—", c: mWins / me.length >= 0.5 ? "#4ade80" : "#f87171" },
                      { l: "WIN DAYS", v: mWins, c: "#4ade80" },
                      { l: "LOSS DAYS", v: mLoss, c: "#f87171" },
                      ...(a ? [
                        { l: "LARGEST WIN DAY", v: fmtPnl(Math.max(...me.map(e => netPnl(e)))), c: "#4ade80" },
                        { l: "LARGEST LOSS DAY", v: fmtPnl(Math.min(...me.map(e => netPnl(e)))), c: "#f87171" },
                        { l: "MAX DRAWDOWN", v: a.maxDD > 0 ? `-$${a.maxDD.toFixed(2)}` : "—", c: "#f87171" },
                      ] : []),
                    ].map(s => (
                      <div key={s.l} style={{ background: s.hi ? "#0d1f3c" : "#0f1729", border: `1px solid ${s.hi ? "#1e3a5f" : "#1e293b"}`, borderRadius: 4, padding: "10px 12px" }}>
                        <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 4 }}>{s.l}</div>
                        <div style={{ fontSize: 16, color: s.c, fontWeight: s.hi ? 700 : 500 }}>{s.v}</div>
                      </div>
                    ))}
          </div>
          )}
              </div>

              {/* Trade Statistics */}
              {a && (
                <div>
                  <MonthSectionHeader label="TRADE STATISTICS" monthKey={expandedMonth} skey="trades" summary={<span>{a.total} trades · <span style={{ color: a.winRate >= 50 ? "#4ade80" : "#f87171" }}>{a.winRate.toFixed(0)}% WR</span></span>} />
                  {!isCollapsed(expandedMonth, "trades") && (
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
                      {[
                        { l: "TOTAL TRADES", v: a.total, c: "#e2e8f0" },
                        { l: "WINNERS", v: a.winners, c: "#4ade80" },
                        { l: "LOSERS", v: a.losers, c: "#f87171" },
                        { l: "WIN RATE", v: `${a.winRate.toFixed(1)}%`, c: a.winRate >= 50 ? "#4ade80" : "#f87171" },
                        { l: "PROFIT FACTOR", v: fmtPF(a.profitFactor), c: pfColor(a.profitFactor) },
                        { l: "AVG CONTRACT SIZE", v: `${a.avgQty.toFixed(1)} cts`, c: "#e2e8f0" },
                        { l: "AVG WIN", v: fmtPnl(a.avgWin), c: "#4ade80" },
                        { l: "AVG LOSS", v: fmtPnl(a.avgLoss), c: "#f87171" },
                        { l: "MAX WIN STREAK", v: `${a.maxConsecWins} trades`, c: "#4ade80" },
                        { l: "MAX LOSS STREAK", v: `${a.maxConsecLoss} trades`, c: "#f87171" },
                        { l: "LARGEST WIN", v: fmtPnl(a.largestWin), c: "#4ade80" },
                        { l: "LARGEST LOSS", v: fmtPnl(a.largestLoss), c: "#f87171" },
                      ].map(s => (
                        <div key={s.l} style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "10px 12px" }}>
                          <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 4 }}>{s.l}</div>
                          <div style={{ fontSize: 16, color: s.c, fontWeight: 500 }}>{s.v}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Long vs Short */}
              {a?.byDirection && (a.byDirection.long.trades > 0 || a.byDirection.short.trades > 0) && (() => {
                const dirs = [
                  { key: "long",  label: "Long",  emoji: "📈", accentColor: "#4ade80", dimColor: "#166534", bgColor: "rgba(16,63,33,0.35)" },
                  { key: "short", label: "Short", emoji: "📉", accentColor: "#f87171", dimColor: "#7f1d1d", bgColor: "rgba(63,16,16,0.35)" },
                ].filter(d => a.byDirection[d.key].trades > 0);
                const totalTrades = (a.byDirection.long.trades || 0) + (a.byDirection.short.trades || 0);
                const maxPnl = Math.max(Math.abs(a.byDirection.long.pnl), Math.abs(a.byDirection.short.pnl), 1);
                return (
                  <div>
                    <MonthSectionHeader label="LONG vs SHORT" monthKey={expandedMonth} skey="direction"
                      summary={<span style={{ color: "#64748b" }}>{totalTrades} trades · {dirs.length === 2 ? `${Math.round(a.byDirection.long.trades / totalTrades * 100)}% long` : dirs[0]?.label + " only"}</span>} />
                    {!isCollapsed(expandedMonth, "direction") && (
                      <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px" }}>
                        <div style={{ display: "grid", gridTemplateColumns: dirs.length === 2 ? "1fr 1fr" : "1fr", gap: 10, marginBottom: dirs.length === 2 ? 14 : 0 }}>
                          {dirs.map(({ key, label, emoji, accentColor, dimColor, bgColor }) => {
                            const d = a.byDirection[key];
                            const wr = Math.round(d.wins / d.trades * 100);
                            const avg = d.pnl / d.trades;
                            const pf = d.grossLoss > 0 ? (d.grossWin / d.grossLoss) : d.grossWin > 0 ? Infinity : null;
                            const sharePct = Math.round(d.trades / totalTrades * 100);
                            return (
                              <div key={key} style={{ background: bgColor, border: `1px solid ${dimColor}`, borderRadius: 5, padding: "12px 14px", position: "relative", overflow: "hidden" }}>
                                <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: accentColor, opacity: 0.5 }} />
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                    <span style={{ fontSize: 14 }}>{emoji}</span>
                                    <span style={{ fontSize: 12, color: accentColor, fontWeight: 600, letterSpacing: "0.08em" }}>{label.toUpperCase()}</span>
                                  </div>
                                  <span style={{ fontSize: 10, color: "#475569" }}>{sharePct}% of trades</span>
                                </div>
                                <div style={{ fontSize: 22, fontWeight: 700, color: d.pnl >= 0 ? "#4ade80" : "#f87171", lineHeight: 1, marginBottom: 8 }}>
                                  {d.pnl >= 0 ? "+" : "-"}${Math.abs(d.pnl).toFixed(0)}
                                </div>
                                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6 }}>
                                  <div style={{ background: "#0a0e1a", borderRadius: 3, padding: "6px 8px" }}>
                                    <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 2 }}>WIN RATE</div>
                                    <div style={{ fontSize: 13, color: wr >= 50 ? "#4ade80" : "#f87171", fontWeight: 600 }}>{wr}%</div>
                                    <div style={{ fontSize: 9, color: "#334155", marginTop: 1 }}><span style={{ color: "#4ade80" }}>{d.wins}</span>W <span style={{ color: "#f87171" }}>{d.losses}</span>L</div>
                                  </div>
                                  <div style={{ background: "#0a0e1a", borderRadius: 3, padding: "6px 8px" }}>
                                    <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 2 }}>AVG/TRADE</div>
                                    <div style={{ fontSize: 13, color: avg >= 0 ? "#4ade80" : "#f87171", fontWeight: 600 }}>{avg >= 0 ? "+" : "-"}${Math.abs(avg).toFixed(0)}</div>
                                    <div style={{ fontSize: 9, color: "#334155", marginTop: 1 }}>{d.trades} trades</div>
                                  </div>
                                  <div style={{ background: "#0a0e1a", borderRadius: 3, padding: "6px 8px" }}>
                                    <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 2 }}>PROF. FACTOR</div>
                                    <div style={{ fontSize: 13, color: pfColor(pf), fontWeight: 600 }}>{fmtPF(pf)}</div>
                                    <div style={{ fontSize: 9, color: "#334155", marginTop: 1 }}>+${d.grossWin.toFixed(0)} / -${d.grossLoss.toFixed(0)}</div>
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                        {dirs.length === 2 && (() => {
                          const l = a.byDirection.long;
                          const s = a.byDirection.short;
                          const lBarW = Math.abs(l.pnl) / maxPnl * 100;
                          const sBarW = Math.abs(s.pnl) / maxPnl * 100;
                          const bestDir = l.pnl >= s.pnl ? "Long" : "Short";
                          return (
                            <div style={{ background: "#0a0e1a", borderRadius: 4, padding: "10px 12px" }}>
                              <div style={{ marginBottom: 8 }}>
                                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3, fontSize: 9, color: "#475569" }}>
                                  <span>📈 Long</span><span style={{ color: l.pnl >= 0 ? "#4ade80" : "#f87171" }}>{l.pnl >= 0 ? "+" : "-"}${Math.abs(l.pnl).toFixed(0)}</span>
                                </div>
                                <div style={{ background: "#0f1729", borderRadius: 2, height: 5, overflow: "hidden" }}>
                                  <div style={{ width: `${lBarW}%`, height: "100%", background: l.pnl >= 0 ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} />
                                </div>
                              </div>
                              <div>
                                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3, fontSize: 9, color: "#475569" }}>
                                  <span>📉 Short</span><span style={{ color: s.pnl >= 0 ? "#4ade80" : "#f87171" }}>{s.pnl >= 0 ? "+" : "-"}${Math.abs(s.pnl).toFixed(0)}</span>
                                </div>
                                <div style={{ background: "#0f1729", borderRadius: 2, height: 5, overflow: "hidden" }}>
                                  <div style={{ width: `${sBarW}%`, height: "100%", background: s.pnl >= 0 ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} />
                                </div>
                              </div>
                              <div style={{ marginTop: 8, fontSize: 10, color: "#64748b", lineHeight: 1.6 }}>
                                <span style={{ color: "#3b82f6" }}>💡 </span>
                                Stronger edge on <span style={{ color: bestDir === "Long" ? "#4ade80" : "#f87171" }}>{bestDir}</span> trades
                                {" · "}{Math.round(l.trades / totalTrades * 100)}% long / {Math.round(s.trades / totalTrades * 100)}% short split
                              </div>
                            </div>
                          );
                        })()}
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* Performance by Symbol */}
              {a && Object.keys(a.bySymbol).length > 0 && (
                <div>
                  <MonthSectionHeader label="PERFORMANCE BY SYMBOL" monthKey={expandedMonth} skey="symbol" summary={<span style={{ color: "#64748b" }}>{Object.keys(a.bySymbol).join(", ")}</span>} />
                  {!isCollapsed(expandedMonth, "symbol") && (
                    <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px", display: "flex", flexDirection: "column", gap: 12 }}>
                      {Object.entries(a.bySymbol).map(([sym, s]) => {
                        const maxPnl = Math.max(...Object.values(a.bySymbol).map(x => Math.abs(x.pnl)), 1);
                        const wr = Math.round(s.wins / s.trades * 100);
                        const barW = Math.abs(s.pnl) / maxPnl * 100;
                        return (
                          <div key={sym}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 5 }}>
                              <span style={{ fontSize: 11, color: "#93c5fd", fontWeight: 500 }}>{sym}</span>
                              <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
                                <span style={{ fontSize: 10, color: wr >= 50 ? "#4ade80" : "#f87171" }}>{wr}% WR</span>
                                <span style={{ fontSize: 10, letterSpacing: 0 }}><span style={{ color: "#4ade80" }}>{s.wins}</span><span style={{ color: "#94a3b8" }}>/</span><span style={{ color: "#f87171" }}>{s.trades - s.wins}</span></span>
                                <span style={{ fontSize: 12, fontWeight: 600, color: s.pnl >= 0 ? "#4ade80" : "#f87171", minWidth: 80, textAlign: "right" }}>{s.pnl >= 0 ? "+" : "-"}${Math.abs(s.pnl).toFixed(2)}</span>
                              </div>
                            </div>
                            <div style={{ background: "#0a0e1a", borderRadius: 2, height: 5, overflow: "hidden" }}>
                              <div style={{ width: `${barW}%`, height: "100%", background: s.pnl >= 0 ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}

                {/* Behavioral Edge Checks — expanded month */}
                {a && (a.afterLoss?.total > 0 || a.afterWin?.total > 0 || a.first3?.total > 0) && (() => (
                  <div>
                    <MonthSectionHeader label="BEHAVIORAL EDGE CHECKS" monthKey={expandedMonth} skey="behavioral"
                      summary={<span style={{ color: "#64748b" }}>after loss · after win · first 3</span>} />
                    {!isCollapsed(expandedMonth, "behavioral") && (
                      <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px" }}>
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 10 }}>
                          {[
                            { label: "AFTER A LOSS",   data: a.afterLoss },
                            { label: "AFTER A WIN",    data: a.afterWin  },
                            { label: "FIRST 3 TRADES", data: a.first3    },
                            { label: "REST OF SESSION", data: a.rest     },
                          ].map(card => (
                            <div key={card.label} style={{ background: "#0a0e1a", border: "1px solid #1e293b", borderRadius: 4, padding: "10px 12px" }}>
                              <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 6 }}>{card.label}</div>
                              {card.data?.total ? (
                                <>
                                  <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
                                    <div style={{ fontSize: 13, color: "#e2e8f0" }}>{card.data.winRate.toFixed(0)}% WR</div>
                                    <div style={{ fontSize: 13, fontWeight: 600, color: (card.label === "AFTER A LOSS" || card.label === "AFTER A WIN") ? (card.data.avgPnl >= 0 ? "#4ade80" : "#f87171") : (card.data.pnl >= 0 ? "#4ade80" : "#f87171") }}>
                                      {card.label === "AFTER A LOSS" || card.label === "AFTER A WIN"
                                        ? `${card.data.avgPnl >= 0 ? "+" : "-"}$${Math.abs(card.data.avgPnl).toFixed(0)}/trade`
                                        : `${card.data.pnl >= 0 ? "+" : "-"}$${Math.abs(card.data.pnl).toFixed(0)}`}
                                    </div>
                                  </div>
                                  <div style={{ fontSize: 10, color: "#64748b", marginTop: 6 }}>n={card.data.total} · avg {card.data.avgQty.toFixed(1)} cts</div>
                                </>
                              ) : <div style={{ fontSize: 12, color: "#64748b" }}>Not enough data</div>}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))()}

                {/* Performance by Holding Time — expanded month */}
                {a?.byDuration && Object.values(a.byDuration).some(b => b.trades > 0) && (
                  <div>
                    <MonthSectionHeader label="PERFORMANCE BY HOLDING TIME" monthKey={expandedMonth} skey="holdingtime"
                      summary={<span style={{ color: "#64748b" }}>{Object.values(a.byDuration).filter(b => b.trades > 0).length} buckets</span>} />
                    {!isCollapsed(expandedMonth, "holdingtime") && (
                      <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px", display: "flex", flexDirection: "column", gap: 12 }}>
                        {(a.DURATION_BUCKETS || []).map(bucket => {
                          const b = a.byDuration[bucket.key];
                          if (!b || b.trades === 0) return null;
                          const wr = Math.round(b.wins / b.trades * 100);
                          const maxPnl = Math.max(...(a.DURATION_BUCKETS || []).map(bk => Math.abs(a.byDuration[bk.key]?.pnl || 0)), 1);
                          const barW = Math.abs(b.pnl) / maxPnl * 100;
                          const avg = b.pnl / b.trades;
                          const fmtSecs = s => !s ? "—" : s < 60 ? `${Math.round(s)}s` : s < 3600 ? `${Math.floor(s/60)}m ${Math.round(s%60)}s` : `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`;
                          return (
                            <div key={bucket.key}>
                              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 5 }}>
                                <div>
                                  <span style={{ fontSize: 11, color: "#e2e8f0", fontWeight: 500 }}>{bucket.label}</span>
                                  <span style={{ fontSize: 9, color: "#64748b", marginLeft: 8 }}>avg {fmtSecs(b.avgSecs)}</span>
                                </div>
                                <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
                                  <span style={{ fontSize: 10, color: wr >= 50 ? "#4ade80" : "#f87171" }}>{wr}% WR</span>
                                  <span style={{ fontSize: 10 }}><span style={{ color: "#4ade80" }}>{b.wins}</span><span style={{ color: "#94a3b8" }}>/</span><span style={{ color: "#f87171" }}>{b.losses}</span></span>
                                  <span style={{ fontSize: 12, fontWeight: 600, color: b.pnl >= 0 ? "#4ade80" : "#f87171", minWidth: 80, textAlign: "right" }}>{avg >= 0 ? "+" : "-"}${Math.abs(avg).toFixed(0)} avg</span>
                                </div>
                              </div>
                              <div style={{ background: "#0a0e1a", borderRadius: 2, height: 5, overflow: "hidden" }}>
                                <div style={{ width: `${barW}%`, height: "100%", background: b.pnl >= 0 ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}

                {/* Mistake Frequency + Mood vs Performance — side by side */}
                {(() => {
                  const mistakeCounts = {};
                  let cleanDays = 0;
                  for (const e of me) {
                    if (e.sessionMistakes?.includes("No Mistakes — Executed the Plan ✓")) cleanDays++;
                    for (const m of (e.sessionMistakes || [])) {
                      if (m === "No Mistakes — Executed the Plan ✓") continue;
                      mistakeCounts[m] = (mistakeCounts[m] || 0) + 1;
                    }
                  }
                  const mistakeSorted = Object.entries(mistakeCounts).sort((a, b) => b[1] - a[1]);
                  const maxCount = mistakeSorted[0]?.[1] || 1;
                  const hasMistakes = mistakeSorted.length > 0 || cleanDays > 0;

                  const moodStats = {};
                  for (const e of me) {
                    const moods = e.moods?.length ? e.moods : e.mood ? [e.mood] : [];
                    const ep = netPnl(e);
                    for (const m of moods) {
                      if (!moodStats[m]) moodStats[m] = { pnl: 0, sessions: 0, wins: 0 };
                      moodStats[m].pnl += ep;
                      moodStats[m].sessions++;
                      if (ep > 0) moodStats[m].wins++;
                    }
                  }
                  const moodList = Object.entries(moodStats).sort((a, b) => (b[1].pnl / b[1].sessions) - (a[1].pnl / a[1].sessions));
                  const maxAvg = Math.max(...moodList.map(([, s]) => Math.abs(s.pnl / s.sessions)), 1);
                  const hasMoods = moodList.length > 0;
                  if (!hasMistakes && !hasMoods) return null;
                  return (
                    <div style={{ display: "flex", flexDirection: "row", gap: 16, alignItems: "stretch" }}>
                      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
                        <MonthSectionHeader label="MISTAKE FREQUENCY" monthKey={expandedMonth} skey="mistakes"
                          summary={<span style={{ color: "#64748b" }}>{mistakeSorted.length} type{mistakeSorted.length !== 1 ? "s" : ""} · {cleanDays} clean</span>} />
                        {!isCollapsed(expandedMonth, "mistakes") && (
                          <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px", flex: 1, display: "flex", flexDirection: "column" }}>
                            {cleanDays > 0 && (
                              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: mistakeSorted.length ? 10 : 0, padding: "6px 10px", background: "rgba(16,63,33,0.55)", border: "1px solid #166534", borderRadius: 4 }}>
                                <span style={{ fontSize: 11, color: "#4ade80" }}>✓ Executed the Plan</span>
                                <span style={{ fontSize: 12, fontWeight: 600, color: "#4ade80" }}>{cleanDays} day{cleanDays !== 1 ? "s" : ""}</span>
                              </div>
                            )}
                            {mistakeSorted.map(([mistake, count]) => (
                              <div key={mistake} style={{ marginBottom: 10 }}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
                                  <span style={{ fontSize: 11, color: "#e2e8f0" }}>{mistake}</span>
                                  <div style={{ display: "flex", gap: 10, alignItems: "center", flexShrink: 0, marginLeft: 8 }}>
                                    <span style={{ fontSize: 9, color: "#94a3b8" }}>{Math.round(count / me.length * 100)}%</span>
                                    <span style={{ fontSize: 12, fontWeight: 600, color: "#f87171" }}>{count}×</span>
                                  </div>
                                </div>
                                <div style={{ background: "#0a0e1a", borderRadius: 2, height: 4, overflow: "hidden" }}>
                                  <div style={{ width: `${(count / maxCount) * 100}%`, height: "100%", background: "#f87171", borderRadius: 2, opacity: 0.7 }} />
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
                        <MonthSectionHeader label="MOOD VS PERFORMANCE" monthKey={expandedMonth} skey="mood"
                          summary={<span style={{ color: "#64748b" }}>{moodList.length} mood{moodList.length !== 1 ? "s" : ""}</span>} />
                        {!isCollapsed(expandedMonth, "mood") && (
                          <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px", display: "flex", flexDirection: "column", gap: 10, flex: 1 }}>
                            {moodList.map(([mood, s]) => {
                              const avg = s.pnl / s.sessions;
                              const wr = Math.round((s.wins / s.sessions) * 100);
                              const isPos = avg >= 0;
                              return (
                                <div key={mood}>
                                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
                                    <span style={{ fontSize: 11, color: "#e2e8f0" }}>{mood}</span>
                                    <div style={{ display: "flex", gap: 10, alignItems: "center", flexShrink: 0, marginLeft: 8 }}>
                                      <span style={{ fontSize: 9, color: "#94a3b8" }}>{s.sessions}× · {wr}% WR</span>
                                      <span style={{ fontSize: 11, fontWeight: 600, color: isPos ? "#4ade80" : "#f87171", minWidth: 62, textAlign: "right" }}>{avg >= 0 ? "+" : "-"}${Math.abs(avg).toFixed(0)} avg</span>
                                    </div>
                                  </div>
                                  <div style={{ background: "#0a0e1a", borderRadius: 2, height: 4, overflow: "hidden" }}>
                                    <div style={{ width: `${Math.abs(avg) / maxAvg * 100}%`, height: "100%", background: isPos ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} />
                                  </div>
                                </div>
                              );
                            })}
                            {moodList.length >= 2 && (() => {
                              const best = moodList[0];
                              const worst = moodList[moodList.length - 1];
                              return (
                                <div style={{ marginTop: 4, padding: "8px 10px", background: "#0a1628", border: "1px solid #1e3a5f", borderRadius: 4, fontSize: 10, color: "#94a3b8", lineHeight: 1.7 }}>
                                  <span style={{ color: "#3b82f6" }}>💡 </span>
                                  Best: <span style={{ color: "#4ade80" }}>{best[0]}</span> (+${(best[1].pnl / best[1].sessions).toFixed(0)} avg).
                                  {(worst[1].pnl / worst[1].sessions) < 0 && <> Watch: <span style={{ color: "#f87171" }}>{worst[0]}</span>.</>}
                                </div>
                              );
                            })()}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })()}


            </div>
          </div>
        );
      })()}
    </div>
  );
}

function AIRecapView({ entries, netPnl: calcNetPnlProp, fmtPnl, pnlColor, initMode = "weekly", ai, activeJournal, propStatus }) {
  const calcNetPnl = calcNetPnlProp;
  const journalId  = activeJournal?.id || "default";
  const RECAP_KEY  = `ai-recaps-v2-${journalId}`;

  const [recapMode, setRecapMode]         = useState(initMode);
  const [selectedPeriod, setSelectedPeriod] = useState(null);
  const [summary, setSummary]             = useState("");
  const [loading, setLoading]             = useState(false);
  const [generated, setGenerated]         = useState({});
  const [savedAt, setSavedAt]             = useState({});   // { [period]: ISO string }
  const [savedHashes, setSavedHashes]     = useState({});   // { [period]: notesHash string } — for staleness detection
  const [taglineId, setTaglineId]         = useState(() => loadTagline());
  const [showTaglinePicker, setShowTaglinePicker] = useState(false);
  const [storageReady, setStorageReady]   = useState(false);

  // ── Load persisted recaps from storage on mount ────────────────────────────
  useEffect(() => {
    setStorageReady(false);
    setGenerated({});
    setSavedAt({});
    setSavedHashes({});
    (async () => {
      try {
        const r = await storage.get(RECAP_KEY);
        if (r?.value) {
          const parsed = JSON.parse(r.value);
          setGenerated(parsed.texts  || {});
          setSavedAt(parsed.meta     || {});
          setSavedHashes(parsed.hashes || {});
        }
      } catch {}
      setStorageReady(true);
    })();
  }, [journalId]);

  // ── Persist recaps to storage whenever generated changes (after initial load) ─
  useEffect(() => {
    if (!storageReady) return;
    const keys = Object.keys(generated);
    if (keys.length === 0) return;
    storage.set(RECAP_KEY, JSON.stringify({ texts: generated, meta: savedAt, hashes: savedHashes })).catch(() => {});
  }, [generated, storageReady]);

  // Group entries into ISO weeks (Mon–Sun)
  // ISO 8601 week key — must match WeeklyPerformance's getISOWeek exactly
  const getWeekKey = (dateStr) => {
    const d = new Date(dateStr + "T12:00:00");
    const jan4 = new Date(d.getFullYear(), 0, 4);
    const startOfWeek1 = new Date(jan4);
    startOfWeek1.setDate(jan4.getDate() - ((jan4.getDay() + 6) % 7));
    const diff = d - startOfWeek1;
    const week = Math.floor(diff / (7 * 24 * 60 * 60 * 1000)) + 1;
    return `${d.getFullYear()}-W${String(week).padStart(2, "0")}`;
  };

  const getWeekLabel = (dateStr) => {
    const d = new Date(dateStr + "T12:00:00");
    const day = d.getDay();
    const diff = d.getDate() - day + (day === 0 ? -6 : 1);
    const mon = new Date(new Date(dateStr + "T12:00:00").setDate(diff));
    const fri = new Date(mon); fri.setDate(mon.getDate() + 4);
    const fmt = dt => dt.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    return `${fmt(mon)} – ${fmt(fri)}, ${mon.getFullYear()}`;
  };

  // Group by week
  const byWeek = {};
  for (const e of entries) {
    if (!e.date) continue;
    const wk = getWeekKey(e.date);
    if (!byWeek[wk]) byWeek[wk] = [];
    byWeek[wk].push(e);
  }
  const weeks = Object.keys(byWeek).sort().reverse();

  // Group by month
  const byMonth = {};
  for (const e of entries) {
    if (!e.date) continue;
    const mo = e.date.slice(0, 7);
    if (!byMonth[mo]) byMonth[mo] = [];
    byMonth[mo].push(e);
  }
  const monthsList = Object.keys(byMonth).sort().reverse();

  const periods = recapMode === "weekly" ? weeks : monthsList;
  const grouped = recapMode === "weekly" ? byWeek : byMonth;

  const periodLabel = (p) => {
    if (recapMode === "monthly") {
      const [y, m] = p.split("-");
      return new Date(+y, +m - 1, 1).toLocaleString("default", { month: "long", year: "numeric" });
    }
    const sample = grouped[p]?.[0]?.date;
    return sample ? getWeekLabel(sample) : p;
  };

  const buildPrompt = (periodEntries, label, period = null) => {
    const sorted = [...periodEntries].sort((a,b) => a.date.localeCompare(b.date));

    // ── Aggregate stats ──────────────────────────────────────────────────────
    const totalPnl = sorted.reduce((s,e) => s + netPnl(e), 0);
    const winDays  = sorted.filter(e => netPnl(e) > 0).length;
    const lossDays = sorted.filter(e => netPnl(e) < 0).length;
    const allTrades   = sorted.flatMap(e => e.parsedTrades || []);
    const tNR = (t) => t.pnl - (t.commission||0);        // net per trade — single source of truth
    const allWinners  = allTrades.filter(t => tNR(t) > 0);
    const allLosers   = allTrades.filter(t => tNR(t) < 0);
    const overallWR   = allTrades.length ? ((allWinners.length/allTrades.length)*100).toFixed(1) : "0";
    const overallPF   = allLosers.length
      ? allWinners.reduce((s,t)=>s+tNR(t),0)/Math.abs(allLosers.reduce((s,t)=>s+tNR(t),0))
      : allWinners.length > 0 ? Infinity : null;
    const grossPnl  = allTrades.reduce((s,t) => s + t.pnl, 0);
    const totalFees = allTrades.reduce((s,t)=>s+(t.commission||0),0);
    const commDragPct = Math.abs(grossPnl) > 0 ? (totalFees / Math.abs(grossPnl) * 100).toFixed(1) : "0";
    const avgWin  = allWinners.length ? (allWinners.reduce((s,t)=>s+tNR(t),0)/allWinners.length).toFixed(0) : "0";
    const avgLoss = allLosers.length  ? Math.abs(allLosers.reduce((s,t)=>s+tNR(t),0)/allLosers.length).toFixed(0) : "0";

    // Direction breakdown (long vs short) — key edge signal
    const byDir = { long: { t:0, w:0, pnl:0, comm:0 }, short: { t:0, w:0, pnl:0, comm:0 } };
    for (const t of allTrades) {
      const d = byDir[t.direction === "short" ? "short" : "long"];
      const nt = tNR(t);
      d.t++; d.pnl += nt; d.comm += (t.commission||0);
      if (nt > 0) d.w++;
    }
    const dirLine = ["long","short"].filter(k=>byDir[k].t>0).map(k => {
      const d = byDir[k];
      const wr = d.t ? Math.round(d.w/d.t*100) : 0;
      const sign = d.pnl >= 0 ? "+" : "";
      return `${k.toUpperCase()}: ${d.t} trades ${wr}% WR net ${sign}$${d.pnl.toFixed(0)}`;
    }).join(" | ") || "none";

    // Session breakdown — proper ET hour buckets matching journal getSession logic
    const getSessET = t => {
      const ts = t.direction === "short" ? (t.buyTime || t.sellTime) : (t.sellTime || t.buyTime);
      if (!ts) return "Unknown";
      const m = ts.match(/^\d{8}\s(\d{2})(\d{2})/);
      if (!m) return "Unknown";
      const h = parseInt(m[1]) + parseInt(m[2]) / 60;
      if (h >= 20) return "Asian";
      if (h < 4)   return "London Overnight";
      if (h < 9.5) return "Pre-Market";
      if (h < 12)  return "NY Morning";
      if (h < 16)  return "NY Afternoon";
      return "After Hours";
    };
    const sessMap = {};
    for (const t of allTrades) {
      const k = getSessET(t);
      if (!sessMap[k]) sessMap[k] = { pnl:0, t:0, w:0, comm:0 };
      const nt = tNR(t);
      sessMap[k].pnl  += nt;
      sessMap[k].t++;
      sessMap[k].comm += (t.commission||0);
      if (nt > 0) sessMap[k].w++;
    }
    const sessOrder = ["Asian","London Overnight","Pre-Market","NY Morning","NY Afternoon","After Hours","Unknown"];
    const sessLine = sessOrder.filter(k=>sessMap[k]).map(k => {
      const d = sessMap[k];
      const wr = Math.round(d.w/d.t*100);
      const sign = d.pnl >= 0 ? "+" : "";
      return `${k}: ${d.t}t ${wr}%WR net ${sign}$${d.pnl.toFixed(0)}`;
    }).join(" | ") || "none";
    const sessEntries = Object.entries(sessMap).filter(([,d])=>d.t>0);
    const bestSess  = sessEntries.length ? sessEntries.reduce((a,b) => b[1].pnl > a[1].pnl ? b : a) : null;
    const worstSess = sessEntries.length ? sessEntries.reduce((a,b) => b[1].pnl < a[1].pnl ? b : a) : null;

    // Carry-forward count
    const carryCount = allTrades.filter(t=>t.notes==="overnight-carry").length;

    // Mistake frequency
    const mCounts = {};
    for (const e of sorted) for (const m of (e.sessionMistakes||[])) if(m!=="No Mistakes — Executed the Plan ✓") mCounts[m]=(mCounts[m]||0)+1;
    const cleanDays = sorted.filter(e=>e.sessionMistakes?.includes("No Mistakes — Executed the Plan ✓")).length;
    const mistakeLine = Object.entries(mCounts).sort((a,b)=>b[1]-a[1]).map(([m,n])=>`"${m}":${n}x`).join(", ")||"none";

    const grades = sorted.filter(e=>e.grade).map(e=>`${e.date.slice(5)}:${e.grade}`).join(", ")||"none";

    // ── Improvement #1: Best/worst day explicit ──────────────────────────────
    const dayNets = sorted.map(e => ({ date: e.date, net: netPnl(e) }));
    const bestDay  = dayNets.length ? dayNets.reduce((a,b) => b.net > a.net ? b : a) : null;
    const worstDay = dayNets.length ? dayNets.reduce((a,b) => b.net < a.net ? b : a) : null;

    // ── Improvement #2: Avg win/loss ratio ───────────────────────────────────
    const rrRatioLine = allLosers.length && allWinners.length
      ? `W:L ratio: ${(allWinners.reduce((s,t)=>s+tNR(t),0)/allWinners.length / Math.abs(allLosers.reduce((s,t)=>s+tNR(t),0)/allLosers.length)).toFixed(2)}x`
      : "";

    // ── Improvement #3: Equity curve shape (first-half vs second-half) ───────
    const halfIdx = Math.floor(sorted.length / 2);
    const firstHalfNet = sorted.slice(0, halfIdx).reduce((s,e) => s + netPnl(e), 0);
    const secondHalfNet = sorted.slice(halfIdx).reduce((s,e) => s + netPnl(e), 0);
    const curveTrend = secondHalfNet > firstHalfNet ? "improving" : secondHalfNet < firstHalfNet ? "fading" : "flat";
    const curveDesc = sorted.length >= 2
      ? `First half: $${firstHalfNet.toFixed(0)} | Second half: $${secondHalfNet.toFixed(0)} | Trend: ${curveTrend.toUpperCase()}`
      : "Not enough days for curve analysis";

    // ── Improvement #4: Clean execution days — flag for special treatment ────
    const cleanExecDays = sorted.filter(e =>
      e.sessionMistakes?.includes("No Mistakes — Executed the Plan ✓") && netPnl(e) > 0
    );

    // ── Per-day summary with per-trade lines ──────────────────────────────────
    const dayBlocks = sorted.map(e => {
      const trades = e.parsedTrades || [];
      const dWins  = trades.filter(t=>(t.pnl-(t.commission||0))>0).length;
      const dGross = trades.reduce((s,t)=>s+t.pnl,0);
      const dComm  = trades.reduce((s,t)=>s+(t.commission||0),0);
      const dNet   = netPnl(e).toFixed(0);
      const moods  = (e.moods?.length ? e.moods : e.mood ? [e.mood] : []).join(", ");
      const isClean = e.sessionMistakes?.includes("No Mistakes — Executed the Plan ✓");
      // Prefix each trade with its date so the AI can place trades in context
      const tradeLines = trades.map((t,i) => {
        const net = (t.pnl-(t.commission||0)).toFixed(2);
        const exitT = t.direction==="short" ? (t.buyTime||t.sellTime) : (t.sellTime||t.buyTime);
        const exitHHMM = exitT ? exitT.replace(/^\d{8}\s/,"").slice(0,4) : "?";
        const carry = t.notes==="overnight-carry" ? " [carry-forward]" : "";
        return `    ${e.date} T${i+1} ${(t.direction||"long").toUpperCase()} @${exitHHMM} gross $${t.pnl.toFixed(2)} net $${net}${carry}`;
      }).join("\n");
      const cleanFlag = isClean ? "  ✓ CLEAN EXECUTION DAY — executed the plan" : "";
      const lines = [`[${e.date}] Net:$${dNet} Gross:$${dGross.toFixed(0)} Fees:-$${dComm.toFixed(0)} | ${dWins}W/${trades.length-dWins}L | Grade:${e.grade||"?"} | Mood:${moods||"?"}`];
      if (cleanFlag) lines.push(cleanFlag);
      if (tradeLines) lines.push(tradeLines);
      if (!isClean && e.sessionMistakes?.length) lines.push(`  Mistakes: ${e.sessionMistakes.filter(m=>m!=="No Mistakes — Executed the Plan ✓").join(" | ")||"none"}`);
      const noteFields = [
        ["Market notes",  e.marketNotes],  ["Rules",        e.rules],
        ["Lessons",       e.lessonsLearned],["Mistakes note",e.mistakes],
        ["Improvements",  e.improvements],  ["Best trade",   e.bestTrade],
        ["Worst trade",   e.worstTrade],    ["Reinforce",    e.reinforceRule],
        ["Tomorrow plan", e.tomorrow],
      ];
      for (const [lbl, val] of noteFields) {
        if (val?.trim()) lines.push(`  ${lbl}: ${val.trim()}`);
      }
      return lines.join("\n");
    }).join("\n\n");

    // ── Plan vs actual ────────────────────────────────────────────────────────
    const planLines = [];
    for (let i=1;i<sorted.length;i++) {
      const prev=sorted[i-1], curr=sorted[i];
      if (prev.tomorrow?.trim()) {
        const currClean = curr.sessionMistakes?.includes("No Mistakes — Executed the Plan ✓");
        const nextDayNotes = [
          curr.marketNotes, curr.lessonsLearned, curr.mistakes,
          curr.rules, curr.bestTrade, curr.worstTrade
        ].filter(Boolean).map(n => n.trim()).join(' | ').slice(0, 300);
        const mistakeTags = (curr.sessionMistakes||[]).filter(m=>m!=="No Mistakes — Executed the Plan ✓").join(", ")||"none";
        planLines.push(
          `  ${prev.date} plan: "${prev.tomorrow.slice(0,200)}"\n` +
          `  ${curr.date} outcome: grade ${curr.grade||"?"} | net $${netPnl(curr).toFixed(0)}${currClean ? " | CLEAN EXECUTION DAY" : ""}\n` +
          `  ${curr.date} notes: ${nextDayNotes || "(none written)"}\n` +
          `  ${curr.date} tagged: ${mistakeTags}`
        );
      }
    }

    // ── Improvement #5: Prior period action plan (continuity) ─────────────────
    const priorPeriodKey = (() => {
      if (!recapMode || periods.length < 2) return null;
      const currentIdx = periods.indexOf(period);
      if (currentIdx === -1) return null;
      return periods[currentIdx + 1] || null; // periods sorted descending, so +1 = prior
    })();
    const priorRecapText = priorPeriodKey ? generated[priorPeriodKey] : null;
    const priorActionPlan = (() => {
      if (!priorRecapText) return null;
      // Extract ACTION PLAN section using string split — avoids regex literal newline issue
      const marker = priorRecapText.split('\n').findIndex(l => l.includes('ACTION PLAN'));
      if (marker === -1) return null;
      const lines = priorRecapText.split('\n').slice(marker + 1);
      const endIdx = lines.findIndex((l, i) => i > 0 && l.startsWith('**'));
      const planLines = endIdx === -1 ? lines : lines.slice(0, endIdx);
      return planLines.join('\n').trim().slice(0, 800) || null;
    })();

    const isMonthly = recapMode === "monthly";

    return `You are a professional futures trading coach. A trader has shared their complete journal for: ${label}

Your job is to write a thorough, honest coaching review. Be direct and specific — cite exact dates, dollar amounts, and quote the trader\'s own words. Be balanced: name what\'s working as clearly as what needs fixing.

PERIOD OVERVIEW
Days: ${sorted.length} | ${winDays}W green / ${lossDays}L red | Gross: $${grossPnl.toFixed(0)} | Fees: -$${totalFees.toFixed(0)} | Net: $${totalPnl.toFixed(0)} | Avg/day net: $${(totalPnl/sorted.length).toFixed(0)}
Trades: ${allTrades.length} | WR: ${overallWR}% | PF: ${fmtPF(overallPF)} | Avg win: +$${avgWin} | Avg loss: -$${avgLoss} | ${rrRatioLine}
Commission drag: $${totalFees.toFixed(0)} = ${commDragPct}% of gross P&L${parseFloat(commDragPct)>30?" \u26a0 HIGH":""}
${carryCount > 0 ? `Overnight carry-forwards: ${carryCount} trade(s) — opened prior session, closed next\n` : ""}DIRECTION: ${dirLine}
SESSION (by exit time ET): ${sessLine}
${bestSess ? `Best session: ${bestSess[0]} net $${(bestSess[1].pnl-bestSess[1].comm).toFixed(0)}` : ""}${worstSess && worstSess[0]!==bestSess?.[0] ? ` | Worst: ${worstSess[0]} net $${(worstSess[1].pnl-worstSess[1].comm).toFixed(0)}` : ""}
${bestDay ? `Best day: ${bestDay.date} net $${bestDay.net.toFixed(0)} | Worst day: ${worstDay.date} net $${worstDay.net.toFixed(0)}` : ""}
EQUITY CURVE SHAPE: ${curveDesc}
${cleanExecDays.length > 0 ? `CLEAN EXECUTION DAYS (✓ executed the plan, positive result): ${cleanExecDays.map(e=>e.date).join(", ")} — treat these as a template\n` : ""}Grades: ${grades}
Mistakes tagged (light context): ${mistakeLine}${cleanDays>0?` | ${cleanDays} clean days`:""}
${planLines.length ? `\nPLAN vs ACTUAL CROSSCHECK\n${planLines.join("\n\n")}` : ""}${priorActionPlan ? `\n\nLAST ${isMonthly?"MONTH":"WEEK"}\'S ACTION PLAN (committed rules — check if they were followed this period):\n${priorActionPlan}` : ""}

FULL JOURNAL — per-trade detail (date-prefixed) + every written note, day by day:
${dayBlocks}

---

BEFORE WRITING: Scan the DIRECTION and SESSION breakdowns first. If one direction is net-negative while the other is profitable, that is a headline insight. If commission drag exceeds 30% of gross, that is structural. Then read every word of the written journal notes — those are the primary source for themes and coaching insights. Mistake tags are light supporting context only.

**\ud83d\udcca HEADLINE INSIGHTS**
Write 3-4 punchy one-line insights derived from the actual trade data and written notes. Balance: surface what's working as clearly as what isn't.
• [Pattern from data or notes]: [specific numbers or quote] → [what it means]
Example style (use real data): "Shorts carrying the book: +$354 gross 82%WR vs longs -$37 — long edge needs a filter"
If a session or direction is genuinely strong, that belongs here too: "NY Morning is the edge: 86% WR, +$296 net — the problem is what happens after noon"

**\ud83d\udcd3 NOTES ANALYSIS**
This is the most important section. Read every word written across all days — market notes, lessons, rules, improvements, best/worst trade descriptions, plans. Find:
• Recurring themes — same insight, intention, or observation written on 2+ days (quote each with date)
• Contradictions — something written one day that the next day's notes or trades contradict
• Unrealized intentions — a plan or lesson written but not reflected in subsequent behavior
• Self-awareness worth reinforcing — something the trader accurately identified that the data confirms
One bullet per finding. Quote their exact words. Mistake dropdown tags can appear here only as corroboration of something already visible in the prose — not as the primary finding.

**\ud83d\udcc8 PERFORMANCE PATTERNS**
3-4 bullets on what the numbers reveal — sourced from trade data, not from mistake tags:
• Which session/direction is helping vs hurting NET P&L (use net figures, not just gross)
• Sequence patterns visible in the trade log — time-of-day drift, sizing changes, re-entry timing
• Commission drag impact if fees exceed 15% of gross — quantify the structural cost
• Any hold-time or order-type edge visible in the data

**\ud83d\udea9 PLAN vs REALITY**
For every "Tomorrow plan" written, cover it in this exact format — no headers, just flowing bullets:
• **[Date] Plan:** Quote the specific intention the trader wrote (exact words, under 30 words)
• **What happened:** State the actual outcome — P&L, grade, key trades. Be specific, not just the label.
• **Verdict:** One sentence starting with either "Plan held — [what they did right]" or "Plan slipped — [the specific rule that broke, with evidence from trades or notes, not just the mistake tags]"

If a plan was honored, name what specifically worked. If it slipped, explain in plain language what the gap was between the intention and the reality — cite a trade time, a P&L figure, or a quote from the day's notes. Never use ✓ or ✗ as the lead word of a sentence. Skip this section entirely if no Tomorrow plans were written.

**\ud83d\udca1 STRENGTHS**
2-3 specific strengths backed by trade data or direct quotes from written notes. Be genuine — if the morning session consistently delivered, say so with exact figures. If the trader wrote something insightful and the data confirms it, reinforce it. No filler praise.

**\ud83c\udfaf ACTION PLAN FOR NEXT ${isMonthly ? "MONTH" : "WEEK"}**
Exactly 3 rules rooted in TRADE DATA and WRITTEN NOTES from this review — not from the mistake dropdown tags. At least one rule should build on a strength or a correct self-observation the trader already made. Format: [Root cause from data/notes] → [Specific measurable rule with threshold]`;
  };

  const generateSummary = async (period, forceRerun = false) => {
    const periodEntries = grouped[period] || [];
    const label = periodLabel(period);
    setSelectedPeriod(period);
    setSummary("");

    // If already saved and not forcing rerun, just display it
    if (generated[period] && !forceRerun) { setSummary(generated[period]); return; }

    const hasNotes = periodEntries.some(e =>
      e.lessonsLearned || e.mistakes || e.improvements || e.marketNotes ||
      e.bestTrade || e.worstTrade || e.tomorrow || e.rules || e.reinforceRule ||
      (e.parsedTrades?.length > 0)
    );
    if (!hasNotes) { setSummary("NO_NOTES"); return; }
    if (!ai?.enabled || !ai?.apiKey) { setSummary("AI_NOT_CONFIGURED"); return; }

    setLoading(true);
    try {
      let prompt = buildPrompt(periodEntries, label, period);

      // Guard: if prompt is very large, strip per-trade detail lines (keep all written notes)
      if (prompt.length > 18000) {
        const lines = prompt.split('\n');
        const trimmed = lines.filter(l => !l.match(/^\s+T\d+:/));
        prompt = trimmed.join('\n');
      }

      const txt = await aiRequestText(ai, {
        max_tokens: 8192,
        timeoutMs: 120000,
        thinkingBudget: 0,
        messages: [{ role: 'user', content: prompt }],
      });
      const now  = new Date().toISOString();
      const hash = calcNotesHash(periodEntries);
      setGenerated(prev   => ({ ...prev, [period]: txt  }));
      setSavedAt(prev     => ({ ...prev, [period]: now  }));
      setSavedHashes(prev => ({ ...prev, [period]: hash }));
      setSummary(txt);
    } catch (err) {
      const f = friendlyAiError(err);
      console.warn('Recap failed:', f.code, f.message, err);
      // Never cache errors — always allow retry on next click
      setSummary(`ERROR:${f.code}:${f.message}`);
    }
    setLoading(false);
  };

  const deleteRecap = async (period) => {
    setGenerated(prev   => { const n = {...prev}; delete n[period]; return n; });
    setSavedAt(prev     => { const n = {...prev}; delete n[period]; return n; });
    setSavedHashes(prev => { const n = {...prev}; delete n[period]; return n; });
    if (selectedPeriod === period) setSummary("");
  };

  const renderSummary = (text) => {
    if (!text || text === "NO_NOTES") return (
      <div style={{ textAlign: "center", padding: "40px 20px", color: "#64748b", fontSize: 13 }}>
        No journal notes found for this period. Add lessons, mistakes, or market notes to generate a recap.
      </div>
    );
    if (text === "AI_NOT_CONFIGURED") return (
      <div style={{ textAlign: "center", padding: "40px 20px", color: "#94a3b8", fontSize: 13, lineHeight: 1.7 }}>
        AI recap is turned off or not configured. Open Settings (⚙) to add your API key or disable AI sections.
      </div>
    );
    if (text?.startsWith("ERROR:")) {
      const parts = text.split(":");
      const code = parts[1] || "unknown";
      const msg = parts.slice(2).join(":") || "Unknown error";
      const helpMap = {
        timeout: "The request timed out. Try ↺ re-run.",
        no_key: "No API key found. Open Settings (⚙) → AI & API Key and add your key.",
        unauthorized: "API key rejected (401). Check your key in Settings (⚙).",
        forbidden: "Request blocked (403). Check your Gemini/Anthropic account permissions.",
        rate_limit: "Gemini rate limit hit (429). The journal will auto-retry with a 35-second wait between attempts. If all 3 retries fail, wait 1–2 minutes then try ↺ re-run. To avoid this: use Gemini 2.5 Flash (higher free limits) or add billing to your Google AI Studio account.",
        provider_down: "API returned a 5xx server error. Try again in a moment.",
        network: "Network error. Check your internet connection and try again.",
      };
      return (
        <div style={{ padding: "32px 24px", background: "#0f1729", border: "1px solid #7f1d1d", borderRadius: 6 }}>
          <div style={{ fontSize: 13, color: "#f87171", fontWeight: 600, marginBottom: 10 }}>⚠ Recap failed — {code}</div>
          <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.7, marginBottom: 14 }}>{helpMap[code] || msg}</div>
          <div style={{ fontSize: 11, color: "#475569", fontFamily: "'DM Mono',monospace", background: "#0a0e1a", padding: "8px 12px", borderRadius: 4 }}>{msg}</div>
        </div>
      );
    }
    return <RenderAI text={text} />;
  };

  return (
    <div>
      <div style={{ background: "linear-gradient(135deg,rgba(56,189,248,0.1),rgba(129,140,248,0.12),rgba(192,132,252,0.08))", border: "1px solid rgba(129,140,248,0.3)", borderRadius: 6, padding: "14px 18px", marginBottom: 20, position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc)" }} />
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 26 }}>🤖</span>
          <span style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 26, letterSpacing: "0.1em", background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>AI RECAP</span>
          <button
            onClick={() => setShowTaglinePicker(p => !p)}
            title="Change tagline"
            style={{ fontSize: 9, color: showTaglinePicker ? "#93c5fd" : "#475569", letterSpacing: "0.15em", background: "transparent", border: "none", cursor: "pointer", fontFamily: "inherit", padding: "2px 4px", transition: "color .15s" }}>
            {getTaglineText(taglineId)}
          </button>
        </div>
        {/* Tagline picker dropdown */}
        {showTaglinePicker && (
          <div style={{ marginTop: 8, background: "#060b18", border: "1px solid #1e293b", borderRadius: 6, padding: "10px 12px", display: "flex", flexDirection: "column", gap: 6, animation: "formSlide 0.18s ease" }}>
            <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.12em", marginBottom: 4 }}>CHOOSE YOUR TAGLINE</div>
            {AI_TAGLINES.map(t => (
              <button key={t.id} onClick={() => { setTaglineId(t.id); saveTagline(t.id); setShowTaglinePicker(false); }}
                style={{ textAlign: "left", background: taglineId === t.id ? "rgba(129,140,248,0.1)" : "transparent", border: `1px solid ${taglineId === t.id ? "rgba(129,140,248,0.35)" : "#0f1729"}`, borderRadius: 4, padding: "7px 12px", fontFamily: "inherit", fontSize: 10, color: taglineId === t.id ? "#93c5fd" : "#64748b", cursor: "pointer", letterSpacing: "0.08em", transition: "all .12s" }}>
                {taglineId === t.id && <span style={{ color: "#818cf8", marginRight: 6 }}>✓</span>}{t.text}
              </button>
            ))}
          </div>
        )}
      </div>

      {entries.length === 0 ? (
        <div style={{ textAlign: "center", padding: "60px 20px", color: "#64748b", fontSize: 13 }}>No journal entries yet. Start logging your trading days to generate AI recaps.</div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 20, alignItems: "stretch" }}>

          {/* Left: period selector */}
          <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 6, overflow: "hidden", display: "flex", flexDirection: "column" }}>
            {/* Mode toggle */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", borderBottom: "1px solid #1e293b", flexShrink: 0 }}>
              {["weekly", "monthly"].map(m => (
                <button key={m} onClick={() => { setRecapMode(m); setSelectedPeriod(null); setSummary(""); }}
                  style={{ position: "relative", padding: "10px 0", fontFamily: "inherit", fontSize: 10, letterSpacing: "0.08em", cursor: "pointer", border: "none", overflow: "hidden",
                    background: recapMode === m ? "#0a1628" : "transparent",
                    color: recapMode === m ? "#818cf8" : "#334155",
                    textTransform: "uppercase", transition: "all .15s" }}>
                  {recapMode === m && <span style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc)" }} />}
                  {m}
                </button>
              ))}
            </div>
            {/* Period list */}
            <div style={{ overflowY: "auto", flex: 1 }}>
              {periods.length === 0 ? (
                <div style={{ padding: "20px 14px", fontSize: 11, color: "#64748b" }}>No entries yet.</div>
              ) : periods.map(p => {
                const es = grouped[p] || [];
                const pnl = es.reduce((s, e) => s + netPnl(e), 0);
                const wins = es.filter(e => netPnl(e) > 0).length;
                const hasNotes = es.some(e => e.lessonsLearned || e.mistakes || e.improvements || e.marketNotes || e.parsedTrades?.length > 0);
                const isSelected = selectedPeriod === p;
                const isGenerated = !!generated[p];
                const isLoading = loading && selectedPeriod === p;
                const ts = savedAt[p] ? new Date(savedAt[p]).toLocaleDateString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : null;
                // Staleness: compare current notes hash against what was hashed when recap was generated
                const currentHash = isGenerated ? calcNotesHash(es) : null;
                const isStale = isGenerated && savedHashes[p] && currentHash !== savedHashes[p];
                return (
                  <div key={p}
                    style={{ padding: "12px 14px", borderBottom: "1px solid #0f1729", background: isSelected ? "#0a1628" : "transparent", borderLeft: isSelected ? "2px solid transparent" : "2px solid transparent", borderImage: isSelected ? "linear-gradient(180deg,#38bdf8,#818cf8,#c084fc) 1" : "none", transition: "all .15s" }}
                    onMouseEnter={e => { if (!isSelected) e.currentTarget.style.background = "#0d1526"; }}
                    onMouseLeave={e => { if (!isSelected) e.currentTarget.style.background = isSelected ? "#0a1628" : "transparent"; }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 3 }}>
                      <div style={{ fontSize: 10, color: isSelected ? "#93c5fd" : "#94a3b8", letterSpacing: "0.05em" }}>{periodLabel(p)}</div>
                      {isGenerated && (
                        <button onClick={e => { e.stopPropagation(); deleteRecap(p); }}
                          title="Clear saved recap"
                          style={{ background: "transparent", border: "none", color: "#1e3a5f", fontSize: 10, cursor: "pointer", padding: "0 2px", lineHeight: 1, fontFamily: "inherit", transition: "color .12s" }}
                          onMouseEnter={e => e.currentTarget.style.color = "#f87171"}
                          onMouseLeave={e => e.currentTarget.style.color = "#1e3a5f"}>✕</button>
                      )}
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: isGenerated ? 6 : 8 }}>
                      <span style={{ fontSize: 12, fontWeight: 600, color: pnl >= 0 ? "#4ade80" : "#f87171" }}>{pnl >= 0 ? "+" : ""}${Math.abs(pnl).toLocaleString("en-US", { maximumFractionDigits: 0 })}</span>
                      <span style={{ fontSize: 9, letterSpacing: 0 }}><span style={{ color: "#4ade80" }}>{wins}</span><span style={{ color: "#94a3b8" }}>/</span><span style={{ color: "#f87171" }}>{es.length - wins}</span></span>
                    </div>
                    {/* Saved indicator + staleness warning */}
                    {isGenerated && ts && (
                      <div style={{ fontSize: 8, letterSpacing: "0.06em", marginBottom: 7, display: "flex", alignItems: "center", gap: 4 }}>
                        <span style={{ color: isStale ? "#f59e0b" : "#166534" }}>●</span>
                        <span style={{ color: isStale ? "#f59e0b" : "#334155" }}>
                          {isStale ? "⚠ NOTES CHANGED · " : "SAVED · "}{ts}
                        </span>
                      </div>
                    )}
                    {!hasNotes && !isGenerated && <div style={{ fontSize: 9, color: "#1e3a5f", marginBottom: 6 }}>no notes</div>}
                    {/* Action buttons */}
                    {isGenerated ? (
                      <div style={{ display: "flex", gap: 5 }}>
                        <button
                          onClick={() => generateSummary(p)}
                          disabled={isLoading}
                          style={{ flex: 2, padding: "6px 8px", borderRadius: 4, fontFamily: "inherit", fontSize: 9, letterSpacing: "0.06em", cursor: isLoading ? "not-allowed" : "pointer", background: isSelected ? "rgba(56,189,248,0.08)" : "transparent", border: `1px solid ${isSelected ? "rgba(56,189,248,0.2)" : "#1e293b"}`, color: isSelected ? "#93c5fd" : "#64748b", transition: "all .12s" }}>
                          {isLoading ? "LOADING…" : "VIEW →"}
                        </button>
                        <button
                          onClick={() => generateSummary(p, true)}
                          disabled={isLoading}
                          title="Re-run AI analysis"
                          style={{ flex: 1, padding: "6px 8px", borderRadius: 4, fontFamily: "inherit", fontSize: 9, letterSpacing: "0.06em", cursor: isLoading ? "not-allowed" : "pointer", background: "transparent", border: "1px solid #1e293b", color: "#475569", transition: "all .12s" }}>
                          ↺
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => generateSummary(p)}
                        disabled={isLoading}
                        style={{ width: "100%", padding: "7px 10px", borderRadius: 4, fontFamily: "inherit", fontSize: 10, letterSpacing: "0.06em", cursor: isLoading ? "not-allowed" : "pointer", transition: "all .15s",
                          background: isLoading ? "transparent" : "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)",
                          border: isLoading ? "1px solid #1e293b" : "none",
                          color: isLoading ? "#475569" : "#070d1a", fontWeight: 700 }}>
                        {isLoading ? "ANALYSING..." : `ANALYSE ${recapMode === "weekly" ? "WEEK" : "MONTH"} →`}
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right: summary panel */}
          <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 6, minHeight: 400 }}>
            {!selectedPeriod ? (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: 400, gap: 12, color: "#64748b" }}>
                <div style={{ fontSize: 28, opacity: 0.3 }}>✦</div>
                <div style={{ fontSize: 12, letterSpacing: "0.1em" }}>SELECT A PERIOD TO VIEW OR GENERATE YOUR RECAP</div>
                <div style={{ fontSize: 10, color: "#1e3a5f", letterSpacing: "0.1em" }}>{getTaglineText(taglineId)}</div>
              </div>
            ) : (
              <div style={{ padding: "20px 24px" }}>
                {/* Period header */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20, paddingBottom: 16, borderBottom: "1px solid #1e293b" }}>
                  <div>
                    <div style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 4 }}>{recapMode.toUpperCase()} RECAP</div>
                    <div style={{ fontSize: 15, color: "#e2e8f0", fontWeight: 600 }}>{periodLabel(selectedPeriod)}</div>
                    {savedAt[selectedPeriod] && (() => {
                      const selectedEs = grouped[selectedPeriod] || [];
                      const panelHash = calcNotesHash(selectedEs);
                      const panelStale = savedHashes[selectedPeriod] && panelHash !== savedHashes[selectedPeriod];
                      return (
                        <div style={{ fontSize: 8, letterSpacing: "0.08em", marginTop: 4, display: "flex", alignItems: "center", gap: 4 }}>
                          <span style={{ color: panelStale ? "#f59e0b" : "#166534" }}>●</span>
                          <span style={{ color: panelStale ? "#f59e0b" : "#334155" }}>
                            {panelStale ? "⚠ NOTES CHANGED — consider re-running · " : "SAVED "}
                            {new Date(savedAt[selectedPeriod]).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit" })}
                          </span>
                        </div>
                      );
                    })()}
                  </div>
                  {(() => {
                    const es = grouped[selectedPeriod] || [];
                    const pnl = es.reduce((s, e) => s + netPnl(e), 0);
                    const wins = es.filter(e => netPnl(e) > 0).length;
                    const wr = es.length ? Math.round(wins / es.length * 100) : 0;
                    return (
                      <div style={{ display: "flex", gap: 16, textAlign: "right" }}>
                        <div><div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.08em" }}>NET P&L</div><div style={{ fontSize: 16, fontWeight: 700, color: pnl >= 0 ? "#4ade80" : "#f87171" }}>{pnl >= 0 ? "+" : ""}${Math.abs(pnl).toLocaleString("en-US", { maximumFractionDigits: 0 })}</div></div>
                        <div><div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.08em" }}>DAYS</div><div style={{ fontSize: 16, fontWeight: 700, color: "#e2e8f0" }}>{es.length}</div></div>
                        <div><div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.08em" }}>WIN RATE</div><div style={{ fontSize: 16, fontWeight: 700, color: wr >= 50 ? "#4ade80" : "#f87171" }}>{wr}%</div></div>
                      </div>
                    );
                  })()}
                </div>

                {/* Summary content */}
                {loading ? (
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: 300, gap: 16 }}>
                    <div style={{ fontSize: 11, color: "#3b82f6", letterSpacing: "0.15em", animation: "pulse 1.5s infinite" }}>✦ GENERATING YOUR RECAP...</div>
                    <div style={{ fontSize: 10, color: "#64748b" }}>Analysing your journal data & notes…</div>
                    <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }`}</style>
                  </div>
                ) : renderSummary(summary)}

                {/* Bottom actions */}
                {summary && !summary.startsWith("ERROR:") && summary !== "NO_NOTES" && summary !== "AI_NOT_CONFIGURED" && !loading && (
                  <div style={{ marginTop: 20, paddingTop: 16, borderTop: "1px solid #1e293b", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div style={{ fontSize: 9, color: "#1e3a5f", letterSpacing: "0.08em" }}>
                      {savedAt[selectedPeriod] ? `● SAVED ${new Date(savedAt[selectedPeriod]).toLocaleDateString("en-US",{month:"short",day:"numeric"})}` : ""}
                    </div>
                    <div style={{ display: "flex", gap: 8 }}>
                      <button onClick={() => deleteRecap(selectedPeriod)}
                        style={{ background: "transparent", border: "1px solid #1e293b", color: "#475569", padding: "8px 14px", borderRadius: 4, fontFamily: "inherit", fontSize: 10, cursor: "pointer", letterSpacing: "0.06em", transition: "all .12s" }}
                        onMouseEnter={e => { e.currentTarget.style.borderColor = "#7f1d1d"; e.currentTarget.style.color = "#f87171"; }}
                        onMouseLeave={e => { e.currentTarget.style.borderColor = "#1e293b"; e.currentTarget.style.color = "#475569"; }}>
                        ✕ CLEAR
                      </button>
                      <button onClick={() => generateSummary(selectedPeriod, true)}
                        style={{ background: "transparent", border: "1px solid #1e293b", color: "#94a3b8", padding: "8px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.06em", transition: "all .12s" }}
                        onMouseEnter={e => { e.currentTarget.style.borderColor = "#3b82f6"; e.currentTarget.style.color = "#93c5fd"; }}
                        onMouseLeave={e => { e.currentTarget.style.borderColor = "#1e293b"; e.currentTarget.style.color = "#94a3b8"; }}>
                        ↺ RE-RUN
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}



const NO_MISTAKES_OPTION = "No Mistakes — Executed the Plan ✓";

function MistakesDropdown({ selected, onChange }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    if (!open) return;
    const handler = (e) => { if (!ref.current?.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const isClean = selected.includes(NO_MISTAKES_OPTION);

  const toggle = (m) => {
    if (m === NO_MISTAKES_OPTION) {
      // Selecting "No Mistakes" clears everything else
      onChange(isClean ? [] : [NO_MISTAKES_OPTION]);
    } else {
      // Selecting any mistake clears "No Mistakes" if it was set
      const without = selected.filter(x => x !== NO_MISTAKES_OPTION);
      onChange(without.includes(m) ? without.filter(x => x !== m) : [...without, m]);
    }
  };

  const mistakeCount = selected.filter(m => m !== NO_MISTAKES_OPTION).length;

  const buttonLabel = () => {
    if (selected.length === 0) return <span style={{ color: "#64748b" }}>Select mistakes made today…</span>;
    if (isClean) return <span style={{ color: "#4ade80" }}>No Mistakes — Executed the Plan ✓</span>;
    return <span style={{ color: "#f87171" }}>{mistakeCount} mistake{mistakeCount > 1 ? "s" : ""} flagged</span>;
  };

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button type="button" onClick={() => setOpen(p => !p)}
        style={{ width: "100%", display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 14px", background: "#0f1729", border: `1px solid ${open ? "#3b82f6" : isClean ? "#166534" : selected.length ? "#1e3a5f" : "#1e3a5f"}`, borderRadius: 4, fontFamily: "inherit", fontSize: 12, color: "#e2e8f0", cursor: "pointer", transition: "all .15s", textAlign: "left" }}>
        <span>{buttonLabel()}</span>
        <span style={{ color: "#64748b", fontSize: 10, marginLeft: 8, transform: open ? "rotate(180deg)" : "rotate(0deg)", display: "inline-block", transition: "transform .2s" }}>▾</span>
      </button>
      {open && (
        <div style={{ position: "absolute", top: "calc(100% + 4px)", left: 0, right: 0, background: "#0f1729", border: "1px solid #1e3a5f", borderRadius: 6, zIndex: 50, overflow: "hidden", boxShadow: "0 8px 24px rgba(0,0,0,0.5)" }}>
          {selected.length > 0 && (
            <div style={{ padding: "6px 12px", borderBottom: "1px solid #0a0e1a", display: "flex", justifyContent: "flex-end" }}>
              <button onClick={() => onChange([])} style={{ background: "transparent", border: "none", color: "#3b82f6", fontSize: 10, cursor: "pointer", fontFamily: "inherit", letterSpacing: "0.05em" }}>CLEAR ALL</button>
            </div>
          )}
          {/* No Mistakes option — first, styled green */}
          {(() => {
            const checked = isClean;
            return (
              <div onClick={() => toggle(NO_MISTAKES_OPTION)}
                style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 14px", cursor: "pointer", borderBottom: "2px solid #0a0e1a", background: checked ? "#061f0f" : "transparent", transition: "background .1s" }}
                onMouseEnter={e => { if (!checked) e.currentTarget.style.background = "#0a1f12"; }}
                onMouseLeave={e => { if (!checked) e.currentTarget.style.background = "transparent"; }}>
                <div style={{ width: 14, height: 14, borderRadius: 3, border: `1px solid ${checked ? "#4ade80" : "#475569"}`, background: checked ? "#166534" : "transparent", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", transition: "all .15s" }}>
                  {checked && <span style={{ color: "#4ade80", fontSize: 9, lineHeight: 1 }}>✓</span>}
                </div>
                <span style={{ fontSize: 12, color: checked ? "#4ade80" : "#4ade80", fontWeight: checked ? 600 : 400, opacity: checked ? 1 : 0.6 }}>{NO_MISTAKES_OPTION}</span>
              </div>
            );
          })()}
          {/* Regular mistake options */}
          {MISTAKE_OPTIONS.map(m => {
            const checked = selected.includes(m);
            return (
              <div key={m} onClick={() => toggle(m)}
                style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 14px", cursor: "pointer", borderBottom: "1px solid #0a0e1a", background: checked ? "#0d1f3c" : "transparent", transition: "background .1s" }}
                onMouseEnter={e => { if (!checked) e.currentTarget.style.background = "#0a1628"; }}
                onMouseLeave={e => { if (!checked) e.currentTarget.style.background = "transparent"; }}>
                <div style={{ width: 14, height: 14, borderRadius: 3, border: `1px solid ${checked ? "#3b82f6" : "#475569"}`, background: checked ? "#3b82f6" : "transparent", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", transition: "all .15s" }}>
                  {checked && <span style={{ color: "white", fontSize: 9, lineHeight: 1 }}>✓</span>}
                </div>
                <span style={{ fontSize: 12, color: checked ? "#e2e8f0" : "#94a3b8" }}>{m}</span>
              </div>
            );
          })}
        </div>
      )}
      {selected.length > 0 && (
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 8 }}>
          {selected.map(m => (
            <span key={m} style={{ display: "inline-flex", alignItems: "center", gap: 4, padding: "3px 8px", borderRadius: 3,
              background: m === NO_MISTAKES_OPTION ? "#052e16" : "#1f0606",
              border: `1px solid ${m === NO_MISTAKES_OPTION ? "#166534" : "#7f1d1d"}`,
              fontSize: 11, color: m === NO_MISTAKES_OPTION ? "#4ade80" : "#f87171" }}>
              {m}
              <button onClick={() => toggle(m)} style={{ background: "transparent", border: "none", color: m === NO_MISTAKES_OPTION ? "#166534" : "#7f1d1d", cursor: "pointer", padding: 0, fontSize: 11, lineHeight: 1, fontFamily: "inherit" }}>✕</button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// QUOTES VIEW
// ─────────────────────────────────────────────────────────────────────────────
const SEED_QUOTES = [
  { id: "seed-1", text: "The goal of a successful trader is to make the best trades. Money is secondary.", author: "Alexander Elder", category: "Mindset" },
  { id: "seed-6", text: "In trading, the one who survives longest wins. Survival is the ultimate edge.", author: "Ed Seykota", category: "Mindset" },
  { id: "seed-m1", text: "Do not pray for an easy life. Pray for the strength to endure a difficult one.", author: "Bruce Lee", category: "Mindset" },
  { id: "seed-m2", text: "The mind is everything. What you think, you become.", author: "Buddha", category: "Mindset" },
  { id: "seed-m3", text: "Whether you think you can or think you can't, you are right.", author: "Henry Ford", category: "Mindset" },
  { id: "seed-m4", text: "I hated every minute of training, but I said to myself: suffer now and live the rest of your life as a champion.", author: "Muhammad Ali", category: "Mindset" },
  { id: "seed-m5", text: "You don't drown by falling in the water. You drown by staying there.", author: "Ed Cole", category: "Mindset" },
  { id: "seed-m6", text: "It's not about the size of the dog in the fight. It's about the size of the fight in the dog.", author: "Mark Twain", category: "Mindset" },
  { id: "seed-m7", text: "The man who moves a mountain begins by carrying away small stones.", author: "Confucius", category: "Mindset" },
  { id: "seed-m8", text: "Amateurs hope. Professionals work.", author: "Garson Kanin", category: "Mindset" },
  { id: "seed-m9", text: "Champions aren't made in gyms. Champions are made from something they have deep inside — a desire, a dream, a vision.", author: "Muhammad Ali", category: "Mindset" },
  { id: "seed-m10", text: "I am not the richest, smartest, or most talented person in the room. But I will be damned if I am not the hardest working.", author: "Will Smith", category: "Mindset" },
  { id: "seed-m11", text: "Every strike brings me closer to the next home run.", author: "Babe Ruth", category: "Mindset" },
  { id: "seed-m12", text: "The brick walls are there for a reason. They're there to stop people who don't want it badly enough.", author: "Randy Pausch", category: "Mindset" },
  { id: "seed-m13", text: "Strength does not come from physical capacity. It comes from an indomitable will.", author: "Mahatma Gandhi", category: "Mindset" },
  { id: "seed-m14", text: "It always seems impossible until it's done.", author: "Nelson Mandela", category: "Mindset" },
  { id: "seed-m15", text: "The harder the battle, the sweeter the victory.", author: "Les Brown", category: "Mindset" },
  { id: "seed-m16", text: "You only live once, but if you do it right, once is enough.", author: "Mae West", category: "Mindset" },
  { id: "seed-m17", text: "Success is not final, failure is not fatal: it is the courage to continue that counts.", author: "Winston Churchill", category: "Mindset" },
  { id: "seed-m18", text: "Fall seven times, stand up eight.", author: "Japanese Proverb", category: "Mindset" },
  { id: "seed-m19", text: "The successful warrior is the average man with laser-like focus.", author: "Bruce Lee", category: "Mindset" },
  { id: "seed-m20", text: "Your biggest competition is yourself.", author: "Unknown", category: "Mindset" },
  { id: "seed-5", text: "The markets are never wrong. Opinions often are.", author: "Jesse Livermore", category: "Discipline" },
  { id: "seed-d1", text: "Pain is temporary. Quitting lasts forever.", author: "Lance Armstrong", category: "Discipline" },
  { id: "seed-d2", text: "Discipline is the bridge between goals and accomplishment.", author: "Jim Rohn", category: "Discipline" },
  { id: "seed-d3", text: "Suffer the pain of discipline or suffer the pain of regret.", author: "Jim Rohn", category: "Discipline" },
  { id: "seed-d4", text: "The market is a device for transferring money from the impatient to the patient.", author: "Warren Buffett", category: "Discipline" },
  { id: "seed-d5", text: "I just wait until there is money lying in the corner. All I have to do is go over there and pick it up.", author: "Jim Rogers", category: "Discipline" },
  { id: "seed-d6", text: "Amateurs sit and wait for inspiration. The rest of us just get up and go to work.", author: "Stephen King", category: "Discipline" },
  { id: "seed-d7", text: "We are what we repeatedly do. Excellence, then, is not an act, but a habit.", author: "Aristotle", category: "Discipline" },
  { id: "seed-d8", text: "The elements of good trading are: cutting losses, cutting losses, and cutting losses.", author: "Ed Seykota", category: "Discipline" },
  { id: "seed-d9", text: "With self-discipline, almost anything is possible.", author: "Theodore Roosevelt", category: "Discipline" },
  { id: "seed-d10", text: "The stock market is filled with individuals who know the price of everything, but the value of nothing.", author: "Philip Fisher", category: "Discipline" },
  { id: "seed-d11", text: "Don't look for the needle in the haystack. Just buy the haystack.", author: "John Bogle", category: "Discipline" },
  { id: "seed-d12", text: "The big money is not in the buying and selling, but in the waiting.", author: "Charlie Munger", category: "Discipline" },
  { id: "seed-d13", text: "Discipline is choosing between what you want now and what you want most.", author: "Abraham Lincoln", category: "Discipline" },
  { id: "seed-d14", text: "In the middle of difficulty lies opportunity.", author: "Albert Einstein", category: "Discipline" },
  { id: "seed-d15", text: "If you don't find a way to make money while you sleep, you will work until you die.", author: "Warren Buffett", category: "Discipline" },
  { id: "seed-2", text: "It's not whether you're right or wrong that's important, but how much money you make when you're right and how much you lose when you're wrong.", author: "George Soros", category: "Risk" },
  { id: "seed-4", text: "The most important thing is to cut your losses quickly. Never let them run.", author: "Jesse Livermore", category: "Risk" },
  { id: "seed-8", text: "Risk comes from not knowing what you're doing.", author: "Warren Buffett", category: "Risk" },
  { id: "seed-r1", text: "I am always thinking about losing money as opposed to making money. Don't focus on making money, focus on protecting what you have.", author: "Paul Tudor Jones", category: "Risk" },
  { id: "seed-r2", text: "Never risk more than you can afford to lose. Never. Full stop.", author: "Larry Hite", category: "Risk" },
  { id: "seed-r3", text: "Protect your capital first. Without it, you cannot play the game.", author: "Jesse Livermore", category: "Risk" },
  { id: "seed-r4", text: "You can be wrong half the time and still make a fortune if you manage your losses.", author: "William O'Neil", category: "Risk" },
  { id: "seed-r5", text: "The biggest risk of all is not taking one.", author: "Mellody Hobson", category: "Risk" },
  { id: "seed-r6", text: "The first rule of trading: never lose big.", author: "Paul Tudor Jones", category: "Risk" },
  { id: "seed-r7", text: "It's not about being right. It's about how much you make when you're right versus how much you lose when you're wrong.", author: "George Soros", category: "Risk" },
  { id: "seed-r8", text: "If you can't take a small loss, sooner or later you will take the mother of all losses.", author: "Ed Seykota", category: "Risk" },
  { id: "seed-r9", text: "Risk management is the most important thing to be well understood.", author: "Paul Tudor Jones", category: "Risk" },
  { id: "seed-r10", text: "Cut your losses short and let your winners run.", author: "Jesse Livermore", category: "Risk" },
  { id: "seed-r11", text: "Opportunities come infrequently. When it rains gold, put out the bucket, not the thimble.", author: "Warren Buffett", category: "Risk" },
  { id: "seed-3", text: "Trade what you see, not what you think.", author: "Linda Bradford Raschke", category: "Execution" },
  { id: "seed-e1", text: "The most important organ in the body as far as the stock market is concerned is the guts, not the head.", author: "Peter Lynch", category: "Execution" },
  { id: "seed-e2", text: "I never buy at the bottom and I always sell too soon.", author: "Baron Rothschild", category: "Execution" },
  { id: "seed-e3", text: "Know what you own, and know why you own it.", author: "Peter Lynch", category: "Execution" },
  { id: "seed-e4", text: "The exit is more important than the entry.", author: "Ed Seykota", category: "Execution" },
  { id: "seed-e5", text: "Patterns repeat, because human nature hasn't changed for thousands of years.", author: "Jesse Livermore", category: "Execution" },
  { id: "seed-e6", text: "Be fearful when others are greedy and greedy when others are fearful.", author: "Warren Buffett", category: "Execution" },
  { id: "seed-e7", text: "There is a time to go long, a time to go short, and a time to go fishing.", author: "Jesse Livermore", category: "Execution" },
  { id: "seed-e8", text: "The best trades are the ones you almost didn't take.", author: "Unknown", category: "Execution" },
  { id: "seed-e9", text: "I don't think about what the market will do. I think about what I will do.", author: "Jack Schwager", category: "Execution" },
  { id: "seed-e10", text: "Acting without thinking is the cause of most of my losses.", author: "Jesse Livermore", category: "Execution" },
  { id: "seed-e11", text: "Speed is fine, but accuracy is final.", author: "Wyatt Earp", category: "Execution" },
  { id: "seed-e12", text: "You miss 100% of the shots you don't take.", author: "Wayne Gretzky", category: "Execution" },
  { id: "seed-7", text: "Every battle is won or lost before it is ever fought.", author: "Sun Tzu", category: "Preparation" },
  { id: "seed-9", text: "The hard work in trading comes in the preparation. The easy part is the execution.", author: "Jack Schwager", category: "Preparation" },
  { id: "seed-p1", text: "Give me six hours to chop down a tree and I will spend the first four sharpening the axe.", author: "Abraham Lincoln", category: "Preparation" },
  { id: "seed-p2", text: "Luck is what happens when preparation meets opportunity.", author: "Seneca", category: "Preparation" },
  { id: "seed-p3", text: "By failing to prepare, you are preparing to fail.", author: "Benjamin Franklin", category: "Preparation" },
  { id: "seed-p4", text: "The will to win is not nearly so important as the will to prepare to win.", author: "Vince Lombardi", category: "Preparation" },
  { id: "seed-p5", text: "The more I practice, the luckier I get.", author: "Gary Player", category: "Preparation" },
  { id: "seed-p6", text: "Success is where preparation and opportunity meet.", author: "Bobby Unser", category: "Preparation" },
  { id: "seed-p7", text: "Before anything else, preparation is the key to success.", author: "Alexander Graham Bell", category: "Preparation" },
  { id: "seed-p8", text: "I will prepare and some day my chance will come.", author: "Abraham Lincoln", category: "Preparation" },
  { id: "seed-p9", text: "Plans are nothing. Planning is everything.", author: "Dwight D. Eisenhower", category: "Preparation" },
  { id: "seed-p10", text: "Today I will do what others won't, so tomorrow I can do what others can't.", author: "Jerry Rice", category: "Preparation" },
  { id: "seed-ps1", text: "The stock market is a device for transferring money from the impatient to the patient.", author: "Warren Buffett", category: "Psychology" },
  { id: "seed-ps2", text: "Fear and greed are the two driving emotions of market participants. Master them and you master the market.", author: "Alexander Elder", category: "Psychology" },
  { id: "seed-ps3", text: "The market is a voting machine in the short run and a weighing machine in the long run.", author: "Benjamin Graham", category: "Psychology" },
  { id: "seed-ps4", text: "The investor's chief problem — and even his worst enemy — is likely to be himself.", author: "Benjamin Graham", category: "Psychology" },
  { id: "seed-ps5", text: "The key to trading success is emotional discipline. If intelligence were the key, there would be a lot more people making money.", author: "Victor Sperandeo", category: "Psychology" },
  { id: "seed-ps6", text: "When you lose, don't lose the lesson.", author: "Dalai Lama", category: "Psychology" },
  { id: "seed-ps7", text: "The markets are never wrong. But our interpretations of them often are.", author: "Jesse Livermore", category: "Psychology" },
  { id: "seed-ps8", text: "Never let a win turn into a loss, and never let a loss turn into a catastrophe.", author: "Unknown", category: "Psychology" },
  { id: "seed-ps9", text: "If you are distressed by anything external, the pain is not due to the thing itself, but to your estimate of it; and this you have the power to revoke at any moment.", author: "Marcus Aurelius", category: "Psychology" },
  { id: "seed-ps10", text: "Your job is not to predict the future. Your job is to react to the present.", author: "Unknown", category: "Psychology" },
  { id: "seed-ps11", text: "The greatest enemy of a good plan is the dream of a perfect plan.", author: "Carl von Clausewitz", category: "Psychology" },
  { id: "seed-ps12", text: "Emotional intelligence is the ability to sense, understand, and effectively apply the power and acumen of emotions.", author: "Robert K. Cooper", category: "Psychology" },
  { id: "seed-ps13", text: "Between stimulus and response there is a space. In that space is our power to choose our response.", author: "Viktor Frankl", category: "Psychology" },
  { id: "seed-ps14", text: "You will never be punished for taking a profit.", author: "Jesse Livermore", category: "Psychology" },
  { id: "seed-ps15", text: "The crowd is always wrong at the extremes.", author: "Humphrey Neill", category: "Psychology" },
  { id: "seed-st1", text: "All men can see these tactics whereby I conquer, but what none can see is the strategy out of which victory is evolved.", author: "Sun Tzu", category: "Strategy" },
  { id: "seed-st2", text: "There are old traders and bold traders, but there are no old, bold traders.", author: "Ed Seykota", category: "Strategy" },
  { id: "seed-st3", text: "The trend is your friend until the end when it bends.", author: "Ed Seykota", category: "Strategy" },
  { id: "seed-st4", text: "Look for opportunities where risk is small and upside is large.", author: "Paul Tudor Jones", category: "Strategy" },
  { id: "seed-st5", text: "Don't try to buy at the bottom and sell at the top. It can't be done except by liars.", author: "Bernard Baruch", category: "Strategy" },
  { id: "seed-st6", text: "Wide diversification is only required when investors do not understand what they are doing.", author: "Warren Buffett", category: "Strategy" },
  { id: "seed-st7", text: "The best way to measure your investing success is not by whether you're beating the market but by whether you've put in place a financial plan and a behavioral discipline that are likely to get you where you want to go.", author: "Benjamin Graham", category: "Strategy" },
  { id: "seed-st8", text: "Trade with the trend. Counter-trend trades are for very advanced traders only.", author: "Alexander Elder", category: "Strategy" },
  { id: "seed-st9", text: "Without a system, every loss is devastating and every win is lucky.", author: "Brett Steenbarger", category: "Strategy" },
  { id: "seed-st10", text: "A good system is better than a brilliant trader with no system.", author: "Ed Seykota", category: "Strategy" },
  { id: "seed-st11", text: "The simpler the strategy, the harder it is to mess up.", author: "Unknown", category: "Strategy" },
  { id: "seed-st12", text: "Don't fight the tape, don't fight the Fed.", author: "Marty Zweig", category: "Strategy" },
  { id: "seed-o1", text: "The obstacle is the way.", author: "Marcus Aurelius", category: "Other" },
  { id: "seed-o2", text: "You have power over your mind, not outside events. Realize this, and you will find strength.", author: "Marcus Aurelius", category: "Other" },
  { id: "seed-o3", text: "It is not what happens to you, but how you react to it that matters.", author: "Epictetus", category: "Other" },
  { id: "seed-o4", text: "Wealth consists not in having great possessions, but in having few wants.", author: "Epictetus", category: "Other" },
  { id: "seed-o5", text: "He suffers more than necessary, who suffers before it is necessary.", author: "Seneca", category: "Other" },
  { id: "seed-o6", text: "Begin at once to live, and count each separate day as a separate life.", author: "Seneca", category: "Other" },
  { id: "seed-o7", text: "No man is free who is not a master of himself.", author: "Epictetus", category: "Other" },
  { id: "seed-o8", text: "The wise man does at once what the fool does finally.", author: "Niccolo Machiavelli", category: "Other" },
  { id: "seed-o9", text: "Very little is needed to make a happy life; it is all within yourself, in your way of thinking.", author: "Marcus Aurelius", category: "Other" },
  { id: "seed-o10", text: "He who has a why to live can bear almost any how.", author: "Friedrich Nietzsche", category: "Other" },
  { id: "seed-o11", text: "In the depth of winter, I finally learned that within me there lay an invincible summer.", author: "Albert Camus", category: "Other" },
  { id: "seed-o12", text: "Do not go where the path may lead, go instead where there is no path and leave a trail.", author: "Ralph Waldo Emerson", category: "Other" },
  { id: "seed-o13", text: "It does not matter how slowly you go as long as you do not stop.", author: "Confucius", category: "Other" },
  { id: "seed-o14", text: "Out of difficulties grow miracles.", author: "Jean de la Bruyere", category: "Other" },
  { id: "seed-o15", text: "We suffer more in imagination than in reality.", author: "Seneca", category: "Other" },
  { id: "seed-o16", text: "First say to yourself what you would be, and then do what you have to do.", author: "Epictetus", category: "Other" },
  { id: "seed-o17", text: "The two most powerful warriors are patience and time.", author: "Leo Tolstoy", category: "Other" },
  { id: "seed-o18", text: "Tomorrow is promised to no one. Act with urgency today.", author: "Walter Payton", category: "Other" },
  { id: "seed-o19", text: "The greatest glory in living lies not in never falling, but in rising every time we fall.", author: "Nelson Mandela", category: "Other" },
  { id: "seed-o20", text: "You are never too old to set another goal or to dream a new dream.", author: "C.S. Lewis", category: "Other" }
];
const QUOTE_CATS = ["All", "Mindset", "Risk", "Discipline", "Execution", "Preparation", "Psychology", "Strategy", "Other"];

function QuotesView() {
  const [quotes, setQuotes] = useState([]);
  const [loaded, setLoaded] = useState(false);
  const [activeCat, setActiveCat] = useState("All");
  const [pageStart, setPageStart] = useState(0);
  const [showForm, setShowForm] = useState(false);
  const [editId, setEditId] = useState(null);
  const [qForm, setQForm] = useState({ text: "", author: "", category: "Mindset" });
  const [qErr, setQErr] = useState("");
  const [delConfirm, setDelConfirm] = useState(null);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const r = await storage.get("trader-quotes-v1");
        setQuotes(r?.value ? JSON.parse(r.value) : SEED_QUOTES);
        if (!r?.value) await storage.set("trader-quotes-v1", JSON.stringify(SEED_QUOTES));
      } catch { setQuotes(SEED_QUOTES); }
      setLoaded(true);
    })();
  }, []);

  const persist = async (updated) => {
    setQuotes(updated);
    try { await storage.set("trader-quotes-v1", JSON.stringify(updated)); } catch(e) { console.warn(e); }
  };

  const filtered = activeCat === "All" ? quotes : quotes.filter(q => q.category === activeCat);

  useEffect(() => { setPageStart(0); }, [activeCat]);

  const openAdd = () => { setEditId(null); setQForm({ text: "", author: "", category: "Mindset" }); setQErr(""); setShowForm(true); };
  const openEdit = (q) => { setEditId(q.id); setQForm({ text: q.text, author: q.author, category: q.category }); setQErr(""); setShowForm(true); };
  const cancelForm = () => { setShowForm(false); setEditId(null); setQErr(""); };

  const submitForm = async () => {
    if (!qForm.text.trim()) { setQErr("Quote text is required."); return; }
    if (!qForm.author.trim()) { setQErr("Author is required."); return; }
    if (editId) {
      await persist(quotes.map(q => q.id === editId ? { ...q, ...qForm } : q));
    } else {
      await persist([...quotes, { id: `q-${Date.now()}`, ...qForm }]);
    }
    setShowForm(false); setEditId(null); setQErr("");
  };

  const doDelete = async (id) => {
    await persist(quotes.filter(q => q.id !== id));
    setDelConfirm(null);
    setPageStart(p => Math.max(0, p - (p % 3 === 0 && filtered.length % 3 === 1 ? 3 : 0)));
  };

  const catCounts = quotes.reduce((a, q) => { a[q.category] = (a[q.category] || 0) + 1; return a; }, {});

  // Featured 3 (sliding window)
  const featuredThree = filtered.slice(pageStart, pageStart + 3);
  const canPrev = pageStart > 0;
  const canNext = pageStart + 3 < filtered.length;

  // Rest of quotes (after the featured 3)
  const restQuotes = showAll ? filtered : filtered.slice(0, Math.min(filtered.length, 12));

  // One spotlight quote per category (when viewing All)
  // MUST be before any early returns to satisfy React hook rules
  const CAT_PALETTE = {
    Mindset:     { accent: "#3b82f6", icon: "🧠", bg: "rgba(59,130,246,0.06)"  },
    Risk:        { accent: "#f87171", icon: "⚡", bg: "rgba(248,113,113,0.06)" },
    Discipline:  { accent: "#f59e0b", icon: "🔱", bg: "rgba(245,158,11,0.06)"  },
    Execution:   { accent: "#4ade80", icon: "🎯", bg: "rgba(74,222,128,0.06)"  },
    Preparation: { accent: "#a78bfa", icon: "📐", bg: "rgba(167,139,250,0.06)" },
    Psychology:  { accent: "#22d3ee", icon: "🌊", bg: "rgba(34,211,238,0.06)"  },
    Strategy:    { accent: "#fb923c", icon: "♟", bg: "rgba(251,146,60,0.06)"   },
    Other:       { accent: "#94a3b8", icon: "✦", bg: "rgba(148,163,184,0.06)"  },
  };

  const spotlightsByCat = useMemo(() => {
    const cats = QUOTE_CATS.filter(c => c !== "All");
    return cats.map(cat => {
      const pool = quotes.filter(q => q.category === cat);
      if (!pool.length) return null;
      return { cat, q: pool[Math.floor(Math.random() * pool.length)], ...CAT_PALETTE[cat] };
    }).filter(Boolean);
  }, [quotes]);

  if (!loaded) return (
    <div style={{ padding: 80, textAlign: "center", color: "#1e3a5f", fontFamily: "'DM Mono',monospace", letterSpacing: "0.15em", fontSize: 12 }}>LOADING QUOTES…</div>
  );

  const ACCENT_COLORS = ["#3b82f6", "#00ff88", "#f59e0b"];



  return (
    <div style={{ fontFamily: "'DM Mono',monospace" }}>
      <style>{`
        @keyframes qCardIn { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
        @keyframes formSlide { from{opacity:0;transform:translateY(-10px)} to{opacity:1;transform:translateY(0)} }
        .q-feat-card { transition: border-color .2s, transform .2s; }
        .q-feat-card:hover { transform: translateY(-3px); }
        .q-grid-card { transition: all .18s ease; }
        .q-grid-card:hover { border-color: #1e3a5f !important; background: #090e1c !important; }
        .q-cat-btn { transition: all .15s; }
        .q-cat-btn:hover { border-color: #1e3a5f !important; color: #93c5fd !important; }
        .q-nav-btn:hover { border-color: #3b82f6 !important; color: #93c5fd !important; }
      `}</style>

      {/* ── PAGE HEADER ── */}
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom: 20 }}>
        <div>
          <div style={{ fontFamily:"'Bebas Neue',sans-serif", fontSize:22, letterSpacing:"0.1em", lineHeight:1, background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>TRADER QUOTES</div>
          <div style={{ fontSize:9, color:"#3b82f6", letterSpacing:"0.12em", marginTop:4 }}>{quotes.length} QUOTES SAVED</div>
        </div>
        <button onClick={openAdd} style={{ background:"#1d4ed8", color:"white", border:"none", padding:"9px 20px", borderRadius:4, fontFamily:"inherit", fontSize:11, cursor:"pointer", letterSpacing:"0.08em" }}>+ ADD QUOTE</button>
      </div>

      {/* ── CATEGORY FILTER ── */}
      <div style={{ display:"flex", gap:6, flexWrap:"wrap", marginBottom:20 }}>
        {QUOTE_CATS.map(cat => {
          const cnt = cat === "All" ? quotes.length : (catCounts[cat]||0);
          const on = activeCat === cat;
          return (
            <button key={cat} className="q-cat-btn" onClick={() => setActiveCat(cat)}
              style={{ padding:"5px 12px", borderRadius:3, fontFamily:"inherit", fontSize:9, cursor:"pointer", letterSpacing:"0.08em", background: on?"#0a1628":"transparent", border:`1px solid ${on?"#1e3a5f":"#0f1729"}`, color: on?"#93c5fd": cnt>0?"#475569":"#1e293b" }}>
              {cat.toUpperCase()}{cnt>0?` · ${cnt}`:""}
            </button>
          );
        })}
      </div>

      {/* ── CATEGORY SPOTLIGHTS (one per category, shown when "All" is active) ── */}
      {activeCat === "All" && spotlightsByCat.length > 0 && (
        <div style={{ marginBottom: 32 }}>
          <div style={{ fontSize: 9, color: "#1e3a5f", letterSpacing: "0.18em", marginBottom: 16 }}>✦ ONE FROM EVERY CATEGORY ✦</div>
          {/* Hero row — first 2 categories big */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
            {spotlightsByCat.slice(0, 2).map(({ cat, q, accent, icon, bg }, i) => (
              <div key={cat} className="q-feat-card"
                style={{ background: "#060b18", border: `1px solid ${accent}28`, borderRadius: 10, padding: "40px 36px 34px", position: "relative", overflow: "hidden", animation: `qCardIn 0.35s ease ${i*0.08}s both` }}>
                <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, transparent, ${accent}cc, transparent)` }} />
                <div style={{ position: "absolute", inset: 0, background: `radial-gradient(ellipse at 20% 0%, ${bg} 0%, transparent 60%)`, pointerEvents: "none" }} />
                <div style={{ position: "absolute", right: 16, top: 10, fontSize: 42, opacity: 0.12, userSelect: "none", lineHeight: 1 }}>{icon}</div>
                <div style={{ position: "absolute", right: 14, bottom: -4, fontFamily: "Georgia,serif", fontSize: 100, color: accent, opacity: 0.04, lineHeight: 1, userSelect: "none" }}>"</div>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
                  <span style={{ fontSize: 14 }}>{icon}</span>
                  <span style={{ fontSize: 9, color: accent, letterSpacing: "0.18em", fontFamily: "'DM Mono',monospace" }}>{cat.toUpperCase()}</span>
                </div>
                <div style={{ fontFamily: "Georgia,'Times New Roman',serif", fontStyle: "italic", fontSize: 17, color: "#dbeafe", lineHeight: 1.85, marginBottom: 22, position: "relative", zIndex: 1 }}>
                  "{q.text}"
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <div style={{ width: 20, height: 1, background: `${accent}66` }} />
                  <span style={{ fontFamily: "'DM Mono',monospace", fontSize: 9, color: accent, letterSpacing: "0.16em" }}>{q.author.toUpperCase()}</span>
                </div>
                <div style={{ position: "absolute", top: 10, right: 10, display: "flex", gap: 3, opacity: 0 }} className="q-feat-actions"
                  onMouseEnter={e=>e.currentTarget.style.opacity="1"} onMouseLeave={e=>e.currentTarget.style.opacity="0"}>
                  <button onClick={()=>openEdit(q)} style={{ background:"transparent",border:"none",color:"#475569",fontSize:11,cursor:"pointer",padding:"2px 4px",transition:"color .15s",fontFamily:"inherit" }}
                    onMouseEnter={e=>e.currentTarget.style.color=accent} onMouseLeave={e=>e.currentTarget.style.color="#334155"}>✎</button>
                  <button onClick={()=>setDelConfirm(q.id)} style={{ background:"transparent",border:"none",color:"#475569",fontSize:11,cursor:"pointer",padding:"2px 4px",transition:"color .15s",fontFamily:"inherit" }}
                    onMouseEnter={e=>e.currentTarget.style.color="#f87171"} onMouseLeave={e=>e.currentTarget.style.color="#334155"}>✕</button>
                </div>
              </div>
            ))}
          </div>
          {/* Medium row — next 3 categories */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 10 }}>
            {spotlightsByCat.slice(2, 5).map(({ cat, q, accent, icon, bg }, i) => (
              <div key={cat} className="q-feat-card"
                style={{ background: "#060b18", border: `1px solid ${accent}22`, borderRadius: 8, padding: "30px 26px 26px", position: "relative", overflow: "hidden", animation: `qCardIn 0.35s ease ${(i+2)*0.07}s both` }}>
                <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 1.5, background: `linear-gradient(90deg, transparent, ${accent}aa, transparent)` }} />
                <div style={{ position: "absolute", inset: 0, background: `radial-gradient(ellipse at 50% 0%, ${bg} 0%, transparent 55%)`, pointerEvents: "none" }} />
                <div style={{ position: "absolute", right: 10, bottom: -2, fontFamily: "Georgia,serif", fontSize: 72, color: accent, opacity: 0.04, lineHeight: 1, userSelect: "none" }}>"</div>
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 11 }}>
                  <span style={{ fontSize: 11 }}>{icon}</span>
                  <span style={{ fontSize: 8, color: accent, letterSpacing: "0.16em", fontFamily: "'DM Mono',monospace" }}>{cat.toUpperCase()}</span>
                </div>
                <div style={{ fontFamily: "Georgia,'Times New Roman',serif", fontStyle: "italic", fontSize: 14, color: "#c8d8f0", lineHeight: 1.82, marginBottom: 18, position: "relative", zIndex: 1 }}>
                  "{q.text.length > 120 ? q.text.slice(0,120)+"…" : q.text}"
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{ width: 14, height: 1, background: `${accent}55` }} />
                  <span style={{ fontFamily: "'DM Mono',monospace", fontSize: 8, color: accent, letterSpacing: "0.14em" }}>{q.author.toUpperCase()}</span>
                </div>
                <div style={{ position: "absolute", top: 7, right: 7, display: "flex", gap: 2, opacity: 0 }} className="q-feat-actions"
                  onMouseEnter={e=>e.currentTarget.style.opacity="1"} onMouseLeave={e=>e.currentTarget.style.opacity="0"}>
                  <button onClick={()=>openEdit(q)} style={{ background:"transparent",border:"none",color:"#475569",fontSize:10,cursor:"pointer",padding:"2px 3px",fontFamily:"inherit" }}
                    onMouseEnter={e=>e.currentTarget.style.color=accent} onMouseLeave={e=>e.currentTarget.style.color="#334155"}>✎</button>
                  <button onClick={()=>setDelConfirm(q.id)} style={{ background:"transparent",border:"none",color:"#475569",fontSize:10,cursor:"pointer",padding:"2px 3px",fontFamily:"inherit" }}
                    onMouseEnter={e=>e.currentTarget.style.color="#f87171"} onMouseLeave={e=>e.currentTarget.style.color="#334155"}>✕</button>
                </div>
              </div>
            ))}
          </div>
          {/* Compact strip — remaining categories */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
            {spotlightsByCat.slice(5).map(({ cat, q, accent, icon, bg }, i) => (
              <div key={cat} className="q-feat-card"
                style={{ background: "#060b18", border: `1px solid ${accent}1a`, borderRadius: 7, padding: "24px 22px 20px", position: "relative", overflow: "hidden", animation: `qCardIn 0.35s ease ${(i+5)*0.06}s both` }}>
                <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 1, background: `linear-gradient(90deg, transparent, ${accent}88, transparent)` }} />
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
                  <span style={{ fontSize: 10 }}>{icon}</span>
                  <span style={{ fontSize: 8, color: accent, letterSpacing: "0.14em", fontFamily: "'DM Mono',monospace" }}>{cat.toUpperCase()}</span>
                </div>
                <div style={{ fontFamily: "Georgia,'Times New Roman',serif", fontStyle: "italic", fontSize: 11, color: "#94a3b8", lineHeight: 1.68, marginBottom: 10, position: "relative", zIndex: 1 }}>
                  "{q.text.length > 100 ? q.text.slice(0,100)+"…" : q.text}"
                </div>
                <div style={{ fontSize: 8, color: `${accent}99`, fontFamily: "'DM Mono',monospace", letterSpacing: "0.12em" }}>— {q.author.toUpperCase()}</div>
                <div style={{ position: "absolute", top: 6, right: 6, display: "flex", gap: 2, opacity: 0 }} className="q-feat-actions"
                  onMouseEnter={e=>e.currentTarget.style.opacity="1"} onMouseLeave={e=>e.currentTarget.style.opacity="0"}>
                  <button onClick={()=>openEdit(q)} style={{ background:"transparent",border:"none",color:"#475569",fontSize:10,cursor:"pointer",padding:"2px 3px",fontFamily:"inherit" }}
                    onMouseEnter={e=>e.currentTarget.style.color=accent} onMouseLeave={e=>e.currentTarget.style.color="#334155"}>✎</button>
                  <button onClick={()=>setDelConfirm(q.id)} style={{ background:"transparent",border:"none",color:"#475569",fontSize:10,cursor:"pointer",padding:"2px 3px",fontFamily:"inherit" }}
                    onMouseEnter={e=>e.currentTarget.style.color="#f87171"} onMouseLeave={e=>e.currentTarget.style.color="#334155"}>✕</button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── FEATURED 3 (when a specific category is active) ── */}
      {activeCat !== "All" && filtered.length > 0 && (
        <div style={{ marginBottom: 28 }}>
          <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:14 }}>
            <div style={{ fontSize:9, color:"#3b82f6", letterSpacing:"0.14em" }}>
              {activeCat.toUpperCase()} · {pageStart + 1}–{Math.min(pageStart + 3, filtered.length)} OF {filtered.length}
            </div>
            {(canPrev || canNext) && (
              <div style={{ display:"flex", gap:6 }}>
                <button className="q-nav-btn" onClick={() => setPageStart(p => Math.max(0, p - 3))} disabled={!canPrev}
                  style={{ background:"transparent", border:`1px solid ${canPrev?"#1e293b":"#0a1220"}`, color: canPrev?"#475569":"#0f1729", width:28, height:28, borderRadius:"50%", cursor: canPrev?"pointer":"default", fontSize:13, display:"flex", alignItems:"center", justifyContent:"center", transition:"all .15s", fontFamily:"inherit" }}>‹</button>
                <button className="q-nav-btn" onClick={() => setPageStart(p => Math.min(p + 3, Math.floor((filtered.length-1)/3)*3))} disabled={!canNext}
                  style={{ background:"transparent", border:`1px solid ${canNext?"#1e293b":"#0a1220"}`, color: canNext?"#475569":"#0f1729", width:28, height:28, borderRadius:"50%", cursor: canNext?"pointer":"default", fontSize:13, display:"flex", alignItems:"center", justifyContent:"center", transition:"all .15s", fontFamily:"inherit" }}>›</button>
              </div>
            )}
          </div>
          <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:14 }}>
            {featuredThree.map((q, i) => {
              const palette = CAT_PALETTE[activeCat] || { accent:"#3b82f6", icon:"✦", bg:"rgba(59,130,246,0.06)" };
              const accent = palette.accent;
              return (
                <div key={q.id} className="q-feat-card"
                  style={{ background:"#060b18", border:`1px solid ${accent}28`, borderRadius:8, padding:"34px 28px 28px", position:"relative", overflow:"hidden", animation:`qCardIn 0.3s ease ${i*0.07}s both` }}>
                  <div style={{ position:"absolute", top:0, left:0, right:0, height:2, background:`linear-gradient(90deg, transparent, ${accent}, transparent)`, opacity:0.7 }}/>
                  <div style={{ position:"absolute", top:-4, right:12, fontFamily:"Georgia,serif", fontSize:80, color:accent, opacity:0.05, lineHeight:1, userSelect:"none", pointerEvents:"none" }}>"</div>
                  <div style={{ fontSize:8, color:accent, letterSpacing:"0.16em", marginBottom:14, opacity:0.8 }}>{palette.icon} {q.category.toUpperCase()}</div>
                  <div style={{ fontFamily:"Georgia,'Times New Roman',serif", fontStyle:"italic", fontSize:13, color:"#c8d8f0", lineHeight:1.75, marginBottom:20, minHeight:80 }}>
                    "{q.text}"
                  </div>
                  <div style={{ display:"flex", alignItems:"center", gap:10 }}>
                    <div style={{ flex:1, height:1, background:`linear-gradient(90deg, ${accent}44, transparent)` }}/>
                    <div style={{ fontFamily:"'DM Mono',monospace", fontStyle:"normal", fontSize:9, color:accent, letterSpacing:"0.16em" }}>{q.author.toUpperCase()}</div>
                  </div>
                  <div style={{ position:"absolute", bottom:14, right:14, display:"flex", gap:4, opacity:0 }} className="q-feat-actions"
                    onMouseEnter={e => e.currentTarget.style.opacity="1"}
                    onMouseLeave={e => e.currentTarget.style.opacity="0"}>
                    <button onClick={() => openEdit(q)} style={{ background:"transparent", border:"none", color:"#64748b", fontSize:11, cursor:"pointer", padding:"2px 4px", transition:"color .15s", fontFamily:"inherit" }}
                      onMouseEnter={e=>e.currentTarget.style.color=accent} onMouseLeave={e=>e.currentTarget.style.color="#334155"}>✎</button>
                    <button onClick={() => setDelConfirm(q.id)} style={{ background:"transparent", border:"none", color:"#64748b", fontSize:11, cursor:"pointer", padding:"2px 4px", transition:"color .15s", fontFamily:"inherit" }}
                      onMouseEnter={e=>e.currentTarget.style.color="#f87171"} onMouseLeave={e=>e.currentTarget.style.color="#334155"}>✕</button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── ADD / EDIT FORM ── */}
      {showForm && (
        <div style={{ background:"#060b18", border:"1px solid #1e3a5f", borderRadius:6, padding:"18px 20px", marginBottom:20, animation:"formSlide 0.22s ease" }}>
          <div style={{ fontSize:10, color:"#93c5fd", letterSpacing:"0.12em", marginBottom:14 }}>{editId ? "EDIT QUOTE" : "NEW QUOTE"}</div>
          {qErr && <div style={{ fontSize:10, color:"#f87171", background:"#1f0606", border:"1px solid #7f1d1d", borderRadius:4, padding:"6px 12px", marginBottom:10 }}>{qErr}</div>}
          <textarea placeholder="Quote text…" value={qForm.text} onChange={e=>setQForm(p=>({...p,text:e.target.value}))} rows={3}
            style={{ width:"100%", fontFamily:"Georgia,serif", fontStyle:"italic", background:"#0a0e1a", border:"1px solid #1e293b", borderRadius:4, color:"#dbeafe", fontSize:13, padding:"10px 12px", resize:"vertical", outline:"none", lineHeight:1.65, marginBottom:10, boxSizing:"border-box" }}
            onFocus={e=>e.target.style.borderColor="#3b82f6"} onBlur={e=>e.target.style.borderColor="#1e293b"}/>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 180px", gap:10, marginBottom:12 }}>
            <input type="text" placeholder="Author…" value={qForm.author} onChange={e=>setQForm(p=>({...p,author:e.target.value}))}
              style={{ fontFamily:"'DM Mono',monospace", background:"#0a0e1a", border:"1px solid #1e293b", borderRadius:4, color:"#e2e8f0", fontSize:12, padding:"9px 12px", outline:"none" }}
              onFocus={e=>e.target.style.borderColor="#3b82f6"} onBlur={e=>e.target.style.borderColor="#1e293b"}/>
            <select value={qForm.category} onChange={e=>setQForm(p=>({...p,category:e.target.value}))}
              style={{ fontFamily:"'DM Mono',monospace", background:"#0a0e1a", border:"1px solid #1e293b", borderRadius:4, color:"#94a3b8", fontSize:11, padding:"9px 10px", outline:"none" }}>
              {QUOTE_CATS.filter(c=>c!=="All").map(c=><option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div style={{ display:"flex", gap:8, justifyContent:"flex-end" }}>
            <button onClick={cancelForm} style={{ background:"transparent", border:"1px solid #1e293b", color:"#3b82f6", padding:"7px 16px", borderRadius:4, fontFamily:"inherit", fontSize:10, cursor:"pointer", letterSpacing:"0.06em" }}>CANCEL</button>
            <button onClick={submitForm} style={{ background:"#1d4ed8", color:"white", border:"none", padding:"7px 18px", borderRadius:4, fontFamily:"inherit", fontSize:10, cursor:"pointer", letterSpacing:"0.06em" }}>{editId?"UPDATE":"SAVE QUOTE"}</button>
          </div>
        </div>
      )}

      {/* ── ALL QUOTES GRID ── */}
      {filtered.length > 3 && (
        <div>
          <div style={{ fontSize:9, color:"#1e293b", letterSpacing:"0.12em", marginBottom:12 }}>
            {activeCat === "All" ? "ALL QUOTES" : activeCat.toUpperCase()} · {filtered.length}
          </div>
          <div style={{ display:"grid", gridTemplateColumns:"repeat(2,1fr)", gap:8 }}>
            {restQuotes.map((q) => (
              <div key={q.id} className="q-grid-card"
                style={{ background:"#060b18", border:"1px solid #0a1220", borderRadius:6, padding:"14px 16px", cursor:"default", position:"relative", minHeight:100 }}>
                <div style={{ fontSize:8, color:"#1e3a5f", letterSpacing:"0.14em", marginBottom:8 }}>{q.category.toUpperCase()}</div>
                <div style={{ fontFamily:"Georgia,serif", fontStyle:"italic", fontSize:11, color:"#64748b", lineHeight:1.65, marginBottom:10 }}>
                  "{q.text.length > 130 ? q.text.slice(0,130)+"…" : q.text}"
                </div>
                <div style={{ display:"flex", alignItems:"center", gap:7 }}>
                  <div style={{ width:12, height:1, background:"#1e293b" }}/>
                  <span style={{ fontSize:9, color:"#3b82f6", letterSpacing:"0.1em", fontFamily:"'DM Mono',monospace", fontStyle:"normal" }}>{q.author.toUpperCase()}</span>
                </div>
                <div style={{ position:"absolute", top:8, right:8, display:"flex", gap:3 }}>
                  <button onClick={()=>openEdit(q)} style={{ background:"transparent",border:"none",color:"#0f1e30",fontSize:10,cursor:"pointer",padding:"1px 3px",transition:"color .15s",fontFamily:"inherit" }}
                    onMouseEnter={e=>e.currentTarget.style.color="#3b82f6"} onMouseLeave={e=>e.currentTarget.style.color="#0f1e30"}>✎</button>
                  <button onClick={()=>setDelConfirm(q.id)} style={{ background:"transparent",border:"none",color:"#0f1e30",fontSize:10,cursor:"pointer",padding:"1px 3px",transition:"color .15s",fontFamily:"inherit" }}
                    onMouseEnter={e=>e.currentTarget.style.color="#f87171"} onMouseLeave={e=>e.currentTarget.style.color="#0f1e30"}>✕</button>
                </div>
              </div>
            ))}
          </div>
          {filtered.length > 12 && (
            <div style={{ textAlign:"center", marginTop:12 }}>
              <button onClick={()=>setShowAll(p=>!p)} style={{ background:"transparent", border:"1px solid #0f1729", color:"#3b82f6", padding:"7px 20px", borderRadius:4, fontFamily:"inherit", fontSize:10, cursor:"pointer", letterSpacing:"0.08em", transition:"all .15s" }}
                onMouseEnter={e=>{e.currentTarget.style.borderColor="#1e3a5f";e.currentTarget.style.color="#94a3b8";}}
                onMouseLeave={e=>{e.currentTarget.style.borderColor="#0f1729";e.currentTarget.style.color="#334155";}}>
                {showAll ? "SHOW LESS ▲" : `SHOW ALL ${filtered.length} ▼`}
              </button>
            </div>
          )}
        </div>
      )}

      {/* ── DELETE CONFIRM ── */}
      {delConfirm && (
        <div onClick={()=>setDelConfirm(null)} style={{ position:"fixed", inset:0, background:"rgba(0,0,0,0.75)", zIndex:9999, display:"flex", alignItems:"center", justifyContent:"center" }}>
          <div onClick={e=>e.stopPropagation()} style={{ background:"#0a0e1a", border:"1px solid #7f1d1d", borderRadius:8, padding:"28px 32px", maxWidth:360, textAlign:"center" }}>
            <div style={{ fontSize:13, color:"#e2e8f0", marginBottom:6, letterSpacing:"0.04em" }}>Remove this quote?</div>
            <div style={{ fontSize:10, color:"#64748b", marginBottom:22 }}>This can't be undone.</div>
            <div style={{ display:"flex", gap:10, justifyContent:"center" }}>
              <button onClick={()=>setDelConfirm(null)} style={{ background:"transparent", border:"1px solid #1e293b", color:"#64748b", padding:"7px 18px", borderRadius:4, fontFamily:"inherit", fontSize:10, cursor:"pointer" }}>CANCEL</button>
              <button onClick={()=>doDelete(delConfirm)} style={{ background:"#450a0a", color:"#f87171", border:"1px solid #7f1d1d", padding:"7px 18px", borderRadius:4, fontFamily:"inherit", fontSize:10, cursor:"pointer" }}>REMOVE</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}


// ─────────────────────────────────────────────────────────────────────────────
// CERTIFICATES & ACHIEVEMENTS TAB
// ─────────────────────────────────────────────────────────────────────────────
function CertificatesTab({ certs, setCerts, certLightbox, setCertLightbox }) {
  const zoneRef = useRef(null);
  const fileInputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [editLabel, setEditLabel] = useState("");

  const persist = (updated) => {
    setCerts(updated);
    try { localStorage.setItem("tj-certs-v1", JSON.stringify(updated)); } catch {}
  };

  const addFromDataUrl = (src, name = "") => {
    const id = `cert-${Date.now()}-${Math.random().toString(36).slice(2,7)}`;
    const label = name.replace(/\.[^.]+$/, "").replace(/[-_]/g, " ") || "";
    persist([...certs, { id, src, label, addedAt: new Date().toISOString() }]);
  };

  const handlePaste = (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of [...items]) {
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        const reader = new FileReader();
        reader.onload = ev => addFromDataUrl(ev.target.result, "");
        reader.readAsDataURL(file);
      }
    }
  };

  const handleDrop = (e) => {
    e.preventDefault(); setDragging(false);
    const files = [...(e.dataTransfer?.files || [])].filter(f => f.type.startsWith("image/"));
    files.forEach(file => {
      const reader = new FileReader();
      reader.onload = ev => addFromDataUrl(ev.target.result, file.name);
      reader.readAsDataURL(file);
    });
  };

  const handleFileInput = (e) => {
    const files = [...(e.target.files || [])].filter(f => f.type.startsWith("image/"));
    files.forEach(file => {
      const reader = new FileReader();
      reader.onload = ev => addFromDataUrl(ev.target.result, file.name);
      reader.readAsDataURL(file);
    });
    e.target.value = "";
  };

  const remove = (id) => persist(certs.filter(c => c.id !== id));

  const saveLabel = (id) => {
    persist(certs.map(c => c.id === id ? { ...c, label: editLabel } : c));
    setEditingId(null);
  };

  const GOLD   = "#f59e0b";
  const SILVER = "#94a3b8";

  return (
    <div style={{ fontFamily: "'DM Mono',monospace" }}>
      <style>{`
        @keyframes certIn { from { opacity:0; transform:translateY(16px) scale(0.97); } to { opacity:1; transform:translateY(0) scale(1); } }
        @keyframes shimmer { 0%,100% { opacity:.5; } 50% { opacity:1; } }
        .cert-card:hover .cert-overlay { opacity: 1 !important; }
        .cert-card:hover { transform: translateY(-4px) !important; box-shadow: 0 12px 40px rgba(245,158,11,0.18) !important; border-color: rgba(245,158,11,0.4) !important; }
        .cert-upload-btn:hover { border-color: #f59e0b !important; color: #f59e0b !important; background: rgba(245,158,11,0.06) !important; }
      `}</style>

      {/* ── HEADER ── */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
          <div>
            <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 22, color: GOLD, letterSpacing: "0.1em", lineHeight: 1 }}>
              🎖 CERTIFICATES & ACHIEVEMENTS
            </div>
            <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.14em", marginTop: 4 }}>
              {certs.length === 0 ? "PASTE · DRAG & DROP · OR UPLOAD YOUR PROP FIRM CERTIFICATES" : `${certs.length} CERTIFICATE${certs.length !== 1 ? "S" : ""} · PASTE, DRAG, OR CLICK UPLOAD TO ADD MORE`}
            </div>
          </div>
          {certs.length > 0 && (
            <div style={{ fontSize: 28, animation: "shimmer 2.5s ease-in-out infinite" }}>🏆</div>
          )}
        </div>
        {/* Gold accent line */}
        <div style={{ height: 1, background: "linear-gradient(90deg, #f59e0b55, #f59e0b22, transparent)", marginTop: 10 }} />
      </div>

      {/* ── PASTE / DROP ZONE ── */}
      <div
        ref={zoneRef} tabIndex={0} onPaste={handlePaste}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => zoneRef.current?.focus()}
        style={{
          border: `2px dashed ${dragging ? GOLD : "rgba(245,158,11,0.25)"}`,
          borderRadius: 10, padding: certs.length ? "14px 20px" : "36px 24px",
          textAlign: "center", outline: "none", cursor: "text",
          background: dragging ? "rgba(245,158,11,0.04)" : "rgba(245,158,11,0.02)",
          transition: "all .2s", marginBottom: 20,
          display: "flex", alignItems: "center", justifyContent: "center", gap: 16,
        }}>
        <div style={{ fontSize: certs.length ? 20 : 36, opacity: dragging ? 1 : 0.35 }}>🏅</div>
        <div>
          <div style={{ fontSize: certs.length ? 11 : 13, color: dragging ? GOLD : "#64748b", lineHeight: 1.6, transition: "color .2s" }}>
            {dragging ? "Drop your certificate here!" : (<>Click here, then <span style={{ color: GOLD }}>Ctrl+V</span> / <span style={{ color: GOLD }}>⌘V</span> to paste{certs.length ? " another" : " a certificate or screenshot"}</>)}
          </div>
          {!certs.length && !dragging && (
            <div style={{ fontSize: 10, color: "#334155", marginTop: 4 }}>Or drag & drop · Or use the upload button below</div>
          )}
        </div>
        {certs.length > 0 && !dragging && (
          <div style={{ fontSize: 10, color: "#334155" }}>or drag & drop</div>
        )}
      </div>

      {/* ── ACTION BUTTONS ── */}
      <div style={{ display: "flex", gap: 8, marginBottom: certs.length ? 28 : 0 }}>
        <button className="cert-upload-btn" onClick={() => fileInputRef.current?.click()}
          style={{ background: "transparent", border: "1px solid #1e293b", color: "#3b82f6", padding: "8px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 10, cursor: "pointer", letterSpacing: "0.1em", transition: "all .18s" }}>
          ↑ UPLOAD FILE
        </button>
        <input ref={fileInputRef} type="file" accept="image/*" multiple onChange={handleFileInput} style={{ display: "none" }} />
        {certs.length > 0 && (
          <div style={{ marginLeft: "auto", fontSize: 9, color: "#1e293b", display: "flex", alignItems: "center", letterSpacing: "0.1em" }}>
            HOVER CARD TO EDIT OR REMOVE
          </div>
        )}
      </div>

      {/* ── EMPTY STATE ── */}
      {certs.length === 0 && (
        <div style={{ textAlign: "center", padding: "40px 20px", border: "1px solid #0f1729", borderRadius: 8, background: "#060b18" }}>
          <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.3 }}>🏆</div>
          <div style={{ fontSize: 13, color: "#334155", letterSpacing: "0.06em", marginBottom: 8 }}>Your wall of achievements is empty</div>
          <div style={{ fontSize: 10, color: "#1e293b", letterSpacing: "0.1em", lineHeight: 1.7 }}>
            PASTE · DRAG & DROP · UPLOAD<br />
            Your funded certificates, passing screenshots, payout confirmations — anything that proves how far you've come.
          </div>
        </div>
      )}

      {/* ── CERTIFICATE GRID ── */}
      {certs.length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 16 }}>
          {certs.map((cert, idx) => (
            <div key={cert.id} className="cert-card"
              style={{ position: "relative", borderRadius: 10, overflow: "hidden", border: "1px solid rgba(245,158,11,0.2)", background: "#060b18", cursor: "zoom-in", transition: "all .25s ease", animation: `certIn 0.35s ease ${idx * 0.07}s both`, boxShadow: "0 4px 20px rgba(0,0,0,0.4)" }}
              onClick={() => setCertLightbox(cert)}>

              {/* Gold top bar */}
              <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, transparent, ${GOLD}, transparent)`, zIndex: 2 }} />

              {/* Certificate image */}
              <img src={cert.src} alt={cert.label || `Certificate ${idx + 1}`}
                style={{ display: "block", width: "100%", height: 200, objectFit: "cover", objectPosition: "top", background: "#030508" }} />

              {/* Bottom label bar */}
              <div style={{ padding: "10px 14px 12px", background: "linear-gradient(to bottom, #060b18, #04060f)" }}>
                {editingId === cert.id ? (
                  <div onClick={e => e.stopPropagation()} style={{ display: "flex", gap: 6 }}>
                    <input
                      autoFocus value={editLabel}
                      onChange={e => setEditLabel(e.target.value)}
                      onKeyDown={e => { if (e.key === "Enter") saveLabel(cert.id); if (e.key === "Escape") setEditingId(null); }}
                      style={{ flex: 1, background: "#0a0e1a", border: "1px solid #f59e0b44", borderRadius: 3, color: "#e2e8f0", fontFamily: "inherit", fontSize: 11, padding: "4px 8px", outline: "none" }}
                      placeholder="Label this certificate…"
                    />
                    <button onClick={() => saveLabel(cert.id)}
                      style={{ background: "#f59e0b22", border: "1px solid #f59e0b44", color: GOLD, padding: "4px 10px", borderRadius: 3, fontSize: 10, cursor: "pointer", fontFamily: "inherit" }}>✓</button>
                  </div>
                ) : (
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                    <div style={{ fontSize: 10, color: cert.label ? "#e2e8f0" : "#334155", letterSpacing: "0.08em", fontStyle: cert.label ? "normal" : "italic", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {cert.label || "click ✎ to add a label"}
                    </div>
                    <div style={{ fontSize: 8, color: "#1e293b", letterSpacing: "0.06em", flexShrink: 0 }}>
                      {new Date(cert.addedAt).toLocaleDateString("en-US", { month: "short", year: "numeric" }).toUpperCase()}
                    </div>
          </div>
          )}
              </div>

              {/* Hover overlay: edit + remove buttons */}
              <div className="cert-overlay"
                style={{ position: "absolute", top: 8, right: 8, display: "flex", gap: 5, opacity: 0, transition: "opacity .18s", zIndex: 3 }}>
                <button
                  onClick={e => { e.stopPropagation(); setEditLabel(cert.label || ""); setEditingId(cert.id); }}
                  style={{ background: "rgba(0,0,0,0.8)", border: "1px solid #1e3a5f", color: "#93c5fd", padding: "4px 8px", borderRadius: 3, fontSize: 10, cursor: "pointer", fontFamily: "inherit", backdropFilter: "blur(4px)" }}>
                  ✎
                </button>
                <button
                  onClick={e => { e.stopPropagation(); remove(cert.id); }}
                  style={{ background: "rgba(0,0,0,0.8)", border: "1px solid #7f1d1d", color: "#f87171", padding: "4px 8px", borderRadius: 3, fontSize: 10, cursor: "pointer", fontFamily: "inherit", backdropFilter: "blur(4px)" }}>
                  ✕
                </button>
              </div>

              {/* Corner badge */}
              <div style={{ position: "absolute", top: 10, left: 12, fontSize: 14, zIndex: 2, filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.6))" }}>🏅</div>
            </div>
          ))}
        </div>
      )}

      {/* ── LIGHTBOX ── */}
      {certLightbox && (
        <div onClick={() => setCertLightbox(null)}
          style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.95)", zIndex: 9999, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", cursor: "zoom-out", padding: 24 }}>
          {/* Gold top accent */}
          <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, transparent, ${GOLD}, transparent)` }} />
          <img src={certLightbox.src} alt={certLightbox.label || "Certificate"}
            style={{ maxWidth: "90vw", maxHeight: "82vh", borderRadius: 8, boxShadow: `0 0 80px rgba(245,158,11,0.2), 0 0 120px rgba(0,0,0,0.8)`, border: "1px solid rgba(245,158,11,0.25)" }} />
          {certLightbox.label && (
            <div style={{ marginTop: 18, fontFamily: "'Bebas Neue',sans-serif", fontSize: 18, color: GOLD, letterSpacing: "0.2em", textAlign: "center" }}>
              {certLightbox.label.toUpperCase()}
            </div>
          )}
          <div style={{ marginTop: 10, fontSize: 10, color: "#334155", letterSpacing: "0.1em" }}>CLICK ANYWHERE TO CLOSE</div>
        </div>
      )}
    </div>
  );
}



function PropDashInner({ journals, entries, activeJournalId, activeJournal, propStatus, propJournals, allPropEntriesMap, saveJournalsMeta, netPnl, pnlColor }) {
        // State lifted from OverviewTab and AccountTab to fix React #310
        // (hooks cannot live in arrow functions redefined each render)
        const [showArchived, setShowArchived] = useState(false);
        const [payoutModal, setPayoutModal] = useState(false);
        const [payoutForm, setPayoutForm] = useState({ date: new Date().toISOString().split("T")[0], amount: "", fee: "", note: "" });
        const [archiveModal, setArchiveModal] = useState(false);
        const [archiveForm, setArchiveForm] = useState({ breachType: "maxLoss", postMortem: "", evalCost: "" });
        const [certs, setCerts] = useState(() => { try { const r = localStorage.getItem("tj-certs-v1"); return r ? JSON.parse(r) : []; } catch { return []; } });
        const [certLightbox, setCertLightbox] = useState(null);

        const cfg = activeJournal?.config;
        const ps = propStatus;
        const fmt$ = v => `$${Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

        const PctBar = ({ pct, color, warn = 70, danger = 90 }) => {
          const c = pct >= danger ? "#f87171" : pct >= warn ? "#fbbf24" : color;
          return (
            <div style={{ height: 5, background: "#0a0e1a", borderRadius: 3, overflow: "hidden", marginTop: 8 }}>
              <div style={{ width: `${Math.min(100, pct)}%`, height: "100%", background: c, borderRadius: 3, transition: "width .4s" }} />
            </div>
          );
        };
        const RuleRow = ({ label, pass, detail }) => (
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "9px 0", borderBottom: "1px solid #0a1220" }}>
            <div style={{ fontSize: 11, color: "#94a3b8" }}>{label}</div>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span style={{ fontSize: 10, color: "#94a3b8" }}>{detail}</span>
              <span style={{ fontSize: 10, padding: "2px 10px", borderRadius: 12, background: pass ? "rgba(74,222,128,0.1)" : "rgba(248,113,113,0.1)", color: pass ? "#4ade80" : "#f87171", border: `1px solid ${pass ? "rgba(74,222,128,0.2)" : "rgba(248,113,113,0.2)"}` }}>{pass ? "✓ PASS" : "✗ FAIL"}</span>
            </div>
          </div>
        );

        const TAB_STYLE = (active) => ({
          padding: "8px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer",
          letterSpacing: "0.08em", transition: "all .15s",
          background: active ? "#0f1a2e" : "transparent",
          border: `1px solid ${active ? "#1e3a5f" : "#0f1729"}`,
          color: active ? "#93c5fd" : "#64748b",
        });

        const OverviewTab = () => {
          if (propJournals.length === 0) return (
            <div style={{ color: "#64748b", fontSize: 12, padding: "40px 0", textAlign: "center" }}>
              No prop accounts found. Create a Prop journal to get started.
            </div>
          );
          const activeAccounts   = propJournals.filter(j => !j.config?.isArchived);
          const archivedAccounts = propJournals.filter(j =>  j.config?.isArchived);
          const visibleAccounts  = showArchived ? propJournals : activeAccounts;
          return (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {/* ── TOTAL HARVESTED hero stat ── */}
              {(() => {
                const allWithdrawn = propJournals.reduce((sum, j) => {
                  const jPs = j.config ? calcPropStatus(
                    allPropEntriesMap[j.id] || (j.id === activeJournalId ? entries : []),
                    j.config
                  ) : null;
                  return sum + (jPs?.totalWithdrawn || 0);
                }, 0);
                if (allWithdrawn <= 0) return null;
                return (
                  <div style={{ background: "linear-gradient(135deg, #0c1a0a, #0a1c0e)", border: "1px solid #14532d", borderRadius: 8, padding: "14px 20px", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10, position: "relative", overflow: "hidden" }}>
                    <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg, transparent, #f59e0b, #fbbf24, transparent)" }} />
                    <div>
                      <div style={{ fontSize: 9, color: "#f59e0b", letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: 4 }}>💰 Total Harvested · Lifetime Payouts</div>
                      <div style={{ fontSize: 32, fontWeight: 700, color: "#fbbf24", fontFamily: "'DM Mono',monospace", letterSpacing: "-0.02em", lineHeight: 1 }}>${allWithdrawn.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
                      <div style={{ fontSize: 9, color: "#64748b", marginTop: 4 }}>Cash extracted from prop firms into your bank</div>
                    </div>
                    <div style={{ fontSize: 36, opacity: 0.15 }}>🏦</div>
                  </div>
                );
              })()}

              {/* Header row: count + active/all toggle */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.12em" }}>
                  {activeAccounts.length} ACTIVE{archivedAccounts.length > 0 ? ` · ${archivedAccounts.length} ARCHIVED` : ""} · CLICK ANY CARD TO OPEN
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  {archivedAccounts.length > 0 && (
                    <div style={{ display: "flex", gap: 0, borderRadius: 4, overflow: "hidden", border: "1px solid #1e293b" }}>
                      {[{id: false, label: "ACTIVE"}, {id: true, label: "ALL"}].map(opt => (
                        <button key={String(opt.id)} onClick={() => setShowArchived(opt.id)}
                          style={{ padding: "4px 14px", background: showArchived === opt.id ? "#1e3a5f" : "transparent", color: showArchived === opt.id ? "#93c5fd" : "#64748b", border: "none", fontFamily: "inherit", fontSize: 10, cursor: "pointer", letterSpacing: "0.08em" }}>
                          {opt.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              {/* Responsive grid */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 14 }}>
                {visibleAccounts.map(j => {
                  const jEntries = allPropEntriesMap[j.id] || (j.id === activeJournalId ? entries : []);
                  const jPs = j.config ? calcPropStatus(jEntries, j.config) : null;
                  const isActive = j.id === activeJournalId;
                  const isEvalJ = j.config?.phase === "eval";
                  const isArchJ = !!j.config?.isArchived;
                  const netPnlJ = jPs ? ((isEvalJ ? jPs.evalStats?.cumPnl : jPs.fundedStats?.cumPnl) ?? 0) : 0;
                  const ddPct = jPs && j.config ? (() => {
                    const stats = isEvalJ ? jPs.evalStats : jPs.fundedStats;
                    const limit = isEvalJ ? j.config.eval?.maxLossLimit : j.config.funded?.maxLossLimit;
                    return limit ? Math.min(100, (stats?.trailingDrawdown || 0) / limit * 100) : 0;
                  })() : 0;
                  const ddLimit = jPs && j.config ? (isEvalJ ? j.config.eval?.maxLossLimit : j.config.funded?.maxLossLimit) : 0;
                  const ddUsed = jPs ? ((isEvalJ ? jPs.evalStats?.trailingDrawdown : jPs.fundedStats?.trailingDrawdown) || 0) : 0;
                  const allPass = jPs ? Object.values(jPs.rules).every(Boolean) : null;
                  const pnlColor = netPnlJ > 0 ? "#4ade80" : netPnlJ < 0 ? "#f87171" : "#64748b";
                  const ddColor = ddPct > 75 ? "#f87171" : ddPct > 50 ? "#fbbf24" : "#94a3b8";
                  const withdrawn = jPs?.totalWithdrawn || 0;
                  return (
                    <div key={j.id}
                      onClick={() => { if (!isActive) switchJournal(j.id); setPropDashTab("account"); }}
                      style={{ background: isArchJ ? "#050810" : "#070d1a", border: `1px solid ${isArchJ ? "#1a2030" : isActive ? "#92400e" : "#1e293b"}`, borderRadius: 10, padding: "18px 18px 16px", cursor: "pointer", transition: "all .18s", position: "relative", overflow: "hidden", display: "flex", flexDirection: "column", gap: 14, minHeight: 220, opacity: isArchJ ? 0.55 : 1, filter: isArchJ ? "saturate(0.3)" : "none" }}
                      onMouseEnter={e => { if (!isArchJ) { e.currentTarget.style.borderColor = "#f59e0b"; e.currentTarget.style.background = "#0b1120"; e.currentTarget.style.transform = "translateY(-1px)"; e.currentTarget.style.boxShadow = "0 6px 24px rgba(245,158,11,0.08)"; } }}
                      onMouseLeave={e => { e.currentTarget.style.borderColor = isArchJ ? "#1a2030" : isActive ? "#92400e" : "#1e293b"; e.currentTarget.style.background = isArchJ ? "#050810" : "#070d1a"; e.currentTarget.style.transform = "none"; e.currentTarget.style.boxShadow = "none"; }}>

                      {/* Top accent line */}
                      {!isArchJ && isActive && <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, transparent, ${!isEvalJ ? "#4ade80" : "#f59e0b"} 40%, ${!isEvalJ ? "#16a34a" : "#92400e"} 60%, transparent)` }} />}
                      {!isArchJ && !isEvalJ && !isActive && <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg, transparent, rgba(74,222,128,0.4) 40%, transparent)" }} />}
                      {isArchJ && <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg, transparent, rgba(248,113,113,0.25) 40%, transparent)" }} />}

                      {/* Header */}
                      <div>
                        <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 15, color: isArchJ ? "#64748b" : "#f59e0b", letterSpacing: "0.1em", lineHeight: 1.2, marginBottom: 6 }}>
                          {j.config?.firmName?.toUpperCase() || j.name.toUpperCase()}
                        </div>
                        <div style={{ fontSize: 11, color: "#64748b", letterSpacing: "0.06em", marginBottom: 8 }}>
                          {j.config?.product || "FLEX"} · {j.config ? `${((j.config.accountSize||0)/1000).toFixed(0)}K` : ""}
                        </div>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                          {isArchJ
                            ? <span style={{ fontSize: 9, padding: "2px 8px", borderRadius: 10, background: "rgba(248,113,113,0.1)", color: "#f87171", border: "1px solid rgba(248,113,113,0.2)", letterSpacing: "0.1em", fontWeight: 600 }}>✕ ARCHIVED</span>
                            : <span style={{ fontSize: 9, padding: "2px 8px", borderRadius: 10, background: isEvalJ ? "rgba(59,130,246,0.12)" : "rgba(74,222,128,0.1)", color: isEvalJ ? "#93c5fd" : "#4ade80", border: `1px solid ${isEvalJ ? "rgba(59,130,246,0.25)" : "rgba(74,222,128,0.25)"}`, letterSpacing: "0.1em", fontWeight: 600 }}>
                                {isEvalJ ? "EVAL" : "✓ FUNDED"}
                              </span>
                          }
                          {!isArchJ && allPass !== null && (
                            <span style={{ fontSize: 9, padding: "2px 8px", borderRadius: 10, background: allPass ? "rgba(74,222,128,0.08)" : "rgba(248,113,113,0.08)", color: allPass ? "#4ade80" : "#f87171", border: `1px solid ${allPass ? "rgba(74,222,128,0.2)" : "rgba(248,113,113,0.2)"}`, letterSpacing: "0.06em" }}>
                              {allPass ? "✓ RULES OK" : "⚠ BREACH"}
                            </span>
                          )}
                          {isActive && !isArchJ && <span style={{ fontSize: 9, padding: "2px 8px", borderRadius: 10, background: "rgba(245,158,11,0.1)", color: "#f59e0b", border: "1px solid rgba(245,158,11,0.25)", letterSpacing: "0.06em" }}>● ACTIVE</span>}
                        </div>
                        {!isArchJ && !isEvalJ && j.config?.passedDate && (
                          <div style={{ marginTop: 7, display: "flex", alignItems: "center", gap: 6 }}>
                            <span style={{ fontSize: 9, padding: "3px 10px", borderRadius: 10, background: "rgba(74,222,128,0.12)", color: "#4ade80", border: "1px solid rgba(74,222,128,0.3)", letterSpacing: "0.07em", fontWeight: 600 }}>✓ FUNDED</span>
                            <span style={{ fontSize: 9, color: "#64748b" }}>since {new Date(j.config.passedDate + "T12:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}</span>
                          </div>
                        )}
                        {isArchJ && j.config?.breachedDate && (
                          <div style={{ marginTop: 6, fontSize: 9, color: "#f87171" }}>
                            Breached {new Date(j.config.breachedDate + "T12:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                          </div>
                        )}
                      </div>

                      {/* P&L hero */}
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.12em", marginBottom: 4, textTransform: "uppercase" }}>{isArchJ ? "Final P&L" : isEvalJ ? "Eval Net P&L" : "Funded Net P&L"}</div>
                        <div style={{ fontSize: 28, fontWeight: 700, color: isArchJ ? "#64748b" : pnlColor, letterSpacing: "-0.01em", lineHeight: 1 }}>
                          {netPnlJ >= 0 ? "+" : ""}{fmt$(netPnlJ)}
                        </div>
                        {!isArchJ && withdrawn > 0 && (
                          <div style={{ fontSize: 9, color: "#60a5fa", marginTop: 4, letterSpacing: "0.06em" }}>💰 {fmt$(withdrawn)} withdrawn</div>
                        )}
                      </div>

                      {/* DD bar (hidden for archived) */}
                      {!isArchJ && (
                        <div>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
                            <span style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em" }}>DRAWDOWN USED</span>
                            <span style={{ fontSize: 13, fontWeight: 600, color: ddColor, fontFamily: "'DM Mono',monospace" }}>{ddPct.toFixed(0)}%</span>
                          </div>
                          <PctBar pct={ddPct} color="#3b82f6" warn={50} danger={75} />
                          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
                            <span style={{ fontSize: 9, color: "#64748b" }}>{fmt$(ddUsed)} used</span>
                            <span style={{ fontSize: 9, color: "#64748b" }}>limit {fmt$(ddLimit)}</span>
                          </div>
                        </div>
                      )}
                      {isArchJ && (
                        <div style={{ fontSize: 9, color: "#64748b", fontStyle: "italic", lineHeight: 1.6 }}>
                          {j.config?.postMortem ? j.config.postMortem.slice(0, 80) + (j.config.postMortem.length > 80 ? "…" : "") : "No post-mortem recorded."}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

            {/* GRAVEYARD: collapsed by default, reveals archived accounts */}
            {archivedAccounts.length > 0 && !showArchived && (
              <div style={{ marginTop: 8 }}>
                <button onClick={() => setShowArchived(true)}
                  style={{ width: "100%", background: "transparent", border: "1px dashed #1e293b", borderRadius: 6, padding: "10px 16px", color: "#3b82f6", fontFamily: "inherit", fontSize: 10, cursor: "pointer", letterSpacing: "0.12em", textTransform: "uppercase", transition: "all .15s" }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = "#f87171"; e.currentTarget.style.color = "#f87171"; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = "#1e293b"; e.currentTarget.style.color = "#475569"; }}>
                  ✕ View Graveyard · {archivedAccounts.length} archived account{archivedAccounts.length !== 1 ? "s" : ""}
                </button>
              </div>
            )}
          </div>
          );
        };

        const AccountTab = () => {
          if (!cfg) return <div style={{ color: "#94a3b8", padding: 32 }}>No prop config found for active account.</div>;
          const isEval = cfg.phase === "eval";
          const isFunded = cfg.phase === "funded";
          const isArchived = !!cfg.isArchived;

          const markAsPassed = async () => {
            const today = new Date().toISOString().split("T")[0];
            const updated = journals.map(j => j.id === activeJournalId ? { ...j, config: { ...j.config, phase: "funded", passedDate: today, fundedStartBalance: j.config.accountSize } } : j);
            await saveJournalsMeta(updated);
          };

          const recordPayout = async () => {
            const amt = parseFloat(payoutForm.amount);
            if (!amt || amt <= 0) return;
            const newPayout = { id: Date.now(), date: payoutForm.date, amount: amt, fee: parseFloat(payoutForm.fee) || 0, note: payoutForm.note };
            const existing = cfg.payouts || [];
            const updated = journals.map(j => j.id === activeJournalId ? { ...j, config: { ...j.config, payouts: [...existing, newPayout] } } : j);
            await saveJournalsMeta(updated);
            setPayoutModal(false);
            setPayoutForm({ date: new Date().toISOString().split("T")[0], amount: "", fee: "", note: "" });
          };

          const archiveAccount = async () => {
            const today = new Date().toISOString().split("T")[0];
            const updated = journals.map(j => j.id === activeJournalId ? { ...j, config: { ...j.config, isArchived: true, breachedDate: today, breachType: archiveForm.breachType, postMortem: archiveForm.postMortem, evalCost: parseFloat(archiveForm.evalCost) || cfg.evalCost || 0 } } : j);
            await saveJournalsMeta(updated);
            setArchiveModal(false);
          };

          const unarchiveAccount = async () => {
            const updated = journals.map(j => j.id === activeJournalId ? { ...j, config: { ...j.config, isArchived: false, breachedDate: null, breachType: null } } : j);
            await saveJournalsMeta(updated);
          };

          const payouts = cfg.payouts || [];
          const totalWithdrawn = payouts.reduce((s, p) => s + (parseFloat(p.amount) || 0), 0);

          return (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {/* Payout Modal */}
              {payoutModal && (
                <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.75)", zIndex: 200, display: "flex", alignItems: "center", justifyContent: "center" }}
                  onClick={e => { if (e.target === e.currentTarget) setPayoutModal(false); }}>
                  <div style={{ background: "#070d1a", border: "1px solid #1e3a5f", borderRadius: 10, padding: 28, width: 380, display: "flex", flexDirection: "column", gap: 16 }}>
                    <div style={{ fontSize: 14, color: "#93c5fd", fontWeight: 600, letterSpacing: "0.08em" }}>💰 RECORD PAYOUT / WITHDRAWAL</div>
                    <div style={{ fontSize: 10, color: "#64748b", lineHeight: 1.6 }}>
                      Payout reduces your account equity on the curve but does not count as a trading loss. Your drawdown floor is unaffected.
                    </div>
                    {[
                      { label: "DATE", key: "date", type: "date" },
                      { label: "WITHDRAWAL AMOUNT ($)", key: "amount", type: "number", placeholder: "1000.00" },
                      { label: "PROP FIRM FEE ($) (optional)", key: "fee", type: "number", placeholder: "0.00" },
                      { label: "NOTE (optional)", key: "note", type: "text", placeholder: "e.g. 1st payout" },
                    ].map(f => (
                      <div key={f.key}>
                        <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 5, textTransform: "uppercase" }}>{f.label}</div>
                        <input type={f.type} value={payoutForm[f.key]} placeholder={f.placeholder}
                          onChange={e => setPayoutForm(p => ({ ...p, [f.key]: e.target.value }))}
                          style={{ width: "100%", boxSizing: "border-box", padding: "9px 12px", background: "#060810", border: "1px solid #1e293b", borderRadius: 5, color: "#e2e8f0", fontFamily: "inherit", fontSize: 12 }} />
                      </div>
                    ))}
                    <div style={{ display: "flex", gap: 10 }}>
                      <button onClick={recordPayout} style={{ flex: 1, background: "#1e3a5f", border: "1px solid #3b82f6", color: "#93c5fd", padding: "10px", borderRadius: 6, fontFamily: "inherit", fontSize: 12, cursor: "pointer", letterSpacing: "0.08em", textTransform: "uppercase" }}>RECORD PAYOUT</button>
                      <button onClick={() => setPayoutModal(false)} style={{ background: "transparent", border: "1px solid #1e293b", color: "#64748b", padding: "10px 18px", borderRadius: 6, fontFamily: "inherit", fontSize: 12, cursor: "pointer" }}>CANCEL</button>
                    </div>
                  </div>
                </div>
              )}
              {/* Archive Modal */}
              {archiveModal && (
                <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.75)", zIndex: 200, display: "flex", alignItems: "center", justifyContent: "center" }}
                  onClick={e => { if (e.target === e.currentTarget) setArchiveModal(false); }}>
                  <div style={{ background: "#0a0610", border: "1px solid #7f1d1d", borderRadius: 10, padding: 28, width: 420, display: "flex", flexDirection: "column", gap: 16 }}>
                    <div style={{ fontSize: 14, color: "#f87171", fontWeight: 600, letterSpacing: "0.08em" }}>⚠ ARCHIVE / BREACH THIS ACCOUNT</div>
                    <div style={{ fontSize: 10, color: "#64748b", lineHeight: 1.6 }}>
                      Account moves to the Graveyard. It stays in your records but disappears from the Active dashboard. Capture why it ended.
                    </div>
                    <div>
                      <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 6, textTransform: "uppercase" }}>BREACH TYPE</div>
                      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                        {[{id:"maxLoss",label:"Max Loss Hit"},{id:"dailyLoss",label:"Daily Loss"},{id:"slow_bleed",label:"Slow Bleed"},{id:"manual",label:"Manual Close"}].map(bt => (
                          <button key={bt.id} onClick={() => setArchiveForm(f => ({...f, breachType: bt.id}))}
                            style={{ padding: "5px 12px", borderRadius: 4, fontFamily: "inherit", fontSize: 10, cursor: "pointer", border: `1px solid ${archiveForm.breachType === bt.id ? "#f87171" : "#1e293b"}`, background: archiveForm.breachType === bt.id ? "rgba(248,113,113,0.12)" : "transparent", color: archiveForm.breachType === bt.id ? "#f87171" : "#64748b" }}>
                            {bt.label}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 5, textTransform: "uppercase" }}>EVAL / CHALLENGE COST ($)</div>
                      <input type="number" value={archiveForm.evalCost} placeholder="e.g. 150"
                        onChange={e => setArchiveForm(f => ({...f, evalCost: e.target.value}))}
                        style={{ width: "100%", boxSizing: "border-box", padding: "9px 12px", background: "#060810", border: "1px solid #1e293b", borderRadius: 5, color: "#e2e8f0", fontFamily: "inherit", fontSize: 12 }} />
                      <div style={{ fontSize: 9, color: "#475569", marginTop: 4 }}>Shows up in your cumulative business expenses.</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 5, textTransform: "uppercase" }}>POST-MORTEM (what happened?)</div>
                      <textarea value={archiveForm.postMortem} rows={4} placeholder="e.g. Averaged down on a counter-trend trade after NFP. Ignored the daily loss limit."
                        onChange={e => setArchiveForm(f => ({...f, postMortem: e.target.value}))}
                        style={{ width: "100%", boxSizing: "border-box", padding: "9px 12px", background: "#060810", border: "1px solid #1e293b", borderRadius: 5, color: "#e2e8f0", fontFamily: "inherit", fontSize: 12, resize: "vertical" }} />
                    </div>
                    <div style={{ display: "flex", gap: 10 }}>
                      <button onClick={archiveAccount} style={{ flex: 1, background: "rgba(248,113,113,0.1)", border: "1px solid #7f1d1d", color: "#f87171", padding: "10px", borderRadius: 6, fontFamily: "inherit", fontSize: 12, cursor: "pointer", letterSpacing: "0.08em", textTransform: "uppercase" }}>ARCHIVE ACCOUNT</button>
                      <button onClick={() => setArchiveModal(false)} style={{ background: "transparent", border: "1px solid #1e293b", color: "#64748b", padding: "10px 18px", borderRadius: 6, fontFamily: "inherit", fontSize: 12, cursor: "pointer" }}>CANCEL</button>
                    </div>
                  </div>
                </div>
              )}

              <div style={{ background: isArchived ? "#080510" : "#070d1a", border: `1px solid ${isArchived ? "#7f1d1d" : "#92400e"}`, borderRadius: 8, padding: "20px 24px", position: "relative", overflow: "hidden" }}>
                <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: isArchived ? "linear-gradient(90deg, transparent, #f87171, #7f1d1d, transparent)" : "linear-gradient(90deg, transparent, #f59e0b, #92400e, transparent)" }} />
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
                  <div>
                    <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 24, color: isArchived ? "#f87171" : "#f59e0b", letterSpacing: "0.1em" }}>{isArchived ? "✕ " : "🏆 "}{cfg.firmName?.toUpperCase()} · {cfg.product || "FLEX"} {(cfg.accountSize/1000).toFixed(0)}K</div>
                    <div style={{ display: "flex", gap: 8, marginTop: 6, flexWrap: "wrap" }}>
                      {isArchived
                        ? <span style={{ fontSize: 9, padding: "2px 10px", borderRadius: 12, background: "rgba(248,113,113,0.1)", color: "#f87171", border: "1px solid rgba(248,113,113,0.2)", letterSpacing: "0.1em" }}>✕ ARCHIVED · {cfg.breachType?.toUpperCase().replace("_"," ") || "BREACHED"}</span>
                        : <span style={{ fontSize: 9, padding: "2px 10px", borderRadius: 12, background: isEval ? "rgba(59,130,246,0.12)" : "rgba(74,222,128,0.1)", color: isEval ? "#93c5fd" : "#4ade80", border: `1px solid ${isEval ? "rgba(59,130,246,0.2)" : "rgba(74,222,128,0.2)"}`, letterSpacing: "0.1em" }}>
                            {isEval ? "📋 EVALUATION" : "✓ FUNDED"}
                          </span>
                      }
                      {cfg.passedDate && !isArchived && <span style={{ fontSize: 9, color: "#94a3b8", padding: "2px 6px" }}>Passed eval {cfg.passedDate}</span>}
                      {isArchived && cfg.breachedDate && <span style={{ fontSize: 9, color: "#f87171", padding: "2px 6px" }}>Archived {cfg.breachedDate}</span>}
                      {totalWithdrawn > 0 && <span style={{ fontSize: 9, color: "#60a5fa", padding: "2px 6px", letterSpacing: "0.06em" }}>💰 {fmt$(totalWithdrawn)} withdrawn</span>}
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {isEval && ps && ps.rules.profitTarget && !isArchived && (
                      <button onClick={markAsPassed} style={{ background: "rgba(74,222,128,0.1)", border: "1px solid #166534", color: "#4ade80", padding: "8px 16px", borderRadius: 6, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.08em" }}>✓ MARK EVAL AS PASSED →</button>
                    )}
                    {isEval && ps && !ps.rules.profitTarget && !isArchived && (
                      <button onClick={markAsPassed} style={{ background: "transparent", border: "1px solid #1e293b", color: "#94a3b8", padding: "8px 16px", borderRadius: 6, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.08em" }}>Mark as Passed (Early)</button>
                    )}
                    {isFunded && !isArchived && (
                      <button onClick={() => setPayoutModal(true)} style={{ background: "rgba(59,130,246,0.1)", border: "1px solid #1e3a5f", color: "#60a5fa", padding: "8px 16px", borderRadius: 6, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.08em" }}>💰 RECORD PAYOUT</button>
                    )}
                    {!isArchived && (
                      <button onClick={() => setArchiveModal(true)} style={{ background: "transparent", border: "1px solid #7f1d1d", color: "#f87171", padding: "8px 16px", borderRadius: 6, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.08em" }}>✕ ARCHIVE</button>
                    )}
                    {isArchived && (
                      <button onClick={unarchiveAccount} style={{ background: "transparent", border: "1px solid #1e293b", color: "#64748b", padding: "8px 16px", borderRadius: 6, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.08em" }}>↩ RESTORE</button>
                    )}
                  </div>
                </div>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: ps?.fundedStats ? "1fr 1fr" : "1fr", gap: 12 }}>
                {ps?.evalStats && (
                  <div style={{ background: "#070d1a", border: `1px solid ${isEval ? "#1e3a5f" : "#0f1729"}`, borderRadius: 6, padding: "16px 18px", opacity: isFunded ? 0.7 : 1 }}>
                    <div style={{ fontSize: 10, color: isEval ? "#93c5fd" : "#475569", letterSpacing: "0.1em", marginBottom: 12 }}>📋 EVALUATION {isFunded ? "· COMPLETED" : "· ACTIVE"}</div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                      {[
                        { l: "NET P&L", v: `${ps.evalStats.cumPnl >= 0 ? "+" : ""}${fmt$(ps.evalStats.cumPnl)}`, c: ps.evalStats.cumPnl >= 0 ? "#4ade80" : "#f87171" },
                        { l: "VS TARGET", v: `${ps.evalProfitTargetPct.toFixed(0)}%`, c: ps.evalProfitTargetPct >= 100 ? "#4ade80" : "#93c5fd" },
                        { l: "TRAIL DD", v: fmt$(ps.evalStats.trailingDrawdown), c: ps.evalStats.trailingDrawdown > cfg.eval.maxLossLimit * 0.75 ? "#f87171" : "#e2e8f0" },
                        { l: "DD LIMIT", v: fmt$(cfg.eval.maxLossLimit), c: "#94a3b8" },
                        { l: "DAYS TRADED", v: ps.evalStats.daysTraded, c: "#e2e8f0" },
                        { l: "CONSISTENCY", v: `${ps.evalStats.consistencyPct.toFixed(0)}%`, c: ps.evalStats.consistencyPct > cfg.eval.consistencyRule ? "#f87171" : "#4ade80" },
                      ].map(s => (
                        <div key={s.l} style={{ background: "#060b18", borderRadius: 4, padding: "7px 10px" }}>
                          <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.08em", marginBottom: 3 }}>{s.l}</div>
                          <div style={{ fontSize: 13, color: s.c }}>{s.v}</div>
                        </div>
                      ))}
                    </div>
                    <PctBar pct={ps.evalProfitTargetPct} color="#3b82f6" warn={75} danger={101} />
                    <div style={{ fontSize: 9, color: "#94a3b8", marginTop: 4 }}>{fmt$(ps.evalStats.cumPnl)} of {fmt$(cfg.eval.profitTarget)} target</div>
          </div>
          )}
                {ps?.fundedStats && (
                  <div style={{ background: "#070d1a", border: "1px solid #166534", borderRadius: 6, padding: "16px 18px" }}>
                    <div style={{ fontSize: 10, color: "#4ade80", letterSpacing: "0.1em", marginBottom: 12 }}>✓ FUNDED · ACTIVE</div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                      {[
                        { l: "NET P&L", v: `${ps.fundedStats.cumPnl >= 0 ? "+" : ""}${fmt$(ps.fundedStats.cumPnl)}`, c: ps.fundedStats.cumPnl >= 0 ? "#4ade80" : "#f87171" },
                        { l: "CURRENT BAL", v: fmt$(ps.fundedStats.currentBalance), c: "#e2e8f0" },
                        { l: "TRAIL DD", v: fmt$(ps.fundedStats.trailingDrawdown), c: ps.fundedStats.trailingDrawdown > cfg.funded.maxLossLimit * 0.75 ? "#f87171" : "#e2e8f0" },
                        { l: "DD LIMIT", v: fmt$(cfg.funded.maxLossLimit), c: "#94a3b8" },
                        { l: "PROFIT DAYS", v: `${ps.qualifyingProfitDays} / ${cfg.funded.minProfitDays}`, c: ps.qualifyingProfitDays >= cfg.funded.minProfitDays ? "#4ade80" : "#94a3b8" },
                        { l: "MIN/DAY", v: fmt$(cfg.funded.minDailyProfit), c: "#94a3b8" },
                      ].map(s => (
                        <div key={s.l} style={{ background: "#060b18", borderRadius: 4, padding: "7px 10px" }}>
                          <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.08em", marginBottom: 3 }}>{s.l}</div>
                          <div style={{ fontSize: 13, color: s.c }}>{s.v}</div>
                        </div>
                      ))}
                    </div>
                    <PctBar pct={(ps.qualifyingProfitDays / cfg.funded.minProfitDays) * 100} color="#4ade80" warn={80} danger={101} />
                    <div style={{ fontSize: 9, color: "#94a3b8", marginTop: 4 }}>{ps.qualifyingProfitDays} of {cfg.funded.minProfitDays} qualifying profit days (≥{fmt$(cfg.funded.minDailyProfit)}/day)</div>
          </div>
          )}
              </div>
              {ps && (
                <div style={{ background: "#070d1a", border: "1px solid #1e293b", borderRadius: 6, padding: "16px 18px" }}>
                  <div style={{ fontSize: 10, color: "#f59e0b", letterSpacing: "0.1em", marginBottom: 4 }}>ACTIVE RULES · {isEval ? "EVALUATION PHASE" : "FUNDED PHASE"}</div>
                  {isEval ? (<>
                    <RuleRow label="Max Loss Limit (Trailing, EOD)" pass={ps.rules.maxLossLimit} detail={`${fmt$(ps.evalStats?.trailingDrawdown || 0)} drawn · limit ${fmt$(cfg.eval.maxLossLimit)}`} />
                    <RuleRow label="Profit Target" pass={ps.rules.profitTarget} detail={`${fmt$(ps.evalStats?.cumPnl || 0)} of ${fmt$(cfg.eval.profitTarget)}`} />
                    <RuleRow label="Consistency Rule (50% cap)" pass={ps.rules.consistency} detail={`Best day is ${(ps.evalStats?.consistencyPct || 0).toFixed(1)}% of total profit`} />
                  </>) : (<>
                    <RuleRow label="Max Loss Limit (Trailing)" pass={ps.rules.maxLossLimit} detail={`${fmt$(ps.fundedStats?.trailingDrawdown || 0)} drawn · limit ${fmt$(cfg.funded.maxLossLimit)}`} />
                    <RuleRow label="Min Qualifying Profit Days" pass={ps.rules.minProfitDays} detail={`${ps.qualifyingProfitDays} of ${cfg.funded.minProfitDays} days ≥ ${fmt$(cfg.funded.minDailyProfit)}`} />
                  </>)}
                </div>
              )}
              {ps && (ps.evalStats || ps.fundedStats) && (
                <div style={{ background: "#070d1a", border: "1px solid #1e293b", borderRadius: 6, padding: "16px 18px" }}>
                  <div style={{ fontSize: 10, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 14 }}>LIFETIME STATS · EVAL + FUNDED COMBINED</div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10 }}>
                    {[
                      { l: "TOTAL NET P&L", v: `${ps.totalCumPnl >= 0 ? "+" : ""}${fmt$(ps.totalCumPnl)}`, c: ps.totalCumPnl >= 0 ? "#4ade80" : "#f87171", hi: true },
                      { l: "TOTAL DAYS", v: ps.totalDaysTraded, c: "#e2e8f0" },
                      { l: "ACCOUNT SIZE", v: fmt$(cfg.accountSize), c: "#94a3b8" },
                    ].map(s => (
                      <div key={s.l} style={{ background: s.hi ? "#0f1a2e" : "#060b18", border: `1px solid ${s.hi ? "#1e3a5f" : "#0f1729"}`, borderRadius: 4, padding: "10px 12px" }}>
                        <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.08em", marginBottom: 4 }}>{s.l}</div>
                        <div style={{ fontSize: 16, color: s.c, fontWeight: s.hi ? 600 : 400 }}>{s.v}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {!ps && <div style={{ color: "#64748b", fontSize: 12, padding: "24px 0", textAlign: "center" }}>No entries yet — start logging sessions to see your prop firm status.</div>}

              {/* Payout history */}
              {payouts.length > 0 && (
                <div style={{ background: "#070d1a", border: "1px solid #1e3a5f", borderRadius: 6, padding: "16px 18px" }}>
                  <div style={{ fontSize: 10, color: "#60a5fa", letterSpacing: "0.1em", marginBottom: 14 }}>💰 PAYOUT HISTORY · {fmt$(totalWithdrawn)} TOTAL WITHDRAWN</div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    {payouts.sort((a, b) => a.date.localeCompare(b.date)).map((p, i) => (
                      <div key={p.id || i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 10px", background: "#060b18", borderRadius: 4, border: "1px solid #0f1729" }}>
                        <div>
                          <span style={{ fontSize: 11, color: "#e2e8f0", fontFamily: "'DM Mono',monospace" }}>{fmt$(p.amount)}</span>
                          {p.fee > 0 && <span style={{ fontSize: 10, color: "#64748b", marginLeft: 8 }}>−{fmt$(p.fee)} fee</span>}
                          {p.note && <span style={{ fontSize: 9, color: "#64748b", marginLeft: 10, fontStyle: "italic" }}>{p.note}</span>}
                        </div>
                        <span style={{ fontSize: 10, color: "#475569", fontFamily: "'DM Mono',monospace" }}>{p.date}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Post-mortem (archived accounts) */}
              {isArchived && cfg.postMortem && (
                <div style={{ background: "#0a0510", border: "1px solid #7f1d1d", borderRadius: 6, padding: "16px 18px" }}>
                  <div style={{ fontSize: 10, color: "#f87171", letterSpacing: "0.1em", marginBottom: 10 }}>📋 POST-MORTEM</div>
                  <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.8, whiteSpace: "pre-wrap" }}>{cfg.postMortem}</div>
                </div>
              )}
            </div>
          );
        };

        const AnalyticsSection = () => {
          if (propJournals.length === 0) return null;
          const accountStats = propJournals.map(j => {
            const jEntries = allPropEntriesMap[j.id] || (j.id === activeJournalId ? entries : []);
            const jPs = j.config ? calcPropStatus(jEntries, j.config) : null;
            const allTrades = jEntries.flatMap(e => e.parsedTrades || []);
            const tNP = (t) => t.pnl - (t.commission||0);
            const winners = allTrades.filter(t => tNP(t) > 0);
            const losers = allTrades.filter(t => tNP(t) < 0);
            const winRate = allTrades.length ? (winners.length / allTrades.length * 100) : 0;
            const netPnl = jPs ? (jPs.totalCumPnl ?? 0) : jEntries.reduce((s, e) => s + (parseFloat(e.pnl) || 0) - (parseFloat(e.commissions) || 0), 0);
            const fees = jEntries.reduce((s, e) => s + (parseFloat(e.commissions) || 0), 0);
            const avgWin = winners.length ? winners.reduce((s, t) => s + tNP(t), 0) / winners.length : 0;
            const avgLoss = losers.length ? Math.abs(losers.reduce((s, t) => s + tNP(t), 0) / losers.length) : 0;
            // Lifetime mistake cost for this account
            const mistakeCostByTag = {};
            for (const e of jEntries) {
              if (!e.mistakeCosts) continue;
              for (const [tag, cost] of Object.entries(e.mistakeCosts)) {
                const n = parseFloat(cost);
                if (Number.isFinite(n) && n > 0) mistakeCostByTag[tag] = (mistakeCostByTag[tag] || 0) + n;
              }
            }
            const accountMistakeCostTotal = Object.values(mistakeCostByTag).reduce((s, v) => s + v, 0);
            return { journal: j, jEntries, allTrades, winners, losers, winRate, netPnl, fees, avgWin, avgLoss, jPs, mistakeCostByTag, accountMistakeCostTotal };
          });
          const totalNetPnl = accountStats.reduce((s, a) => s + a.netPnl, 0);
          const totalTrades = accountStats.reduce((s, a) => s + a.allTrades.length, 0);
          const totalWinners = accountStats.reduce((s, a) => s + a.winners.length, 0);
          const totalDays = accountStats.reduce((s, a) => s + a.jEntries.length, 0);
          const totalFees = accountStats.reduce((s, a) => s + a.fees, 0);
          const overallWinRate = totalTrades > 0 ? (totalWinners / totalTrades * 100) : 0;
          const avgWinRateAcrossAccounts = accountStats.filter(a => a.allTrades.length > 0).reduce((s, a) => s + a.winRate, 0) / Math.max(1, accountStats.filter(a => a.allTrades.length > 0).length);
          // Business ledger totals
          const totalWithdrawnAll = accountStats.reduce((s, a) => s + (a.jPs?.totalWithdrawn || 0), 0);
          const totalEvalCosts = accountStats.reduce((s, a) => s + (parseFloat(a.journal.config?.evalCost) || 0), 0);
          const archivedCount = accountStats.filter(a => a.journal.config?.isArchived).length;
          // Lifetime mistake cost aggregated across ALL prop accounts
          const lifetimeMistakeCostByTag = {};
          for (const { mistakeCostByTag } of accountStats) {
            for (const [tag, cost] of Object.entries(mistakeCostByTag)) {
              lifetimeMistakeCostByTag[tag] = (lifetimeMistakeCostByTag[tag] || 0) + cost;
            }
          }
          const lifetimeMistakeCostTotal = Object.values(lifetimeMistakeCostByTag).reduce((s, v) => s + v, 0);
          const topMistakeTags = Object.entries(lifetimeMistakeCostByTag).sort((a, b) => b[1] - a[1]).slice(0, 5);

          if (propJournals.length < 2 && totalTrades === 0) return null; // hide if only 1 empty account

          return (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {/* Divider */}
              <div style={{ display: "flex", alignItems: "center", gap: 10, margin: "4px 0" }}>
                <div style={{ height: 1, flex: 1, background: "linear-gradient(90deg, #1e3a5f, transparent)" }} />
                <span style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.14em" }}>📊 CUMULATIVE ANALYTICS</span>
                <div style={{ height: 1, flex: 1, background: "linear-gradient(270deg, #1e3a5f, transparent)" }} />
              </div>

              {/* Lifetime Withdrawn hero banner */}
              {totalWithdrawnAll > 0 && (
                <div style={{ background: "linear-gradient(135deg, #0a1628, #0d1e38)", border: "1px solid #1e3a5f", borderRadius: 8, padding: "16px 20px", position: "relative", overflow: "hidden" }}>
                  <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg, transparent, #60a5fa, #3b82f6, transparent)" }} />
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12 }}>
                    <div>
                      <div style={{ fontSize: 9, color: "#60a5fa", letterSpacing: "0.18em", marginBottom: 6, textTransform: "uppercase" }}>💰 LIFETIME WITHDRAWN · REALIZED INCOME</div>
                      <div style={{ fontSize: 36, fontWeight: 700, color: "#93c5fd", letterSpacing: "-0.02em", lineHeight: 1, fontFamily: "'DM Mono',monospace" }}>{fmt$(totalWithdrawnAll)}</div>
                      <div style={{ fontSize: 10, color: "#475569", marginTop: 4 }}>Paid out across {accountStats.filter(a => (a.jPs?.totalWithdrawn || 0) > 0).length} account{accountStats.filter(a => (a.jPs?.totalWithdrawn || 0) > 0).length !== 1 ? "s" : ""}</div>
                    </div>
                    {totalEvalCosts > 0 && (
                      <div style={{ textAlign: "right" }}>
                        <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.12em", marginBottom: 4 }}>EVAL COSTS (EXPENSES)</div>
                        <div style={{ fontSize: 18, color: "#f87171", fontFamily: "'DM Mono',monospace" }}>-{fmt$(totalEvalCosts)}</div>
                        <div style={{ fontSize: 9, color: "#475569", marginTop: 3 }}>Net: <span style={{ color: totalWithdrawnAll - totalEvalCosts >= 0 ? "#4ade80" : "#f87171" }}>{fmt$(totalWithdrawnAll - totalEvalCosts)}</span></div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Headline numbers */}
              <div style={{ background: "#070d1a", border: "1px solid #1e3a5f", borderRadius: 8, padding: "16px 20px", position: "relative", overflow: "hidden" }}>
                <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg, transparent, #3b82f6, transparent)" }} />
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
                  {[
                    { l: "TOTAL NET P&L", v: `${totalNetPnl >= 0 ? "+" : ""}${fmt$(totalNetPnl)}`, c: totalNetPnl >= 0 ? "#4ade80" : "#f87171", big: true },
                    { l: "OVERALL WIN RATE", v: `${overallWinRate.toFixed(1)}%`, c: overallWinRate >= 50 ? "#4ade80" : "#f87171" },
                    { l: "TOTAL TRADES", v: totalTrades, c: "#e2e8f0" },
                    { l: "TRADING DAYS", v: totalDays, c: "#e2e8f0" },
                  ].map(s => (
                    <div key={s.l} style={{ background: "#060b18", borderRadius: 5, padding: "10px 12px", border: "1px solid #0f1729" }}>
                      <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 4 }}>{s.l}</div>
                      <div style={{ fontSize: s.big ? 18 : 14, color: s.c, fontWeight: s.big ? 700 : 500, fontFamily: "'DM Mono',monospace" }}>{s.v}</div>
                    </div>
                  ))}
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8, marginTop: 8 }}>
                  {[
                    { l: "AVG WIN RATE", v: `${avgWinRateAcrossAccounts.toFixed(1)}%`, c: avgWinRateAcrossAccounts >= 50 ? "#4ade80" : "#fbbf24" },
                    { l: "WINNERS", v: `${totalWinners} / ${totalTrades}`, c: "#4ade80" },
                    { l: "TOTAL FEES PAID", v: totalFees > 0 ? `-${fmt$(totalFees)}` : "—", c: "#f87171" },
                  ].map(s => (
                    <div key={s.l} style={{ background: "#060b18", borderRadius: 5, padding: "8px 10px", border: "1px solid #0f1729" }}>
                      <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 3 }}>{s.l}</div>
                      <div style={{ fontSize: 12, color: s.c, fontFamily: "'DM Mono',monospace" }}>{s.v}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Lifetime Mistake Cost Card */}
              {lifetimeMistakeCostTotal > 0 && (
                <div style={{ background: "#0d0505", border: "1px solid #7f1d1d", borderRadius: 8, padding: "16px 20px", position: "relative", overflow: "hidden" }}>
                  <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg, transparent, #ef4444, transparent)" }} />
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
                    <div>
                      <div style={{ fontSize: 9, color: "#f87171", letterSpacing: "0.14em", marginBottom: 4 }}>💸 LIFETIME MISTAKE COST</div>
                      <div style={{ fontSize: 9, color: "#64748b" }}>money left on the table from behavioral errors</div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontSize: 26, fontWeight: 700, color: "#f87171", fontFamily: "'DM Mono',monospace", lineHeight: 1 }}>-{fmt$(lifetimeMistakeCostTotal)}</div>
                      {totalNetPnl !== 0 && (
                        <div style={{ fontSize: 9, color: "#64748b", marginTop: 4 }}>
                          {(lifetimeMistakeCostTotal / Math.abs(totalNetPnl) * 100).toFixed(1)}% of gross P&L
                        </div>
                      )}
                    </div>
                  </div>
                  {topMistakeTags.length > 0 && (
                    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                      {topMistakeTags.map(([tag, cost], i) => {
                        const pct = lifetimeMistakeCostTotal > 0 ? cost / lifetimeMistakeCostTotal : 0;
                        const barColor = i === 0 ? "#ef4444" : i === 1 ? "#f87171" : "#fca5a5";
                        return (
                          <div key={tag}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
                              <span style={{ fontSize: 10, color: "#e2e8f0", letterSpacing: "0.03em" }}>{tag}</span>
                              <span style={{ fontSize: 10, fontFamily: "'DM Mono',monospace", color: barColor }}>-{fmt$(cost)}</span>
                            </div>
                            <div style={{ height: 3, background: "#1a0505", borderRadius: 2, overflow: "hidden" }}>
                              <div style={{ width: `${pct * 100}%`, height: "100%", background: barColor, borderRadius: 2, opacity: 0.75, transition: "width .4s" }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}

              {/* Per-account bars */}
              {accountStats.length > 1 && (
                <div style={{ background: "#070d1a", border: "1px solid #1e293b", borderRadius: 6, padding: "14px 16px" }}>
                  <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 10 }}>P&L BY ACCOUNT</div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {accountStats.map(({ journal: j, allTrades, winRate, netPnl, avgWin, avgLoss, jEntries }) => {
                      const maxAbsPnl = Math.max(...accountStats.map(a => Math.abs(a.netPnl)), 1);
                      const barW = Math.abs(netPnl) / maxAbsPnl * 100;
                      const isEvalJ = j.config?.phase === "eval";
                      return (
                        <div key={j.id}
                          onClick={() => { if (j.id !== activeJournalId) switchJournal(j.id); setPropDashTab("account"); }}
                          style={{ cursor: "pointer", padding: "10px 12px", background: "#060b18", borderRadius: 5, border: `1px solid ${j.id === activeJournalId ? "#1e3a5f" : "#0f1729"}`, transition: "all .15s" }}
                          onMouseEnter={e => e.currentTarget.style.borderColor = "#f59e0b"}
                          onMouseLeave={e => { e.currentTarget.style.borderColor = j.id === activeJournalId ? "#1e3a5f" : "#0f1729"; }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                              <span style={{ fontSize: 10, color: "#e2e8f0", fontWeight: 500 }}>{j.config?.firmName || j.name}</span>
                              <span style={{ fontSize: 8, padding: "1px 7px", borderRadius: 10, background: isEvalJ ? "rgba(59,130,246,0.1)" : "rgba(74,222,128,0.1)", color: isEvalJ ? "#93c5fd" : "#4ade80", border: `1px solid ${isEvalJ ? "rgba(59,130,246,0.2)" : "rgba(74,222,128,0.2)"}` }}>{isEvalJ ? "EVAL" : "FUNDED"}</span>
                              {allTrades.length > 0 && <span style={{ fontSize: 8, color: winRate >= 50 ? "#4ade80" : "#f87171" }}>{winRate.toFixed(0)}% WR</span>}
                            </div>
                            <span style={{ fontSize: 13, fontWeight: 700, color: netPnl >= 0 ? "#4ade80" : "#f87171", fontFamily: "'DM Mono',monospace" }}>{netPnl >= 0 ? "+" : ""}{fmt$(netPnl)}</span>
                          </div>
                          <div style={{ height: 4, background: "#0a0e1a", borderRadius: 2, overflow: "hidden" }}>
                            <div style={{ width: `${barW}%`, height: "100%", background: netPnl >= 0 ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.65, transition: "width .4s" }} />
                          </div>
                          {allTrades.length > 0 && avgWin > 0 && avgLoss > 0 && (
                            <div style={{ display: "flex", gap: 12, marginTop: 5, fontSize: 8, color: "#64748b" }}>
                              <span>Avg W: <span style={{ color: "#4ade80" }}>{fmt$(avgWin)}</span></span>
                              <span>Avg L: <span style={{ color: "#f87171" }}>-{fmt$(avgLoss)}</span></span>
                              <span>R:R <span style={{ color: "#93c5fd" }}>{(avgWin / avgLoss).toFixed(2)}x</span></span>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          );
        };

        return (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

            {/* ── TAB BAR (only when not in account detail) ── */}
            {propDashTab !== "account" && (
              <div style={{ display: "flex", gap: 6, borderBottom: "1px solid #0f1729", paddingBottom: 12 }}>
                {[
                  { id: "overview",      label: "🏆 ACCOUNTS" },
                  { id: "cumulative",    label: "📊 CUMULATIVE" },
                  { id: "certificates",  label: "🎖 ACHIEVEMENTS" },
                ].map(({ id, label }) => (
                  <button key={id} onClick={() => setPropDashTab(id)}
                    style={{ padding: "7px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.08em", transition: "all .15s", background: propDashTab === id ? "#0f1a2e" : "transparent", border: `1px solid ${propDashTab === id ? "#1e3a5f" : "#0f1729"}`, color: propDashTab === id ? "#93c5fd" : "#64748b", textTransform: "uppercase" }}>
                    {label}
                  </button>
                ))}
              </div>
            )}

            {propDashTab === "overview" && <OverviewTab />}
            {propDashTab === "cumulative" && <AnalyticsSection />}
            {propDashTab === "certificates" && <CertificatesTab certs={certs} setCerts={setCerts} certLightbox={certLightbox} setCertLightbox={setCertLightbox} />}
            {propDashTab === "account" && (
              <>
                <button onClick={() => setPropDashTab("overview")}
                  style={{ alignSelf: "flex-start", background: "transparent", border: "1px solid #1e293b", color: "#94a3b8", padding: "6px 14px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.06em" }}>
                  ← ALL ACCOUNTS
                </button>
                <AccountTab />
              </>
            )}
          </div>
        );
}

// ─────────────────────────────────────────────────────────────────────────────
// REFERENCE VIEW — Futures Trading Sessions
// ─────────────────────────────────────────────────────────────────────────────

// ── Risk calculator extracted from ReferenceView IIFE to fix React hook rules ──
function RiskCalcPanel() {
  const [mode, setMode] = useState("futures");
  const [innerTab, setInnerTab] = useState("calc"); // "calc" | "specs"
        const [acctSize, setAcctSize] = useState("50000");
        const [riskPct, setRiskPct] = useState("1");
        const [stopTicks, setStopTicks] = useState("4");
        const [stopDollar, setStopDollar] = useState("50");
        const [ticker, setTicker] = useState("ES");
        const [stockPrice, setStockPrice] = useState("500");
        const [stockStop, setStockStop] = useState("5");

        const FUTURES_SPECS = [
          { sym: "ES",  name: "E-mini S&P 500",   tick: 0.25, tickVal: 12.50 },
          { sym: "MES", name: "Micro E-mini S&P",  tick: 0.25, tickVal: 1.25  },
          { sym: "NQ",  name: "E-mini Nasdaq",      tick: 0.25, tickVal: 5.00  },
          { sym: "MNQ", name: "Micro E-mini Nasdaq",tick: 0.25, tickVal: 0.50  },
          { sym: "YM",  name: "E-mini Dow",         tick: 1.00, tickVal: 5.00  },
          { sym: "MYM", name: "Micro E-mini Dow",   tick: 1.00, tickVal: 0.50  },
          { sym: "RTY", name: "E-mini Russell 2000",tick: 0.10, tickVal: 5.00  },
          { sym: "M2K", name: "Micro E-mini Russell",tick: 0.10, tickVal: 0.50 },
          { sym: "CL",  name: "Crude Oil",          tick: 0.01, tickVal: 10.00 },
          { sym: "GC",  name: "Gold",               tick: 0.10, tickVal: 10.00 },
        ];

        const acct = parseFloat(acctSize) || 0;
        const rPct = parseFloat(riskPct) || 0;
        const riskDollars = acct * rPct / 100;

        // Futures calc
        const spec = FUTURES_SPECS.find(s => s.sym === ticker) || FUTURES_SPECS[0];
        const ticks = parseFloat(stopTicks) || 0;
        const stopVal = ticks * spec.tickVal;
        const contracts = stopVal > 0 ? Math.floor(riskDollars / stopVal) : 0;
        const actualRisk = contracts * stopVal;
        const actualRiskPct = acct > 0 ? actualRisk / acct * 100 : 0;
        const reward2x = actualRisk * 2;
        const reward3x = actualRisk * 3;

        // Stocks calc
        const sPrice = parseFloat(stockPrice) || 0;
        const sStop = parseFloat(stockStop) || 0;
        const sStopPct = sPrice > 0 ? sStop / sPrice * 100 : 0;
        const shares = sStop > 0 ? Math.floor(riskDollars / sStop) : 0;
        const sActualRisk = shares * sStop;
        const sPositionValue = shares * sPrice;
        const sPositionPct = acct > 0 ? sPositionValue / acct * 100 : 0;
        const sReward2x = sActualRisk * 2;
        const sReward3x = sActualRisk * 3;

        const Stat = ({ label, value, color = "#e2e8f0", sub, big }) => (
          <div style={{ background: "#060b18", border: "1px solid #0f1e30", borderRadius: 6, padding: big ? "14px 16px" : "10px 14px" }}>
            <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.12em", marginBottom: big ? 6 : 4, textTransform: "uppercase" }}>{label}</div>
            <div style={{ fontSize: big ? 22 : 15, color, fontWeight: big ? 700 : 500, fontFamily: "'DM Mono',monospace", letterSpacing: "0.02em" }}>{value}</div>
            {sub && <div style={{ fontSize: 9, color: "#64748b", marginTop: 4 }}>{sub}</div>}
          </div>
        );

        const InputRow = ({ label, value, onChange, placeholder, prefix, suffix, type = "number", min }) => (
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase" }}>{label}</label>
            <div style={{ display: "flex", alignItems: "center", background: "#060b18", border: "1px solid #1e293b", borderRadius: 4, overflow: "hidden" }}>
              {prefix && <span style={{ padding: "0 10px", fontSize: 11, color: "#64748b", borderRight: "1px solid #1e293b", lineHeight: "34px" }}>{prefix}</span>}
              <input type={type} value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} min={min}
                style={{ flex: 1, background: "transparent", border: "none", color: "#e2e8f0", padding: "7px 10px", fontSize: 13, fontFamily: "inherit", outline: "none", width: 0 }} />
              {suffix && <span style={{ padding: "0 10px", fontSize: 11, color: "#64748b", borderLeft: "1px solid #1e293b", lineHeight: "34px" }}>{suffix}</span>}
            </div>
          </div>
        );

        return (
          <div style={{ animation: "refFadeIn .3s ease", display: "flex", flexDirection: "column", gap: 16 }}>

            {/* Inner tab strip */}
            <div style={{ display: "flex", gap: 6 }}>
              {[["calc", "⚡ RISK CALCULATOR"], ["specs", "📋 CONTRACT SPECS"]].map(([id, label]) => (
                <button key={id} onClick={() => setInnerTab(id)}
                  style={{ padding: "8px 20px", borderRadius: 4, fontFamily: "inherit", fontSize: 12, cursor: "pointer", letterSpacing: "0.06em", transition: "all .15s",
                    background: innerTab === id ? "#0a1628" : "transparent",
                    border: `1px solid ${innerTab === id ? "#3b82f6" : "#1e293b"}`,
                    color: innerTab === id ? "#93c5fd" : "#64748b" }}>
                  {label}
                </button>
              ))}
            </div>

            {innerTab === "calc" && <>
            {/* Account + Risk% — shared */}
            <div style={{ background: "#070d1a", border: "1px solid #1e3a5f", borderRadius: 8, padding: "18px 20px" }}>
              <div style={{ fontSize: 10, color: "#93c5fd", letterSpacing: "0.12em", marginBottom: 14 }}>⚡ POSITION SIZE CALCULATOR</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 14 }}>
                <InputRow label="Account Size" value={acctSize} onChange={setAcctSize} prefix="$" placeholder="50000" />
                <InputRow label="Risk Per Trade" value={riskPct} onChange={setRiskPct} suffix="%" placeholder="1" />
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <span style={{ fontSize: 10, color: "#64748b" }}>Risk $:</span>
                <span style={{ fontSize: 15, color: riskDollars > 0 ? "#f59e0b" : "#475569", fontWeight: 700, fontFamily: "'DM Mono',monospace" }}>
                  {riskDollars > 0 ? `$${riskDollars.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "—"}
                </span>
                <div style={{ display: "flex", gap: 4, marginLeft: "auto" }}>
                  {[0.5, 1, 1.5, 2].map(p => (
                    <button key={p} onClick={() => setRiskPct(String(p))}
                      style={{ padding: "3px 10px", borderRadius: 3, fontFamily: "inherit", fontSize: 10, cursor: "pointer", background: riskPct === String(p) ? "#1e3a5f" : "transparent", border: `1px solid ${riskPct === String(p) ? "#3b82f6" : "#1e293b"}`, color: riskPct === String(p) ? "#93c5fd" : "#475569" }}>
                      {p}%
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Mode toggle */}
            <div style={{ display: "flex", gap: 6 }}>
              {[["futures", "📈 FUTURES"], ["stocks", "🏦 STOCKS"]].map(([m, label]) => (
                <button key={m} onClick={() => setMode(m)}
                  style={{ flex: 1, padding: "10px 0", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.08em", transition: "all .15s", background: mode === m ? (m === "futures" ? "#0f1a2e" : "#0a1a0f") : "transparent", border: `1px solid ${mode === m ? (m === "futures" ? "#1e3a5f" : "#166534") : "#1e293b"}`, color: mode === m ? (m === "futures" ? "#93c5fd" : "#4ade80") : "#475569" }}>
                  {label}
                </button>
              ))}
            </div>

            {mode === "futures" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {/* Instrument selector */}
                <div>
                  <label style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>INSTRUMENT</label>
                  <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {FUTURES_SPECS.map(s => (
                      <button key={s.sym} onClick={() => setTicker(s.sym)}
                        style={{ padding: "6px 14px", borderRadius: 3, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.06em", background: ticker === s.sym ? "#1e3a5f" : "transparent", border: `1px solid ${ticker === s.sym ? "#3b82f6" : "#1e293b"}`, color: ticker === s.sym ? "#93c5fd" : "#64748b", transition: "all .12s" }}>
                        {s.sym}
                      </button>
                    ))}
                  </div>
                  {spec && <div style={{ fontSize: 9, color: "#64748b", marginTop: 6 }}>{spec.name} · Tick: {spec.tick} pts = ${spec.tickVal.toFixed(2)}/contract</div>}
                </div>

                {/* Stop input */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <InputRow label="Stop (ticks)" value={stopTicks} onChange={setStopTicks} placeholder="4" suffix="ticks" />
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    <label style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase" }}>Stop (points)</label>
                    <div style={{ background: "#060b18", border: "1px solid #0f1729", borderRadius: 4, padding: "7px 12px", fontSize: 13, color: "#64748b", fontFamily: "'DM Mono',monospace" }}>
                      {ticks > 0 ? `${(ticks * spec.tick).toFixed(2)} pts` : "—"}
                    </div>
                  </div>
                </div>

                {/* Results */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
                  <Stat big label="Contracts" value={contracts > 0 ? contracts : "—"} color="#f59e0b" sub={contracts > 0 ? `${spec.sym} @ ${ticks} ticks stop` : "Increase account or reduce stop"} />
                  <Stat big label="Actual Risk $" value={actualRisk > 0 ? `$${actualRisk.toLocaleString()}` : "—"} color={actualRiskPct > 2 ? "#f87171" : "#e2e8f0"} sub={actualRisk > 0 ? `${actualRiskPct.toFixed(2)}% of account` : ""} />
                  <Stat big label="Stop Value" value={stopVal > 0 ? `$${stopVal.toFixed(2)}` : "—"} color="#94a3b8" sub="per contract" />
                </div>

                {contracts > 0 && (
                  <div style={{ background: "#070d1a", border: "1px solid #0f1729", borderRadius: 6, padding: "14px 16px" }}>
                    <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.12em", marginBottom: 12 }}>REWARD TARGETS · {contracts} CONTRACT{contracts !== 1 ? "S" : ""}</div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
                      <Stat label="1R Target" value={`$${actualRisk.toLocaleString()}`} color="#fbbf24" sub={`${(ticks).toFixed(0)} ticks`} />
                      <Stat label="2R Target" value={`$${reward2x.toLocaleString()}`} color="#4ade80" sub={`${(ticks*2).toFixed(0)} ticks`} />
                      <Stat label="3R Target" value={`$${reward3x.toLocaleString()}`} color="#22d3ee" sub={`${(ticks*3).toFixed(0)} ticks`} />
                    </div>
          </div>
          )}

                {/* Quick presets */}
                <div style={{ background: "#070d1a", border: "1px solid #0f1729", borderRadius: 6, padding: "12px 14px" }}>
                  <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 2 }}>COMMON STOP SIZES · {spec.sym}</div>
                  <div style={{ fontSize: 9, color: "#475569", marginBottom: 10, lineHeight: 1.5 }}>Click any preset to load it. Each button shows: <strong style={{color:"#64748b"}}>ticks · $ per contract · max contracts</strong> at your risk settings.</div>
                  <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {[2, 4, 6, 8, 10, 12, 16, 20].map(t => {
                      const sv = t * spec.tickVal;
                      const c = riskDollars > 0 && sv > 0 ? Math.floor(riskDollars / sv) : 0;
                      return (
                        <button key={t} onClick={() => setStopTicks(String(t))}
                          style={{ padding: "6px 12px", borderRadius: 3, fontFamily: "inherit", fontSize: 10, cursor: "pointer", background: stopTicks === String(t) ? "#1e3a5f" : "#060b18", border: `1px solid ${stopTicks === String(t) ? "#3b82f6" : "#0f1729"}`, color: stopTicks === String(t) ? "#93c5fd" : "#64748b", textAlign: "center", minWidth: 60 }}>
                          <div style={{fontWeight:600}}>{t}T</div>
                          <div style={{ fontSize: 8, color: stopTicks === String(t) ? "#60a5fa" : "#475569", marginTop: 1 }}>${sv.toFixed(0)}/ct</div>
                          <div style={{ fontSize: 8, color: stopTicks === String(t) ? "#4ade80" : "#334155", marginTop: 1 }}>{c > 0 ? c+"ct max" : "—"}</div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {mode === "stocks" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <InputRow label="Stock Price" value={stockPrice} onChange={setStockPrice} prefix="$" placeholder="500.00" />
                  <InputRow label="Stop Loss ($)" value={stockStop} onChange={setStockStop} prefix="-$" placeholder="5.00" suffix={sPrice > 0 && sStop > 0 ? `${sStopPct.toFixed(2)}%` : ""} />
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
                  <Stat big label="Shares" value={shares > 0 ? shares.toLocaleString() : "—"} color="#4ade80" sub={shares > 0 ? `@ $${sPrice} per share` : "Check inputs"} />
                  <Stat big label="Position Size" value={sPositionValue > 0 ? `$${sPositionValue.toLocaleString()}` : "—"} color="#93c5fd" sub={sPositionValue > 0 ? `${sPositionPct.toFixed(1)}% of account` : ""} />
                  <Stat big label="Actual Risk $" value={sActualRisk > 0 ? `$${sActualRisk.toLocaleString()}` : "—"} color={sActualRisk > riskDollars * 1.05 ? "#f87171" : "#e2e8f0"} sub={sActualRisk > 0 ? `${(sActualRisk/acct*100).toFixed(2)}% of account` : ""} />
                </div>

                {shares > 0 && (
                  <div style={{ background: "#070d1a", border: "1px solid #0f1729", borderRadius: 6, padding: "14px 16px" }}>
                    <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.12em", marginBottom: 12 }}>REWARD TARGETS · {shares.toLocaleString()} SHARES</div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
                      {[
                        { r: "1R", mult: 1, c: "#fbbf24" },
                        { r: "2R", mult: 2, c: "#4ade80" },
                        { r: "3R", mult: 3, c: "#22d3ee" },
                      ].map(({ r, mult, c }) => {
                        const target = sPrice + (sStop * mult);
                        const profit = shares * sStop * mult;
                        return <Stat key={r} label={`${r} Target`} value={`$${profit.toLocaleString()}`} color={c} sub={`$${target.toFixed(2)} per share`} />;
                      })}
                    </div>
          </div>
          )}

                {/* Position concentration warning */}
                {sPositionPct > 25 && (
                  <div style={{ background: "rgba(248,113,113,0.06)", border: "1px solid rgba(248,113,113,0.2)", borderRadius: 6, padding: "10px 14px", fontSize: 11, color: "#f87171" }}>
                    ⚠ Position is {sPositionPct.toFixed(0)}% of account — consider reducing size or using options for defined risk.
          </div>
          )}
              </div>
            )}
            </> /* end innerTab === calc */}

            {innerTab === "specs" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16, paddingBottom: 24 }}>

          <div>
            <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.18em", marginBottom: 4 }}>📋 CONTRACT SPECIFICATIONS — TICK & POINT VALUES</div>
            <div style={{ fontSize: 10, color: "#64748b", lineHeight: 1.6 }}>
              Cross-check your AI parser: <strong style={{ color: "#94a3b8" }}>Points Gained × Multiplier = $ P&L</strong>.
              If the numbers don't match, there's a parsing error.
            </div>
          </div>
          {/* Spec table */}
          <div style={{ background: "rgba(7,13,26,0.8)", border: "1px solid #1e293b", borderRadius: 8, overflow: "hidden" }}>
            <div style={{ display: "grid", gridTemplateColumns: "72px 1fr 80px 90px 90px 90px", padding: "8px 14px", background: "#060810", borderBottom: "1px solid #1e293b" }}>
              {["SYMBOL","NAME","TICK","TICK $","PT $","MULTIPLIER"].map(h => (
                <div key={h} style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em" }}>{h}</div>
              ))}
            </div>
            {[
              { sym:"ES",  name:"E-mini S&P 500",     tick:0.25,  tickVal:12.50, ptVal:50,   mult:50    },
              { sym:"MES", name:"Micro E-mini S&P",   tick:0.25,  tickVal:1.25,  ptVal:5,    mult:5     },
              { sym:"NQ",  name:"E-mini Nasdaq",       tick:0.25,  tickVal:5.00,  ptVal:20,   mult:20    },
              { sym:"MNQ", name:"Micro E-mini Nasdaq", tick:0.25,  tickVal:0.50,  ptVal:2,    mult:2     },
              { sym:"YM",  name:"E-mini Dow",          tick:1.00,  tickVal:5.00,  ptVal:5,    mult:5     },
              { sym:"MYM", name:"Micro E-mini Dow",    tick:1.00,  tickVal:0.50,  ptVal:0.5,  mult:0.5   },
              { sym:"RTY", name:"E-mini Russell 2000", tick:0.10,  tickVal:5.00,  ptVal:50,   mult:50    },
              { sym:"M2K", name:"Micro E-mini Russell",tick:0.10,  tickVal:0.50,  ptVal:5,    mult:5     },
              { sym:"CL",  name:"Crude Oil",           tick:0.01,  tickVal:10.00, ptVal:1000, mult:1000  },
              { sym:"GC",  name:"Gold",                tick:0.10,  tickVal:10.00, ptVal:100,  mult:100   },
            ].map((s, i, arr) => (
              <div key={s.sym} style={{ display:"grid", gridTemplateColumns:"72px 1fr 80px 90px 90px 90px", padding:"9px 14px", borderBottom: i < arr.length-1 ? "1px solid #0a1220" : "none", background: i%2===0 ? "transparent" : "rgba(255,255,255,0.01)" }}>
                <div style={{ fontSize:12, color:"#f59e0b", fontFamily:"'DM Mono',monospace", fontWeight:600 }}>{s.sym}</div>
                <div style={{ fontSize:10, color:"#94a3b8" }}>{s.name}</div>
                <div style={{ fontSize:11, color:"#e2e8f0", fontFamily:"'DM Mono',monospace" }}>{s.tick}</div>
                <div style={{ fontSize:11, color:"#4ade80", fontFamily:"'DM Mono',monospace" }}>${s.tickVal.toFixed(2)}</div>
                <div style={{ fontSize:11, color:"#93c5fd", fontFamily:"'DM Mono',monospace" }}>${s.ptVal}</div>
                <div style={{ fontSize:11, color:"#e2e8f0", fontFamily:"'DM Mono',monospace" }}>{s.mult}</div>
              </div>
            ))}
          </div>
          {/* Verification examples */}
          <div style={{ background: "rgba(7,13,26,0.8)", border: "1px solid #1e293b", borderRadius: 8, padding: "16px 18px" }}>
            <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.14em", marginBottom: 10, textTransform: "uppercase" }}>🔍 P&L VERIFICATION EXAMPLES</div>
            {[
              { sym:"ES",  ex:"Long 2 @ 5280.00 → 5282.50  =  (2.50 pts × 2 cts × $50)  = +$250.00" },
              { sym:"MES", ex:"Long 5 @ 5280.00 → 5281.00  =  (1.00 pt  × 5 cts × $5)   = +$25.00" },
              { sym:"NQ",  ex:"Short 1 @ 20000 → 19990     =  (10 pts   × 1 ct  × $20)   = +$200.00" },
              { sym:"MNQ", ex:"Long 10 @ 20000 → 20008     =  (8 pts    × 10 cts × $2)   = +$160.00" },
            ].map(r => (
              <div key={r.sym} style={{ background:"#060810", borderRadius:4, padding:"8px 12px", marginBottom:6 }}>
                <span style={{ fontSize:9, color:"#f59e0b", fontFamily:"'DM Mono',monospace", marginRight:10, fontWeight:600 }}>{r.sym}</span>
                <span style={{ fontSize:9, color:"#64748b", fontFamily:"'DM Mono',monospace" }}>{r.ex}</span>
              </div>
            ))}
          </div>

              </div>
            )}
          </div>
        );
}

function ReferenceView() {
  const [activeSection, setActiveSection] = useState("sessions");
  const [sessInnerTab, setSessInnerTab] = useState("map");

  // ── Resource Links ──
  const LINK_CATS = ["News", "Education", "Tools", "Brokers", "Charts", "Other"];

  // ── Trading Album ──
  const [album, setAlbum] = useState(() => {
    try { const r = localStorage.getItem("tj-album-v1"); return r ? JSON.parse(r) : []; } catch { return []; }
  });
  const ALBUM_CATS = [
    { id: "chart",     label: "Chart",      emoji: "📊", color: "#3b82f6" },
    { id: "lesson",    label: "Lesson",     emoji: "📚", color: "#8b5cf6" },
    { id: "motivation",label: "Motivation", emoji: "🔥", color: "#f59e0b" },
    { id: "news",      label: "News",       emoji: "📰", color: "#ef4444" },
    { id: "setup",     label: "Setup",      emoji: "📈", color: "#10b981" },
    { id: "mistake",   label: "Mistake",    emoji: "⚠️",  color: "#f97316" },
    { id: "strategy",  label: "Strategy",   emoji: "🧠", color: "#ec4899" },
    { id: "other",     label: "Other",      emoji: "📌", color: "#64748b" },
  ];
  const [albumFilter, setAlbumFilter] = useState("all");
  const [albumForm, setAlbumForm] = useState({ title: "", caption: "", type: "chart" });
  const [albumPending, setAlbumPending] = useState(null); // base64 image waiting for title
  const [albumLightbox, setAlbumLightbox] = useState(null); // id of image to show full size
  const [albumEditing, setAlbumEditing] = useState(null); // id being edited
  const [albumConfirmDelete, setAlbumConfirmDelete] = useState(null);
  const [albumPasteActive, setAlbumPasteActive] = useState(false);

  const saveAlbum = (updated) => {
    setAlbum(updated);
    try { localStorage.setItem("tj-album-v1", JSON.stringify(updated)); } catch {}
  };

  const handleAlbumPaste = (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        e.preventDefault();
        const blob = item.getAsFile();
        const reader = new FileReader();
        reader.onload = (ev) => {
          setAlbumPending(ev.target.result);
          setAlbumForm({ title: "", caption: "" });
        };
        reader.readAsDataURL(blob);
        break;
      }
    }
  };

  const confirmAddAlbumImage = () => {
    if (!albumPending) return;
    const entry = { id: Date.now(), src: albumPending, title: albumForm.title.trim() || "Untitled", caption: albumForm.caption.trim(), type: albumForm.type || "chart", addedAt: new Date().toISOString() };
    saveAlbum([entry, ...album]);
    setAlbumPending(null);
    setAlbumForm({ title: "", caption: "", type: "chart" });
  };

  const saveAlbumEdit = (id) => {
    saveAlbum(album.map(img => img.id === id ? { ...img, title: albumForm.title.trim() || img.title, caption: albumForm.caption.trim(), type: albumForm.type || img.type || "setup" } : img));
    setAlbumEditing(null);
  };
  const [links, setLinks] = useState(() => {
    try { const r = localStorage.getItem("tj-links-v1"); return r ? JSON.parse(r) : []; } catch { return []; }
  });
  const [linkForm, setLinkForm] = useState({ name: "", url: "", desc: "", cat: "Tools" });
  const [linkError, setLinkError] = useState("");
  const [linkFilter, setLinkFilter] = useState("All");
  const [confirmDeleteLink, setConfirmDeleteLink] = useState(null);
  const [editingLink, setEditingLink] = useState(null); // id of link being edited

  const saveLinks = (updated) => {
    setLinks(updated);
    try { localStorage.setItem("tj-links-v1", JSON.stringify(updated)); } catch {}
  };

  const handleAddLink = () => {
    const name = linkForm.name.trim();
    let url = linkForm.url.trim();
    if (!name) { setLinkError("Name is required."); return; }
    if (!url)  { setLinkError("URL is required."); return; }
    if (!/^https?:\/\//i.test(url)) url = "https://" + url;
    if (editingLink) {
      saveLinks(links.map(l => l.id === editingLink ? { ...l, name, url, desc: linkForm.desc.trim(), cat: linkForm.cat } : l));
      setEditingLink(null);
    } else {
      saveLinks([...links, { id: Date.now(), name, url, desc: linkForm.desc.trim(), cat: linkForm.cat, addedAt: new Date().toISOString() }]);
    }
    setLinkForm({ name: "", url: "", desc: "", cat: "Tools" });
    setLinkError("");
  };

  const handleEditLink = (link) => {
    setLinkForm({ name: link.name, url: link.url, desc: link.desc || "", cat: link.cat || "Tools" });
    setEditingLink(link.id);
    setLinkError("");
  };

  const handleCancelEdit = () => {
    setEditingLink(null);
    setLinkForm({ name: "", url: "", desc: "", cat: "Tools" });
    setLinkError("");
  };

  const SECTIONS = [
    { id: "sessions", label: "🗺 SESSION MAP" },
    { id: "risk",    label: "⚡ RISK CALC" },

    { id: "links",   label: "🔗 RESOURCE LINKS" },
    { id: "album",   label: "📸 TRADING ALBUM" },
    { id: "quotes",  label: "💬 QUOTES" },
  ];

  // 24h timeline bar — 6PM to 5PM next day (23hrs)
  // Each segment: [label, flex, bgColor, borderColor]
  const BAR_SEGMENTS = [
    ["ASIA",     9,   "rgba(0,100,200,0.22)",   "rgba(0,170,255,0.25)"],
    ["LONDON",   6,   "rgba(180,120,0,0.22)",   "rgba(255,180,0,0.25)"],
    ["PRE",      1,   "rgba(255,100,0,0.32)",   "rgba(255,140,0,0.45)"],
    ["NY OPEN",  2,   "rgba(0,180,90,0.30)",    "rgba(0,255,136,0.40)"],
    ["DEAD",     1.5, "rgba(20,30,50,0.80)",    "rgba(60,80,110,0.25)"],
    ["AFT",      1.5, "rgba(180,120,0,0.20)",   "rgba(255,180,0,0.22)"],
    ["POWER",    1,   "rgba(255,80,0,0.30)",    "rgba(255,100,0,0.45)"],
    ["CLOSE",    1,   "rgba(80,50,0,0.28)",     "rgba(130,70,0,0.28)"],
  ];

  // Tick labels for the 24h bar (key times only)
  const TICKS = [
    { label: "6PM",   pct: 0 },
    { label: "9PM",   pct: 13.0 },
    { label: "12AM",  pct: 26.1 },
    { label: "3AM",   pct: 39.1 },
    { label: "6AM",   pct: 52.2 },
    { label: "8:30",  pct: 63.0 },
    { label: "9:30",  pct: 67.4 },
    { label: "11:30", pct: 76.1 },
    { label: "1:30",  pct: 84.8 },
    { label: "3PM",   pct: 91.3 },
    { label: "4PM",   pct: 95.7 },
    { label: "5PM",   pct: 100  },
  ];

  const NY_SLOTS = [
    { emoji:"⚡", time:"8:30–9:30 AM",  name:"Pre-Market",         kind:"good",  vol:70,  note:"News releases hit — high volatility, wide spreads", volColor:"#ff8c00" },
    { emoji:"🔥", time:"9:30 AM–12 PM", name:"NY Open Kill Zone",  kind:"best",  vol:95,  note:"Highest volume of the day — best setups",            volColor:"#00ff88" },
    { emoji:"😴", time:"12:00–3:00 PM", name:"Dead Zone",          kind:"avoid", vol:15,  note:"Choppy & random — avoid trading",                    volColor:"#ff3355" },
    { emoji:"⚡", time:"3:00–4:00 PM",  name:"Power Hour",         kind:"best",  vol:80,  note:"2nd highest volume — strong directional moves",      volColor:"#ff8c00" },
    { emoji:"🔔", time:"4:00–4:45 PM",  name:"Lucid Cutoff",       kind:"avoid", vol:30,  note:"All positions MUST close by 4:45 PM EST",            volColor:"#ffd700" },
  ];

  const EVENTS = [
    { emoji:"🏦", name:"FOMC Announcement",   time:"2:00 PM EST — 8× per year" },
    { emoji:"📊", name:"CPI Inflation Data",   time:"8:30 AM EST — Monthly" },
    { emoji:"💼", name:"NFP (Non-Farm Payrolls)", time:"8:30 AM EST — 1st Friday/month" },
    { emoji:"📉", name:"GDP Data",             time:"8:30 AM EST — Quarterly" },
    { emoji:"📋", name:"Jobless Claims",       time:"8:30 AM EST — Every Thursday" },
    { emoji:"🛢️", name:"EIA Oil Inventories",  time:"10:30 AM EST — Every Wednesday" },
  ];

  const slotBg = (kind) => kind === "best" ? "rgba(0,255,136,0.05)" : kind === "avoid" ? "rgba(255,51,85,0.04)" : "rgba(255,140,0,0.04)";
  const slotBorder = (kind) => kind === "best" ? "rgba(0,255,136,0.30)" : kind === "avoid" ? "rgba(255,51,85,0.25)" : "rgba(255,140,0,0.25)";
  const slotNameColor = (kind) => kind === "best" ? "#00ff88" : kind === "avoid" ? "#ff3355" : "#ff8c00";

  return (
    <div style={{ fontFamily:"'DM Mono',monospace", overflow:"visible" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Bodoni+Moda:ital,wght@1,900&display=swap');
        @keyframes refFadeIn { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
        .ref-card { transition: border-color .18s, transform .18s; }
        .ref-card:hover { transform: translateY(-1px); }
        .bar-seg:hover { filter: brightness(1.35); }
        .ref-sec-btn:hover { border-color: #1e3a5f !important; color: #93c5fd !important; }
      `}</style>

      {/* ── PAGE HEADER ── */}
      <div style={{ marginBottom:24, animation:"refFadeIn .35s ease" }}>
        <div style={{ height:2, background:"linear-gradient(90deg,#38bdf8,#818cf8,#c084fc,transparent)", borderRadius:1, marginBottom:14 }} />
        <div style={{ fontFamily:"'Bebas Neue',sans-serif", fontSize:32, letterSpacing:"0.12em", background:"linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent", backgroundClip:"text", lineHeight:1 }}>✦ TRADING SESSIONS</div>
        <div style={{ fontFamily:"'DM Mono',monospace", fontSize:8, color:"#334155", letterSpacing:"0.25em", marginTop:4 }}>SESSIONS · REMINDERS · RISK · LINKS · ALBUM · QUOTES</div>
      </div>

      {/* ── SECTION TABS ── */}
      <div style={{ display:"flex", gap:6, marginBottom:20, flexWrap:"wrap" }}>
        {SECTIONS.map(s => {
          const _fc = [...s.label][0];
          const emoji = _fc.codePointAt(0) > 127 ? _fc : null;
          const text  = emoji ? s.label.slice(_fc.length).trim() : s.label;
          const isActive = activeSection === s.id;
          return isActive ? (
            <button key={s.id} className="ref-sec-btn" onClick={() => setActiveSection(s.id)}
              style={{ position:"relative", padding:"9px 20px", borderRadius:6, fontFamily:"inherit", fontSize:13, cursor:"pointer", letterSpacing:"0.06em", transition:"all .15s", display:"flex", alignItems:"center", gap:6, overflow:"hidden",
                background:"rgba(10,18,32,0.95)", border:"1px solid #1e3a5f", color:"#e2e8f0" }}>
              <span style={{ position:"absolute", bottom:0, left:0, right:0, height:2, background:"linear-gradient(90deg,#38bdf8,#818cf8,#c084fc)", borderRadius:"0 0 5px 5px" }} />
              {emoji && <span style={{ fontSize:15, filter:"saturate(4) brightness(1.4)" }}>{emoji}</span>}
              {text}
            </button>
          ) : (
            <span key={s.id} style={{ display:"inline-block", padding:1, borderRadius:7, background:"linear-gradient(135deg,rgba(56,189,248,0.45),rgba(129,140,248,0.45),rgba(192,132,252,0.45))" }}>
              <button className="ref-sec-btn" onClick={() => setActiveSection(s.id)}
                style={{ display:"flex", alignItems:"center", gap:6, background:"#070d1a", color:"#64748b", border:"none", padding:"8px 19px", borderRadius:6, fontFamily:"inherit", fontSize:13, cursor:"pointer", letterSpacing:"0.06em", transition:"all .15s", whiteSpace:"nowrap" }}>
                {emoji && <span style={{ fontSize:15, filter:"saturate(4) brightness(1.4)" }}>{emoji}</span>}
                {text}
              </button>
            </span>
          );
        })}
      </div>

      {/* ════════════════════════════════════════════════
          SESSION MAP TAB
      ════════════════════════════════════════════════ */}
      {activeSection === "sessions" && (
        <div style={{ animation:"refFadeIn .3s ease" }}>
          <div style={{ display:"flex", gap:6, marginBottom:18 }}>
            {[
              ["map",       "🗺", "SESSION MAP", "hue-rotate(180deg) saturate(4) brightness(1.5)"],
              ["reminders", "📌", "REMINDERS",   "hue-rotate(340deg) saturate(5) brightness(1.4)"],
            ].map(([id, emoji, text, emojiFilter]) => {
              const isSel = sessInnerTab === id;
              return isSel ? (
                <button key={id} onClick={() => setSessInnerTab(id)}
                  style={{ position:"relative", padding:"5px 13px", borderRadius:4, fontFamily:"inherit", fontSize:10, cursor:"pointer", letterSpacing:"0.07em", transition:"all .15s", display:"flex", alignItems:"center", gap:5, overflow:"hidden",
                    background:"rgba(10,18,32,0.95)", border:"1px solid #1e3a5f", color:"#93c5fd" }}>
                  <span style={{ position:"absolute", bottom:0, left:0, right:0, height:1, background:"linear-gradient(90deg,#38bdf8,#c084fc)", borderRadius:"0 0 3px 3px" }} />
                  <span style={{ fontSize:12, filter: emojiFilter }}>{emoji}</span>
                  {text}
                </button>
              ) : (
                <span key={id} style={{ display:"inline-block", padding:1, borderRadius:5, background:"linear-gradient(135deg,rgba(56,189,248,0.45),rgba(129,140,248,0.45),rgba(192,132,252,0.45))" }}>
                  <button onClick={() => setSessInnerTab(id)}
                    style={{ display:"flex", alignItems:"center", gap:5, background:"#070d1a", color:"#64748b", border:"none", padding:"4px 12px", borderRadius:3, fontFamily:"inherit", fontSize:10, cursor:"pointer", letterSpacing:"0.07em", whiteSpace:"nowrap" }}>
                    <span style={{ fontSize:12, filter: emojiFilter }}>{emoji}</span>
                    {text}
                  </button>
                </span>
              );
            })}
          </div>

          {sessInnerTab === "map" && <>

          {/* 24-Hour Timeline */}
          <div style={{ background:"#060b18", border:"1px solid #0f1e30", borderRadius:8, padding:"22px 24px 20px", marginBottom:20, position:"relative", overflow:"hidden" }}>
            {/* Top accent bar */}
            <div style={{ position:"absolute", top:0, left:0, right:0, height:2, background:"linear-gradient(90deg, #00aaff, #00ff88, #ff8c00)" }}/>
            <div style={{ fontSize:9, color:"#3b82f6", letterSpacing:"0.15em", marginBottom:18 }}>⏱ 24-HOUR SESSION MAP (EST) — 6PM TO 5PM NEXT DAY</div>

            {/* Tick ruler */}
            <div style={{ position:"relative", height:26, marginBottom:4 }}>
              {TICKS.map(t => (
                <div key={t.label} style={{ position:"absolute", left:`${t.pct}%`, transform:"translateX(-50%)", display:"flex", flexDirection:"column", alignItems:"center" }}>
                  <div style={{ width:1, height:8, background:"#1e293b", opacity:0.8 }}/>
                  <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:8, color:"#64748b", marginTop:2, whiteSpace:"nowrap" }}>{t.label}</div>
                </div>
              ))}
            </div>

            {/* Bar */}
            <div style={{ display:"flex", height:48, borderRadius:6, overflow:"hidden", border:"1px solid #0f1e30" }}>
              {BAR_SEGMENTS.map(([label, flex, bg, bd]) => (
                <div key={label} className="bar-seg"
                  style={{ flex, background:bg, borderLeft:`1px solid ${bd}`, display:"flex", alignItems:"center", justifyContent:"center", cursor:"default", transition:"filter .2s" }}>
                  <span style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:9, color:"rgba(255,255,255,0.55)", letterSpacing:"0.5px", whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis", padding:"0 3px" }}>{label}</span>
                </div>
              ))}
            </div>

            {/* Legend */}
            <div style={{ display:"flex", gap:18, flexWrap:"wrap", marginTop:14 }}>
              {[
                ["Asia",        "rgba(0,170,255,0.55)"],
                ["London",      "rgba(255,200,0,0.55)"],
                ["Pre-Market",  "rgba(255,100,0,0.65)"],
                ["NY Open 🔥",  "rgba(0,255,136,0.65)"],
                ["Dead Zone",   "rgba(50,60,80,0.85)"],
                ["Power Hour",  "rgba(255,100,0,0.60)"],
              ].map(([lbl, c]) => (
                <div key={lbl} style={{ display:"flex", alignItems:"center", gap:6, fontSize:10, color:"#64748b", fontFamily:"'Share Tech Mono',monospace" }}>
                  <div style={{ width:10, height:10, borderRadius:2, background:c }}/>
                  {lbl}
                </div>
              ))}
            </div>
          </div>

          {/* Session Cards — Asia + London */}
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14, marginBottom:20 }}>
            {/* Asia */}
            <div className="ref-card" style={{ background:"#060b18", border:"1px solid rgba(0,170,255,0.18)", borderRadius:8, padding:"20px 22px", position:"relative", overflow:"hidden" }}>
              <div style={{ position:"absolute", bottom:0, left:0, right:0, height:2, background:"#00aaff", opacity:0.5 }}/>
              <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:12 }}>
                <span style={{ fontSize:26 }}>🌏</span>
                <div style={{ fontFamily:"'Rajdhani',sans-serif", fontSize:18, fontWeight:700, color:"#00aaff", letterSpacing:"0.08em", textTransform:"uppercase" }}>Asia Session</div>
              </div>
              <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:12, color:"#e2e8f0", background:"rgba(255,255,255,0.05)", border:"1px solid rgba(255,255,255,0.08)", borderRadius:5, padding:"5px 10px", marginBottom:12, display:"inline-block" }}>6:00 PM – 3:00 AM EST</div>
              <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
                {["Low ES/MES volume — choppy & ranging", "Tokyo peak: 7:00 PM – 1:00 AM", "Better for currencies & Asian indices", "⚠️ Not recommended for ES trading"].map(r => (
                  <div key={r} style={{ display:"flex", alignItems:"flex-start", gap:8, fontSize:12, color:"#94a3b8", lineHeight:1.4 }}>
                    <div style={{ width:5, height:5, borderRadius:"50%", background:"#00aaff", marginTop:5, flexShrink:0 }}/>
                    {r}
                  </div>
                ))}
              </div>
            </div>

            {/* London */}
            <div className="ref-card" style={{ background:"#060b18", border:"1px solid rgba(255,200,0,0.18)", borderRadius:8, padding:"20px 22px", position:"relative", overflow:"hidden" }}>
              <div style={{ position:"absolute", bottom:0, left:0, right:0, height:2, background:"#ffd700", opacity:0.5 }}/>
              <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:12 }}>
                <span style={{ fontSize:26 }}>🇬🇧</span>
                <div style={{ fontFamily:"'Rajdhani',sans-serif", fontSize:18, fontWeight:700, color:"#ffd700", letterSpacing:"0.08em", textTransform:"uppercase" }}>London Session</div>
              </div>
              <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:12, color:"#e2e8f0", background:"rgba(255,255,255,0.05)", border:"1px solid rgba(255,255,255,0.08)", borderRadius:5, padding:"5px 10px", marginBottom:12, display:"inline-block" }}>3:00 AM – 12:00 PM EST</div>
              <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
                {["ES volume builds from 3–4 AM", "Kill zone sweet spot: 2–5 AM", "Big moves start 7–9:30 AM as EU matures", "London close 10 AM–12 PM adds volume"].map(r => (
                  <div key={r} style={{ display:"flex", alignItems:"flex-start", gap:8, fontSize:12, color:"#94a3b8", lineHeight:1.4 }}>
                    <div style={{ width:5, height:5, borderRadius:"50%", background:"#ffd700", marginTop:5, flexShrink:0 }}/>
                    {r}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* NY teaser card */}
          {/* NEW YORK SESSION — inline below session map */}
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize:9, color:"#3b82f6", letterSpacing:"0.15em", marginBottom:12 }}>🗽 NEW YORK SESSION WINDOWS</div>

          <div style={{ background:"#060b18", border:"1px solid rgba(0,255,136,0.18)", borderRadius:8, padding:"22px 24px", marginBottom:20, position:"relative", overflow:"hidden" }}>
            <div style={{ position:"absolute", top:0, left:0, right:0, height:2, background:"#00ff88", opacity:0.5 }}/>
            <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:18 }}>
              <span style={{ fontSize:28 }}>🗽</span>
              <div>
                <div style={{ fontFamily:"'Rajdhani',sans-serif", fontSize:20, fontWeight:700, color:"#00ff88", letterSpacing:"0.08em" }}>NEW YORK SESSION</div>
                <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:11, color:"#64748b", marginTop:2 }}>9:30 AM – 5:00 PM EST &nbsp;|&nbsp; Futures: 6:00 PM – 5:00 PM next day</div>
              </div>
            </div>
            <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:12 }}>
              {NY_SLOTS.map(s => (
                <div key={s.name} style={{ background:slotBg(s.kind), border:`1px solid ${slotBorder(s.kind)}`, borderRadius:8, padding:"14px 14px 12px" }}>
                  <div style={{ fontSize:18, marginBottom:4 }}>{s.emoji}</div>
                  <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:10, color:"#64748b", marginBottom:4 }}>{s.time}</div>
                  <div style={{ fontFamily:"'Rajdhani',sans-serif", fontSize:13, fontWeight:700, color:slotNameColor(s.kind), textTransform:"uppercase", letterSpacing:"0.04em", marginBottom:6 }}>{s.name}</div>
                  <div style={{ display:"flex", alignItems:"flex-start", gap:6, fontSize:11, color:"#64748b", lineHeight:1.4, marginBottom:8 }}>
                    <div style={{ width:5, height:5, borderRadius:"50%", background:s.volColor, marginTop:4, flexShrink:0 }}/>
                    {s.note}
                  </div>
                  {/* Volume bar */}
                  <div style={{ height:4, borderRadius:2, background:"rgba(255,255,255,0.06)", overflow:"hidden" }}>
                    <div style={{ height:"100%", width:`${s.vol}%`, background:s.volColor, borderRadius:2, transition:"width 1s ease" }}/>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* NEWS EVENTS — merged into Session Map */}
          <div style={{ marginTop: 20 }}>
            <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.15em", marginBottom: 12 }}>📅 HIGH VOLUME NEWS EVENTS</div>

          <div style={{ background:"#060b18", border:"1px solid #0f1e30", borderRadius:8, padding:"22px 24px", position:"relative", overflow:"hidden" }}>
            <div style={{ position:"absolute", top:0, left:0, right:0, height:2, background:"linear-gradient(90deg,#ff8c00,#ffd700)" }}/>
            <div style={{ fontSize:10, color:"#3b82f6", letterSpacing:"0.15em", marginBottom:18 }}>📅 HIGH VOLUME NEWS EVENTS — WATCH THESE</div>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
              {EVENTS.map(ev => (
                <div key={ev.name} className="ref-card" style={{ display:"flex", alignItems:"center", gap:12, background:"rgba(0,0,0,0.25)", border:"1px solid #0f1e30", borderRadius:8, padding:"12px 14px" }}>
                  <span style={{ fontSize:20 }}>{ev.emoji}</span>
                  <div>
                    <div style={{ fontFamily:"'Rajdhani',sans-serif", fontSize:14, fontWeight:600, color:"#e2e8f0", letterSpacing:"0.03em" }}>{ev.name}</div>
                    <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:10, color:"#ff8c00", marginTop:2 }}>{ev.time}</div>
                  </div>
                </div>
              ))}
            </div>
            <div style={{ marginTop:18, padding:"14px 16px", background:"rgba(255,140,0,0.04)", border:"1px solid rgba(255,140,0,0.2)", borderRadius:6 }}>
              <div style={{ fontSize:11, color:"#ff8c00", letterSpacing:"0.1em", marginBottom:6 }}>⚠️ NEWS TRADING PROTOCOL</div>
              <div style={{ fontSize:11, color:"#64748b", lineHeight:1.7 }}>
                News releases on Lucid Flex are <span style={{ color:"#00ff88" }}>fully allowed</span>. Expect wide spreads and sharp moves in the 2 minutes before and after. Let the initial spike complete before entering. Best entries are usually the <span style={{ color:"#e2e8f0" }}>retest</span> of the news level.
              </div>
            </div>
          </div>
          </div>
          </div>

          </> /* end sessInnerTab map */}

          {sessInnerTab === "reminders" && (
                  <div style={{ display:"flex", flexDirection:"column", gap:14 }}>


          {/* Ontario / EST advantage */}
          <div style={{ background:"rgba(0,255,136,0.04)", border:"1px solid rgba(0,255,136,0.18)", borderRadius:8, padding:"18px 22px", display:"flex", alignItems:"flex-start", gap:16 }}>
            <span style={{ fontSize:28, flexShrink:0 }}>🍁</span>
            <div>
              <div style={{ fontFamily:"'Rajdhani',sans-serif", fontSize:14, fontWeight:700, color:"#00ff88", letterSpacing:"0.08em", marginBottom:8 }}>ONTARIO / EST ADVANTAGE</div>
              <div style={{ fontSize:12, color:"#94a3b8", lineHeight:1.8 }}>
                You're perfectly aligned with peak US futures volume.<br/>
                Primary window: <strong style={{ color:"#00ff88" }}>9:30 AM–12:00 PM EST</strong><br/>
                Secondary window: <strong style={{ color:"#ff8c00" }}>3:00–4:00 PM EST</strong><br/>
                <span style={{ color:"#ff3355" }}>All Lucid Flex positions must close by <strong style={{ color:"#ff3355" }}>4:45 PM EST</strong> daily.</span><br/>
                News trading is fully allowed on Lucid Flex.
              </div>
            </div>
          </div>

          {/* Quick reference grid */}
          <div style={{ background:"#060b18", border:"1px solid #0f1e30", borderRadius:8, padding:"18px 22px" }}>
            <div style={{ fontSize:9, color:"#3b82f6", letterSpacing:"0.15em", marginBottom:14 }}>QUICK REFERENCE</div>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
              {[
                { label:"BEST SESSION",      value:"NY Open 9:30 AM–12 PM",   c:"#00ff88" },
                { label:"2ND BEST",          value:"Power Hour 3:00–4:00 PM", c:"#ff8c00" },
                { label:"AVOID",             value:"Dead Zone 12:00–3:00 PM", c:"#ff3355" },
                { label:"POSITION CUTOFF",   value:"4:45 PM EST (Lucid Flex)", c:"#ff3355" },
                { label:"FUTURES OPEN",      value:"6:00 PM EST (Sunday–Friday)", c:"#94a3b8" },
                { label:"FUTURES CLOSE",     value:"5:00 PM EST daily",       c:"#94a3b8" },
                { label:"LONDON KILL ZONE",  value:"2:00–5:00 AM EST",        c:"#ffd700" },
                { label:"ASIA SESSION",      value:"6:00 PM – 3:00 AM EST",   c:"#00aaff" },
              ].map(item => (
                <div key={item.label} style={{ background:"rgba(0,0,0,0.25)", border:"1px solid #0a1220", borderRadius:6, padding:"10px 14px" }}>
                  <div style={{ fontSize:8, color:"#3b82f6", letterSpacing:"0.12em", marginBottom:4 }}>{item.label}</div>
                  <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:12, color:item.c }}>{item.value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Pre-session checklist */}
          <div style={{ background:"#060b18", border:"1px solid #0f1e30", borderRadius:8, padding:"18px 22px" }}>
            <div style={{ fontSize:9, color:"#3b82f6", letterSpacing:"0.15em", marginBottom:14 }}>✅ PRE-SESSION CHECKLIST</div>
            <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
              {[
                "Check economic calendar for 8:30 AM news before market open",
                "Identify overnight high / low and key levels from Asia/London session",
                "Mark pre-market high and low (8:30–9:30 AM)",
                "Note any gap between yesterday's close and today's open",
                "Set hard stop on all trades — no exceptions on Lucid Flex",
                "Confirm max daily loss limit not already hit",
                "Close all positions before 4:45 PM EST",
              ].map((item, i) => (
                <div key={i} style={{ display:"flex", alignItems:"flex-start", gap:10, fontSize:12, color:"#64748b", lineHeight:1.5 }}>
                  <div style={{ width:16, height:16, borderRadius:3, border:"1px solid #1e293b", flexShrink:0, marginTop:1, display:"flex", alignItems:"center", justifyContent:"center" }}>
                    <div style={{ width:6, height:6, borderRadius:1, background:"#1e3a5f" }}/>
                  </div>
                  {item}
                </div>
              ))}
            </div>
          </div>


          </div>
          )}

        </div>
      )}



      {activeSection === "risk" && <RiskCalcPanel />}

      {activeSection === "links" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 20, paddingBottom: 24 }}>

          {/* Header */}
          <div>
            <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.18em", marginBottom: 4 }}>🔗 RESOURCE LINKS — USEFUL TRADING REFERENCES</div>
            <div style={{ fontSize: 10, color: "#64748b", lineHeight: 1.6 }}>Save links you come across — news sources, tools, education, brokers. Click any link to open it directly.</div>
          </div>

          {/* Add / Edit Form */}
          <div style={{ background: "#070d1a", border: `1px solid ${editingLink ? "#3b82f6" : "#1e3a5f"}`, borderRadius: 6, padding: "16px 18px" }}>
            <div style={{ fontSize: 10, color: editingLink ? "#93c5fd" : "#64748b", letterSpacing: "0.12em", marginBottom: 12 }}>
              {editingLink ? "✏ EDIT LINK" : "+ ADD NEW LINK"}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <label style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em" }}>NAME *</label>
                <input
                  type="text" placeholder="e.g. TradingView, Investopedia..."
                  value={linkForm.name}
                  onChange={e => { setLinkForm(p => ({ ...p, name: e.target.value })); setLinkError(""); }}
                  style={{ fontSize: 12, padding: "7px 10px" }}
                />
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <label style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em" }}>CATEGORY</label>
                <select value={linkForm.cat} onChange={e => setLinkForm(p => ({ ...p, cat: e.target.value }))}
                  style={{ fontSize: 12, padding: "7px 10px" }}>
                  {LINK_CATS.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 10 }}>
              <label style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em" }}>URL *</label>
              <input
                type="text" placeholder="https://..."
                value={linkForm.url}
                onChange={e => { setLinkForm(p => ({ ...p, url: e.target.value })); setLinkError(""); }}
                style={{ fontSize: 12, padding: "7px 10px" }}
              />
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 12 }}>
              <label style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em" }}>SHORT DESCRIPTION <span style={{ color: "#475569" }}>(optional)</span></label>
              <input
                type="text" placeholder="What is this for?"
                value={linkForm.desc}
                onChange={e => setLinkForm(p => ({ ...p, desc: e.target.value }))}
                style={{ fontSize: 12, padding: "7px 10px" }}
              />
            </div>
            {linkError && <div style={{ fontSize: 11, color: "#f87171", marginBottom: 8 }}>⚠ {linkError}</div>}
            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={handleAddLink}
                style={{ background: editingLink ? "#1e3a5f" : "#1d4ed8", color: "white", border: "none", padding: "9px 22px", borderRadius: 4, fontFamily: "inherit", fontSize: 12, cursor: "pointer", letterSpacing: "0.06em" }}>
                {editingLink ? "✓ SAVE CHANGES" : "+ ADD LINK"}
              </button>
              {editingLink && (
                <button onClick={handleCancelEdit}
                  style={{ background: "transparent", border: "1px solid #1e293b", color: "#94a3b8", padding: "9px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 12, cursor: "pointer", letterSpacing: "0.06em" }}>
                  CANCEL
                </button>
              )}
            </div>
          </div>

          {/* Category filter */}
          {links.length > 0 && (
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {["All", ...LINK_CATS.filter(c => links.some(l => l.cat === c))].map(c => (
                <button key={c} onClick={() => setLinkFilter(c)}
                  style={{ padding: "5px 14px", borderRadius: 3, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.06em", transition: "all .15s",
                    background: linkFilter === c ? "#1e3a5f" : "transparent",
                    border: `1px solid ${linkFilter === c ? "#3b82f6" : "#1e293b"}`,
                    color: linkFilter === c ? "#93c5fd" : "#64748b" }}>
                  {c} {c !== "All" && <span style={{ color: "#475569" }}>({links.filter(l => l.cat === c).length})</span>}
                </button>
              ))}
            </div>
          )}

          {/* Links list */}
          {links.length === 0 ? (
            <div style={{ textAlign: "center", padding: "32px 0", color: "#334155", fontSize: 12, borderRadius: 6, border: "1px dashed #1e293b" }}>
              <div style={{ fontSize: 28, marginBottom: 8, opacity: 0.4 }}>🔗</div>
              No links saved yet — add your first one above
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              {(() => {
                const cats = linkFilter === "All"
                  ? LINK_CATS.filter(c => links.some(l => l.cat === c))
                  : [linkFilter];
                return cats.map(cat => {
                  const catLinks = links.filter(l => l.cat === cat);
                  if (!catLinks.length) return null;
                  return (
                    <div key={cat} style={{ marginBottom: 14 }}>
                      {/* Category header */}
                      <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.16em", textTransform: "uppercase", padding: "4px 0 6px", borderBottom: "1px solid #1e293b", marginBottom: 6 }}>
                        {cat} <span style={{ color: "#334155" }}>· {catLinks.length}</span>
                      </div>
                      {/* Rows */}
                      {catLinks.map(link => (
                        <div key={link.id}
                          style={{ display: "grid", gridTemplateColumns: "1fr auto", alignItems: "center", gap: 10, padding: "9px 12px", borderRadius: 4, background: editingLink === link.id ? "#070d1a" : "transparent", border: `1px solid ${editingLink === link.id ? "#1e3a5f" : "transparent"}`, transition: "all .15s" }}
                          onMouseEnter={e => { if (editingLink !== link.id) e.currentTarget.style.background = "#0a0e1a"; }}
                          onMouseLeave={e => { if (editingLink !== link.id) e.currentTarget.style.background = "transparent"; }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 12, minWidth: 0 }}>
                            {/* Favicon */}
                            <img
                              src={`https://www.google.com/s2/favicons?domain=${encodeURIComponent(link.url)}&sz=16`}
                              alt="" width={16} height={16}
                              style={{ borderRadius: 3, flexShrink: 0, opacity: 0.8 }}
                              onError={e => { e.target.style.display = "none"; }}
                            />
                            <div style={{ minWidth: 0 }}>
                              <a href={link.url} target="_blank" rel="noopener noreferrer"
                                onClick={e => e.stopPropagation()}
                                style={{ fontSize: 13, color: "#93c5fd", fontWeight: 500, textDecoration: "none", letterSpacing: "0.02em", display: "block", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
                                onMouseEnter={e => e.target.style.textDecoration = "underline"}
                                onMouseLeave={e => e.target.style.textDecoration = "none"}>
                                {link.name}
                              </a>
                              {link.desc && <div style={{ fontSize: 11, color: "#475569", marginTop: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{link.desc}</div>}
                            </div>
                          </div>
                          {/* Actions */}
                          <div style={{ display: "flex", gap: 4, flexShrink: 0, alignItems: "center" }}>
                            {confirmDeleteLink === link.id ? (
                              <>
                                <span style={{ fontSize: 10, color: "#f87171", marginRight: 4 }}>Delete?</span>
                                <button onClick={() => { saveLinks(links.filter(l => l.id !== link.id)); setConfirmDeleteLink(null); }}
                                  style={{ fontSize: 10, padding: "3px 10px", borderRadius: 3, fontFamily: "inherit", cursor: "pointer", background: "#450a0a", border: "1px solid #7f1d1d", color: "#f87171", letterSpacing: "0.04em" }}>YES</button>
                                <button onClick={() => setConfirmDeleteLink(null)}
                                  style={{ fontSize: 10, padding: "3px 10px", borderRadius: 3, fontFamily: "inherit", cursor: "pointer", background: "transparent", border: "1px solid #1e293b", color: "#64748b" }}>NO</button>
                              </>
                            ) : (
                              <>
                                <button onClick={() => handleEditLink(link)}
                                  style={{ fontSize: 10, padding: "3px 10px", borderRadius: 3, fontFamily: "inherit", cursor: "pointer", background: "transparent", border: "1px solid #1e293b", color: "#3b82f6", transition: "all .15s", letterSpacing: "0.04em" }}
                                  onMouseEnter={e => { e.target.style.borderColor = "#3b82f6"; e.target.style.color = "#93c5fd"; }}
                                  onMouseLeave={e => { e.target.style.borderColor = "#1e293b"; e.target.style.color = "#64748b"; }}>
                                  EDIT
                                </button>
                                <button onClick={() => setConfirmDeleteLink(link.id)}
                                  style={{ fontSize: 10, padding: "3px 10px", borderRadius: 3, fontFamily: "inherit", cursor: "pointer", background: "transparent", border: "1px solid #1e293b", color: "#3b82f6", transition: "all .15s", letterSpacing: "0.04em" }}
                                  onMouseEnter={e => { e.target.style.borderColor = "#7f1d1d"; e.target.style.color = "#f87171"; }}
                                  onMouseLeave={e => { e.target.style.borderColor = "#1e293b"; e.target.style.color = "#64748b"; }}>
                                  ✕
                                </button>
                              </>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  );
                });
              })()}
            </div>
          )}
        </div>
      )}

      {/* ══════════════════ TRADING ALBUM TAB ══════════════════ */}
      {activeSection === "album" && (
        <div style={{ animation: "refFadeIn .3s ease" }}
          onPaste={handleAlbumPaste}>

          {/* Header */}
          <div style={{ background: "#060b18", border: "1px solid #1e3a5f", borderRadius: 8, padding: "18px 22px", marginBottom: 20, position: "relative", overflow: "hidden" }}>
            <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg,#3b82f6,#8b5cf6,#ec4899)" }} />
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <span style={{ fontSize: 26 }}>📸</span>
              <div>
                <div style={{ fontFamily: "'Rajdhani',sans-serif", fontSize: 18, fontWeight: 700, color: "#93c5fd", letterSpacing: "0.08em" }}>TRADING ALBUM</div>
                <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>Chart setups, lessons, marked-up screenshots — your visual trading library</div>
              </div>
              <div style={{ marginLeft: "auto", fontSize: 10, color: "#334155", background: "#0a0e1a", border: "1px solid #1e293b", borderRadius: 4, padding: "6px 12px", letterSpacing: "0.06em" }}>
                {album.length} image{album.length !== 1 ? "s" : ""}
              </div>
            </div>
          </div>

          {/* Paste zone */}
          <div
            style={{ border: `2px dashed ${albumPasteActive ? "#3b82f6" : "#1e293b"}`, borderRadius: 8, padding: "28px 24px", textAlign: "center", marginBottom: 20, cursor: "pointer", background: albumPasteActive ? "rgba(59,130,246,0.04)" : "transparent", transition: "all .2s" }}
            onFocus={() => setAlbumPasteActive(true)}
            onBlur={() => setAlbumPasteActive(false)}
            onClick={() => { setAlbumPasteActive(true); }}
            tabIndex={0}
            onPaste={handleAlbumPaste}>
            {albumPending ? (
              <div style={{ display: "flex", flexDirection: "column", gap: 14, alignItems: "center" }}>
                <img src={albumPending} alt="preview" style={{ maxHeight: 220, maxWidth: "100%", borderRadius: 6, border: "1px solid #1e3a5f", objectFit: "contain" }} />
                <div style={{ display: "flex", flexDirection: "column", gap: 10, width: "100%", maxWidth: 560 }}>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                      <label style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em" }}>TITLE</label>
                      <input type="text" placeholder="e.g. MES breakout setup" value={albumForm.title} onChange={e => setAlbumForm(p => ({ ...p, title: e.target.value }))}
                        style={{ fontSize: 12, padding: "7px 10px" }} autoFocus />
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                      <label style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em" }}>CAPTION / NOTE</label>
                      <input type="text" placeholder="What does this chart show?" value={albumForm.caption} onChange={e => setAlbumForm(p => ({ ...p, caption: e.target.value }))}
                        style={{ fontSize: 12, padding: "7px 10px" }} />
                    </div>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    <label style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em" }}>TYPE</label>
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                      {ALBUM_CATS.map(cat => (
                        <button key={cat.id} type="button" onClick={() => setAlbumForm(p => ({ ...p, type: cat.id }))}
                          style={{ padding: "6px 14px", borderRadius: 20, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.04em", transition: "all .15s",
                            background: albumForm.type === cat.id ? cat.color + "22" : "transparent",
                            border: `1px solid ${albumForm.type === cat.id ? cat.color : "#1e293b"}`,
                            color: albumForm.type === cat.id ? cat.color : "#64748b" }}>
                          {cat.emoji} {cat.label}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <button onClick={confirmAddAlbumImage}
                    style={{ background: "#1d4ed8", color: "white", border: "none", padding: "9px 22px", borderRadius: 4, fontFamily: "inherit", fontSize: 12, cursor: "pointer", letterSpacing: "0.06em" }}>
                    + SAVE TO ALBUM
                  </button>
                  <button onClick={() => { setAlbumPending(null); setAlbumForm({ title: "", caption: "" }); }}
                    style={{ background: "transparent", border: "1px solid #1e293b", color: "#64748b", padding: "9px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 12, cursor: "pointer", letterSpacing: "0.06em" }}>
                    CANCEL
                  </button>
                </div>
              </div>
            ) : (
              <div>
                <div style={{ fontSize: 32, marginBottom: 10, opacity: 0.4 }}>📋</div>
                <div style={{ fontSize: 14, color: "#334155", marginBottom: 4 }}>Click here, then paste a screenshot</div>
                <div style={{ fontSize: 11, color: "#1e293b" }}>Ctrl+V / Cmd+V — works with any chart, screenshot, or image</div>
              </div>
            )}
          </div>

          {/* Filter bar */}
          {album.length > 0 && (
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 16, alignItems: "center" }}>
              <button onClick={() => setAlbumFilter("all")}
                style={{ padding: "6px 16px", borderRadius: 20, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.04em", transition: "all .15s",
                  background: albumFilter === "all" ? "rgba(255,255,255,0.08)" : "transparent",
                  border: `1px solid ${albumFilter === "all" ? "#475569" : "#1e293b"}`,
                  color: albumFilter === "all" ? "#e2e8f0" : "#64748b" }}>
                All ({album.length})
              </button>
              {ALBUM_CATS.map(cat => {
                const count = album.filter(img => (img.type || "chart") === cat.id).length;
                if (count === 0) return null;
                return (
                  <button key={cat.id} onClick={() => setAlbumFilter(cat.id)}
                    style={{ padding: "6px 16px", borderRadius: 20, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.04em", transition: "all .15s",
                      background: albumFilter === cat.id ? cat.color + "22" : "transparent",
                      border: `1px solid ${albumFilter === cat.id ? cat.color : "#1e293b"}`,
                      color: albumFilter === cat.id ? cat.color : "#64748b" }}>
                    {cat.emoji} {cat.label} ({count})
                  </button>
                );
              })}
            </div>
          )}

          {/* Grid */}
          {(() => {
            const filtered = albumFilter === "all" ? album : album.filter(img => (img.type || "chart") === albumFilter);
            return filtered.length === 0 ? (
            <div style={{ textAlign: "center", padding: "40px 0", color: "#1e293b", fontSize: 12 }}>
              {album.length === 0 ? "No images yet — paste your first chart above" : "No images in this category"}
            </div>
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 14 }}>
              {filtered.map(img => (
                <div key={img.id}
                  style={{ background: "#0a0e1a", border: "1px solid #1e293b", borderRadius: 6, overflow: "hidden", transition: "border-color .15s" }}
                  onMouseEnter={e => e.currentTarget.style.borderColor = "#3b82f6"}
                  onMouseLeave={e => e.currentTarget.style.borderColor = "#1e293b"}>

                  {/* Thumbnail */}
                  <div style={{ position: "relative", cursor: "pointer" }} onClick={() => setAlbumLightbox(img.id)}>
                    <img src={img.src} alt={img.title} style={{ width: "100%", height: 160, objectFit: "cover", display: "block" }} />
                    <div style={{ position: "absolute", inset: 0, background: "rgba(0,0,0,0)", transition: "background .15s", display: "flex", alignItems: "center", justifyContent: "center" }}
                      onMouseEnter={e => e.currentTarget.style.background = "rgba(0,0,0,0.35)"}
                      onMouseLeave={e => e.currentTarget.style.background = "rgba(0,0,0,0)"}>
                      <span style={{ fontSize: 22, opacity: 0, transition: "opacity .15s" }}
                        onMouseEnter={e => { e.currentTarget.style.opacity = 1; }}
                        onMouseLeave={e => { e.currentTarget.style.opacity = 0; }}>🔍</span>
                    </div>
                  </div>

                  {/* Info */}
                  <div style={{ padding: "10px 12px" }}>
                    {albumEditing === img.id ? (
                      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                        <input type="text" value={albumForm.title} onChange={e => setAlbumForm(p => ({ ...p, title: e.target.value }))}
                          placeholder="Title" style={{ fontSize: 11, padding: "5px 8px" }} autoFocus />
                        <input type="text" value={albumForm.caption} onChange={e => setAlbumForm(p => ({ ...p, caption: e.target.value }))}
                          placeholder="Caption / note" style={{ fontSize: 11, padding: "5px 8px" }} />
                        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                          {ALBUM_CATS.map(cat => (
                            <button key={cat.id} type="button" onClick={() => setAlbumForm(p => ({ ...p, type: cat.id }))}
                              style={{ padding: "3px 10px", borderRadius: 20, fontFamily: "inherit", fontSize: 10, cursor: "pointer", transition: "all .15s",
                                background: albumForm.type === cat.id ? cat.color + "22" : "transparent",
                                border: `1px solid ${albumForm.type === cat.id ? cat.color : "#1e293b"}`,
                                color: albumForm.type === cat.id ? cat.color : "#475569" }}>
                              {cat.emoji} {cat.label}
                            </button>
                          ))}
                        </div>
                        <div style={{ display: "flex", gap: 6 }}>
                          <button onClick={() => saveAlbumEdit(img.id)}
                            style={{ flex: 1, background: "#1d4ed8", color: "white", border: "none", padding: "5px 0", borderRadius: 3, fontFamily: "inherit", fontSize: 10, cursor: "pointer", letterSpacing: "0.06em" }}>SAVE</button>
                          <button onClick={() => setAlbumEditing(null)}
                            style={{ flex: 1, background: "transparent", border: "1px solid #1e293b", color: "#64748b", padding: "5px 0", borderRadius: 3, fontFamily: "inherit", fontSize: 10, cursor: "pointer" }}>CANCEL</button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                          {(() => { const cat = ALBUM_CATS.find(c => c.id === (img.type || "chart")); return cat ? <span style={{ fontSize: 9, padding: "1px 7px", borderRadius: 10, background: cat.color + "18", border: `1px solid ${cat.color}44`, color: cat.color, letterSpacing: "0.06em", flexShrink: 0 }}>{cat.emoji} {cat.label}</span> : null; })()}
                        </div>
                        <div style={{ fontSize: 12, color: "#e2e8f0", fontWeight: 600, marginBottom: 3, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{img.title}</div>
                        {img.caption && <div style={{ fontSize: 10, color: "#64748b", lineHeight: 1.5, marginBottom: 6 }}>{img.caption}</div>}
                        <div style={{ fontSize: 9, color: "#334155", marginBottom: 8 }}>{new Date(img.addedAt).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}</div>
                        <div style={{ display: "flex", gap: 6 }}>
                          <button onClick={() => { setAlbumEditing(img.id); setAlbumForm({ title: img.title, caption: img.caption || "", type: img.type || "chart" }); }}
                            style={{ flex: 1, background: "transparent", border: "1px solid #1e293b", color: "#3b82f6", padding: "4px 0", borderRadius: 3, fontFamily: "inherit", fontSize: 10, cursor: "pointer", letterSpacing: "0.04em", transition: "all .15s" }}
                            onMouseEnter={e => { e.currentTarget.style.borderColor = "#3b82f6"; e.currentTarget.style.color = "#93c5fd"; }}
                            onMouseLeave={e => { e.currentTarget.style.borderColor = "#1e293b"; e.currentTarget.style.color = "#64748b"; }}>
                            EDIT
                          </button>
                          {albumConfirmDelete === img.id ? (
                            <>
                              <button onClick={() => { saveAlbum(album.filter(a => a.id !== img.id)); setAlbumConfirmDelete(null); }}
                                style={{ flex: 1, background: "#450a0a", border: "1px solid #7f1d1d", color: "#f87171", padding: "4px 0", borderRadius: 3, fontFamily: "inherit", fontSize: 10, cursor: "pointer" }}>DELETE</button>
                              <button onClick={() => setAlbumConfirmDelete(null)}
                                style={{ flex: 1, background: "transparent", border: "1px solid #1e293b", color: "#64748b", padding: "4px 0", borderRadius: 3, fontFamily: "inherit", fontSize: 10, cursor: "pointer" }}>NO</button>
                            </>
                          ) : (
                            <button onClick={() => setAlbumConfirmDelete(img.id)}
                              style={{ flex: 1, background: "transparent", border: "1px solid #1e293b", color: "#475569", padding: "4px 0", borderRadius: 3, fontFamily: "inherit", fontSize: 10, cursor: "pointer", transition: "all .15s" }}
                              onMouseEnter={e => { e.currentTarget.style.borderColor = "#7f1d1d"; e.currentTarget.style.color = "#f87171"; }}
                              onMouseLeave={e => { e.currentTarget.style.borderColor = "#1e293b"; e.currentTarget.style.color = "#475569"; }}>
                              ✕
                            </button>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>
          );
          })()}

          {/* Lightbox */}
          {albumLightbox && (() => {
            const img = album.find(a => a.id === albumLightbox);
            if (!img) return null;
            return (
              <div onClick={() => setAlbumLightbox(null)}
                style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.92)", zIndex: 9999, display: "flex", alignItems: "center", justifyContent: "center", padding: 24 }}>
                <div onClick={e => e.stopPropagation()} style={{ display: "flex", flexDirection: "column", alignItems: "center", maxWidth: "90vw", maxHeight: "90vh", gap: 14 }}>
                  <img src={img.src} alt={img.title} style={{ maxWidth: "100%", maxHeight: "78vh", objectFit: "contain", borderRadius: 6, border: "1px solid #1e3a5f" }} />
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 15, color: "#e2e8f0", fontWeight: 600, marginBottom: 4 }}>{img.title}</div>
                    {img.caption && <div style={{ fontSize: 12, color: "#64748b" }}>{img.caption}</div>}
                  </div>
                  <button onClick={() => setAlbumLightbox(null)}
                    style={{ background: "transparent", border: "1px solid #334155", color: "#94a3b8", padding: "8px 24px", borderRadius: 4, fontFamily: "inherit", fontSize: 12, cursor: "pointer", letterSpacing: "0.06em" }}>
                    CLOSE ✕
                  </button>
                </div>
              </div>
            );
          })()}
        </div>
      )}


      {/* ══════════════════ QUOTES TAB ══════════════════ */}
      {activeSection === "quotes" && (
        <div style={{ animation: "refFadeIn .3s ease" }}>
          <QuotesView />
        </div>
      )}

    </div>
  );
}
// ─────────────────────────────────────────────────────────────────────────────
const DURATION_HARD_LIMIT_SECS = 0; // No hard duration cap — overnight trades allowed

const validateTradesHard = (trades, allowOvernight = false) => {
  const limit = 0; // Duration no longer used as a rejection criterion
  const accepted = [], rejected = [];
  const seen = new Set();
  for (const t of trades) {
    const reasons = [];
    if (!t.symbol || !t.symbol.trim()) reasons.push("Missing symbol");
    if (!Number.isFinite(t.qty) || t.qty <= 0) reasons.push("Invalid qty");
    if (!Number.isFinite(t.pnl)) reasons.push("Non-finite P&L");
    // Price=0 is a soft issue (IBKR may omit prices when P&L is provided directly)
    // Duration cap removed — overnight/multi-day trades are valid
    const h = [t.symbol, t.buyTime, t.sellTime, t.qty, t.buyPrice, t.sellPrice].join("|");
    if (seen.has(h)) reasons.push("Duplicate");
    else seen.add(h);
    if (reasons.length) rejected.push({ trade: t, reasons });
    else accepted.push(t);
  }
  return { accepted, rejected };
};

const validateTradesSoft = (trades) => {
  const clean = [], flagged = [];
  for (const t of trades) {
    const warnings = [];
    const isCarry = t.notes === 'overnight-carry';
    if (t.durationSecs > 0 && t.durationSecs < 2) warnings.push("⚡ Sub-2s duration — verify fill");
    // Overnight-carry trades legitimately have one missing price (entry was in prior session's file)
    if (!isCarry && (t.buyPrice <= 0 || t.sellPrice <= 0)) warnings.push("⚠ Price missing — P&L taken from broker data");
    if (isCarry) warnings.push("🌙 Overnight carry-forward — entry opened in prior session's CSV · P&L is accurate");
    if (Math.abs(t.pnl) > 50000) warnings.push("💰 Very large P&L — confirm correct");
    else if (Math.abs(t.pnl) > 5000 && (t.symbol?.startsWith("M") || t.qty < 3)) warnings.push("💰 Large P&L for micro — confirm correct");
    if (t.pnl === 0) warnings.push("⚪ Zero P&L — breakeven or parse error?");
    // Carry-forward trades have one empty timestamp by design — not an error
    if (!isCarry && !t.buyTime && !t.sellTime) warnings.push("🕐 No timestamps — session attribution unavailable");
    if (warnings.length) flagged.push({ trade: t, warnings });
    else clean.push(t);
  }
  return { clean, flagged };
};

// ─────────────────────────────────────────────────────────────────────────────
// Feature 2: Parse Review Modal
// ─────────────────────────────────────────────────────────────────────────────
function ParseReviewModal({ data, onConfirm, onCancel }) {
  const { flagged, clean } = data;
  const [checked, setChecked] = useState(() => Object.fromEntries(flagged.map((_, i) => [i, true])));
  const [allowOvernight, setAllowOvernight] = useState(false);
  const [showRejected, setShowRejected] = useState(true);
  const [showClean, setShowClean] = useState(false);
  const [forceAccepted, setForceAccepted] = useState({}); // hard-rejected trades the user overrides

  // Re-run hard validation when overnight toggle changes
  const allTrades = [...clean, ...flagged.map(f => f.trade), ...data.hardRejected.map(r => r.trade)];
  const revalidated = validateTradesHard(allTrades, allowOvernight);
  const hardRejected = revalidated.rejected;
  const revalidatedClean = revalidated.accepted.filter(t =>
    !flagged.some(f => f.trade === t)
  );

  const acceptedFlagged = flagged.filter((_, i) => checked[i]).map(f => f.trade);
  const rejectedCount   = flagged.filter((_, i) => !checked[i]).length;
  const forcedTrades    = hardRejected.filter((_, i) => forceAccepted[i]).map(r => r.trade);
  const finalTrades     = [...clean, ...acceptedFlagged, ...forcedTrades];

  const fmtD = (s) => !s ? "—" : s < 60 ? `${s}s` : s < 3600 ? `${Math.floor(s/60)}m${s%60}s` : `${(s/3600).toFixed(1)}h`;
  const fmtP = (n) => `${n >= 0 ? "+" : ""}$${Math.abs(n).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  const TRow = ({ t }) => (
    <div style={{ display: "grid", gridTemplateColumns: "90px 40px 48px 80px 80px 80px 70px", gap: 8, fontSize: 11, color: "#94a3b8", padding: "5px 0", borderBottom: "1px solid #0f1729" }}>
      <span style={{ color: "#e2e8f0", fontWeight: 500 }}>{t.symbol}</span>
      <span>{t.qty}</span>
      <span style={{ color: t.direction === "short" ? "#f87171" : "#4ade80" }}>{t.direction}</span>
      <span>${t.buyPrice}</span>
      <span>${t.sellPrice}</span>
      <span style={{ color: t.pnl >= 0 ? "#4ade80" : "#f87171" }}>{fmtP(t.pnl)}</span>
      <span>{fmtD(t.durationSecs)}</span>
    </div>
  );

  const ColHdr = () => (
    <div style={{ display: "grid", gridTemplateColumns: "90px 40px 48px 80px 80px 80px 70px", gap: 8, fontSize: 9, color: "#3b82f6", letterSpacing: "0.08em", marginBottom: 4 }}>
      <span>SYMBOL</span><span>QTY</span><span>DIR</span><span>BUY</span><span>SELL</span><span>P&L</span><span>DUR</span>
    </div>
  );

  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.88)", zIndex: 900, display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}>
      <div style={{ background: "#0f1729", border: "1px solid #1e3a5f", borderRadius: 8, width: "100%", maxWidth: 960, maxHeight: "88vh", display: "flex", flexDirection: "column", boxShadow: "0 24px 80px rgba(0,0,0,0.8)" }}>
        {/* Header */}
        <div style={{ padding: "16px 20px", borderBottom: "1px solid #1e293b", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div style={{ fontSize: 13, color: "#93c5fd", fontWeight: 600, letterSpacing: "0.08em" }}>🔍 REVIEW PARSED TRADES</div>
            <div style={{ fontSize: 10, color: "#64748b", marginTop: 3 }}>Review before saving — accept or reject flagged trades individually.</div>
          </div>
          <div style={{ display: "flex", gap: 12, fontSize: 11 }}>
            <span style={{ color: "#4ade80" }}>✓ {clean.length} clean</span>
            {flagged.length > 0 && <span style={{ color: "#f59e0b" }}>⚠ {flagged.length} flagged</span>}
            {hardRejected.length > 0 && <span style={{ color: "#f87171" }}>✕ {hardRejected.length} rejected</span>}
          </div>
        </div>

        {/* Body */}
        <div style={{ flex: 1, overflowY: "auto", padding: "16px 20px", display: "flex", flexDirection: "column", gap: 16 }}>

          {/* Flagged trades — expanded, accept/reject per trade */}
          {flagged.length > 0 && (
            <div>
              <div style={{ fontSize: 10, color: "#f59e0b", letterSpacing: "0.1em", marginBottom: 10 }}>⚠ FLAGGED — REVIEW REQUIRED ({flagged.length})</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {flagged.map((item, i) => (
                  <div key={i} style={{ background: "#0a0e1a", border: `1px solid ${checked[i] ? "#92400e" : "#1e293b"}`, borderRadius: 6, padding: "12px 14px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                      <div style={{ flex: 1 }}>
                        <TRow t={item.trade} />
                        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 8 }}>
                          {item.warnings.map((w, wi) => (
                            <span key={wi} style={{ fontSize: 9, color: "#f59e0b", background: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.25)", padding: "2px 8px", borderRadius: 3 }}>{w}</span>
                          ))}
                        </div>
                      </div>
                      <div style={{ display: "flex", gap: 6, marginLeft: 16, flexShrink: 0 }}>
                        <button onClick={() => setChecked(p => ({ ...p, [i]: true }))}
                          style={{ padding: "5px 12px", borderRadius: 3, fontSize: 10, fontFamily: "inherit", cursor: "pointer", letterSpacing: "0.06em", background: checked[i] ? "#052e16" : "transparent", border: `1px solid ${checked[i] ? "#166534" : "#1e293b"}`, color: checked[i] ? "#4ade80" : "#475569", transition: "all .15s" }}>
                          ✓ ACCEPT
                        </button>
                        <button onClick={() => setChecked(p => ({ ...p, [i]: false }))}
                          style={{ padding: "5px 12px", borderRadius: 3, fontSize: 10, fontFamily: "inherit", cursor: "pointer", letterSpacing: "0.06em", background: !checked[i] ? "#1f0606" : "transparent", border: `1px solid ${!checked[i] ? "#7f1d1d" : "#1e293b"}`, color: !checked[i] ? "#f87171" : "#475569", transition: "all .15s" }}>
                          ✕ REJECT
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Hard rejected — collapsed, read-only */}
          {hardRejected.length > 0 && (
            <div>
              <button onClick={() => setShowRejected(p => !p)}
                style={{ background: "transparent", border: "none", cursor: "pointer", color: "#f87171", fontSize: 10, letterSpacing: "0.1em", padding: "4px 0", fontFamily: "inherit", display: "flex", alignItems: "center", gap: 6 }}>
                ✕ AUTO-REJECTED ({hardRejected.length}) {showRejected ? "▾" : "▸"} — cannot be saved
              </button>
              {showRejected && (
                <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
                  {hardRejected.map((item, i) => (
                    <div key={i} style={{ background: "#0a0e1a", border: `1px solid ${forceAccepted[i] ? "#166534" : "#450a0a"}`, borderRadius: 6, padding: "10px 14px" }}>
                      <TRow t={item.trade} />
                      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 6, alignItems: "center" }}>
                        {item.reasons.map((r, ri) => (
                          <span key={ri} style={{ fontSize: 9, color: "#f87171", background: "rgba(248,113,113,0.07)", border: "1px solid #450a0a", padding: "2px 8px", borderRadius: 3 }}>{r}</span>
                        ))}
                        <button onClick={() => setForceAccepted(p => ({ ...p, [i]: !p[i] }))}
                          style={{ marginLeft: "auto", fontSize: 9, padding: "2px 10px", borderRadius: 3, fontFamily: "inherit", cursor: "pointer", letterSpacing: "0.06em", transition: "all .15s",
                            background: forceAccepted[i] ? "#052e16" : "transparent",
                            border: `1px solid ${forceAccepted[i] ? "#166534" : "#7f1d1d"}`,
                            color: forceAccepted[i] ? "#4ade80" : "#f87171" }}>
                          {forceAccepted[i] ? "✓ FORCE ACCEPTED" : "FORCE ACCEPT"}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              {/* Overnight override toggle */}
              <label style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 10, cursor: "pointer" }}>
                <div onClick={() => setAllowOvernight(p => !p)}
                  style={{ width: 14, height: 14, borderRadius: 3, border: `1px solid ${allowOvernight ? "#3b82f6" : "#475569"}`, background: allowOvernight ? "#3b82f6" : "transparent", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, cursor: "pointer" }}>
                  {allowOvernight && <span style={{ color: "white", fontSize: 9, lineHeight: 1 }}>✓</span>}
                </div>
                <span style={{ fontSize: 10, color: "#64748b" }}>This session included overnight holds — relax duration limit to 24h</span>
              </label>
            </div>
          )}

          {/* Clean trades summary — collapsed */}
          {clean.length > 0 && (
            <div>
              <button onClick={() => setShowClean(p => !p)}
                style={{ background: "transparent", border: "none", cursor: "pointer", color: "#4ade80", fontSize: 10, letterSpacing: "0.1em", padding: "4px 0", fontFamily: "inherit", display: "flex", alignItems: "center", gap: 6 }}>
                ✓ CLEAN TRADES ({clean.length}) {showClean ? "▾" : "▸"}
              </button>
              {showClean && (
                <div style={{ marginTop: 8, background: "#0a0e1a", border: "1px solid #166534", borderRadius: 6, padding: "12px 14px" }}>
                  <ColHdr />
                  {clean.map((t, i) => <TRow key={i} t={t} />)}
                </div>
              )}
            </div>
          )}

          {/* Warning if > 50% hard rejected */}
          {hardRejected.length > (hardRejected.length + flagged.length + clean.length) / 2 && (
            <div style={{ background: "rgba(63,16,16,0.5)", border: "1px solid #7f1d1d", borderRadius: 4, padding: "10px 14px", fontSize: 11, color: "#f87171" }}>
              ⚠ More than half of parsed trades were auto-rejected. Check that your export includes both entry and exit fills.
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{ padding: "14px 20px", borderTop: "1px solid #1e293b", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ fontSize: 11, color: "#64748b" }}>
            Saving <span style={{ color: "#e2e8f0", fontWeight: 600 }}>{finalTrades.length}</span> trade{finalTrades.length !== 1 ? "s" : ""}
            {rejectedCount > 0 && <> · discarding <span style={{ color: "#f87171" }}>{rejectedCount}</span></>}
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button onClick={onCancel}
              style={{ background: "transparent", border: "1px solid #1e293b", color: "#64748b", padding: "8px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.06em" }}>
              CANCEL
            </button>
            <button
              disabled={finalTrades.length === 0}
              onClick={() => onConfirm(finalTrades, allowOvernight, {
                hardRejected: hardRejected.length - forcedTrades.length,
                softFlagged: flagged.length,
                userRejected: rejectedCount,
                accepted: finalTrades.length,
              })}
              style={{ background: finalTrades.length ? "#1d4ed8" : "#1e293b", color: "white", border: "none", padding: "8px 22px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: finalTrades.length ? "pointer" : "not-allowed", letterSpacing: "0.06em", opacity: finalTrades.length ? 1 : 0.5, transition: "all .15s" }}>
              CONFIRM & SAVE {finalTrades.length} TRADES →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

const DEFAULT_JOURNAL_ID = "default";
const DEFAULT_JOURNAL_NAME = "Futures Journal";

export default function TradingJournal() {
  const [journals, setJournals] = useState([{ id: DEFAULT_JOURNAL_ID, name: DEFAULT_JOURNAL_NAME, createdAt: Date.now(), type: JOURNAL_TYPES.PERSONAL, config: defaultPersonalConfig() }]);
  const [activeJournalId, setActiveJournalId] = useState(DEFAULT_JOURNAL_ID);
  const [entries, setEntries] = useState([]);
  const [journalsLoaded, setJournalsLoaded] = useState(false);
  // Journal management UI state
  const [showJournalMgr, setShowJournalMgr] = useState(false);
  const [newJournalName, setNewJournalName] = useState("");
  const [newJournalType, setNewJournalType] = useState(JOURNAL_TYPES.PERSONAL);
  const [newJournalConfig, setNewJournalConfig] = useState(defaultPersonalConfig());
  const [renamingId, setRenamingId] = useState(null);
  const [renameValue, setRenameValue] = useState("");
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);
  const [confirmDeleteEntry, setConfirmDeleteEntry] = useState(false);

  const [view, setView] = useState("list");
  const [propDashTab, setPropDashTab] = useState("overview"); // "overview" | "account" | "cumulative"
  const [recapInitMode, setRecapInitMode] = useState("weekly");
  const [activeEntry, setActiveEntry] = useState(null);
  const [form, setForm] = useState(emptyEntry());
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [filterMonth, setFilterMonth] = useState("");
  const [tab, setTab] = useState("session");
  const [sessionInnerTab, setSessionInnerTab] = useState("session");
  const [importRaw, setImportRaw] = useState("");
  const [importError, setImportError] = useState("");
  const [importSuccess, setImportSuccess] = useState(false);
  const [analyticsTab, setAnalyticsTab] = useState("overview");
  const [listMode, setListMode] = useState("calendar");
  const [calMonth, setCalMonth] = useState(() => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`; });
  // calendarNotes state is now managed inside CalendarView to avoid
  // full-app re-renders on every keystroke
  const [headerQuotes, setHeaderQuotes] = useState(() => { const s=[...SEED_QUOTES].sort(()=>Math.random()-0.5); return [s[0],s[1],s[2]].filter(Boolean); });

  // AI settings (local only)
  const [aiCfg, setAiCfg] = useState(() => loadAiSettings());
  const [aiTestStatus, setAiTestStatus] = useState(null); // { type: 'ok'|'err', text }

  // Data health UI
  const [dataHealthReport, setDataHealthReport] = useState(null);
  const [showDataHealthDetails, setShowDataHealthDetails] = useState(false);

  // Load journals meta + active journal entries on mount
  useEffect(() => {
    const load = async () => {
      try {
        // Load journals meta
        let loadedJournals = [{ id: DEFAULT_JOURNAL_ID, name: DEFAULT_JOURNAL_NAME, createdAt: Date.now() }];
        try {
          const metaResult = await storage.get("journal-meta");
          if (metaResult?.value) loadedJournals = JSON.parse(metaResult.value);
        } catch {}
        setJournals(loadedJournals);

        // Always select the first journal in the list on load
        const activeId = loadedJournals[0]?.id || DEFAULT_JOURNAL_ID;
        setActiveJournalId(activeId);
        let entriesData = null;

        // Try new key first
        try {
          const r = await storage.get(`journal-entries-${activeId}`);
          if (r?.value) entriesData = r.value;
        } catch {}

        // If not found, migrate from old journal-v3 key
        if (!entriesData) {
          try {
            const oldResult = await storage.get("journal-v3");
            if (oldResult?.value) {
              entriesData = oldResult.value;
              // Save under new key so migration only runs once
              await storage.set(`journal-entries-${activeId}`, entriesData);
            }
          } catch {}
        }

        if (entriesData) {
          const parsed = JSON.parse(entriesData);
          const norm = normalizeEntries(parsed);
          setEntries(norm.entries);
          if (norm.report?.changed) {
            try { await storage.set(`journal-entries-${activeId}`, JSON.stringify(norm.entries)); } catch {}
            setDataHealthReport(norm.report);
          }
        }

        // Load a random quote for the header
        try {
          const qr = await storage.get("trader-quotes-v1");
          const allQuotes = qr?.value ? JSON.parse(qr.value) : SEED_QUOTES;
          if (allQuotes.length > 0) { const sh=[...allQuotes].sort(()=>Math.random()-0.5); setHeaderQuotes([sh[0],sh[1],sh[2]].filter(Boolean)); }
        } catch {}
      } catch {}
      setJournalsLoaded(true);
    };
    load();
  }, []);

  const saveJournalsMeta = async (updated) => {
    setJournals(updated);
    await storage.set("journal-meta", JSON.stringify(updated));
  };

  const saveEntries = async (updated, journalId = activeJournalId) => {
    if (journalId === activeJournalId) setEntries(updated);
    await storage.set(`journal-entries-${journalId}`, JSON.stringify(updated));
  };

  const switchJournal = async (id) => {
    if (id === activeJournalId) return;
    setActiveJournalId(id);
    setView("list"); setActiveEntry(null); setForm(emptyEntry());
    setFilterMonth(""); setListMode("calendar");
    try {
      const result = await storage.get(`journal-entries-${id}`);
      const parsed = result?.value ? JSON.parse(result.value) : [];
      const norm = normalizeEntries(parsed);
      setEntries(norm.entries);
      if (norm.report?.changed) {
        try { await storage.set(`journal-entries-${id}`, JSON.stringify(norm.entries)); } catch {}
        setDataHealthReport(norm.report);
      }
    } catch { setEntries([]); }
  };

  const createJournal = async () => {
    const name = newJournalName.trim() || `Journal ${journals.length + 1}`;
    const id = `journal-${Date.now()}`;
    const newJ = { id, name, createdAt: Date.now(), type: newJournalType, config: newJournalConfig };
    await saveJournalsMeta([...journals, newJ]);
    await storage.set(`journal-entries-${id}`, JSON.stringify([]));
    // Reset form state
    setNewJournalName("");
    setNewJournalType(JOURNAL_TYPES.PERSONAL);
    setNewJournalConfig(defaultPersonalConfig());
    // Close modal immediately
    setShowJournalMgr(false);
    // Navigate to new journal home — set everything directly since switchJournal
    // guards against same-id switch which can race with state updates
    setActiveJournalId(id);
    setEntries([]);
    setView("list");
    setActiveEntry(null);
    setForm(emptyEntry());
    setFilterMonth("");
    setListMode("calendar");
  };

  const renameJournal = async (id, name) => {
    const updated = journals.map(j => j.id === id ? { ...j, name: name.trim() || j.name } : j);
    await saveJournalsMeta(updated);
    setRenamingId(null);
  };

  const deleteJournal = async (id) => {
    if (journals.length <= 1) return; // can't delete last journal
    const updated = journals.filter(j => j.id !== id);
    await saveJournalsMeta(updated);
    await storage.delete(`journal-entries-${id}`);
    setConfirmDeleteId(null);
    if (activeJournalId === id) switchJournal(updated[0].id);
  };

  const [aiParsing, setAiParsing] = useState(false);
  const [aiParseError, setAiParseError] = useState("");
  const [detectedFormat, setDetectedFormat] = useState("");
  const [csvFileName, setCsvFileName] = useState("");
  const [csvConfirmPending, setCsvConfirmPending] = useState(null); // { text, name } waiting for confirm
  // Feature 2: Trade review modal state
  const [parseReviewData, setParseReviewData] = useState(null);

  const handleImport = async () => {
    setImportError(""); setImportSuccess(false); setAiParseError(""); setDetectedFormat("");
    if (!importRaw.trim()) { setImportError("Please paste your broker data first."); return; }

    setAiParsing(true);
    try {
      // Try deterministic IBKR parser first — faster and more accurate than AI for known format
      const ibkrResult = parseIbkrCsv(importRaw);
      if (ibkrResult) {
        const tnorm = normalizeTrades(ibkrResult);
        if (tnorm.changed) setDataHealthReport({ changed: true, summary: `Data health update: ${tnorm.changes.join(", ")}.`, details: tnorm.changes });
        const trades = tnorm.trades;
        if (trades && trades.length > 0) {
          const { accepted: hardAccepted, rejected: hardRejected } = validateTradesHard(trades, false);
          const { clean, flagged } = validateTradesSoft(hardAccepted);
          const detectedFmt = "Interactive Brokers";
          setDetectedFormat(detectedFmt);
          if (hardRejected.length === 0 && flagged.length === 0) {
            applyParsedTrades(clean, detectedFmt, { hardRejected: 0, softFlagged: 0, userRejected: 0, accepted: clean.length });
          } else {
            setParseReviewData({ hardRejected, flagged, clean, allowOvernight: false, detectedFmt });
          }
          setAiParsing(false);
          return;
        }
      }
      // Fall back to AI parser for unknown formats
      const tradesRaw = await parseTradesWithAI(importRaw, aiCfg);
      const tnorm = normalizeTrades(tradesRaw);
      if (tnorm.changed) setDataHealthReport({ changed: true, summary: `Data health update: ${tnorm.changes.join(", ")}.`, details: tnorm.changes });
      const trades = tnorm.trades;
      if (!trades || trades.length === 0) {
        setAiParseError("No completed round-trip trades found. Check your data includes both entry and exit fills.");
        setAiParsing(false); return;
      }
      // Feature 2: Run validation pipeline
      const { accepted: hardAccepted, rejected: hardRejected } = validateTradesHard(trades, false);
      const { clean, flagged } = validateTradesSoft(hardAccepted);
      // Detect broker format from raw data
      const raw = importRaw.slice(0, 500).toLowerCase();
      const detectedFmt = raw.includes("tradovate") || raw.includes("lfe0") ? "Tradovate" :
                  raw.includes("interactive brokers") || raw.includes("ib ") ? "Interactive Brokers" :
                  raw.includes("questrade") ? "Questrade" :
                  raw.includes("ninjatrader") ? "NinjaTrader" :
                  raw.includes("rithmic") ? "Rithmic" : "Auto-detected";
      // If nothing to review, apply directly
      if (hardRejected.length === 0 && flagged.length === 0) {
        applyParsedTrades(clean, detectedFmt, { hardRejected: 0, softFlagged: 0, userRejected: 0, accepted: clean.length });
      } else {
        setParseReviewData({ hardRejected, flagged, clean, allowOvernight: false, detectedFmt });
      }
    } catch (err) {
      setAiParseError("AI parsing failed — " + (err.message || "unknown error") + ". Try again or check your data.");
    }
    setAiParsing(false);
  };

  const handleCsvUpload = (file) => {
    if (!file) return;
    if (!file.name.match(/\.csv$/i) && !file.name.match(/\.txt$/i) && !file.name.match(/\.tsv$/i)) {
      setImportError("Please upload a CSV, TSV, or TXT file."); return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      // If trades already exist, ask for confirmation
      if (form.parsedTrades?.length > 0) {
        setCsvConfirmPending({ text, name: file.name });
      } else {
        setCsvFileName(file.name);
        setImportRaw(text);
        setImportError(""); setImportSuccess(false); setDetectedFormat(""); setAiParseError("");
      }
    };
    reader.readAsText(file);
  };

  const confirmCsvReplace = () => {
    if (!csvConfirmPending) return;
    setCsvFileName(csvConfirmPending.name);
    setImportRaw(csvConfirmPending.text);
    setImportError(""); setImportSuccess(false); setDetectedFormat(""); setAiParseError("");
    setCsvConfirmPending(null);
  };

  const applyParsedTrades = (trades, fmt, logEntry) => {
    const totalPnL = trades.reduce((s, t) => s + t.pnl, 0);
    const totalComm = trades.reduce((s, t) => s + (t.commission || 0), 0);

    // Auto-detect trading day from trade timestamps.
    // Uses the MODE (most-frequent calendar date) across ALL entry+exit timestamps so that
    // overnight sessions starting on the prior evening are attributed to the correct
    // trading day (e.g. a file spanning Mar 11 evening → Mar 12 close = "2026-03-12").
    const autoDate = (() => {
      // Helper: extract YYYY-MM-DD from any supported broker timestamp format
      const extractDate = (raw) => {
        if (!raw) return null;
        const s = raw.trim();
        const patterns = [
          // YYYYMMDD HHMMSS  → "20260311 132635"
          { re: /^(\d{4})(\d{2})(\d{2})[\s;]/, fn: m => `${m[1]}-${m[2]}-${m[3]}` },
          // YYYYMMDD (no time)
          { re: /^(\d{4})(\d{2})(\d{2})$/, fn: m => `${m[1]}-${m[2]}-${m[3]}` },
          // YYYY/MM/DD or YYYY-MM-DD
          { re: /^(\d{4})[\/\-](\d{2})[\/\-](\d{2})/, fn: m => `${m[1]}-${m[2]}-${m[3]}` },
          // MM/DD/YYYY
          { re: /^(\d{1,2})\/(\d{1,2})\/(\d{4})/, fn: m => `${m[3]}-${m[1].padStart(2,'0')}-${m[2].padStart(2,'0')}` },
          // MM-DD-YYYY
          { re: /^(\d{1,2})-(\d{1,2})-(\d{4})/, fn: m => `${m[3]}-${m[1].padStart(2,'0')}-${m[2].padStart(2,'0')}` },
        ];
        for (const { re, fn } of patterns) { const m = s.match(re); if (m) return fn(m); }
        return null;
      };

      // Collect both entry and exit timestamps from every trade.
      // Overnight-carry trades have one empty timestamp — skip it.
      const dateCounts = {};
      for (const t of trades) {
        for (const ts of [t.buyTime, t.sellTime]) {
          const d = extractDate(ts);
          if (d) dateCounts[d] = (dateCounts[d] || 0) + 1;
        }
      }
      const dates = Object.keys(dateCounts);
      if (!dates.length) return null;
      // FIX: Use the LATEST date in the file as the trading day.
      // IBKR overnight sessions start at 6 PM the prior evening, so the MODE
      // (most-frequent date) incorrectly picks the prior calendar day.
      // The latest date is always the actual trading session date.
      return dates.sort().reverse()[0];
    })();

    // Use local date (not UTC) for today comparison to avoid timezone issues
    const localToday = new Date();
    const localTodayStr = `${localToday.getFullYear()}-${String(localToday.getMonth()+1).padStart(2,'0')}-${String(localToday.getDate()).padStart(2,'0')}`;

    // Auto-detect instruments from symbols
    const syms = [...new Set(trades.map(t => t.symbol))];
    const autoInstrs = [...new Set(syms.map(s =>
      s.startsWith("MNQ") ? "MNQ" : s.startsWith("NQ") ? "NQ" :
      s.startsWith("MES") ? "MES" : s.startsWith("ES") ? "ES" :
      s.startsWith("MGC") ? "MGC" : s.startsWith("GC") ? "GC" :
      s.startsWith("MCL") ? "MCL" : s.startsWith("CL") ? "CL" : s
    ))];

    setDetectedFormat(fmt);
    setForm(prev => ({
      ...prev,
      rawTradeData: importRaw,
      rawCsvFile: csvFileName ? { name: csvFileName, content: importRaw, savedAt: new Date().toISOString() } : (prev.rawCsvFile || null),
      parsedTrades: trades,
      // Auto-fill date: set if field is empty OR still showing today's placeholder date
      date:        autoDate && (!prev.date || prev.date === localTodayStr) ? autoDate : prev.date,
      // FIX: Always overwrite pnl and commissions on import — stale values from a
      // previous import would otherwise persist even after re-parsing corrected data.
      pnl:         totalPnL.toFixed(2),
      commissions: totalComm.toFixed(2),
      instruments: prev.instruments?.length ? prev.instruments : autoInstrs,
      parseValidationLog: [...(prev.parseValidationLog || []).slice(-4), { ts: Date.now(), ...logEntry }],
    }));
    setImportSuccess(true);
    setTab("analysis");
    setParseReviewData(null);
  };

  const handleSave = async () => {
    setSaving(true);
    const entry = { ...form };
    const entryWithId = { ...entry, id: entry.id || Date.now() };
    let updated = activeEntry ? entries.map(e => e.id === form.id ? entryWithId : e) : [entryWithId, ...entries];
    // Update state immediately so calendar renders right away
    setEntries(updated);
    // Persist to storage
    try {
      await storage.set(`journal-entries-${activeJournalId}`, JSON.stringify(updated));
    } catch (e) {
      console.warn("Save error:", e);
    }
    setSaved(true);
    // Auto-backup: download JSON snapshot on save if enabled
    if (aiCfg?.autoBackup) {
      try {
        const snap = { exportedAt: new Date().toISOString(), schemaVersion: 4, journals, entries: updated };
        const blob = new Blob([JSON.stringify(snap, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = `tj-auto-${new Date().toISOString().slice(0,10)}.json`; a.click();
        URL.revokeObjectURL(url);
      } catch {}
    }
    // Navigate to list — set calendar to the saved entry's month so it appears immediately
    const savedMonth = entry.date ? entry.date.slice(0, 7) : calMonth; // "2026-03"
    setCalMonth(savedMonth);
    setListMode("list");
    setView("list");
    setSortNewest(true);
    setActiveEntry(null);
    setForm(emptyEntry());
    setTab("session");
    setSessionInnerTab("session");
    setImportRaw(""); setImportError(""); setImportSuccess(false);
    setCsvFileName(""); setCsvConfirmPending(null);
    setSaving(false);
    setTimeout(() => setSaved(false), 500);
  };

  const handleDelete = async (id) => {
    await saveEntries(entries.filter(e => e.id !== id));
    setView("list"); setActiveEntry(null);
  };

  const getYesterdayPlan = () => {
    const today = new Date().toISOString().split("T")[0];
    const prior = [...entries]
      .filter(e => e.date < today)
      .sort((a, b) => b.date.localeCompare(a.date))[0];
    return prior ? { plan: prior.tomorrow, reinforceRule: prior.reinforceRule, date: prior.date } : null;
  };

  const openNew = () => { setForm(emptyEntry()); setActiveEntry(null); setTab("session"); setSessionInnerTab("session"); setImportRaw(""); setImportError(""); setImportSuccess(false); setCsvFileName(""); setCsvConfirmPending(null); setAnalyticsTab("overview"); setView("new"); };
  const openEdit = (entry) => { setForm({ ...entry }); setActiveEntry(entry); setTab("session"); setSessionInnerTab("session"); setImportRaw(entry.rawTradeData || ""); setImportError(""); setImportSuccess(false); setCsvFileName(entry.rawCsvFile?.name || ""); setCsvConfirmPending(null); setAnalyticsTab("overview"); setView("new"); };
  const viewDetail = (entry) => { setActiveEntry(entry); setView("detail"); };

  const f = (field, val) => setForm(p => ({ ...p, [field]: val }));

  // Auto-bullet: on paste into any notes field, prefix every non-empty line with "- "
  // Normalises •, *, –, — and existing - to uniform "- " prefix
  const autoBulletPaste = (field) => (e) => {
    const pasted = e.clipboardData?.getData('text');
    if (!pasted) return;
    e.preventDefault();
    const lines = pasted.split('\n');
    const formatted = lines
      .map(line => {
        const trimmed = line.trimEnd();
        if (!trimmed.trim()) return ''; // keep blank lines as blank
        // Strip any existing bullet-like prefix
        const stripped = trimmed.replace(/^[\s]*[-–—•*]\s*/, '');
        return `- ${stripped}`;
      })
      .join('\n');
    // Insert at cursor position if possible, otherwise replace full value
    const el = e.target;
    const start = el.selectionStart ?? 0;
    const end   = el.selectionEnd   ?? 0;
    const current = el.value;
    const newVal = current.slice(0, start) + formatted + current.slice(end);
    f(field, newVal);
    // Restore cursor after state update
    requestAnimationFrame(() => {
      el.selectionStart = el.selectionEnd = start + formatted.length;
    });
  };
  const pnlColor = (n) => { const v = parseFloat(n); return isNaN(v) ? "#e2e8f0" : v > 0 ? "#4ade80" : v < 0 ? "#f87171" : "#e2e8f0"; };
  const gradeColor = (g) => {
    if (!g) return "#94a3b8";
    const s = g.toUpperCase();
    if (s === "A+" || s === "A")  return "#4ade80"; // bright green
    if (s === "A-")               return "#86efac"; // light green
    if (s === "B+" || s === "B")  return "#bef264"; // lime
    if (s === "B-")               return "#d9f99d"; // pale lime
    if (s === "C+" || s === "C")  return "#fde047"; // yellow
    if (s === "C-")               return "#fb923c"; // orange
    return "#f87171";             // red for D and below
  };
  const fmtPnl = (n) => { const v = parseFloat(n); if (isNaN(v)) return "-"; return `${v >= 0 ? "+" : "-"}$${Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`; };
  // Format profit factor — handles Infinity (perfect day: 0 losses) gracefully
  const fmtPF = (pf) => { if (pf == null) return "—"; if (!isFinite(pf)) return "∞"; return pf.toFixed(2); };
  const pfColor = (pf) => { if (pf == null) return "#64748b"; if (!isFinite(pf)) return "#4ade80"; return pf >= 1 ? "#4ade80" : "#f87171"; };

  const months = [...new Set(entries.map(e => e.date?.slice(0, 7)))].sort().reverse();
  const filtered = filterMonth ? entries.filter(e => e.date?.startsWith(filterMonth)) : entries;
  const totalPnL = filtered.reduce((s, e) => s + netPnl(e), 0);
  const winDays = filtered.filter(e => netPnl(e) > 0).length;
  const lossDays = filtered.filter(e => netPnl(e) < 0).length;

  // Active journal meta
  const activeJournal = journals.find(j => j.id === activeJournalId) || journals[0];
  const isProp = activeJournal?.type === JOURNAL_TYPES.PROP;
  const isPersonal = !isProp;
  const propStatus = useMemo(() => isProp && activeJournal?.config ? calcPropStatus(entries, activeJournal.config) : null, [entries, activeJournal, isProp]);
  const personalBalance = useMemo(() => isPersonal && activeJournal?.config ? calcPersonalBalance(entries, activeJournal.config) : null, [entries, activeJournal, isPersonal]);
  const analytics = useMemo(() => calcAnalytics(form.parsedTrades, aiCfg?.tzLock !== false), [form.parsedTrades]);
  const detailAnalytics = useMemo(() => activeEntry ? calcAnalytics(activeEntry.parsedTrades || [], aiCfg?.tzLock !== false) : null, [activeEntry]);

  // All prop journals for overview + cumulative analytics
  const propJournals = useMemo(() => journals.filter(j => j.type === JOURNAL_TYPES.PROP), [journals]);
  const [allPropEntriesMap, setAllPropEntriesMap] = useState({}); // { journalId: entries[] }

  // Auto-resize all textareas — JS fallback for browsers without CSS field-sizing support
  useEffect(() => {
    const resize = (e) => {
      const el = e.target;
      if (el.tagName !== 'TEXTAREA' || el.classList.contains('no-autoresize')) return;
      el.style.height = 'auto';
      el.style.height = el.scrollHeight + 'px';
    };
    // Set initial heights for any pre-filled textareas
    const initAll = () => {
      document.querySelectorAll('textarea:not(.no-autoresize)').forEach(el => {
        if (el.value) { el.style.height = 'auto'; el.style.height = el.scrollHeight + 'px'; }
      });
    };
    document.addEventListener('input', resize);
    initAll();
    return () => document.removeEventListener('input', resize);
  }, [view, tab]); // re-init when view/tab changes to catch newly mounted textareas

  // Load all prop journal entries when propDash view is opened
  useEffect(() => {
    if (view !== "propdash") return;
    if (propJournals.length === 0) { setView("list"); return; }
    let cancelled = false;
    (async () => {
      const map = {};
      for (const j of propJournals) {
        if (j.id === activeJournalId) { map[j.id] = entries; continue; }
        try {
          const r = await storage.get(`entries_${j.id}`);
          map[j.id] = r?.value ? JSON.parse(r.value).map(normalizeEntry).map(e => e.entry || e) : [];
        } catch { map[j.id] = []; }
      }
      if (!cancelled) setAllPropEntriesMap(map);
    })();
    return () => { cancelled = true; };
  }, [view, propJournals.length, activeJournalId]); // eslint-disable-line

  // Memoize per-entry analytics for trade list — avoids recalculating on every render
  const entryAnalyticsMap = useMemo(() => {
    const map = {};
    for (const e of entries) { map[e.id] = calcAnalytics(e.parsedTrades || [], true); }
    return map;
  }, [entries]);

  // Global trade analytics across ALL entries
  const globalAnalytics = useMemo(() => {
    const allTrades = entries.flatMap(e => e.parsedTrades || []);
    if (!allTrades.length) return null;
    return calcAnalytics(allTrades, true);
  }, [entries]);

  // Total account balance = starting capital + cumulative net P&L
  const totalAccountBalance = useMemo(() => {
    if (!activeJournal?.config?.startingBalance) return null;
    const cum = entries.reduce((s, e) => s + (parseFloat(e.pnl || 0) - parseFloat(e.commissions || 0)), 0);
    return activeJournal.config.startingBalance + cum;
  }, [entries, activeJournal]);

  // Search & filter state
  const [searchQuery, setSearchQuery] = useState("");
  const [sortNewest, setSortNewest] = useState(true);
  const [listSubTab, setListSubTab] = useState("journal"); // "journal" | "csv"
  const [csvDatePreset, setCsvDatePreset] = useState("all"); // "all"|"today"|"week"|"month"|"custom"
  const [csvDateFrom, setCsvDateFrom] = useState("");
  const [csvDateTo, setCsvDateTo] = useState("");
  const [searchField, setSearchField] = useState("all"); // "all" | "notes" | "instruments" | "grade" | "mistakes"
  const filteredAndSearched = useMemo(() => {
    let result = filtered;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = filtered.filter(e => {
        if (searchField === "all" || searchField === "instruments") {
          const instrs = (e.instruments?.length ? e.instruments : e.instrument ? [e.instrument] : []).join(" ").toLowerCase();
          if (instrs.includes(q)) return true;
        }
        if (searchField === "all" || searchField === "grade") {
          if ((e.grade || "").toLowerCase().includes(q)) return true;
        }
        if (searchField === "all" || searchField === "notes") {
          const noteText = [e.marketNotes, e.lessonsLearned, e.mistakes, e.improvements, e.bestTrade, e.worstTrade, e.rules, e.tomorrow, e.reinforceRule].filter(Boolean).join(" ").toLowerCase();
          if (noteText.includes(q)) return true;
        }
        if (searchField === "all" || searchField === "mistakes") {
          const mText = (e.sessionMistakes || []).join(" ").toLowerCase();
          if (mText.includes(q)) return true;
        }
        if (searchField === "all") {
          if ((e.bias || "").toLowerCase().includes(q)) return true;
          if ((e.date || "").includes(q)) return true;
        }
        return false;
      });
    }
    return sortNewest
      ? [...result].sort((a, b) => b.date.localeCompare(a.date))
      : [...result].sort((a, b) => a.date.localeCompare(b.date));
  }, [filtered, searchQuery, searchField, sortNewest]);

  const TABS = [
    { id: "session", label: "Session" },
    { id: "import", label: "Import Trades" },
    { id: "analysis", label: "Analysis", disabled: !form.parsedTrades?.length },
    { id: "tomorrow", label: "Tomorrow" },
  ];

  // ── Import / Export ──────────────────────────────────────────────
  const [showSettings, setShowSettings] = useState(false);
  const [settingsTab, setSettingsTab] = useState("backup");
  const [importMsg, setImportMsg] = useState(null);
  const importFileRef = useRef(null);
  const csvInputRef = useRef(null);

  const [exportData, setExportData] = useState(null);
  const [copied, setCopied] = useState(false);
  const [exportMeta, setExportMeta] = useState({ title: "EXPORT BACKUP", filename: "trading-journal-backup.json", desc: "Copy all of the text below \u2192 paste into a new file \u2192 save as", isCSV: false });
  // Single-journal export controls
  const [exportJournalId, setExportJournalId] = useState("");
  const [exportMonth, setExportMonth] = useState(""); // "YYYY-MM" or "" for all
  const [csvJournalId, setCsvJournalId] = useState(""); // "" = all journals
  const [csvYear, setCsvYear] = useState("");            // "" = all years
  const [csvMonth, setCsvMonth] = useState("");          // "" = all months in year

  // ── Reliable download helper — works inside sandboxed iframes ──
  const triggerDownload = (content, filename, mime) => {
    try {
      const blob = new Blob([content], { type: mime });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.style.display = "none";
      document.body.appendChild(a);
      a.click();
      setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 1000);
    } catch {
      // Fallback: show copy modal if download is truly blocked
      return false;
    }
    return true;
  };

  const handleExport = async () => {
    try {
      const linksBackup  = (() => { try { const r = localStorage.getItem("tj-links-v1");  return r ? JSON.parse(r) : []; } catch { return []; } })();
      const albumBackup  = (() => { try { const r = localStorage.getItem("tj-album-v1");  return r ? JSON.parse(r) : []; } catch { return []; } })();
      const quotesResult = await storage.get("trader-quotes-v1");
      const quotesBackup = quotesResult?.value ? JSON.parse(quotesResult.value) : [];
      const allData = { version: 2, exportedAt: new Date().toISOString(), journals, entriesByJournal: {}, links: linksBackup, album: albumBackup, quotes: quotesBackup };
      for (const j of journals) {
        try {
          const r = await storage.get(`journal-entries-${j.id}`);
          allData.entriesByJournal[j.id] = r?.value ? JSON.parse(r.value) : [];
        } catch { allData.entriesByJournal[j.id] = []; }
      }
      const json = JSON.stringify(allData, null, 2);
      const ok = triggerDownload(json, 'trading-journal-backup.json', 'application/json');
      if (!ok) {
        setExportMeta({ title: 'FULL BACKUP · ALL JOURNALS', filename: 'trading-journal-backup.json', desc: 'Copy all of the text below → paste into a new file → save as', isCSV: false });
        setExportData(json);
      }
      setShowSettings(false);
    } catch (e) { alert("Export failed: " + e.message); }
  };

  // Export a single journal (optionally filtered to a month)
  const handleExportSingle = async () => {
    try {
      const jId = exportJournalId || activeJournalId;
      const j = journals.find(x => x.id === jId);
      if (!j) { alert("Journal not found."); return; }
      const r = await storage.get(`journal-entries-${jId}`);
      let jEntries = r?.value ? JSON.parse(r.value) : [];
      if (exportMonth) jEntries = jEntries.filter(e => e.date?.startsWith(exportMonth));
      const data = { version: 2, exportedAt: new Date().toISOString(), journal: j, entries: jEntries };
      const suffix = exportMonth ? `-${exportMonth}` : "";
      const filename = `journal-${j.name.replace(/[^a-z0-9]/gi,"_")}${suffix}.json`;
      const json = JSON.stringify(data, null, 2);
      const ok = triggerDownload(json, filename, 'application/json');
      if (!ok) {
        setExportMeta({ title: `EXPORT · ${j.name.toUpperCase()}${exportMonth ? " · " + exportMonth : ""}`, filename, desc: `Copy all of the text below → paste into a new file → save as`, isCSV: false });
        setExportData(json);
      }
      setShowSettings(false);
    } catch(e) { alert("Export failed: " + e.message); }
  };

  // Export CSV — all trades across all journals (or current journal only)
  const handleExportCSV = async () => {
    try {
      const jId = csvJournalId || null;
      const targetJournals = jId ? journals.filter(j => j.id === jId) : journals;
      const rows = [["Date","Journal","Symbol","Direction","Contracts","Entry","Exit","Points","P&L","Commissions","Session","Duration","Grade","Tags","Notes"]];
      // Build date prefix filter: year only, year+month, or none
      const datePrefix = csvYear ? (csvMonth ? `${csvYear}-${csvMonth}` : csvYear) : "";
      for (const j of targetJournals) {
        const r = await storage.get(`journal-entries-${j.id}`);
        let jEntries = r?.value ? JSON.parse(r.value) : [];
        if (datePrefix) jEntries = jEntries.filter(e => e.date?.startsWith(datePrefix));
        for (const e of jEntries) {
          const trades = e.parsedTrades || [];
          if (trades.length > 0) {
            for (const t of trades) {
              rows.push([
                e.date, j.name,
                t.symbol||"", t.direction||"",
                t.contracts||"", t.entry||"", t.exit||"",
                t.points != null ? t.points : "",
                t.pnl != null ? t.pnl.toFixed(2) : "",
                e.commissions||"",
                t.session||"", t.duration||"",
                e.grade||"",
                (e.mistakeTags||[]).join(";"),
                (e.notes||"").replace(/"/g,'""').slice(0,200)
              ]);
            }
          } else {
            // Daily summary row when no parsed trades
            rows.push([
              e.date, j.name,
              "","","","","","",
              e.pnl||"", e.commissions||"",
              "","",
              e.grade||"",
              (e.mistakeTags||[]).join(";"),
              (e.notes||"").replace(/"/g,'""').slice(0,200)
            ]);
          }
        }
      }
      const csv = rows.map(r => r.map(v => `"${String(v).replace(/"/g,'""')}"`).join(",")).join("\n");
      const jLabel = jId ? journals.find(j=>j.id===jId)?.name?.replace(/[^a-z0-9]/gi,"_") || "all-journals" : "all-journals";
      const dateSuffix = csvYear ? (csvMonth ? `-${csvYear}-${csvMonth}` : `-${csvYear}`) : "";
      const filename = `trades-${jLabel}${dateSuffix}.csv`;
      const titleDate = csvYear ? (csvMonth ? ` · ${csvYear}-${csvMonth}` : ` · ${csvYear}`) : "";
      const ok = triggerDownload(csv, filename, 'text/csv');
      if (!ok) {
        setExportMeta({ title: `CSV EXPORT · ${jLabel.toUpperCase().replace(/_/g," ")}${titleDate}`, filename, desc: `Copy all of the text below → paste into a new .csv file → open in Excel`, isCSV: true });
        setExportData(csv);
      }
      setShowSettings(false);
    } catch(e) { alert("CSV export failed: " + e.message); }
  };

  const handleImportFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      if (!data.journals || !data.entriesByJournal) throw new Error("Invalid backup file format.");
      await saveJournalsMeta(data.journals);
      for (const j of data.journals) {
        const jes = data.entriesByJournal[j.id] || [];
        await storage.set(`journal-entries-${j.id}`, JSON.stringify(jes));
      }
      // Restore resource links if present in backup
      if (data.links?.length)  { try { localStorage.setItem("tj-links-v1",  JSON.stringify(data.links));  } catch {} }
      // Restore trading album if present in backup
      if (data.album?.length)  { try { localStorage.setItem("tj-album-v1",  JSON.stringify(data.album));  } catch {} }
      // Restore custom quotes if present in backup
      if (data.quotes?.length) { try { await storage.set("trader-quotes-v1", JSON.stringify(data.quotes)); } catch {} }
      const firstId = data.journals[0]?.id || DEFAULT_JOURNAL_ID;
      setActiveJournalId(firstId);
      setEntries(data.entriesByJournal[firstId] || []);
      setView("list"); setActiveEntry(null); setForm(emptyEntry());
      setImportMsg({ type: "ok", text: `Restored ${data.journals.length} journal${data.journals.length > 1 ? "s" : ""} successfully.` });
      setShowSettings(false);
    } catch (err) {
      setImportMsg({ type: "err", text: "Import failed: " + err.message });
    }
    e.target.value = "";
  };

  // ── Prop dashboard extracted to named component to satisfy React hook rules ──
  // Uses closure to access TradingJournal state without prop-drilling
  // PropDashInner moved to module scope below

  // ─────────────────────────────────────────────────────────────────

  return (
    <>
    <div style={{ fontFamily: "'Space Grotesk','DM Mono',sans-serif", background: "#0a0e1a", minHeight: "100vh", color: "#e2e8f0" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Bebas+Neue&family=Bodoni+Moda:ital,wght@1,900&family=Cinzel:wght@700;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        html,body{width:100%;margin:0;padding:0}*{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:#0a0e1a}::-webkit-scrollbar-thumb{background:#1e3a5f;border-radius:2px}
        textarea,input,select{background:#0f1729!important;color:#e2e8f0!important;border:1px solid #1e3a5f!important;border-radius:4px;padding:10px 12px;font-family:'DM Mono',monospace;font-size:13px;width:100%;outline:none;transition:border-color .2s;resize:none;overflow:hidden;field-sizing:content}
        textarea:focus,input:focus,select:focus{border-color:#3b82f6!important}
        textarea.cal-note{color:#fbbf24!important;background:transparent!important;border:none!important;padding:0!important;field-sizing:content!important;resize:none!important;overflow:hidden!important;min-height:20px}
        textarea.cal-note:focus{color:#fde68a!important;border:none!important;outline:none!important}
        textarea::placeholder,input::placeholder{color:#1e3a5f}
        select option{background:#0f1729}
        textarea.no-autoresize{resize:vertical;overflow:auto;field-sizing:unset}
        .entry-card{transition:all .15s;cursor:pointer}
        
        .entry-card:hover{background:#0f1729!important;border-color:#1e3a5f!important}
        .pill{padding:6px 14px;border-radius:3px;border:1px solid #1e293b;font-size:12px;cursor:pointer;transition:all .15s;color:#94a3b8;background:transparent;letter-spacing:.05em;font-family:'DM Mono',monospace}
        .pill.sel{border-color:#3b82f6;background:#1e3a5f;color:#93c5fd}
        .pill:hover:not(.sel){border-color:#334155;color:#e2e8f0}
        @keyframes qotdFade { from{opacity:0;transform:translateY(-6px)} to{opacity:1;transform:translateY(0)} }
        @keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
        .helper-text { font-family:"DM Mono",monospace; font-size:8px; color:#64748b; letter-spacing:0.25em; margin-top:4px; }
      `}</style>

      {/* Header */}
      <div style={{ borderBottom: "1px solid #1e293b", padding: "0 40px", display: "flex", alignItems: "center", justifyContent: "space-between", position: "sticky", top: 0, background: "#0a0e1a", zIndex: 10, overflow: "visible", height: 80 }}>

        {/* LEFT: C4 signature — AyeOh + rule + TRADING JOURNAL horizontal + tagline */}
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexShrink: 0, overflow: "visible" }}>
          <div onClick={() => { setView("list"); setActiveEntry(null); setListMode("calendar"); }}
            style={{ display: "flex", alignItems: "center", gap: 14, cursor: "pointer", userSelect: "none", overflow: "visible" }}
            onMouseEnter={e => e.currentTarget.querySelector(".ayeoh-sig").style.opacity = "0.75"}
            onMouseLeave={e => e.currentTarget.querySelector(".ayeoh-sig").style.opacity = "1"}>
            <div className="ayeoh-sig" style={{ fontFamily: "'Cinzel',serif", fontSize: 32, fontWeight: 900, fontStyle: "normal", letterSpacing: "0.12em", lineHeight: 1, background: "linear-gradient(135deg,#38bdf8 0%,#818cf8 55%,#c084fc 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text", transition: "opacity .15s", flexShrink: 0 }}>AYEOH</div>
            <div style={{ width: 1, height: 40, background: "linear-gradient(180deg,transparent,#334155,transparent)", flexShrink: 0 }} />
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <div style={{ fontFamily: "'DM Mono',monospace", fontSize: 12, color: "#38bdf8", letterSpacing: "0.3em", whiteSpace: "nowrap", opacity: 0.9 }}>TRADING JOURNAL</div>
              <div style={{ height: 1.5, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc,transparent)", borderRadius: 1 }} />
              <div style={{ fontFamily: "'DM Mono',monospace", fontSize: 8, letterSpacing: "0.22em", whiteSpace: "nowrap", background: "linear-gradient(135deg,rgba(56,189,248,0.6),rgba(129,140,248,0.6),rgba(192,132,252,0.6))", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>CAPTURE · ANALYSE · GROW</div>
            </div>
          </div>
          {/* Settings — unobtrusive */}
          <button
            onClick={() => { setShowSettings(true); setSettingsTab("backup"); setImportMsg(null); }}
            title="Settings"
            style={{ background: "transparent", border: "none", color: "#818cf8", fontSize: 20, cursor: "pointer", padding: "4px 6px", lineHeight: 1, fontFamily: "inherit", transition: "color .15s", flexShrink: 0 }}
            onMouseEnter={e => e.currentTarget.style.color = "#c084fc"}
            onMouseLeave={e => e.currentTarget.style.color = "#818cf8"}>
            ⚙
          </button>
        </div>

        {/* CENTRE: Market structure SVG art */}
        <div style={{ flex: 1, alignSelf: "stretch", display: "flex", alignItems: "stretch", padding: "0 16px", overflow: "hidden", minWidth: 0 }}>
          <svg width="100%" height="100%" viewBox="0 0 500 64" preserveAspectRatio="none" style={{ display: "block", width: "100%", height: "100%" }}>
            <defs>
              <linearGradient id="navLine" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#38bdf8" stopOpacity="0"/>
                <stop offset="10%" stopColor="#38bdf8" stopOpacity="1"/>
                <stop offset="42%" stopColor="#818cf8" stopOpacity="1"/>
                <stop offset="58%" stopColor="#818cf8" stopOpacity="1"/>
                <stop offset="90%" stopColor="#c084fc" stopOpacity="1"/>
                <stop offset="100%" stopColor="#c084fc" stopOpacity="0"/>
              </linearGradient>
              <linearGradient id="navFill" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#818cf8" stopOpacity="0.16"/>
                <stop offset="100%" stopColor="#818cf8" stopOpacity="0"/>
              </linearGradient>
            </defs>
            {/* Area fill — bottom-anchored polygon */}
            <polygon points="15,62 55,44 85,56 130,28 165,40 210,12 248,26 285,2 325,18 360,34 392,42 435,24 490,36 490,64 15,64" fill="url(#navFill)"/>
            {/* Main price path — peaks at y=2, troughs at y=62 */}
            <polyline points="15,62 55,44 85,56 130,28 165,40 210,12 248,26 285,2 325,18 360,34 392,42 435,24 490,36" fill="none" stroke="url(#navLine)" strokeWidth="3.5" strokeLinejoin="round" strokeLinecap="round"/>
            {/* Support dashed */}
            <line x1="50" y1="54" x2="500" y2="54" stroke="#38bdf8" strokeWidth="1" strokeDasharray="7 5" opacity="0.4"/>
            {/* Resistance dashed */}
            <line x1="0" y1="10" x2="400" y2="10" stroke="#c084fc" strokeWidth="1" strokeDasharray="7 5" opacity="0.4"/>
            {/* Labels */}


            {/* Peak marker */}
            <circle cx="285" cy="2" r="5" fill="#818cf8" opacity="1"/>
            <circle cx="285" cy="2" r="11" fill="none" stroke="#818cf8" strokeWidth="1.5" opacity="0.4"/>
            <circle cx="285" cy="2" r="17" fill="none" stroke="#818cf8" strokeWidth="0.6" opacity="0.15"/>
          </svg>
        </div>

        {/* RIGHT: Nav buttons */}
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexShrink: 0 }}>
          {view === "list" && propJournals.length > 0 && <button onClick={() => { setPropDashTab("overview"); setView("propdash"); }} style={{ background: "rgba(120,53,15,0.15)", color: "#f59e0b", border: "1px solid rgba(245,158,11,0.35)", padding: "12px 24px", borderRadius: 6, fontFamily: "inherit", fontSize: 14, cursor: "pointer", letterSpacing: "0.06em", fontWeight: 600 }}>🏆 PROP DASHBOARD</button>}
          {(view === "list" || view === "recap") && (
            <span style={{ display:"inline-block", padding:1, borderRadius:7, background: view === "recap" ? "linear-gradient(135deg,#7c3aed,#a855f7,#818cf8)" : "linear-gradient(135deg,#1e1b4b,#2e1065)", flexShrink:0 }}>
              <button onClick={() => setView("recap")} style={{ display:"block", background: view === "recap" ? "#0d0a1a" : "#070d1a", color: view === "recap" ? "#c4b5fd" : "#475569", border:"none", padding:"11px 24px", borderRadius:6, fontFamily:"inherit", fontSize:14, cursor:"pointer", letterSpacing:"0.06em", fontWeight:700, whiteSpace:"nowrap" }}>🤖 AI RECAP</button>
            </span>
          )}
          {view === "list" && (
            <span style={{ display:"inline-block", padding:1, borderRadius:7, background:"linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", flexShrink:0 }}>
              <button onClick={openNew} style={{ display:"block", background:"#070d1a", color:"#e2e8f0", border:"none", padding:"11px 24px", borderRadius:6, fontFamily:"inherit", fontSize:14, cursor:"pointer", letterSpacing:"0.06em", fontWeight:700, whiteSpace:"nowrap" }}>+ NEW ENTRY</button>
            </span>
          )}
          {view === "recap" && <button onClick={() => setView("list")} style={{ background: "transparent", border: "1px solid rgba(129,140,248,0.35)", padding: "12px 24px", borderRadius: 6, fontFamily: "inherit", fontSize: 14, cursor: "pointer", letterSpacing: "0.06em", fontWeight: 600, background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>← BACK</button>}
          {view === "quotes" && <button onClick={() => setView("list")} style={{ background: "transparent", border: "1px solid rgba(129,140,248,0.35)", padding: "12px 24px", borderRadius: 6, fontFamily: "inherit", fontSize: 14, cursor: "pointer", letterSpacing: "0.06em", fontWeight: 600, background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>← BACK</button>}
          {view === "reference" && <button onClick={() => setView("list")} style={{ background: "transparent", border: "1px solid rgba(129,140,248,0.35)", padding: "12px 24px", borderRadius: 6, fontFamily: "inherit", fontSize: 14, cursor: "pointer", letterSpacing: "0.06em", fontWeight: 600, background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>← BACK</button>}
          {view === "propdash" && <button onClick={() => setView("list")} style={{ background: "rgba(120,53,15,0.1)", color: "#f59e0b", border: "1px solid rgba(245,158,11,0.25)", padding: "12px 24px", borderRadius: 6, fontFamily: "inherit", fontSize: 14, cursor: "pointer", letterSpacing: "0.06em", fontWeight: 600 }}>← BACK</button>}
          {view === "new" && <>
            <button onClick={() => { setView("list"); setActiveEntry(null); }} style={{ background: "transparent", border: "1px solid rgba(129,140,248,0.35)", padding: "12px 24px", borderRadius: 6, fontFamily: "inherit", fontSize: 14, cursor: "pointer", letterSpacing: "0.06em", fontWeight: 600, background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>CANCEL</button>
            <button onClick={handleSave} disabled={saving} style={{ background: "#0a0e1a", color: "#4ade80", border: "1px solid rgba(74,222,128,0.4)", padding: "12px 24px", borderRadius: 6, fontFamily: "inherit", fontSize: 14, cursor: "pointer", letterSpacing: "0.06em", fontWeight: 700, boxShadow: "0 0 0 1px rgba(74,222,128,0.15)" }}>{saving ? "SAVING..." : saved ? "✓ SAVED" : activeEntry ? "✓ UPDATE" : "+ APPLY"}</button>
          </>}
          {view === "detail" && <>
            <button onClick={() => { setView("list"); setConfirmDeleteEntry(false); }} style={{ background: "transparent", border: "1px solid rgba(129,140,248,0.35)", padding: "11px 20px", borderRadius: 6, fontFamily: "inherit", fontSize: 13, cursor: "pointer", letterSpacing: "0.06em", fontWeight: 600, background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>← BACK</button>
            <button onClick={() => { openEdit(activeEntry); setConfirmDeleteEntry(false); }} style={{ background: "rgba(10,18,32,0.5)", color: "#64748b", border: "1px solid #1e293b", padding: "12px 24px", borderRadius: 6, fontFamily: "inherit", fontSize: 14, cursor: "pointer", letterSpacing: "0.06em", fontWeight: 600 }}>EDIT</button>
            {confirmDeleteEntry ? (
              <span style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <span style={{ fontSize: 11, color: "#f87171", letterSpacing: "0.06em" }}>Delete this entry?</span>
                <button onClick={() => { handleDelete(activeEntry.id); setConfirmDeleteEntry(false); }}
                  style={{ background: "rgba(127,29,29,0.25)", border: "1px solid rgba(248,113,113,0.35)", color: "#f87171", padding: "12px 20px", borderRadius: 6, fontFamily: "inherit", fontSize: 13, cursor: "pointer", letterSpacing: "0.06em", fontWeight: 700 }}>YES DELETE</button>
                <button onClick={() => setConfirmDeleteEntry(false)}
                  style={{ background: "transparent", border: "1px solid #1e293b", color: "#475569", padding: "12px 20px", borderRadius: 6, fontFamily: "inherit", fontSize: 13, cursor: "pointer", letterSpacing: "0.06em", fontWeight: 600 }}>CANCEL</button>
              </span>
            ) : (
              <button onClick={() => setConfirmDeleteEntry(true)} style={{ background: "transparent", color: "#f87171", border: "1px solid rgba(248,113,113,0.25)", padding: "12px 22px", borderRadius: 6, fontFamily: "inherit", fontSize: 14, cursor: "pointer", letterSpacing: "0.06em", fontWeight: 600 }}>DELETE</button>
            )}
          </>}
        </div>
      </div>

      {/* Export modal */}
      {exportData && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.8)", zIndex: 200, display: "flex", alignItems: "center", justifyContent: "center", padding: 24 }}>
          <div style={{ background: "#0f1729", border: "1px solid #1e3a5f", borderRadius: 8, width: "100%", maxWidth: 680, maxHeight: "80vh", display: "flex", flexDirection: "column" }}>
            <div style={{ padding: "16px 20px", borderBottom: "1px solid #1e293b", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <div style={{ fontSize: 13, color: "#93c5fd", fontWeight: 600, letterSpacing: "0.08em" }}>{exportMeta.title}</div>
                <div style={{ fontSize: 10, color: "#64748b", marginTop: 3 }}>{exportMeta.desc} <span style={{ color: "#94a3b8" }}>{exportMeta.filename}</span></div>
              </div>
              <button onClick={() => setExportData(null)} style={{ background: "transparent", border: "1px solid #1e293b", color: "#64748b", padding: "5px 12px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer" }}>✕ CLOSE</button>
            </div>
            <div style={{ padding: "12px 16px", borderBottom: "1px solid #1e293b", display: "flex", gap: 8, alignItems: "center" }}>
              <button
                onClick={() => { navigator.clipboard?.writeText(exportData).then(() => { setCopied(true); setTimeout(() => setCopied(false), 2000); }).catch(() => {}); }}
                
                style={{ background: copied ? "#166534" : "#1d4ed8", color: "white", border: "none", padding: "8px 20px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.06em", transition: "background .2s" }}>
                {copied ? "✓ COPIED!" : "COPY ALL"}
              </button>
              <div style={{ fontSize: 10, color: "#64748b" }}>
                {exportData.length.toLocaleString()} characters · save contents as <strong style={{ color: "#94a3b8" }}>{exportMeta.filename}</strong>
              </div>
            </div>
            <textarea
              readOnly
              value={exportData}
              onClick={e => e.target.select()}
              style={{ flex: 1, margin: 0, resize: "none", fontFamily: "monospace", fontSize: 11, background: "#060810", border: "none", borderRadius: "0 0 8px 8px", color: "#64748b", padding: "14px 16px", outline: "none", overflowY: "auto" }}
            />
          </div>
        </div>
      )}

      {/* Data Health strip */}
      {dataHealthReport?.changed && dataHealthReport.summary && (
        <div style={{ margin: '12px 24px 0', background: '#0a1628', border: '1px solid #1e3a5f', borderRadius: 6, padding: '10px 12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10 }}>
          <div style={{ fontSize: 12, color: '#93c5fd', lineHeight: 1.5 }}>🧪 {dataHealthReport.summary}</div>
          <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
            <button onClick={() => setShowDataHealthDetails(true)} style={{ background: 'transparent', border: '1px solid #1e293b', color: '#93c5fd', padding: '6px 10px', borderRadius: 4, fontFamily: 'inherit', fontSize: 11, cursor: 'pointer', letterSpacing: '0.05em' }}>VIEW</button>
            <button onClick={() => setDataHealthReport(null)} style={{ background: 'transparent', border: '1px solid #1e293b', color: '#64748b', padding: '6px 10px', borderRadius: 4, fontFamily: 'inherit', fontSize: 11, cursor: 'pointer' }}>DISMISS</button>
          </div>
        </div>
      )}

      {/* Data Health details modal */}
      {showDataHealthDetails && dataHealthReport && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.8)', zIndex: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24 }}>
          <div style={{ background: '#0f1729', border: '1px solid #1e3a5f', borderRadius: 8, width: '100%', maxWidth: 720, maxHeight: '80vh', display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '16px 20px', borderBottom: '1px solid #1e293b', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: 13, color: '#93c5fd', fontWeight: 600, letterSpacing: '0.08em' }}>DATA HEALTH</div>
                <div style={{ fontSize: 10, color: '#64748b', marginTop: 3 }}>What was auto-fixed so your stats stay accurate.</div>
              </div>
              <button onClick={() => setShowDataHealthDetails(false)} style={{ background: 'transparent', border: '1px solid #1e293b', color: '#64748b', padding: '5px 12px', borderRadius: 4, fontFamily: 'inherit', fontSize: 11, cursor: 'pointer' }}>✕ CLOSE</button>
            </div>
            <div style={{ padding: '14px 18px', overflowY: 'auto' }}>
              {(dataHealthReport.details || []).length ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {dataHealthReport.details.map((d, i) => (
                    <div key={i} style={{ background: '#0a0e1a', border: '1px solid #1e293b', borderRadius: 6, padding: '10px 12px', fontSize: 12, color: '#94a3b8', lineHeight: 1.6 }}>{d}</div>
                  ))}
                </div>
              ) : (
                <div style={{ fontSize: 12, color: '#64748b' }}>No details.</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── SETTINGS MODAL ─────────────────────────────────────────── */}
      {showSettings && (
        <div onClick={() => setShowSettings(false)} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.75)', zIndex: 230, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24 }}>
          <div onClick={e => e.stopPropagation()} style={{ background: '#0a0f1e', border: '1px solid #1e3a5f', borderRadius: 10, width: '100%', maxWidth: 860, maxHeight: '88vh', display: 'flex', overflow: 'hidden', boxShadow: '0 24px 64px rgba(0,0,0,0.7)' }}>

            {/* ── LEFT SIDEBAR NAV ── */}
            <div style={{ width: 192, background: '#070d1a', borderRight: '1px solid #0f1729', display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
              <div style={{ padding: '20px 18px 14px', borderBottom: '1px solid #0f1729' }}>
                <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 16, color: '#93c5fd', letterSpacing: '0.14em' }}>SETTINGS</div>
                <div style={{ fontSize: 9, color: '#64748b', letterSpacing: '0.1em', marginTop: 3 }}>TRADING JOURNAL</div>
              </div>
              <nav style={{ flex: 1, padding: '10px 8px', display: 'flex', flexDirection: 'column', gap: 2 }}>
                {[
                  { id: 'backup',  icon: '💾', label: 'Backup & Restore' },
                  { id: 'ai',      icon: '🤖', label: 'AI & API Key' },
                  { id: 'general', icon: '⚙',  label: 'General' },
                ].map(({ id, icon, label }) => (
                  <button key={id} onClick={() => setSettingsTab(id)}
                    style={{ display: 'flex', alignItems: 'center', gap: 10, width: '100%', padding: '9px 12px', borderRadius: 6, background: settingsTab === id ? '#0f1a2e' : 'transparent', border: `1px solid ${settingsTab === id ? '#1e3a5f' : 'transparent'}`, color: settingsTab === id ? '#93c5fd' : '#64748b', fontFamily: 'inherit', fontSize: 12, cursor: 'pointer', textAlign: 'left', letterSpacing: '0.03em', transition: 'all .12s' }}
                    onMouseEnter={e => { if (settingsTab !== id) { e.currentTarget.style.background = '#0a1220'; e.currentTarget.style.color = '#94a3b8'; }}}
                    onMouseLeave={e => { if (settingsTab !== id) { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = '#64748b'; }}}>
                    <span style={{ fontSize: 14, opacity: settingsTab === id ? 1 : 0.6 }}>{icon}</span>
                    {label}
                  </button>
                ))}
              </nav>
              <div style={{ padding: '12px 8px', borderTop: '1px solid #0f1729' }}>
                <button onClick={() => setShowSettings(false)}
                  style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%', padding: '8px 12px', borderRadius: 6, background: 'transparent', border: '1px solid #0f1729', color: '#64748b', fontFamily: 'inherit', fontSize: 11, cursor: 'pointer', letterSpacing: '0.04em', transition: 'all .12s' }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = '#1e293b'; e.currentTarget.style.color = '#64748b'; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = '#0f1729'; e.currentTarget.style.color = '#334155'; }}>
                  ✕ Close
                </button>
              </div>
            </div>

            {/* ── RIGHT CONTENT PANE ── */}
            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>

              {/* ── BACKUP & RESTORE ── */}
              {settingsTab === 'backup' && (
                <div style={{ padding: '28px 28px', display: 'flex', flexDirection: 'column', gap: 20 }}>
                  <div>
                    <div style={{ fontSize: 16, color: '#e2e8f0', fontWeight: 600, letterSpacing: '0.06em', marginBottom: 4 }}>Backup & Restore</div>
                    <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.6 }}>Export all journals to a single JSON file, or restore from a previous backup.</div>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>

                    {/* ── Export All ── */}
                    <div style={{ background: '#070d1a', border: '1px solid #1e293b', borderRadius: 8, padding: '18px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
                      <div>
                        <div style={{ fontSize: 12, color: '#e2e8f0', fontWeight: 500, marginBottom: 4 }}>Export All Journals (Full Backup)</div>
                        <div style={{ fontSize: 10, color: '#64748b', lineHeight: 1.5 }}>Downloads all {journals.length} journal{journals.length !== 1 ? 's' : ''} and entries as a .json backup file.</div>
                      </div>
                      <button onClick={handleExport}
                        style={{ background: '#1d4ed8', color: 'white', border: 'none', padding: '9px 20px', borderRadius: 6, fontFamily: 'inherit', fontSize: 12, cursor: 'pointer', letterSpacing: '0.06em', whiteSpace: 'nowrap', flexShrink: 0 }}>
                        ↓ Export (.json)
                      </button>
                    </div>

                    {/* ── Export Single Journal ── */}
                    <div style={{ background: '#070d1a', border: '1px solid #1e293b', borderRadius: 8, padding: '18px 20px', display: 'flex', flexDirection: 'column', gap: 12 }}>
                      <div>
                        <div style={{ fontSize: 12, color: '#e2e8f0', fontWeight: 500, marginBottom: 4 }}>Export Single Journal</div>
                        <div style={{ fontSize: 10, color: '#64748b', lineHeight: 1.5 }}>Export one journal, optionally filtered to a specific month.</div>
                      </div>
                      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                        <select value={exportJournalId} onChange={e => setExportJournalId(e.target.value)}
                          style={{ flex: 1, minWidth: 140, padding: '8px 10px', background: '#060810', border: '1px solid #1e293b', borderRadius: 5, color: '#e2e8f0', fontFamily: 'inherit', fontSize: 11 }}>
                          <option value="">— Active journal —</option>
                          {journals.map(j => <option key={j.id} value={j.id}>{j.name}</option>)}
                        </select>
                        <input type="month" value={exportMonth} onChange={e => setExportMonth(e.target.value)}
                          style={{ padding: '8px 10px', background: '#060810', border: '1px solid #1e293b', borderRadius: 5, color: '#e2e8f0', fontFamily: 'inherit', fontSize: 11 }} />
                        <button onClick={handleExportSingle}
                          style={{ background: '#0c2a4a', color: '#93c5fd', border: '1px solid #1e3a5f', padding: '9px 18px', borderRadius: 6, fontFamily: 'inherit', fontSize: 12, cursor: 'pointer', letterSpacing: '0.06em', whiteSpace: 'nowrap' }}>
                          ↓ Export (.json)
                        </button>
                      </div>
                    </div>

                    {/* ── Export CSV ── */}
                    <div style={{ background: '#070d1a', border: '1px solid #1e293b', borderRadius: 8, padding: '18px 20px', display: 'flex', flexDirection: 'column', gap: 14 }}>
                      <div>
                        <div style={{ fontSize: 12, color: '#e2e8f0', fontWeight: 500, marginBottom: 4 }}>Export CSV (Excel / Sheets Compatible)</div>
                        <div style={{ fontSize: 10, color: '#64748b', lineHeight: 1.5 }}>Filter by journal, year, and/or month — then download as a spreadsheet. Columns: Date, Journal, Symbol, Direction, Contracts, Entry, Exit, Points, P&amp;L, Commissions, Session, Grade, Tags, Notes.</div>
                      </div>

                      {/* Filter row */}
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                        {/* Journal */}
                        <div>
                          <div style={{ fontSize: 9, color: '#475569', letterSpacing: '0.1em', marginBottom: 5 }}>JOURNAL</div>
                          <select value={csvJournalId} onChange={e => setCsvJournalId(e.target.value)}
                            style={{ width: '100%', padding: '8px 10px', background: '#060810', border: `1px solid ${csvJournalId ? '#3b82f6' : '#1e293b'}`, borderRadius: 5, color: csvJournalId ? '#e2e8f0' : '#64748b', fontFamily: 'inherit', fontSize: 11 }}>
                            <option value="">All journals</option>
                            {journals.map(j => <option key={j.id} value={j.id}>{j.name}</option>)}
                          </select>
                        </div>

                        {/* Year */}
                        <div>
                          <div style={{ fontSize: 9, color: '#475569', letterSpacing: '0.1em', marginBottom: 5 }}>YEAR</div>
                          <select value={csvYear} onChange={e => { setCsvYear(e.target.value); setCsvMonth(''); }}
                            style={{ width: '100%', padding: '8px 10px', background: '#060810', border: `1px solid ${csvYear ? '#3b82f6' : '#1e293b'}`, borderRadius: 5, color: csvYear ? '#e2e8f0' : '#64748b', fontFamily: 'inherit', fontSize: 11 }}>
                            <option value="">All years</option>
                            {(() => {
                              // Derive years that actually have data across all journals
                              const yearsSet = new Set();
                              for (const j of journals) {
                                const jEntries = j.id === activeJournalId ? entries : (allPropEntriesMap?.[j.id] || []);
                                jEntries.forEach(e => { if (e.date) yearsSet.add(e.date.slice(0, 4)); });
                              }
                              return [...yearsSet].sort((a, b) => b - a).map(y => (
                                <option key={y} value={y}>{y}</option>
                              ));
                            })()}
                          </select>
                        </div>

                        {/* Month — only active when a year is selected */}
                        <div>
                          <div style={{ fontSize: 9, color: csvYear ? '#475569' : '#334155', letterSpacing: '0.1em', marginBottom: 5 }}>MONTH</div>
                          <select value={csvMonth} onChange={e => setCsvMonth(e.target.value)} disabled={!csvYear}
                            style={{ width: '100%', padding: '8px 10px', background: '#060810', border: `1px solid ${csvMonth ? '#3b82f6' : '#1e293b'}`, borderRadius: 5, color: csvMonth ? '#e2e8f0' : '#64748b', fontFamily: 'inherit', fontSize: 11, opacity: csvYear ? 1 : 0.4, cursor: csvYear ? 'pointer' : 'default' }}>
                            <option value="">All months</option>
                            {['01','02','03','04','05','06','07','08','09','10','11','12'].map((m, i) => (
                              <option key={m} value={m}>{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][i]} ({m})</option>
                            ))}
                          </select>
                        </div>
                      </div>

                      {/* Preview label + download button */}
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
                        <div style={{ fontSize: 10, color: '#475569', fontStyle: 'italic' }}>
                          {(() => {
                            const jName = csvJournalId ? journals.find(j => j.id === csvJournalId)?.name || 'Selected journal' : 'All journals';
                            const dateLabel = csvYear ? (csvMonth ? `${['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][parseInt(csvMonth)-1]} ${csvYear}` : csvYear) : 'all time';
                            return `Will export: ${jName} · ${dateLabel}`;
                          })()}
                        </div>
                        <button onClick={handleExportCSV}
                          style={{ background: '#052e16', color: '#4ade80', border: '1px solid #166534', padding: '9px 20px', borderRadius: 6, fontFamily: 'inherit', fontSize: 12, cursor: 'pointer', letterSpacing: '0.06em', whiteSpace: 'nowrap' }}>
                          ↓ Download .csv
                        </button>
                      </div>
                    </div>

                    {/* ── Import ── */}
                    <div style={{ background: '#070d1a', border: '1px solid #1e293b', borderRadius: 8, padding: '18px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
                      <div>
                        <div style={{ fontSize: 12, color: '#e2e8f0', fontWeight: 500, marginBottom: 4 }}>Restore from Backup</div>
                        <div style={{ fontSize: 10, color: '#f87171', lineHeight: 1.5 }}>⚠ Replaces all current data. Make sure to export first.</div>
                      </div>
                      <button onClick={() => importFileRef.current?.click()}
                        style={{ background: 'transparent', color: '#94a3b8', border: '1px solid #1e293b', padding: '9px 20px', borderRadius: 6, fontFamily: 'inherit', fontSize: 12, cursor: 'pointer', letterSpacing: '0.06em', whiteSpace: 'nowrap', flexShrink: 0 }}>
                        ↑ Restore (.json)
                      </button>
                    </div>
                    <input ref={importFileRef} type="file" accept=".json" onChange={handleImportFile} style={{ display: 'none' }} />
                  </div>

                  {importMsg && (
                    <div style={{ padding: '11px 14px', borderRadius: 6, fontSize: 12, background: importMsg.type === 'ok' ? '#052e16' : '#450a0a', color: importMsg.type === 'ok' ? '#4ade80' : '#f87171', border: `1px solid ${importMsg.type === 'ok' ? '#166534' : '#7f1d1d'}` }}>
                      {importMsg.type === 'ok' ? '✓ ' : '⚠ '}{importMsg.text}
                    </div>
                  )}

                  <div style={{ background: '#060b18', border: '1px solid #0f1729', borderRadius: 6, padding: '12px 16px' }}>
                    <div style={{ fontSize: 9, color: '#64748b', letterSpacing: '0.12em', marginBottom: 8 }}>STORAGE INFO</div>
                    <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.8 }}>
                      {journals.length} journal{journals.length !== 1 ? 's' : ''} · Data stored in your browser locally<br />
                      AI settings are stored separately in <span style={{ color: '#64748b' }}>localStorage</span> and not included in exports.
                    </div>
                  </div>
                </div>
              )}

              {/* ── AI & API KEY ── */}
              {settingsTab === 'ai' && (() => {
                const activeAdapter = getProviderAdapter(aiCfg.provider);
                return (
                  <div style={{ padding: '28px 28px', display: 'flex', flexDirection: 'column', gap: 20 }}>
                    <div>
                      <div style={{ fontSize: 16, color: '#e2e8f0', fontWeight: 600, letterSpacing: '0.06em', marginBottom: 4 }}>AI & API Key</div>
                      <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.6 }}>Stored locally in your browser only. Never included in exports or backups.</div>
                    </div>

                    {/* Enable toggle */}
                    <div style={{ background: '#070d1a', border: '1px solid #1e293b', borderRadius: 8, padding: '16px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16 }}>
                      <div>
                        <div style={{ fontSize: 12, color: '#e2e8f0', fontWeight: 500, marginBottom: 3 }}>Enable AI Features</div>
                        <div style={{ fontSize: 10, color: '#64748b', lineHeight: 1.5 }}>Top findings, note rewrites, daily analysis, weekly & monthly recap.</div>
                      </div>
                      <button onClick={() => { const next = { ...aiCfg, enabled: !aiCfg.enabled }; setAiCfg(next); saveAiSettings(next); }}
                        style={{ background: aiCfg.enabled ? '#052e16' : '#450a0a', border: `1px solid ${aiCfg.enabled ? '#166534' : '#7f1d1d'}`, color: aiCfg.enabled ? '#4ade80' : '#f87171', padding: '8px 18px', borderRadius: 6, fontFamily: 'inherit', fontSize: 12, cursor: 'pointer', fontWeight: 600, letterSpacing: '0.08em', flexShrink: 0 }}>
                        {aiCfg.enabled ? '● ON' : '○ OFF'}
                      </button>
                    </div>

                    {/* Provider selector */}
                    <div>
                      <div style={{ fontSize: 10, color: '#94a3b8', letterSpacing: '0.1em', marginBottom: 8, textTransform: 'uppercase' }}>Provider</div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
                        {AI_PROVIDER_REGISTRY.map(p => {
                          const isActive = aiCfg.provider === p.id;
                          return (
                            <button key={p.id} type="button"
                              onClick={() => setAiCfg(c => ({ ...c, provider: p.id, model: p.defaultModel }))}
                              style={{ padding: '10px 14px', borderRadius: 6, fontFamily: 'inherit', fontSize: 11, cursor: 'pointer', textAlign: 'left', letterSpacing: '0.04em', transition: 'all .15s', background: isActive ? '#0f1a2e' : '#060810', border: `1px solid ${isActive ? '#3b82f6' : '#0f1729'}`, color: isActive ? '#93c5fd' : '#64748b' }}>
                              <div style={{ fontWeight: isActive ? 600 : 400 }}>{p.label}</div>
                              <div style={{ fontSize: 9, color: isActive ? '#475569' : '#334155', marginTop: 2 }}>{p.keyHint}</div>
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    {/* Model selector — preset pills + custom input */}
                    <div>
                      <div style={{ fontSize: 10, color: '#94a3b8', letterSpacing: '0.1em', marginBottom: 8, textTransform: 'uppercase' }}>Model</div>
                      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
                        {activeAdapter.models.map(m => {
                          const isSel = aiCfg.model === m.id;
                          return (
                            <button key={m.id} type="button"
                              onClick={() => setAiCfg(c => ({ ...c, model: m.id }))}
                              style={{ padding: '6px 14px', borderRadius: 4, fontFamily: 'inherit', fontSize: 11, cursor: 'pointer', transition: 'all .12s', background: isSel ? '#1e3a5f' : 'transparent', border: `1px solid ${isSel ? '#3b82f6' : '#1e293b'}`, color: isSel ? '#93c5fd' : '#64748b', letterSpacing: '0.04em' }}>
                              {m.label}
                            </button>
                          );
                        })}
                      </div>
                      <input value={aiCfg.model} onChange={e => setAiCfg(c => ({ ...c, model: e.target.value }))}
                        placeholder={activeAdapter.defaultModel}
                        style={{ width: '100%', boxSizing: 'border-box', padding: '9px 12px', background: '#060810', border: '1px solid #1e293b', borderRadius: 6, color: '#94a3b8', fontFamily: 'inherit', fontSize: 11 }} />
                      <div style={{ fontSize: 9, color: '#64748b', marginTop: 4 }}>Or type any custom model string above</div>
                    </div>

                    {/* API Key — label and placeholder adapt to selected provider */}
                    <div>
                      <div style={{ fontSize: 10, color: '#94a3b8', letterSpacing: '0.1em', marginBottom: 8, textTransform: 'uppercase' }}>{activeAdapter.label} API Key</div>
                      <input type="password" value={aiCfg.apiKey} onChange={e => setAiCfg(c => ({ ...c, apiKey: e.target.value }))}
                        placeholder={activeAdapter.keyPlaceholder}
                        style={{ width: '100%', boxSizing: 'border-box', padding: '11px 14px', background: '#060810', border: '1px solid #1e293b', borderRadius: 6, color: '#e2e8f0', fontFamily: 'inherit', fontSize: 13 }} />
                      <div style={{ fontSize: 10, color: '#64748b', marginTop: 8, lineHeight: 1.6 }}>
                        🔒 Stays in your browser only. API calls go directly from your browser to {activeAdapter.label} — key never touches any server.
                        <br />⚠ Keep out of screen recordings and screenshots.
                      </div>
                    </div>

                    {aiTestStatus && (
                      <div style={{ padding: '10px 14px', borderRadius: 6, fontSize: 12, background: aiTestStatus.type === 'ok' ? '#052e16' : '#450a0a', color: aiTestStatus.type === 'ok' ? '#4ade80' : '#f87171', border: `1px solid ${aiTestStatus.type === 'ok' ? '#166534' : '#7f1d1d'}` }}>
                        {aiTestStatus.text}
                      </div>
                    )}

                    {/* Actions */}
                    <div style={{ display: 'flex', gap: 10 }}>
                      <button onClick={async () => {
                        try { saveAiSettings({ ...aiCfg }); setAiCfg(loadAiSettings()); setAiTestStatus({ type: 'ok', text: '✓ Settings saved.' }); }
                        catch (e) { setAiTestStatus({ type: 'err', text: e.message || 'Save failed.' }); }
                      }} style={{ background: '#1d4ed8', color: 'white', border: 'none', padding: '10px 22px', borderRadius: 6, fontFamily: 'inherit', fontSize: 12, cursor: 'pointer', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                        SAVE
                      </button>
                      <button onClick={async () => {
                        try {
                          saveAiSettings({ ...aiCfg }); setAiCfg({ ...aiCfg });
                          setAiTestStatus({ type: 'ok', text: `Testing ${activeAdapter.label}…` });
                          const txt = await aiRequestText({ ...aiCfg }, { max_tokens: 8, timeoutMs: 8000, messages: [{ role: 'user', content: 'Reply with OK.' }] });
                          setAiTestStatus({ type: 'ok', text: `✓ ${activeAdapter.label} connected — ${txt.slice(0, 50)}` });
                        } catch (e) { setAiTestStatus({ type: 'err', text: e.message || 'Test failed.' }); }
                      }} style={{ background: 'transparent', color: '#93c5fd', border: '1px solid #1e3a5f', padding: '10px 22px', borderRadius: 6, fontFamily: 'inherit', fontSize: 12, cursor: 'pointer', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                        TEST CONNECTION
                      </button>
                    </div>
                  </div>
                );
              })()}

              {/* ── GENERAL ── */}
              {settingsTab === 'general' && (
                <div style={{ padding: '28px 28px', display: 'flex', flexDirection: 'column', gap: 20 }}>
                  <div>
                    <div style={{ fontSize: 16, color: '#e2e8f0', fontWeight: 600, letterSpacing: '0.06em', marginBottom: 4 }}>General</div>
                    <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.6 }}>App-wide preferences. Stored locally alongside AI settings.</div>
                  </div>

                  {/* Broker Preset */}
                  <div style={{ background: '#070d1a', border: '1px solid #1e293b', borderRadius: 8, padding: '20px' }}>
                    <div style={{ fontSize: 9, color: '#94a3b8', letterSpacing: '0.12em', marginBottom: 6, textTransform: 'uppercase' }}>🏦 BROKER PRESET</div>
                    <div style={{ fontSize: 11, color: '#64748b', marginBottom: 12, lineHeight: 1.6 }}>
                      Tells the AI parser which column headers and date format to expect from your export, improving trade parse accuracy.
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
                      {Object.entries(BROKER_PRESETS).map(([id, p]) => {
                        const isSel = (aiCfg.brokerPreset || 'none') === id;
                        return (
                          <button key={id} type="button"
                            onClick={() => { const next = { ...aiCfg, brokerPreset: id }; setAiCfg(next); saveAiSettings(next); }}
                            style={{ padding: '10px 14px', borderRadius: 6, fontFamily: 'inherit', fontSize: 11, cursor: 'pointer', textAlign: 'left', transition: 'all .15s', background: isSel ? '#0f1a2e' : '#060810', border: `1px solid ${isSel ? '#3b82f6' : '#0f1729'}`, color: isSel ? '#93c5fd' : '#64748b', fontWeight: isSel ? 600 : 400 }}>
                            {p.label}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Timezone Lock */}
                  <div style={{ background: '#070d1a', border: '1px solid #1e293b', borderRadius: 8, padding: '16px 20px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16 }}>
                      <div>
                        <div style={{ fontSize: 12, color: '#e2e8f0', fontWeight: 500, marginBottom: 4 }}>🕐 Lock Sessions to EST/EDT</div>
                        <div style={{ fontSize: 10, color: '#64748b', lineHeight: 1.6, maxWidth: 320 }}>
                          Forces NY Open, London, and Asian session windows to always use Eastern time — even when traveling.
                          Turn off only if your broker timestamps in your local timezone.
                        </div>
                      </div>
                      <button onClick={() => { const next = { ...aiCfg, tzLock: !aiCfg.tzLock }; setAiCfg(next); saveAiSettings(next); }}
                        style={{ background: aiCfg.tzLock !== false ? '#052e16' : '#450a0a', border: `1px solid ${aiCfg.tzLock !== false ? '#166534' : '#7f1d1d'}`, color: aiCfg.tzLock !== false ? '#4ade80' : '#f87171', padding: '8px 18px', borderRadius: 6, fontFamily: 'inherit', fontSize: 12, cursor: 'pointer', fontWeight: 600, letterSpacing: '0.08em', flexShrink: 0, whiteSpace: 'nowrap' }}>
                        {aiCfg.tzLock !== false ? '● ON' : '○ OFF'}
                      </button>
                    </div>
                  </div>

                  {/* Auto-Backup */}
                  <div style={{ background: '#070d1a', border: '1px solid #1e293b', borderRadius: 8, padding: '16px 20px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16 }}>
                      <div>
                        <div style={{ fontSize: 12, color: '#e2e8f0', fontWeight: 500, marginBottom: 4 }}>💾 Auto-Download Backup on Save</div>
                        <div style={{ fontSize: 10, color: '#64748b', lineHeight: 1.6, maxWidth: 320 }}>
                          Every time you save an entry, a .json backup file is downloaded automatically. Best insurance against cleared browser storage.
                        </div>
                      </div>
                      <button onClick={() => { const next = { ...aiCfg, autoBackup: !aiCfg.autoBackup }; setAiCfg(next); saveAiSettings(next); }}
                        style={{ background: aiCfg.autoBackup ? '#052e16' : '#450a0a', border: `1px solid ${aiCfg.autoBackup ? '#166534' : '#7f1d1d'}`, color: aiCfg.autoBackup ? '#4ade80' : '#f87171', padding: '8px 18px', borderRadius: 6, fontFamily: 'inherit', fontSize: 12, cursor: 'pointer', fontWeight: 600, letterSpacing: '0.08em', flexShrink: 0, whiteSpace: 'nowrap' }}>
                        {aiCfg.autoBackup ? '● ON' : '○ OFF'}
                      </button>
                    </div>
                  </div>

                  {/* Journal Stats */}
                  <div style={{ background: '#070d1a', border: '1px solid #1e293b', borderRadius: 8, padding: '20px' }}>
                    <div style={{ fontSize: 9, color: '#94a3b8', letterSpacing: '0.12em', marginBottom: 14, textTransform: 'uppercase' }}>JOURNAL OVERVIEW</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
                      {[
                        { l: 'Total Journals', v: journals.length },
                        { l: 'Prop Accounts', v: journals.filter(j => j.type === JOURNAL_TYPES.PROP).length },
                        { l: 'Personal Accounts', v: journals.filter(j => j.type === JOURNAL_TYPES.PERSONAL).length },
                        { l: 'Total Entries', v: entries.length },
                      ].map(({ l, v }) => (
                        <div key={l} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: '1px solid #0a1220' }}>
                          <span style={{ fontSize: 11, color: '#64748b' }}>{l}</span>
                          <span style={{ fontSize: 12, color: '#e2e8f0', fontFamily: "'DM Mono',monospace" }}>{v}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

            </div>
          </div>
        </div>
      )}

      {/* Journal tab strip */}
      <div style={{ borderBottom: "1px solid #1e293b", background: "#060810", padding: "0 40px", display: "flex", alignItems: "stretch", gap: 2, overflowX: "auto", position: "sticky", top: 80, zIndex: 9 }}>
        {journals.map(j => {
          const isActive = j.id === activeJournalId;
          const isRenaming = renamingId === j.id;
          return (
            <div key={j.id} style={{ display: "flex", alignItems: "center", gap: 0, position: "relative" }}>
              {isRenaming ? (
                <div style={{ display: "flex", alignItems: "center", gap: 4, padding: "8px 10px", borderBottom: "2px solid #3b82f6" }}>
                  <input
                    autoFocus
                    value={renameValue}
                    onChange={e => setRenameValue(e.target.value)}
                    onKeyDown={e => { if (e.key === "Enter") renameJournal(j.id, renameValue); if (e.key === "Escape") setRenamingId(null); }}
                    style={{ width: 120, padding: "2px 6px!important", fontSize: 11, background: "#0f1729!important", border: "1px solid #3b82f6!important", borderRadius: 3, color: "#e2e8f0!important" }}
                  />
                  <button onClick={() => renameJournal(j.id, renameValue)} style={{ background: "#1d4ed8", border: "none", color: "white", padding: "2px 8px", borderRadius: 3, fontSize: 10, cursor: "pointer", fontFamily: "inherit" }}>✓</button>
                  <button onClick={() => setRenamingId(null)} style={{ background: "transparent", border: "none", color: "#64748b", padding: "2px 6px", fontSize: 10, cursor: "pointer", fontFamily: "inherit" }}>✕</button>
                </div>
              ) : (
                <div
                  onClick={() => switchJournal(j.id)}
                  style={{ display: "flex", alignItems: "center", gap: 8, padding: isActive ? "10px 18px" : "14px 20px", cursor: "pointer", borderBottom: isActive ? "none" : "3px solid transparent", background: "transparent", transition: "all .15s", whiteSpace: "nowrap", position: "relative" }}
                  onMouseEnter={e => { if (!isActive) e.currentTarget.style.borderBottomColor = "#1e3a5f"; }}
                  onMouseLeave={e => { if (!isActive) e.currentTarget.style.borderBottomColor = "transparent"; }}>
                  {isActive ? (
                    <div style={{ position: "relative", display: "flex", alignItems: "center", gap: 10,
                      background: "rgba(56,189,248,0.04)", border: "1px solid rgba(56,189,248,0.2)",
                      borderRadius: 6, padding: "8px 16px",
                      outline: "1px solid rgba(192,132,252,0.08)", outlineOffset: 3 }}>
                      {/* All 4 corner brackets */}
                      <div style={{ position:"absolute", top:-1, left:-1, width:20, height:3, background:"linear-gradient(90deg,#38bdf8,transparent)" }} />
                      <div style={{ position:"absolute", top:-1, left:-1, width:3, height:20, background:"linear-gradient(180deg,#38bdf8,transparent)" }} />
                      <div style={{ position:"absolute", top:-1, right:-1, width:20, height:3, background:"linear-gradient(90deg,transparent,#818cf8)" }} />
                      <div style={{ position:"absolute", top:-1, right:-1, width:3, height:20, background:"linear-gradient(180deg,#818cf8,transparent)" }} />
                      <div style={{ position:"absolute", bottom:-1, left:-1, width:20, height:3, background:"linear-gradient(90deg,#818cf8,transparent)" }} />
                      <div style={{ position:"absolute", bottom:-1, left:-1, width:3, height:20, background:"linear-gradient(180deg,transparent,#818cf8)" }} />
                      <div style={{ position:"absolute", bottom:-1, right:-1, width:20, height:3, background:"linear-gradient(90deg,transparent,#c084fc)" }} />
                      <div style={{ position:"absolute", bottom:-1, right:-1, width:3, height:20, background:"linear-gradient(180deg,transparent,#c084fc)" }} />
                      <div style={{ display:"flex", flexDirection:"column", gap:2 }}>
                        <div style={{ fontFamily:"'DM Mono',monospace", fontSize:7, letterSpacing:"0.28em", display:"flex", alignItems:"center", gap:2 }}>
                          <span style={{ fontSize:9 }}>{j.type === JOURNAL_TYPES.PROP ? "🏆" : "💼"}</span>
                          <span style={{ background:"linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent", backgroundClip:"text" }}>
                            {j.type === JOURNAL_TYPES.PROP ? "PROP" : "PERSONAL"}
                          </span>
                        </div>
                        <div style={{ fontFamily:"'Cinzel',serif", fontWeight:900, fontSize:20, letterSpacing:"0.08em",
                          background:"linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)",
                          WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent", backgroundClip:"text", lineHeight:1.1 }}>
                          {j.name.toUpperCase()}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <span style={{ fontSize: 13, color: "#64748b", letterSpacing: "0.06em", fontWeight: 400 }}>
                      {j.type === JOURNAL_TYPES.PROP ? "🏆" : "💼"} {j.name.toUpperCase()}
                    </span>
                  )}
                  {isActive && (
                    <button onClick={e => { e.stopPropagation(); setRenamingId(j.id); setRenameValue(j.name); }}
                      style={{ background: "#1e293b", border: "1px solid #334155", color: "#94a3b8", padding: "2px 7px", borderRadius: 3, fontSize: 10, cursor: "pointer", fontFamily: "inherit", lineHeight: 1.4, transition: "all .15s", marginLeft: 2 }}
                      title="Rename"
                      onMouseEnter={e => { e.currentTarget.style.background="#1e3a5f"; e.currentTarget.style.borderColor="#3b82f6"; e.currentTarget.style.color="#93c5fd"; }}
                      onMouseLeave={e => { e.currentTarget.style.background="#1e293b"; e.currentTarget.style.borderColor="#334155"; e.currentTarget.style.color="#94a3b8"; }}>✎</button>
                  )}
                  {isActive && journals.length > 1 && (
                    confirmDeleteId === j.id ? (
                      <span style={{ display: "flex", gap: 3, alignItems: "center" }}>
                        <button onClick={e => { e.stopPropagation(); deleteJournal(j.id); }} style={{ background: "#450a0a", border: "1px solid #7f1d1d", color: "#f87171", padding: "2px 8px", borderRadius: 3, fontSize: 10, cursor: "pointer", fontFamily: "inherit" }}>YES DELETE</button>
                        <button onClick={e => { e.stopPropagation(); setConfirmDeleteId(null); }} style={{ background: "#1e293b", border: "1px solid #334155", color: "#94a3b8", padding: "2px 7px", borderRadius: 3, fontSize: 10, cursor: "pointer", fontFamily: "inherit" }}>CANCEL</button>
                      </span>
                    ) : (
                      <button onClick={e => { e.stopPropagation(); setConfirmDeleteId(j.id); }}
                        style={{ background: "#1e293b", border: "1px solid #334155", color: "#94a3b8", padding: "2px 7px", borderRadius: 3, fontSize: 10, cursor: "pointer", fontFamily: "inherit", lineHeight: 1.4, transition: "all .15s", marginLeft: 2 }}
                        title="Delete journal"
                        onMouseEnter={e => { e.currentTarget.style.background="#450a0a"; e.currentTarget.style.borderColor="#7f1d1d"; e.currentTarget.style.color="#f87171"; }}
                        onMouseLeave={e => { e.currentTarget.style.background="#1e293b"; e.currentTarget.style.borderColor="#334155"; e.currentTarget.style.color="#94a3b8"; }}>🗑</button>
                    )
                  )}
                </div>
              )}
            </div>
          );
        })}

        {/* New journal button */}
        <button onClick={() => setShowJournalMgr(true)}
          style={{ background: "transparent", border: "none", color: "#1e3a5f", padding: "10px 14px", cursor: "pointer", fontFamily: "inherit", fontSize: 11, letterSpacing: "0.06em", transition: "color .15s", whiteSpace: "nowrap", borderBottom: "2px solid transparent", marginLeft: "auto" }}
          onMouseEnter={e => e.currentTarget.style.color = "#3b82f6"}
          onMouseLeave={e => e.currentTarget.style.color = "#1e3a5f"}>
          + NEW JOURNAL
        </button>
      </div>

      {/* ── NEW JOURNAL MODAL ── */}
      {showJournalMgr && (() => {
        const sz = newJournalConfig.accountSize || 50000;
        const tmpl = LUCID_FLEX_RULES[sz] || LUCID_FLEX_RULES[50000];
        const isPropType = newJournalType === JOURNAL_TYPES.PROP;
        const close = () => { setShowJournalMgr(false); setNewJournalName(""); setNewJournalType(JOURNAL_TYPES.PERSONAL); setNewJournalConfig(defaultPersonalConfig()); };
        return (
          <div onClick={close} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.8)", zIndex: 9000, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <div onClick={e => e.stopPropagation()}
              style={{ background: "#070d1a", border: `1px solid ${isPropType ? "#92400e" : "#1e3a5f"}`, borderRadius: 10, padding: "28px 32px", width: 520, maxWidth: "95vw", maxHeight: "92vh", overflowY: "auto" }}>

              {/* Header */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
                <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 22, letterSpacing: "0.12em", color: isPropType ? "#f59e0b" : "transparent", background: isPropType ? "transparent" : "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: isPropType ? "unset" : "text", WebkitTextFillColor: isPropType ? "unset" : "transparent", backgroundClip: isPropType ? "unset" : "text" }}>NEW JOURNAL</div>
                <button onClick={close} style={{ background: "transparent", border: "none", color: "#64748b", fontSize: 16, cursor: "pointer" }}>✕</button>
              </div>

              {/* Name */}
              <div style={{ marginBottom: 20 }}>
                <label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", display: "block", marginBottom: 6 }}>JOURNAL NAME</label>
                <input autoFocus placeholder="e.g. Lucid 50K Flex, Questrade TFSA…"
                  value={newJournalName} onChange={e => setNewJournalName(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && createJournal()}
                  style={{ width: "100%", boxSizing: "border-box" }} />
              </div>

              {/* Type picker */}
              <div style={{ marginBottom: 24 }}>
                <label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", display: "block", marginBottom: 10 }}>ACCOUNT TYPE</label>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                  {[
                    { type: JOURNAL_TYPES.PERSONAL, emoji: "💼", label: "Personal", sub: "Cash, Margin, TFSA/RRSP — your own capital" },
                    { type: JOURNAL_TYPES.PROP, emoji: "🏆", label: "Prop Firm", sub: "Tracks full lifecycle: Eval → Funded in one journal" },
                  ].map(({ type, emoji, label, sub }) => (
                    <div key={type} onClick={() => { setNewJournalType(type); setNewJournalConfig(type === JOURNAL_TYPES.PROP ? defaultPropConfig(50000) : defaultPersonalConfig()); }}
                      style={{ background: newJournalType === type ? (type === JOURNAL_TYPES.PROP ? "rgba(245,158,11,0.08)" : "#0f1a2e") : "#060b18", border: `1px solid ${newJournalType === type ? (type === JOURNAL_TYPES.PROP ? "#f59e0b" : "#3b82f6") : "#1e293b"}`, borderRadius: 8, padding: "14px 16px", cursor: "pointer", transition: "all .15s" }}>
                      <div style={{ fontSize: 22, marginBottom: 6 }}>{emoji}</div>
                      <div style={{ fontSize: 12, color: newJournalType === type ? (type === JOURNAL_TYPES.PROP ? "#f59e0b" : "#93c5fd") : "#e2e8f0", fontWeight: 500, marginBottom: 4 }}>{label}</div>
                      <div style={{ fontSize: 10, color: "#64748b", lineHeight: 1.5 }}>{sub}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Personal config */}
              {newJournalType === JOURNAL_TYPES.PERSONAL && (
                <div style={{ background: "#060b18", border: "1px solid #1e293b", borderRadius: 8, padding: "16px 18px", marginBottom: 20 }}>
                  <div style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 14 }}>💼 PERSONAL ACCOUNT SETUP</div>
                  <label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.08em", display: "block", marginBottom: 6 }}>STARTING BALANCE ($)</label>
                  <input type="number" placeholder="e.g. 10000" value={newJournalConfig.startingBalance || ""}
                    onChange={e => setNewJournalConfig(c => ({ ...c, startingBalance: parseFloat(e.target.value) || 0 }))}
                    style={{ width: "100%", boxSizing: "border-box" }} />
                  <div style={{ fontSize: 9, color: "#64748b", marginTop: 4 }}>Auto-calculates as you log trades. Add cash deposits on individual entries.</div>
                </div>
              )}

              {/* Prop config */}
              {newJournalType === JOURNAL_TYPES.PROP && (
                <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

                  {/* Firm + account size */}
                  <div style={{ background: "#060b18", border: "1px solid #92400e", borderRadius: 8, padding: "16px 18px" }}>
                    <div style={{ fontSize: 10, color: "#f59e0b", letterSpacing: "0.1em", marginBottom: 14 }}>🏆 PROP FIRM</div>
                    <div style={{ marginBottom: 12 }}>
                      <label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.08em", display: "block", marginBottom: 6 }}>FIRM NAME</label>
                      <input value={newJournalConfig.firmName || "Lucid Trading"}
                        onChange={e => setNewJournalConfig(c => ({ ...c, firmName: e.target.value }))}
                        style={{ width: "100%", boxSizing: "border-box" }} />
                    </div>
                    <div>
                      <label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.08em", display: "block", marginBottom: 8 }}>ACCOUNT SIZE</label>
                      <div style={{ display: "flex", gap: 8 }}>
                        {PROP_ACCOUNT_SIZES.map(s => (
                          <button key={s} onClick={() => setNewJournalConfig(c => ({ ...defaultPropConfig(s), firmName: c.firmName }))}
                            style={{ flex: 1, padding: "8px 0", borderRadius: 4, fontSize: 11, fontFamily: "inherit", cursor: "pointer", border: `1px solid ${sz === s ? "#f59e0b" : "#1e293b"}`, background: sz === s ? "rgba(245,158,11,0.12)" : "#0f1729", color: sz === s ? "#f59e0b" : "#94a3b8", transition: "all .15s" }}>
                            ${(s/1000).toFixed(0)}K
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Eval rules — pre-filled, editable */}
                  <div style={{ background: "#060b18", border: "1px solid #1e293b", borderRadius: 8, padding: "16px 18px" }}>
                    <div style={{ fontSize: 10, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 4 }}>📋 EVALUATION PHASE RULES</div>
                    <div style={{ fontSize: 9, color: "#64748b", marginBottom: 14 }}>Pre-filled from LucidFlex {tmpl.label} · Edit if needed</div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                      {[
                        { key: "profitTarget", label: "PROFIT TARGET ($)", path: "eval" },
                        { key: "maxLossLimit", label: "MAX LOSS LIMIT ($)", path: "eval" },
                        { key: "consistencyRule", label: "CONSISTENCY RULE (%)", path: "eval" },
                        { key: "minTradingDays", label: "MIN TRADING DAYS", path: "eval" },
                      ].map(({ key, label, path }) => (
                        <div key={key}>
                          <label style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.08em", display: "block", marginBottom: 5 }}>{label}</label>
                          <input type="number" value={newJournalConfig[path]?.[key] ?? ""}
                            onChange={e => setNewJournalConfig(c => ({ ...c, [path]: { ...c[path], [key]: parseFloat(e.target.value) || 0 } }))} />
                        </div>
                      ))}
                    </div>
                    <div style={{ marginTop: 10, fontSize: 9, color: "#64748b" }}>Drawdown type: <span style={{ color: "#64748b" }}>{newJournalConfig.eval?.drawdownType || "EOD"}</span> · Daily loss limit: <span style={{ color: "#4ade80" }}>NONE</span> · Max size: {tmpl.maxSize}</div>
                  </div>

                  {/* Funded rules — pre-filled, editable */}
                  <div style={{ background: "#060b18", border: "1px solid #1e293b", borderRadius: 8, padding: "16px 18px" }}>
                    <div style={{ fontSize: 10, color: "#4ade80", letterSpacing: "0.1em", marginBottom: 4 }}>✓ FUNDED PHASE RULES</div>
                    <div style={{ fontSize: 9, color: "#64748b", marginBottom: 14 }}>Applied automatically when you mark eval as passed</div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                      {[
                        { key: "maxLossLimit", label: "MAX LOSS LIMIT ($)", path: "funded" },
                        { key: "minProfitDays", label: "MIN PROFIT DAYS", path: "funded" },
                        { key: "minDailyProfit", label: "MIN DAILY PROFIT ($)", path: "funded" },
                        { key: "daysToPayout", label: "DAYS TO PAYOUT", path: "funded" },
                      ].map(({ key, label, path }) => (
                        <div key={key}>
                          <label style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.08em", display: "block", marginBottom: 5 }}>{label}</label>
                          <input type="number" value={newJournalConfig[path]?.[key] ?? ""}
                            onChange={e => setNewJournalConfig(c => ({ ...c, [path]: { ...c[path], [key]: parseFloat(e.target.value) || 0 } }))} />
                        </div>
                      ))}
                    </div>
                    <div style={{ marginTop: 10, fontSize: 9, color: "#64748b" }}>No consistency rule · No daily loss limit · Payouts to live: <span style={{ color: "#64748b" }}>{newJournalConfig.funded?.payoutsToLive || 5}</span></div>
                  </div>

                  <div style={{ fontSize: 9, color: "#64748b", lineHeight: 1.6, padding: "0 2px" }}>
                    📌 This journal tracks your <strong style={{ color: "#64748b" }}>full account lifecycle</strong> — eval and funded stats stay in the same journal. When you pass the eval, hit "Mark as Passed" on the dashboard to switch phases with a fresh balance. All historical data is preserved.
                  </div>
                </div>
              )}

              {/* Create button */}
              <button onClick={createJournal}
                style={{ width: "100%", marginTop: 20, background: isPropType ? "#92400e" : "#1d4ed8", border: `1px solid ${isPropType ? "#f59e0b" : "#3b82f6"}`, color: "white", padding: "12px 0", borderRadius: 6, fontFamily: "inherit", fontSize: 13, letterSpacing: "0.08em", cursor: "pointer" }}>
                {isPropType ? "🏆" : "💼"} CREATE {(newJournalName.trim() || "JOURNAL").toUpperCase()}
              </button>
            </div>
          </div>
        );
      })()}

      {/* ── QUOTE OF THE DAY — independent full-width section ── */}
      {headerQuotes.length > 0 && (view === "list" || view === "recap") && (() => {
        const CARD_STYLES = [
          { accent: "#f59e0b", glow: "rgba(245,158,11,0.06)",  label: "✦ DISCIPLINE ✦",       textSize: 16, layout: "left"   },
          { accent: "#3b82f6", glow: "rgba(59,130,246,0.07)", label: "✦ QUOTE OF THE DAY ✦", textSize: 22, layout: "center" },
          { accent: "#4ade80", glow: "rgba(74,222,128,0.06)",  label: "✦ EDGE ✦",              textSize: 16, layout: "left"   },
        ];
        // Reorder quotes so hero (QUOTE OF THE DAY) is centre
        const orderedQuotes = headerQuotes.length >= 3
          ? [headerQuotes[1], headerQuotes[0], headerQuotes[2]]
          : headerQuotes;
        return (
          <div style={{ width: "100%", padding: "28px 40px 0", boxSizing: "border-box", display: "grid", gridTemplateColumns: "1fr 1.4fr 1fr", gap: 20, marginBottom: 0 }}>
              {orderedQuotes.map((q, i) => {
                const cs = CARD_STYLES[i] || CARD_STYLES[2];
                const isHero = i === 1;
                return (
                  <div key={q.id} onClick={() => setView("quotes")} style={{ borderRadius: 8, border: `1px solid ${cs.accent}28`, background: "#070d1a", padding: isHero ? "52px 44px 48px" : "44px 32px 40px", position: "relative", overflow: "hidden", textAlign: cs.layout, cursor: "pointer", transition: "border-color .15s" }} onMouseEnter={e => e.currentTarget.style.borderColor=`${cs.accent}55`} onMouseLeave={e => e.currentTarget.style.borderColor=`${cs.accent}28`}>
                    {/* Top accent line */}
                    <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, transparent, ${cs.accent}, transparent)`, opacity: 0.7 }} />
                    {/* Glow */}
                    <div style={{ position: "absolute", inset: 0, background: `radial-gradient(ellipse at 50% 0%, ${cs.glow} 0%, transparent 65%)`, pointerEvents: "none" }} />
                    {/* Ghost quote mark */}
                    <div style={{ position: "absolute", right: isHero ? 14 : 8, top: isHero ? 2 : -2, fontFamily: "Georgia,serif", fontSize: isHero ? 140 : 100, color: cs.accent, opacity: 0.07, lineHeight: 1, userSelect: "none", pointerEvents: "none" }}>"</div>
                    {/* Label */}
                    <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: isHero ? 14 : 13, color: cs.accent, letterSpacing: "0.3em", marginBottom: isHero ? 20 : 14, opacity: 0.9 }}>{cs.label}</div>
                    {/* Category pill */}
                    <div style={{ display: "inline-block", fontSize: 9, color: cs.accent, background: `${cs.accent}18`, border: `1px solid ${cs.accent}33`, padding: "3px 11px", borderRadius: 20, letterSpacing: "0.12em", marginBottom: isHero ? 18 : 14 }}>{q.category.toUpperCase()}</div>
                    {/* Quote text */}
                    <div style={{ fontFamily: "Georgia,'Times New Roman',serif", fontStyle: "italic", fontSize: isHero ? cs.textSize + 2 : cs.textSize + 3, color: isHero ? "#e2e8f0" : "#c8ddf0", lineHeight: 1.85, marginBottom: isHero ? 24 : 18, position: "relative", zIndex: 1 }}>
                      "{q.text}"
                    </div>
                    {/* Divider + Author */}
                    <div style={{ display: "flex", alignItems: "center", gap: 10, justifyContent: cs.layout === "center" ? "center" : "flex-start" }}>
                      <div style={{ width: isHero ? 32 : 18, height: 1, background: `${cs.accent}66` }} />
                      <span style={{ fontFamily: "'DM Mono',monospace", fontSize: isHero ? 11 : 10, color: cs.accent, letterSpacing: "0.16em" }}>{q.author.toUpperCase()}</span>
                    </div>
                  </div>
                );
              })}
            </div>
        );
      })()}

      <div style={{ width: "100%", padding: "28px 40px", boxSizing: "border-box" }}>
        {/* LIST */}
        {view === "list" && (<div>
          {filtered.length > 0 && (
            <div style={{ marginBottom: 24 }}>
              {/* Row 1 — Total P&L prominent + Account Balance */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
                <div style={{ background: "linear-gradient(135deg,rgba(56,189,248,0.12),rgba(129,140,248,0.14),rgba(192,132,252,0.09))", border: "1px solid rgba(129,140,248,0.3)", borderRadius: 6, padding: "14px 20px", position: "relative", overflow: "hidden", backdropFilter: "blur(8px)" }}>
                  <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc)" }} />
                  <div style={{ fontSize: 10, color: "#818cf8", letterSpacing: "0.12em", marginBottom: 4 }}>TOTAL NET P&L</div>
                  <div style={{ fontSize: 28, color: pnlColor(totalPnL), fontWeight: 700, letterSpacing: "0.02em" }}>{totalPnL >= 0 ? "+" : "-"}${Math.abs(totalPnL).toLocaleString("en-US", { maximumFractionDigits: 0 })}</div>
                  <div style={{ fontSize: 10, color: "#334155", marginTop: 2 }}>{filtered.length} trading days · {winDays}W {filtered.length - winDays}L</div>
                </div>
                <div style={{ background: "linear-gradient(135deg,rgba(192,132,252,0.12),rgba(129,140,248,0.14),rgba(56,189,248,0.09))", border: "1px solid rgba(192,132,252,0.3)", borderRadius: 6, padding: "14px 20px", position: "relative", overflow: "hidden", backdropFilter: "blur(8px)" }}>
                  <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg,#c084fc,#818cf8,#38bdf8)" }} />
                  <div style={{ fontSize: 10, color: "#818cf8", letterSpacing: "0.12em", marginBottom: 4 }}>ACCOUNT BALANCE</div>
                  {totalAccountBalance !== null ? (
                    <>
                      <div style={{ fontSize: 28, color: totalAccountBalance > (activeJournal?.config?.startingBalance || 0) ? "#4ade80" : totalAccountBalance < (activeJournal?.config?.startingBalance || 0) ? "#f87171" : "#e2e8f0", fontWeight: 700 }}>${totalAccountBalance.toLocaleString("en-US", { maximumFractionDigits: 0 })}</div>
                      <div style={{ fontSize: 10, color: "#334155", marginTop: 2 }}>started ${activeJournal?.config?.startingBalance?.toLocaleString("en-US", { maximumFractionDigits: 0 }) || "—"}</div>
                    </>
                  ) : (
                    <div style={{ fontSize: 13, color: "#334155", marginTop: 6 }}>Set starting capital in journal settings</div>
                  )}
                </div>
              </div>
              {/* Row 2 — Trade statistics */}
              {globalAnalytics && (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(6,1fr)", gap: 8 }}>
                  {[
                    { l: "TRADE WIN RATE", v: `${Math.round(globalAnalytics.winRate)}%`, c: globalAnalytics.winRate >= 50 ? "#4ade80" : "#f87171" },
                    { l: "PROFIT FACTOR", v: fmtPF(globalAnalytics.profitFactor), c: pfColor(globalAnalytics.profitFactor) },
                    { l: "EXPECTANCY", v: globalAnalytics.expectancy != null ? `${globalAnalytics.expectancy >= 0 ? "+" : ""}$${Math.abs(globalAnalytics.expectancy * Math.abs(globalAnalytics.avgLoss || 1)).toFixed(0)}` : "—", c: globalAnalytics.expectancy != null && globalAnalytics.expectancy >= 0 ? "#4ade80" : "#f87171" },
                    { l: "MAX DRAWDOWN", v: globalAnalytics.maxDD != null ? `-$${Math.abs(globalAnalytics.maxDD).toFixed(0)}` : "—", c: "#f87171" },
                    { l: "TOTAL TRADES", v: globalAnalytics.total || filtered.length, c: "#e2e8f0" },
                  ].map(s => (
                    <div key={s.l} style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 5, padding: "8px 10px" }}>
                      <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 3 }}>{s.l}</div>
                      <div style={{ fontSize: 13, color: s.c, fontWeight: 600 }}>{s.v}</div>
                    </div>
                  ))}
                  {/* AVG WIN / LOSS — split colors */}
                  <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 5, padding: "8px 10px" }}>
                    <div style={{ fontSize: 8, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 3 }}>AVG WIN / LOSS</div>
                    <div style={{ fontSize: 13, fontWeight: 600 }}>
                      {globalAnalytics.avgWin && globalAnalytics.avgLoss ? (
                        <>
                          <span style={{ color: "#4ade80" }}>${Math.abs(globalAnalytics.avgWin).toFixed(0)}</span>
                          <span style={{ color: "#475569" }}> / </span>
                          <span style={{ color: "#f87171" }}>-${Math.abs(globalAnalytics.avgLoss).toFixed(0)}</span>
                        </>
                      ) : "—"}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* View toggle + month nav */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 24, flexWrap: "wrap", gap: 10 }}>
              {[["calendar","📅 MONTHLY PERFORMANCE"],["list","📋 TRADE LIST"],["weekly","📈 WEEKLY PERFORMANCE"],["performance","📊 ANNUAL/QUARTERLY PERFORMANCE"]].map(([m, label]) => (
                listMode === m ? (
                  <button key={m} onClick={() => setListMode(m)}
                    style={{ position: "relative", padding: "11px 20px", borderRadius: 6, fontFamily: "inherit", fontSize: 13, cursor: "pointer", letterSpacing: "0.06em", transition: "all .15s", whiteSpace: "nowrap", overflow: "hidden",
                      background: "rgba(10,18,32,0.95)", border: "1px solid #1e3a5f", color: "#e2e8f0" }}>
                    <span style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc)", borderRadius: "0 0 5px 5px" }} />
                    {label}
                  </button>
                ) : (
                  <span key={m} style={{ display: "inline-block", padding: 1, borderRadius: 7, background: "linear-gradient(135deg,rgba(56,189,248,0.45),rgba(129,140,248,0.45),rgba(192,132,252,0.45))", flexShrink: 0 }}>
                    <button onClick={() => setListMode(m)}
                      style={{ display: "block", background: "#070d1a", color: "#64748b", border: "none", padding: "10px 19px", borderRadius: 6, fontFamily: "inherit", fontSize: 13, cursor: "pointer", letterSpacing: "0.06em", transition: "all .15s", whiteSpace: "nowrap" }}>
                      {label}
                    </button>
                  </span>
                )
              ))}
              <span style={{ display: "inline-block", padding: 1, borderRadius: 7, background: "linear-gradient(135deg,rgba(56,189,248,0.45),rgba(129,140,248,0.45),rgba(192,132,252,0.45))", flexShrink: 0 }}>
                <button onClick={() => setView("reference")}
                  style={{ display: "block", background: "#070d1a", color: "#64748b", border: "none", padding: "10px 19px", borderRadius: 6, fontFamily: "inherit", fontSize: 13, cursor: "pointer", letterSpacing: "0.06em", transition: "all .15s", whiteSpace: "nowrap" }}>
                  📊 REFERENCE
                </button>
              </span>

              {listMode === "calendar" ? (
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ display:"inline-block", padding:1, borderRadius:5, background:"linear-gradient(135deg,rgba(56,189,248,0.45),rgba(129,140,248,0.45),rgba(192,132,252,0.45))" }}>
                  <button onClick={() => { const [y, m] = calMonth.split("-").map(Number); const d = new Date(y, m - 2, 1); setCalMonth(`${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`); }}
                    style={{ display:"block", background:"#070d1a", color:"#64748b", border:"none", padding:"10px 15px", borderRadius:4, fontFamily:"inherit", fontSize:13, cursor:"pointer" }}>‹</button>
                </span>
                <span style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 22, letterSpacing: "0.12em", minWidth: 160, textAlign: "center", background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text", lineHeight: 1 }}>
                  {(() => { const [y, m] = calMonth.split("-").map(Number); return new Date(y, m - 1, 1).toLocaleString("default", { month: "long", year: "numeric" }).toUpperCase(); })()}
                </span>
                <span style={{ display:"inline-block", padding:1, borderRadius:5, background:"linear-gradient(135deg,rgba(56,189,248,0.45),rgba(129,140,248,0.45),rgba(192,132,252,0.45))" }}>
                  <button onClick={() => { const [y, m] = calMonth.split("-").map(Number); const d = new Date(y, m, 1); setCalMonth(`${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`); }}
                    style={{ display:"block", background:"#070d1a", color:"#64748b", border:"none", padding:"10px 15px", borderRadius:4, fontFamily:"inherit", fontSize:13, cursor:"pointer" }}>›</button>
                </span>
              </div>
            ) : listMode === "performance" || listMode === "weekly" ? null : (
              months.length > 1 && (
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  <button className={`pill ${filterMonth === "" ? "sel" : ""}`} onClick={() => setFilterMonth("")}>ALL</button>
                  {months.map(m => <button key={m} className={`pill ${filterMonth === m ? "sel" : ""}`} onClick={() => setFilterMonth(m)}>{new Date(m + "-01T12:00:00").toLocaleString("default", { month: "short", year: "numeric" }).toUpperCase()}</button>)}
                </div>
              )
            )}
          </div>

          {entries.length === 0 ? (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "60vh", textAlign: "center", padding: "60px 0" }}>
              <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 56, color: "#1e3a5f", marginBottom: 14, letterSpacing: "0.08em" }}>NO ENTRIES YET</div>
              <div style={{ fontSize: 13, color: "#475569", marginBottom: 32, letterSpacing: "0.04em" }}>Start journaling today. Your future self will thank you.</div>
              <button onClick={openNew} style={{ background: "#1d4ed8", color: "white", border: "none", padding: "14px 36px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer", letterSpacing: "0.08em" }}>+ CREATE FIRST ENTRY</button>
            </div>
          ) : listMode === "calendar" ? (
            <>
              <div style={{ marginBottom: 20 }}>
  <div style={{ height: 2, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc,transparent)", borderRadius: 1, marginBottom: 14 }} />
  <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
    <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 32, letterSpacing: "0.12em", background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text", lineHeight: 1 }}>✦ MONTHLY PERFORMANCE</div>
  </div>
  <div className="helper-text" style={{ marginTop: 4 }}>REVIEW YOUR CALENDAR · TRACK YOUR EDGE</div>
</div>
              <CalendarView
              month={calMonth}
              entries={entries}
              onDayClick={(entry) => viewDetail(entry)}
              onNewDay={(date) => { const e = emptyEntry(); e.date = date; setForm(e); setActiveEntry(null); setTab("session"); setImportRaw(""); setImportError(""); setImportSuccess(false); setAnalyticsTab("overview"); setView("new"); }}
              pnlColor={pnlColor}
              fmtPnl={fmtPnl}
              netPnl={netPnl}
              gradeColor={gradeColor}
              calcAnalytics={calcAnalytics}
            />
            </>
          ) : listMode === "weekly" ? (
            <>
              <div style={{ marginBottom: 20 }}>
  <div style={{ height: 2, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc,transparent)", borderRadius: 1, marginBottom: 14 }} />
  <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
    <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 32, letterSpacing: "0.12em", background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text", lineHeight: 1 }}>✦ WEEKLY PERFORMANCE</div>
  </div>
  <div className="helper-text" style={{ marginTop: 4 }}>WEEK BY WEEK PROGRESS · SPOT YOUR PATTERNS</div>
</div>
              <WeeklyPerformance entries={entries} netPnl={netPnl} fmtPnl={fmtPnl} pnlColor={pnlColor} calcAnalytics={calcAnalytics} ai={aiCfg} activeJournal={activeJournal} />
            </>
          ) : listMode === "performance" ? (
            <>
              <div style={{ marginBottom: 20 }}>
  <div style={{ height: 2, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc,transparent)", borderRadius: 1, marginBottom: 14 }} />
  <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
    <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 32, letterSpacing: "0.12em", background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text", lineHeight: 1 }}>✦ ANNUAL PERFORMANCE BREAKDOWN</div>
  </div>
  <div className="helper-text" style={{ marginTop: 4 }}>YEARLY OVERVIEW · MEASURE YOUR GROWTH</div>
</div>
              <PerformanceOverview entries={entries} netPnl={netPnl} fmtPnl={fmtPnl} pnlColor={pnlColor} />
            </>
          ) : (
            <div>
              <div style={{ marginBottom: 16 }}>
  <div style={{ height: 2, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc,transparent)", borderRadius: 1, marginBottom: 14 }} />
  <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
    <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 32, letterSpacing: "0.12em", background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text", lineHeight: 1 }}>✦ TRADE LIST</div>
  </div>
  <div className="helper-text" style={{ marginTop: 4 }}>ALL SESSIONS · SEARCH · FILTER · REVIEW</div>
</div>

              {/* Sub-tab strip: Journal Entries | CSV Import Log */}
              <div style={{ display: "flex", gap: 6, marginBottom: 16, alignItems: "center", justifyContent: "space-between" }}>
                <div style={{ display: "flex", gap: 6 }}>
                  {[["journal","📋 JOURNAL ENTRIES"],["csv","📂 CSV IMPORT LOG"]].map(([id,label]) => (
                    listSubTab === id ? (
                      <button key={id} onClick={() => setListSubTab(id)}
                        style={{ position:"relative", padding:"9px 18px", borderRadius:6, fontFamily:"inherit", fontSize:12, cursor:"pointer", letterSpacing:"0.06em", fontWeight:600, overflow:"hidden", transition:"all .15s", background:"rgba(10,18,32,0.95)", border:"1px solid #1e3a5f", color:"#e2e8f0" }}>
                        <span style={{ position:"absolute", bottom:0, left:0, right:0, height:2, background:"linear-gradient(90deg,#38bdf8,#818cf8,#c084fc)" }} />
                        {label}
                      </button>
                    ) : (
                      <span key={id} style={{ display:"inline-block", padding:1, borderRadius:7, background:"linear-gradient(135deg,rgba(56,189,248,0.45),rgba(129,140,248,0.45),rgba(192,132,252,0.45))" }}>
                        <button onClick={() => setListSubTab(id)}
                          style={{ display:"block", background:"#070d1a", color:"#64748b", border:"none", padding:"8px 17px", borderRadius:6, fontFamily:"inherit", fontSize:12, cursor:"pointer", letterSpacing:"0.06em", fontWeight:600, whiteSpace:"nowrap" }}>
                          {label}
                        </button>
                      </span>
                    )
                  ))}
                </div>
                {listSubTab === "journal" && (
                  <button onClick={() => setSortNewest(p => !p)}
                    style={{ background:"transparent", border:"1px solid #1e293b", borderRadius:6, color:"#64748b", fontSize:11, padding:"7px 14px", cursor:"pointer", fontFamily:"inherit", letterSpacing:"0.06em", display:"flex", alignItems:"center", gap:5 }}>
                    {sortNewest ? "↓ NEWEST FIRST" : "↑ OLDEST FIRST"}
                  </button>
                )}
              </div>

              {/* ── CSV IMPORT LOG sub-view ── */}
              {listSubTab === "csv" && (() => {
                const today = new Date().toISOString().slice(0,10);
                const weekAgo = new Date(Date.now() - 7*86400000).toISOString().slice(0,10);
                const monthAgo = new Date(Date.now() - 30*86400000).toISOString().slice(0,10);
                // Flatten all parsedTrades from all entries
                let allTrades = entries.flatMap(e =>
                  (e.parsedTrades || []).map(t => ({ ...t, _entryDate: e.date, _entryId: e.id }))
                ).filter(t => t.buyTime || t.sellTime || t.symbol);
                // Apply date filter
                const fromDate = csvDatePreset === "today" ? today
                  : csvDatePreset === "week" ? weekAgo
                  : csvDatePreset === "month" ? monthAgo
                  : csvDatePreset === "custom" ? csvDateFrom
                  : null;
                const toDate = csvDatePreset === "custom" ? csvDateTo : null;
                if (fromDate) allTrades = allTrades.filter(t => (t._entryDate || "") >= fromDate);
                if (toDate)   allTrades = allTrades.filter(t => (t._entryDate || "") <= toDate);
                allTrades = allTrades.sort((a, b) => (b._entryDate || "").localeCompare(a._entryDate || ""));
                const tNCSV = (t) => Number.isFinite(t.pnl) ? t.pnl - (t.commission||0) : 0;
                const totalPnL = allTrades.reduce((s,t) => s + tNCSV(t), 0);
                const wins = allTrades.filter(t => tNCSV(t) > 0).length;
                const wr = allTrades.length ? Math.round(wins/allTrades.length*100) : 0;
                return (
                  <div>
                    {/* Date filter strip */}
                    <div style={{ display:"flex", gap:6, marginBottom:14, alignItems:"center", flexWrap:"wrap" }}>
                      <span style={{ fontSize:9, color:"#3b82f6", letterSpacing:"0.1em" }}>RANGE:</span>
                      {[["all","ALL TIME"],["today","TODAY"],["week","LAST 7D"],["month","LAST 30D"],["custom","CUSTOM"]].map(([id,label]) => (
                        <button key={id} onClick={() => setCsvDatePreset(id)}
                          style={{ padding:"5px 12px", borderRadius:4, fontSize:10, fontFamily:"inherit", cursor:"pointer", letterSpacing:"0.06em",
                            background: csvDatePreset===id ? "#1e3a5f" : "transparent",
                            border: `1px solid ${csvDatePreset===id ? "#3b82f6" : "#1e293b"}`,
                            color: csvDatePreset===id ? "#93c5fd" : "#64748b" }}>
                          {label}
                        </button>
                      ))}
                      {csvDatePreset === "custom" && (
                        <>
                          <input type="date" value={csvDateFrom} onChange={e => setCsvDateFrom(e.target.value)} style={{ width:140, fontSize:11, padding:"4px 8px" }}/>
                          <span style={{ color:"#475569", fontSize:11 }}>→</span>
                          <input type="date" value={csvDateTo} onChange={e => setCsvDateTo(e.target.value)} style={{ width:140, fontSize:11, padding:"4px 8px" }}/>
                        </>
                      )}
                    </div>
                    {/* Summary strip */}
                    {allTrades.length > 0 && (
                      <div style={{ display:"flex", gap:16, marginBottom:12, padding:"10px 16px", background:"#0a0e1a", border:"1px solid #1e293b", borderRadius:6, flexWrap:"wrap" }}>
                        {[
                          { l:"TRADES", v:allTrades.length, c:"#e2e8f0" },
                          { l:"WIN RATE", v:`${wr}%`, c: wr>=50?"#4ade80":"#f87171" },
                          { l:"TOTAL P&L", v:(totalPnL>=0?"+":"")+`$${totalPnL.toFixed(2)}`, c:totalPnL>=0?"#4ade80":"#f87171" },
                          { l:"WINNERS", v:wins, c:"#4ade80" },
                          { l:"LOSERS", v:allTrades.length-wins, c:"#f87171" },
                        ].map(s => (
                          <div key={s.l} style={{ display:"flex", flexDirection:"column", gap:2 }}>
                            <div style={{ fontSize:8, color:"#3b82f6", letterSpacing:"0.12em" }}>{s.l}</div>
                            <div style={{ fontSize:14, color:s.c, fontWeight:600 }}>{s.v}</div>
                          </div>
                        ))}
                      </div>
                    )}
                    {/* Table */}
                    {allTrades.length === 0 ? (
                      <div style={{ textAlign:"center", padding:"40px 0", color:"#475569", fontSize:12 }}>No imported trade data for this range.</div>
                    ) : (
                      <div style={{ overflowX:"auto" }}>
                        <table style={{ width:"100%", borderCollapse:"collapse", fontSize:11 }}>
                          <thead>
                            <tr style={{ background:"#0a0e1a", borderBottom:"1px solid #1e3a5f" }}>
                              {["DATE","SYMBOL","DIRECTION","QTY","ENTRY","EXIT","DURATION","GROSS P&L","COMMISSION","NET P&L","ORDER TYPE"].map(h => (
                                <th key={h} style={{ padding:"8px 10px", textAlign:"left", fontSize:8, color:"#3b82f6", letterSpacing:"0.1em", fontWeight:400, whiteSpace:"nowrap" }}>{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {allTrades.map((t,i) => {
                              const net = Number.isFinite(t.pnl) ? t.pnl - (t.commission||0) : null;
                              const dir = (t.direction||"").toUpperCase();
                              return (
                                <tr key={i} style={{ borderBottom:"1px solid #0f1729", background: i%2===0?"#0a0e1a":"#0d1525" }}
                                  onMouseEnter={e=>e.currentTarget.style.background="#0f1e38"} onMouseLeave={e=>e.currentTarget.style.background=i%2===0?"#0a0e1a":"#0d1525"}>
                                  <td style={{ padding:"7px 10px", color:"#64748b", whiteSpace:"nowrap" }}>{t._entryDate}</td>
                                  <td style={{ padding:"7px 10px", color:"#93c5fd", fontWeight:600 }}>{t.symbol||"—"}</td>
                                  <td style={{ padding:"7px 10px" }}>
                                    <span style={{ padding:"2px 8px", borderRadius:3, fontSize:9, fontWeight:600, letterSpacing:"0.06em",
                                      background: dir==="LONG"?"rgba(74,222,128,0.12)":dir==="SHORT"?"rgba(248,113,113,0.12)":"rgba(148,163,184,0.1)",
                                      color: dir==="LONG"?"#4ade80":dir==="SHORT"?"#f87171":"#94a3b8" }}>{dir||"—"}</span>
                                  </td>
                                  <td style={{ padding:"7px 10px", color:"#e2e8f0" }}>{t.qty||"—"}</td>
                                  <td style={{ padding:"7px 10px", color:"#94a3b8", whiteSpace:"nowrap" }}>{t.buyPrice!=null?`$${Number(t.buyPrice).toFixed(2)}`:"—"}</td>
                                  <td style={{ padding:"7px 10px", color:"#94a3b8", whiteSpace:"nowrap" }}>{t.sellPrice!=null?`$${Number(t.sellPrice).toFixed(2)}`:"—"}</td>
                                  <td style={{ padding:"7px 10px", color:"#64748b" }}>{t.duration||"—"}</td>
                                  <td style={{ padding:"7px 10px", color:Number.isFinite(t.pnl)?(t.pnl>=0?"#4ade80":"#f87171"):"#64748b", fontWeight:600 }}>{Number.isFinite(t.pnl)?(t.pnl>=0?"+":"")+`$${t.pnl.toFixed(2)}`:"—"}</td>
                                  <td style={{ padding:"7px 10px", color:"#f87171" }}>{t.commission>0?`-$${t.commission.toFixed(2)}`:"—"}</td>
                                  <td style={{ padding:"7px 10px", color:net!=null?(net>=0?"#4ade80":"#f87171"):"#64748b", fontWeight:600 }}>{net!=null?(net>=0?"+":"")+`$${net.toFixed(2)}`:"—"}</td>
                                  <td style={{ padding:"7px 10px", color:"#64748b" }}>{t.orderType||"—"}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* ── JOURNAL ENTRIES sub-view ── */}
              {listSubTab === "journal" && <>

              {/* Search bar */}
              <div style={{ display: "flex", gap: 8, marginBottom: 14, alignItems: "center" }}>
                <div style={{ position: "relative", flex: 1 }}>
                  <span style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", fontSize: 12, color: "#64748b", pointerEvents: "none" }}>🔍</span>
                  <input
                    type="text"
                    placeholder="Search entries…"
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                    style={{ paddingLeft: 32, paddingRight: searchQuery ? 32 : 12 }}
                  />
                  {searchQuery && (
                    <button onClick={() => setSearchQuery("")}
                      style={{ position: "absolute", right: 8, top: "50%", transform: "translateY(-50%)", background: "transparent", border: "none", color: "#64748b", cursor: "pointer", fontSize: 13, lineHeight: 1, padding: 0 }}>✕</button>
                  )}
                </div>
                <select value={searchField} onChange={e => setSearchField(e.target.value)} style={{ width: 140, padding: "9px 10px", fontSize: 12 }}>
                  <option value="all">All fields</option>
                  <option value="notes">Notes only</option>
                  <option value="instruments">Instruments</option>
                  <option value="grade">Grade</option>
                  <option value="mistakes">Mistakes</option>
                </select>
              </div>
              {searchQuery && (
                <div style={{ fontSize: 11, color: "#64748b", marginBottom: 10, letterSpacing: "0.06em" }}>
                  {filteredAndSearched.length} result{filteredAndSearched.length !== 1 ? "s" : ""} for "<span style={{ color: "#93c5fd" }}>{searchQuery}</span>"
                </div>
              )}

              {filteredAndSearched.length === 0 ? (
                <div style={{ textAlign: "center", padding: "40px 0", color: "#64748b", fontSize: 12 }}>
                  {searchQuery ? `No entries match "${searchQuery}"` : "No entries for this period."}
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {filteredAndSearched.map(entry => {
                    const a = entryAnalyticsMap[entry.id];
                    return (
                      <div key={entry.id} className="entry-card" onClick={() => viewDetail(entry)}
                        style={{ background: "#0a0e1a", border: "1px solid #1e293b", borderBottom: "2px solid transparent", borderImage: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc,transparent) 1", borderRadius: 6, padding: "18px 24px", display: "grid", gridTemplateColumns: "160px 1fr auto auto", gap: 24, alignItems: "center" }}>

                        {/* DATE COLUMN */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                          <div style={{ fontSize: 11, color: "#475569", letterSpacing: "0.12em", textTransform: "uppercase" }}>
                            {new Date(entry.date + "T12:00:00").toLocaleDateString("en-US", { weekday: "long" }).toUpperCase()}
                          </div>
                          <div style={{ fontSize: 18, color: "#e2e8f0", fontWeight: 600, letterSpacing: "0.02em" }}>
                            {(() => { const d = new Date(entry.date + "T12:00:00"); return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }); })()}
                          </div>
                          <div style={{ display: "flex", gap: 6, marginTop: 2, flexWrap: "wrap" }}>
                            {(entry.instruments?.length ? entry.instruments : entry.instrument ? [entry.instrument] : []).map(sym => (
                              <span key={sym} style={{ padding: "3px 10px", borderRadius: 4, fontSize: 11, fontWeight: 600, background: "#1e3a5f", color: "#93c5fd", letterSpacing: "0.06em" }}>{sym}</span>
                            ))}
                          </div>
                        </div>

                        {/* STATS + TAGS COLUMN */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                          {/* Stats row */}
                          <div style={{ display: "flex", gap: 20, alignItems: "center", flexWrap: "wrap" }}>
                            {a && <>
                              <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                                <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase" }}>Trades</div>
                                <div style={{ fontSize: 15, color: "#e2e8f0", fontWeight: 600 }}>{a.trades}</div>
                              </div>
                              <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                                <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase" }}>Win Rate</div>
                                <div style={{ fontSize: 15, color: a.winRate >= 50 ? "#4ade80" : "#f87171", fontWeight: 600 }}>{Math.round(a.winRate)}%</div>
                              </div>
                              <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                                <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase" }}>W / L</div>
                                <div style={{ fontSize: 15, fontWeight: 600 }}>
                                  <span style={{ color: "#4ade80" }}>{a.winners}</span>
                                  <span style={{ color: "#475569" }}> / </span>
                                  <span style={{ color: "#f87171" }}>{a.losers}</span>
                                </div>
                              </div>
                              {a.profitFactor != null && (
                                <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                                  <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase" }}>Prof. Factor</div>
                                  <div style={{ fontSize: 15, color: pfColor(a.profitFactor), fontWeight: 600 }}>{fmtPF(a.profitFactor)}</div>
                                </div>
                              )}
                              {parseFloat(entry.commissions) > 0 && (
                                <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                                  <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase" }}>Fees</div>
                                  <div style={{ fontSize: 11, color: "#475569" }}>-${parseFloat(entry.commissions).toFixed(2)}</div>
                                </div>
                              )}
                            </>}
                          </div>
                          {/* Tags row */}
                          <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                            {entry.grade && <span style={{ padding: "3px 10px", borderRadius: 4, fontSize: 11, fontWeight: 600, background: "#0f172a", border: `1px solid ${gradeColor(entry.grade)}`, color: gradeColor(entry.grade) }}>{entry.grade}</span>}
                            {entry.bias && <span style={{ padding: "3px 10px", borderRadius: 4, fontSize: 11, background: entry.bias === "Bullish" ? "#052e16" : entry.bias === "Bearish" ? "#450a0a" : "#1e1b4b", color: entry.bias === "Bullish" ? "#4ade80" : entry.bias === "Bearish" ? "#f87171" : "#a5b4fc" }}>{entry.bias.toUpperCase()}</span>}
                            {(entry.moods?.length ? entry.moods : entry.mood ? [entry.mood] : []).map(m => (
                              <span key={m} style={{ fontSize: 12, color: "#64748b" }}>{m}</span>
                            ))}
                            {entry.sessionMistakes?.length > 0 && (
                              entry.sessionMistakes.includes("No Mistakes — Executed the Plan ✓") ? (
                                <span style={{ display: "inline-flex", alignItems: "center", gap: 4, padding: "3px 10px", borderRadius: 4, fontSize: 11, background: "#052e16", border: "1px solid #166534", color: "#4ade80" }}>✓ Clean Session</span>
                              ) : (
                                <>
                                  {entry.sessionMistakes.map((m, mi) => (
                                    <span key={mi} style={{ display: "inline-flex", alignItems: "center", gap: 3, padding: "3px 9px", borderRadius: 4, fontSize: 10, background: "rgba(63,16,16,0.5)", border: "1px solid #7f1d1d", color: "#f87171", letterSpacing: "0.01em" }}>
                                      ⚠ {m}
                                    </span>
                                  ))}
                                </>
                              )
                            )}
                          </div>
                        </div>

                        {/* EQUITY CURVE */}
                        <div style={{ width: 120 }}>{a && <MiniCurve curve={a.equityCurve} w={120} h={40} />}</div>

                        {/* P&L COLUMN */}
                        <div style={{ textAlign: "right", minWidth: 110 }}>
                          {entry.pnl !== "" && <>
                            <div style={{ fontSize: 22, color: pnlColor(netPnl(entry)), fontWeight: 700, letterSpacing: "0.02em" }}>{fmtPnl(netPnl(entry))}</div>
                            {parseFloat(entry.commissions) > 0 && <div style={{ fontSize: 10, color: "#475569", marginTop: 3 }}>net of fees</div>}
                            {entry.pnl !== "" && parseFloat(entry.pnl) !== netPnl(entry) && (
                              <div style={{ fontSize: 11, color: "#334155", marginTop: 2 }}>gross {fmtPnl(parseFloat(entry.pnl))}</div>
                            )}
                          </>}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </>}
            </div>
          )}

          {/* AI Recap banner */}
          {entries.length > 0 && (
            <div style={{ marginTop: 20, background: "#0a0e1a", border: "1px solid #1e3a5f", borderRadius: 6, overflow: "hidden", position: "relative" }}>
              <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc)" }} />
              <div style={{ padding: "18px 24px", background: "#0a1628", borderBottom: "1px solid #1e293b", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: "0.1em", background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>✦ AI WEEKLY & MONTHLY RECAP</div>
                  <div style={{ fontSize: 9, color: "#475569", marginTop: 3, letterSpacing: "0.1em" }}>ANALYSE · REFLECT · IMPROVE</div>
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <button onClick={() => { setRecapInitMode("weekly"); setView("recap"); }}
                    style={{ background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", color: "#070d1a", border: "none", padding: "8px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.06em", fontWeight: 700, transition: "all .15s" }}>
                    ANALYSE WEEK →
                  </button>
                  <button onClick={() => { setRecapInitMode("monthly"); setView("recap"); }}
                    style={{ background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", color: "#070d1a", border: "none", padding: "8px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.06em", fontWeight: 700, transition: "all .15s" }}>
                    ANALYSE MONTH →
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>)}

        {/* NEW/EDIT */}
        {view === "new" && (
          <div>
            <div style={{ marginBottom: 18 }}>
              <div style={{ marginBottom: 4 }}>
  <div style={{ height: 2, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc,transparent)", borderRadius: 1, marginBottom: 14 }} />
  <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
    <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 32, letterSpacing: "0.12em", background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text", lineHeight: 1 }}>{activeEntry ? "✎ EDIT ENTRY" : "✦ NEW ENTRY"}</div>
  </div>
  <div className="helper-text" style={{ marginTop: 4 }}>{activeEntry ? "UPDATE YOUR SESSION · REVISE YOUR NOTES" : "LOG YOUR SESSION · CAPTURE YOUR EDGE · GROW DAILY"}</div>
</div>
            </div>
            {!activeEntry && (() => {
              const yp = getYesterdayPlan();
              if (!yp || (!yp.plan && !yp.reinforceRule)) return null;
              return (
                <div style={{ marginBottom: 18, background: "#0a1628", border: "1px solid #1e3a5f", borderRadius: 6, padding: "12px 16px" }}>
                  <div style={{ fontSize: 10, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 10, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span>📋 FROM YESTERDAY · {yp.date}</span>
                    <span style={{ fontSize: 9, color: "#64748b" }}>your plan coming into today</span>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {yp.reinforceRule && (
                      <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                        <span style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.08em", whiteSpace: "nowrap", marginTop: 1 }}>RULE TO REINFORCE</span>
                        <span style={{ fontSize: 12, color: "#e2e8f0", lineHeight: 1.5 }}>{yp.reinforceRule}</span>
                      </div>
                    )}
                    {yp.plan && (
                      <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                        <span style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.08em", whiteSpace: "nowrap", marginTop: 1 }}>PLAN FOR TODAY</span>
                        <span style={{ fontSize: 12, color: "#e2e8f0", lineHeight: 1.5 }}>{yp.plan}</span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })()}
            <div style={{ display: "grid", gridTemplateColumns: "180px 1fr 180px", gap: 18, alignItems: "start", marginBottom: 18 }}>

              {/* ── DATE ── */}
              <div>
                <label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>DATE</label>
                <input type="date" value={form.date} onChange={e => f("date", e.target.value)} style={{ width: "100%" }} />
              </div>

              {/* ── TICKER / INSTRUMENTS ── */}
              <div>
                <label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>{isProp ? "INSTRUMENTS TRADED" : "TICKER / SYMBOL"}</label>
                {isProp ? (() => {
                  const known = ["ES", "MES", "NQ", "MNQ"];
                  const selected = form.instruments || [];
                  const toggle = (sym) => {
                    const next = selected.includes(sym) ? selected.filter(s => s !== sym) : [...selected, sym];
                    f("instruments", next);
                  };
                  const customVals = selected.filter(s => !known.includes(s));
                  return (
                    <div>
                      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center", marginBottom: customVals.length ? 6 : 0 }}>
                        {known.map(sym => (
                          <button key={sym} type="button" onClick={() => toggle(sym)}
                            style={{ padding: "8px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer", letterSpacing: "0.06em", transition: "all .15s", background: selected.includes(sym) ? "#1e3a5f" : "transparent", border: `1px solid ${selected.includes(sym) ? "#3b82f6" : "#1e293b"}`, color: selected.includes(sym) ? "#93c5fd" : "#94a3b8" }}>
                            {sym}
                          </button>
                        ))}
                        <button type="button" onClick={() => { if (!customVals.length) f("instruments", [...selected, ""]); }}
                          style={{ padding: "8px 14px", borderRadius: 4, fontFamily: "inherit", fontSize: 12, cursor: "pointer", letterSpacing: "0.06em", background: "transparent", border: "1px solid #1e293b", color: "#94a3b8" }}>
                          + Custom
                        </button>
                      </div>
                      {selected.map((sym, idx) => !known.includes(sym) && (
                        <div key={idx} style={{ display: "flex", gap: 6, marginTop: 4, alignItems: "center" }}>
                          <input type="text" placeholder="e.g. CL, RTY, YM..." value={sym}
                            onChange={e => { const next = [...selected]; next[idx] = e.target.value.toUpperCase(); f("instruments", next); }}
                            style={{ flex: 1 }} />
                          <button type="button" onClick={() => f("instruments", selected.filter((_, i) => i !== idx))}
                            style={{ background: "transparent", border: "1px solid #1e293b", color: "#94a3b8", padding: "6px 10px", borderRadius: 3, fontFamily: "inherit", fontSize: 11, cursor: "pointer" }}>✕</button>
                        </div>
                      ))}
                    </div>
                  );
                })() : (
                  <input
                    type="text"
                    placeholder="e.g. AAPL, TSLA, SPY, BTC… (stocks, ETFs, crypto)"
                    value={(form.instruments || []).join(", ")}
                    onChange={e => {
                      const raw = e.target.value;
                      const parsed = raw.split(/[,\s]+/).map(s => s.trim().toUpperCase()).filter(Boolean);
                      f("instruments", raw.trim() ? parsed : []);
                    }}
                    style={{ width: "100%", fontSize: 13 }}
                  />
                )}
              </div>

              {/* ── GROSS P&L ── */}
              <div>
                <label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>GROSS P&L ($)</label>
                <input type="number" placeholder={isProp ? "Auto on import" : "0.00"} value={form.pnl} onChange={e => f("pnl", e.target.value)}
                  style={{ width: "100%", fontSize: 16, fontWeight: 600, padding: "8px 12px", textAlign: "right" }} />
              </div>

            </div>
            {/* Fees + Net PnL + optional cash deposit row */}
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14, background: "#0a0e1a", border: "1px solid #1e293b", borderRadius: 4, padding: "8px 14px", flexWrap: "wrap" }}>
              <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", whiteSpace: "nowrap" }}>FEES ($)</div>
              <input type="number" min="0" placeholder="0.00" value={form.commissions} onChange={e => f("commissions", e.target.value)}
                style={{ width: 90, fontSize: 13, padding: "4px 8px" }} />
              <div style={{ width: "1px", height: 20, background: "#1e293b", margin: "0 4px" }} />
              <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", whiteSpace: "nowrap" }}>NET P&L</div>
              {(() => {
                const gross = parseFloat(form.pnl);
                const comm = parseFloat(form.commissions) || 0;
                if (isNaN(gross)) return <div style={{ fontSize: 14, color: "#64748b" }}>—</div>;
                const net = gross - comm;
                return <div style={{ fontSize: 15, color: net >= 0 ? "#4ade80" : "#f87171", fontWeight: 600 }}>{fmtPnl(net)}</div>;
              })()}
              {isPersonal && (<>
                <div style={{ width: "1px", height: 20, background: "#1e293b", margin: "0 4px" }} />
                <div style={{ fontSize: 9, color: "#4ade80", letterSpacing: "0.1em", textTransform: "uppercase", whiteSpace: "nowrap" }}>💵 CASH DEPOSIT ($)</div>
                <input type="number" min="0" placeholder="0.00" value={form.cashDeposit || ""} onChange={e => f("cashDeposit", e.target.value)}
                  style={{ width: 100, fontSize: 13, padding: "4px 8px" }} />
                <div style={{ fontSize: 9, color: "#64748b" }}>added to balance</div>
              </>)}
            </div>
            <div style={{ display: "flex", gap: 6, marginBottom: 22 }}>
              {TABS.map(t => (
                <button key={t.id} disabled={t.disabled}
                  onClick={() => !t.disabled && setTab(t.id)}
                  style={{ position: "relative", flex: 1, padding: "11px 0", borderRadius: 6, fontFamily: "inherit", fontSize: 12, cursor: t.disabled ? "not-allowed" : "pointer", transition: "all .15s", letterSpacing: "0.06em", opacity: t.disabled ? 0.35 : 1, overflow: "hidden", textAlign: "center",
                    background: tab === t.id ? "rgba(10,18,32,0.95)" : "transparent",
                    border: `1px solid ${tab === t.id ? "#1e3a5f" : "#1e293b"}`,
                    color: tab === t.id ? "#e2e8f0" : "#64748b" }}>
                  {tab === t.id && <span style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc)", borderRadius: "0 0 5px 5px" }} />}
                  {t.label.toUpperCase()}
                </button>
              ))}
            </div>

            {tab === "session" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

                {/* Sub-tab strip: SESSION | LESSONS */}
                <div style={{ display: "flex", gap: 6 }}>
                  {[
                    ["session", "📋", "SESSION", "hue-rotate(180deg) saturate(3) brightness(1.4)"],
                    ["lessons", "📖", "LESSONS",  "hue-rotate(30deg)  saturate(4) brightness(1.2)"],
                  ].map(([id, emoji, text, emojiFilter]) => (
                    <button key={id} onClick={() => setSessionInnerTab(id)}
                      style={{ padding: "8px 20px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.06em", transition: "all .15s", display: "flex", alignItems: "center", gap: 6,
                        background: sessionInnerTab === id ? "#1e3a5f" : "transparent",
                        border: `1px solid ${sessionInnerTab === id ? "#3b82f6" : "#1e293b"}`,
                        color: sessionInnerTab === id ? "#93c5fd" : "#94a3b8" }}>
                      <span style={{ fontSize: 13, filter: emojiFilter }}>{emoji}</span>
                      {text}
                    </button>
                  ))}
                </div>

                {/* ── SESSION sub-tab ── */}
                {sessionInnerTab === "session" && <>
                <div><label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8, display: "block" }}>MARKET BIAS</label><div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>{BIAS_OPTIONS.map(b => <button key={b} className={`pill ${form.bias === b ? "sel" : ""}`} onClick={() => f("bias", b)}>{b.toUpperCase()}</button>)}</div></div>
                <div><label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8, display: "block" }}>MOOD / MENTAL STATE <span style={{ color: "#64748b", fontWeight: 400, letterSpacing: 0 }}>· select all that apply</span></label><div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>{MOOD_OPTIONS.map(m => { const sel = (form.moods || []).includes(m); return <button key={m} className={`pill ${sel ? "sel" : ""}`} onClick={() => { const cur = form.moods || []; f("moods", sel ? cur.filter(x => x !== m) : [...cur, m]); }}>{m}</button>; })}</div></div>
                <div>
                  <label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8, display: "block" }}>SESSION MISTAKES <span style={{ color: "#64748b", fontWeight: 400, letterSpacing: 0 }}>· select all that apply</span></label>
                  <MistakesDropdown selected={form.sessionMistakes || []} onChange={v => f("sessionMistakes", v)} />
                </div>
                <div><label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8, display: "block" }}>SESSION GRADE / SYSTEM ADHERENCE SCORE</label><div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>{GRADE_OPTIONS.map(g => <button key={g} className={`pill ${form.grade === g ? "sel" : ""}`} onClick={() => f("grade", g)} style={form.grade === g ? { borderColor: gradeColor(g), background: "#0f172a", color: gradeColor(g) } : {}}>{g}</button>)}</div></div>
                {/* Feature 3: Dual Scores */}
                {(() => {
                  // Dynamic color scale: red(1) → amber(5) → green(10)
                  const scoreColor = (n) => {
                    if (n <= 3) return '#f87171';   // red
                    if (n <= 5) return '#fb923c';   // orange
                    if (n <= 6) return '#fbbf24';   // amber
                    if (n <= 7) return '#a3e635';   // yellow-green
                    if (n <= 8) return '#4ade80';   // green
                    return '#22d3ee';               // teal/bright for 9-10
                  };
                  const ScoreStrip = ({ label, fieldKey, subtitle }) => {
                    const val = form[fieldKey];
                    const derived = (() => {
                      const ex = Number(form.executionScore), dec = Number(form.decisionScore);
                      if (!Number.isFinite(ex) || !Number.isFinite(dec)) return null;
                      const composite = ex * 0.6 + dec * 0.4;
                      if (composite >= 9) return 'A+';
                      if (composite >= 8) return 'A';
                      if (composite >= 7) return 'B+';
                      if (composite >= 6) return 'B';
                      if (composite >= 5) return 'C';
                      if (composite >= 3) return 'D';
                      return 'F';
                    })();
                    return (
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 6 }}>
                          <label style={{ fontSize: 10, color: '#94a3b8', letterSpacing: '0.1em', textTransform: 'uppercase' }}>{label}</label>
                          <span style={{ fontSize: 9, color: '#64748b' }}>{subtitle}</span>
                        </div>
                        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                          {[1,2,3,4,5,6,7,8,9,10].map(n => {
                            const c = scoreColor(n);
                            const isSelected = val === n;
                            return (
                              <button key={n} onClick={() => f(fieldKey, isSelected ? null : n)}
                                style={{ width: 32, height: 32, borderRadius: 4, border: `1px solid ${isSelected ? c : '#1e293b'}`, background: isSelected ? c + '33' : 'transparent', color: isSelected ? c : '#64748b', fontSize: 12, cursor: 'pointer', fontFamily: 'inherit', fontWeight: isSelected ? 700 : 400, transition: 'all .12s' }}>
                                {n}
                              </button>
                            );
                          })}
                          {val !== null && val !== undefined && (
                            <button onClick={() => f(fieldKey, null)} style={{ padding: '0 8px', height: 32, borderRadius: 4, border: '1px solid #1e293b', background: 'transparent', color: '#64748b', fontSize: 10, cursor: 'pointer', fontFamily: 'inherit' }}>✕</button>
                          )}
                        </div>
                        {fieldKey === 'decisionScore' && derived && (
                          <div style={{ marginTop: 6, fontSize: 10, color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 6 }}>
                            Derived overall grade:
                            <button onClick={() => f('grade', derived)}
                              style={{ padding: '2px 8px', borderRadius: 3, border: `1px solid ${gradeColor(derived)}`, background: '#0f172a', color: gradeColor(derived), fontSize: 10, cursor: 'pointer', fontFamily: 'inherit', letterSpacing: '0.04em' }}>
                              {derived} → + Apply
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  };
                  return (
                    <div style={{ background: '#0a0e1a', border: '1px solid #1e293b', borderRadius: 6, padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 14 }}>
                      <div style={{ fontSize: 10, color: '#93c5fd', letterSpacing: '0.1em' }}>📊 PERFORMANCE SCORES <span style={{ color: '#94a3b8', fontWeight: 400 }}>· both optional</span></div>
                      <ScoreStrip label="Execution Score" fieldKey="executionScore" subtitle="Did I follow the plan? Entry/exit/sizing discipline?" />
                      <ScoreStrip label="Decision Quality" fieldKey="decisionScore" subtitle="Was the trade idea sound in context?" />
                    </div>
                  );
                })()}
                {/* Feature 4: Mistake Cost Attribution */}
                {(() => {
                  const activeMistakes = (form.sessionMistakes || []).filter(m => m !== NO_MISTAKES_OPTION);
                  if (activeMistakes.length === 0) return null;
                  const costs = form.mistakeCosts || {};
                  const total = activeMistakes.reduce((s, m) => s + (parseFloat(costs[m]) || 0), 0);
                  const netLoss = Math.abs(Math.min(0, parseFloat(form.pnl) || 0));
                  const highWarning = netLoss > 0 && total > netLoss * 3;
                  return (
                    <div style={{ background: '#0f0a00', border: '1px solid #451a03', borderRadius: 6, padding: '14px 16px' }}>
                      <div style={{ fontSize: 10, color: '#fb923c', letterSpacing: '0.1em', marginBottom: 12 }}>💸 MISTAKE COST ATTRIBUTION <span style={{ color: '#7c3d12', fontWeight: 400 }}>· optional, rough estimate ok</span></div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {activeMistakes.map(m => (
                          <div key={m} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                            <span style={{ fontSize: 11, color: '#e2e8f0', flex: 1 }}>{m}</span>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                              <span style={{ fontSize: 10, color: '#7c3d12' }}>$</span>
                              <input type="number" min="0" step="1" placeholder="0"
                                value={costs[m] !== null && costs[m] !== undefined ? costs[m] : ''}
                                onChange={e => {
                                  const raw = e.target.value;
                                  const n = raw === '' ? null : Math.max(0, parseFloat(raw) || 0);
                                  f('mistakeCosts', { ...costs, [m]: n });
                                }}
                                style={{ width: 90, fontSize: 13, padding: '4px 8px', textAlign: 'right' }} />
                            </div>
                          </div>
                        ))}
                      </div>
                      {total > 0 && (
                        <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid #451a03', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: 10, color: '#7c3d12', letterSpacing: '0.08em' }}>TOTAL ATTRIBUTED</span>
                          <span style={{ fontSize: 15, color: '#fb923c', fontWeight: 600 }}>${total.toFixed(0)}</span>
                        </div>
                      )}
                      {highWarning && (
                        <div style={{ marginTop: 8, fontSize: 10, color: '#fbbf24', background: 'rgba(251,191,36,0.08)', border: '1px solid rgba(251,191,36,0.2)', borderRadius: 4, padding: '6px 10px' }}>
                          ⚠ Attributed cost exceeds 3× today's loss — double-check this
                        </div>
                      )}
                      <div style={{ marginTop: 10 }}>
                        <input type="text" placeholder="Optional note on cost breakdown…" value={form.mistakeCostNotes || ''} onChange={e => f('mistakeCostNotes', e.target.value)} style={{ width: '100%', fontSize: 11, padding: '6px 10px', boxSizing: 'border-box' }} />
                      </div>
                    </div>
                  );
                })()}
                <div><label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>MARKET / TAPE NOTES</label><textarea rows={4} placeholder="Price action, key levels, macro context, structure..." value={form.marketNotes} onChange={e => f("marketNotes", e.target.value)} onPaste={autoBulletPaste("marketNotes")} /></div>
                <div><label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>RULES FOLLOWED / BROKEN</label><textarea rows={3} placeholder="Did you follow your trading plan?" value={form.rules} onChange={e => f("rules", e.target.value)} onPaste={autoBulletPaste("rules")} /></div>
                </> /* end session sub-tab */}

                {/* ── LESSONS sub-tab ── */}
                {sessionInnerTab === "lessons" && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                    <div><label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>LESSONS LEARNED TODAY</label><textarea rows={5} placeholder="What did the market teach you today?" value={form.lessonsLearned} onChange={e => f("lessonsLearned", e.target.value)} onPaste={autoBulletPaste("lessonsLearned")} /></div>
                    <div><label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>MISTAKES TO AVOID</label><textarea rows={4} placeholder="Be specific. e.g. 'Don't trade after 2 back-to-back losses'" value={form.mistakes} onChange={e => f("mistakes", e.target.value)} onPaste={autoBulletPaste("mistakes")} /></div>
                    <div><label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>AREAS FOR IMPROVEMENT</label><textarea rows={3} placeholder="Entry timing? Holding winners? Cutting losers?" value={form.improvements} onChange={e => f("improvements", e.target.value)} onPaste={autoBulletPaste("improvements")} /></div>
                    <div><label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>BEST TRADE OF THE DAY</label><textarea rows={3} placeholder="What setup worked? What went right?" value={form.bestTrade} onChange={e => f("bestTrade", e.target.value)} onPaste={autoBulletPaste("bestTrade")} /></div>
                    <div><label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>WORST TRADE / MISTAKE</label><textarea rows={3} placeholder="What went wrong? Revenge trade? Poor sizing?" value={form.worstTrade} onChange={e => f("worstTrade", e.target.value)} onPaste={autoBulletPaste("worstTrade")} /></div>
                    <div style={{ border: "1px solid #1e3a5f", borderRadius: 6, padding: "14px 16px", background: "#0a1628" }}>
                      <label style={{ fontSize: 10, color: "#93c5fd", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>ONE RULE TO REINFORCE TOMORROW?</label>
                      <textarea rows={2} placeholder="e.g. 'No trades during the Afternoon Deadzone' or 'Size down after first loss'" value={form.reinforceRule || ""} onChange={e => f("reinforceRule", e.target.value)} onPaste={autoBulletPaste("reinforceRule")} />
                    </div>
                  </div>
                )}
              </div>
            )}

            {tab === "import" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

                {/* Header card */}
                <div style={{ background: "#070d1a", border: "1px solid #1e3a5f", borderRadius: 6, padding: "16px 18px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                    <div style={{ fontSize: 18 }}>🤖</div>
                    <div>
                      <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.12em", fontWeight: 600 }}>AI SMART PARSER</div>
                      <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>Paste raw data from any broker — Claude will identify the format, match round-trips, and calculate P&L automatically</div>
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {["Tradovate", "Interactive Brokers", "Questrade", "NinjaTrader", "Rithmic", "Any Format"].map(b => (
                      <span key={b} style={{ fontSize: 9, color: "#3b82f6", border: "1px solid #1e293b", padding: "2px 8px", borderRadius: 12, letterSpacing: "0.08em" }}>{b}</span>
                    ))}
                  </div>
                </div>

                {/* CSV Upload */}
                <div>
                  <label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>UPLOAD CSV FILE</label>
                  <div
                    style={{ border: "2px dashed #1e3a5f", borderRadius: 6, padding: "20px 24px", textAlign: "center", cursor: "pointer", background: csvFileName ? "rgba(59,130,246,0.05)" : "transparent", transition: "all .15s" }}
                    onClick={() => csvInputRef.current && csvInputRef.current.click()}
                    onDragOver={e => { e.preventDefault(); e.currentTarget.style.borderColor = "#3b82f6"; e.currentTarget.style.background = "rgba(59,130,246,0.08)"; }}
                    onDragLeave={e => { e.currentTarget.style.borderColor = "#1e3a5f"; e.currentTarget.style.background = csvFileName ? "rgba(59,130,246,0.05)" : "transparent"; }}
                    onDrop={e => { e.preventDefault(); e.currentTarget.style.borderColor = "#1e3a5f"; e.currentTarget.style.background = "transparent"; const droppedFile = e.dataTransfer.files[0]; if (droppedFile) handleCsvUpload(droppedFile); }}>
                    <input ref={csvInputRef} type="file" accept=".csv,.txt,.tsv" style={{ display: "none" }} onChange={e => { handleCsvUpload(e.target.files[0]); e.target.value = ""; }} />
                    {csvFileName ? (
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10 }}>
                        <span style={{ fontSize: 16 }}>📄</span>
                        <div style={{ textAlign: "left" }}>
                          <div style={{ fontSize: 12, color: "#93c5fd", fontWeight: 500 }}>{csvFileName}</div>
                          <div style={{ fontSize: 10, color: "#4ade80", marginTop: 2 }}>✓ File loaded — ready to parse</div>
                        </div>
                        <button onClick={e => { e.stopPropagation(); setCsvFileName(""); setImportRaw(""); setImportSuccess(false); }}
                          style={{ background: "transparent", border: "none", color: "#475569", cursor: "pointer", fontSize: 13, padding: "2px 6px", marginLeft: 8 }}>✕</button>
                      </div>
                    ) : (
                      <div>
                        <div style={{ fontSize: 22, marginBottom: 8, opacity: 0.5 }}>📂</div>
                        <div style={{ fontSize: 12, color: "#64748b" }}>Click to upload or drag & drop your CSV</div>
                        <div style={{ fontSize: 10, color: "#334155", marginTop: 4 }}>Supports .csv · .txt · .tsv from any broker</div>
                      </div>
                    )}
                  </div>
                </div>

                {/* OR divider */}
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <div style={{ flex: 1, height: 1, background: "#1e293b" }} />
                  <span style={{ fontSize: 10, color: "#334155", letterSpacing: "0.12em" }}>OR PASTE BELOW</span>
                  <div style={{ flex: 1, height: 1, background: "#1e293b" }} />
                </div>

                {/* Paste textarea */}
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                    <label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase" }}>PASTE BROKER DATA</label>
                    {detectedFormat && <span style={{ fontSize: 9, color: "#3b82f6", background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.2)", padding: "2px 8px", borderRadius: 12 }}>✓ {detectedFormat}</span>}
                  </div>
                  <textarea rows={10} className="no-autoresize" placeholder={"Paste your broker export here — any format works:\n\n• Tradovate Orders CSV\n• Interactive Brokers Flex Query\n• Questrade Activity Export\n• NinjaTrader Trade Log\n• Any tab or comma separated fills data\n\nJust paste and hit the button — AI handles the rest."} value={importRaw} onChange={e => { setImportRaw(e.target.value); setCsvFileName(""); setDetectedFormat(""); setImportSuccess(false); setAiParseError(""); }} style={{ fontFamily: "monospace", fontSize: 11 }} />
                </div>

                {/* CSV confirm replace dialog */}
                {csvConfirmPending && (
                  <div style={{ background: "#0a0e1a", border: "1px solid #92400e", borderRadius: 6, padding: "14px 16px" }}>
                    <div style={{ fontSize: 12, color: "#fbbf24", marginBottom: 10 }}>⚠ Trades already parsed for this entry. Replace with <strong>{csvConfirmPending.name}</strong>?</div>
                    <div style={{ display: "flex", gap: 8 }}>
                      <button onClick={confirmCsvReplace} style={{ background: "#1d4ed8", color: "white", border: "none", padding: "8px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 12, cursor: "pointer", letterSpacing: "0.06em" }}>YES, REPLACE</button>
                      <button onClick={() => setCsvConfirmPending(null)} style={{ background: "transparent", border: "1px solid #334155", color: "#94a3b8", padding: "8px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 12, cursor: "pointer", letterSpacing: "0.06em" }}>CANCEL</button>
                    </div>
          </div>
          )}

                {/* Errors */}
                {(importError || aiParseError) && (
                  <div style={{ color: "#f87171", fontSize: 12, background: "#1f060622", border: "1px solid #450a0a", padding: "10px 14px", borderRadius: 4 }}>
                    ⚠ {importError || aiParseError}
          </div>
          )}

                {/* Success */}
                {importSuccess && (
                  <div style={{ color: "#4ade80", fontSize: 12, background: "#052e1622", border: "1px solid #166534", padding: "12px 14px", borderRadius: 4 }}>
                    <div>✓ <strong>{form.parsedTrades.length} trades</strong> parsed · Total P&L: <strong>{fmtPnl(form.pnl)}</strong></div>
                    <div style={{ fontSize: 10, color: "#4ade8099", marginTop: 4 }}>Analysis tab is now unlocked →</div>
          </div>
          )}

                {/* Parse button */}
                <button
                  onClick={handleImport}
                  disabled={aiParsing || !importRaw.trim()}
                  style={{ width: "fit-content", background: aiParsing ? "#1e3a5f" : "#1d4ed8", color: "white", border: "none", padding: "11px 24px", borderRadius: 4, fontFamily: "inherit", fontSize: 12, cursor: aiParsing ? "wait" : "pointer", letterSpacing: "0.08em", display: "flex", alignItems: "center", gap: 8, opacity: !importRaw.trim() ? 0.5 : 1, transition: "all .2s" }}>
                  {aiParsing ? (
                    <><span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>⟳</span> ANALYSING WITH AI…</>
                  ) : (
                    <>🤖 SMART PARSE & ANALYSE →</>
                  )}
                </button>

                {/* Tip */}
                <div style={{ fontSize: 10, color: "#1e3a5f", lineHeight: 1.6 }}>
                  💡 <strong style={{ color: "#64748b" }}>Tip:</strong> Export the <strong style={{ color: "#64748b" }}>Orders</strong> or <strong style={{ color: "#64748b" }}>Trade History</strong> report from your broker. Cancelled and rejected orders are automatically ignored.
                </div>

                {/* ── TRADE LOG RECAP — appears after successful import ── */}
                {form.parsedTrades?.length > 0 && (() => {
                  const trades = form.parsedTrades;
                  // Must use calcAnalytics for the full `a` object — AnalyticsPanel accesses
                  // a.winRate, a.profitFactor, a.byDirection, a.byDuration, a.equityCurve, etc.
                  // Passing a partial object crashes the component (white screen).
                  const a = calcAnalytics(trades, aiCfg?.tzLock !== false) || {};
                  return (
                    <div style={{ marginTop: 4 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                        <div style={{ height: 1, flex: 1, background: "linear-gradient(90deg, #1e3a5f, transparent)" }} />
                        <div style={{ fontSize: 9, color: "#3b82f6", letterSpacing: "0.2em" }}>✓ {trades.length} TRADES IMPORTED — REVIEW BELOW</div>
                        <div style={{ height: 1, flex: 1, background: "linear-gradient(90deg, transparent, #1e3a5f)" }} />
                      </div>
                      <AnalyticsPanel
                        a={a}
                        trades={trades}
                        pnlColor={pnlColor}
                        fmtPnl={fmtPnl}
                        analyticsTab={analyticsTab}
                        setAnalyticsTab={setAnalyticsTab}
                        totalFees={parseFloat(form.commissions) || 0}
                        rawCsvFile={form.rawCsvFile || null}
                      />
                    </div>
                  );
                })()}

              </div>
            )}

            {tab === "analysis" && analytics && (
              <AnalyticsPanel a={analytics} trades={form.parsedTrades} pnlColor={pnlColor} fmtPnl={fmtPnl} analyticsTab={analyticsTab} setAnalyticsTab={setAnalyticsTab} totalFees={parseFloat(form.commissions) || 0} rawCsvFile={form.rawCsvFile || null} />
            )}



            {tab === "tomorrow" && (
              <div><label style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>PLAN FOR TOMORROW</label><textarea rows={7} placeholder="Key levels, economic events, intended setups, max loss for the day, goals..." value={form.tomorrow} onChange={e => f("tomorrow", e.target.value)} onPaste={autoBulletPaste("tomorrow")} /></div>
            )}

            <ChartScreenshotZone
              screenshots={form.chartScreenshots || []}
              onChange={shots => f("chartScreenshots", shots)}
            />

            <div style={{ marginTop: 28, display: "flex", gap: 12, justifyContent: "flex-end" }}>
              <button onClick={() => { setView("list"); setActiveEntry(null); }} style={{ background: "transparent", color: "#94a3b8", border: "1px solid #1e293b", padding: "10px 22px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer" }}>CANCEL</button>
              <button onClick={handleSave} disabled={saving} style={{ background: "#1d4ed8", color: "white", border: "none", padding: "10px 22px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer" }}>{saving ? "SAVING..." : saved ? "✓ SAVED" : activeEntry ? "✓ UPDATE" : "+ APPLY"}</button>
            </div>
          </div>
        )}

        {/* DETAIL */}
        {view === "detail" && activeEntry && (() => {
          const cfg = activeJournal?.config;
          const dangerLine = cfg ? -(isProp ? (cfg.phase === "eval" ? cfg.eval?.maxLossLimit : cfg.funded?.maxLossLimit) || 0 : 0) : null;
          const priorPlan = (() => {
            const prior = [...entries].filter(e => e.date < activeEntry.date).sort((a, b) => b.date.localeCompare(a.date))[0];
            return prior ? { plan: prior.tomorrow, reinforceRule: prior.reinforceRule, date: prior.date } : null;
          })();
          return (
            <DetailView entry={activeEntry} a={detailAnalytics} pnlColor={pnlColor} fmtPnl={fmtPnl} gradeColor={gradeColor}
              dangerLine={dangerLine && dangerLine < 0 ? dangerLine : null}
              priorPlan={priorPlan}
              onRecap={(mode) => { setRecapInitMode(mode || "weekly"); setView("recap"); }} ai={aiCfg} onUpdateEntry={(id, patch) => {
                const updated = entries.map(e => e.id === id ? { ...e, ...patch } : e);
                saveEntries(updated);
                if (activeEntry?.id === id) setActiveEntry(prev => prev ? { ...prev, ...patch } : prev);
              }} />
          );
        })()}

        {/* AI RECAP */}
        {view === "recap" && (
          <AIRecapView entries={entries} netPnl={netPnl} fmtPnl={fmtPnl} pnlColor={pnlColor} initMode={recapInitMode} ai={aiCfg} activeJournal={activeJournal} propStatus={propStatus} />
        )}

        {/* QUOTES */}
        {view === "quotes" && (
          <QuotesView />
        )}

        {/* REFERENCE */}
        {view === "reference" && (
          <ReferenceView />
        )}

        {/* PROP DASHBOARD */}
        {view === "propdash" && <PropDashInner journals={journals} entries={entries} activeJournalId={activeJournalId} activeJournal={activeJournal} propStatus={propStatus} propJournals={propJournals} allPropEntriesMap={allPropEntriesMap} saveJournalsMeta={saveJournalsMeta} netPnl={netPnl} pnlColor={pnlColor} />}

        {/* PERSONAL BALANCE BANNER — shown on list view for personal accounts */}
        {view === "list" && isPersonal && personalBalance && personalBalance.currentBalance > 0 && (
          <div style={{ background: "#0a0e1a", border: "1px solid #1e3a5f", borderRadius: 6, overflow: "hidden", position: "relative", marginBottom: 16 }}>
            <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc)" }} />
            <div style={{ padding: "14px 18px", background: "#0a1628", borderBottom: "1px solid #1e293b", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
              {/* Emoji rendered separately so gradient doesn't swallow it */}
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 16 }}>💼</span>
                <span style={{ fontSize: 15, fontWeight: 700, letterSpacing: "0.1em", background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>ACCOUNT BALANCE</span>
              </div>
              <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
                {[
                  { l: "STARTING", v: `$${personalBalance.startingBalance.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, c: "#475569" },
                  { l: "NET P&L", v: fmtPnl(totalPnL), c: pnlColor(totalPnL) },
                  { l: "CURRENT BALANCE", v: `${personalBalance.currentBalance >= personalBalance.startingBalance ? "+" : "-"}$${personalBalance.currentBalance.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, c: personalBalance.currentBalance > personalBalance.startingBalance ? "#4ade80" : personalBalance.currentBalance < personalBalance.startingBalance ? "#f87171" : "#e2e8f0" },
                ].map(s => (
                  <div key={s.l} style={{ textAlign: "right" }}>
                    <div style={{ fontSize: 9, letterSpacing: "0.1em", marginBottom: 3, background: "linear-gradient(135deg,#38bdf8,#818cf8,#c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>{s.l}</div>
                    <div style={{ fontSize: 15, color: s.c, fontWeight: 600 }}>{s.v}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

      </div>
    </div>

    {/* Feature 2: Trade parse review modal */}
    {parseReviewData && (
      <ParseReviewModal
        data={parseReviewData}
        onConfirm={(trades, allowOvernight, logEntry) => applyParsedTrades(trades, parseReviewData.detectedFmt, logEntry)}
        onCancel={() => setParseReviewData(null)}
      />
    )}
    </>
  );
}
