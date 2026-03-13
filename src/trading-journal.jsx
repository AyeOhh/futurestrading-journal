import { useState, useEffect, useMemo, useRef } from "react";

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
const MOOD_OPTIONS = ["Focused 🎯", "Disciplined 🧠", "Confident 💪", "Patient 🧘", "Calm 😌", "Excited ⚡", "Tired 😴", "Anxious 😬", "Fearful 😰", "Distracted 😵", "Frustrated 😤", "Overconfident 😎", "Greedy 🤑", "Revenge mode 😡"];
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
        const clean = text.replace(/```json|```/g, '').trim();
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
        const clean = text.replace(/```json|```/g, '').trim();
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
      { id: 'gemini-2.5-pro',        label: 'Gemini 2.5 Pro'        },
      { id: 'gemini-2.5-flash',      label: 'Gemini 2.5 Flash'      },
      { id: 'gemini-2.0-flash',      label: 'Gemini 2.0 Flash'      },
      { id: 'gemini-1.5-pro',        label: 'Gemini 1.5 Pro'        },
      { id: 'gemini-1.5-flash',      label: 'Gemini 1.5 Flash'      },
      { id: 'gemini-1.5-flash-8b',   label: 'Gemini 1.5 Flash 8B'   },
    ],
    async request(ai, { messages, max_tokens = 600, model, timeoutMs = 22000 }) {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), timeoutMs);
      // Convert OpenAI-style messages → Gemini contents format
      const contents = messages.map(m => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }],
      }));
      const mdl = model || ai.model;
      try {
        const res = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/${mdl}:generateContent?key=${ai.apiKey}`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: ctrl.signal,
            body: JSON.stringify({ contents, generationConfig: { maxOutputTokens: max_tokens } }),
          }
        );
        if (!res.ok) throw new Error(`API error ${res.status}`);
        const data = await res.json();
        if (data?.error) throw new Error(data.error.message || 'API returned an error');
        const text = data.candidates?.[0]?.content?.parts?.map(p => p.text || '').join('\n') || '';
        const clean = text.replace(/```json|```/g, '').trim();
        if (!clean) throw new Error('Empty response from API');
        return clean;
      } finally { clearTimeout(t); }
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
        const clean = text.replace(/```json|```/g, '').trim();
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
    const raw = window?.localStorage?.getItem(AI_CACHE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return (parsed && typeof parsed === 'object') ? parsed : {};
  } catch { return {}; }
};

const saveAiCache = (cache) => {
  try { window?.localStorage?.setItem(AI_CACHE_KEY, JSON.stringify(cache)); } catch {}
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
    const raw = window?.localStorage?.getItem(AI_SETTINGS_KEY);
    if (!raw) return { ...DEFAULT_AI_SETTINGS };
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_AI_SETTINGS, ...parsed };
  } catch {
    return { ...DEFAULT_AI_SETTINGS };
  }
};

const saveAiSettings = (s) => {
  try { window?.localStorage?.setItem(AI_SETTINGS_KEY, JSON.stringify(s)); } catch {}
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
      if (['no_key','disabled','unauthorized','forbidden'].includes(f.code)) throw err;
      if (attempt < maxAttempts) await sleep(450 * attempt);
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
const parseTradesWithAI = async (raw, ai) => {
  const brokerHint = BROKER_PRESETS[ai?.brokerPreset || 'none']?.hint || '';
  const prompt = `You are a futures trade parser. Given raw broker export data, extract all COMPLETED round-trip trades.
${brokerHint ? `
BROKER CONTEXT: ${brokerHint}
` : ''}
RULES:
1. Skip cancelled, rejected, or open orders — only include fully filled round-trip trades
2. Match buys to sells by symbol and time proximity (FIFO). A round-trip = one entry + one exit
3. For futures, calculate P&L as: (exitPrice - entryPrice) × qty × multiplier for longs, reversed for shorts
4. Use these multipliers: MES=$5/pt, ES=$50/pt, MNQ=$2/pt, NQ=$20/pt, MYM=$0.50/pt, YM=$5/pt, MGC=$10/pt, GC=$100/pt, MCL=$100/pt, CL=$1000/pt, RTY=$10/pt, M2K=$10/pt
5. If the data already has P&L calculated, use it directly
6. For partial fills on the same order, combine them into one trade
7. For "Exit Market" orders, these are closing trades — match to the most recent open position in that symbol

CRITICAL OUTPUT RULE: Return ONLY a raw JSON array. No markdown, no code fences, no \`\`\`json, no \`\`\`, no explanation before or after. Your response must begin with [ and end with ]. Any wrapper will break the parser. Each trade object must have exactly these fields:
{
  "symbol": "MNQH6",
  "qty": 5,
  "direction": "long",
  "buyPrice": 24904.00,
  "buyTime": "03/05/2026 15:16:35",
  "sellTime": "03/05/2026 15:16:56",
  "sellPrice": 24918.00,
  "pnl": 140.00,
  "duration": "21s",
  "durationSecs": 21,
  "orderType": "LMT",
  "commission": 1.24,
  "multiplier": 2,
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
  // ── JSON repair pipeline ──
  let jsonStr = clean.trim();
  // 1. Strip markdown fences
  jsonStr = jsonStr.replace(/^```(?:json)?\s*/i, '').replace(/\s*```\s*$/, '');
  // 2. Extract just the array
  const arrStart = jsonStr.indexOf('[');
  const arrEnd = jsonStr.lastIndexOf(']');
  if (arrStart === -1 || arrEnd === -1 || arrEnd <= arrStart) throw new Error("No JSON array found in AI response");
  jsonStr = jsonStr.slice(arrStart, arrEnd + 1);
  // 3. Fix common AI JSON mistakes
  jsonStr = jsonStr
    .replace(/,\s*([}\]])/g, '$1')          // trailing commas before } or ]
    .replace(/([{,]\s*)(\w+)\s*:/g, '$1"$2":') // unquoted keys → quoted
    .replace(/:\s*'([^']*)'/g, ': "$1"')      // single-quoted values → double-quoted
    .replace(/[\u0000-\u001F]/g, ' ');        // strip control chars that break JSON
  // 4. If still broken, recover by trimming to last complete object
  try {
    JSON.parse(jsonStr);
  } catch {
    const lastClose = jsonStr.lastIndexOf('}');
    if (lastClose > 0) {
      jsonStr = jsonStr.slice(0, lastClose + 1);
      // Remove trailing comma if present
      jsonStr = jsonStr.replace(/,\s*$/, '');
      jsonStr = '[' + jsonStr.slice(jsonStr.indexOf('{')) + ']';
    }
  }
  const trades = JSON.parse(jsonStr);
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
  const spaceIdx = str.indexOf(" ");
  if (spaceIdx === -1) return "Unknown";
  const timePart = str.slice(spaceIdx + 1).trim();
  const timeLower = timePart.toLowerCase();
  const colonParts = timePart.split(":");
  if (colonParts.length < 2) return "Unknown";
  let h = parseInt(colonParts[0]);
  const mn = parseInt(colonParts[1]);
  if (isNaN(h) || isNaN(mn)) return "Unknown";
  if (timeLower.includes("pm")) { if (h !== 12) h += 12; }
  else if (timeLower.includes("am")) { if (h === 12) h = 0; }
  let t = h * 60 + mn;
  if (tzLock) t = ((t - getEstOffsetMinutes()) % 1440 + 1440) % 1440;
  if (t >= 1080) return "Asian Session (6PM–12AM)";
  if (t < 570)   return "London Session (12AM–9:30AM)";
  if (t < 720)   return "NY Open (9:30AM–12PM)";
  if (t < 900)   return "Afternoon Deadzone (12–3PM)";
  if (t < 960)   return "Power Hour (3–4PM)";
  return "After Hours (4–6PM)";
};

const calcAnalytics = (trades, tzLock = false) => {
  if (!trades || trades.length === 0) return null;
  const winners = trades.filter(t => t.pnl > 0);
  const losers = trades.filter(t => t.pnl < 0);
  const totalPnL = trades.reduce((s, t) => s + t.pnl, 0);
  const avgWin = winners.length ? winners.reduce((s, t) => s + t.pnl, 0) / winners.length : 0;
  const avgLoss = losers.length ? losers.reduce((s, t) => s + t.pnl, 0) / losers.length : 0;
  const winRate = trades.length ? (winners.length / trades.length) * 100 : 0;
  const grossWin = winners.reduce((s, t) => s + t.pnl, 0);
  const grossLoss = Math.abs(losers.reduce((s, t) => s + t.pnl, 0));
  const profitFactor = grossLoss > 0 ? grossWin / grossLoss : null;
  const largestWin = Math.max(...trades.map(t => t.pnl));
  const largestLoss = Math.min(...trades.map(t => t.pnl));
  const avgQty = trades.reduce((s, t) => s + t.qty, 0) / trades.length;

  let running = 0;
  const equityCurve = trades.map((t) => { running += t.pnl; return { pnl: running, trade: t }; });

  let runPeak = 0, maxDD = 0;
  for (const pt of equityCurve) {
    if (pt.pnl > runPeak) runPeak = pt.pnl;
    const dd = runPeak - pt.pnl;
    if (dd > maxDD) maxDD = dd;
  }

  const bySymbol = {};
  for (const t of trades) {
    if (!bySymbol[t.symbol]) bySymbol[t.symbol] = { trades: 0, pnl: 0, wins: 0 };
    bySymbol[t.symbol].trades++;
    bySymbol[t.symbol].pnl += t.pnl;
    if (t.pnl > 0) bySymbol[t.symbol].wins++;
  }

  const bySession = {};
  for (const t of trades) {
    const sess = getSession(t.sellTime, tzLock);
    if (!bySession[sess]) bySession[sess] = { trades: 0, pnl: 0, wins: 0 };
    bySession[sess].trades++;
    bySession[sess].pnl += t.pnl;
    if (t.pnl > 0) bySession[sess].wins++;
  }

  let maxConsecWins = 0, maxConsecLoss = 0, curW = 0, curL = 0;
  for (const t of trades) {
    if (t.pnl > 0) { curW++; curL = 0; maxConsecWins = Math.max(maxConsecWins, curW); }
    else if (t.pnl < 0) { curL++; curW = 0; maxConsecLoss = Math.max(maxConsecLoss, curL); }
    else { curW = 0; curL = 0; }
  }

  // Long vs Short breakdown
  const byDirection = { long: { trades: 0, pnl: 0, wins: 0, losses: 0, grossWin: 0, grossLoss: 0 }, short: { trades: 0, pnl: 0, wins: 0, losses: 0, grossWin: 0, grossLoss: 0 } };
  for (const t of trades) {
    const dir = t.direction === "short" ? "short" : "long";
    byDirection[dir].trades++;
    byDirection[dir].pnl += t.pnl;
    if (t.pnl > 0) { byDirection[dir].wins++; byDirection[dir].grossWin += t.pnl; }
    else if (t.pnl < 0) { byDirection[dir].losses++; byDirection[dir].grossLoss += Math.abs(t.pnl); }
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
    b.trades++; b.pnl += t.pnl; b.totalSecs += secs;
    if (t.pnl > 0) { b.wins++; b.grossWin += t.pnl; }
    else if (t.pnl < 0) { b.losses++; b.grossLoss += Math.abs(t.pnl); }
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
    const wins = arr.filter(t => t.pnl > 0);
    const pnl = arr.reduce((s, t) => s + (t.pnl || 0), 0);
    const avgPnl = total ? pnl / total : 0;
    const avgQty = total ? arr.reduce((s, t) => s + (Number(t.qty) || 0), 0) / total : 0;
    const winRate = total ? (wins.length / total) * 100 : 0;
    return { total, wins: wins.length, pnl, avgPnl, avgQty, winRate };
  };

  const afterLossTrades = ordered.filter((t, i) => i > 0 && (ordered[i - 1]?.pnl || 0) < 0);
  const afterWinTrades  = ordered.filter((t, i) => i > 0 && (ordered[i - 1]?.pnl || 0) > 0);
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
    byOrderType[ot].trades++;
    byOrderType[ot].pnl += t.pnl;
    if (t.pnl > 0) { byOrderType[ot].wins++; byOrderType[ot].grossWin += t.pnl; }
    else if (t.pnl < 0) { byOrderType[ot].grossLoss += Math.abs(t.pnl); }
  }

  // Commission totals
  const totalCommission = trades.reduce((s, t) => s + (t.commission || 0), 0);

  return {
    total: trades.length, winners: winners.length, losers: losers.length, breakeven: trades.filter(t => t.pnl === 0).length,
    totalPnL, avgWin, avgLoss, winRate, profitFactor,
    largestWin: winners.length ? Math.max(...winners.map(t => t.pnl)) : 0,
    largestLoss: losers.length ? Math.min(...losers.map(t => t.pnl)) : 0,
    equityCurve, maxDD, bySymbol, bySession,
    maxConsecWins, maxConsecLoss, avgQty,
    byDirection, byDuration, DURATION_BUCKETS,
    avgWinDuration, avgLossDuration,
    afterLoss, afterWin, first3, rest,
    byOrderType, totalCommission,
    // R-based metrics — use avg loss as 1R baseline
    avgWinR: (avgLoss !== 0 && winners.length) ? avgWin / Math.abs(avgLoss) : null,
    avgLossR: (avgLoss !== 0 && losers.length) ? avgLoss / Math.abs(avgLoss) : null,
    netR: avgLoss !== 0 ? totalPnL / Math.abs(avgLoss) : null,
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
  rawTradeData: "", parsedTrades: [], cashDeposit: "",
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
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={lineColor} stopOpacity="0.18" />
          <stop offset="100%" stopColor={lineColor} stopOpacity="0" />
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
      {/* Area fill */}
      <path d={fillPath} fill={`url(#${gradientId})`} />
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
      {/* Line */}
      <polyline points={pts} fill="none" stroke={lineColor} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
      {/* Dots (optional) */}
      {dots && values.map((v, i) => (
        <circle key={i} cx={toX(i)} cy={toY(v)} r="4"
          fill={v >= 0 ? "#4ade80" : "#f87171"} stroke="#0f1729" strokeWidth="2" />
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
      <div style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 10 }}>CHART SCREENSHOTS</div>

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

function AnalyticsPanel({ a, trades, pnlColor, fmtPnl, analyticsTab, setAnalyticsTab, totalFees = 0, dangerLine = null }) {
  const ATABS = ["overview", "by session", "trade log"];
  const netTotal = a.totalPnL - totalFees;
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
              { l: "TOTAL FEES", v: totalFees > 0 ? `-$${totalFees.toFixed(2)}` : "—", c: totalFees > 0 ? "#f87171" : "#94a3b8" },
              { l: "NET P&L", v: fmtPnl(netTotal), c: pnlColor(netTotal), highlight: true },
              { l: "LARGEST WIN", v: a.largestWin ? fmtPnl(a.largestWin) : "—", c: "#4ade80" },
              { l: "LARGEST LOSS", v: a.largestLoss ? fmtPnl(a.largestLoss) : "—", c: "#f87171" },
              { l: "MAX DRAWDOWN", v: `$${a.maxDD.toFixed(2)}`, c: "#f87171" },
            ].map(s => (
              <div key={s.l} style={{ background: s.highlight ? "#0f1a2e" : "#0f1729", border: `1px solid ${s.highlight ? "#1e3a5f" : "#1e293b"}`, borderRadius: 4, padding: "10px 12px" }}>
                <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 4 }}>{s.l}</div>
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
              { l: "PROFIT FACTOR", v: a.profitFactor ? a.profitFactor.toFixed(2) : "—", c: a.profitFactor >= 1 ? "#4ade80" : "#f87171" },
              { l: "AVG CONTRACT SIZE", v: `${a.avgQty.toFixed(1)} cts`, c: "#e2e8f0" },
              { l: "AVG WIN", v: fmtPnl(a.avgWin), c: "#4ade80" },
              { l: "AVG LOSS", v: fmtPnl(a.avgLoss), c: "#f87171" },
              { l: "MAX WIN STREAK", v: `${a.maxConsecWins} trades`, c: "#4ade80" },
              { l: "MAX LOSS STREAK", v: `${a.maxConsecLoss} trades`, c: "#f87171" },
              { l: "MAX DRAWDOWN", v: `$${a.maxDD.toFixed(2)}`, c: "#f87171" },
            ].map(s => (
              <div key={s.l} style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "10px 12px" }}>
                <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 4 }}>{s.l}</div>
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
                            ? `${card.data.avgPnl >= 0 ? '+' : ''}$${Math.abs(card.data.avgPnl).toFixed(2)}/trade`
                            : `${card.data.pnl >= 0 ? '+' : ''}$${Math.abs(card.data.pnl).toFixed(2)}`}
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

            {totalFees > 0 && (
              <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid #1e293b', display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12 }}>
                <div>
                  <div style={{ fontSize: 9, color: '#94a3b8', letterSpacing: '0.1em', marginBottom: 4 }}>FEES IMPACT</div>
                  <div style={{ fontSize: 12, color: '#e2e8f0' }}>${(totalFees / Math.max(a.total || 1, 1)).toFixed(2)} per trade</div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: 12, color: netTotal >= 0 ? '#4ade80' : '#f87171', fontWeight: 600 }}>{fmtPnl(netTotal)} net</div>
                  <div style={{ fontSize: 10, color: '#64748b' }}>{a.totalPnL !== 0 ? `${Math.min(100, Math.abs(totalFees / a.totalPnL) * 100).toFixed(1)}% of gross P&L` : ''}</div>
                </div>
              </div>
            )}
          </div>

          {/* Performance by Time of Day */}
          {(() => {
            const sessions = ["Asian Session (6PM–12AM)", "London Session (12AM–9:30AM)", "NY Open (9:30AM–12PM)", "Afternoon Deadzone (12–3PM)", "Power Hour (3–4PM)", "After Hours (4–6PM)"];
            const sessionData = sessions.map(s => ({ name: s, ...(a.bySession[s] || { trades: 0, pnl: 0, wins: 0 }) })).filter(s => s.trades > 0);
            if (sessionData.length === 0) return null;
            const maxAbs = Math.max(...sessionData.map(s => Math.abs(s.pnl)), 1);
            return (
              <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px" }}>
                <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 14 }}>PERFORMANCE BY TIME OF DAY</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  {sessionData.map(s => {
                    const wr = Math.round(s.wins / s.trades * 100);
                    const barW = Math.abs(s.pnl) / maxAbs * 100;
                    const shortName = s.name === "Asian Session (6PM–12AM)" ? "Asian Session" : s.name === "London Session (12AM–9:30AM)" ? "London Session" : s.name === "NY Open (9:30AM–12PM)" ? "NY Open" : s.name === "Afternoon Deadzone (12–3PM)" ? "Deadzone" : s.name === "Power Hour (3–4PM)" ? "Power Hour" : "After Hours";
                    const timeLabel = s.name === "Asian Session (6PM–12AM)" ? "6:00 PM – 12:00 AM EST" : s.name === "London Session (12AM–9:30AM)" ? "12:00 AM – 9:30 AM EST" : s.name === "NY Open (9:30AM–12PM)" ? "9:30 AM – 12:00 PM EST" : s.name === "Afternoon Deadzone (12–3PM)" ? "12:00 – 3:00 PM EST" : s.name === "Power Hour (3–4PM)" ? "3:00 – 4:00 PM EST" : "4:00 – 6:00 PM EST";
                    return (
                      <div key={s.name}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 5 }}>
                          <div>
                            <span style={{ fontSize: 11, color: "#e2e8f0", fontWeight: 500 }}>{shortName}</span>
                            <span style={{ fontSize: 9, color: "#64748b", marginLeft: 8 }}>{timeLabel}</span>
                          </div>
                          <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
                            <span style={{ fontSize: 10, color: wr >= 50 ? "#4ade80" : "#f87171" }}>{wr}% WR</span>
                            <span style={{ fontSize: 10, letterSpacing: 0 }}><span style={{ letterSpacing: 0 }}><span style={{ color: "#4ade80" }}>{s.wins}</span><span style={{ color: "#94a3b8" }}>/</span><span style={{ color: "#f87171" }}>{s.trades - s.wins}</span></span></span>
                            <span style={{ fontSize: 12, fontWeight: 600, color: s.pnl >= 0 ? "#4ade80" : "#f87171", minWidth: 80, textAlign: "right" }}>{s.pnl >= 0 ? "+" : ""}${s.pnl.toFixed(2)}</span>
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
            );
          })()}

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
                          <span style={{ fontSize: 12, fontWeight: 600, color: s.pnl >= 0 ? "#4ade80" : "#f87171", minWidth: 80, textAlign: "right" }}>{s.pnl >= 0 ? "+" : ""}${s.pnl.toFixed(2)}</span>
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
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {[
                  { key: "long",  label: "Long  📈", color: "#4ade80" },
                  { key: "short", label: "Short 📉", color: "#f87171" },
                ].map(({ key, label, color }) => {
                  const d = a.byDirection[key];
                  if (!d || d.trades === 0) return null;
                  const wr = Math.round(d.wins / d.trades * 100);
                  const maxPnl = Math.max(Math.abs(a.byDirection.long.pnl), Math.abs(a.byDirection.short.pnl), 1);
                  const barW = Math.abs(d.pnl) / maxPnl * 100;
                  const avg = d.pnl / d.trades;
                  return (
                    <div key={key}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 5 }}>
                        <div>
                          <span style={{ fontSize: 11, color: "#e2e8f0", fontWeight: 500 }}>{label}</span>
                          <span style={{ fontSize: 9, color: "#64748b", marginLeft: 8 }}>{d.trades} trades · avg {avg >= 0 ? "+" : ""}${avg.toFixed(2)}</span>
                        </div>
                        <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
                          <span style={{ fontSize: 10, color: wr >= 50 ? "#4ade80" : "#f87171" }}>{wr}% WR</span>
                          <span style={{ fontSize: 10 }}><span style={{ color: "#4ade80" }}>{d.wins}</span><span style={{ color: "#94a3b8" }}>/</span><span style={{ color: "#f87171" }}>{d.losses}</span></span>
                          <span style={{ fontSize: 12, fontWeight: 600, color: d.pnl >= 0 ? "#4ade80" : "#f87171", minWidth: 80, textAlign: "right" }}>{d.pnl >= 0 ? "+" : ""}${d.pnl.toFixed(2)}</span>
                        </div>
                      </div>
                      <div style={{ background: "#0a0e1a", borderRadius: 2, height: 5, overflow: "hidden" }}>
                        <div style={{ width: `${barW}%`, height: "100%", background: d.pnl >= 0 ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} />
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
                          <span style={{ fontSize: 12, fontWeight: 600, color: b.pnl >= 0 ? "#4ade80" : "#f87171", minWidth: 80, textAlign: "right" }}>{b.pnl >= 0 ? "+" : ""}${b.pnl.toFixed(2)}</span>
                          <span style={{ fontSize: 9, color: "#64748b", minWidth: 60, textAlign: "right" }}>avg {avg >= 0 ? "+" : ""}${avg.toFixed(2)}</span>
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
                      <div style={{ fontSize: 9, color: "#64748b", letterSpacing: "0.1em", marginBottom: 4 }}>{s.l}</div>
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
                <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 12 }}>ORDER TYPE BREAKDOWN</div>
                <div style={{ display: "grid", gridTemplateColumns: activeOTs.length === 3 ? "1fr 1fr 1fr" : activeOTs.length === 2 ? "1fr 1fr" : "1fr", gap: 10 }}>
                  {[
                    { key: "LMT", label: "LIMIT",  color: "#93c5fd", bg: "rgba(147,197,253,0.08)", border: "rgba(147,197,253,0.2)"  },
                    { key: "STP", label: "STOP",   color: "#fb923c", bg: "rgba(251,146,60,0.08)",  border: "rgba(251,146,60,0.2)"   },
                    { key: "MKT", label: "MARKET", color: "#94a3b8", bg: "rgba(148,163,184,0.06)", border: "rgba(148,163,184,0.15)" },
                  ].filter(({ key }) => (a.byOrderType[key] || {}).trades > 0).map(({ key, label, color, bg, border }) => {
                    const d = a.byOrderType[key];
                    const wr = d.trades ? Math.round(d.wins / d.trades * 100) : 0;
                    const pf = d.grossLoss > 0 ? (d.grossWin / d.grossLoss).toFixed(2) : d.grossWin > 0 ? "∞" : "—";
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
                            { l: "NET P&L", v: `${d.pnl >= 0 ? "+" : ""}$${d.pnl.toFixed(0)}`, c: d.pnl >= 0 ? "#4ade80" : "#f87171" },
                            { l: "AVG/TRADE", v: `${avgPnl >= 0 ? "+" : ""}$${avgPnl.toFixed(0)}`, c: avgPnl >= 0 ? "#4ade80" : "#f87171" },
                            { l: "PROF FACTOR", v: pf, c: "#e2e8f0" },
                          ].map(s => (
                            <div key={s.l} style={{ background: "#0a0e1a", borderRadius: 4, padding: "6px 8px" }}>
                              <div style={{ fontSize: 8, color: "#475569", letterSpacing: "0.1em", marginBottom: 3 }}>{s.l}</div>
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
                    <span style={{ fontSize: 9, color: "#475569", letterSpacing: "0.1em" }}>TOTAL COMMISSIONS</span>
                    <span style={{ fontSize: 12, color: "#f87171" }}>-${a.totalCommission.toFixed(2)}</span>
                  </div>
                )}
              </div>
            );
          })()}

          {/* Equity Curve */}
          <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 14px 10px" }}>
            <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 10 }}>EQUITY CURVE</div>
            {(() => {
              const vals = a.equityCurve.map(p => p.pnl);
              const chartVals = vals.length === 1 ? [0, vals[0]] : vals;
              return <EquityCurveChart values={chartVals} height={90} gradientId="ec1" dangerLine={dangerLine} />;
            })()}
          </div>
        </>
      )}


      {analyticsTab === "by session" && (
        <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "16px" }}>
          <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 14 }}>P&L BY TIME OF DAY</div>
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
      {analyticsTab === "trade log" && (
        <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, overflow: "hidden" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 12px", borderBottom: "1px solid #1e293b" }}>
            <div style={{ display: "grid", gridTemplateColumns: "100px 44px 90px 90px 96px 52px 70px 90px", width: "100%", fontSize: 10, color: "#94a3b8", letterSpacing: "0.08em" }}>
              <div>SYMBOL</div><div>QTY</div><div style={{ textAlign: "right" }}>BUY</div><div style={{ textAlign: "right" }}>SELL</div><div style={{ textAlign: "right" }}>TIME</div><div style={{ textAlign: "right" }}>TYPE</div><div style={{ textAlign: "right" }}>COMM</div><div style={{ textAlign: "right" }}>NET P&L</div>
            </div>
            <button
              onClick={() => {
                const header = "Symbol,Qty,Buy Price,Buy Time,Duration,Sell Time,Sell Price,Gross P&L,Commission,Net P&L,Order Type,Multiplier,Notes\n";
                const rows = trades.map(t =>
                  `${t.symbol},${t.qty},${t.buyPrice},${t.buyTime || ""},${t.duration || ""},${t.sellTime || ""},${t.sellPrice},${t.pnl.toFixed(2)},${(t.commission||0).toFixed(2)},${(t.pnl-(t.commission||0)).toFixed(2)},${t.orderType||"MKT"},${t.multiplier||""},${t.notes||""}`
                ).join("\n");
                const blob = new Blob([header + rows], { type: "text/csv" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url; a.download = `trades-export.csv`; a.click();
                URL.revokeObjectURL(url);
              }}
              style={{ background: "transparent", border: "1px solid #1e293b", color: "#64748b", padding: "3px 10px", borderRadius: 3, fontSize: 9, cursor: "pointer", fontFamily: "inherit", letterSpacing: "0.06em", whiteSpace: "nowrap", marginLeft: 10, flexShrink: 0, transition: "all .15s" }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = "#3b82f6"; e.currentTarget.style.color = "#93c5fd"; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = "#1e293b"; e.currentTarget.style.color = "#475569"; }}>
              ↓ CSV
            </button>
          </div>
          <div style={{ maxHeight: 360, overflowY: "auto" }}>
            {trades.map((t, i) => (
              <div key={i} style={{ display: "grid", gridTemplateColumns: "100px 44px 90px 90px 96px 52px 70px 90px", padding: "8px 14px", borderBottom: "1px solid #0a0e1a", fontSize: 12, background: i % 2 === 0 ? "#0f1729" : "#0d1525" }}>
                <div style={{ color: "#93c5fd" }}>{t.symbol}{t.notes ? <span style={{ fontSize: 9, color: "#f59e0b", marginLeft: 3 }} title={t.notes}>●</span> : null}</div>
                <div style={{ color: "#94a3b8" }}>{t.qty}</div>
                <div style={{ textAlign: "right", color: "#e2e8f0" }}>{t.buyPrice.toFixed(2)}</div>
                <div style={{ textAlign: "right", color: "#e2e8f0" }}>{t.sellPrice.toFixed(2)}</div>
                <div style={{ textAlign: "right", color: "#94a3b8", fontSize: 10 }}>{t.sellTime?.split(" ")[1] || ""}</div>
                <div style={{ textAlign: "right", fontSize: 10 }}>
                  <span style={{ padding: "1px 5px", borderRadius: 3, background: t.orderType === "LMT" ? "rgba(147,197,253,0.12)" : t.orderType === "STP" ? "rgba(251,146,60,0.12)" : "rgba(148,163,184,0.08)", color: t.orderType === "LMT" ? "#93c5fd" : t.orderType === "STP" ? "#fb923c" : "#64748b", letterSpacing: "0.05em" }}>{t.orderType || "MKT"}</span>
                </div>
                <div style={{ textAlign: "right", color: "#475569", fontSize: 11 }}>{t.commission > 0 ? `-$${t.commission.toFixed(2)}` : "—"}</div>
                <div style={{ textAlign: "right", color: t.pnl > 0 ? "#4ade80" : t.pnl < 0 ? "#f87171" : "#94a3b8", fontWeight: 500 }}>{fmtPnl(t.pnl - (t.commission||0))}</div>
              </div>
            ))}
          </div>
        </div>
      )}
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
    const winners = trades.filter(t => t.pnl > 0);
    const losers = trades.filter(t => t.pnl < 0);
    const avgWin = winners.length ? (winners.reduce((s,t) => s+t.pnl, 0)/winners.length).toFixed(2) : 0;
    const avgLoss = losers.length ? Math.abs(losers.reduce((s,t) => s+t.pnl, 0)/losers.length).toFixed(2) : 0;
    const biggestWin = winners.length ? Math.max(...winners.map(t=>t.pnl)).toFixed(2) : 0;
    const biggestLoss = losers.length ? Math.min(...losers.map(t=>t.pnl)).toFixed(2) : 0;
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
    const flaggedNotes = trades.filter(t=>t.notes).map(t=>`Trade ${trades.indexOf(t)+1}: ${t.notes}`).join(", ") || "none";
    const durationBreakdown = a?.DURATION_BUCKETS ? a.DURATION_BUCKETS.filter(b=>a.byDuration[b.key]?.trades>0).map(b=>{
      const bd = a.byDuration[b.key]; return `${b.key}: ${bd.trades}t ${Math.round(bd.wins/bd.trades*100)}%WR $${bd.pnl.toFixed(0)}`;
    }).join(" | ") : "none";

    const prompt = `You are an expert futures trading analyst. Analyze this trader's full day of data and give exactly 3 findings — the most insightful, non-obvious patterns that will directly improve their profitability. Each finding must be data-backed with specific numbers. No fluff, no generic advice.

DATE: ${entry.date} (${new Date(entry.date+"T12:00:00").toLocaleDateString("en-US",{weekday:"long"})})
INSTRUMENTS: ${(entry.instruments?.length ? entry.instruments : [entry.instrument||"?"]).join(", ")}
GRADE: ${entry.grade||"?"} | BIAS: ${entry.bias||"?"} | MOOD: ${(entry.moods||[entry.mood]).filter(Boolean).join(", ")||"?"}
SESSION MISTAKES: ${(entry.sessionMistakes||[]).join(", ")||"none"}
GROSS P&L: $${entry.pnl||0} | NET P&L: $${(parseFloat(entry.pnl||0)-parseFloat(entry.commissions||0)).toFixed(2)} | FEES: $${entry.commissions||0}
TOTAL TRADES: ${trades.length} | WINNERS: ${winners.length} | LOSERS: ${losers.length}
WIN RATE: ${trades.length ? ((winners.length/trades.length)*100).toFixed(1) : 0}%
AVG WIN: $${avgWin} | AVG LOSS: $${avgLoss} | BIGGEST WIN: $${biggestWin} | BIGGEST LOSS: $${biggestLoss}
PROFIT FACTOR: ${a?.profitFactor?.toFixed(2)||"N/A"} | EXPECTANCY: ${a?.expectancy?.toFixed(2)||"N/A"}R
BY SESSION: ${sessionBreakdown}
BY SYMBOL: ${bySymbol}
BY ORDER TYPE: ${orderTypeBreakdown}
BY DURATION BUCKET: ${durationBreakdown}
COMMISSIONS: total $${totalComm.toFixed(2)} (${commDragPct}% of gross P&L)
BROKER FLAGS (partial fills etc): ${flaggedNotes}
GRADE: ${entry.grade||"none"} | EXECUTION SCORE: ${entry.executionScore != null ? entry.executionScore+"/10" : "not logged"} | DECISION SCORE: ${entry.decisionScore != null ? entry.decisionScore+"/10" : "not logged"}
MOOD: ${(entry.moods?.length ? entry.moods : entry.mood ? [entry.mood] : []).join(", ")||"not logged"}
BIAS: ${entry.bias||"none"}
${noteSummary ? `JOURNAL NARRATIVE (AI-consolidated from all notes):\n${noteSummary}\n` : ""}MARKET NOTES: ${note("marketNotes")}
RULES FOLLOWED/BROKEN: ${note("rules")}
LESSONS LEARNED: ${note("lessonsLearned")}
MISTAKES (freeform): ${note("mistakes")}
SESSION MISTAKES FLAGGED: ${(entry.sessionMistakes||[]).join(", ")||"none"}
AREAS FOR IMPROVEMENT: ${note("improvements")}
BEST TRADE (own words): ${note("bestTrade")}
WORST TRADE (own words): ${note("worstTrade")}
RULE TO REINFORCE: ${note("reinforceRule")}
PLAN FOR TOMORROW: ${note("tomorrow")}
${(() => { const costs = entry.mistakeCosts||{}; const rows = Object.entries(costs).filter(([,v])=>v!=null&&Number(v)>0); if(!rows.length) return "MISTAKE COSTS: none attributed"; const total = rows.reduce((s,[,v])=>s+Number(v),0); return `MISTAKE COST BREAKDOWN: ${rows.map(([tag,v])=>`${tag}: $${Number(v).toFixed(0)}`).join(", ")} — total $${total.toFixed(0)}${entry.mistakeCostNotes?" · Notes: "+entry.mistakeCostNotes:""}`; })()}

FULL TRADE LOG:
${tradeLog||"No trade data imported"}

CRITICAL INSTRUCTION: Cross-reference the trader's OWN WRITTEN NOTES against the trade data. Look for: (1) contradictions — they said they'd do X but the trades show Y, (2) patterns they haven't named — recurring timing or behavior visible in the log but absent from their notes, (3) cost of their stated mistakes — if they flagged a mistake, find the exact trade(s) it cost them and quantify it. At least one finding must reference something from their written notes directly.

Return EXACTLY this format, nothing else:

**FINDING 1: [short punchy title]**
[2-3 sentences. Specific data point → what it means → one concrete action to fix or reinforce it]

**FINDING 2: [short punchy title]**
[2-3 sentences. Specific data point → what it means → one concrete action to fix or reinforce it]

**FINDING 3: [short punchy title]**
[2-3 sentences. Specific data point → what it means → one concrete action to fix or reinforce it]`;

    try {
      const txt = await aiRequestText(ai, {
        max_tokens: 600,
        timeoutMs: 24000,
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
          <button onClick={generate} style={{ background: "transparent", border: "1px solid #1e293b", color: "#64748b", padding: "2px 8px", borderRadius: 3, fontSize: 9, cursor: "pointer", fontFamily: "inherit", letterSpacing: "0.06em" }}>↺ regenerate</button>
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
        <div style={{ background: "#1f0606", border: "1px solid #7f1d1d", borderRadius: 6, padding: "12px 16px", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
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
    const winners = trades.filter(t => t.pnl > 0);
    const losers = trades.filter(t => t.pnl < 0);
    const avgWin = winners.length ? winners.reduce((s, t) => s + t.pnl, 0) / winners.length : 0;
    const avgLoss = losers.length ? Math.abs(losers.reduce((s, t) => s + t.pnl, 0) / losers.length) : 0;
    const rrRatio = avgLoss > 0 ? (avgWin / avgLoss).toFixed(2) : "N/A";
    const winRate = trades.length ? ((winners.length / trades.length) * 100).toFixed(1) : 0;
    const profitFactor = a?.profitFactor ? a.profitFactor.toFixed(2) : "N/A";
    const expectancy = a?.expectancy !== null && a?.expectancy !== undefined ? a.expectancy.toFixed(2) + "R" : "N/A";
    // Use AI-polished rewrites where available, fall back to raw original
    const rw = entry.aiRewrites || {};
    const note = (key) => (rw[key]?.trim() || entry[key] || "None");
    const noteSummary = entry.aiNoteSummary || "";

    const sessionBreakdown = a?.bySession ? Object.entries(a.bySession).map(([s, d]) =>
      `  ${s}: ${d.trades} trades, ${Math.round(d.wins/d.trades*100)}% win rate, $${d.pnl.toFixed(2)} P&L`
    ).join("\n") : "No session data";

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
MOOD: ${(entry.moods?.length ? entry.moods : entry.mood ? [entry.mood] : ["Not logged"]).join(", ")}
SESSION MISTAKES FLAGGED: ${(entry.sessionMistakes?.length ? entry.sessionMistakes : ["None flagged"]).join(", ")}
GRADE: ${entry.grade || "Not logged"}${entry.executionScore != null || entry.decisionScore != null ? ` (Execution: ${entry.executionScore != null ? entry.executionScore + "/10" : "—"}, Decision: ${entry.decisionScore != null ? entry.decisionScore + "/10" : "—"})` : ""}
BIAS: ${entry.bias || "Not logged"}
${(() => {
  const costs = entry.mistakeCosts || {};
  const rows = Object.entries(costs).filter(([, v]) => v != null && v > 0);
  if (rows.length === 0) return "";
  const total = rows.reduce((s, [, v]) => s + parseFloat(v), 0);
  const netPnlVal = parseFloat(entry.pnl || 0) - parseFloat(entry.commissions || 0);
  return `MISTAKE COST ATTRIBUTION: ${rows.map(([tag, v]) => `${tag}: $${parseFloat(v).toFixed(0)}`).join(", ")} — Total attributed: $${total.toFixed(0)} (vs net P&L of ${netPnlVal >= 0 ? "+" : ""}$${netPnlVal.toFixed(0)})${entry.mistakeCostNotes ? ` · Notes: ${entry.mistakeCostNotes}` : ""}\n`;
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

SESSION BREAKDOWN (P&L by time window):
${sessionBreakdown}

ORDER TYPE BREAKDOWN (LMT/STP/MKT performance):
${orderTypeStats}

HOLD TIME BREAKDOWN (duration buckets):
${durationStats}

BROKER FLAGS (partial fills, manual orders — trades to scrutinize):
${flaggedTrades}

FULL TRADE LOG (chronological — use this to detect sequencing, sizing, and revenge patterns):
${tradeLog}

TRADER'S OWN NOTES:
${noteSummary ? `  JOURNAL NARRATIVE (AI-consolidated):\n  ${noteSummary}\n` : ""}  Market notes: ${note("marketNotes")}
  Rules followed/broken: ${note("rules")}
  Lessons learned: ${note("lessonsLearned")}
  Mistakes (freeform): ${note("mistakes")}
  Session mistakes flagged: ${(entry.sessionMistakes?.length ? entry.sessionMistakes : ["None"]).join(", ")}
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

Provide a deep, data-driven daily analysis with exactly these sections. Use specific numbers from the data. Be like a sharp trading coach, not a cheerleader. Cross-reference the trade log with stated notes — call out discrepancies.

**🔬 DATA SNAPSHOT**
3-4 sentences on what the raw numbers reveal. Lead with the single most important stat (profit factor, expectancy, R-ratio, or drawdown). Mention if fees are eating a material % of gross. If execution/decision scores were logged, note whether they align with the actual trade performance — a high execution score on a losing day signals self-rating bias worth flagging. Be specific.

**✅ WHAT WORKED TODAY**
2-3 specific things backed by the data. Reference actual trades by number, time window, or price. If a session window dominated, name it. If the R-ratio was strong, say so. Be concrete — no generic praise.

**⚠️ WHAT NEEDS WORK**
2-3 specific weaknesses from the data AND flagged mistakes. Directly cross-reference: if "Cut winner early" was flagged and avg winner < avg loser, say exactly that and quantify the gap. If trades after a losing trade show a pattern (bigger size, quicker exit, worse entry), name it. Don't soften.

**🧠 BEHAVIORAL INSIGHTS**
Two-part analysis: (A) TRADE SEQUENCE — scan the log for revenge trading (losses followed by bigger size/faster re-entry), FOMO (late entries into fast moves), discipline failures (dead zone trades, overtrading after a win), emotional escalation, or size inconsistency. Name each pattern with specific trade numbers as evidence. (B) NOTES VS. REALITY — read every word the trader wrote today (market notes, lessons learned, mistakes, rules, best/worst trade descriptions). Find contradictions between what they wrote and what the trades show. If they wrote "I was disciplined" but the log shows a revenge trade, name it. If they identified a lesson but the same pattern appeared earlier in the day, flag it. If mood logged correlates with a known behavioral pattern in today's data, call it out explicitly.
${priorPlan && (priorPlan.plan || priorPlan.reinforceRule) ? `
**🚩 RED FLAGS — PLAN VS. EXECUTION**
Cross-reference today's trades directly against the previous session's stated plan. For each contradiction found: (1) quote what was planned, (2) describe what actually happened with specific trade evidence, (3) label the violation type (e.g. rule break, timing violation, sizing violation, setup deviation). If no violations were found, state that explicitly and note which parts of the plan were honored. This section is non-negotiable if prior plan data exists.
` : ""}

**📈 MISTAKE TREND WATCH**
For each flagged session mistake AND each mistake mentioned in freeform notes: (1) find the specific trade(s) where it occurred — trade number, time, and P&L impact, (2) classify root cause as emotional / mechanical / situational, (3) calculate the dollar cost if quantifiable (e.g. "held loser 4 extra minutes = -$X vs avg loss"), (4) write one precise rule to prevent it tomorrow with a measurable threshold. If no mistakes were flagged in any notes field, scan the trade log independently for patterns the trader missed — unacknowledged mistakes are the most dangerous ones. If mistake cost data was provided, cross-reference it: does the trader's dollar attribution match what the trade data actually shows?

**🎯 TOMORROW'S EDGE**
3 concrete, measurable action points. At least one must directly address a flagged session mistake with a specific rule (time, size, setup condition). Make them actionable: "Max 2 contracts until first profitable trade confirmed" not "size down."

Keep each section tight. No filler. Maximum value per word. Total response should be 400-600 words.`;
  };

  const generate = async () => {
    setLoading(true);
    setAnalysis("");
    setDone(false);
    try {
      const txt = await aiRequestText(ai, {
        max_tokens: 900,
        timeoutMs: 24000,
        messages: [{ role: 'user', content: prompt }],
      });
      setAnalysis(txt || 'ERROR');
      setDone(true);
    } catch {
      setAnalysis("ERROR");
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
          <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>Deep coaching insights powered by Claude · Based on your full trade data & notes</div>
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
      {!loading && analysis && analysis !== "ERROR" && (
        <div style={{ padding: "20px 24px" }}>
          <RenderAI text={analysis} />
        </div>
      )}

      {/* Error */}
      {!loading && analysis === "ERROR" && (
        <div style={{ padding: "20px 24px", color: "#f87171", fontSize: 13 }}>Failed to generate analysis. Please try again.</div>
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
              <div style={{ fontSize: 10, color: "#94a3b8", fontWeight: 600, marginBottom: 3, letterSpacing: "0.05em" }}>{s.label}</div>
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
      <div style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 12 }}>CHART SCREENSHOTS <span style={{ color: "#64748b", fontWeight: 400 }}>· {screenshots.length}</span></div>
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
          timeoutMs: 24000,
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
            style={{ background: "transparent", border: "none", color: "#64748b", fontSize: 10, cursor: "pointer", padding: "2px 4px", fontFamily: "inherit", letterSpacing: "0.04em", lineHeight: 1, transition: "color .15s" }}
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
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 24 }}>
        <div>
          <div style={{ fontSize: 11, color: "#94a3b8", letterSpacing: "0.1em" }}>{new Date(entry.date + "T12:00:00").toLocaleDateString("en-US", { weekday: "long" }).toUpperCase()}</div>
          <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 34, color: "#e2e8f0", letterSpacing: "0.1em" }}>{entry.date}</div>
        </div>
        <div style={{ flex: 1 }} />
        {entry.pnl !== "" && (
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 11, color: "#94a3b8", letterSpacing: "0.08em", marginBottom: 2 }}>NET P&L</div>
            <div style={{ fontSize: 36, color: pnlColor(netPnl(entry)), fontWeight: 500 }}>{fmtPnl(netPnl(entry))}</div>
            {(parseFloat(entry.commissions)) ? (
              <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 3, lineHeight: 1.6 }}>
                <span style={{ color: "#94a3b8" }}>gross {fmtPnl(entry.pnl)}</span>
                {parseFloat(entry.commissions) > 0 && <span style={{ color: "#f87171" }}> − ${parseFloat(entry.commissions).toFixed(2)} fees</span>}
              </div>
            ) : null}
          </div>
        )}
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
          <AnalyticsPanel a={a} trades={entry.parsedTrades || []} pnlColor={pnlColor} fmtPnl={fmtPnl} analyticsTab={atab} setAnalyticsTab={setAtab} totalFees={parseFloat(entry.commissions) || 0} dangerLine={dangerLine} />
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
                        <span key={m} style={{ padding: "4px 10px", borderRadius: 20, fontSize: 11, background: "#1f0606", border: "1px solid #7f1d1d", color: "#f87171" }}>{m}</span>
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
                  <RewriteBtn fieldKey="marketNotes" text={entry.marketNotes} />
                </div>
                <div style={{ fontSize: 13, color: rewrites.marketNotes && !showOriginal.marketNotes ? "#93c5fd" : "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap", fontStyle: rewrites.marketNotes && !showOriginal.marketNotes ? "italic" : "normal" }}>{rewrites.marketNotes && !showOriginal.marketNotes ? rewrites.marketNotes : entry.marketNotes}</div>
              </div>
            )}
            {entry.rules && (
              <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 6, padding: "14px 16px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <div style={{ fontSize: 10, color: "#93c5fd", letterSpacing: "0.12em" }}>📋 RULES FOLLOWED / BROKEN</div>
                  <RewriteBtn fieldKey="rules" text={entry.rules} />
                </div>
                <div style={{ fontSize: 13, color: rewrites.rules && !showOriginal.rules ? "#93c5fd" : "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap", fontStyle: rewrites.rules && !showOriginal.rules ? "italic" : "normal" }}>{rewrites.rules && !showOriginal.rules ? rewrites.rules : entry.rules}</div>
              </div>
            )}

            {/* Trades */}
            {(entry.bestTrade || entry.worstTrade) && (
              <div style={{ display: "grid", gridTemplateColumns: entry.bestTrade && entry.worstTrade ? "1fr 1fr" : "1fr", gap: 10 }}>
                {entry.bestTrade && (
                  <div style={{ background: "#061f0f", border: "1px solid #166534", borderRadius: 6, padding: "14px 16px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                      <div style={{ fontSize: 10, color: "#4ade80", letterSpacing: "0.12em" }}>✅ BEST TRADE</div>
                      <RewriteBtn fieldKey="bestTrade" text={entry.bestTrade} />
                    </div>
                    <div style={{ fontSize: 13, color: rewrites.bestTrade && !showOriginal.bestTrade ? "#93c5fd" : "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap", fontStyle: rewrites.bestTrade && !showOriginal.bestTrade ? "italic" : "normal" }}>{rewrites.bestTrade && !showOriginal.bestTrade ? rewrites.bestTrade : entry.bestTrade}</div>
                  </div>
                )}
                {entry.worstTrade && (
                  <div style={{ background: "#1f0606", border: "1px solid #7f1d1d", borderRadius: 6, padding: "14px 16px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                      <div style={{ fontSize: 10, color: "#f87171", letterSpacing: "0.12em" }}>❌ WORST TRADE / MISTAKE</div>
                      <RewriteBtn fieldKey="worstTrade" text={entry.worstTrade} />
                    </div>
                    <div style={{ fontSize: 13, color: rewrites.worstTrade && !showOriginal.worstTrade ? "#93c5fd" : "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap", fontStyle: rewrites.worstTrade && !showOriginal.worstTrade ? "italic" : "normal" }}>{rewrites.worstTrade && !showOriginal.worstTrade ? rewrites.worstTrade : entry.worstTrade}</div>
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
                      <div style={{ fontSize: 10, color: "#64748b", letterSpacing: "0.1em" }}>LESSONS LEARNED</div>
                      <RewriteBtn fieldKey="lessonsLearned" text={entry.lessonsLearned} />
                    </div>
                    <div style={{ fontSize: 13, color: rewrites.lessonsLearned && !showOriginal.lessonsLearned ? "#93c5fd" : "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap", fontStyle: rewrites.lessonsLearned && !showOriginal.lessonsLearned ? "italic" : "normal" }}>{rewrites.lessonsLearned && !showOriginal.lessonsLearned ? rewrites.lessonsLearned : entry.lessonsLearned}</div>
                  </div>
                )}
                {entry.mistakes && (
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                      <div style={{ fontSize: 10, color: "#64748b", letterSpacing: "0.1em" }}>MISTAKES TO AVOID</div>
                      <RewriteBtn fieldKey="mistakes" text={entry.mistakes} />
                    </div>
                    <div style={{ fontSize: 13, color: rewrites.mistakes && !showOriginal.mistakes ? "#93c5fd" : "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap", fontStyle: rewrites.mistakes && !showOriginal.mistakes ? "italic" : "normal" }}>{rewrites.mistakes && !showOriginal.mistakes ? rewrites.mistakes : entry.mistakes}</div>
                  </div>
                )}
                {entry.improvements && (
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                      <div style={{ fontSize: 10, color: "#64748b", letterSpacing: "0.1em" }}>AREAS FOR IMPROVEMENT</div>
                      <RewriteBtn fieldKey="improvements" text={entry.improvements} />
                    </div>
                    <div style={{ fontSize: 13, color: rewrites.improvements && !showOriginal.improvements ? "#93c5fd" : "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap", fontStyle: rewrites.improvements && !showOriginal.improvements ? "italic" : "normal" }}>{rewrites.improvements && !showOriginal.improvements ? rewrites.improvements : entry.improvements}</div>
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
                      <RewriteBtn fieldKey="reinforceRule" text={entry.reinforceRule} />
                    </div>
                    <div style={{ fontSize: 13, color: rewrites.reinforceRule && !showOriginal.reinforceRule ? "#93c5fd" : "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap", fontStyle: rewrites.reinforceRule && !showOriginal.reinforceRule ? "italic" : "normal" }}>{rewrites.reinforceRule && !showOriginal.reinforceRule ? rewrites.reinforceRule : entry.reinforceRule}</div>
                  </div>
                )}
                {entry.tomorrow && (
                  <div style={{ background: "#0f1729", border: "1px solid #1e3a5f", borderRadius: 6, padding: "14px 16px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                      <div style={{ fontSize: 10, color: "#93c5fd", letterSpacing: "0.12em" }}>🗓 PLAN FOR TOMORROW</div>
                      <RewriteBtn fieldKey="tomorrow" text={entry.tomorrow} />
                    </div>
                    <div style={{ fontSize: 13, color: rewrites.tomorrow && !showOriginal.tomorrow ? "#93c5fd" : "#e2e8f0", lineHeight: 1.7, whiteSpace: "pre-wrap", fontStyle: rewrites.tomorrow && !showOriginal.tomorrow ? "italic" : "normal" }}>{rewrites.tomorrow && !showOriginal.tomorrow ? rewrites.tomorrow : entry.tomorrow}</div>
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
                        <div style={{ fontSize: 9, color: "#64748b", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 4 }}>{f.label}</div>
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
        <div style={{ marginTop: 20, background: "#0a0e1a", border: "1px solid #1e3a5f", borderRadius: 6, overflow: "hidden" }}>
          <div style={{ padding: "14px 20px", background: "#0a1628", borderBottom: "1px solid #1e293b", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontSize: 12, color: "#93c5fd", letterSpacing: "0.1em", fontWeight: 600 }}>✦ AI WEEKLY & MONTHLY RECAP</div>
              <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>Patterns, lessons & coaching insights across your journal</div>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={() => onRecap("weekly")}
                style={{ background: "#1d4ed8", color: "white", border: "none", padding: "8px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.06em", transition: "all .15s" }}>
                ANALYSE WEEK →
              </button>
              <button onClick={() => onRecap("monthly")}
                style={{ background: "#1d4ed8", color: "white", border: "none", padding: "8px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.06em", transition: "all .15s" }}>
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
  const calcNetPnl = calcNetPnlProp;
  const [year, mon] = month.split("-").map(Number);
  const [collapsed, setCollapsed] = useState({ pnl: false, trades: false, time: false, symbol: false, dow: false, mistakes: false, mistakecost: false, mood: false });
  const toggleSection = (key) => setCollapsed(prev => ({ ...prev, [key]: !prev[key] }));
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
  const allMonthWinningTrades = allMonthTrades.filter(t => t.pnl > 0);
  const tradeWinRate = allMonthTrades.length ? Math.round(allMonthWinningTrades.length / allMonthTrades.length * 100) : null;

  // Full monthly analytics
  const monthAnalytics = allMonthTrades.length > 0 ? calcAnalytics(allMonthTrades, aiCfg?.tzLock !== false) : null;
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

            {/* P&L Summary */}
            <div>
              <SectionHeader label="P&L SUMMARY" skey="pnl" summary={<span style={{ color: pnlColor(monthPnL), fontWeight: 600 }}>{fmtPnl(monthPnL)} net</span>} />
              {!collapsed.pnl && (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
                {[
                  { l: "GROSS P&L", v: fmtPnl(monthGross), c: pnlColor(monthGross) },
                  { l: "TOTAL FEES", v: monthFees > 0 ? `-$${monthFees.toFixed(2)}` : "—", c: monthFees > 0 ? "#f87171" : "#94a3b8" },
                  { l: "NET P&L", v: fmtPnl(monthPnL), c: pnlColor(monthPnL), hi: true },
                  { l: "DAY WIN RATE", v: monthEntries.length ? `${Math.round(monthWins / monthEntries.length * 100)}%` : "—", c: monthWins / monthEntries.length >= 0.5 ? "#4ade80" : "#f87171" },
                  { l: "WIN DAYS", v: monthWins, c: "#4ade80" },
                  { l: "LOSS DAYS", v: monthLoss, c: "#f87171" },
                  ...(monthAnalytics ? [
                    { l: "LARGEST WIN DAY", v: fmtPnl(Math.max(...monthEntries.map(e => netPnl(e)))), c: "#4ade80" },
                    { l: "LARGEST LOSS DAY", v: fmtPnl(Math.min(...monthEntries.map(e => netPnl(e)))), c: "#f87171" },
                    { l: "MAX DRAWDOWN", v: `$${monthAnalytics.maxDD.toFixed(2)}`, c: "#f87171" },
                  ] : []),
                ].map(s => (
                  <div key={s.l} style={{ background: s.hi ? "#0d1f3c" : "#0f1729", border: `1px solid ${s.hi ? "#1e3a5f" : "#1e293b"}`, borderRadius: 4, padding: "10px 12px" }}>
                    <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 4 }}>{s.l}</div>
                    <div style={{ fontSize: 16, color: s.c, fontWeight: s.hi ? 700 : 500 }}>{s.v}</div>
                  </div>
                ))}
              </div>
              )}
            </div>

      {/* Monthly P&L Chart */}
      {monthEntries.length > 0 && (() => {
        const sorted = [...monthEntries].sort((a, b) => a.date.localeCompare(b.date));
        let running = 0;
        const points = sorted.map(e => { running += netPnl(e); return { date: e.date, cum: running, daily: netPnl(e) }; });
        const lineColor = points[points.length - 1].cum >= 0 ? "#4ade80" : "#f87171";
        // Pad to 2 points so the chart always renders (single day = flat line)
        const chartVals = points.length === 1 ? [0, points[0].cum] : points.map(p => p.cum);

        return (
          <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 6, padding: "14px 16px", marginBottom: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 10 }}>
              <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em" }}>CUMULATIVE P&L</div>
              <div style={{ fontSize: 13, color: lineColor, fontWeight: 500 }}>{fmtPnl(monthPnL)}</div>
            </div>
            <EquityCurveChart values={chartVals} dots={points.length > 1} height={100} gradientId="ec2" />
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
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
        );
      })()}

      {/* Calendar grid header */}
      <div style={{ fontSize: 22, color: "#93c5fd", letterSpacing: "0.08em", fontWeight: 700, fontFamily: "'Bebas Neue', sans-serif", marginBottom: 12 }}>
        {(() => { const [y, m] = month.split("-").map(Number); return new Date(y, m - 1, 1).toLocaleString("default", { month: "long", year: "numeric" }).toUpperCase(); })()}
      </div>

      {/* Day headers */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 3, marginBottom: 3 }}>
        {DAYS.map(d => (
          <div key={d} style={{ textAlign: "center", fontSize: 9, letterSpacing: "0.1em", padding: "3px 0",
            color: d === "SAT" || d === "SUN" ? "#475569" : "#94a3b8" }}>{d}</div>
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
            bgColor = n > 0 ? "#061f0f" : n < 0 ? "#1f0606" : "#0f1729";
            borderColor = n > 0 ? "#166534" : n < 0 ? "#7f1d1d" : "#1e293b";
          } else if (isWeekend) {
            bgColor = "#060810";
            borderColor = "#0d1018";
          } else if (isFuture) {
            bgColor = "#060810";
            borderColor = "#0f1420";
          }
          if (isToday) borderColor = "#3b82f6";

          return (
            <div key={dateStr}
              onClick={() => hasEntry ? onDayClick(entry) : (!isWeekend && isPast || dateStr === today) ? onNewDay(dateStr) : null}
              style={{
                background: bgColor, border: `1px solid ${borderColor}`, borderRadius: 4,
                padding: "10px 8px", minHeight: 115, position: "relative",
                cursor: hasEntry ? "pointer" : (!isWeekend && (isPast || dateStr === today)) ? "pointer" : "default",
                transition: "all .15s", opacity: isWeekend && !hasEntry ? 0.3 : isFuture && !isWeekend ? 0.5 : 1,
              }}
              onMouseEnter={e => { if (hasEntry || (!isWeekend && (isPast || dateStr === today))) e.currentTarget.style.borderColor = hasEntry ? (n > 0 ? "#22c55e" : n < 0 ? "#ef4444" : "#3b82f6") : "#475569"; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = borderColor; }}
            >
              {/* Row 1: day number + emoji + grade */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                <span style={{ fontSize: 14, fontWeight: 600, color: isToday ? "#3b82f6" : hasEntry ? "#e2e8f0" : isWeekend ? "#475569" : "#94a3b8" }}>{day}</span>
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
                  <div style={{ fontSize: 17, fontWeight: 700, color: n > 0 ? "#4ade80" : n < 0 ? "#f87171" : "#e2e8f0", lineHeight: 1, marginBottom: 5 }}>
                    {n > 0 ? "+" : ""}{n !== null ? `$${Math.abs(n).toLocaleString("en-US", { minimumFractionDigits: n % 1 === 0 ? 0 : 2, maximumFractionDigits: 2 })}` : "—"}
                  </div>
                  {/* Instruments — secondary */}
                  <div style={{ display: "flex", gap: 3, flexWrap: "wrap", alignItems: "center", marginBottom: 4 }}>
                    {(entry.instruments?.length ? entry.instruments : entry.instrument ? [entry.instrument] : []).map(sym => (
                      <span key={sym} style={{ fontSize: 10, padding: "1px 6px", borderRadius: 2, background: "#1e3a5f22", color: "#94a3b8", border: "1px solid #1e3a5f44", letterSpacing: "0.04em" }}>{sym}</span>
                    ))}
                    {!entry.grade && (entry.moods?.length ? entry.moods : entry.mood ? [entry.mood] : []).map(m => (
                      <span key={m} style={{ fontSize: 11 }}>{m.split(" ").pop()}</span>
                    ))}
                  </div>
                  {/* Trades + Win Rate — supporting info, smallest */}
                  {entry.parsedTrades?.length > 0 && (() => {
                    const wins = entry.parsedTrades.filter(t => t.pnl > 0).length;
                    const wr = Math.round(wins / entry.parsedTrades.length * 100);
                    return (
                      <>
                        <div style={{ fontSize: 9, color: "#94a3b8", lineHeight: 1.5 }}>Trades: {entry.parsedTrades.length}</div>
                        <div style={{ fontSize: 9, color: wr >= 50 ? "#4ade80" : "#f87171", lineHeight: 1.5 }}>Win Rate: {wr}%</div>
                      </>
                    );
                  })()}
                </>
              ) : !isWeekend && (isPast || dateStr === today) ? (
                <div style={{ fontSize: 9, color: "#1e293b", marginTop: 6, textAlign: "center" }}>+ add</div>
              ) : null}
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div style={{ display: "flex", gap: 16, marginTop: 12, fontSize: 10, color: "#64748b" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}><div style={{ width: 8, height: 8, borderRadius: 2, background: "#061f0f", border: "1px solid #166534" }} /> Win day</div>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}><div style={{ width: 8, height: 8, borderRadius: 2, background: "#1f0606", border: "1px solid #7f1d1d" }} /> Loss day</div>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}><div style={{ width: 8, height: 8, borderRadius: 2, border: "1px solid #3b82f6" }} /> Today</div>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}><div style={{ width: 8, height: 8, borderRadius: 2, background: "#0a0e1a", border: "1px solid #334155", opacity: 0.4 }} /> Weekend / no trade</div>
      </div>
            {/* Trade Statistics */}
            {monthAnalytics && (
              <div>
                <SectionHeader label="TRADE STATISTICS" skey="trades" summary={<span>{monthAnalytics.total} trades · <span style={{ color: monthAnalytics.winRate >= 50 ? "#4ade80" : "#f87171" }}>{monthAnalytics.winRate.toFixed(0)}% WR</span></span>} />
                {!collapsed.trades && (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
                  {[
                    { l: "TOTAL TRADES", v: monthAnalytics.total, c: "#e2e8f0" },
                    { l: "WINNERS", v: monthAnalytics.winners, c: "#4ade80" },
                    { l: "LOSERS", v: monthAnalytics.losers, c: "#f87171" },
                    { l: "WIN RATE", v: `${monthAnalytics.winRate.toFixed(1)}%`, c: monthAnalytics.winRate >= 50 ? "#4ade80" : "#f87171" },
                    { l: "PROFIT FACTOR", v: monthAnalytics.profitFactor ? monthAnalytics.profitFactor.toFixed(2) : "—", c: monthAnalytics.profitFactor >= 1 ? "#4ade80" : "#f87171" },
                    { l: "AVG CONTRACT SIZE", v: `${monthAnalytics.avgQty.toFixed(1)} cts`, c: "#e2e8f0" },
                    { l: "AVG WIN", v: fmtPnl(monthAnalytics.avgWin), c: "#4ade80" },
                    { l: "AVG LOSS", v: fmtPnl(monthAnalytics.avgLoss), c: "#f87171" },
                    { l: "MAX WIN STREAK", v: `${monthAnalytics.maxConsecWins} trades`, c: "#4ade80" },
                    { l: "MAX LOSS STREAK", v: `${monthAnalytics.maxConsecLoss} trades`, c: "#f87171" },
                    { l: "LARGEST WIN", v: fmtPnl(monthAnalytics.largestWin), c: "#4ade80" },
                    { l: "LARGEST LOSS", v: fmtPnl(monthAnalytics.largestLoss), c: "#f87171" },
                  ].map(s => (
                    <div key={s.l} style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "10px 12px" }}>
                      <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 4 }}>{s.l}</div>
                      <div style={{ fontSize: 16, color: s.c, fontWeight: 500 }}>{s.v}</div>
                    </div>
                  ))}
                </div>
                )}
              </div>
            )}

            {/* Performance by Time of Day */}
            {monthAnalytics && (() => {
              const sessions = ["Asian Session (6PM–12AM)", "London Session (12AM–9:30AM)", "NY Open (9:30AM–12PM)", "Afternoon Deadzone (12–3PM)", "Power Hour (3–4PM)", "After Hours (4–6PM)"];
              const sessionData = sessions.map(s => ({ name: s, ...(monthAnalytics.bySession[s] || { trades: 0, pnl: 0, wins: 0 }) })).filter(s => s.trades > 0);
              if (!sessionData.length) return null;
              const maxAbs = Math.max(...sessionData.map(s => Math.abs(s.pnl)), 1);
              return (
                <div>
                  <SectionHeader label="PERFORMANCE BY TIME OF DAY" skey="time" summary={<span style={{ color: "#64748b" }}>{sessionData.length} sessions</span>} />
                  {!collapsed.time && (
                  <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px", display: "flex", flexDirection: "column", gap: 12 }}>
                    {sessionData.map(s => {
                      const wr = Math.round(s.wins / s.trades * 100);
                      const barW = Math.abs(s.pnl) / maxAbs * 100;
                      const shortName = s.name === "Asian Session (6PM–12AM)" ? "Asian Session" : s.name === "London Session (12AM–9:30AM)" ? "London Session" : s.name === "NY Open (9:30AM–12PM)" ? "NY Open" : s.name === "Afternoon Deadzone (12–3PM)" ? "Deadzone" : s.name === "Power Hour (3–4PM)" ? "Power Hour" : "After Hours";
                      const timeLabel = s.name === "Asian Session (6PM–12AM)" ? "6:00 PM – 12:00 AM EST" : s.name === "London Session (12AM–9:30AM)" ? "12:00 AM – 9:30 AM EST" : s.name === "NY Open (9:30AM–12PM)" ? "9:30 AM – 12:00 PM EST" : s.name === "Afternoon Deadzone (12–3PM)" ? "12:00 – 3:00 PM EST" : s.name === "Power Hour (3–4PM)" ? "3:00 – 4:00 PM EST" : "4:00 – 6:00 PM EST";
                      return (
                        <div key={s.name}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 5 }}>
                            <div>
                              <span style={{ fontSize: 11, color: "#e2e8f0", fontWeight: 500 }}>{shortName}</span>
                              <span style={{ fontSize: 9, color: "#64748b", marginLeft: 8 }}>{timeLabel}</span>
                            </div>
                            <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
                              <span style={{ fontSize: 10, color: wr >= 50 ? "#4ade80" : "#f87171" }}>{wr}% WR</span>
                              <span style={{ fontSize: 10, letterSpacing: 0 }}><span style={{ letterSpacing: 0 }}><span style={{ color: "#4ade80" }}>{s.wins}</span><span style={{ color: "#94a3b8" }}>/</span><span style={{ color: "#f87171" }}>{s.trades - s.wins}</span></span></span>
                              <span style={{ fontSize: 12, fontWeight: 600, color: s.pnl >= 0 ? "#4ade80" : "#f87171", minWidth: 80, textAlign: "right" }}>{s.pnl >= 0 ? "+" : ""}${s.pnl.toFixed(2)}</span>
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
                dowStats[day].tradeWins += trades.filter(t => t.pnl > 0).length;
                dowStats[day].tradeLosses += trades.filter(t => t.pnl < 0).length;
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
                                    {s.pnl >= 0 ? "+" : ""}${Math.abs(s.pnl).toFixed(0)}
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
                            <span style={{ fontSize: 12, fontWeight: 600, color: s.pnl >= 0 ? "#4ade80" : "#f87171", minWidth: 80, textAlign: "right" }}>{s.pnl >= 0 ? "+" : ""}${s.pnl.toFixed(2)}</span>
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

            {/* Mistake Frequency */}
            {monthEntries.length > 0 && (() => {
              const mistakeCounts = {};
              let totalSessions = 0;
              for (const e of monthEntries) {
                const mistakes = (e.sessionMistakes || []).filter(m => m !== "No Mistakes — Executed the Plan ✓");
                if (mistakes.length > 0) totalSessions++;
                for (const m of mistakes) {
                  mistakeCounts[m] = (mistakeCounts[m] || 0) + 1;
                }
              }
              const sorted = Object.entries(mistakeCounts).sort((a, b) => b[1] - a[1]);
              const cleanSessions = monthEntries.filter(e => e.sessionMistakes?.includes("No Mistakes — Executed the Plan ✓")).length;
              if (sorted.length === 0 && cleanSessions === 0) return null;
              const maxCount = sorted[0]?.[1] || 1;
              return (
                <div>
                  <SectionHeader label="MISTAKE FREQUENCY" skey="mistakes"
                    summary={<span style={{ color: "#64748b" }}>{sorted.length} types · {cleanSessions} clean day{cleanSessions !== 1 ? "s" : ""}</span>} />
                  {!collapsed.mistakes && (
                    <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px" }}>
                      {cleanSessions > 0 && (
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: sorted.length ? 10 : 0, padding: "6px 10px", background: "#061f0f", border: "1px solid #166534", borderRadius: 4 }}>
                          <span style={{ fontSize: 11, color: "#4ade80" }}>✓ Executed the Plan</span>
                          <span style={{ fontSize: 12, fontWeight: 600, color: "#4ade80" }}>{cleanSessions} day{cleanSessions !== 1 ? "s" : ""}</span>
                        </div>
                      )}
                      {sorted.map(([mistake, count]) => {
                        const barW = (count / maxCount) * 100;
                        const freq = Math.round((count / monthEntries.length) * 100);
                        return (
                          <div key={mistake} style={{ marginBottom: 10 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
                              <span style={{ fontSize: 11, color: "#e2e8f0" }}>{mistake}</span>
                              <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                                <span style={{ fontSize: 9, color: "#94a3b8" }}>{freq}% of sessions</span>
                                <span style={{ fontSize: 12, fontWeight: 600, color: "#f87171" }}>{count}×</span>
                              </div>
                            </div>
                            <div style={{ background: "#0a0e1a", borderRadius: 2, height: 4, overflow: "hidden" }}>
                              <div style={{ width: `${barW}%`, height: "100%", background: "#f87171", borderRadius: 2, opacity: 0.7 }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })()}

            {/* Feature 4: Mistake Cost Accounting */}
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

            {/* Mood vs Performance */}
            {monthEntries.length > 0 && (() => {
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
              if (moodEntries.length === 0) return null;
              const maxAvg = Math.max(...moodEntries.map(([, s]) => Math.abs(s.pnl / s.sessions)), 1);
              return (
                <div>
                  <SectionHeader label="MOOD VS PERFORMANCE" skey="mood"
                    summary={<span style={{ color: "#64748b" }}>{moodEntries.length} mood{moodEntries.length !== 1 ? "s" : ""} tracked</span>} />
                  {!collapsed.mood && (
                    <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px", display: "flex", flexDirection: "column", gap: 10 }}>
                      {moodEntries.map(([mood, s]) => {
                        const avg = s.pnl / s.sessions;
                        const wr = Math.round((s.wins / s.sessions) * 100);
                        const barW = Math.abs(avg) / maxAvg * 100;
                        const isPos = avg >= 0;
                        return (
                          <div key={mood}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
                              <span style={{ fontSize: 11, color: "#e2e8f0" }}>{mood}</span>
                              <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                                <span style={{ fontSize: 9, color: "#94a3b8" }}>{s.sessions} session{s.sessions !== 1 ? "s" : ""} · {wr}% WR</span>
                                <span style={{ fontSize: 11, fontWeight: 600, color: isPos ? "#4ade80" : "#f87171", minWidth: 70, textAlign: "right" }}>{avg >= 0 ? "+" : ""}${avg.toFixed(0)} avg</span>
                              </div>
                            </div>
                            <div style={{ background: "#0a0e1a", borderRadius: 2, height: 4, overflow: "hidden" }}>
                              <div style={{ width: `${barW}%`, height: "100%", background: isPos ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} />
                            </div>
                          </div>
                        );
                      })}
                      {moodEntries.length >= 2 && (() => {
                        const best = moodEntries[0];
                        const worst = moodEntries[moodEntries.length - 1];
                        return (
                          <div style={{ marginTop: 4, padding: "8px 10px", background: "#0a1628", border: "1px solid #1e3a5f", borderRadius: 4, fontSize: 10, color: "#94a3b8", lineHeight: 1.7 }}>
                            <span style={{ color: "#3b82f6" }}>💡 </span>
                            Best mindset: <span style={{ color: "#4ade80" }}>{best[0]}</span> (+${(best[1].pnl / best[1].sessions).toFixed(0)} avg).
                            {(worst[1].pnl / worst[1].sessions) < 0 && <> Avoid trading in <span style={{ color: "#f87171" }}>{worst[0]}</span> state (${(worst[1].pnl / worst[1].sessions).toFixed(0)} avg).</>}
                          </div>
                        );
                      })()}
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

function WeeklyPerformance({ entries, netPnl: calcNetPnlProp, fmtPnl, pnlColor, calcAnalytics }) {
  const calcNetPnl = calcNetPnlProp;
  const [selectedYear, setSelectedYear] = useState(() => new Date().getFullYear());

  const getISOWeek = (dateStr) => {
    const d = new Date(dateStr + "T12:00:00");
    const jan4 = new Date(d.getFullYear(), 0, 4);
    const startOfWeek1 = new Date(jan4);
    startOfWeek1.setDate(jan4.getDate() - ((jan4.getDay() + 6) % 7));
    const diff = d - startOfWeek1;
    const week = Math.floor(diff / (7 * 24 * 60 * 60 * 1000)) + 1;
    const yr = d.getFullYear();
    return `${yr}-W${String(week).padStart(2, "0")}`;
  };

  const getWeekRange = (weekKey) => {
    const [yr, wStr] = weekKey.split("-W");
    const year = parseInt(yr), week = parseInt(wStr);
    const jan4 = new Date(year, 0, 4);
    const startOfWeek1 = new Date(jan4);
    startOfWeek1.setDate(jan4.getDate() - ((jan4.getDay() + 6) % 7));
    const monday = new Date(startOfWeek1);
    monday.setDate(startOfWeek1.getDate() + (week - 1) * 7);
    const friday = new Date(monday);
    friday.setDate(monday.getDate() + 4);
    const fmt = d => d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    return `${fmt(monday)} – ${fmt(friday)}`;
  };

  const years = [...new Set(entries.map(e => e.date?.slice(0, 4)).filter(Boolean))].map(Number).sort((a, b) => b - a);
  if (!years.includes(new Date().getFullYear())) years.unshift(new Date().getFullYear());

  const yearEntries = entries.filter(e => e.date?.startsWith(String(selectedYear)));

  // Group by ISO week
  const byWeek = {};
  for (const e of yearEntries) {
    const wk = getISOWeek(e.date);
    if (!byWeek[wk]) byWeek[wk] = [];
    byWeek[wk].push(e);
  }
  const weeks = Object.keys(byWeek).sort();

  if (weeks.length === 0) return (
    <div style={{ textAlign: "center", padding: "60px 0", color: "#64748b", fontSize: 12 }}>No entries for {selectedYear}.</div>
  );

  // Year summary
  const yearNet = yearEntries.reduce((s, e) => s + netPnl(e), 0);
  const yearWinDays = yearEntries.filter(e => netPnl(e) > 0).length;
  const yearLossDays = yearEntries.filter(e => netPnl(e) < 0).length;
  const yearTrades = yearEntries.flatMap(e => e.parsedTrades || []);
  const yearA = calcAnalytics(yearTrades, aiCfg?.tzLock !== false);

  // Equity curve across weeks
  let running = 0;
  const equityPoints = weeks.map(wk => {
    const wEntries = byWeek[wk] || [];
    const wNet = wEntries.reduce((s, e) => s + netPnl(e), 0);
    running += wNet;
    return running;
  });

  return (
    <div>
      {/* Year selector */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20, flexWrap: "wrap", gap: 10 }}>
        <div style={{ display: "flex", gap: 6 }}>
          {years.map(y => (
            <button key={y} onClick={() => setSelectedYear(y)}
              style={{ padding: "5px 12px", borderRadius: 3, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.05em", transition: "all .15s", background: selectedYear === y ? "#1e3a5f" : "transparent", border: `1px solid ${selectedYear === y ? "#3b82f6" : "#1e293b"}`, color: selectedYear === y ? "#93c5fd" : "#94a3b8" }}>
              {y}
            </button>
          ))}
        </div>
        <div style={{ fontSize: 10, color: "#64748b" }}>{weeks.length} weeks · {yearEntries.length} trading days</div>
      </div>

      {/* Year summary bar */}
      <div style={{ background: "#0a1628", border: "1px solid #1e3a5f", borderRadius: 6, padding: "14px 18px", marginBottom: 20 }}>
        <div style={{ fontSize: 11, color: "#93c5fd", letterSpacing: "0.1em", marginBottom: 12 }}>YEAR SUMMARY · {selectedYear}</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 14, marginBottom: equityPoints.length > 1 ? 18 : 0 }}>
          {[
            { l: "NET P&L", v: fmtPnl(yearNet), c: pnlColor(yearNet), hi: true },
            { l: "WIN WEEKS", v: `${weeks.filter(wk => byWeek[wk].reduce((s,e)=>s+netPnl(e),0) > 0).length}/${weeks.length}`, c: "#e2e8f0" },
            { l: "WIN DAYS", v: `${yearWinDays}/${yearLossDays}`, c: "#e2e8f0" },
            { l: "TOTAL TRADES", v: yearTrades.length || "—", c: "#e2e8f0" },
            { l: "WIN RATE", v: yearA ? `${yearA.winRate.toFixed(1)}%` : "—", c: yearA?.winRate >= 50 ? "#4ade80" : "#f87171" },
            { l: "PROFIT FACTOR", v: yearA?.profitFactor ? yearA.profitFactor.toFixed(2) : "—", c: yearA?.profitFactor >= 1 ? "#4ade80" : "#f87171" },
          ].map(s => (
            <div key={s.l}>
              <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.08em", marginBottom: 4 }}>{s.l}</div>
              <div style={{ fontSize: 15, color: s.c, fontWeight: s.hi ? 700 : 500 }}>{s.v}</div>
            </div>
          ))}
        </div>
        {equityPoints.length >= 1 && (
          <EquityCurveChart values={equityPoints.length === 1 ? [0, equityPoints[0]] : equityPoints} height={70} gradientId="ec4" />
        )}
      </div>

      {/* Weekly cards */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {weeks.slice().reverse().map((wk, idx) => {
          const wEntries = byWeek[wk] || [];
          const wNet = wEntries.reduce((s, e) => s + netPnl(e), 0);
          const wGross = wEntries.reduce((s, e) => s + (parseFloat(e.pnl) || 0), 0);
          const wFees = wEntries.reduce((s, e) => s + (parseFloat(e.commissions) || 0), 0);
          const wWinDays = wEntries.filter(e => netPnl(e) > 0).length;
          const wLossDays = wEntries.filter(e => netPnl(e) < 0).length;
          const wTrades = wEntries.flatMap(e => e.parsedTrades || []);
          const wA = calcAnalytics(wTrades, aiCfg?.tzLock !== false);
          const isPos = wNet >= 0;
          const weekNum = wk.split("-W")[1];
          return (
            <div key={wk} style={{ background: "#0a0e1a", border: `1px solid ${isPos ? "#166534" : wNet < 0 ? "#7f1d1d" : "#1e293b"}`, borderRadius: 6, overflow: "hidden" }}>
              {/* Week header */}
              <div style={{ padding: "14px 18px", background: isPos ? "#061f0f" : wNet < 0 ? "#1f0606" : "#0f1729", display: "grid", gridTemplateColumns: "1fr auto", gap: 16, alignItems: "center" }}>
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                    <span style={{ fontSize: 11, color: isPos ? "#4ade80" : wNet < 0 ? "#f87171" : "#93c5fd", letterSpacing: "0.1em", fontWeight: 600 }}>WEEK {weekNum}</span>
                    <span style={{ fontSize: 11, color: "#94a3b8" }}>{getWeekRange(wk)}</span>
                  </div>
                  <div style={{ display: "flex", gap: 20, alignItems: "center", flexWrap: "wrap" }}>
                    <div>
                      <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.08em", marginBottom: 2 }}>TRADING DAYS</div>
                      <div style={{ fontSize: 12, color: "#e2e8f0" }}>
                        <span style={{ color: "#4ade80" }}>{wWinDays}W</span>
                        <span style={{ color: "#475569" }}> / </span>
                        <span style={{ color: "#f87171" }}>{wLossDays}L</span>
                        <span style={{ color: "#64748b", fontSize: 10 }}> · {wEntries.length} day{wEntries.length !== 1 ? "s" : ""}</span>
                      </div>
                    </div>
                    {wA && <>
                      <div>
                        <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.08em", marginBottom: 2 }}>TRADES</div>
                        <div style={{ fontSize: 12, color: "#e2e8f0" }}>
                          <span style={{ color: "#4ade80" }}>{wA.winners}W</span>
                          <span style={{ color: "#475569" }}> / </span>
                          <span style={{ color: "#f87171" }}>{wA.losers}L</span>
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.08em", marginBottom: 2 }}>WIN RATE</div>
                        <div style={{ fontSize: 12, color: wA.winRate >= 50 ? "#4ade80" : "#f87171", fontWeight: 600 }}>{wA.winRate.toFixed(0)}%</div>
                      </div>
                      {wA.profitFactor && <div>
                        <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.08em", marginBottom: 2 }}>PROFIT FACTOR</div>
                        <div style={{ fontSize: 12, color: wA.profitFactor >= 1 ? "#4ade80" : "#f87171", fontWeight: 600 }}>{wA.profitFactor.toFixed(2)}</div>
                      </div>}
                    </>}
                  </div>
                </div>
                <div style={{ display: "flex", gap: 20, alignItems: "flex-end", textAlign: "right" }}>
                  {wFees > 0 && <>
                    <div>
                      <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.08em", marginBottom: 2 }}>GROSS</div>
                      <div style={{ fontSize: 13, color: "#94a3b8" }}>{fmtPnl(wGross)}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.08em", marginBottom: 2 }}>FEES</div>
                      <div style={{ fontSize: 13, color: "#f87171" }}>−${wFees.toFixed(0)}</div>
                    </div>
                  </>}
                  <div>
                    <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.08em", marginBottom: 2 }}>NET P&L</div>
                    <div style={{ fontSize: 22, color: pnlColor(wNet), fontWeight: 700, lineHeight: 1 }}>{fmtPnl(wNet)}</div>
                  </div>
                </div>
              </div>

              {/* Day breakdown */}
              <div style={{ padding: "10px 18px 0", display: "flex", flexDirection: "column", gap: 6 }}>
                {[...wEntries].sort((a, b) => a.date.localeCompare(b.date)).map(e => {
                  const dayNet = netPnl(e);
                  const dayGross = parseFloat(e.pnl) || 0;
                  const dayFees = parseFloat(e.commissions) || 0;
                  const dayA = calcAnalytics(e.parsedTrades || [], aiCfg?.tzLock !== false);
                  const isClean = e.sessionMistakes?.includes("No Mistakes — Executed the Plan ✓");
                  const mistakeCount = e.sessionMistakes?.filter(m => m !== "No Mistakes — Executed the Plan ✓").length || 0;
                  return (
                    <div key={e.id} style={{ display: "grid", gridTemplateColumns: "140px 1fr auto", gap: 14, alignItems: "center", padding: "8px 12px", borderRadius: 4, background: "#0f1729", border: "1px solid #1e293b" }}>
                      <div>
                        <div style={{ fontSize: 11, color: "#e2e8f0", fontWeight: 500 }}>{new Date(e.date + "T12:00:00").toLocaleDateString("en-US", { weekday: "long" })}</div>
                        <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 2 }}>{new Date(e.date + "T12:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric" })}</div>
                      </div>
                      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                        {(e.instruments?.length ? e.instruments : e.instrument ? [e.instrument] : []).map(sym => (
                          <span key={sym} style={{ padding: "2px 8px", borderRadius: 3, fontSize: 10, background: "#1e3a5f", color: "#93c5fd" }}>{sym}</span>
                        ))}
                        {e.grade && <span style={{ padding: "2px 8px", borderRadius: 3, fontSize: 10, background: "#0a0e1a", border: `1px solid ${pnlColor(dayNet)}55`, color: pnlColor(dayNet) }}>{e.grade}</span>}
                        {e.bias && <span style={{ padding: "2px 8px", borderRadius: 3, fontSize: 10, background: e.bias === "Bullish" ? "#052e16" : e.bias === "Bearish" ? "#450a0a" : "#1e1b4b", color: e.bias === "Bullish" ? "#4ade80" : e.bias === "Bearish" ? "#f87171" : "#a5b4fc" }}>{e.bias}</span>}
                        {dayA && <span style={{ fontSize: 10, color: "#94a3b8" }}><span style={{ color: "#4ade80" }}>{dayA.winners}W</span> / <span style={{ color: "#f87171" }}>{dayA.losers}L</span> · <span style={{ color: dayA.winRate >= 50 ? "#4ade80" : "#f87171" }}>{dayA.winRate.toFixed(0)}%</span></span>}
                        {isClean && <span style={{ fontSize: 9, color: "#4ade80", background: "#052e16", border: "1px solid #166534", padding: "1px 6px", borderRadius: 2 }}>✓ Clean</span>}
                        {!isClean && mistakeCount > 0 && <span style={{ fontSize: 9, color: "#f87171", background: "#1f0606", border: "1px solid #7f1d1d", padding: "1px 6px", borderRadius: 2 }}>⚠ {mistakeCount} mistake{mistakeCount !== 1 ? "s" : ""}</span>}
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <div style={{ fontSize: 15, color: pnlColor(dayNet), fontWeight: 700 }}>{fmtPnl(dayNet)}</div>
                        {dayFees > 0 && <div style={{ fontSize: 9, color: "#94a3b8", marginTop: 2 }}>gross {fmtPnl(dayGross)}</div>}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Week-level Mistake Frequency + Mood vs Performance */}
              {(() => {
                // Mistakes
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

                // Moods
                const moodStats = {};
                for (const e of wEntries) {
                  const moods = e.moods?.length ? e.moods : e.mood ? [e.mood] : [];
                  const ep = netPnl(e);
                  for (const m of moods) {
                    if (!moodStats[m]) moodStats[m] = { pnl: 0, sessions: 0, wins: 0 };
                    moodStats[m].pnl += ep;
                    moodStats[m].sessions++;
                    if (ep > 0) moodStats[m].wins++;
                  }
                }
                const moodSorted = Object.entries(moodStats).sort((a, b) => (b[1].pnl / b[1].sessions) - (a[1].pnl / a[1].sessions));
                const maxMoodAvg = Math.max(...moodSorted.map(([, s]) => Math.abs(s.pnl / s.sessions)), 1);

                const hasData = mistakeSorted.length > 0 || cleanDays > 0 || moodSorted.length > 0;
                if (!hasData) return null;

                return (
                  <div style={{ display: "grid", gridTemplateColumns: mistakeSorted.length > 0 && moodSorted.length > 0 ? "1fr 1fr" : "1fr", gap: 10, padding: "10px 18px 14px", borderTop: "1px solid #1e293b", marginTop: 8 }}>
                    {/* Mistake Frequency */}
                    {(mistakeSorted.length > 0 || cleanDays > 0) && (
                      <div style={{ background: "#0a0e1a", borderRadius: 4, padding: "10px 12px" }}>
                        <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 8 }}>MISTAKE FREQUENCY</div>
                        {cleanDays > 0 && (
                          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: mistakeSorted.length ? 6 : 0, padding: "4px 8px", background: "#061f0f", border: "1px solid #166534", borderRadius: 3 }}>
                            <span style={{ fontSize: 10, color: "#4ade80" }}>✓ Clean days</span>
                            <span style={{ fontSize: 10, fontWeight: 600, color: "#4ade80" }}>{cleanDays}</span>
                          </div>
                        )}
                        {mistakeSorted.map(([mistake, count]) => (
                          <div key={mistake} style={{ marginBottom: 6 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                              <span style={{ fontSize: 10, color: "#e2e8f0" }}>{mistake}</span>
                              <span style={{ fontSize: 10, fontWeight: 600, color: "#f87171" }}>{count}×</span>
                            </div>
                            <div style={{ background: "#0f1729", borderRadius: 2, height: 3 }}>
                              <div style={{ width: `${(count / maxMistake) * 100}%`, height: "100%", background: "#f87171", borderRadius: 2, opacity: 0.7 }} />
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Mood vs Performance */}
                    {moodSorted.length > 0 && (
                      <div style={{ background: "#0a0e1a", borderRadius: 4, padding: "10px 12px" }}>
                        <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 8 }}>MOOD VS P&L</div>
                        {moodSorted.map(([mood, s]) => {
                          const avg = s.pnl / s.sessions;
                          const isPos = avg >= 0;
                          return (
                            <div key={mood} style={{ marginBottom: 6 }}>
                              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                                <span style={{ fontSize: 10, color: "#e2e8f0" }}>{mood}</span>
                                <span style={{ fontSize: 10, fontWeight: 600, color: isPos ? "#4ade80" : "#f87171" }}>{avg >= 0 ? "+" : ""}${avg.toFixed(0)}</span>
                              </div>
                              <div style={{ background: "#0f1729", borderRadius: 2, height: 3 }}>
                                <div style={{ width: `${Math.abs(avg) / maxMoodAvg * 100}%`, height: "100%", background: isPos ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })()}
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
    const a = calcAnalytics(trades, aiCfg?.tzLock !== false);
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
  const periodA = calcAnalytics(periodTrades, aiCfg?.tzLock !== false);
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
          <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 22, color: "#e2e8f0", letterSpacing: "0.1em" }}>PERFORMANCE OVERVIEW</div>
          <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>Cumulative results by month · click any month to expand full breakdown</div>
        </div>
        {/* Year selector */}
        <div style={{ display: "flex", gap: 6 }}>
          {years.map(y => (
            <button key={y} onClick={() => { setSelectedYear(y); setExpandedMonth(null); }}
              style={{ padding: "5px 12px", borderRadius: 3, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.05em", transition: "all .15s", background: selectedYear === y ? "#1e3a5f" : "transparent", border: `1px solid ${selectedYear === y ? "#3b82f6" : "#1e293b"}`, color: selectedYear === y ? "#93c5fd" : "#94a3b8" }}>
              {y}
            </button>
          ))}
        </div>
      </div>

      {/* Period selector */}
      <div style={{ display: "flex", gap: 6, marginBottom: 20, flexWrap: "wrap" }}>
        {Object.entries(PERIOD_LABELS).map(([key, label]) => (
          <button key={key} onClick={() => { setPeriod(key); setExpandedMonth(null); }}
            style={{ padding: "6px 14px", borderRadius: 3, fontFamily: "inherit", fontSize: 10, cursor: "pointer", letterSpacing: "0.06em", transition: "all .15s", background: period === key ? "#1e3a5f" : "transparent", border: `1px solid ${period === key ? "#3b82f6" : "#1e293b"}`, color: period === key ? "#93c5fd" : "#94a3b8" }}>
            {label}
          </button>
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
              { l: "TOTAL FEES", v: periodFees > 0 ? `-$${periodFees.toFixed(0)}` : "—", c: "#f87171" },
              { l: "WIN DAYS", v: `${periodWinDays}/${periodLossDays}`, c: "#e2e8f0" },
              { l: "TOTAL TRADES", v: periodTrades.length, c: "#e2e8f0" },
              { l: "WIN RATE", v: periodA ? `${periodA.winRate.toFixed(1)}%` : "—", c: periodA?.winRate >= 50 ? "#4ade80" : "#f87171" },
            ].map(s => (
              <div key={s.l}>
                <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.08em", marginBottom: 4 }}>{s.l}</div>
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
      <div style={{ fontSize: 22, color: "#93c5fd", letterSpacing: "0.08em", fontWeight: 700, fontFamily: "'Bebas Neue', sans-serif", marginBottom: 12 }}>
        {PERIOD_LABELS[period]} · {selectedYear}
      </div>
      {/* Month grid */}
      <div style={{ display: "grid", gridTemplateColumns: period === "year" ? "repeat(4, 1fr)" : period === "h1" || period === "h2" ? "repeat(3, 1fr)" : "repeat(3, 1fr)", gap: 10, marginBottom: 16 }}>
        {monthData.map(({ mi, monthStr, monthEntries, trades, a, netTotal, winDays, lossDays }) => {
          const hasData = monthEntries.length > 0;
          const isExpanded = expandedMonth === monthStr;
          const isFuture = monthStr > `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}`;

          return (
            <div key={monthStr}
              onClick={() => hasData && setExpandedMonth(isExpanded ? null : monthStr)}
              style={{ background: hasData ? (netTotal > 0 ? "#061f0f" : netTotal < 0 ? "#1f0606" : "#0f1729") : isFuture ? "#060810" : "#0a0e1a", border: `1px solid ${isExpanded ? "#3b82f6" : hasData ? (netTotal > 0 ? "#166534" : netTotal < 0 ? "#7f1d1d" : "#1e293b") : "#0f1729"}`, borderRadius: 6, padding: "14px 16px", cursor: hasData ? "pointer" : "default", transition: "all .15s", opacity: isFuture ? 0.4 : 1 }}
              onMouseEnter={e => { if (hasData) e.currentTarget.style.borderColor = netTotal > 0 ? "#22c55e" : netTotal < 0 ? "#ef4444" : "#3b82f6"; }}
              onMouseLeave={e => { if (hasData) e.currentTarget.style.borderColor = isExpanded ? "#3b82f6" : netTotal > 0 ? "#166534" : netTotal < 0 ? "#7f1d1d" : "#1e293b"; }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: hasData ? 10 : 0 }}>
                <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 18, color: hasData ? "#e2e8f0" : "#1e3a5f", letterSpacing: "0.1em" }}>{MONTH_NAMES[mi]}</div>
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
                    {a && <div style={{ fontSize: 10, color: "#64748b" }}>PF {a.profitFactor ? a.profitFactor.toFixed(2) : "—"}</div>}
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
          dowStats[day].tradeWins += trades.filter(t => t.pnl > 0).length;
          dowStats[day].tradeLosses += trades.filter(t => t.pnl < 0).length;
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
                      { l: "TOTAL FEES", v: mFees > 0 ? `-$${mFees.toFixed(2)}` : "—", c: mFees > 0 ? "#f87171" : "#94a3b8" },
                      { l: "NET P&L", v: fmtPnl(mNet), c: pnlColor(mNet), hi: true },
                      { l: "DAY WIN RATE", v: me.length ? `${Math.round(mWins / me.length * 100)}%` : "—", c: mWins / me.length >= 0.5 ? "#4ade80" : "#f87171" },
                      { l: "WIN DAYS", v: mWins, c: "#4ade80" },
                      { l: "LOSS DAYS", v: mLoss, c: "#f87171" },
                      ...(a ? [
                        { l: "LARGEST WIN DAY", v: fmtPnl(Math.max(...me.map(e => netPnl(e)))), c: "#4ade80" },
                        { l: "LARGEST LOSS DAY", v: fmtPnl(Math.min(...me.map(e => netPnl(e)))), c: "#f87171" },
                        { l: "MAX DRAWDOWN", v: `$${a.maxDD.toFixed(2)}`, c: "#f87171" },
                      ] : []),
                    ].map(s => (
                      <div key={s.l} style={{ background: s.hi ? "#0d1f3c" : "#0f1729", border: `1px solid ${s.hi ? "#1e3a5f" : "#1e293b"}`, borderRadius: 4, padding: "10px 12px" }}>
                        <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 4 }}>{s.l}</div>
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
                        { l: "PROFIT FACTOR", v: a.profitFactor ? a.profitFactor.toFixed(2) : "—", c: a.profitFactor >= 1 ? "#4ade80" : "#f87171" },
                        { l: "AVG CONTRACT SIZE", v: `${a.avgQty.toFixed(1)} cts`, c: "#e2e8f0" },
                        { l: "AVG WIN", v: fmtPnl(a.avgWin), c: "#4ade80" },
                        { l: "AVG LOSS", v: fmtPnl(a.avgLoss), c: "#f87171" },
                        { l: "MAX WIN STREAK", v: `${a.maxConsecWins} trades`, c: "#4ade80" },
                        { l: "MAX LOSS STREAK", v: `${a.maxConsecLoss} trades`, c: "#f87171" },
                        { l: "LARGEST WIN", v: fmtPnl(a.largestWin), c: "#4ade80" },
                        { l: "LARGEST LOSS", v: fmtPnl(a.largestLoss), c: "#f87171" },
                      ].map(s => (
                        <div key={s.l} style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "10px 12px" }}>
                          <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 4 }}>{s.l}</div>
                          <div style={{ fontSize: 16, color: s.c, fontWeight: 500 }}>{s.v}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Performance by Time of Day */}
              {a && (() => {
                const sessions = ["Asian Session (6PM–12AM)", "London Session (12AM–9:30AM)", "NY Open (9:30AM–12PM)", "Afternoon Deadzone (12–3PM)", "Power Hour (3–4PM)", "After Hours (4–6PM)"];
                const sessionData = sessions.map(s => ({ name: s, ...(a.bySession[s] || { trades: 0, pnl: 0, wins: 0 }) })).filter(s => s.trades > 0);
                if (!sessionData.length) return null;
                const maxAbs = Math.max(...sessionData.map(s => Math.abs(s.pnl)), 1);
                return (
                  <div>
                    <MonthSectionHeader label="PERFORMANCE BY TIME OF DAY" monthKey={expandedMonth} skey="time" summary={<span style={{ color: "#64748b" }}>{sessionData.length} sessions</span>} />
                    {!isCollapsed(expandedMonth, "time") && (
                      <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px", display: "flex", flexDirection: "column", gap: 12 }}>
                        {sessionData.map(s => {
                          const wr = Math.round(s.wins / s.trades * 100);
                          const barW = Math.abs(s.pnl) / maxAbs * 100;
                          const shortName = s.name === "Asian Session (6PM–12AM)" ? "Asian Session" : s.name === "London Session (12AM–9:30AM)" ? "London Session" : s.name === "NY Open (9:30AM–12PM)" ? "NY Open" : s.name === "Afternoon Deadzone (12–3PM)" ? "Deadzone" : s.name === "Power Hour (3–4PM)" ? "Power Hour" : "After Hours";
                          const timeLabel = s.name === "Asian Session (6PM–12AM)" ? "6:00 PM – 12:00 AM EST" : s.name === "London Session (12AM–9:30AM)" ? "12:00 AM – 9:30 AM EST" : s.name === "NY Open (9:30AM–12PM)" ? "9:30 AM – 12:00 PM EST" : s.name === "Afternoon Deadzone (12–3PM)" ? "12:00 – 3:00 PM EST" : s.name === "Power Hour (3–4PM)" ? "3:00 – 4:00 PM EST" : "4:00 – 6:00 PM EST";
                          return (
                            <div key={s.name}>
                              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 5 }}>
                                <div><span style={{ fontSize: 11, color: "#e2e8f0", fontWeight: 500 }}>{shortName}</span><span style={{ fontSize: 9, color: "#64748b", marginLeft: 8 }}>{timeLabel}</span></div>
                                <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
                                  <span style={{ fontSize: 10, color: wr >= 50 ? "#4ade80" : "#f87171" }}>{wr}% WR</span>
                                  <span style={{ fontSize: 10, letterSpacing: 0 }}><span style={{ color: "#4ade80" }}>{s.wins}</span><span style={{ color: "#94a3b8" }}>/</span><span style={{ color: "#f87171" }}>{s.trades - s.wins}</span></span>
                                  <span style={{ fontSize: 12, fontWeight: 600, color: s.pnl >= 0 ? "#4ade80" : "#f87171", minWidth: 80, textAlign: "right" }}>{s.pnl >= 0 ? "+" : ""}${s.pnl.toFixed(2)}</span>
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
                                <span style={{ fontSize: 12, fontWeight: 600, color: s.pnl >= 0 ? "#4ade80" : "#f87171", minWidth: 80, textAlign: "right" }}>{s.pnl >= 0 ? "+" : ""}${s.pnl.toFixed(2)}</span>
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

                {/* Mistake Frequency — expanded month */}
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
                  const sorted = Object.entries(mistakeCounts).sort((a, b) => b[1] - a[1]);
                  if (sorted.length === 0 && cleanDays === 0) return null;
                  const maxCount = sorted[0]?.[1] || 1;
                  return (
                    <div>
                      <MonthSectionHeader label="MISTAKE FREQUENCY" monthKey={expandedMonth} skey="mistakes"
                        summary={<span style={{ color: "#64748b" }}>{sorted.length} type{sorted.length !== 1 ? "s" : ""} · {cleanDays} clean</span>} />
                      {!isCollapsed(expandedMonth, "mistakes") && (
                        <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px" }}>
                          {cleanDays > 0 && (
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: sorted.length ? 10 : 0, padding: "6px 10px", background: "#061f0f", border: "1px solid #166534", borderRadius: 4 }}>
                              <span style={{ fontSize: 11, color: "#4ade80" }}>✓ Executed the Plan</span>
                              <span style={{ fontSize: 12, fontWeight: 600, color: "#4ade80" }}>{cleanDays} day{cleanDays !== 1 ? "s" : ""}</span>
                            </div>
                          )}
                          {sorted.map(([mistake, count]) => {
                            const freq = Math.round((count / me.length) * 100);
                            return (
                              <div key={mistake} style={{ marginBottom: 10 }}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
                                  <span style={{ fontSize: 11, color: "#e2e8f0" }}>{mistake}</span>
                                  <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                                    <span style={{ fontSize: 9, color: "#94a3b8" }}>{freq}% of sessions</span>
                                    <span style={{ fontSize: 12, fontWeight: 600, color: "#f87171" }}>{count}×</span>
                                  </div>
                                </div>
                                <div style={{ background: "#0a0e1a", borderRadius: 2, height: 4, overflow: "hidden" }}>
                                  <div style={{ width: `${(count / maxCount) * 100}%`, height: "100%", background: "#f87171", borderRadius: 2, opacity: 0.7 }} />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })()}

                {/* Mood vs Performance — expanded month */}
                {(() => {
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
                  if (moodList.length === 0) return null;
                  const maxAvg = Math.max(...moodList.map(([, s]) => Math.abs(s.pnl / s.sessions)), 1);
                  return (
                    <div>
                      <MonthSectionHeader label="MOOD VS PERFORMANCE" monthKey={expandedMonth} skey="mood"
                        summary={<span style={{ color: "#64748b" }}>{moodList.length} mood{moodList.length !== 1 ? "s" : ""} tracked</span>} />
                      {!isCollapsed(expandedMonth, "mood") && (
                        <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 4, padding: "14px 16px", display: "flex", flexDirection: "column", gap: 10 }}>
                          {moodList.map(([mood, s]) => {
                            const avg = s.pnl / s.sessions;
                            const wr = Math.round((s.wins / s.sessions) * 100);
                            const isPos = avg >= 0;
                            return (
                              <div key={mood}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
                                  <span style={{ fontSize: 11, color: "#e2e8f0" }}>{mood}</span>
                                  <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                                    <span style={{ fontSize: 9, color: "#94a3b8" }}>{s.sessions} session{s.sessions !== 1 ? "s" : ""} · {wr}% WR</span>
                                    <span style={{ fontSize: 11, fontWeight: 600, color: isPos ? "#4ade80" : "#f87171", minWidth: 70, textAlign: "right" }}>{avg >= 0 ? "+" : ""}${avg.toFixed(0)} avg</span>
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
                                Best mindset: <span style={{ color: "#4ade80" }}>{best[0]}</span> (+${(best[1].pnl / best[1].sessions).toFixed(0)} avg).
                                {(worst[1].pnl / worst[1].sessions) < 0 && <> Watch for <span style={{ color: "#f87171" }}>{worst[0]}</span> sessions (${(worst[1].pnl / worst[1].sessions).toFixed(0)} avg).</>}
                              </div>
                            );
                          })()}
                        </div>
                      )}
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

function AIRecapView({ entries, netPnl: calcNetPnlProp, fmtPnl, pnlColor, initMode = "weekly", ai }) {
  const calcNetPnl = calcNetPnlProp;
  const [recapMode, setRecapMode] = useState(initMode);
  const [selectedPeriod, setSelectedPeriod] = useState(null);
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);
  const [generated, setGenerated] = useState({});

  // Group entries into ISO weeks (Mon–Sun)
  const getWeekKey = (dateStr) => {
    const d = new Date(dateStr + "T12:00:00");
    const day = d.getDay();
    const diff = d.getDate() - day + (day === 0 ? -6 : 1);
    const mon = new Date(d.setDate(diff));
    return `${mon.getFullYear()}-W${String(Math.ceil((mon.getDate() + new Date(mon.getFullYear(), 0, 1).getDay() - 1) / 7)).padStart(2, "0")}`;
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

  const buildPrompt = (periodEntries, label) => {
    // Per-entry full data including computed trade stats
    const notes = periodEntries.map(e => {
      const trades = e.parsedTrades || [];
      const winners = trades.filter(t => t.pnl > 0);
      const losers = trades.filter(t => t.pnl < 0);
      const avgWin = winners.length ? (winners.reduce((s,t) => s+t.pnl,0)/winners.length).toFixed(2) : null;
      const avgLoss = losers.length ? Math.abs(losers.reduce((s,t) => s+t.pnl,0)/losers.length).toFixed(2) : null;
      const profitFactor = losers.length && avgLoss > 0 ? (winners.reduce((s,t)=>s+t.pnl,0)/Math.abs(losers.reduce((s,t)=>s+t.pnl,0))).toFixed(2) : null;
      const maxDD = trades.reduce((acc, t) => { acc.peak = Math.max(acc.peak, acc.running += t.pnl); acc.dd = Math.min(acc.dd, acc.running - acc.peak); return acc; }, { running:0, peak:0, dd:0 }).dd;
      const tradeLog = trades.map((t,i) => {
        const netT = (t.pnl-(t.commission||0)).toFixed(2);
        const notesStr = t.notes ? ` [${t.notes}]` : "";
        return `    Trade ${i+1}: ${t.symbol} qty:${t.qty} ${t.direction||""} | ${t.orderType||"MKT"} | entry $${t.buyPrice}@${t.buyTime?.split(" ")[1]||"?"} exit $${t.sellPrice}@${t.sellTime?.split(" ")[1]||"?"} hold:${t.duration||"?"} | gross $${t.pnl?.toFixed(2)} comm -$${(t.commission||0).toFixed(2)} net $${netT}${notesStr}`;
      }).join("\n");

      const parts = [];
      parts.push(`DATE: ${e.date} (${new Date(e.date+"T12:00:00").toLocaleDateString("en-US",{weekday:"long"})})`);
      if (e.instruments?.length || e.instrument) parts.push(`  Instruments: ${(e.instruments?.length ? e.instruments : [e.instrument]).join(", ")}`);
      parts.push(`  Net P&L: $${netPnl(e).toFixed(2)} | Gross: $${parseFloat(e.pnl||0).toFixed(2)} | Fees: $${parseFloat(e.commissions||0).toFixed(2)}`);
      if (e.grade) parts.push(`  Grade: ${e.grade}`);
      if (e.bias) parts.push(`  Bias: ${e.bias}`);
      const moods = e.moods?.length ? e.moods : e.mood ? [e.mood] : [];
      if (moods.length) parts.push(`  Mood: ${moods.join(", ")}`);
      if (e.sessionMistakes?.length) parts.push(`  Session Mistakes: ${e.sessionMistakes.join(", ")}`);
      // Feature 3: Dual scores
      if (e.executionScore != null || e.decisionScore != null) {
        const scoreStr = [
          e.executionScore != null && `Execution ${e.executionScore}/10`,
          e.decisionScore  != null && `Decision Quality ${e.decisionScore}/10`,
        ].filter(Boolean).join(" | ");
        parts.push(`  Scores: ${scoreStr}`);
      }
      // Feature 4: Mistake costs
      if (e.mistakeCosts && Object.keys(e.mistakeCosts).length) {
        const costLines = Object.entries(e.mistakeCosts)
          .filter(([, v]) => v != null && Number(v) > 0)
          .map(([tag, cost]) => `    - ${tag}: $${Number(cost).toFixed(2)}`);
        if (costLines.length) {
          const total = costLines.reduce((s, line) => s + parseFloat(line.split("$")[1] || 0), 0);
          parts.push(`  Mistake Cost Attribution:\n${costLines.join("\n")}\n    Total: $${total.toFixed(2)} (vs net P&L $${netPnl(e).toFixed(2)})${e.mistakeCostNotes ? `\n    Cost Notes: ${e.mistakeCostNotes}` : ""}`);
        }
      }
      if (trades.length) {
        parts.push(`  Trades: ${trades.length} total | ${winners.length}W / ${losers.length}L | ${trades.length ? ((winners.length/trades.length)*100).toFixed(0) : 0}% WR`);
        if (avgWin) parts.push(`  Avg Win: $${avgWin} | Avg Loss: $${avgLoss} | Profit Factor: ${profitFactor||"N/A"}`);
        if (maxDD < 0) parts.push(`  Max Intraday Drawdown: $${maxDD.toFixed(2)}`);
        parts.push(`  Trade Log:\n${tradeLog}`);
      }
      // Use AI-polished rewrites where available
      const erw = e.aiRewrites || {};
      const enote = (key) => (erw[key]?.trim() || e[key] || "");
      if (e.aiNoteSummary) parts.push(`  Journal Narrative: ${e.aiNoteSummary}`);
      if (enote("marketNotes")) parts.push(`  Market Notes: ${enote("marketNotes")}`);
      if (enote("rules")) parts.push(`  Rules Followed/Broken: ${enote("rules")}`);
      if (enote("lessonsLearned")) parts.push(`  Lessons Learned: ${enote("lessonsLearned")}`);
      if (enote("mistakes")) parts.push(`  Mistakes (freeform): ${enote("mistakes")}`);
      if (enote("improvements")) parts.push(`  Improvements: ${enote("improvements")}`);
      if (enote("reinforceRule")) parts.push(`  Rule To Reinforce: ${enote("reinforceRule")}`);
      if (enote("bestTrade")) parts.push(`  Best Trade: ${enote("bestTrade")}`);
      if (enote("worstTrade")) parts.push(`  Worst Trade: ${enote("worstTrade")}`);
      if (enote("tomorrow")) parts.push(`  Tomorrow's Plan: ${enote("tomorrow")}`);
      if (e.mistakeCostNotes) parts.push(`  Mistake Cost Notes: ${e.mistakeCostNotes}`);
      return parts.join("\n");
    }).join("\n\n---\n\n");

    const totalPnl = periodEntries.reduce((s, e) => s + netPnl(e), 0);
    const wins = periodEntries.filter(e => netPnl(e) > 0).length;
    const losses = periodEntries.filter(e => netPnl(e) < 0).length;
    const allTrades = periodEntries.flatMap(e => e.parsedTrades || []);
    const allWinners = allTrades.filter(t => t.pnl > 0);
    const allLosers = allTrades.filter(t => t.pnl < 0);
    const overallWR = allTrades.length ? ((allWinners.length/allTrades.length)*100).toFixed(1) : "N/A";
    const overallPF = allLosers.length ? (allWinners.reduce((s,t)=>s+t.pnl,0)/Math.abs(allLosers.reduce((s,t)=>s+t.pnl,0))).toFixed(2) : "N/A";
    const avgDailyPnl = (totalPnl / periodEntries.length).toFixed(2);
    // Order type aggregation across period
    const periodOT = {};
    for (const t of allTrades) { const k = t.orderType||"MKT"; if (!periodOT[k]) periodOT[k]={trades:0,pnl:0,wins:0}; periodOT[k].trades++; periodOT[k].pnl+=t.pnl; if(t.pnl>0)periodOT[k].wins++; }
    const orderTypeSummary = Object.entries(periodOT).filter(([,d])=>d.trades>0).map(([ot,d])=>`${ot}: ${d.trades} trades, ${Math.round(d.wins/d.trades*100)}% WR, $${d.pnl.toFixed(2)} gross`).join(" | ") || "none";
    // Commission drag across period
    const periodTotalComm = allTrades.reduce((s,t)=>s+(t.commission||0),0);
    const periodGross = allTrades.reduce((s,t)=>s+t.pnl,0);
    const periodCommDrag = Math.abs(periodGross)>0 ? (periodTotalComm/Math.abs(periodGross)*100).toFixed(1) : "N/A";
    // Duration breakdown across period
    const periodDurBuckets = {"<1m":{t:0,w:0,pnl:0},"1-5m":{t:0,w:0,pnl:0},"5-15m":{t:0,w:0,pnl:0},"15-60m":{t:0,w:0,pnl:0},">1h":{t:0,w:0,pnl:0}};
    for (const t of allTrades) { const s=t.durationSecs||0; const k=s<60?"<1m":s<300?"1-5m":s<900?"5-15m":s<3600?"15-60m":">1h"; periodDurBuckets[k].t++; periodDurBuckets[k].pnl+=t.pnl; if(t.pnl>0)periodDurBuckets[k].w++; }
    const durationSummary = Object.entries(periodDurBuckets).filter(([,d])=>d.t>0).map(([k,d])=>`${k}: ${d.t} trades ${Math.round(d.w/d.t*100)}% WR $${d.pnl.toFixed(0)}`).join(" | ") || "none";
    // Session breakdown across period
    const periodSess = {};
    const _getSess = (sellTime) => { if (!sellTime) return "Unknown"; const p=sellTime.split(" ")[1]||sellTime; const [h,m]=(p.split(":")).map(Number); const mins=h*60+m; return mins<360?"Asian":mins<570?"London":mins<720?"NY Open":mins<900?"Afternoon Deadzone":mins<960?"Power Hour":"After Hours"; };
    for (const t of allTrades) { const k=_getSess(t.sellTime); if(!periodSess[k])periodSess[k]={trades:0,pnl:0,wins:0}; periodSess[k].trades++; periodSess[k].pnl+=t.pnl; if(t.pnl>0)periodSess[k].wins++; }
    const periodSessionSummary = Object.entries(periodSess).filter(([,d])=>d.trades>0).sort((a,b)=>b[1].pnl-a[1].pnl).map(([k,d])=>`${k}: ${d.trades} trades ${Math.round(d.wins/d.trades*100)}% WR $${d.pnl.toFixed(2)}`).join(" | ") || "none";
    // Symbol breakdown across period
    const periodSyms = {};
    for (const t of allTrades) { if(!periodSyms[t.symbol])periodSyms[t.symbol]={trades:0,pnl:0,wins:0}; periodSyms[t.symbol].trades++; periodSyms[t.symbol].pnl+=t.pnl; if(t.pnl>0)periodSyms[t.symbol].wins++; }
    const symbolSummary = Object.entries(periodSyms).sort((a,b)=>b[1].pnl-a[1].pnl).map(([sym,d])=>`${sym}: ${d.trades} trades ${Math.round(d.wins/d.trades*100)}% WR $${d.pnl.toFixed(2)}`).join(" | ") || "none";

    // Mistake frequency tally
    const mistakeCounts = {};
    for (const e of periodEntries) for (const m of (e.sessionMistakes || [])) mistakeCounts[m] = (mistakeCounts[m] || 0) + 1;
    const mistakeTally = Object.entries(mistakeCounts).sort((a,b)=>b[1]-a[1])
      .map(([m,n]) => `  ${m}: ${n}x (${Math.round(n/periodEntries.length*100)}% of days)`)
      .join("\n") || "  None flagged";

    // Compile all written notes by field across the period for theme mining
    // Prefer polished rewrites over raw text
    const getNote = (e, key) => (e.aiRewrites?.[key]?.trim() || e[key] || "");
    const allNarratives = periodEntries.filter(e=>e.aiNoteSummary).map(e=>`[${e.date}] ${e.aiNoteSummary}`).join("\n");
    const allLessons = periodEntries.filter(e=>getNote(e,"lessonsLearned")).map(e=>`[${e.date}] ${getNote(e,"lessonsLearned")}`).join("\n");
    const allMistakeNotes = periodEntries.filter(e=>getNote(e,"mistakes")).map(e=>`[${e.date}] ${getNote(e,"mistakes")}`).join("\n");
    const allImprovements = periodEntries.filter(e=>getNote(e,"improvements")).map(e=>`[${e.date}] ${getNote(e,"improvements")}`).join("\n");
    const allRules = periodEntries.filter(e=>getNote(e,"rules")).map(e=>`[${e.date}] ${getNote(e,"rules")}`).join("\n");
    const allMarketNotes = periodEntries.filter(e=>getNote(e,"marketNotes")).map(e=>`[${e.date}] ${getNote(e,"marketNotes")}`).join("\n");
    const allReinforceRules = periodEntries.filter(e=>getNote(e,"reinforceRule")).map(e=>`[${e.date}] ${getNote(e,"reinforceRule")}`).join("\n");
    const allTomorrowPlans = periodEntries.filter(e=>getNote(e,"tomorrow")).map(e=>`[${e.date}] ${getNote(e,"tomorrow")}`).join("\n");
    const allBestTrades = periodEntries.filter(e=>getNote(e,"bestTrade")).map(e=>`[${e.date}] ${getNote(e,"bestTrade")}`).join("\n");
    const allWorstTrades = periodEntries.filter(e=>getNote(e,"worstTrade")).map(e=>`[${e.date}] ${getNote(e,"worstTrade")}`).join("\n");
    const allMistakeCostNotes = periodEntries.filter(e=>e.mistakeCostNotes).map(e=>`[${e.date}] ${e.mistakeCostNotes}`).join("\n");
    const scoreEntries = periodEntries.filter(e=>e.executionScore!=null||e.decisionScore!=null);
    const scoreVsPnl = scoreEntries.map(e=>{ const net=netPnl(e); return `[${e.date}] exec:${e.executionScore??"—"}/10 dec:${e.decisionScore??"—"}/10 net:$${net.toFixed(0)} ${net<0&&(e.executionScore||0)>=7?"⚠ high score on losing day":""}`; }).join("\n");

    // Feature 4: Mistake cost tally across period
    const mistakeCostTotals = {};
    for (const e of periodEntries) {
      if (!e.mistakeCosts) continue;
      for (const [tag, cost] of Object.entries(e.mistakeCosts)) {
        if (cost == null || !Number.isFinite(Number(cost)) || Number(cost) <= 0) continue;
        mistakeCostTotals[tag] = (mistakeCostTotals[tag] || 0) + Number(cost);
      }
    }
    const totalMistakeCost = Object.values(mistakeCostTotals).reduce((s, v) => s + v, 0);
    const mistakeCostSummary = Object.entries(mistakeCostTotals).sort((a,b)=>b[1]-a[1])
      .map(([tag, cost]) => `  ${tag}: $${cost.toFixed(2)}`)
      .join("\n") || "  No cost data attributed";

    // Feature 3: Average scores
    const scoredEntries = periodEntries.filter(e => e.executionScore != null || e.decisionScore != null);
    const avgExec = scoredEntries.length ? (scoredEntries.filter(e=>e.executionScore!=null).reduce((s,e)=>s+e.executionScore,0) / Math.max(1,scoredEntries.filter(e=>e.executionScore!=null).length)).toFixed(1) : null;
    const avgDec  = scoredEntries.length ? (scoredEntries.filter(e=>e.decisionScore!=null).reduce((s,e)=>s+e.decisionScore,0)  / Math.max(1,scoredEntries.filter(e=>e.decisionScore!=null).length)).toFixed(1)  : null;

    // Grade distribution
    const grades = periodEntries.filter(e => e.grade).map(e => e.grade);
    const gradeDist = grades.length ? [...new Set(grades)].map(g=>`${g}:${grades.filter(x=>x===g).length}`).join(", ") : "None logged";

    // Day of week breakdown
    const DOW = ["Monday","Tuesday","Wednesday","Thursday","Friday"];
    const dowStats = {};
    for (const day of DOW) dowStats[day] = { pnl:0, wins:0, days:0 };
    for (const e of periodEntries) {
      const day = DOW[new Date(e.date+"T12:00:00").getDay()-1];
      if (!day) continue;
      dowStats[day].pnl += netPnl(e);
      dowStats[day].days++;
      if (netPnl(e) > 0) dowStats[day].wins++;
    }
    const dowSummary = DOW.filter(d => dowStats[d].days > 0)
      .map(d => `  ${d}: ${dowStats[d].days} days, $${dowStats[d].pnl.toFixed(2)} net, ${Math.round(dowStats[d].wins/dowStats[d].days*100)}% WR`)
      .join("\n") || "  No data";

    // Mood/grade correlation
    const moodGrade = periodEntries.filter(e => (e.moods?.length || e.mood) && e.grade)
      .map(e => `${(e.moods?.length ? e.moods : [e.mood]).join("+")}→${e.grade}`)
      .join(", ") || "None";

    // Plan-violation cross-reference: for each entry, look at the PREVIOUS entry's "tomorrow" plan
    const sortedEntries = [...periodEntries].sort((a, b) => a.date.localeCompare(b.date));
    const planViolations = [];
    for (let i = 1; i < sortedEntries.length; i++) {
      const prev = sortedEntries[i - 1];
      const curr = sortedEntries[i];
      if (prev.tomorrow && prev.tomorrow.trim()) {
        planViolations.push({
          planDate: prev.date,
          tradeDate: curr.date,
          plan: prev.tomorrow.trim(),
          rule: prev.reinforceRule || null,
          actualMistakes: curr.sessionMistakes || [],
          actualGrade: curr.grade || null,
          actualNetPnl: netPnl(curr),
        });
      }
    }
    const planViolationSummary = planViolations.length > 0
      ? planViolations.map(v =>
          `  [${v.planDate} plan → ${v.tradeDate} actual]\n  Plan stated: "${v.plan}"${v.rule ? `\n  Rule to reinforce: "${v.rule}"` : ""}\n  Actual session grade: ${v.actualGrade || "not logged"} | Net P&L: $${v.actualNetPnl.toFixed(2)}${v.actualMistakes.length ? `\n  Mistakes flagged on actual day: ${v.actualMistakes.join(", ")}` : ""}`
        ).join("\n\n")
      : "  No consecutive entries with plans in this period";

    return `You are a professional trading coach with access to a futures trader's COMPLETE journal data for the period: ${label}. You have every trade, every written note, and all behavioral data. Your analysis should feel like it was written by someone who has personally reviewed every session.

ACCOUNT TYPE: ${activeJournal?.type === JOURNAL_TYPES.PROP ? `Prop Firm — ${activeJournal?.config?.firmName || "Prop"} (${activeJournal?.config?.phase === "funded" ? "Funded Account" : "Challenge/Evaluation"}, $${(activeJournal?.config?.accountSize || 0).toLocaleString()} account)` : "Personal Account"}
${activeJournal?.type === JOURNAL_TYPES.PROP && propStatus ? `PROP STATUS: Profit target ${propStatus.profitTargetPct.toFixed(1)}% complete ($${propStatus.profitTargetProgress.toFixed(2)} of $${activeJournal.config.profitTarget}), trailing drawdown ${propStatus.trailingDrawdownPct.toFixed(1)}% used, consistency rule at ${propStatus.consistencyPct.toFixed(1)}% (limit ${activeJournal.config.consistencyRule}%), ${propStatus.daysTraded} of ${activeJournal.config.minTradingDays} min days traded` : ""}

PERIOD SUMMARY:
- Trading days: ${periodEntries.length} (${wins}W / ${losses}L) | Avg daily P&L: $${avgDailyPnl}
- Net P&L: $${totalPnl.toFixed(2)}
- Total trades: ${allTrades.length} | Trade win rate: ${overallWR}% | Profit factor: ${overallPF}
- Grade distribution: ${gradeDist}
${avgExec || avgDec ? `- Avg Scores: ${[avgExec && `Execution ${avgExec}/10`, avgDec && `Decision ${avgDec}/10`].filter(Boolean).join(" | ")}` : ""}
- Total commissions: $${periodTotalComm.toFixed(2)} (${periodCommDrag}% commission drag on gross $${periodGross.toFixed(2)})

ORDER TYPE BREAKDOWN (aggregated across all trades this period):
${orderTypeSummary}

HOLD TIME BREAKDOWN (duration buckets — where is edge concentrated?):
${durationSummary}

SESSION WINDOW BREAKDOWN (aggregated across all trades this period):
${periodSessionSummary}

SYMBOL BREAKDOWN:
${symbolSummary}

DAY OF WEEK BREAKDOWN:
${dowSummary}

SESSION MISTAKE FREQUENCY:
${mistakeTally}

MISTAKE COST ATTRIBUTION (dollar impact of behavioral habits):
${mistakeCostSummary}
${totalMistakeCost > 0 ? `Total attributed mistake cost: $${totalMistakeCost.toFixed(2)} over the period` : ""}

MOOD → GRADE CORRELATION PAIRS:
${moodGrade}

PLAN-VS-EXECUTION LOG (previous day's plan vs actual next-day behavior):
${planViolationSummary}

SCORE VS P&L LOG (flag days with high self-scores but negative results — rating bias):
${scoreVsPnl||"  No scores logged"}

COMPILED WRITTEN NOTES — ALL ENTRIES (for theme and pattern mining):
${allNarratives ? `JOURNAL NARRATIVES (AI-consolidated per day):\n${allNarratives}\n\n` : ""}LESSONS LEARNED:
${allLessons||"  None written"}

MISTAKES (freeform):
${allMistakeNotes||"  None written"}

RULES FOLLOWED/BROKEN:
${allRules||"  None written"}

AREAS FOR IMPROVEMENT:
${allImprovements||"  None written"}

MARKET NOTES:
${allMarketNotes||"  None written"}

RULES TO REINFORCE:
${allReinforceRules||"  None written"}

TOMORROW PLANS WRITTEN:
${allTomorrowPlans||"  None written"}

BEST TRADE DESCRIPTIONS:
${allBestTrades||"  None written"}

WORST TRADE DESCRIPTIONS:
${allWorstTrades||"  None written"}
${allMistakeCostNotes ? `\nMISTAKE COST NOTES:\n${allMistakeCostNotes}` : ""}

FULL JOURNAL ENTRIES (with complete trade logs and all written notes):
${notes}

You have everything. Every trade price, time, and P&L. Every word the trader wrote across every day. Every mistake they flagged. Every lesson, plan, and rule. Cross-reference all of it ruthlessly.

Provide a structured recap with exactly these 7 sections:

**📊 PERFORMANCE OVERVIEW**
3-4 sentences. Lead with the key number (P&L, win rate, profit factor). Note consistency — were wins and losses spread evenly or clustered? Call out any meaningful patterns in the order type, session, or duration breakdowns. Any grade or day-of-week pattern worth flagging immediately?

**🔑 KEY LESSONS IDENTIFIED**
Read ALL compiled written notes above (lessons learned, mistakes, rules, improvements, plans). Extract the top 3-4 recurring themes by looking for: the same word, concept, or situation appearing across multiple days; plans that were written but never acted on; lessons that were re-learned (same lesson written on multiple dates). For each theme: (1) quote or closely paraphrase the trader's exact words with dates, (2) count how many days it appeared, (3) cross-reference against the trade data — did the trade results improve on days they acknowledged this lesson? Show them you read every word they actually wrote — do not generalize or invent.

**⚠️ PATTERNS & MISTAKES**
This is the most important section. Two layers: (A) FLAGGED MISTAKES — lead with the session mistake frequency tally. For each repeated mistake: name it, how many times it was flagged, what the trade data shows on those days (worse P&L? more trades? larger losses?), dollar cost if quantifiable from mistake cost data, and root cause. (B) WRITTEN NOTES PATTERNS — scan ALL the compiled freeform notes above. Find: the same mistake described in different words across multiple days (name the underlying pattern); rules the trader wrote but the trades show were broken; self-improvement intentions that never materialized (written 3+ times = habitual blind spot). For unacknowledged patterns visible in the trade data but absent from any notes, name them explicitly — these are the trader's blind spots. Be direct. Use dates and dollar amounts.

**🚩 RED FLAGS — PLAN VS. EXECUTION**
Analyze the Plan-vs-Execution log above. For every consecutive entry pair where a plan existed: (1) State what was planned in quotes, (2) Identify whether the next day's actual behavior honored or violated the plan — use the session mistakes, grade, and P&L as evidence, (3) Label each as HONORED or VIOLATED with a one-line explanation. If violations are repeated across multiple days (e.g. the trader keeps writing "no revenge trading" then flagging revenge trading), call out the pattern explicitly with dates. If no plan data exists for this period, skip this section.

**🧠 BEHAVIORAL TRENDS**
3 observations the trader likely hasn't noticed themselves. Use the order type breakdown, hold time breakdown, and session window breakdown — cross-reference them against P&L. Look for: which day of week and session their edge is strongest/weakest, whether mood correlates with grade, whether mistake clusters appear on winning or losing days, whether LMT vs STP vs MKT entries show meaningfully different results, whether shorter or longer holds produce better outcomes. Commission drag: if fees are >5% of gross P&L, flag it explicitly. Cite specific dates or data points.

**🪞 SELF-AWARENESS AUDIT**
Cross-reference the Score vs P&L log and written plans vs actual execution. Answer: (1) Is the trader's self-assessment (execution/decision scores) calibrated to their actual results, or do they rate themselves highly on losing days? Name specific dates. (2) Plans written for tomorrow — were they actually followed the next day? Quote the plan and describe what happened. (3) Rules written to reinforce — did the trader honor them the following session? This section is about whether the trader's self-knowledge matches reality. Skip if no score or plan data exists.

**💡 STRENGTHS TO BUILD ON**
2-3 genuine positives backed by data. Which sessions, setups, or behavioral patterns produced the best results? Be specific — "Tuesday NY Open trades averaged +$X" not "you trade well sometimes."

**🎯 ACTIONABLE FOCUS POINTS**
3 rules for next period. Format strictly as: [Root Cause] → [Measurable Rule]. Tie each directly to the most damaging repeated mistake or behavioral pattern — including any repeated plan violations. Make them specific enough to evaluate at end of next session.

Tone: trusted mentor who has studied every trade. Direct, data-driven, no filler. When calling out plan violations, be unflinching — the trader wrote the plan themselves. Accountability is the point.`;
  };

  const generateSummary = async (period) => {
    const periodEntries = grouped[period] || [];
    const label = periodLabel(period);
    setSelectedPeriod(period);
    setSummary("");

    if (generated[period]) { setSummary(generated[period]); return; }

    const hasNotes = periodEntries.some(e =>
      e.lessonsLearned || e.mistakes || e.improvements || e.marketNotes ||
      e.bestTrade || e.worstTrade || e.tomorrow || e.rules || e.reinforceRule ||
      (e.parsedTrades?.length > 0)
    );
    if (!hasNotes) { setSummary("NO_NOTES"); return; }
    if (!ai?.enabled || !ai?.apiKey) { setSummary("AI_NOT_CONFIGURED"); return; }

    setLoading(true);
    try {
      const prompt = buildPrompt(periodEntries, label);
      const cacheKey = await sha256(`${ai?.provider || "anthropic"}|${ai?.model || ''}|periodRecap|${prompt}`);
      const cached = getCachedAiText(cacheKey);
      if (cached) {
        setSummary(cached);
        setGenerated(prev => ({ ...prev, [period]: cached }));
        setLoading(false);
        return;
      }

      const txt = await aiRequestText(ai, {
        max_tokens: 2000,
        timeoutMs: 30000,
        messages: [{ role: 'user', content: prompt }],
      });
      setCachedAiText(cacheKey, txt);
      setSummary(txt);
      setGenerated(prev => ({ ...prev, [period]: txt }));
    } catch (err) {
      const f = friendlyAiError(err);
      console.warn('Recap failed:', f.code, f.message);
      setSummary("ERROR");
    }
    setLoading(false);
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
    if (text === "ERROR") return (
      <div style={{ textAlign: "center", padding: "40px 20px", color: "#f87171", fontSize: 13 }}>
        Failed to generate summary. Please try again.
      </div>
    );
    return <RenderAI text={text} />;
  };

  return (
    <div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 20 }}>
        <span style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 22, color: "#93c5fd", letterSpacing: "0.08em", fontWeight: 700 }}>🤖 AI RECAP</span>
        <span style={{ fontSize: 10, color: "#64748b", letterSpacing: "0.1em" }}>POWERED BY CLAUDE</span>
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
                  style={{ padding: "10px 0", fontFamily: "inherit", fontSize: 10, letterSpacing: "0.08em", cursor: "pointer", border: "none", background: recapMode === m ? "#1e3a5f" : "transparent", color: recapMode === m ? "#93c5fd" : "#94a3b8", textTransform: "uppercase", transition: "all .15s" }}>
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
                const hasNotes = es.some(e => e.lessonsLearned || e.mistakes || e.improvements || e.marketNotes);
                const isSelected = selectedPeriod === p;
                const isGenerated = !!generated[p];
                const isLoading = loading && selectedPeriod === p;
                return (
                  <div key={p}
                    style={{ padding: "12px 14px", borderBottom: "1px solid #0f1729", background: isSelected ? "#0a1628" : "transparent", borderLeft: isSelected ? "2px solid #3b82f6" : "2px solid transparent", transition: "all .15s" }}
                    onMouseEnter={e => { if (!isSelected) e.currentTarget.style.background = "#0d1526"; }}
                    onMouseLeave={e => { if (!isSelected) e.currentTarget.style.background = isSelected ? "#0a1628" : "transparent"; }}>
                    <div style={{ fontSize: 10, color: isSelected ? "#93c5fd" : "#94a3b8", marginBottom: 4, letterSpacing: "0.05em" }}>{periodLabel(p)}</div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                      <span style={{ fontSize: 12, fontWeight: 600, color: pnl >= 0 ? "#4ade80" : "#f87171" }}>{pnl >= 0 ? "+" : ""}${Math.abs(pnl).toLocaleString("en-US", { maximumFractionDigits: 0 })}</span>
                      <span style={{ fontSize: 9, letterSpacing: 0 }}><span style={{ color: "#4ade80" }}>{wins}</span><span style={{ color: "#94a3b8" }}>/</span><span style={{ color: "#f87171" }}>{es.length - wins}</span></span>
                    </div>
                    {!hasNotes && <div style={{ fontSize: 9, color: "#1e3a5f", marginBottom: 6 }}>no notes</div>}
                    <button
                      onClick={() => generateSummary(p)}
                      disabled={isLoading}
                      style={{ width: "100%", padding: "7px 10px", borderRadius: 4, fontFamily: "inherit", fontSize: 10, letterSpacing: "0.06em", cursor: isLoading ? "not-allowed" : "pointer", transition: "all .15s", background: isLoading ? "transparent" : isGenerated ? "transparent" : "#1d4ed8", border: isLoading ? "1px solid #1e293b" : isGenerated ? "1px solid #1e293b" : "none", color: isLoading ? "#475569" : isGenerated ? "#94a3b8" : "white" }}>
                      {isLoading ? "ANALYSING..." : isGenerated ? "↺ REGENERATE" : `ANALYSE ${recapMode === "weekly" ? "WEEK" : "MONTH"} →`}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right: summary panel */}
          <div style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 6, minHeight: 400 }}>
            {!selectedPeriod ? (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: 400, gap: 12, color: "#64748b" }}>
                <div style={{ fontSize: 28 }}>✦</div>
                <div style={{ fontSize: 12, letterSpacing: "0.1em" }}>SELECT A PERIOD TO GENERATE YOUR RECAP</div>
                <div style={{ fontSize: 11, color: "#1e3a5f" }}>Powered by Claude AI</div>
              </div>
            ) : (
              <div style={{ padding: "20px 24px" }}>
                {/* Period header */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20, paddingBottom: 16, borderBottom: "1px solid #1e293b" }}>
                  <div>
                    <div style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 4 }}>{recapMode.toUpperCase()} RECAP</div>
                    <div style={{ fontSize: 15, color: "#e2e8f0", fontWeight: 600 }}>{periodLabel(selectedPeriod)}</div>
                  </div>
                  {(() => {
                    const es = grouped[selectedPeriod] || [];
                    const pnl = es.reduce((s, e) => s + netPnl(e), 0);
                    const wins = es.filter(e => netPnl(e) > 0).length;
                    const wr = es.length ? Math.round(wins / es.length * 100) : 0;
                    return (
                      <div style={{ display: "flex", gap: 16, textAlign: "right" }}>
                        <div><div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.08em" }}>NET P&L</div><div style={{ fontSize: 16, fontWeight: 700, color: pnl >= 0 ? "#4ade80" : "#f87171" }}>{pnl >= 0 ? "+" : ""}${Math.abs(pnl).toLocaleString("en-US", { maximumFractionDigits: 0 })}</div></div>
                        <div><div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.08em" }}>DAYS</div><div style={{ fontSize: 16, fontWeight: 700, color: "#e2e8f0" }}>{es.length}</div></div>
                        <div><div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.08em" }}>WIN RATE</div><div style={{ fontSize: 16, fontWeight: 700, color: wr >= 50 ? "#4ade80" : "#f87171" }}>{wr}%</div></div>
                      </div>
                    );
                  })()}
                </div>

                {/* Summary content */}
                {loading ? (
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: 300, gap: 16 }}>
                    <div style={{ fontSize: 11, color: "#3b82f6", letterSpacing: "0.15em", animation: "pulse 1.5s infinite" }}>✦ GENERATING YOUR RECAP...</div>
                    <div style={{ fontSize: 10, color: "#64748b" }}>Claude is analyzing your journal notes</div>
                    <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }`}</style>
                  </div>
                ) : renderSummary(summary)}

                {/* Regenerate */}
                {summary && summary !== "NO_NOTES" && summary !== "ERROR" && !loading && (
                  <div style={{ marginTop: 20, paddingTop: 16, borderTop: "1px solid #1e293b", display: "flex", justifyContent: "flex-end" }}>
                    <button onClick={() => { setGenerated(prev => { const n = {...prev}; delete n[selectedPeriod]; return n; }); generateSummary(selectedPeriod); }}
                      style={{ background: "transparent", border: "1px solid #1e293b", color: "#94a3b8", padding: "8px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.06em" }}>↺ REGENERATE</button>
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
              <button onClick={() => onChange([])} style={{ background: "transparent", border: "none", color: "#64748b", fontSize: 10, cursor: "pointer", fontFamily: "inherit", letterSpacing: "0.05em" }}>CLEAR ALL</button>
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
        const r = await window.storage.get("trader-quotes-v1");
        setQuotes(r?.value ? JSON.parse(r.value) : SEED_QUOTES);
        if (!r?.value) await window.storage.set("trader-quotes-v1", JSON.stringify(SEED_QUOTES));
      } catch { setQuotes(SEED_QUOTES); }
      setLoaded(true);
    })();
  }, []);

  const persist = async (updated) => {
    setQuotes(updated);
    try { await window.storage.set("trader-quotes-v1", JSON.stringify(updated)); } catch(e) { console.warn(e); }
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
          <div style={{ fontFamily:"'Bebas Neue',sans-serif", fontSize:22, color:"#93c5fd", letterSpacing:"0.1em", lineHeight:1 }}>TRADER QUOTES</div>
          <div style={{ fontSize:9, color:"#64748b", letterSpacing:"0.12em", marginTop:4 }}>{quotes.length} QUOTES SAVED</div>
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
            <div style={{ fontSize:9, color:"#64748b", letterSpacing:"0.14em" }}>
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
            <button onClick={cancelForm} style={{ background:"transparent", border:"1px solid #1e293b", color:"#64748b", padding:"7px 16px", borderRadius:4, fontFamily:"inherit", fontSize:10, cursor:"pointer", letterSpacing:"0.06em" }}>CANCEL</button>
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
                  <span style={{ fontSize:9, color:"#64748b", letterSpacing:"0.1em", fontFamily:"'DM Mono',monospace", fontStyle:"normal" }}>{q.author.toUpperCase()}</span>
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
              <button onClick={()=>setShowAll(p=>!p)} style={{ background:"transparent", border:"1px solid #0f1729", color:"#64748b", padding:"7px 20px", borderRadius:4, fontFamily:"inherit", fontSize:10, cursor:"pointer", letterSpacing:"0.08em", transition:"all .15s" }}
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
    try { window.localStorage.setItem("tj-certs-v1", JSON.stringify(updated)); } catch {}
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
            <div style={{ fontSize: 9, color: "#475569", letterSpacing: "0.14em", marginTop: 4 }}>
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
          style={{ background: "transparent", border: "1px solid #1e293b", color: "#64748b", padding: "8px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 10, cursor: "pointer", letterSpacing: "0.1em", transition: "all .18s" }}>
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
        const [certs, setCerts] = useState(() => { try { const r = window.localStorage.getItem("tj-certs-v1"); return r ? JSON.parse(r) : []; } catch { return []; } });
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
                <div style={{ fontSize: 10, color: "#64748b", letterSpacing: "0.12em" }}>
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
                        <div style={{ fontSize: 9, color: "#64748b", letterSpacing: "0.12em", marginBottom: 4, textTransform: "uppercase" }}>{isArchJ ? "Final P&L" : isEvalJ ? "Eval Net P&L" : "Funded Net P&L"}</div>
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
                            <span style={{ fontSize: 9, color: "#64748b", letterSpacing: "0.1em" }}>DRAWDOWN USED</span>
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
                  style={{ width: "100%", background: "transparent", border: "1px dashed #1e293b", borderRadius: 6, padding: "10px 16px", color: "#475569", fontFamily: "inherit", fontSize: 10, cursor: "pointer", letterSpacing: "0.12em", textTransform: "uppercase", transition: "all .15s" }}
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
                        <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 5, textTransform: "uppercase" }}>{f.label}</div>
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
                      <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 6, textTransform: "uppercase" }}>BREACH TYPE</div>
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
                      <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 5, textTransform: "uppercase" }}>EVAL / CHALLENGE COST ($)</div>
                      <input type="number" value={archiveForm.evalCost} placeholder="e.g. 150"
                        onChange={e => setArchiveForm(f => ({...f, evalCost: e.target.value}))}
                        style={{ width: "100%", boxSizing: "border-box", padding: "9px 12px", background: "#060810", border: "1px solid #1e293b", borderRadius: 5, color: "#e2e8f0", fontFamily: "inherit", fontSize: 12 }} />
                      <div style={{ fontSize: 9, color: "#475569", marginTop: 4 }}>Shows up in your cumulative business expenses.</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 5, textTransform: "uppercase" }}>POST-MORTEM (what happened?)</div>
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
                          <div style={{ fontSize: 8, color: "#94a3b8", letterSpacing: "0.08em", marginBottom: 3 }}>{s.l}</div>
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
                          <div style={{ fontSize: 8, color: "#94a3b8", letterSpacing: "0.08em", marginBottom: 3 }}>{s.l}</div>
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
                        <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.08em", marginBottom: 4 }}>{s.l}</div>
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
            const winners = allTrades.filter(t => t.pnl > 0);
            const losers = allTrades.filter(t => t.pnl < 0);
            const winRate = allTrades.length ? (winners.length / allTrades.length * 100) : 0;
            const netPnl = jPs ? (jPs.totalCumPnl ?? 0) : jEntries.reduce((s, e) => s + (parseFloat(e.pnl) || 0) - (parseFloat(e.commissions) || 0), 0);
            const fees = jEntries.reduce((s, e) => s + (parseFloat(e.commissions) || 0), 0);
            const avgWin = winners.length ? winners.reduce((s, t) => s + t.pnl, 0) / winners.length : 0;
            const avgLoss = losers.length ? Math.abs(losers.reduce((s, t) => s + t.pnl, 0) / losers.length) : 0;
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
                        <div style={{ fontSize: 9, color: "#64748b", letterSpacing: "0.12em", marginBottom: 4 }}>EVAL COSTS (EXPENSES)</div>
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
                      <div style={{ fontSize: 8, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 4 }}>{s.l}</div>
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
                      <div style={{ fontSize: 8, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 3 }}>{s.l}</div>
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
                  <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 10 }}>P&L BY ACCOUNT</div>
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
            <div style={{ fontSize: 8, color: "#94a3b8", letterSpacing: "0.12em", marginBottom: big ? 6 : 4, textTransform: "uppercase" }}>{label}</div>
            <div style={{ fontSize: big ? 22 : 15, color, fontWeight: big ? 700 : 500, fontFamily: "'DM Mono',monospace", letterSpacing: "0.02em" }}>{value}</div>
            {sub && <div style={{ fontSize: 9, color: "#64748b", marginTop: 4 }}>{sub}</div>}
          </div>
        );

        const InputRow = ({ label, value, onChange, placeholder, prefix, suffix, type = "number", min }) => (
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase" }}>{label}</label>
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
                  <label style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>INSTRUMENT</label>
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
                    <label style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase" }}>Stop (points)</label>
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
                    <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.12em", marginBottom: 12 }}>REWARD TARGETS · {contracts} CONTRACT{contracts !== 1 ? "S" : ""}</div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
                      <Stat label="1R Target" value={`$${actualRisk.toLocaleString()}`} color="#fbbf24" sub={`${(ticks).toFixed(0)} ticks`} />
                      <Stat label="2R Target" value={`$${reward2x.toLocaleString()}`} color="#4ade80" sub={`${(ticks*2).toFixed(0)} ticks`} />
                      <Stat label="3R Target" value={`$${reward3x.toLocaleString()}`} color="#22d3ee" sub={`${(ticks*3).toFixed(0)} ticks`} />
                    </div>
                  </div>
                )}

                {/* Quick presets */}
                <div style={{ background: "#070d1a", border: "1px solid #0f1729", borderRadius: 6, padding: "12px 14px" }}>
                  <div style={{ fontSize: 9, color: "#64748b", letterSpacing: "0.1em", marginBottom: 2 }}>COMMON STOP SIZES · {spec.sym}</div>
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
                    <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.12em", marginBottom: 12 }}>REWARD TARGETS · {shares.toLocaleString()} SHARES</div>
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
          </div>
        );
}

function ReferenceView() {
  const [activeSection, setActiveSection] = useState("sessions");

  const SECTIONS = [
    { id: "sessions", label: "SESSION MAP" },
    { id: "newyork", label: "NEW YORK" },
    { id: "events",  label: "NEWS EVENTS" },
    { id: "notes",   label: "REMINDERS" },
    { id: "risk",    label: "⚡ RISK CALC" },
    { id: "specs",   label: "📋 CONTRACT SPECS" },
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
    <div style={{ fontFamily:"'DM Mono',monospace" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
        @keyframes refFadeIn { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
        .ref-card { transition: border-color .18s, transform .18s; }
        .ref-card:hover { transform: translateY(-1px); }
        .bar-seg:hover { filter: brightness(1.35); }
        .ref-sec-btn:hover { border-color: #1e3a5f !important; color: #93c5fd !important; }
      `}</style>

      {/* ── PAGE HEADER ── */}
      <div style={{ marginBottom:24, animation:"refFadeIn .35s ease" }}>
        <div style={{ fontSize:9, color:"#3b82f6", letterSpacing:"0.2em", marginBottom:8 }}>📊 ES · MES · FUTURES REFERENCE</div>
        <div style={{ fontFamily:"'Bebas Neue',sans-serif", fontSize:28, color:"#e2e8f0", letterSpacing:"0.1em", lineHeight:1 }}>
          TRADING <span style={{ color:"#00ff88" }}>SESSIONS</span>
        </div>
        <div style={{ fontSize:10, color:"#64748b", letterSpacing:"0.1em", marginTop:6 }}>EASTERN TIME ZONE — NORTH AMERICAN TRADER</div>
      </div>

      {/* ── SECTION TABS ── */}
      <div style={{ display:"flex", gap:6, marginBottom:20, flexWrap:"wrap" }}>
        {SECTIONS.map(s => (
          <button key={s.id} className="ref-sec-btn" onClick={() => setActiveSection(s.id)}
            style={{ padding:"6px 16px", borderRadius:3, fontFamily:"inherit", fontSize:10, cursor:"pointer", letterSpacing:"0.1em", transition:"all .15s", background: activeSection===s.id?"#0a1628":"transparent", border:`1px solid ${activeSection===s.id?"#1e3a5f":"#0f1729"}`, color: activeSection===s.id?"#93c5fd":"#64748b" }}>
            {s.label}
          </button>
        ))}
      </div>

      {/* ════════════════════════════════════════════════
          SESSION MAP TAB
      ════════════════════════════════════════════════ */}
      {activeSection === "sessions" && (
        <div style={{ animation:"refFadeIn .3s ease" }}>

          {/* 24-Hour Timeline */}
          <div style={{ background:"#060b18", border:"1px solid #0f1e30", borderRadius:8, padding:"22px 24px 20px", marginBottom:20, position:"relative", overflow:"hidden" }}>
            {/* Top accent bar */}
            <div style={{ position:"absolute", top:0, left:0, right:0, height:2, background:"linear-gradient(90deg, #00aaff, #00ff88, #ff8c00)" }}/>
            <div style={{ fontSize:9, color:"#64748b", letterSpacing:"0.15em", marginBottom:18 }}>⏱ 24-HOUR SESSION MAP (EST) — 6PM TO 5PM NEXT DAY</div>

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
          <div className="ref-card" style={{ background:"#060b18", border:"1px solid rgba(0,255,136,0.18)", borderRadius:8, padding:"18px 22px", display:"flex", alignItems:"center", justifyContent:"space-between", cursor:"pointer" }}
            onClick={() => setActiveSection("newyork")}>
            <div style={{ display:"flex", alignItems:"center", gap:12 }}>
              <span style={{ fontSize:26 }}>🗽</span>
              <div>
                <div style={{ fontFamily:"'Rajdhani',sans-serif", fontSize:18, fontWeight:700, color:"#00ff88", letterSpacing:"0.08em" }}>NEW YORK SESSION</div>
                <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:11, color:"#64748b", marginTop:3 }}>9:30 AM – 5:00 PM EST · 6 distinct windows</div>
              </div>
            </div>
            <div style={{ fontSize:11, color:"#1e3a5f", letterSpacing:"0.08em" }}>VIEW DETAIL →</div>
          </div>
        </div>
      )}

      {/* ════════════════════════════════════════════════
          NEW YORK TAB
      ════════════════════════════════════════════════ */}
      {activeSection === "newyork" && (
        <div style={{ animation:"refFadeIn .3s ease" }}>
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
        </div>
      )}

      {/* ════════════════════════════════════════════════
          NEWS EVENTS TAB
      ════════════════════════════════════════════════ */}
      {activeSection === "events" && (
        <div style={{ animation:"refFadeIn .3s ease" }}>
          <div style={{ background:"#060b18", border:"1px solid #0f1e30", borderRadius:8, padding:"22px 24px", position:"relative", overflow:"hidden" }}>
            <div style={{ position:"absolute", top:0, left:0, right:0, height:2, background:"linear-gradient(90deg,#ff8c00,#ffd700)" }}/>
            <div style={{ fontSize:10, color:"#64748b", letterSpacing:"0.15em", marginBottom:18 }}>📅 HIGH VOLUME NEWS EVENTS — WATCH THESE</div>
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
      )}

      {/* ════════════════════════════════════════════════
          REMINDERS TAB
      ════════════════════════════════════════════════ */}
      {activeSection === "notes" && (
        <div style={{ animation:"refFadeIn .3s ease", display:"flex", flexDirection:"column", gap:14 }}>

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
            <div style={{ fontSize:9, color:"#64748b", letterSpacing:"0.15em", marginBottom:14 }}>QUICK REFERENCE</div>
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
                  <div style={{ fontSize:8, color:"#64748b", letterSpacing:"0.12em", marginBottom:4 }}>{item.label}</div>
                  <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:12, color:item.c }}>{item.value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Pre-session checklist */}
          <div style={{ background:"#060b18", border:"1px solid #0f1e30", borderRadius:8, padding:"18px 22px" }}>
            <div style={{ fontSize:9, color:"#64748b", letterSpacing:"0.15em", marginBottom:14 }}>✅ PRE-SESSION CHECKLIST</div>
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
      {activeSection === "risk" && <RiskCalcPanel />}

      {activeSection === "specs" && (
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
                <div key={h} style={{ fontSize: 8, color: "#64748b", letterSpacing: "0.1em" }}>{h}</div>
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
            <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.14em", marginBottom: 10, textTransform: "uppercase" }}>🔍 P&L VERIFICATION EXAMPLES</div>
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
// ─────────────────────────────────────────────────────────────────────────────
const DURATION_HARD_LIMIT_SECS = 28800; // 8h intraday cap

const validateTradesHard = (trades, allowOvernight = false) => {
  const limit = allowOvernight ? 86400 : DURATION_HARD_LIMIT_SECS;
  const accepted = [], rejected = [];
  const seen = new Set();
  for (const t of trades) {
    const reasons = [];
    if (!t.symbol || !t.symbol.trim()) reasons.push("Missing symbol");
    if (!Number.isFinite(t.qty) || t.qty <= 0) reasons.push("Invalid qty");
    if (!Number.isFinite(t.pnl)) reasons.push("Non-finite P&L");
    if (t.buyPrice <= 0 || t.sellPrice <= 0) reasons.push("Invalid price");
    if (t.durationSecs > limit) reasons.push(`Duration ${(t.durationSecs/3600).toFixed(1)}h > ${limit/3600}h limit`);
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
    if (t.durationSecs > 0 && t.durationSecs < 2) warnings.push("⚡ Sub-2s duration — verify fill");
    if (Math.abs(t.pnl) > 50000) warnings.push("💰 Very large P&L — confirm correct");
    else if (Math.abs(t.pnl) > 5000 && (t.symbol?.startsWith("M") || t.qty < 3)) warnings.push("💰 Large P&L for micro — confirm correct");
    if (t.pnl === 0) warnings.push("⚪ Zero P&L — breakeven or parse error?");
    if (!t.buyTime && !t.sellTime) warnings.push("🕐 No timestamps — session attribution unavailable");
    if (warnings.length) flagged.push({ trade: t, warnings });
    else clean.push(t);
  }
  return { clean, flagged };
};

// ─────────────────────────────────────────────────────────────────────────────
// Feature 2: Parse Review Modal
// ─────────────────────────────────────────────────────────────────────────────
function ParseReviewModal({ data, onConfirm, onCancel }) {
  const { hardRejected, flagged, clean } = data;
  const [checked, setChecked] = useState(() => Object.fromEntries(flagged.map((_, i) => [i, true])));
  const [allowOvernight, setAllowOvernight] = useState(false);
  const [showRejected, setShowRejected] = useState(false);
  const [showClean, setShowClean] = useState(false);

  const acceptedFlagged = flagged.filter((_, i) => checked[i]).map(f => f.trade);
  const rejectedCount   = flagged.filter((_, i) => !checked[i]).length;
  const finalTrades     = [...clean, ...acceptedFlagged];

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
    <div style={{ display: "grid", gridTemplateColumns: "90px 40px 48px 80px 80px 80px 70px", gap: 8, fontSize: 9, color: "#64748b", letterSpacing: "0.08em", marginBottom: 4 }}>
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
                    <div key={i} style={{ background: "#0a0e1a", border: "1px solid #450a0a", borderRadius: 6, padding: "10px 14px" }}>
                      <TRow t={item.trade} />
                      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 6 }}>
                        {item.reasons.map((r, ri) => (
                          <span key={ri} style={{ fontSize: 9, color: "#f87171", background: "rgba(248,113,113,0.07)", border: "1px solid #450a0a", padding: "2px 8px", borderRadius: 3 }}>{r}</span>
                        ))}
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
            <div style={{ background: "#1f0606", border: "1px solid #7f1d1d", borderRadius: 4, padding: "10px 14px", fontSize: 11, color: "#f87171" }}>
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
                hardRejected: hardRejected.length,
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

  const [view, setView] = useState("list");
  const [propDashTab, setPropDashTab] = useState("overview"); // "overview" | "account" | "cumulative"
  const [recapInitMode, setRecapInitMode] = useState("weekly");
  const [activeEntry, setActiveEntry] = useState(null);
  const [form, setForm] = useState(emptyEntry());
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [filterMonth, setFilterMonth] = useState("");
  const [tab, setTab] = useState("session");
  const [importRaw, setImportRaw] = useState("");
  const [importError, setImportError] = useState("");
  const [importSuccess, setImportSuccess] = useState(false);
  const [analyticsTab, setAnalyticsTab] = useState("overview");
  const [listMode, setListMode] = useState("calendar");
  const [calMonth, setCalMonth] = useState(() => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`; });
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
          const metaResult = await window.storage.get("journal-meta");
          if (metaResult?.value) loadedJournals = JSON.parse(metaResult.value);
        } catch {}
        setJournals(loadedJournals);

        // Load active journal entries
        const activeId = DEFAULT_JOURNAL_ID;
        let entriesData = null;

        // Try new key first
        try {
          const r = await window.storage.get(`journal-entries-${activeId}`);
          if (r?.value) entriesData = r.value;
        } catch {}

        // If not found, migrate from old journal-v3 key
        if (!entriesData) {
          try {
            const oldResult = await window.storage.get("journal-v3");
            if (oldResult?.value) {
              entriesData = oldResult.value;
              // Save under new key so migration only runs once
              await window.storage.set(`journal-entries-${activeId}`, entriesData);
            }
          } catch {}
        }

        if (entriesData) {
          const parsed = JSON.parse(entriesData);
          const norm = normalizeEntries(parsed);
          setEntries(norm.entries);
          if (norm.report?.changed) {
            try { await window.storage.set(`journal-entries-${activeId}`, JSON.stringify(norm.entries)); } catch {}
            setDataHealthReport(norm.report);
          }
        }

        // Load a random quote for the header
        try {
          const qr = await window.storage.get("trader-quotes-v1");
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
    await window.storage.set("journal-meta", JSON.stringify(updated));
  };

  const saveEntries = async (updated, journalId = activeJournalId) => {
    if (journalId === activeJournalId) setEntries(updated);
    await window.storage.set(`journal-entries-${journalId}`, JSON.stringify(updated));
  };

  const switchJournal = async (id) => {
    if (id === activeJournalId) return;
    setActiveJournalId(id);
    setView("list"); setActiveEntry(null); setForm(emptyEntry());
    setFilterMonth(""); setListMode("calendar");
    try {
      const result = await window.storage.get(`journal-entries-${id}`);
      const parsed = result?.value ? JSON.parse(result.value) : [];
      const norm = normalizeEntries(parsed);
      setEntries(norm.entries);
      if (norm.report?.changed) {
        try { await window.storage.set(`journal-entries-${id}`, JSON.stringify(norm.entries)); } catch {}
        setDataHealthReport(norm.report);
      }
    } catch { setEntries([]); }
  };

  const createJournal = async () => {
    const name = newJournalName.trim() || `Journal ${journals.length + 1}`;
    const id = `journal-${Date.now()}`;
    const newJ = { id, name, createdAt: Date.now(), type: newJournalType, config: newJournalConfig };
    await saveJournalsMeta([...journals, newJ]);
    await window.storage.set(`journal-entries-${id}`, JSON.stringify([]));
    setNewJournalName("");
    setNewJournalType(JOURNAL_TYPES.PERSONAL);
    setNewJournalConfig(defaultPersonalConfig());
    setShowJournalMgr(false);
    switchJournal(id);
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
    await window.storage.delete(`journal-entries-${id}`);
    setConfirmDeleteId(null);
    if (activeJournalId === id) switchJournal(updated[0].id);
  };

  const [aiParsing, setAiParsing] = useState(false);
  const [aiParseError, setAiParseError] = useState("");
  const [detectedFormat, setDetectedFormat] = useState("");
  // Feature 2: Trade review modal state
  const [parseReviewData, setParseReviewData] = useState(null);

  const handleImport = async () => {
    setImportError(""); setImportSuccess(false); setAiParseError(""); setDetectedFormat("");
    if (!importRaw.trim()) { setImportError("Please paste your broker data first."); return; }

    setAiParsing(true);
    try {
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

  const applyParsedTrades = (trades, fmt, logEntry) => {
    const totalPnL = trades.reduce((s, t) => s + t.pnl, 0);
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
      parsedTrades: trades,
      pnl: totalPnL.toFixed(2),
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
    let updated = activeEntry ? entries.map(e => e.id === form.id ? entry : e) : [{ ...entry, id: Date.now() }, ...entries];
    await saveEntries(updated);
    setSaving(false); setSaved(true);
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
    setTimeout(() => { setSaved(false); setView("list"); setActiveEntry(null); setForm(emptyEntry()); setTab("session"); setImportRaw(""); setImportError(""); setImportSuccess(false); }, 900);
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

  const openNew = () => { setForm(emptyEntry()); setActiveEntry(null); setTab("session"); setImportRaw(""); setImportError(""); setImportSuccess(false); setAnalyticsTab("overview"); setView("new"); };
  const openEdit = (entry) => { setForm({ ...entry }); setActiveEntry(entry); setTab("session"); setImportRaw(entry.rawTradeData || ""); setImportError(""); setImportSuccess(false); setAnalyticsTab("overview"); setView("new"); };
  const viewDetail = (entry) => { setActiveEntry(entry); setView("detail"); };

  const f = (field, val) => setForm(p => ({ ...p, [field]: val }));
  const pnlColor = (n) => { const v = parseFloat(n); return isNaN(v) ? "#e2e8f0" : v > 0 ? "#4ade80" : v < 0 ? "#f87171" : "#e2e8f0"; };
  const gradeColor = (g) => { if (!g) return "#94a3b8"; if (g.startsWith("A")) return "#4ade80"; if (g.startsWith("B")) return "#60a5fa"; if (g.startsWith("C")) return "#facc15"; return "#f87171"; };
  const fmtPnl = (n) => { const v = parseFloat(n); if (isNaN(v)) return "-"; return `${v >= 0 ? "+" : ""}$${Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`; };

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
          const r = await window.storage.get(`entries_${j.id}`);
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
    for (const e of entries) { map[e.id] = calcAnalytics(e.parsedTrades || [], aiCfg?.tzLock !== false); }
    return map;
  }, [entries]);

  // Search & filter state
  const [searchQuery, setSearchQuery] = useState("");
  const [searchField, setSearchField] = useState("all"); // "all" | "notes" | "instruments" | "grade" | "mistakes"
  const filteredAndSearched = useMemo(() => {
    if (!searchQuery.trim()) return filtered;
    const q = searchQuery.toLowerCase();
    return filtered.filter(e => {
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
  }, [filtered, searchQuery, searchField]);

  const TABS = [
    { id: "session", label: "Session" },
    { id: "import", label: "Import Trades" },
    { id: "analysis", label: "Analysis", disabled: !form.parsedTrades?.length },
    { id: "lessons", label: "Lessons" },
    { id: "tomorrow", label: "Tomorrow" },
  ];

  // ── Import / Export ──────────────────────────────────────────────
  const [showSettings, setShowSettings] = useState(false);
  const [settingsTab, setSettingsTab] = useState("backup");
  const [importMsg, setImportMsg] = useState(null);
  const importFileRef = useRef(null);

  const [exportData, setExportData] = useState(null);
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
      const allData = { version: 2, exportedAt: new Date().toISOString(), journals, entriesByJournal: {} };
      for (const j of journals) {
        try {
          const r = await window.storage.get(`journal-entries-${j.id}`);
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
      const r = await window.storage.get(`journal-entries-${jId}`);
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
        const r = await window.storage.get(`journal-entries-${j.id}`);
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
        await window.storage.set(`journal-entries-${j.id}`, JSON.stringify(jes));
      }
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
    <div style={{ fontFamily: "'DM Mono','Courier New',monospace", background: "#0a0e1a", minHeight: "100vh", color: "#e2e8f0" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Bebas+Neue&display=swap');
        html,body{width:100%;margin:0;padding:0}*{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:#0a0e1a}::-webkit-scrollbar-thumb{background:#1e3a5f;border-radius:2px}
        textarea,input,select{background:#0f1729!important;color:#e2e8f0!important;border:1px solid #1e3a5f!important;border-radius:4px;padding:10px 12px;font-family:'DM Mono',monospace;font-size:13px;width:100%;outline:none;transition:border-color .2s;resize:vertical}
        textarea:focus,input:focus,select:focus{border-color:#3b82f6!important}
        textarea::placeholder,input::placeholder{color:#1e3a5f}
        select option{background:#0f1729}
        .entry-card{transition:all .15s;cursor:pointer}
        
        .entry-card:hover{background:#0f1729!important;border-color:#1e3a5f!important}
        .pill{padding:6px 14px;border-radius:3px;border:1px solid #1e293b;font-size:12px;cursor:pointer;transition:all .15s;color:#94a3b8;background:transparent;letter-spacing:.05em;font-family:'DM Mono',monospace}
        .pill.sel{border-color:#3b82f6;background:#1e3a5f;color:#93c5fd}
        .pill:hover:not(.sel){border-color:#334155;color:#e2e8f0}
        @keyframes qotdFade { from{opacity:0;transform:translateY(-6px)} to{opacity:1;transform:translateY(0)} }
        @keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
      `}</style>

      {/* Header */}
      <div style={{ borderBottom: "1px solid #1e293b", padding: "14px 32px", display: "flex", alignItems: "center", justifyContent: "space-between", position: "sticky", top: 0, background: "#0a0e1a", zIndex: 10 }}>

        {/* LEFT: Title + backup menu */}
        <div style={{ display: "flex", alignItems: "baseline", gap: 14, position: "relative", flexShrink: 0 }}>
          <span onClick={() => { setView("list"); setActiveEntry(null); }} style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 30, color: "#3b82f6", letterSpacing: "0.15em", cursor: "pointer", transition: "color .15s" }} onMouseEnter={e => e.currentTarget.style.color="#60a5fa"} onMouseLeave={e => e.currentTarget.style.color="#3b82f6"}>TRADING JOURNAL</span>
          <span style={{ fontSize: 11, color: "#334155", letterSpacing: "0.12em" }}>FUTURES</span>
          {/* Settings trigger */}
          <button
            onClick={() => { setShowSettings(true); setSettingsTab("backup"); setImportMsg(null); }}
            title="Settings"
            style={{ background: "transparent", border: "none", color: "#1e293b", fontSize: 14, cursor: "pointer", padding: "0 4px", lineHeight: 1, fontFamily: "inherit", transition: "color .15s" }}
            onMouseEnter={e => e.currentTarget.style.color = "#475569"}
            onMouseLeave={e => e.currentTarget.style.color = "#1e293b"}>
            ⋯
          </button>
        </div>

        {/* RIGHT: Nav buttons */}
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexShrink: 0 }}>
          {view === "list" && propJournals.length > 0 && <button onClick={() => { setPropDashTab("overview"); setView("propdash"); }} style={{ background: "transparent", color: "#f59e0b", border: "1px solid #92400e", padding: "11px 24px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer", letterSpacing: "0.05em" }}>🏆 PROP DASHBOARD</button>}
          {(view === "list" || view === "recap") && <button onClick={() => setView("recap")} style={{ background: view === "recap" ? "#1e3a5f" : "transparent", color: view === "recap" ? "#93c5fd" : "#94a3b8", border: `1px solid ${view === "recap" ? "#3b82f6" : "#1e293b"}`, padding: "11px 24px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer", letterSpacing: "0.05em" }}>🤖 AI RECAP</button>}
          {view === "list" && <button onClick={openNew} style={{ background: "#1d4ed8", color: "white", border: "none", padding: "11px 24px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer", letterSpacing: "0.05em" }}>+ NEW ENTRY</button>}
          {view === "recap" && <button onClick={() => setView("list")} style={{ background: "transparent", color: "#94a3b8", border: "1px solid #1e293b", padding: "11px 24px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer" }}>← BACK</button>}
          {view === "quotes" && <button onClick={() => setView("list")} style={{ background: "transparent", color: "#94a3b8", border: "1px solid #1e293b", padding: "11px 24px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer" }}>← BACK</button>}
          {view === "reference" && <button onClick={() => setView("list")} style={{ background: "transparent", color: "#94a3b8", border: "1px solid #1e293b", padding: "11px 24px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer" }}>← BACK</button>}
          {view === "propdash" && <button onClick={() => setView("list")} style={{ background: "transparent", color: "#f59e0b", border: "1px solid #92400e", padding: "11px 24px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer" }}>← BACK</button>}
          {view === "new" && <>
            <button onClick={() => { setView("list"); setActiveEntry(null); }} style={{ background: "transparent", color: "#94a3b8", border: "1px solid #1e293b", padding: "11px 24px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer" }}>CANCEL</button>
            <button onClick={handleSave} disabled={saving} style={{ background: "#1d4ed8", color: "white", border: "none", padding: "11px 24px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer", textTransform: "uppercase" }}>{saving ? "SAVING..." : saved ? "✓ SAVED" : activeEntry ? "✓ UPDATE" : "+ APPLY"}</button>
          </>}
          {view === "detail" && <>
            <button onClick={() => setView("list")} style={{ background: "transparent", color: "#94a3b8", border: "1px solid #1e293b", padding: "9px 20px", borderRadius: 4, fontFamily: "inherit", fontSize: 12, cursor: "pointer" }}>← BACK</button>
            <button onClick={() => openEdit(activeEntry)} style={{ background: "transparent", color: "#94a3b8", border: "1px solid #1e293b", padding: "11px 24px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer" }}>EDIT</button>
            <button onClick={() => handleDelete(activeEntry.id)} style={{ background: "transparent", color: "#ef4444", border: "1px solid #450a0a", padding: "11px 20px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer" }}>DELETE</button>
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
                onClick={() => { navigator.clipboard.writeText(exportData).then(() => { const b = document.getElementById("copy-btn"); if (b) { b.textContent = "✓ COPIED!"; b.style.color = "#4ade80"; setTimeout(() => { b.textContent = "COPY ALL"; b.style.color = "white"; }, 2000); } }); }}
                id="copy-btn"
                style={{ background: "#1d4ed8", color: "white", border: "none", padding: "8px 20px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.06em" }}>
                COPY ALL
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
                      {journals.length} journal{journals.length !== 1 ? 's' : ''} · Data stored in browser via <span style={{ color: '#64748b' }}>window.storage</span><br />
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
      <div style={{ borderBottom: "1px solid #1e293b", background: "#060810", padding: "0 32px", display: "flex", alignItems: "stretch", gap: 2, overflowX: "auto", position: "sticky", top: 58, zIndex: 9 }}>
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
                  style={{ display: "flex", alignItems: "center", gap: 8, padding: "14px 20px", cursor: "pointer", borderBottom: `3px solid ${isActive ? "#3b82f6" : "transparent"}`, background: "transparent", transition: "all .15s", whiteSpace: "nowrap" }}
                  onMouseEnter={e => { if (!isActive) e.currentTarget.style.borderBottomColor = "#1e3a5f"; }}
                  onMouseLeave={e => { if (!isActive) e.currentTarget.style.borderBottomColor = "transparent"; }}>
                  <span style={{ fontSize: 13, color: isActive ? "#93c5fd" : "#64748b", letterSpacing: "0.06em", fontWeight: isActive ? 600 : 400 }}>
                    {j.type === JOURNAL_TYPES.PROP ? "🏆" : "💼"} {j.name.toUpperCase()}
                  </span>
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
                <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 22, color: isPropType ? "#f59e0b" : "#93c5fd", letterSpacing: "0.12em" }}>NEW JOURNAL</div>
                <button onClick={close} style={{ background: "transparent", border: "none", color: "#64748b", fontSize: 16, cursor: "pointer" }}>✕</button>
              </div>

              {/* Name */}
              <div style={{ marginBottom: 20 }}>
                <label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", display: "block", marginBottom: 6 }}>JOURNAL NAME</label>
                <input autoFocus placeholder="e.g. Lucid 50K Flex, Questrade TFSA…"
                  value={newJournalName} onChange={e => setNewJournalName(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && createJournal()}
                  style={{ width: "100%", boxSizing: "border-box" }} />
              </div>

              {/* Type picker */}
              <div style={{ marginBottom: 24 }}>
                <label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", display: "block", marginBottom: 10 }}>ACCOUNT TYPE</label>
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
                  <label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.08em", display: "block", marginBottom: 6 }}>STARTING BALANCE ($)</label>
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
                      <label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.08em", display: "block", marginBottom: 6 }}>FIRM NAME</label>
                      <input value={newJournalConfig.firmName || "Lucid Trading"}
                        onChange={e => setNewJournalConfig(c => ({ ...c, firmName: e.target.value }))}
                        style={{ width: "100%", boxSizing: "border-box" }} />
                    </div>
                    <div>
                      <label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.08em", display: "block", marginBottom: 8 }}>ACCOUNT SIZE</label>
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
                          <label style={{ fontSize: 9, color: "#64748b", letterSpacing: "0.08em", display: "block", marginBottom: 5 }}>{label}</label>
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
                          <label style={{ fontSize: 9, color: "#64748b", letterSpacing: "0.08em", display: "block", marginBottom: 5 }}>{label}</label>
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
          { accent: "#3b82f6", glow: "rgba(59,130,246,0.07)", label: "✦ QUOTE OF THE DAY ✦", textSize: 22, layout: "center" },
          { accent: "#f59e0b", glow: "rgba(245,158,11,0.06)",  label: "✦ DISCIPLINE ✦",       textSize: 16, layout: "left"   },
          { accent: "#4ade80", glow: "rgba(74,222,128,0.06)",  label: "✦ EDGE ✦",              textSize: 16, layout: "left"   },
        ];
        return (
          <div style={{ width: "100%", padding: "28px 40px 0", boxSizing: "border-box", display: "grid", gridTemplateColumns: "1.55fr 1fr 1fr", gap: 20, marginBottom: 0 }}>
              {headerQuotes.map((q, i) => {
                const cs = CARD_STYLES[i] || CARD_STYLES[2];
                const isHero = i === 0;
                return (
                  <div key={q.id} onClick={() => setView("quotes")} style={{ borderRadius: 8, border: `1px solid ${cs.accent}28`, background: "#070d1a", padding: isHero ? "52px 44px 48px" : "44px 32px 40px", position: "relative", overflow: "hidden", textAlign: cs.layout, cursor: "pointer", transition: "border-color .15s" }} onMouseEnter={e => e.currentTarget.style.borderColor=`${cs.accent}55`} onMouseLeave={e => e.currentTarget.style.borderColor=`${cs.accent}28`}>
                    {/* Top accent line */}
                    <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, transparent, ${cs.accent}, transparent)`, opacity: 0.7 }} />
                    {/* Glow */}
                    <div style={{ position: "absolute", inset: 0, background: `radial-gradient(ellipse at 50% 0%, ${cs.glow} 0%, transparent 65%)`, pointerEvents: "none" }} />
                    {/* Ghost quote mark */}
                    <div style={{ position: "absolute", right: isHero ? 14 : 8, top: isHero ? 2 : -2, fontFamily: "Georgia,serif", fontSize: isHero ? 140 : 100, color: cs.accent, opacity: 0.07, lineHeight: 1, userSelect: "none", pointerEvents: "none" }}>"</div>
                    {/* Label */}
                    <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 11, color: cs.accent, letterSpacing: "0.3em", marginBottom: isHero ? 20 : 14, opacity: 0.85 }}>{cs.label}</div>
                    {/* Category pill */}
                    <div style={{ display: "inline-block", fontSize: 9, color: cs.accent, background: `${cs.accent}18`, border: `1px solid ${cs.accent}33`, padding: "3px 11px", borderRadius: 20, letterSpacing: "0.12em", marginBottom: isHero ? 18 : 14 }}>{q.category.toUpperCase()}</div>
                    {/* Quote text */}
                    <div style={{ fontFamily: "Georgia,'Times New Roman',serif", fontStyle: "italic", fontSize: isHero ? cs.textSize + 2 : cs.textSize + 1, color: isHero ? "#e2e8f0" : "#c8ddf0", lineHeight: 1.85, marginBottom: isHero ? 24 : 18, position: "relative", zIndex: 1 }}>
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

      <div style={{ width: "100%", padding: "28px 40px" }}>
        {/* LIST */}
        {view === "list" && (<>
          {filtered.length > 0 && (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,minmax(0,280px))", gap: 12, marginBottom: 24, justifyContent: "center" }}>
              {[
                { l: "TOTAL P&L", v: `${totalPnL >= 0 ? "+" : ""}$${Math.abs(totalPnL).toLocaleString("en-US", { maximumFractionDigits: 0 })}`, c: pnlColor(totalPnL) },
                { l: "WIN DAYS", v: `${winDays} / ${filtered.length}`, c: "#4ade80" },
                { l: "LOSS DAYS", v: lossDays, c: "#f87171" },
                { l: "DAY WIN RATE", v: `${filtered.length ? Math.round(winDays / filtered.length * 100) : 0}%`, c: filtered.length ? (winDays / filtered.length >= 0.5 ? "#4ade80" : "#f87171") : "#94a3b8" },
              ].map(s => (
                <div key={s.l} style={{ background: "#0f1729", border: "1px solid #1e293b", borderRadius: 5, padding: "12px 16px" }}>
                  <div style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", marginBottom: 4 }}>{s.l}</div>
                  <div style={{ fontSize: 20, color: s.c, fontWeight: 500 }}>{s.v}</div>
                </div>
              ))}
            </div>
          )}

          {/* View toggle + month nav */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 24, flexWrap: "wrap", gap: 10 }}>
              {[["calendar","📅 MONTHLY PERFORMANCE"],["list","📋 TRADE LIST"],["weekly","📈 WEEKLY PERFORMANCE"],["performance","📊 ANNUAL/QUARTERLY PERFORMANCE"]].map(([m, label]) => (
                <button key={m} onClick={() => setListMode(m)}
                  style={{ padding: "11px 20px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer", letterSpacing: "0.06em", transition: "all .15s", background: listMode === m ? "#1e3a5f" : "transparent", border: `1px solid ${listMode === m ? "#3b82f6" : "#1e293b"}`, color: listMode === m ? "#93c5fd" : "#94a3b8", whiteSpace: "nowrap" }}>
                  {label}
                </button>
              ))}
              <button onClick={() => setView("reference")}
                style={{ padding: "11px 20px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer", letterSpacing: "0.06em", transition: "all .15s", background: "transparent", border: "1px solid #1e293b", color: "#94a3b8", whiteSpace: "nowrap" }}>
                📊 REFERENCE
              </button>
              <button onClick={() => setView("quotes")}
                style={{ padding: "11px 20px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer", letterSpacing: "0.06em", transition: "all .15s", background: "transparent", border: "1px solid #1e293b", color: "#94a3b8", whiteSpace: "nowrap" }}>
                💬 QUOTES
              </button>
              {listMode === "calendar" ? (
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <button onClick={() => { const [y, m] = calMonth.split("-").map(Number); const d = new Date(y, m - 2, 1); setCalMonth(`${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`); }}
                  style={{ background: "transparent", border: "1px solid #1e293b", color: "#94a3b8", padding: "11px 16px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer" }}>‹</button>
                <span style={{ fontSize: 13, color: "#e2e8f0", letterSpacing: "0.08em", minWidth: 130, textAlign: "center", fontWeight: 500 }}>
                  {(() => { const [y, m] = calMonth.split("-").map(Number); return new Date(y, m - 1, 1).toLocaleString("default", { month: "long", year: "numeric" }).toUpperCase(); })()}
                </span>
                <button onClick={() => { const [y, m] = calMonth.split("-").map(Number); const d = new Date(y, m, 1); setCalMonth(`${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`); }}
                  style={{ background: "transparent", border: "1px solid #1e293b", color: "#94a3b8", padding: "11px 16px", borderRadius: 4, fontFamily: "inherit", fontSize: 13, cursor: "pointer" }}>›</button>
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
              <div style={{ fontSize: 22, color: "#93c5fd", letterSpacing: "0.08em", fontWeight: 700, fontFamily: "'Bebas Neue', sans-serif", marginBottom: 14 }}>📅 MONTHLY PERFORMANCE</div>
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
              <div style={{ fontSize: 22, color: "#93c5fd", letterSpacing: "0.08em", fontWeight: 700, fontFamily: "'Bebas Neue', sans-serif", marginBottom: 14 }}>📈 WEEKLY PERFORMANCE</div>
              <WeeklyPerformance entries={entries} netPnl={netPnl} fmtPnl={fmtPnl} pnlColor={pnlColor} calcAnalytics={calcAnalytics} />
            </>
          ) : listMode === "performance" ? (
            <>
              <div style={{ fontSize: 22, color: "#93c5fd", letterSpacing: "0.08em", fontWeight: 700, fontFamily: "'Bebas Neue', sans-serif", marginBottom: 14 }}>📊 ANNUAL PERFORMANCE BREAKDOWN</div>
              <PerformanceOverview entries={entries} netPnl={netPnl} fmtPnl={fmtPnl} pnlColor={pnlColor} />
            </>
          ) : (
            <>
              <div style={{ fontSize: 22, color: "#93c5fd", letterSpacing: "0.08em", fontWeight: 700, fontFamily: "'Bebas Neue', sans-serif", marginBottom: 14 }}>📋 TRADE LIST</div>

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
                        style={{ background: "#0a0e1a", border: "1px solid #1e293b", borderRadius: 5, padding: "12px 16px", display: "grid", gridTemplateColumns: "100px 1fr 100px 90px", gap: 12, alignItems: "center" }}>
                        <div>
                          <div style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.08em" }}>{new Date(entry.date + "T12:00:00").toLocaleDateString("en-US", { weekday: "short" }).toUpperCase()}</div>
                          <div style={{ fontSize: 13, color: "#e2e8f0", marginTop: 1 }}>{entry.date}</div>
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                            {(entry.instruments?.length ? entry.instruments : entry.instrument ? [entry.instrument] : []).map(sym => (
                              <span key={sym} style={{ padding: "2px 8px", borderRadius: 3, fontSize: 10, background: "#1e3a5f", color: "#93c5fd" }}>{sym}</span>
                            ))}
                            {entry.bias && <span style={{ padding: "2px 8px", borderRadius: 3, fontSize: 10, background: entry.bias === "Bullish" ? "#052e16" : entry.bias === "Bearish" ? "#450a0a" : "#1e1b4b", color: entry.bias === "Bullish" ? "#4ade80" : entry.bias === "Bearish" ? "#f87171" : "#a5b4fc" }}>{entry.bias.toUpperCase()}</span>}
                            {(entry.moods?.length ? entry.moods : entry.mood ? [entry.mood] : []).map(m => (
                              <span key={m} style={{ fontSize: 12, color: "#94a3b8" }}>{m}</span>
                            ))}
                            {entry.grade && <span style={{ padding: "2px 8px", borderRadius: 3, fontSize: 10, background: "#0f172a", border: `1px solid ${gradeColor(entry.grade)}`, color: gradeColor(entry.grade) }}>{entry.grade}</span>}
                            {a && <span style={{ fontSize: 10, color: "#94a3b8" }}><span style={{ letterSpacing: 0 }}><span style={{ color: "#4ade80" }}>{a.winners}</span><span style={{ color: "#94a3b8" }}>/</span><span style={{ color: "#f87171" }}>{a.losers}</span></span> · {Math.round(a.winRate)}% WR</span>}
                          </div>
                          {entry.sessionMistakes?.length > 0 && (
                            <div style={{ display: "flex", gap: 5, flexWrap: "wrap", alignItems: "center" }}>
                              {entry.sessionMistakes.includes("No Mistakes — Executed the Plan ✓") ? (
                                <span style={{ display: "inline-flex", alignItems: "center", gap: 4, padding: "2px 8px", borderRadius: 3, fontSize: 10, background: "#052e16", border: "1px solid #166534", color: "#4ade80" }}>
                                  ✓ Clean Session
                                </span>
                              ) : (
                                <>
                                  <span style={{ fontSize: 9, color: "#64748b", letterSpacing: "0.08em", textTransform: "uppercase", marginRight: 2 }}>Mistakes:</span>
                                  {entry.sessionMistakes.map(m => (
                                    <span key={m} style={{ padding: "2px 8px", borderRadius: 3, fontSize: 10, background: "#1f0606", border: "1px solid #7f1d1d", color: "#f87171" }}>{m}</span>
                                  ))}
                                </>
                              )}
                            </div>
                          )}
                        </div>
                        <div>{a && <MiniCurve curve={a.equityCurve} />}</div>
                        <div style={{ textAlign: "right" }}>
                          {entry.pnl !== "" && <>
                            <div style={{ fontSize: 16, color: pnlColor(netPnl(entry)), fontWeight: 500 }}>{fmtPnl(netPnl(entry))}</div>
                            {parseFloat(entry.commissions) ? <div style={{ fontSize: 9, color: "#64748b", marginTop: 1 }}>net of fees</div> : null}
                          </>}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </>
          )}

          {/* AI Recap banner */}
          {entries.length > 0 && (
            <div style={{ marginTop: 20, background: "#0a0e1a", border: "1px solid #1e3a5f", borderRadius: 6, overflow: "hidden" }}>
              <div style={{ padding: "18px 24px", background: "#0a1628", borderBottom: "1px solid #1e293b", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <div style={{ fontSize: 12, color: "#93c5fd", letterSpacing: "0.1em", fontWeight: 600 }}>✦ AI WEEKLY & MONTHLY RECAP</div>
                  <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>Claude analyzes your journal notes and identifies patterns, lessons & focus areas</div>
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <button onClick={() => { setRecapInitMode("weekly"); setView("recap"); }}
                    style={{ background: "#1d4ed8", color: "white", border: "none", padding: "8px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.06em", transition: "all .15s" }}>
                    ANALYSE WEEK →
                  </button>
                  <button onClick={() => { setRecapInitMode("monthly"); setView("recap"); }}
                    style={{ background: "#1d4ed8", color: "white", border: "none", padding: "8px 18px", borderRadius: 4, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: "0.06em", transition: "all .15s" }}>
                    ANALYSE MONTH →
                  </button>
                </div>
              </div>
            </div>
          )}
        </>)}

        {/* NEW/EDIT */}
        {view === "new" && (
          <div>
            <div style={{ marginBottom: 18 }}>
              <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 26, color: "#94a3b8", letterSpacing: "0.1em" }}>{activeEntry ? "EDIT ENTRY" : "NEW ENTRY"}</div>
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
            <div style={{ display: "grid", gridTemplateColumns: "200px 1fr", gap: 18, marginBottom: 18 }}>
              <div><label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>DATE</label><input type="date" value={form.date} onChange={e => f("date", e.target.value)} /></div>
              <div style={{ gridColumn: "2 / -1" }}><label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>{isProp ? "INSTRUMENTS TRADED" : "TICKER / SYMBOL"}</label>
                {isProp ? (() => {
                  // Prop accounts: existing pill selector + custom
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
                        {/* Gross P&L inline — right side */}
                        <div style={{ marginLeft: "auto", display: "flex", flexDirection: "column", gap: 4 }}>
                          <label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", whiteSpace: "nowrap" }}>GROSS P&L ($)</label>
                          <input type="number" placeholder="Auto on import" value={form.pnl} onChange={e => f("pnl", e.target.value)}
                            style={{ width: 160, fontSize: 16, fontWeight: 600, padding: "8px 12px", textAlign: "right" }} />
                        </div>
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
                  // Personal accounts: free-form text input, defaults to Stock context
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <input
                      type="text"
                      placeholder="e.g. AAPL, TSLA, SPY, BTC… (stocks, ETFs, crypto)"
                      value={(form.instruments || []).join(", ")}
                      onChange={e => {
                        const raw = e.target.value;
                        const parsed = raw.split(/[,\s]+/).map(s => s.trim().toUpperCase()).filter(Boolean);
                        f("instruments", raw.trim() ? parsed : []);
                      }}
                      style={{ flex: 1, fontSize: 13 }}
                    />
                    {/* Gross P&L inline — right side */}
                    <div style={{ display: "flex", flexDirection: "column", gap: 4, flexShrink: 0 }}>
                      <label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", whiteSpace: "nowrap" }}>GROSS P&L ($)</label>
                      <input type="number" placeholder="0.00" value={form.pnl} onChange={e => f("pnl", e.target.value)}
                        style={{ width: 160, fontSize: 16, fontWeight: 600, padding: "8px 12px", textAlign: "right" }} />
                    </div>
                  </div>
                )}
              </div>
            </div>
            {/* Fees + Net PnL + optional cash deposit row */}
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14, background: "#0a0e1a", border: "1px solid #1e293b", borderRadius: 4, padding: "8px 14px", flexWrap: "wrap" }}>
              <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", whiteSpace: "nowrap" }}>FEES ($)</div>
              <input type="number" min="0" placeholder="0.00" value={form.commissions} onChange={e => f("commissions", e.target.value)}
                style={{ width: 90, fontSize: 13, padding: "4px 8px" }} />
              <div style={{ width: "1px", height: 20, background: "#1e293b", margin: "0 4px" }} />
              <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", whiteSpace: "nowrap" }}>NET P&L</div>
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
                  style={{ flex: 1, padding: "11px 0", borderRadius: 4, fontFamily: "inherit", fontSize: 12, cursor: t.disabled ? "not-allowed" : "pointer", transition: "all .15s", letterSpacing: "0.06em", opacity: t.disabled ? 0.35 : 1, background: tab === t.id ? "#1e3a5f" : "transparent", border: `1px solid ${tab === t.id ? "#3b82f6" : "#1e293b"}`, color: tab === t.id ? "#93c5fd" : "#94a3b8", textAlign: "center" }}>
                  {t.label.toUpperCase()}
                </button>
              ))}
            </div>

            {tab === "session" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                <div><label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8, display: "block" }}>MARKET BIAS</label><div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>{BIAS_OPTIONS.map(b => <button key={b} className={`pill ${form.bias === b ? "sel" : ""}`} onClick={() => f("bias", b)}>{b.toUpperCase()}</button>)}</div></div>
                <div><label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8, display: "block" }}>MOOD / MENTAL STATE <span style={{ color: "#64748b", fontWeight: 400, letterSpacing: 0 }}>· select all that apply</span></label><div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>{MOOD_OPTIONS.map(m => { const sel = (form.moods || []).includes(m); return <button key={m} className={`pill ${sel ? "sel" : ""}`} onClick={() => { const cur = form.moods || []; f("moods", sel ? cur.filter(x => x !== m) : [...cur, m]); }}>{m}</button>; })}</div></div>
                <div>
                  <label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8, display: "block" }}>SESSION MISTAKES <span style={{ color: "#64748b", fontWeight: 400, letterSpacing: 0 }}>· select all that apply</span></label>
                  <MistakesDropdown selected={form.sessionMistakes || []} onChange={v => f("sessionMistakes", v)} />
                </div>
                <div><label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8, display: "block" }}>SESSION GRADE / SYSTEM ADHERENCE SCORE</label><div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>{GRADE_OPTIONS.map(g => <button key={g} className={`pill ${form.grade === g ? "sel" : ""}`} onClick={() => f("grade", g)} style={form.grade === g ? { borderColor: gradeColor(g), background: "#0f172a", color: gradeColor(g) } : {}}>{g}</button>)}</div></div>
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
                <div><label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>MARKET / TAPE NOTES</label><textarea rows={4} placeholder="Price action, key levels, macro context, structure..." value={form.marketNotes} onChange={e => f("marketNotes", e.target.value)} /></div>
                <div><label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>RULES FOLLOWED / BROKEN</label><textarea rows={3} placeholder="Did you follow your trading plan?" value={form.rules} onChange={e => f("rules", e.target.value)} /></div>
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
                      <span key={b} style={{ fontSize: 9, color: "#64748b", border: "1px solid #1e293b", padding: "2px 8px", borderRadius: 12, letterSpacing: "0.08em" }}>{b}</span>
                    ))}
                  </div>
                </div>

                {/* Textarea */}
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                    <label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase" }}>PASTE BROKER DATA</label>
                    {detectedFormat && <span style={{ fontSize: 9, color: "#3b82f6", background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.2)", padding: "2px 8px", borderRadius: 12 }}>✓ {detectedFormat}</span>}
                  </div>
                  <textarea rows={14} placeholder={"Paste your broker export here — any format works:\n\n• Tradovate Orders CSV\n• Interactive Brokers Flex Query\n• Questrade Activity Export\n• NinjaTrader Trade Log\n• Any tab or comma separated fills data\n\nJust paste and hit the button — AI handles the rest."} value={importRaw} onChange={e => { setImportRaw(e.target.value); setDetectedFormat(""); setImportSuccess(false); setAiParseError(""); }} style={{ fontFamily: "monospace", fontSize: 11 }} />
                </div>

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
              </div>
            )}

            {tab === "analysis" && analytics && (
              <AnalyticsPanel a={analytics} trades={form.parsedTrades} pnlColor={pnlColor} fmtPnl={fmtPnl} analyticsTab={analyticsTab} setAnalyticsTab={setAnalyticsTab} totalFees={parseFloat(form.commissions) || 0} />
            )}

            {tab === "lessons" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                <div><label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>LESSONS LEARNED TODAY</label><textarea rows={5} placeholder="What did the market teach you today?" value={form.lessonsLearned} onChange={e => f("lessonsLearned", e.target.value)} /></div>
                <div><label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>MISTAKES TO AVOID</label><textarea rows={4} placeholder="Be specific. e.g. 'Don't trade after 2 back-to-back losses'" value={form.mistakes} onChange={e => f("mistakes", e.target.value)} /></div>
                <div><label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>AREAS FOR IMPROVEMENT</label><textarea rows={3} placeholder="Entry timing? Holding winners? Cutting losers?" value={form.improvements} onChange={e => f("improvements", e.target.value)} /></div>
                <div><label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>BEST TRADE OF THE DAY</label><textarea rows={3} placeholder="What setup worked? What went right?" value={form.bestTrade} onChange={e => f("bestTrade", e.target.value)} /></div>
                <div><label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>WORST TRADE / MISTAKE</label><textarea rows={3} placeholder="What went wrong? Revenge trade? Poor sizing?" value={form.worstTrade} onChange={e => f("worstTrade", e.target.value)} /></div>
                <div style={{ border: "1px solid #1e3a5f", borderRadius: 6, padding: "14px 16px", background: "#0a1628" }}>
                  <label style={{ fontSize: 10, color: "#93c5fd", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>ONE RULE TO REINFORCE TOMORROW?</label>
                  <textarea rows={2} placeholder="e.g. 'No trades during the Afternoon Deadzone' or 'Size down after first loss'" value={form.reinforceRule || ""} onChange={e => f("reinforceRule", e.target.value)} />
                </div>
              </div>
            )}

            {tab === "tomorrow" && (
              <div><label style={{ fontSize: 10, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" }}>PLAN FOR TOMORROW</label><textarea rows={7} placeholder="Key levels, economic events, intended setups, max loss for the day, goals..." value={form.tomorrow} onChange={e => f("tomorrow", e.target.value)} /></div>
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
          <AIRecapView entries={entries} netPnl={netPnl} fmtPnl={fmtPnl} pnlColor={pnlColor} initMode={recapInitMode} ai={aiCfg} />
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
          <div style={{ background: "#070d1a", border: "1px solid #1e3a5f", borderRadius: 6, padding: "14px 18px", marginBottom: 16, display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
            <div style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.12em" }}>💼 ACCOUNT BALANCE</div>
            <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
              {[
                { l: "STARTING", v: `$${personalBalance.startingBalance.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, c: "#475569" },
                { l: "NET P&L", v: fmtPnl(totalPnL), c: pnlColor(totalPnL) },
                { l: "CURRENT BALANCE", v: `$${personalBalance.currentBalance.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, c: "#e2e8f0" },
              ].map(s => (
                <div key={s.l} style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 9, color: "#94a3b8", letterSpacing: "0.08em", marginBottom: 2 }}>{s.l}</div>
                  <div style={{ fontSize: 14, color: s.c, fontWeight: 500 }}>{s.v}</div>
                </div>
              ))}
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
