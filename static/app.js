/* 板块情绪雷达 前端逻辑（纯静态，无框架） */
'use strict';

const UP = '#ea3943', DOWN = '#0ecb81', ACCENT = '#58a6ff', GOLD = '#f0b90b', PURPLE = '#bc8cff';
const UP_SOFT = '#ff6b74', DOWN_SOFT = '#4fe3a5';
const $ = s => document.querySelector(s);

/* ---------------- 工具 ---------------- */
function toast(msg, ok = false) {
  const el = document.createElement('div');
  el.className = 'toast' + (ok ? ' ok' : '');
  el.textContent = msg;
  $('#toast-wrap').appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

/* 顶部加载条：按进行中的请求数开关 */
let _loading = 0;
function loadingOn() { _loading++; $('#loadbar').classList.add('on'); }
function loadingOff() { _loading = Math.max(0, _loading - 1); if (!_loading) $('#loadbar').classList.remove('on'); }

async function api(path, opts) {
  loadingOn();
  try {
    const r = await fetch(path, opts);
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(d.detail || `HTTP ${r.status}`);
    }
    return await r.json();
  } catch (e) {
    toast('数据加载失败：' + e.message);
    throw e;
  } finally {
    loadingOff();
  }
}

const fmtScore = v => v == null ? '--' : (v > 0 ? '+' : '') + Number(v).toFixed(2);
const fmtPct = (v, d = 0) => v == null ? '--' : (v * 100).toFixed(d) + '%';
const fmtPctSign = (v, d = 1) => v == null ? '--' : (v > 0 ? '+' : '') + (v * 100).toFixed(d) + '%';
const fmtNum = v => v == null ? '--' : Number(v).toLocaleString('zh-CN', { maximumFractionDigits: 2 });
const fmtAmt = v => v == null ? '' : (v >= 1e8 ? (v / 1e8).toFixed(1) + '亿' : (v / 1e4).toFixed(0) + '万');
const cls = v => v == null ? '' : (v >= 0 ? 'num-up' : 'num-down');

/* 数字滚动动画 */
function countUp(el, target, fmt, dur = 900) {
  if (target == null || !isFinite(target)) { el.textContent = '--'; return; }
  const t0 = performance.now();
  const step = now => {
    const k = Math.min(1, (now - t0) / dur);
    const eased = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(target * eased);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ETF pill / 无 ETF 占位 */
function etfPills(etfs) {
  if (!etfs || !etfs.length) return '<span class="etf-none">无直接ETF</span>';
  return etfs.map(e =>
    `<span class="etf-pill"><b>${e.code}</b><span>${e.name}</span></span>`).join('');
}
const firstEtfCode = item => (item.etf && item.etf.length) ? item.etf[0].code : null;

/* 龙头股：无 ETF 板块的替代可操作标的 */
function leadersOf(code) {
  const l = state.overview && state.overview.latest && state.overview.latest.leaders;
  return l ? l[code] : null;
}
function firstLeaderCode(code) {
  const l = leadersOf(code);
  return (l && l.liquidity && l.liquidity.length) ? l.liquidity[0].code : null;
}
function leaderPill(code) {
  const l = leadersOf(code);
  const s = l && l.liquidity && l.liquidity[0];
  if (!s) return '<span class="etf-none">无直接ETF</span>';
  return `<span class="leader-pill" title="无直接ETF，参考板块龙头股">` +
    `<span class="tag">龙头</span><span>${s.name}</span><b>${s.code}</b></span>`;
}
/* 排名表/榜单用：有 ETF 显示 ETF，否则显示龙头股 */
function tradablePills(item) {
  return (item.etf && item.etf.length) ? etfPills(item.etf) : leaderPill(item.code);
}

function skeleton(target, h = 240) {
  const el = typeof target === 'string' ? $(target) : target;
  el.innerHTML = `<div class="skeleton sk-block" style="height:${h}px"></div>`;
}

/* ---------------- 图表管理 ---------------- */
const charts = {};
function getChart(id) {
  if (!charts[id]) charts[id] = echarts.init($('#' + id), null, { renderer: 'canvas' });
  return charts[id];
}
window.addEventListener('resize', () => Object.values(charts).forEach(c => c.resize()));

const AXIS_STYLE = {
  axisLine: { lineStyle: { color: '#30363d' } },
  axisLabel: { color: '#8b949e' },
  splitLine: { lineStyle: { color: '#21262d' } },
};
const TOOLTIP_STYLE = {
  backgroundColor: '#1c2129', borderColor: '#30363d',
  textStyle: { color: '#e6edf3', fontSize: 12 },
};

/* ---------------- 业务字典 ---------------- */
const SIG_META = {
  heating:   { title: '升温榜',  cls: 'sig-heating', desc: '情绪分连续 3 天上升，资金正在进场' },
  cooling:   { title: '降温榜',  cls: 'sig-cooling', desc: '情绪分连续 3 天下降，资金正在撤退' },
  hot:       { title: '亢奋区',  cls: 'sig-hot',     desc: '情绪处于自身历史 90% 以上，不宜追高' },
  cold:      { title: '冰点区',  cls: 'sig-cold',    desc: '情绪处于自身历史 10% 以下，留意反转机会' },
  switching: { title: '切换预警', cls: 'sig-switch',  desc: '前期领涨板块降温、低位板块升温，热点可能正在切换' },
};
const BADGE_NAME = { heating: '升温', cooling: '降温', hot: '亢奋', cold: '冰点', switching_from: '切换' };

const FACTOR_META = {
  limit_up_strength: { name: '涨停强度', desc: '板块里涨停股票的比例，越高说明资金进攻性越强', fmt: v => fmtPct(v, 1) },
  money_effect:      { name: '赚钱效应', desc: '昨天涨停的股票今天的中位涨幅，打板的人有没有赚到钱', fmt: v => fmtPctSign(v / 100, 2) },
  breadth:           { name: '涨跌宽度', desc: '板块内上涨家数比下跌家数多多少，衡量整体人气', fmt: v => fmtPct(v, 1) },
  rps:               { name: '相对强度', desc: '最近 20 天涨幅在全部板块中所处的位置', fmt: v => '超过 ' + fmtPct(v, 0) + ' 板块' },
  turnover_heat:     { name: '热度变化', desc: '成交额占全市场比例最近 5 天的变化（百分点）', fmt: v => (v > 0 ? '+' : '') + Number(v).toFixed(2) + ' pp' },
  promotion_rate:    { name: '晋级率',   desc: '昨天涨停的股票今天继续涨停的比例，衡量资金承接意愿', fmt: v => fmtPct(v, 0) },
};
const FACTOR_ORDER = ['limit_up_strength', 'money_effect', 'breadth', 'rps', 'turnover_heat', 'promotion_rate'];

/* ---------------- 全局状态 ---------------- */
const state = {
  overview: null, backtest: null, health: null,
  boards: {},          // code -> board detail
  currentBoard: null,
  rankSort: { key: 'score', dir: -1 },
  loaded: { today: false, backtest: false },
};

function codeBadges() {
  // code -> [badgeKey,...]
  const m = {};
  const sigs = state.overview.latest.signals;
  for (const key of ['heating', 'cooling', 'hot', 'cold'])
    for (const x of sigs[key]) (m[x.code] = m[x.code] || []).push(key);
  for (const x of sigs.switching) (m[x.from_code] = m[x.from_code] || []).push('switching_from');
  return m;
}
function badgeHtml(keys) {
  return (keys || []).map(k => `<span class="badge badge-${k}">${BADGE_NAME[k]}</span>`).join('');
}

/* ---------------- Tab1 今日榜单 ---------------- */
function renderHealthLamp() {
  const h = state.overview.latest.health;
  const dot = $('#lamp-dot'), txt = $('#lamp-text');
  $('#health-lamp').style.display = 'flex';
  let level, text;
  if (h.ic_mean >= 0.03 && h.ic_win_rate >= 0.55) {
    level = 'g'; text = '信号有效，可以顺势参考';
  } else if (h.ic_mean >= 0) {
    level = 'y'; text = '信号偏弱，参考时打个折扣';
  } else {
    level = 'r'; text = '信号正在失效，建议观望';
  }
  dot.className = 'lamp-dot ' + level;
  txt.textContent = text;
  $('#lamp-sub').textContent =
    `最近60天胜率 ${fmtPct(h.ic_win_rate)} · IC ${h.ic_mean == null ? '--' : h.ic_mean.toFixed(3)}`;
}

function renderSignalCards() {
  const sigs = state.overview.latest.signals;
  const wrap = $('#sig-cards');
  wrap.innerHTML = '';
  for (const key of Object.keys(SIG_META)) {
    const meta = SIG_META[key];
    const items = sigs[key];
    const card = document.createElement('div');
    card.className = `card sig-card ${meta.cls}`;
    const chipsId = `chips-${key}`;
    card.innerHTML = `
      <div class="sig-title"><b>${meta.title}</b><span class="cnt">${items.length} 个</span></div>
      <div class="sig-desc">${meta.desc}</div>
      <div id="${chipsId}"></div>`;
    wrap.appendChild(card);
    const box = card.querySelector('#' + chipsId);
    if (!items.length) {
      box.innerHTML = '<span style="color:var(--text-faint);font-size:12px">今天没有符合条件的板块</span>';
      continue;
    }
    const renderChips = expanded => {
      box.innerHTML = '';
      const show = expanded ? items : items.slice(0, 8);
      for (const it of show) {
        const c = document.createElement('span');
        c.className = 'chip';
        const etfC = firstEtfCode(it);
        const leaderC = etfC ? null : firstLeaderCode(key === 'switching' ? it.from_code : it.code);
        const codeHtml = etfC ? `<span class="c">${etfC}</span>`
          : leaderC ? `<span class="lc" title="无直接ETF，参考板块龙头股">${leaderC}</span>` : '';
        if (key === 'switching') {
          c.innerHTML = `${it.from_name} → ${it.to.map(t => t.name).join('/')}` + codeHtml;
          c.onclick = () => gotoBoard(it.from_code);
        } else {
          c.innerHTML = it.name + codeHtml;
          c.onclick = () => gotoBoard(it.code);
        }
        box.appendChild(c);
      }
      if (!expanded && items.length > 8) {
        const more = document.createElement('span');
        more.className = 'chip more clickable';
        more.textContent = `+${items.length - 8} 展开`;
        more.onclick = () => renderChips(true);
        box.appendChild(more);
      }
    };
    renderChips(false);
  }
}

function renderDiverge() {
  const { top, bottom } = state.overview.latest;
  const rows = [...bottom].reverse().concat([...top].reverse());
  rows.sort((a, b) => a.score - b.score);
  const gradUp = new echarts.graphic.LinearGradient(0, 0, 1, 0,
    [{ offset: 0, color: 'rgba(234,57,67,.25)' }, { offset: 1, color: UP }]);
  const gradDown = new echarts.graphic.LinearGradient(0, 0, 1, 0,
    [{ offset: 0, color: DOWN }, { offset: 1, color: 'rgba(14,203,129,.25)' }]);
  const chart = getChart('chart-diverge');
  chart.setOption({
    animationDuration: 700, animationEasing: 'cubicOut',
    grid: { left: 90, right: 60, top: 10, bottom: 24 },
    tooltip: {
      ...TOOLTIP_STYLE, trigger: 'item',
      formatter: p => {
        const r = rows[p.dataIndex];
        const etf = r.etf && r.etf.length ? `<br>对应ETF：<b style="color:${GOLD}">${r.etf[0].code} ${r.etf[0].name}</b>` : '';
        return `<b>${r.name}</b><br>情绪分：${fmtScore(r.score)}<br>时序分位：${fmtPct(r.pct)}${etf}`;
      },
    },
    xAxis: { type: 'value', ...AXIS_STYLE, axisLabel: { color: '#8b949e', formatter: v => v.toFixed(1) } },
    yAxis: {
      type: 'category', data: rows.map(r => r.name), ...AXIS_STYLE, splitLine: { show: false },
      axisLabel: { color: '#c9d1d9', fontSize: 12 },
    },
    series: [{
      type: 'bar', barWidth: '62%',
      data: rows.map(r => ({
        value: +r.score.toFixed(3),
        itemStyle: {
          color: r.score >= 0 ? gradUp : gradDown,
          borderRadius: r.score >= 0 ? [0, 6, 6, 0] : [6, 0, 0, 6],
        },
      })),
      label: {
        show: true, position: 'right', fontSize: 10, color: '#8b949e',
        formatter: p => p.value > 0 ? p.value.toFixed(2) : '',
      },
    }],
  }, true);
  chart.off('click');
  chart.on('click', p => gotoBoard(rows[p.dataIndex].code));
}

function renderRankTable() {
  const { key, dir } = state.rankSort;
  const rows = [...state.overview.latest.ranking];
  rows.sort((a, b) => ((a[key] ?? -Infinity) - (b[key] ?? -Infinity)) * dir);
  const badges = codeBadges();
  document.querySelectorAll('#rank-table th.sortable').forEach(th => {
    th.querySelector('.sort-mark').textContent =
      th.dataset.sort === key ? (dir === -1 ? '↓' : '↑') : '';
  });
  const tbody = $('#rank-table tbody');
  tbody.innerHTML = rows.map((r, i) => {
    const pctW = r.pct == null ? 0 : Math.round(r.pct * 100);
    const barColor = r.pct >= 0.9 ? GOLD : r.pct <= 0.1 ? ACCENT : (r.score >= 0 ? UP : DOWN);
    return `<tr data-code="${r.code}">
      <td class="l" style="color:var(--text-faint)">${i + 1}</td>
      <td class="l">${r.name}</td>
      <td class="${cls(r.score)} score-cell">${fmtScore(r.score)}</td>
      <td><span class="pct-bar"><i style="width:${pctW}%;background:${barColor}"></i></span>${fmtPct(r.pct)}</td>
      <td class="l">${tradablePills(r)}</td>
      <td class="l">${badgeHtml(badges[r.code])}</td>
    </tr>`;
  }).join('');
  tbody.querySelectorAll('tr').forEach(tr =>
    tr.addEventListener('click', () => gotoBoard(tr.dataset.code)));
}

async function loadToday() {
  if (!state.overview) state.overview = await api('/api/overview');
  const meta = state.overview.meta;
  $('#data-date').textContent =
    `数据日期：${state.overview.latest.date}（共 ${meta.n_boards} 个板块，更新于 ${meta.generated_at}）`;
  renderHealthLamp();
  renderSignalCards();
  renderDiverge();
  renderRankTable();
  state.loaded.today = true;
}

/* ---------------- Tab2 回测验证 ---------------- */
function renderBtHero() {
  const bt = state.backtest;
  const t = bt.rotation.top;
  $('#bt-hero').innerHTML =
    `最近 <b>${bt.window}</b> 个交易日：每周跟着榜单前 ${bt.top_n} 名板块<b>对应的 ETF</b> 买，年化
     <b class="${cls(t.strategy.annual_ret)}" id="hero-strat">--</b>，
     同期大盘（等权全板块）只有
     <b class="${cls(t.benchmark.annual_ret)}" id="hero-bench">--</b>。
     <span style="color:var(--text-faint);font-size:13px">（${bt.start} ~ ${bt.end}，回测用板块指数近似代替 ETF；历史表现，不代表未来）</span>`;
  countUp($('#hero-strat'), t.strategy.annual_ret, v => fmtPctSign(v), 1000);
  countUp($('#hero-bench'), t.benchmark.annual_ret, v => fmtPctSign(v), 1000);
}

function renderIcChart() {
  const s = state.health.series;
  const chart = getChart('chart-ic');
  chart.setOption({
    grid: { left: 50, right: 20, top: 20, bottom: 28 },
    tooltip: { ...TOOLTIP_STYLE, trigger: 'axis',
      formatter: ps => `${ps[0].axisValue}<br>滚动IC：<b class="${cls(ps[0].value)}">${Number(ps[0].value).toFixed(3)}</b>` },
    xAxis: { type: 'category', data: s.map(x => x.date), ...AXIS_STYLE, splitLine: { show: false } },
    yAxis: { type: 'value', ...AXIS_STYLE, axisLabel: { color: '#8b949e', formatter: v => v.toFixed(2) } },
    visualMap: {
      show: false, seriesIndex: 0, dimension: 1,
      pieces: [{ gte: 0, color: UP }, { lt: 0, color: DOWN }],
    },
    series: [{
      type: 'line', data: s.map(x => +x.ic.toFixed(4)), showSymbol: false, lineStyle: { width: 1.6 },
      areaStyle: { opacity: 0.15 },
      markLine: {
        silent: true, symbol: 'none',
        data: [{ yAxis: 0, lineStyle: { color: '#8b949e', type: 'dashed' } },
               { yAxis: 0.03, lineStyle: { color: GOLD, type: 'dotted' },
                 label: { formatter: '0.03 有效线', color: GOLD, fontSize: 10 } }],
      },
    }],
  }, true);
}

function renderQuantChart() {
  const q = state.backtest.quantile;
  const chart = getChart('chart-quant');
  chart.setOption({
    grid: { left: 56, right: 20, top: 20, bottom: 40 },
    tooltip: { ...TOOLTIP_STYLE, trigger: 'item',
      formatter: p => `第 ${p.dataIndex + 1} 组<br>随后5天平均收益：<b>${fmtPctSign(q[p.dataIndex].mean_fwd_ret, 2)}</b><br>样本数：${fmtNum(q[p.dataIndex].n_obs)}` },
    xAxis: {
      type: 'category', ...AXIS_STYLE, splitLine: { show: false },
      data: q.map((_, i) => i === 0 ? 'Q1\n(分数最低)' : i === q.length - 1 ? `Q${i + 1}\n(分数最高)` : `Q${i + 1}`),
      axisLabel: { color: '#c9d1d9', interval: 0, lineHeight: 16 },
    },
    yAxis: { type: 'value', ...AXIS_STYLE, axisLabel: { color: '#8b949e', formatter: v => (v * 100).toFixed(1) + '%' } },
    series: [{
      type: 'bar', barWidth: '50%',
      data: q.map(r => ({
        value: +r.mean_fwd_ret.toFixed(5),
        itemStyle: { color: r.mean_fwd_ret >= 0 ? UP : DOWN, borderRadius: [6, 6, 0, 0] },
      })),
      label: { show: true, position: 'top', fontSize: 11, color: '#c9d1d9', formatter: p => fmtPctSign(p.value, 2) },
    }],
  }, true);
}

const ROT_META = [
  ['top', '追涨：每周买前 N 名', UP],
  ['bottom', '抄底：每周买后 N 名', DOWN],
  ['avoid_top', '回避亢奋：买其余全部', GOLD],
];
function renderNavChart() {
  const rot = state.backtest.rotation;
  const dates = rot.top.nav.map(x => x.date);
  const toMap = arr => Object.fromEntries(arr.map(x => [x.date, x.nav]));
  const series = ROT_META.map(([key, name, color]) => ({
    name, type: 'line', showSymbol: false, lineStyle: { width: 1.8, color },
    itemStyle: { color }, data: dates.map(d => { const v = toMap(rot[key].nav)[d]; return v == null ? null : +v.toFixed(4); }),
  }));
  series.push({
    name: '基准：等权全板块', type: 'line', showSymbol: false,
    lineStyle: { width: 1.4, color: '#8b949e', type: 'dashed' }, itemStyle: { color: '#8b949e' },
    data: dates.map(d => { const v = toMap(rot.top.benchmark_nav)[d]; return v == null ? null : +v.toFixed(4); }),
  });
  const chart = getChart('chart-nav');
  chart.setOption({
    grid: { left: 50, right: 20, top: 40, bottom: 28 },
    legend: { textStyle: { color: '#c9d1d9' }, top: 4 },
    tooltip: { ...TOOLTIP_STYLE, trigger: 'axis', valueFormatter: v => v == null ? '--' : Number(v).toFixed(3) },
    xAxis: { type: 'category', data: dates, ...AXIS_STYLE, splitLine: { show: false } },
    yAxis: { type: 'value', ...AXIS_STYLE, scale: true, axisLabel: { color: '#8b949e', formatter: v => v.toFixed(2) } },
    series,
  }, true);
}

function renderPerfTable() {
  const rot = state.backtest.rotation;
  const rows = ROT_META.map(([key, name]) => {
    const r = rot[key];
    return `<tr>
      <td class="l">${name}</td>
      <td class="${cls(r.strategy.annual_ret)}">${fmtPctSign(r.strategy.annual_ret)}</td>
      <td>${r.strategy.sharpe == null ? '--' : r.strategy.sharpe.toFixed(2)}</td>
      <td class="num-down">${fmtPct(r.strategy.max_dd, 1)}</td>
      <td>${r.final_nav == null ? '--' : r.final_nav.toFixed(3)}</td>
      <td style="color:var(--text-dim)">${fmtPct(r.total_cost_drag, 2)}</td>
    </tr>`;
  });
  const b = rot.top;
  rows.push(`<tr style="color:var(--text-dim)">
    <td class="l">基准：等权全板块</td>
    <td class="${cls(b.benchmark.annual_ret)}">${fmtPctSign(b.benchmark.annual_ret)}</td>
    <td>${b.benchmark.sharpe == null ? '--' : b.benchmark.sharpe.toFixed(2)}</td>
    <td class="num-down">${fmtPct(b.benchmark.max_dd, 1)}</td>
    <td>${b.final_benchmark == null ? '--' : b.final_benchmark.toFixed(3)}</td>
    <td>—</td></tr>`);
  $('#perf-table tbody').innerHTML = rows.join('');
}

function renderIcTable() {
  const ic = state.backtest.ic;
  $('#ic-table tbody').innerHTML = ic.map(r => `<tr>
    <td class="l">${r.horizon} 日</td>
    <td class="${cls(r.IC_mean)}">${r.IC_mean == null ? '--' : r.IC_mean.toFixed(4)}</td>
    <td>${r.IR == null ? '--' : r.IR.toFixed(3)}</td>
    <td>${fmtPct(r.IC_win_rate)}</td>
    <td>${fmtNum(r.n_obs)}</td></tr>`).join('');
  const h5 = ic.find(r => r.horizon === 5) || ic[0];
  const verdict = h5.IC_mean >= 0.03 ? '打分有区分度，可以参考'
    : h5.IC_mean >= 0 ? '区分度偏弱，谨慎参考' : '打分近期没有区分度';
  $('#ic-note').innerHTML =
    `💡 IC 是“打分”和“未来收益”的相关性，<b>&gt; 0.03 才算有用</b>。` +
    `本窗口 ${h5.horizon} 日 IC = <b class="${cls(h5.IC_mean)}">${h5.IC_mean.toFixed(3)}</b>，` +
    `胜率 ${fmtPct(h5.IC_win_rate)} —— ${verdict}。`;
}

async function loadBacktest() {
  if (!state.backtest) state.backtest = await api('/api/backtest');
  if (!state.health) state.health = await api('/api/health_series?days=120');
  renderBtHero();
  renderIcChart();
  renderQuantChart();
  renderNavChart();
  renderPerfTable();
  renderIcTable();
  state.loaded.backtest = true;
}

/* ---------------- Tab3 板块详情 ---------------- */
function ensureBoardSelect() {
  const sel = $('#board-select');
  if (sel.options.length) return;
  for (const r of state.overview.latest.ranking) {
    const o = document.createElement('option');
    o.value = r.code; o.textContent = `${r.name}（${r.code}）`;
    sel.appendChild(o);
  }
  sel.addEventListener('change', () => loadBoard(sel.value));
}

async function loadBoard(code) {
  ensureBoardSelect();
  $('#board-select').value = code;
  state.currentBoard = code;
  if (!state.boards[code]) {
    skeleton('#chart-board-dual', 380); skeleton('#chart-board-pct', 260);
    state.boards[code] = await api(`/api/board/${code}?days=180`);
  }
  renderBoard(state.boards[code]);
}

function renderBoard(b) {
  $('#board-title').textContent = `${b.name}`;
  $('#board-badges').innerHTML = badgeHtml(b.badges);
  $('#board-sub').textContent = b.latest
    ? `情绪分 ${fmtScore(b.latest.score)}（全市场第 ${state.overview.latest.ranking.findIndex(r => r.code === b.code) + 1} 名），时序分位 ${fmtPct(b.latest.pct)}`
    : '';
  $('#factor-date').textContent = b.factor_date;

  // 对应 ETF 卡片
  const ewrap = $('#board-etfs');
  if (b.etf && b.etf.length) {
    ewrap.innerHTML = b.etf.map(e => `
      <div class="etf-card">
        <div>
          <div class="e-code">${e.code}</div>
          <div class="e-name">${e.name}</div>
        </div>
        ${e.price != null ? `<div>
          <div class="e-price ${cls(e.pct_chg)}">${e.price.toFixed(3)}
            <span style="font-size:12px">${e.pct_chg == null ? '' : (e.pct_chg > 0 ? '+' : '') + e.pct_chg.toFixed(2) + '%'}</span></div>
          <div class="e-sub">成交额 ${fmtAmt(e.amount)}</div>
        </div>` : ''}
      </div>`).join('');
  } else {
    ewrap.innerHTML = '<div class="etf-none" style="padding:6px 2px">该板块无直接对应 ETF，可参考下方龙头股</div>';
  }

  // 板块龙头：流动性龙头 Top3 + 情绪龙头（连板）
  const lwrap = $('#board-leaders');
  const ld = b.leaders || { liquidity: [], momentum: [] };
  let lhtml = '';
  if (ld.liquidity && ld.liquidity.length) {
    lhtml += '<div class="leader-card"><div class="l-title">流动性龙头 · 近5日日均成交额</div>' +
      ld.liquidity.map(s =>
        `<div class="l-row"><span>${s.name}</span><span class="code">${s.code}</span>` +
        `<span class="amt">${fmtAmt(s.avg_amount)}</span></div>`).join('') + '</div>';
  }
  if (ld.momentum && ld.momentum.length) {
    lhtml += '<div class="leader-card"><div class="l-title">情绪龙头 · 近10日连板</div>' +
      ld.momentum.map(s =>
        `<div class="l-row"><span>${s.name}</span><span class="code">${s.code}</span>` +
        `<span class="streak-badge">${s.max_streak >= 2 ? s.max_streak + '连板' : '首板'}</span></div>`).join('') + '</div>';
  }
  lwrap.innerHTML = lhtml;

  const s120 = b.score.slice(-120);
  const kMap = Object.fromEntries(b.kline.map(k => [k.date, k.close]));
  const dates = s120.map(x => x.date);
  getChart('chart-board-dual').setOption({
    grid: { left: 56, right: 64, top: 36, bottom: 28 },
    legend: { textStyle: { color: '#c9d1d9' }, top: 4 },
    tooltip: { ...TOOLTIP_STYLE, trigger: 'axis' },
    xAxis: { type: 'category', data: dates, ...AXIS_STYLE, splitLine: { show: false } },
    yAxis: [
      { type: 'value', name: '情绪分', nameTextStyle: { color: '#8b949e' }, ...AXIS_STYLE, scale: true },
      { type: 'value', name: '指数收盘', nameTextStyle: { color: '#8b949e' }, ...AXIS_STYLE, scale: true,
        splitLine: { show: false } },
    ],
    series: [
      { name: '情绪分', type: 'line', data: s120.map(x => x.score == null ? null : +x.score.toFixed(3)),
        showSymbol: false, lineStyle: { width: 2, color: UP }, itemStyle: { color: UP } },
      { name: '板块指数', type: 'line', yAxisIndex: 1, data: dates.map(d => kMap[d] ?? null),
        showSymbol: false, lineStyle: { width: 1.4, color: ACCENT }, itemStyle: { color: ACCENT } },
    ],
  }, true);

  const p120 = b.pct.slice(-120);
  getChart('chart-board-pct').setOption({
    grid: { left: 50, right: 20, top: 16, bottom: 28 },
    tooltip: { ...TOOLTIP_STYLE, trigger: 'axis', valueFormatter: v => v == null ? '--' : fmtPct(v) },
    xAxis: { type: 'category', data: p120.map(x => x.date), ...AXIS_STYLE, splitLine: { show: false } },
    yAxis: { type: 'value', min: 0, max: 1, ...AXIS_STYLE, axisLabel: { color: '#8b949e', formatter: v => (v * 100) + '%' } },
    series: [{
      name: '时序分位', type: 'line', data: p120.map(x => x.pct == null ? null : +x.pct.toFixed(3)),
      showSymbol: false, lineStyle: { width: 1.6, color: GOLD }, itemStyle: { color: GOLD },
      areaStyle: { color: GOLD, opacity: 0.12 },
      markLine: {
        silent: true, symbol: 'none',
        data: [
          { yAxis: 0.9, lineStyle: { color: UP, type: 'dashed' }, label: { formatter: '90% 亢奋线', color: UP, fontSize: 10 } },
          { yAxis: 0.1, lineStyle: { color: ACCENT, type: 'dashed' }, label: { formatter: '10% 冰点线', color: ACCENT, fontSize: 10 } },
        ],
      },
    }],
  }, true);

  const wrap = $('#factor-cards');
  wrap.innerHTML = '';
  for (const fk of FACTOR_ORDER) {
    const meta = FACTOR_META[fk];
    const f = b.factors[fk];
    const card = document.createElement('div');
    card.className = 'card factor-card';
    if (!f) {
      card.innerHTML = `<div class="f-name">${meta.name}</div><div class="f-desc">${meta.desc}</div>
        <div class="f-value" style="color:var(--text-faint)">--</div>
        <div class="f-rank">当日无数据（样本不足）</div>`;
    } else {
      const rk = f.rank == null ? 0 : Math.round(f.rank * 100);
      const rkColor = rk >= 80 ? UP : rk <= 20 ? DOWN : ACCENT;
      card.innerHTML = `<div class="f-name">${meta.name}</div><div class="f-desc">${meta.desc}</div>
        <div class="f-value">${meta.fmt(f.value)}</div>
        <div class="f-rank"><span class="pct-bar" style="width:110px"><i style="width:${rk}%;background:${rkColor}"></i></span>
        超过全市场 ${rk}% 的板块</div>`;
    }
    wrap.appendChild(card);
  }
}

async function gotoBoard(code) {
  state.currentBoard = code;   // 先占位，避免 switchTab 默认加载榜首覆盖目标板块
  await switchTab('board');
}

/* ---------------- Tab 切换 ---------------- */
async function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.tab-page').forEach(p => p.classList.toggle('active', p.id === 'page-' + name));
  try {
    if (name === 'today' && !state.loaded.today) await loadToday();
    if (name === 'backtest' && !state.loaded.backtest) {
      skeleton('#chart-ic', 300); skeleton('#chart-quant', 300); skeleton('#chart-nav', 420);
      await loadBacktest();
    }
    if (name === 'board') {
      if (!state.overview) state.overview = await api('/api/overview');
      ensureBoardSelect();
      if (!state.currentBoard) await loadBoard(state.overview.latest.top[0].code);
      else await loadBoard(state.currentBoard);
    }
    // tab 显示后触发 resize，避免图表宽度为 0
    requestAnimationFrame(() => Object.values(charts).forEach(c => c.resize()));
  } catch (e) { /* toast 已提示 */ }
}
document.querySelectorAll('.tab').forEach(t =>
  t.addEventListener('click', () => switchTab(t.dataset.tab)));

/* ---------------- 刷新数据 ---------------- */
let polling = false;
async function refreshData() {
  const btn = $('#btn-refresh');
  btn.disabled = true;
  btn.innerHTML = '<span class="spin">⟳</span> 计算中…';
  try {
    const r = await api('/api/refresh', { method: 'POST' });
    $('#refresh-msg').textContent = r.message;
    if (!r.started) { btn.disabled = false; btn.innerHTML = '⟳ 刷新数据'; return; }
    if (polling) return;
    polling = true;
    const timer = setInterval(async () => {
      try {
        const st = await api('/api/refresh_status');
        if (!st.running) {
          clearInterval(timer); polling = false;
          btn.disabled = false; btn.innerHTML = '⟳ 刷新数据';
          if (st.ok) {
            $('#refresh-msg').textContent = `已更新（${st.finished_at}）`;
            toast('数据刷新完成', true);
            state.overview = null; state.backtest = null; state.health = null;
            state.boards = {}; state.loaded = { today: false, backtest: false };
            const active = document.querySelector('.tab.active').dataset.tab;
            state.currentBoard = null;
            $('#board-select').innerHTML = '';   // 排名可能变化，强制重建下拉选项
            await switchTab(active);
          } else {
            $('#refresh-msg').textContent = '刷新失败';
            toast('刷新失败：' + (st.log_tail || '未知错误').split('\n').pop());
          }
        }
      } catch (e) { /* 轮询失败下轮再试 */ }
    }, 3000);
  } catch (e) {
    btn.disabled = false; btn.innerHTML = '⟳ 刷新数据';
  }
}
$('#btn-refresh').addEventListener('click', refreshData);

/* ---------------- 表头排序 ---------------- */
document.querySelectorAll('#rank-table th.sortable').forEach(th =>
  th.addEventListener('click', () => {
    const key = th.dataset.sort;
    if (state.rankSort.key === key) state.rankSort.dir *= -1;
    else state.rankSort = { key, dir: -1 };
    renderRankTable();
  }));

/* ---------------- 启动 ---------------- */
(async function boot() {
  skeleton('#chart-diverge', 520);
  $('#sig-cards').innerHTML = Array.from({ length: 5 },
    () => '<div class="card"><div class="skeleton sk-line" style="width:60%"></div><div class="skeleton sk-line"></div><div class="skeleton sk-line" style="width:80%"></div></div>').join('');
  try { await loadToday(); } catch (e) { /* toast 已提示 */ }
})();
