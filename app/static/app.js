/* 墙面霉菌检测 · 手机端 H5 */
(() => {
'use strict';

const $ = (id) => document.getElementById(id);
const shots = [];              // {file, url, hint}
let current = null;            // 当前检测结果
let activeSrc = 0;             // 当前显示的图片下标
const imgCache = {};

const LEVEL_VARS = {L0:'--l0',L1:'--l1',L2:'--l2',L3:'--l3',L4:'--l4',L5:'--l5',
                    UNKNOWN:'--lu',L0_UNVERIFIED:'--l0'};
const CH_NAME = {H:'荧光活性', M:'覆盖', L:'深度厚度'};

/* ---------- 视图切换 ---------- */
function show(view){
  ['view-capture','view-result','view-history'].forEach(v => $(v).classList.toggle('hidden', v!==view));
  window.scrollTo(0,0);
}
function toast(msg, ms=2000){
  const t=$('toast'); t.textContent=msg; t.classList.remove('hidden');
  clearTimeout(t._h); t._h=setTimeout(()=>t.classList.add('hidden'), ms);
}
function loading(on, text){
  $('loading').classList.toggle('hidden', !on);
  if(text) $('loading-text').textContent = text;
}

/* ---------- 采集 ---------- */
$('file-input').addEventListener('change', (e) => {
  for (const f of e.target.files){
    if (shots.length >= 6){ toast('一次最多 6 张'); break; }
    shots.push({file:f, url:URL.createObjectURL(f), hint:''});
  }
  e.target.value = '';
  renderShots();
});

function renderShots(){
  const box = $('shots');
  box.innerHTML = '';
  shots.forEach((s, i) => {
    const el = document.createElement('div');
    el.className = 'shot';
    el.innerHTML = `<img src="${s.url}" alt="照片 ${i+1}">
      <div class="shot-bar">
        <select data-i="${i}">
          <option value=""${s.hint===''?' selected':''}>自动判别光源</option>
          <option value="white"${s.hint==='white'?' selected':''}>白光照</option>
          <option value="uv"${s.hint==='uv'?' selected':''}>紫光照 395nm</option>
        </select>
        <button class="del" data-del="${i}" aria-label="删除">×</button>
      </div>`;
    box.appendChild(el);
  });
  box.querySelectorAll('select').forEach(sel => {
    sel.addEventListener('change', (e) => { shots[+e.target.dataset.i].hint = e.target.value; updateHint(); });
  });
  box.querySelectorAll('[data-del]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const i = +e.target.dataset.del;
      URL.revokeObjectURL(shots[i].url);
      shots.splice(i,1); renderShots();
    });
  });
  $('btn-detect').disabled = shots.length === 0;
  updateHint();
}

function updateHint(){
  const t = $('coverage-text');
  if (!shots.length){
    t.textContent = '还没有添加照片。白光照可检测 L2–L5，395nm 紫光照可检测 L1–L4，两者都有可覆盖 L1–L5。';
    return;
  }
  const hasWhite = shots.some(s => s.hint === 'white');
  const hasUV = shots.some(s => s.hint === 'uv');
  const auto = shots.some(s => s.hint === '');
  if (hasWhite && hasUV) t.textContent = '证据完整，可判 L1–L5。';
  else if (hasUV) t.textContent = '已可检测 L1–L4。补一张白光照可确认面积口径与是否存在基材损伤，并把 L4/L5 区分开。';
  else if (hasWhite) t.textContent = '已可检测 L2–L5。补一张 395nm 紫光照可识别 L1 隐形萌发期。';
  else if (auto) t.textContent = '将自动判别每张照片的光源类型。也可以手动指定，判别更稳。';
}

$('btn-detect').addEventListener('click', async () => {
  if (!shots.length) return;
  const fd = new FormData();
  shots.forEach(s => fd.append('images', s.file, s.file.name || 'photo.jpg'));
  fd.append('modality_hint', JSON.stringify(shots.map(s => s.hint || null)));

  loading(true, '检测中…');
  try{
    const res = await fetch('/api/detect', {method:'POST', body:fd});
    if(!res.ok){
      const err = await res.json().catch(()=>({detail:res.statusText}));
      throw new Error(err.detail || '检测失败');
    }
    current = await res.json();
    activeSrc = 0;
    renderResult();
    show('view-result');
  }catch(err){
    toast(err.message || '检测失败', 3500);
  }finally{
    loading(false);
  }
});

$('btn-again').addEventListener('click', () => {
  shots.splice(0).forEach(s => URL.revokeObjectURL(s.url));
  renderShots(); show('view-capture');
});

/* ---------- 结果渲染 ---------- */
function renderResult(){
  const s = current.summary;
  const color = `var(${LEVEL_VARS[s.level] || '--lu'})`;

  $('level-banner').style.borderLeftColor = color;
  $('level-tag').textContent = s.level.replace('_UNVERIFIED','');
  $('level-tag').style.color = color;
  $('level-name').textContent = s.level_name;

  const r = s.level_range || [];
  $('level-range').textContent = (r.length===2 && r[0]!==r[1])
    ? `可确定区间 ${r[0]} – ${r[1]}　·　I_mold ${s.i_mold}　·　置信度 ${s.confidence}`
    : `I_mold ${s.i_mold}　·　置信度 ${s.confidence}` + (s.conflict ? '　·　规则与指数有冲突，已按更严重的一级' : '');

  const sev = s.severity_in_level || 0;
  $('severity-wrap').classList.toggle('hidden', s.level === 'UNKNOWN' || s.level.startsWith('L0'));
  $('severity-val').textContent = sev.toFixed(2);
  $('severity-bar').style.width = (sev*100).toFixed(0) + '%';
  $('severity-bar').style.background = color;

  const hint = $('confirm-hint');
  const msg = s.unknown_reasons && s.unknown_reasons.length
      ? '无法判定：' + s.unknown_reasons.join('；')
      : (s.needs_confirmation ? s.confirmation_hint : '');
  hint.textContent = msg;
  hint.classList.toggle('hidden', !msg);

  // 三通道
  const ch = $('channels'); ch.innerHTML = '';
  ['H','M','L'].forEach(k => {
    const v = s.channels[k], w = s.channel_weights[k] || 0;
    const el = document.createElement('div');
    el.className = 'ch' + (v===null||v===undefined ? ' off' : '');
    el.innerHTML = `<div class="ch-h"><span class="ch-sym">${k}</span><span class="ch-w">权重 ${w.toFixed(2)}</span></div>
      <div class="ch-name">${CH_NAME[k]}</div>
      <div class="ch-val">${v===null||v===undefined ? '本次输入无此证据' : v.toFixed(3)}</div>
      <div class="bar" style="margin-top:8px"><div class="bar-fill" style="width:${((v||0)*100).toFixed(0)}%"></div></div>`;
    ch.appendChild(el);
  });

  renderTabs();
  renderStats();
  renderAdvice();
  drawCanvas();

  $('btn-save-img').href = `/api/records/${current.record_id}/annotated.png?src=${activeSrc}`;
  $('btn-save-img').setAttribute('download', `${current.record_id}_${activeSrc}.png`);
  $('btn-pdf').href = `/api/records/${current.record_id}/report.pdf`;
}

function renderTabs(){
  const tabs = $('canvas-tabs'); tabs.innerHTML = '';
  current.inputs.forEach((inp, i) => {
    const b = document.createElement('button');
    b.textContent = `${i+1}. ${inp.modality === 'uv' ? '紫光 395nm' : (inp.modality === 'white' ? '白光' : '未判别')}`;
    b.className = i === activeSrc ? 'on' : '';
    b.addEventListener('click', () => { activeSrc = i; renderTabs(); drawCanvas();
      $('btn-save-img').href = `/api/records/${current.record_id}/annotated.png?src=${i}`; });
    tabs.appendChild(b);
  });
}

function renderStats(){
  const s = current.summary, box = $('stats');
  const rows = [];
  const names = {}; current.instances.forEach(i => names[i.cls] = i.cls_cn);
  const fmtCount = (obj) => Object.entries(obj).map(([k,v]) => `${names[k]||k} ${v}`).join('、') || '无';

  rows.push(['检出目标', `${s.instance_count} 个　（${fmtCount(s.by_class)}）`]);
  if (Object.keys(s.excluded||{}).length) rows.push(['已排除的干扰项', fmtCount(s.excluded)]);
  if (s.visible_area_ratio !== null && s.visible_area_ratio !== undefined)
    rows.push(['白光可见霉斑面积占比', (s.visible_area_ratio*100).toFixed(2) + '%']);
  if (s.fluorescent_area_ratio !== null && s.fluorescent_area_ratio !== undefined)
    rows.push(['紫外荧光区面积占比', (s.fluorescent_area_ratio*100).toFixed(2) + '%　（口径与白光不同）']);
  if (s.depth_band) rows.push(['深度档位', s.depth_band + '　（视觉估计，非实测）']);
  if (s.dominant_fluor_pattern) rows.push(['主要荧光形态', names[s.dominant_fluor_pattern] || s.dominant_fluor_pattern]);
  if (s.rules_hit && s.rules_hit.length) rows.push(['命中判定规则', s.rules_hit.join('、')]);
  const inp = current.inputs[activeSrc];
  if (inp) rows.push(['有效墙面占比', (inp.wall_area_ratio*100).toFixed(0) + '%' + (inp.wall_area_fallback ? '　（分割失败，已回退整幅画面）' : '')]);
  if (current.robot_action) rows.push(['机器人指令', current.robot_action.action + (current.robot_action.note ? '　·　' + current.robot_action.note : '')]);

  box.innerHTML = rows.map(([k,v]) => `<div class="stat-row"><span class="k">${k}</span><span class="v">${v}</span></div>`).join('');
}

function renderAdvice(){
  const rec = current.recommendations;
  const li = (arr) => arr.map(x => `<li>${x}</li>`).join('');
  $('advice').innerHTML = `
    <h3>${rec.title}</h3>
    <div class="act">建议处置：${rec.action}</div>
    <ul>${li(rec.steps)}</ul>
    ${rec.ppe ? `<div class="sub">防护</div><div style="font-size:14px;color:var(--ink-2)">${rec.ppe}</div>` : ''}
    ${rec.recheck ? `<div class="sub">复检</div><div style="font-size:14px;color:var(--ink-2)">${rec.recheck}</div>` : ''}
    ${rec.class_notes && rec.class_notes.length ? `<div class="sub">按类别补充</div><ul>${li(rec.class_notes)}</ul>` : ''}
    <div class="sub">安全提示</div><ul>${li(rec.safety || [])}</ul>
    <div class="disclaimer">${current.disclaimer || ''}</div>`;
}

/* ---------- Canvas ---------- */
function drawCanvas(){
  const cv = $('canvas'), ctx = cv.getContext('2d');
  const url = `/api/records/${current.record_id}/source.jpg?src=${activeSrc}`;
  const img = imgCache[url] || new Image();
  imgCache[url] = img;

  const paint = () => {
    const maxW = cv.parentElement.clientWidth - 20;
    const scale = maxW / img.naturalWidth;
    cv.width = img.naturalWidth; cv.height = img.naturalHeight;
    cv.style.height = (img.naturalHeight * scale) + 'px';
    ctx.drawImage(img, 0, 0);

    const insts = current.instances.filter(i => i.source_index === activeSrc);
    insts.forEach(inst => {
      if (!inst.polygon || inst.polygon.length < 3) return;
      const neg = inst.group === 'negative' || inst.group === 'fluor_negative';
      ctx.beginPath();
      inst.polygon.forEach(([x,y], k) => k ? ctx.lineTo(x,y) : ctx.moveTo(x,y));
      ctx.closePath();
      ctx.fillStyle = neg ? 'rgba(120,140,190,.28)' : 'rgba(255,90,60,.30)';
      ctx.fill();
      ctx.lineWidth = Math.max(2, img.naturalWidth/450);
      ctx.strokeStyle = neg ? 'rgba(150,170,220,.95)' : 'rgba(255,120,80,.95)';
      ctx.setLineDash(neg ? [10,7] : []);
      ctx.stroke();
      ctx.setLineDash([]);

      const r = Math.max(13, img.naturalWidth/58);
      const cx = inst.bbox[0] + r, cy = Math.max(inst.bbox[1] - r, r);
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI*2);
      ctx.fillStyle = 'rgba(15,20,16,.9)'; ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.font = `600 ${Math.round(r*1.15)}px sans-serif`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(String(inst.id), cx, cy);
    });
  };

  if (img.complete && img.naturalWidth) paint();
  else { img.onload = paint; img.src = url; }
}

$('canvas').addEventListener('click', (e) => {
  if (!current) return;
  const cv = $('canvas'), rect = cv.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (cv.width / rect.width);
  const y = (e.clientY - rect.top) * (cv.height / rect.height);

  const hit = current.instances
    .filter(i => i.source_index === activeSrc)
    .filter(i => pointInPoly(x, y, i.polygon) || inBox(x, y, i.bbox))
    .sort((a,b) => a.area_px - b.area_px)[0];     // 命中多个时取最小的那个，更符合直觉
  if (hit) openSheet(hit); 
});

function inBox(x,y,b){ return b && x>=b[0] && x<=b[2] && y>=b[1] && y<=b[3]; }
function pointInPoly(x, y, poly){
  if (!poly || poly.length < 3) return false;
  let inside = false;
  for (let i=0, j=poly.length-1; i<poly.length; j=i++){
    const [xi,yi] = poly[i], [xj,yj] = poly[j];
    if (((yi>y) !== (yj>y)) && (x < (xj-xi)*(y-yi)/((yj-yi)||1e-9)+xi)) inside = !inside;
  }
  return inside;
}

/* ---------- 详情弹层 ---------- */
let sheetInstance = null;
function openSheet(inst){
  sheetInstance = inst;
  $('sheet-title').textContent = `#${inst.id}　${inst.cls_cn}`;
  $('sheet-crop').src = `/api/records/${current.record_id}/instances/${inst.id}/crop.jpg`;

  const f = inst.features || {};
  const rows = [
    ['光源', inst.modality === 'uv' ? '紫光 395nm' : '白光'],
    ['置信度', inst.conf.toFixed(2)],
    ['位置', inst.position_hint],
    ['面积', `${inst.area_px} px　·　占有效墙面 ${(inst.area_ratio*100).toFixed(2)}%` +
             (inst.area_cm2 ? `　·　${inst.area_cm2} cm²` : '')],
  ];
  if (inst.density) rows.push(['密度档', `${inst.density}　(${inst.density_score})`]);
  if (inst.depth_band) rows.push(['深度档位', `${inst.depth_band}　(厚度分 ${inst.thickness_score})　视觉估计`]);
  if (inst.uv_response !== null && inst.uv_response !== undefined) rows.push(['荧光响应', inst.uv_response.toFixed(3)]);
  if (inst.group === 'negative' || inst.group === 'fluor_negative') rows.push(['归类', '已排除项（不计入霉菌统计）']);

  const barRow = (k, label) => f[k] === undefined ? '' :
    `<div class="field"><span class="k">${label}</span><span class="v">${f[k].toFixed(2)}
      <div class="feat-bar"><i style="width:${Math.min(100,f[k]*100).toFixed(0)}%"></i></div></span></div>`;

  $('sheet-fields').innerHTML =
    rows.map(([k,v]) => `<div class="field"><span class="k">${k}</span><span class="v">${v}</span></div>`).join('')
    + barRow('dark_ratio','暗像素比') + barRow('fill_ratio','致密度')
    + barRow('shadow_grad','阴影梯度') + barRow('texture_energy','纹理能量');

  const note = (current.recommendations.class_notes || []).find(n => n.includes(inst.cls_cn));
  $('sheet-advice').innerHTML = note
    ? `<div class="sub" style="font-size:12px;color:var(--ink-3);letter-spacing:.08em;margin-top:14px">处理建议</div>
       <div style="font-size:14px;color:var(--ink-2);margin-top:6px">${note}</div>`
    : '';

  $('sheet').classList.remove('hidden');
}
$('sheet-close').addEventListener('click', () => $('sheet').classList.add('hidden'));
$('sheet').addEventListener('click', (e) => { if (e.target === $('sheet')) $('sheet').classList.add('hidden'); });

document.querySelectorAll('.fb-btns button').forEach(btn => {
  btn.addEventListener('click', async () => {
    if (!current) return;
    const fd = new FormData();
    fd.append('kind', btn.dataset.kind);
    if (sheetInstance) fd.append('instance_id', sheetInstance.id);
    try{
      await fetch(`/api/records/${current.record_id}/feedback`, {method:'POST', body:fd});
      toast('已记录反馈，会用于后续模型改进');
      $('sheet').classList.add('hidden');
    }catch(e){ toast('反馈提交失败'); }
  });
});

/* ---------- 历史 ---------- */
$('btn-history').addEventListener('click', async () => {
  try{
    const data = await (await fetch('/api/records?limit=50')).json();
    $('history-list').innerHTML = data.items.length
      ? data.items.map(r => `<div class="hrec" data-id="${r.record_id}">
          <span class="lv" style="color:var(${LEVEL_VARS[r.level]||'--lu'})">${r.level.replace('_UNVERIFIED','')}</span>
          <span class="meta">${r.created_at.replace('T',' ').slice(0,19)}<br>${r.modalities}　·　${r.n_instances} 个目标　·　I=${r.i_mold}</span>
        </div>`).join('')
      : '<div class="hint-card"><div class="hint-body">还没有检测记录。</div></div>';
    $('history-list').querySelectorAll('.hrec').forEach(el => {
      el.addEventListener('click', async () => {
        current = await (await fetch(`/api/records/${el.dataset.id}`)).json();
        activeSrc = 0; renderResult(); show('view-result');
      });
    });
    show('view-history');
  }catch(e){ toast('读取历史失败'); }
});
$('btn-back').addEventListener('click', () => show(current ? 'view-result' : 'view-capture'));

renderShots();
})();
