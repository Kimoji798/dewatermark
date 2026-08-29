"use strict";
// 豆包AI水印处理分两步,职责分离:
//
// 【识别】有没有水印:在右下角区域做白色字形连通域分析,并查集聚成文字行,
//   判断是否存在「贴右边缘 + 高度/宽高比/位置像水印」的白色文字行。
//   多个亮度阈值扫描以对抗锯齿(阈值过高断字、过低与背景亮区粘连),
//   这一步只回答 found: true / false。
//
// 【画框】水印在哪:识别出的字形坐标不可靠——背景亮物与字形粘连时,
//   聚类会把无关区域并进同一行,导致框漂移(实测红框跑到水印上方)。
//   豆包水印固定在右下角、相对图片的位置和大小稳定(样图 doubaoAI.jpg 实测:
//   文字行 x≈62%-100%W、y≈71%-83%H),因此红框按右下角模板生成,
//   并在模板内取识别出的文字字形笔画做逐像素掩膜(背景亮物不进掩膜),
//   保证位置固定、去除精准。
(function () {
  const MAX_ANALYZE = 800;   // 分析用缩小图最大边(快且抗噪)
  const THRESHOLDS = [210, 215, 205, 220, 200, 225, 230, 235, 240, 195, 245];

  // 豆包水印固定在右下角,相对位置/大小稳定(基于样图实测并留余量):
  // 文字行实际约 x62%-100%W、y71%-83%H。
  const BOX_LEFT = 0.58, BOX_TOP = 0.69, BOX_BOTTOM = 0.85;

  // 右下角模板框(分析图坐标)
  function templateBox(W, H) {
    const x0 = Math.round(W * BOX_LEFT);
    const x1 = W - 1;
    const y0 = Math.round(H * BOX_TOP);
    const y1 = Math.round(H * BOX_BOTTOM);
    return { x0, x1, y0, y1, w: x1 - x0 + 1, h: y1 - y0 + 1 };
  }

  // 从 RGBA 像素里找候选文字行(纯函数,方便在 Node 里验证)
  function findLines(data, W, H, T) {
    const x0 = Math.floor(W * 0.55);
    const y0 = Math.floor(H * 0.60);
    const visited = new Uint8Array(W * H);
    const stack = new Int32Array(W * H);
    const comps = [];
    for (let sy = y0; sy < H; sy++) {
      for (let sx = x0; sx < W; sx++) {
        const si = sy * W + sx;
        if (visited[si]) continue;
        const p = si * 4;
        if (data[p] < T || data[p + 1] < T || data[p + 2] < T) continue;
        let sp = 0;
        stack[sp++] = si;
        visited[si] = 1;
        let minX = sx, maxX = sx, minY = sy, maxY = sy, area = 0;
        while (sp > 0) {
          const ci = stack[--sp];
          const cx = ci % W, cy = (ci / W) | 0;
          if (cx < minX) minX = cx;
          if (cx > maxX) maxX = cx;
          if (cy < minY) minY = cy;
          if (cy > maxY) maxY = cy;
          area++;
          for (let dy = -1; dy <= 1; dy++) {
            const ny = cy + dy;
            if (ny < y0 || ny >= H) continue;
            for (let dx = -1; dx <= 1; dx++) {
              if (dx === 0 && dy === 0) continue;
              const nx = cx + dx;
              if (nx < x0 || nx >= W) continue;
              const ni = ny * W + nx;
              if (visited[ni]) continue;
              const np = ni * 4;
              if (data[np] >= T && data[np + 1] >= T && data[np + 2] >= T) {
                visited[ni] = 1;
                stack[sp++] = ni;
              }
            }
          }
        }
        comps.push({ minX, maxX, minY, maxY, w: maxX - minX + 1, h: maxY - minY + 1, area });
      }
    }
    // 字形候选：过滤大块亮物/噪声/近实心色块
    const glyphs = comps.filter((c) =>
      c.h >= 4 && c.h <= H * 0.12 &&
      c.w >= 3 && c.w <= W * 0.6 &&
      c.area >= 4 &&
      c.area / (c.w * c.h) >= 0.03 && c.area / (c.w * c.h) <= 0.92
    );
    // 并查集聚成行：水平间隙小且垂直中心对齐
    const parent = glyphs.map((_, i) => i);
    const find = (i) => (parent[i] === i ? i : (parent[i] = find(parent[i])));
    const union = (a, b) => { parent[find(a)] = find(b); };
    for (let i = 0; i < glyphs.length; i++) {
      for (let j = i + 1; j < glyphs.length; j++) {
        const a = glyphs[i], b = glyphs[j];
        const gapX = Math.max(0, Math.max(a.minX, b.minX) - Math.min(a.maxX, b.maxX));
        const dcy = Math.abs((a.minY + a.maxY) / 2 - (b.minY + b.maxY) / 2);
        const maxH = Math.max(a.h, b.h);
        if (gapX <= 2.5 * maxH && dcy <= 0.8 * maxH) union(i, j);
      }
    }
    const groups = new Map();
    glyphs.forEach((c, i) => {
      const r = find(i);
      if (!groups.has(r)) groups.set(r, []);
      groups.get(r).push(c);
    });
    const lines = [];
    for (const g of groups.values()) {
      if (g.length < 2) continue;
      let ux0 = Infinity, ux1 = -Infinity, uy0 = Infinity, uy1 = -Infinity, area = 0;
      for (const c of g) {
        ux0 = Math.min(ux0, c.minX); ux1 = Math.max(ux1, c.maxX);
        uy0 = Math.min(uy0, c.minY); uy1 = Math.max(uy1, c.maxY);
        area += c.area;
      }
      const uw = ux1 - ux0 + 1, uh = uy1 - uy0 + 1;
      lines.push({ ux0, ux1, uy0, uy1, uw, uh, area, n: g.length, comps: g });
    }
    // 行过滤：贴右边缘 + 尺寸/位置合理
    const rightTol = Math.max(3, Math.round(W * 0.02));
    return lines.filter((l) => {
      const rightMargin = W - 1 - l.ux1;
      const cy = (l.uy0 + l.uy1) / 2;
      if (rightMargin < 0 || rightMargin > rightTol) return false;
      if (l.uh < H * 0.02 || l.uh > H * 0.15) return false;
      if (l.uw / l.uh < 1.8 || l.uw / l.uh > 14) return false;
      if (cy < H * 0.58 || l.uy1 < H * 0.7) return false;
      const fill = l.area / (l.uw * l.uh);
      return fill >= 0.02 && fill <= 0.7;
    });
  }

  function detectFromPixels(data, W, H, opts) {
    const debug = !!(opts && opts.debug);
    const dbg = debug ? { byT: [] } : null;
    let best = null;
    for (const T of THRESHOLDS) {
      const lines = findLines(data, W, H, T);
      if (debug) {
        dbg.byT.push({
          T,
          lines: lines.map((l) => ({
            x: l.ux0, y: l.uy0, w: l.uw, h: l.uh, n: l.n,
            rightMargin: W - 1 - l.ux1,
            aspect: +(l.uw / l.uh).toFixed(2),
            cy: +(((l.uy0 + l.uy1) / 2) / H).toFixed(3),
          })),
        });
      }
      for (const l of lines) {
        // 主字形数:高度达到行高 35% 的字形(排除高阈值下笔画碎裂的小碎片)
        const nSub = l.comps.filter((c) => c.h >= l.uh * 0.35).length;
        const score = nSub * 1000 + l.area - (W - 1 - l.ux1) * 10;
        if (!best || score > best.score) best = Object.assign({}, l, { score, T, nSub });
      }
    }
    if (!best) return debug ? { found: false, debug: dbg } : { found: false };
    const r = {
      found: true,
      x: best.ux0, y: best.uy0, w: best.uw, h: best.uh,
      comps: best.comps, T: best.T, n: best.n, nSub: best.nSub,
    };
    if (debug) {
      r.debug = dbg;
      r.debug.best = { x: r.x, y: r.y, w: r.w, h: r.h, n: r.n, T: r.T };
    }
    return r;
  }

  // 二值掩膜膨胀 rounds 轮(3x3),纯函数,方便 Node 里验证
  function dilateMask(mask, W, H, rounds) {
    let cur = mask;
    for (let r = 0; r < rounds; r++) {
      const next = new Uint8Array(W * H);
      next.set(cur);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          if (cur[y * W + x]) continue;
          let hit = false;
          for (let dy = -1; dy <= 1 && !hit; dy++) {
            const ny = y + dy;
            if (ny < 0 || ny >= H) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const nx = x + dx;
              if (nx < 0 || nx >= W) continue;
              if (cur[ny * W + nx]) { hit = true; break; }
            }
          }
          if (hit) next[y * W + x] = 1;
        }
      }
      cur = next;
    }
    return cur;
  }

  // 生成掩膜:只取「完全落在右下角模板框内」的字形组件笔画,
  // 再外扩 2px 覆盖抗锯齿灰边。背景亮物即使与字形粘连,
  // 也因超出行高被字形过滤排除,不会进掩膜;位置固定右下角。
  // 若组件笔画太少(检测到水印但笔画没落在框内),整框填满兜底。
  function computeMask(data, W, H, box, T, comps) {
    const mask = new Uint8Array(W * H);
    let strokes = 0;
    const paintRegion = (sx, sy, ex, ey) => {
      for (let y = sy; y <= ey; y++) {
        for (let x = sx; x <= ex; x++) {
          const p = (y * W + x) * 4;
          if (data[p] >= T && data[p + 1] >= T && data[p + 2] >= T) {
            mask[y * W + x] = 1;
            strokes++;
          }
        }
      }
    };
    const inside = (c) =>
      c.minX >= box.x0 && c.maxX <= box.x1 && c.minY >= box.y0 && c.maxY <= box.y1;
    const list = (comps && comps.length) ? comps.filter(inside) : null;
    if (list && list.length) {
      for (const c of list) paintRegion(c.minX, c.minY, c.maxX, c.maxY);
    } else {
      // 没有组件信息:退回整框扫描(老行为,仅兜底)
      paintRegion(box.x0, box.y0, box.x1, box.y1);
    }
    return { mask: dilateMask(mask, W, H, 2), strokes };
  }

  // 把计算出的二值掩膜转成白底透明画布,供 HTML 端 source-in 染红
  function buildTemplateMask(data, W, H, box, T, comps) {
    const computed = computeMask(data, W, H, box, T, comps);
    let mask = computed.mask;
    if (computed.strokes < Math.max(16, box.w * box.h * 0.002)) {
      // 兜底:模板框内没找到足够的高亮笔画,直接整框涂白(界面会染红)
      mask = new Uint8Array(W * H);
      for (let y = box.y0; y <= box.y1; y++) {
        for (let x = box.x0; x <= box.x1; x++) mask[y * W + x] = 1;
      }
    }
    const mc = document.createElement("canvas");
    mc.width = W; mc.height = H;
    const mctx = mc.getContext("2d");
    const img = mctx.createImageData(W, H);
    for (let i = 0; i < W * H; i++) {
      const v = mask[i] ? 255 : 0;
      img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = v;
    }
    mctx.putImageData(img, 0, 0);
    return mc;
  }

  function detect(canvas, opts) {
    const debug = !!(opts && opts.debug);
    try {
      const W = canvas.width, H = canvas.height;
      if (!W || !H) return debug ? { found: false, debug: {} } : { found: false };
      const k = Math.min(1, MAX_ANALYZE / Math.max(W, H));
      const aw = Math.max(16, Math.round(W * k));
      const ah = Math.max(16, Math.round(H * k));
      const ac = document.createElement("canvas");
      ac.width = aw; ac.height = ah;
      const actx = ac.getContext("2d", { willReadFrequently: true });
      actx.drawImage(canvas, 0, 0, aw, ah);
      const data = actx.getImageData(0, 0, aw, ah).data;
      const r = detectFromPixels(data, aw, ah, opts);
      if (!r.found) return r;
      // 红框不取识别出的字形坐标(会漂移),改用右下角固定模板;
      // 识别结果只负责「有没有水印」。
      const box = templateBox(aw, ah);
      const out = {
        found: true,
        x: box.x0 / k, y: box.y0 / k, w: box.w / k, h: box.h / k,
        mask: buildTemplateMask(data, aw, ah, box, r.T, r.comps),
      };
      if (debug) out.debug = r.debug;
      return out;
    } catch (err) {
      console.warn("Doubao detect failed:", err);
      return debug ? { found: false, debug: {} } : { found: false };
    }
  }

  function detectFromFile(file) {
    return new Promise((resolve) => {
      try {
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => {
          try {
            const c = document.createElement("canvas");
            c.width = img.naturalWidth;
            c.height = img.naturalHeight;
            c.getContext("2d").drawImage(img, 0, 0);
            resolve(detect(c));
          } catch (e) {
            resolve({ found: false });
          } finally {
            URL.revokeObjectURL(url);
          }
        };
        img.onerror = () => { URL.revokeObjectURL(url); resolve({ found: false }); };
        img.src = url;
      } catch (e) {
        resolve({ found: false });
      }
    });
  }

  const api = { detect, detectFromFile, detectFromPixels, findLines, templateBox, computeMask, dilateMask };
  if (typeof window !== "undefined") window.Doubao = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
