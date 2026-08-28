"use strict";
// 豆包AI水印检测：在图片右下角寻找白色的「豆包AI生成」文字行。
// 判据(基于样图 doubaoAI.jpg 的连通域分析)：
//   1. 白色像素(R/G/B >= 200)的连通域；
//   2. 位于图片右下角区域(x >= 50% 宽, y >= 50% 高)；
//   3. 连通域 bbox 贴住右边缘(x+w >= W-2)；
//   4. 面积合理(>= 阈值)、填充率在文字笔画范围(2% ~ 70%)、尺寸比例合理；
//   5. 文字笔画常被 AA 分成多个小连通域：从贴右边缘的锚点向左收拢
//      「同一行」的相邻笔画域(水平间距小、高度相近、垂直中心对齐)。
(function () {
  const T = 200;            // 白色阈值
  const MAX_ANALYZE = 800;  // 分析用缩小图最大边(快且抗噪)

  function detect(canvas, opts) {
    const debug = !!(opts && opts.debug);
    const dbg = { comps: [], rights: [] };
    try {
      const W = canvas.width, H = canvas.height;
      if (!W || !H) return debug ? { found: false, debug: dbg } : { found: false };
      const k = Math.min(1, MAX_ANALYZE / Math.max(W, H));
      const aw = Math.max(16, Math.round(W * k));
      const ah = Math.max(16, Math.round(H * k));
      const ac = document.createElement("canvas");
      ac.width = aw; ac.height = ah;
      const actx = ac.getContext("2d", { willReadFrequently: true });
      actx.drawImage(canvas, 0, 0, aw, ah);
      const data = actx.getImageData(0, 0, aw, ah).data;

      // 搜索窗口:右下角
      const x0 = Math.floor(aw * 0.5);
      const y0 = Math.floor(ah * 0.5);
      const visited = new Uint8Array(aw * ah);
      const stack = new Int32Array(aw * ah);
      const comps = [];
      for (let sy = y0; sy < ah; sy++) {
        for (let sx = x0; sx < aw; sx++) {
          const si = sy * aw + sx;
          if (visited[si]) continue;
          const p = si * 4;
          if (data[p] < T || data[p + 1] < T || data[p + 2] < T) continue;
          // 8-邻域 BFS
          let sp = 0;
          stack[sp++] = si;
          visited[si] = 1;
          let minX = sx, maxX = sx, minY = sy, maxY = sy, area = 0;
          while (sp > 0) {
            const ci = stack[--sp];
            const cx = ci % aw, cy = (ci / aw) | 0;
            if (cx < minX) minX = cx;
            if (cx > maxX) maxX = cx;
            if (cy < minY) minY = cy;
            if (cy > maxY) maxY = cy;
            area++;
            for (let dy = -1; dy <= 1; dy++) {
              const ny = cy + dy;
              if (ny < y0 || ny >= ah) continue;
              for (let dx = -1; dx <= 1; dx++) {
                if (dx === 0 && dy === 0) continue;
                const nx = cx + dx;
                if (nx < x0 || nx >= aw) continue;
                const ni = ny * aw + nx;
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
      if (!comps.length) return debug ? { found: false, debug: dbg } : { found: false };

      const minArea = Math.max(120, aw * ah * 0.0003);
      const valid = (c) =>
        c.w >= 6 && c.h >= 6 &&
        c.w <= aw * 0.55 && c.h <= ah * 0.45 &&
        c.area >= minArea &&
        c.area / (c.w * c.h) >= 0.02 && c.area / (c.w * c.h) <= 0.7;

      const touchesRight = (c) => c.maxX >= aw - 2;
      const rights = comps.filter((c) => touchesRight(c) && valid(c));
      if (debug) {
        dbg.comps = comps.map((c) => ({
          x: c.minX / k, y: c.minY / k, w: c.w / k, h: c.h / k,
          area: c.area, fill: +(c.area / (c.w * c.h)).toFixed(3),
          right: c.maxX >= aw - 2,
        })).sort((a, b) => b.area - a.area).slice(0, 20);
        dbg.rights = rights.map((c) => ({ x: c.minX / k, y: c.minY / k, w: c.w / k, h: c.h / k, area: c.area }));
      }
      if (!rights.length) return debug ? { found: false, debug: dbg } : { found: false };
      // 锚点:贴右边缘里面积最大(最像文字主体的)的域
      rights.sort((a, b) => b.area - a.area);
      const anchor = rights[0];
      let ux0 = anchor.minX, ux1 = anchor.maxX, uy0 = anchor.minY, uy1 = anchor.maxY;
      let uarea = anchor.area;
      let changed = true;
      while (changed) {
        changed = false;
        const uw = ux1 - ux0 + 1, uh = uy1 - uy0 + 1;
        const ucy = (uy0 + uy1) / 2;
        for (const c of comps) {
          if (!valid(c) || rights.includes(c)) continue;
          if (c.maxX > ux0 || c.minX >= ux0) continue;             // 只在当前行左侧
          const gap = ux0 - c.maxX;
          if (gap <= 0 || gap > uh * 0.75) continue;               // 间距不超过约一个字符
          const ccy = (c.minY + c.maxY) / 2;
          if (Math.abs(ccy - ucy) > uh * 0.35) continue;           // 垂直中心要对齐
          if (c.h > uh * 1.4 || c.h < uh * 0.4) continue;          // 高度相近
          ux0 = Math.min(ux0, c.minX);
          uy0 = Math.min(uy0, c.minY);
          uy1 = Math.max(uy1, c.maxY);
          uarea += c.area;
          changed = true;
        }
      }
      const uw = ux1 - ux0 + 1, uh = uy1 - uy0 + 1;
      const fill = uarea / (uw * uh);
      if (debug) {
        dbg.union = { x: ux0 / k, y: uy0 / k, w: uw / k, h: uh / k, fill: +fill.toFixed(3), anchorArea: anchor.area };
      }
      if (fill < 0.02 || fill > 0.7) return debug ? { found: false, debug: dbg } : { found: false };
      if (uh > ah * 0.45 || uw > aw * 0.6) return debug ? { found: false, debug: dbg } : { found: false };
      // 必须落在右下角(文字行下缘接近图片底部)
      if (uy0 < ah * 0.5 || uy1 < ah * 0.6) return debug ? { found: false, debug: dbg } : { found: false };
      const r = {
        found: true,
        x: ux0 / k,
        y: uy0 / k,
        w: uw / k,
        h: uh / k,
      };
      return debug ? Object.assign(r, { debug: dbg }) : r;
    } catch (err) {
      console.warn("Doubao detect failed:", err);
      return debug ? { found: false, debug: dbg } : { found: false };
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

  window.Doubao = { detect, detectFromFile };
})();
