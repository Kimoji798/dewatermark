"use strict";
/*
 * 智能抠除(自动识别物体边界)—— 本地显著性分割,不上传图片。
 * ------------------------------------------------------------------
 * 流程:用户在图片上框出一个小的范围(矩形/圆形/套索),本模块对该范围
 * 跑一次 U2-Net 小型显著性分割模型(u2netp,约 4.4MB,已本地化到 vendor/),
 * 得到前景概率图,再提取连通域作为候选区域。用户点选候选区域后,
 * 由页面把它转成红色涂抹掩膜,交给既有的去水印(AI/经典)算法消除。
 *
 * 模型不可用时自动回退到"梯度边缘 + 背景洪泛"的经典算法,保证功能可用。
 * U2-Net 权重 Apache-2.0 许可;ONNX 导出来自 rembg(MIT)。
 */
window.AutoCut = (function () {
  "use strict";

  const MODEL_URL = "./vendor/u2netp.onnx";
  const INPUT_SIZE = 320; // u2netp 固定输入
  const MEAN = [0.485, 0.456, 0.406]; // rembg 同款归一化
  const STD = [0.229, 0.224, 0.225];

  let ortReady = null;
  let session = null;

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.onload = resolve;
      s.onerror = () => reject(new Error("脚本加载失败: " + src));
      document.head.appendChild(s);
    });
  }

  // 与 ai-inpaint.js 相同的本地 onnxruntime 加载方式(iOS Safari 绝对路径坑)
  function ensureOrt() {
    if (ortReady) return ortReady;
    ortReady = (async () => {
      if (typeof self.ort === "undefined") {
        await loadScript(new URL("./vendor/ort/ort.min.js", self.location.href).href);
      }
      if (typeof self.ort === "undefined") throw new Error("onnxruntime 未能加载");
      self.ort.env.wasm.wasmPaths = new URL("./vendor/ort/", self.location.href).href;
      const canThread = (typeof self.crossOriginIsolated !== "undefined") && self.crossOriginIsolated
        && (typeof self.SharedArrayBuffer !== "undefined");
      self.ort.env.wasm.numThreads = canThread ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
      self.ort.env.wasm.simd = true;
      return self.ort;
    })().catch((e) => { ortReady = null; throw e; });
    return ortReady;
  }

  async function ensureSession() {
    if (session) return session;
    const ort = await ensureOrt();
    const resp = await fetch(new URL(MODEL_URL, self.location.href).href);
    if (!resp.ok) throw new Error("分割模型加载失败 HTTP " + resp.status);
    const buf = await resp.arrayBuffer();
    session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
    return session;
  }

  // ---------- AI 分割:u2netp 前向,输出缩放到裁剪图尺寸的概率图 ----------
  async function aiProbability(crop) {
    const sess = await ensureSession();
    const ort = await ensureOrt();
    const IN = INPUT_SIZE;
    const cv = document.createElement("canvas");
    cv.width = IN; cv.height = IN;
    const cx = cv.getContext("2d", { willReadFrequently: true });
    cx.drawImage(crop, 0, 0, IN, IN);
    const img = cx.getImageData(0, 0, IN, IN).data;
    const data = new Float32Array(3 * IN * IN);
    for (let p = 0, i = 0; p < IN * IN; p++, i += 4) {
      data[p] = ((img[i] / 255) - MEAN[0]) / STD[0];
      data[IN * IN + p] = ((img[i + 1] / 255) - MEAN[1]) / STD[1];
      data[2 * IN * IN + p] = ((img[i + 2] / 255) - MEAN[2]) / STD[2];
    }
    const inputName = sess.inputNames[0];
    const outputName = sess.outputNames[0];
    const feeds = { [inputName]: new ort.Tensor("float32", data, [1, 3, IN, IN]) };
    const out = await sess.run(feeds);
    const t = out[outputName];
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < t.data.length; i++) {
      const v = t.data[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    // 自适应:如果输出明显超出 [0,1],说明模型没内嵌 sigmoid,这里补一个
    const needSigmoid = mx > 1.5 || mn < -1.5;
    // 双线性采样回裁剪图尺寸
    const W = crop.width, H = crop.height;
    const map = new Float32Array(W * H);
    const fx = IN / W, fy = IN / H;
    for (let y = 0; y < H; y++) {
      const sy = Math.min(IN - 1, y * fy);
      const y0 = Math.floor(sy), y1 = Math.min(IN - 1, y0 + 1), wy = sy - y0;
      const r0 = y0 * IN, r1 = y1 * IN;
      for (let x = 0; x < W; x++) {
        const sx = Math.min(IN - 1, x * fx);
        const x0 = Math.floor(sx), x1 = Math.min(IN - 1, x0 + 1), wx = sx - x0;
        const a = t.data[r0 + x0], b = t.data[r0 + x1], c = t.data[r1 + x0], d = t.data[r1 + x1];
        let v = (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy;
        if (needSigmoid) v = 1 / (1 + Math.exp(-v));
        map[y * W + x] = v;
      }
    }
    return map;
  }

  // ---------- 经典回退:梯度边缘 + 背景洪泛 ----------
  function classicalBinary(crop, regionMask) {
    const W = crop.width, H = crop.height;
    const img = crop.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, W, H).data;
    const gray = new Float32Array(W * H);
    for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
      gray[i] = 0.299 * img[p] + 0.587 * img[p + 1] + 0.114 * img[p + 2];
    }
    const g = new Float32Array(W * H);
    let sum = 0, n = 0;
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const i = y * W + x;
        const gx = gray[y * W + x + 1] - gray[y * W + x - 1];
        const gy = gray[(y + 1) * W + x] - gray[(y - 1) * W + x];
        const v = Math.sqrt(gx * gx + gy * gy);
        g[i] = v; sum += v; n++;
      }
    }
    const mean = sum / Math.max(1, n);
    let vsum = 0;
    for (let i = 0; i < g.length; i++) { const d = g[i] - mean; vsum += d * d; }
    const std = Math.sqrt(vsum / Math.max(1, n));
    const thr = mean + 0.8 * std;
    let edge = new Uint8Array(W * H);
    for (let i = 0; i < g.length; i++) edge[i] = g[i] > thr ? 1 : 0;
    // 膨胀两次闭合边缘断口
    edge = morph(edge, W, H, true);
    edge = morph(edge, W, H, true);

    const bg = new Uint8Array(W * H);
    const q = [];
    if (regionMask) {
      // 形状外直接当背景;与形状外相邻的非边缘点作为洪泛种子
      for (let i = 0; i < bg.length; i++) if (!regionMask[i]) bg[i] = 1;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const i = y * W + x;
          if (!regionMask[i] || edge[i] || bg[i]) continue;
          let border = false;
          if (x > 0 && !regionMask[i - 1]) border = true;
          else if (x < W - 1 && !regionMask[i + 1]) border = true;
          else if (y > 0 && !regionMask[i - W]) border = true;
          else if (y < H - 1 && !regionMask[i + W]) border = true;
          if (border) { bg[i] = 1; q.push(x, y); }
        }
      }
    } else {
      for (let x = 0; x < W; x++) {
        if (!edge[x]) { bg[x] = 1; q.push(x, 0); }
        const b = (H - 1) * W + x;
        if (!edge[b]) { bg[b] = 1; q.push(x, H - 1); }
      }
      for (let y = 0; y < H; y++) {
        const l = y * W, r = y * W + W - 1;
        if (!edge[l]) { bg[l] = 1; q.push(0, y); }
        if (!edge[r]) { bg[r] = 1; q.push(W - 1, y); }
      }
    }
    let head = 0;
    while (head < q.length) {
      const x = q[head++], y = q[head++];
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
          const i = ny * W + nx;
          if (!edge[i] && !bg[i] && (!regionMask || regionMask[i])) { bg[i] = 1; q.push(nx, ny); }
        }
      }
    }
    const bin = new Uint8Array(W * H);
    for (let i = 0; i < bin.length; i++) {
      bin[i] = (!edge[i] && !bg[i] && (!regionMask || regionMask[i])) ? 1 : 0;
    }
    return bin;
  }

  // 二值形态学(膨胀/腐蚀),3x3
  function morph(bin, W, H, isDilate) {
    const out = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        if (isDilate && bin[i]) { out[i] = 1; continue; }
        if (!isDilate && !bin[i]) continue;
        let hit = false;
        for (let dy = -1; dy <= 1 && !hit; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            const nx = x + dx, ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
            const v = bin[ny * W + nx];
            if (isDilate ? v : !v) { hit = true; break; }
          }
        }
        out[i] = isDilate ? (hit ? 1 : 0) : (hit ? 0 : 1);
      }
    }
    return out;
  }

  // 8-连通域标记,返回组件(面积/包围盒)与标签图
  function componentsWithLabels(bin, W, H) {
    const labels = new Int32Array(W * H);
    const comps = [];
    const qx = new Int32Array(W * H), qy = new Int32Array(W * H);
    for (let s = 0; s < W * H; s++) {
      if (!bin[s] || labels[s]) continue;
      const id = comps.length + 1;
      let head = 0, tail = 0;
      qx[tail] = s % W; qy[tail] = (s / W) | 0; tail++;
      labels[s] = id;
      let minX = qx[0], minY = qy[0], maxX = qx[0], maxY = qy[0], area = 0;
      while (head < tail) {
        const x = qx[head], y = qy[head]; head++; area++;
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            const nx = x + dx, ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
            const ni = ny * W + nx;
            if (bin[ni] && !labels[ni]) { labels[ni] = id; qx[tail] = nx; qy[tail] = ny; tail++; }
          }
        }
      }
      comps.push({ id, minX, minY, maxX, maxY, area });
    }
    return { comps, labels };
  }

  // ---------- 对外主入口 ----------
  // sourceCanvas: 原图 canvas;rect:{x,y,w,h} 整数;regionMask: 可选 Uint8Array(w*h),1=形状内
  // 返回 { candidates:[{x,y,w,h,mask,area}], regionW, regionH, method }
  async function segment(sourceCanvas, rect, regionMask) {
    const W = Math.round(rect.w), H = Math.round(rect.h);
    const crop = document.createElement("canvas");
    crop.width = W; crop.height = H;
    crop.getContext("2d", { willReadFrequently: true })
      .drawImage(sourceCanvas, rect.x, rect.y, W, H, 0, 0, W, H);

    let bin = null, method = "ai";
    try {
      const prob = await aiProbability(crop);
      bin = new Uint8Array(W * H);
      for (let i = 0; i < bin.length; i++) bin[i] = prob[i] > 0.5 ? 1 : 0;
    } catch (err) {
      console.warn("u2netp 分割失败,回退经典算法:", err);
      method = "classic";
      bin = classicalBinary(crop, regionMask);
    }
    if (regionMask) {
      for (let i = 0; i < bin.length; i++) if (!regionMask[i]) bin[i] = 0;
    }
    const { comps, labels } = componentsWithLabels(bin, W, H);
    const minArea = Math.max(16, Math.round(W * H * 0.0005));
    const picked = comps
      .filter((c) => c.area >= minArea)
      .sort((a, b) => b.area - a.area)
      .slice(0, 40);
    const candidates = picked.map((c) => {
      const mask = new Uint8Array(W * H);
      for (let i = 0; i < W * H; i++) if (labels[i] === c.id) mask[i] = 1;
      return {
        x: c.minX, y: c.minY,
        w: c.maxX - c.minX + 1, h: c.maxY - c.minY + 1,
        mask, area: c.area,
      };
    });
    return { candidates, regionW: W, regionH: H, method };
  }

  return {
    segment,
    isLoaded: () => !!session,
  };
})();
