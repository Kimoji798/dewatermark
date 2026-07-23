"use strict";
/*
 * AI 去水印模块（LaMa inpainting，浏览器端运行）
 * ------------------------------------------------------------------
 * 使用 onnxruntime-web 加载 LaMa ONNX 模型，在本地对涂抹区域做内容感知修复，
 * 效果接近 Photoshop 的“内容识别填充”/ 网上 AI 去水印工具。
 *
 * 全部在浏览器本地运行，图片不会上传。首次使用需联网下载模型（之后可被缓存）。
 *
 * ⚠️ 模型与运行时的地址集中在下面 CONFIG。若某个 CDN 地址失效，
 *    只需替换对应 URL 即可，无需改动其它代码。
 */

const AIInpaint = (function () {

  // ============ 可配置区（地址失效时改这里） ============
  const CONFIG = {
    // onnxruntime-web 运行时（含 wasm）。用 jsDelivr CDN（有 CORS + 国内节点）。
    ortScript: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js",
    wasmPaths: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/",

    // MI-GAN inpainting 模型（migan_pipeline_v2.onnx，约 28MB，内置前后处理）。
    // 已放进本仓库，同源加载 —— 不再有 CORS/跨域问题，国内可直连 GitHub Pages。
    // 若想换源，把下面改成完整 URL 列表即可（会按顺序尝试）。
    modelFull: ["./migan_pipeline_v2.onnx"],
    modelMobile: ["./migan_pipeline_v2.onnx"],

    // 处理时的最大边长（原生分辨率太大手机会 OOM，超过则等比缩小，
    // 最终只把掩膜区域贴回原图，非掩膜区域仍是原始清晰像素）。
    maxSide: 1024,
    // 尺寸需为该值的整数倍（MI-GAN 下采样要求，8 通常安全）
    multiple: 8,

    // 张量名（从模型文件解析所得）
    inputImageName: "image",   // uint8 [1,3,H,W], RGB, 0..255
    inputMaskName: "mask",     // uint8 [1,1,H,W]
    // 掩膜极性：涂抹处（待去除）应填哪个值。
    // MI-GAN 约定通常为“已知区=255，待修复=0”；若结果反了（把该留的抹了），
    // 把此值从 0 改成 255（并相应把 keep 值改成 0）即可一键翻转。
    maskHoleValue: 0,          // 待去除区域的值
    maskKeepValue: 255,        // 保留区域的值
  };
  // ====================================================

  let ortReady = null;   // Promise，加载 ort 运行时
  let session = null;    // ONNX 推理会话
  let loadedModelUrl = null;

  function isMobile() {
    return /Android|iPhone|iPad|iPod|Mobile|Windows Phone/i.test(navigator.userAgent);
  }

  // 动态加载一个 <script>
  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.onload = resolve;
      s.onerror = () => reject(new Error("脚本加载失败: " + src));
      document.head.appendChild(s);
    });
  }

  // 确保 onnxruntime-web 已就绪
  function ensureOrt() {
    if (ortReady) return ortReady;
    ortReady = (async () => {
      if (typeof self.ort === "undefined") {
        await loadScript(CONFIG.ortScript);
      }
      if (typeof self.ort === "undefined") throw new Error("onnxruntime 未能加载");
      self.ort.env.wasm.wasmPaths = CONFIG.wasmPaths;
      // 关键：GitHub Pages 等静态托管无法设置 COOP/COEP 响应头，
      // 因此 SharedArrayBuffer 不可用，多线程 wasm 会初始化失败。
      // 只有页面确实处于跨域隔离状态时才启用多线程，否则强制单线程。
      const canThread = (typeof self.crossOriginIsolated !== "undefined") && self.crossOriginIsolated
        && (typeof self.SharedArrayBuffer !== "undefined");
      self.ort.env.wasm.numThreads = canThread ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
      self.ort.env.wasm.simd = true;
      return self.ort;
    })().catch((e) => { ortReady = null; throw e; });
    return ortReady;
  }

  // 加载模型会话（带进度回调）
  async function ensureSession(onProgress) {
    const urls = isMobile() ? CONFIG.modelMobile : CONFIG.modelFull;
    const urlList = Array.isArray(urls) ? urls : [urls];
    const key = urlList.join("|");
    if (session && loadedModelUrl === key) return session;

    const ort = await ensureOrt();

    // 依次尝试各下载源，任一成功即用（带进度，结果放进缓存以便离线复用）
    let buf = null, lastErr = null;
    for (const u of urlList) {
      try {
        buf = await fetchModel(u, onProgress);
        break;
      } catch (e) {
        lastErr = e;
        console.warn("模型源失败，尝试下一个：", u, e && e.message);
      }
    }
    if (!buf) throw new Error("所有模型下载源都失败了：" + (lastErr && lastErr.message ? lastErr.message : lastErr));

    session = await ort.InferenceSession.create(buf, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    loadedModelUrl = key;
    return session;
  }

  // 下载模型，优先走 Cache Storage（装了 PWA 后可离线）
  async function fetchModel(url, onProgress) {
    try {
      if ("caches" in self) {
        const cache = await caches.open("ai-models-v1");
        const hit = await cache.match(url);
        if (hit) { onProgress && onProgress(1); return await hit.arrayBuffer(); }
      }
    } catch (_) { /* 缓存不可用则直接下载 */ }

    const res = await fetch(url);
    if (!res.ok) throw new Error("模型下载失败: HTTP " + res.status);

    const total = +res.headers.get("Content-Length") || 0;
    if (!res.body || !total) {
      const ab = await res.arrayBuffer();
      onProgress && onProgress(1);
      await putCache(url, ab);
      return ab;
    }
    // 流式读取以显示进度
    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress && onProgress(received / total);
    }
    const out = new Uint8Array(received);
    let pos = 0;
    for (const c of chunks) { out.set(c, pos); pos += c.length; }
    await putCache(url, out.buffer);
    return out.buffer;
  }

  async function putCache(url, arrayBuffer) {
    try {
      if ("caches" in self) {
        const cache = await caches.open("ai-models-v1");
        await cache.put(url, new Response(arrayBuffer));
      }
    } catch (_) { /* 存不下就算了，不影响本次使用 */ }
  }

  // ---------------------------------------------------------------
  // 核心推理（MI-GAN pipeline，uint8 输入/输出，原生分辨率、动态尺寸）：
  //   image: uint8 [1,3,H,W] RGB 0..255（NCHW）
  //   mask:  uint8 [1,1,H,W]（保留区/待修复区，极性见 CONFIG）
  //   result:uint8 [1,3,H,W]
  //
  // 做法：
  //   1. 把原图等比缩放到 <=maxSide 且宽高为 multiple 的整数倍（模型下采样要求）
  //   2. 掩膜同样缩放（最近邻，保持硬边）
  //   3. 推理得到同尺寸结果
  //   4. 只把“掩膜命中的区域”缩放回原图对应位置贴上，
  //      非掩膜区域保持原始像素 → 图片其它部分 100% 清晰无损
  // ---------------------------------------------------------------
  async function inpaint(srcCanvas, maskCanvas, onProgress) {
    const ort = await ensureOrt();
    const sess = await ensureSession(onProgress);
    const W = srcCanvas.width, H = srcCanvas.height;
    const mult = CONFIG.multiple, maxSide = CONFIG.maxSide;

    // 处理尺寸：等比缩放到 <=maxSide，再各自向下取整到 mult 的倍数
    let scale = Math.min(1, maxSide / Math.max(W, H));
    let pw = Math.max(mult, Math.round(W * scale / mult) * mult);
    let ph = Math.max(mult, Math.round(H * scale / mult) * mult);

    // 源图缩放到 pw x ph
    const tmp = document.createElement("canvas");
    tmp.width = pw; tmp.height = ph;
    const tctx = tmp.getContext("2d", { willReadFrequently: true });
    tctx.drawImage(srcCanvas, 0, 0, W, H, 0, 0, pw, ph);
    const imgData = tctx.getImageData(0, 0, pw, ph).data;

    // 掩膜缩放到 pw x ph（关闭平滑，保持硬边）
    const mtmp = document.createElement("canvas");
    mtmp.width = pw; mtmp.height = ph;
    const mctx = mtmp.getContext("2d", { willReadFrequently: true });
    mctx.imageSmoothingEnabled = false;
    mctx.drawImage(maskCanvas, 0, 0, W, H, 0, 0, pw, ph);
    const mData = mctx.getImageData(0, 0, pw, ph).data;

    // 组装 uint8 NCHW 张量
    const plane = pw * ph;
    const imgArr = new Uint8Array(3 * plane);
    const maskArr = new Uint8Array(plane);
    const hole = CONFIG.maskHoleValue & 255, keep = CONFIG.maskKeepValue & 255;
    for (let i = 0; i < plane; i++) {
      imgArr[i] = imgData[i * 4];               // R
      imgArr[plane + i] = imgData[i * 4 + 1];   // G
      imgArr[2 * plane + i] = imgData[i * 4 + 2]; // B
      const isHole = (mData[i * 4] > 80 && mData[i * 4 + 3] > 40);
      maskArr[i] = isHole ? hole : keep;
    }

    const feeds = {};
    feeds[CONFIG.inputImageName] = new ort.Tensor("uint8", imgArr, [1, 3, ph, pw]);
    feeds[CONFIG.inputMaskName] = new ort.Tensor("uint8", maskArr, [1, 1, ph, pw]);

    const results = await sess.run(feeds);
    const outName = sess.outputNames && sess.outputNames.length ? sess.outputNames[0]
      : Object.keys(results)[0];
    const od = results[outName].data; // uint8 [1,3,ph,pw]

    // 结果写进处理尺寸的 canvas
    const rc = document.createElement("canvas");
    rc.width = pw; rc.height = ph;
    const rctx = rc.getContext("2d", { willReadFrequently: true });
    const resImg = rctx.createImageData(pw, ph);
    for (let i = 0; i < plane; i++) {
      resImg.data[i * 4]     = od[i];
      resImg.data[i * 4 + 1] = od[plane + i];
      resImg.data[i * 4 + 2] = od[2 * plane + i];
      resImg.data[i * 4 + 3] = 255;
    }
    rctx.putImageData(resImg, 0, 0);

    // 合成：原图打底，仅掩膜区域贴回 AI 结果（从 pw×ph 双线性采样回 W×H）
    const fc = document.createElement("canvas");
    fc.width = W; fc.height = H;
    const fctx = fc.getContext("2d", { willReadFrequently: true });
    fctx.drawImage(srcCanvas, 0, 0);
    const finalData = fctx.getImageData(0, 0, W, H);
    const fd = finalData.data;
    const resData = rctx.getImageData(0, 0, pw, ph).data;
    const mfull = maskCanvas.getContext("2d", { willReadFrequently: true })
      .getImageData(0, 0, W, H).data;
    const sxK = pw / W, syK = ph / H;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = y * W + x;
        if (!(mfull[idx * 4] > 80 && mfull[idx * 4 + 3] > 40)) continue;
        const c = sampleBilinear(resData, pw, ph, x * sxK, y * syK);
        fd[idx * 4] = c[0]; fd[idx * 4 + 1] = c[1]; fd[idx * 4 + 2] = c[2]; fd[idx * 4 + 3] = 255;
      }
    }
    return finalData;
  }

  function clamp255(v) { return v < 0 ? 0 : (v > 255 ? 255 : (v + 0.5) | 0); }

  function sampleBilinear(data, W, H, x, y) {
    let x0 = Math.floor(x), y0 = Math.floor(y);
    const tx = x - x0, ty = y - y0;
    let x1 = x0 + 1, y1 = y0 + 1;
    x0 = Math.max(0, Math.min(W - 1, x0)); x1 = Math.max(0, Math.min(W - 1, x1));
    y0 = Math.max(0, Math.min(H - 1, y0)); y1 = Math.max(0, Math.min(H - 1, y1));
    const out = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      const a = data[(y0 * W + x0) * 4 + c], b = data[(y0 * W + x1) * 4 + c];
      const cc = data[(y1 * W + x0) * 4 + c], dd = data[(y1 * W + x1) * 4 + c];
      const top = a + (b - a) * tx, bot = cc + (dd - cc) * tx;
      out[c] = top + (bot - top) * ty;
    }
    return out;
  }

  return { inpaint, ensureSession, ensureOrt, isMobile, CONFIG };
})();

