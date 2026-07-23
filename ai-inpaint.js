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
    // onnxruntime-web 运行时（含 wasm）。用 jsDelivr CDN。
    ortScript: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js",
    wasmPaths: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/",

    // LaMa ONNX 模型。固定 512x512 输入。约 200MB。
    // 多个下载源，按顺序尝试，任一成功即可（国内优先用 hf-mirror 镜像，
    // 原站 huggingface.co 在国内常超时）。地址失效时增删这里即可。
    modelFull: [
      "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
      "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
    ],
    // 手机同源列表（暂与桌面相同；若找到稳定量化小模型，替换这里）
    modelMobile: [
      "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
      "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
    ],

    // 模型固定输入尺寸
    size: 512,

    // 模型的输入/输出张量名与数值约定（Carve/LaMa-ONNX 规格）
    inputImageName: "image",   // float32 [1,3,512,512], RGB, 0..1
    inputMaskName: "mask",     // float32 [1,1,512,512], 值 0 或 1，1=待修复
    // 输出名在加载后自动读取（不同导出可能叫 output / out 等）
    outputRange255: true,      // Carve 版输出是 0..255（不是 0..1）
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
  // 核心推理：给定源图 canvas 和掩膜 canvas（红色=待去除），
  // 返回一个填好水印区域的 ImageData（与原图同尺寸）。
  //
  // LaMa 固定 512x512，所以对任意尺寸的图，做法是：
  //   1. 把整图缩放进 512x512（保持比例，pad 到方形）
  //   2. 掩膜同样缩放
  //   3. 推理得到 512x512 修复结果
  //   4. 只把“掩膜命中的区域”从结果缩放回原图对应位置（未涂抹区域保持原始像素，避免全图被重采样变糊）
  //
  // 说明：整图塞进 512 会损失分辨率，但 LaMa 对水印这类局部修复足够；
  //       只回贴掩膜区域可保证图片其它部分 100% 清晰无损。
  // ---------------------------------------------------------------
  async function inpaint(srcCanvas, maskCanvas, onProgress) {
    const ort = await ensureOrt();
    const sess = await ensureSession(onProgress);
    const S = CONFIG.size;
    const W = srcCanvas.width, H = srcCanvas.height;

    // 1) 用 letterbox 把原图铺进 SxS（居中，短边留边）
    const scale = Math.min(S / W, S / H);
    const dw = Math.round(W * scale), dh = Math.round(H * scale);
    const ox = ((S - dw) / 2) | 0, oy = ((S - dh) / 2) | 0;

    const tmp = document.createElement("canvas");
    tmp.width = S; tmp.height = S;
    const tctx = tmp.getContext("2d", { willReadFrequently: true });
    tctx.fillStyle = "#000"; tctx.fillRect(0, 0, S, S);
    tctx.drawImage(srcCanvas, 0, 0, W, H, ox, oy, dw, dh);
    const imgData = tctx.getImageData(0, 0, S, S).data;

    // 掩膜同样铺进 SxS
    const mtmp = document.createElement("canvas");
    mtmp.width = S; mtmp.height = S;
    const mctx = mtmp.getContext("2d", { willReadFrequently: true });
    mctx.fillStyle = "#000"; mctx.fillRect(0, 0, S, S);
    mctx.drawImage(maskCanvas, 0, 0, W, H, ox, oy, dw, dh);
    const mData = mctx.getImageData(0, 0, S, S).data;

    // 2) 组装输入张量
    // image: float32 [1,3,S,S], RGB, 0..1, planar (NCHW)
    const imgArr = new Float32Array(3 * S * S);
    const maskArr = new Float32Array(1 * S * S);
    const plane = S * S;
    for (let i = 0; i < plane; i++) {
      imgArr[i] = imgData[i * 4] / 255;                 // R
      imgArr[plane + i] = imgData[i * 4 + 1] / 255;     // G
      imgArr[2 * plane + i] = imgData[i * 4 + 2] / 255; // B
      // 掩膜：红色通道（涂抹用 #ff3b3b）或 alpha 高 => 1（待修复）
      const isMask = (mData[i * 4] > 80 && mData[i * 4 + 3] > 40) ? 1 : 0;
      maskArr[i] = isMask;
    }

    const feeds = {};
    feeds[CONFIG.inputImageName] = new ort.Tensor("float32", imgArr, [1, 3, S, S]);
    feeds[CONFIG.inputMaskName] = new ort.Tensor("float32", maskArr, [1, 1, S, S]);

    // 3) 推理
    const results = await sess.run(feeds);
    const outName = sess.outputNames && sess.outputNames.length ? sess.outputNames[0]
      : Object.keys(results)[0];
    const out = results[outName];
    const od = out.data;
    // 输出形状 [1,3,S,S]，值域 0..255（Carve 版）或 0..1
    const div = CONFIG.outputRange255 ? 1 : 255;

    // 4) 把 512 结果画回一个临时 canvas，再只将掩膜区域缩放回原图
    const resCanvas = document.createElement("canvas");
    resCanvas.width = S; resCanvas.height = S;
    const rctx = resCanvas.getContext("2d");
    const resImg = rctx.createImageData(S, S);
    const mul = CONFIG.outputRange255 ? 1 : 255; // 输出 0..1 时乘 255
    for (let i = 0; i < plane; i++) {
      resImg.data[i * 4]     = clamp255(od[i] * mul);
      resImg.data[i * 4 + 1] = clamp255(od[plane + i] * mul);
      resImg.data[i * 4 + 2] = clamp255(od[2 * plane + i] * mul);
      resImg.data[i * 4 + 3] = 255;
    }
    rctx.putImageData(resImg, 0, 0);

    // 组合：以原图为底，仅在掩膜区域贴上放大的 AI 结果
    const finalCanvas = document.createElement("canvas");
    finalCanvas.width = W; finalCanvas.height = H;
    const fctx = finalCanvas.getContext("2d", { willReadFrequently: true });
    fctx.drawImage(srcCanvas, 0, 0);

    // 逐像素合成：仅把掩膜命中的像素替换为 AI 结果（双线性从 512 采样回原图），
    // 其余像素保持原始清晰度不动。
    const finalData = fctx.getImageData(0, 0, W, H);
    const fd = finalData.data;
    const resData = rctx.getImageData(0, 0, S, S).data;
    // 对原图每个像素，若其在掩膜内，则从 AI 结果对应位置双线性取样
    const mfull = maskCanvas.getContext("2d", { willReadFrequently: true })
      .getImageData(0, 0, W, H).data;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = (y * W + x);
        if (!(mfull[idx * 4] > 80 && mfull[idx * 4 + 3] > 40)) continue;
        // 原图坐标 -> 512 坐标
        const sx = ox + x * scale, sy = oy + y * scale;
        const c = sampleBilinear(resData, S, S, sx, sy);
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

