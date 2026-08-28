"use strict";
// Minimal static server for local testing of the PWA (no dependencies).
// Usage:  node serve.js   ->  open http://localhost:8080/mobile.html
const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = process.env.PORT || 8080;
const ROOT = __dirname;
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".webmanifest": "application/manifest+json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".svg": "image/svg+xml",
  ".css": "text/css; charset=utf-8",
  ".ico": "image/x-icon",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
};

const server = http.createServer((req, res) => {
  let urlPath = decodeURIComponent(req.url.split("?")[0]);
  if (urlPath === "/") urlPath = "/mobile.html";
  const filePath = path.join(ROOT, path.normalize(urlPath).replace(/^(\.\.[/\\])+/, ""));
  if (!filePath.startsWith(ROOT)) { res.writeHead(403); res.end("Forbidden"); return; }

  fs.readFile(filePath, (err, data) => {
    if (err) { res.writeHead(404); res.end("Not found"); return; }
    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, {
      "Content-Type": TYPES[ext] || "application/octet-stream",
      "Cache-Control": "no-cache",
    });
    res.end(data);
  });
});

server.listen(PORT, () => {
  const nets = require("os").networkInterfaces();
  const ips = new Set();
  for (const name of Object.keys(nets))
    for (const n of nets[name])
      // 过滤回环与 169.254.x.x 这类连不上的 link-local 地址
      if (n.family === "IPv4" && !n.internal && !n.address.startsWith("169.254.")) ips.add(n.address);
  const list = [...ips].sort((a, b) => {
    const priv = (s) => /^(192\.168\.|10\.|172\.(1[6-9]|2\d|3[01])\.)/.test(s) ? 0 : 1;
    return priv(a) - priv(b) || a.localeCompare(b, undefined, { numeric: true });
  });
  console.log(`本地服务已启动：`);
  console.log(`  - 手机和电脑需连同一个 WiFi`);
  console.log(`  - 手机浏览器输入下面的"手机访问"地址即可查看`);
  console.log(`  - 手机打不开时，请允许 Windows 防火墙放行 Node.js`);
  console.log(`  - 停止服务器：按 Ctrl+C，或直接关闭窗口`);
  console.log(`  电脑访问:  http://localhost:${PORT}/mobile.html`);
  if (list.length === 0) {
    console.log(`  未找到局域网地址，请确认电脑已连接 WiFi。`);
  } else {
    list.forEach((ip) => console.log(`  手机访问:  http://${ip}:${PORT}/mobile.html   (需连同一 WiFi)`));
  }
  console.log(`\n按 Ctrl+C 停止。`);
});

server.on("error", (err) => {
  if (err.code === "EADDRINUSE") {
    console.error(`\n端口 ${PORT} 已被占用，可能是上一次的服务器还没关。`);
    console.error(`解决办法：`);
    console.error(`  1) 找到之前打开的命令行窗口，按 Ctrl+C 关掉它；`);
    console.error(`  2) 或者换一个端口再启动：set PORT=8090 && node serve.js`);
    process.exit(1);
  }
  throw err;
});
