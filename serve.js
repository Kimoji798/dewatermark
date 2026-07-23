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
  ".json": "application/json; charset=utf-8",
  ".webmanifest": "application/manifest+json; charset=utf-8",
  ".png": "image/png",
  ".ico": "image/x-icon",
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
  const ips = [];
  for (const name of Object.keys(nets))
    for (const n of nets[name])
      if (n.family === "IPv4" && !n.internal) ips.push(n.address);
  console.log(`本地服务已启动：`);
  console.log(`  电脑访问:  http://localhost:${PORT}/mobile.html`);
  ips.forEach((ip) => console.log(`  手机访问:  http://${ip}:${PORT}/mobile.html   (需连同一 WiFi)`));
  console.log(`\n按 Ctrl+C 停止。`);
});
