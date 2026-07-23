"use strict";
// Service worker: cache the app shell so it works fully offline.
const CACHE = "dewatermark-v9";
const ASSETS = [
  "./mobile.html",
  "./ai-inpaint.js",
  "./migan_pipeline_v2.onnx",
  "./manifest.webmanifest",
  "./icon-192.png",
  "./icon-512.png"
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// 大文件（模型、图标）走“缓存优先”——避免每次重下 28MB 模型。
const CACHE_FIRST = /\.(onnx|png|webmanifest)(\?|$)/i;

// 其余（HTML 页面、JS 代码）走“网络优先”：
// 有网就总是取最新，拿到后更新缓存；没网才回退到缓存。
// 这样发布更新后，用户正常刷新即可看到新版，无需手动清缓存
// （尤其解决 iOS Safari 缓存顽固、刷新不更新的问题）。
self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;

  if (CACHE_FIRST.test(new URL(req.url).pathname)) {
    // 缓存优先
    e.respondWith(
      caches.match(req).then((cached) => {
        if (cached) return cached;
        return fetch(req).then((res) => {
          if (res && res.status === 200 && res.type === "basic") {
            const copy = res.clone();
            caches.open(CACHE).then((c) => c.put(req, copy));
          }
          return res;
        });
      })
    );
    return;
  }

  // 网络优先
  e.respondWith(
    fetch(req).then((res) => {
      if (res && res.status === 200 && res.type === "basic") {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(req, copy));
      }
      return res;
    }).catch(() => caches.match(req))
  );
});
