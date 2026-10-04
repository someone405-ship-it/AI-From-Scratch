// Basic service worker for AI From Scratch PWA
const CACHE_NAME = "ai-from-scratch-v1";

self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  // Network-first for API / dynamic content
  event.respondWith(
    fetch(event.request).catch(() => caches.match(event.request))
  );
});
