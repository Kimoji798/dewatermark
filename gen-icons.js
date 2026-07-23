"use strict";
// Generates PNG icons for the PWA using only Node built-ins (zlib + a tiny CRC32).
const fs = require("fs");
const zlib = require("zlib");

// ---- CRC32 ----
const crcTable = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(buf) {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < buf.length; i++) c = crcTable[(c ^ buf[i]) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, "ascii");
  const body = Buffer.concat([typeBuf, data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(body), 0);
  return Buffer.concat([len, body, crc]);
}

// ---- Draw one icon (RGBA pixel buffer) ----
function draw(size) {
  const px = Buffer.alloc(size * size * 4);
  const cx = size / 2, cy = size / 2;
  const rOuter = size * 0.33;   // white ring outer
  const rInner = size * 0.17;   // hole
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      // background vertical gradient: #4f8cff -> #3a6fd8
      const t = y / size;
      let r = Math.round(0x4f + (0x3a - 0x4f) * t);
      let g = Math.round(0x8c + (0x6f - 0x8c) * t);
      let b = Math.round(0xff + (0xd8 - 0xff) * t);
      // centered ring (the brush cursor motif)
      const d = Math.hypot(x - cx + 0.5, y - cy + 0.5);
      if (d <= rOuter && d >= rInner) {
        r = 255; g = 255; b = 255;
      }
      const i = (y * size + x) * 4;
      px[i] = r; px[i + 1] = g; px[i + 2] = b; px[i + 3] = 255;
    }
  }
  return px;
}

function encodePNG(size) {
  const raw = draw(size);
  // add filter byte (0) at the start of each scanline
  const stride = size * 4;
  const filtered = Buffer.alloc((stride + 1) * size);
  for (let y = 0; y < size; y++) {
    filtered[y * (stride + 1)] = 0;
    raw.copy(filtered, y * (stride + 1) + 1, y * stride, y * stride + stride);
  }
  const idat = zlib.deflateSync(filtered, { level: 9 });

  const sig = Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8;   // bit depth
  ihdr[9] = 6;   // color type RGBA
  ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;

  return Buffer.concat([
    sig,
    chunk("IHDR", ihdr),
    chunk("IDAT", idat),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

for (const size of [192, 512]) {
  fs.writeFileSync(`icon-${size}.png`, encodePNG(size));
  console.log(`wrote icon-${size}.png`);
}
