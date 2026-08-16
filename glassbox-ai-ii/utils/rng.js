export function hashSeed(seed) {
  const text = String(seed);
  let h = 2166136261 >>> 0;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export class SeededRandom {
  constructor(seed = 42) {
    this.state = hashSeed(seed) || 0x6d2b79f5;
    this.spareNormal = null;
  }

  next() {
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  uniform(min = 0, max = 1) {
    return min + (max - min) * this.next();
  }

  integer(maxExclusive) {
    return Math.floor(this.next() * maxExclusive);
  }

  normal(mean = 0, std = 1) {
    if (this.spareNormal !== null) {
      const value = this.spareNormal;
      this.spareNormal = null;
      return mean + std * value;
    }
    const u = Math.max(this.next(), Number.EPSILON);
    const v = this.next();
    const mag = Math.sqrt(-2 * Math.log(u));
    const angle = 2 * Math.PI * v;
    this.spareNormal = mag * Math.sin(angle);
    return mean + std * mag * Math.cos(angle);
  }

  shuffle(values) {
    const out = [...values];
    for (let i = out.length - 1; i > 0; i -= 1) {
      const j = this.integer(i + 1);
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  }
}
