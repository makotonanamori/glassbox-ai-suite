function hashSeed(seed) {
  const text = String(seed);
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export class SeededRandom {
  constructor(seed) {
    this.seed = String(seed);
    this.state = hashSeed(this.seed);
  }

  next() {
    let value = (this.state += 0x6d2b79f5);
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  }

  uniform(min, max) {
    return min + (max - min) * this.next();
  }
}

export function createSeededRandom(seed) {
  return new SeededRandom(seed);
}
