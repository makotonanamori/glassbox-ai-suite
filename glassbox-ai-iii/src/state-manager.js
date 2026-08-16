export class OperationLog {
  constructor() {
    this.entries = [];
    this.sequence = 0;
  }

  add(message) {
    this.sequence += 1;
    const entry = {
      sequence: this.sequence,
      timestamp: new Date().toISOString(),
      message,
    };
    this.entries.push(entry);
    return entry;
  }

  clear() {
    this.entries = [];
    this.sequence = 0;
  }

  toText() {
    return this.entries
      .map((entry) => `${String(entry.sequence).padStart(4, '0')} ${entry.timestamp} ${entry.message}`)
      .join('\n');
  }

  toJson() {
    return JSON.stringify(
      {
        format: 'glassbox-ai-log',
        version: 1,
        exportedAt: new Date().toISOString(),
        entries: this.entries,
      },
      null,
      2,
    );
  }
}

export function downloadTextFile(filename, text, mimeType = 'text/plain;charset=utf-8') {
  const blob = new Blob([text], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
