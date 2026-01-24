import JSZip from 'jszip';

export interface RecorderMetadata {
  data_dir?: string;
  julia_resolution?: number;
  julia_max_iter?: number;
  timestamp?: string;
  notes?: string;
}

export interface RecorderRecord {
  t: number;
  c: [number, number];
  controller: any;
  h: number | null;
  band_energies: number[] | null;
  audio_features: number[] | null;
  proxy_frame_name?: string;
  deltaV?: number | null;
  notes?: string | null;
}

export class FrontendFlightRecorder {
  private runId: string;
  private metadata: RecorderMetadata | null = null;
  private records: RecorderRecord[] = [];
  private images: Map<string, Blob> = new Map();
  private stepCounter = 0;

  constructor(runId?: string) {
    this.runId = runId || `frontend_run_${Date.now()}`;
  }

  startRun(meta?: RecorderMetadata) {
    this.metadata = meta || {};
    this.records = [];
    this.images = new Map();
    this.stepCounter = 0;
  }

  async recordStep(record: Omit<RecorderRecord, 't' | 'proxy_frame_name'>, proxyBlob?: Blob) {
    const t = this.stepCounter++;
    const rec: RecorderRecord = { t, ...record };

    if (proxyBlob) {
      const name = `${String(t).padStart(6, '0')}.png`;
      this.images.set(name, proxyBlob);
      rec.proxy_frame_name = `proxy_frames/${name}`;
    }

    this.records.push(rec);
  }

  // Return an in-memory zip Blob of the run
  async exportZip(): Promise<Blob> {
    const zip = new JSZip();

    // records.ndjson: metadata as first line then record per line
    const ndjsonLines: string[] = [];
    ndjsonLines.push(JSON.stringify({ _meta: true, metadata: this.metadata || {} }));
    for (const r of this.records) ndjsonLines.push(JSON.stringify(r));
    zip.file('records.ndjson', ndjsonLines.join('\n') + '\n');

    // add images
    for (const [name, blob] of this.images.entries()) {
      zip.file(`proxy_frames/${name}`, blob);
    }

    const content = await zip.generateAsync({ type: 'blob', compression: 'DEFLATE' });
    return content;
  }

  // Upload the current run Blob to backend via fetch
  async uploadToServer(endpoint: string = '/api/flight_recorder/upload'): Promise<Response> {
    const zipBlob = await this.exportZip();
    const form = new FormData();
    const filename = `${this.runId}.zip`;
    form.append('file', zipBlob, filename);
    return fetch(endpoint, { method: 'POST', body: form });
  }

  getRecordCount(): number {
    return this.records.length;
  }

  getRunId(): string {
    return this.runId;
  }
}
