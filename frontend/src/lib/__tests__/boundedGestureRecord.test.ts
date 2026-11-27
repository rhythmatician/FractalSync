import { describe, expect, it } from 'vitest';
import { BoundedGestureRecord } from '../boundedGestureRecord';

describe('gesture diagnostic recording', () => {
  it('limits retained events and samples and reports omitted entries', () => {
    const record = new BoundedGestureRecord<string, number>('start', 2, 2);
    record.addEvent('change');
    record.addEvent('stop');
    record.addSample(1);
    record.addSample(2);
    record.addSample(3);
    expect(record.events).toEqual(['start', 'change']);
    expect(record.samples).toEqual([1, 2]);
    expect(record.droppedEvents).toBe(1);
    expect(record.droppedSamples).toBe(1);
  });
});
