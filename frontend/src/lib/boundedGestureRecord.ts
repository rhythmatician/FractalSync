/** Retain the start of a diagnostic trace and count entries omitted at its limits. */
export class BoundedGestureRecord<Event, Sample> {
  readonly events: Event[];
  readonly samples: Sample[] = [];
  droppedEvents = 0;
  droppedSamples = 0;

  constructor(initial: Event, private readonly eventLimit = 2048, private readonly sampleLimit = 36000) {
    this.events = [initial];
  }

  addEvent(event: Event): void {
    if (this.events.length < this.eventLimit) this.events.push(event);
    else this.droppedEvents++;
  }

  addSample(sample: Sample): void {
    if (this.samples.length < this.sampleLimit) this.samples.push(sample);
    else this.droppedSamples++;
  }
}
