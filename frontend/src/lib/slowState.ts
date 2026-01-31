export type BeatClockState = {
  t_sec: number;
  spb: number;
  phase: number;
  beat_count: number;
  conf: number;
};

export type StructureState = {
  section_probs: number[];
  hazard_probs: number[];
};

export type SlowState = {
  beat: BeatClockState;
  structure: StructureState;
  features: number[];
};
