export interface AnimalClass {
  id: string;
  name: string;
  group: string;
}

export interface Prediction {
  label: string;
  confidence: number;
}

export interface Sample {
  id: string;
  name: string;
  photo: string;
  palette: string[];
  accent: string;
}

export interface ImageSource {
  kind: "sample" | "upload";
  id: string;
  name: string;
  canvas: HTMLCanvasElement;
}

export interface InferenceResponse {
  maskedInputBase64: string;
  maeReconBase64: string;
  unetReconBase64: string;
  maeMse: number;
  unetMse: number;
  betterModel: "mae" | "unet";
  predictions: Prediction[];
  device: string;
  latencyMs: number;
}

export type AppStep = 0 | 1 | 2;
