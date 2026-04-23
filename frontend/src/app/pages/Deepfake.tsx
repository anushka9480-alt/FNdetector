import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Eye,
  Image as ImageIcon,
  LoaderCircle,
  ScanFace,
  ShieldAlert,
  Sparkles,
  Target,
  UploadCloud,
} from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, XAxis, YAxis } from "recharts";

import {
  ChartContainer,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "../components/ui/chart";

type MetricSet = {
  accuracy?: number;
  precision_fake?: number;
  recall_fake?: number;
  f1_fake?: number;
  avg_confidence?: number;
  confusion_matrix?: number[][];
};

type DeepfakePredictionResponse = {
  prediction: "real" | "fake";
  confidence: number;
  model_name: string;
  model_status: string;
  model_metrics?: MetricSet;
  scores: {
    real: number;
    fake: number;
  };
  top_signals: Array<{
    name: string;
    value: number;
  }>;
  image_size: {
    width: number;
    height: number;
  };
};

type DeepfakeMetricsResponse = {
  status: string;
  snapshot: {
    available: boolean;
    feature_names: string[];
    summary: {
      status?: string;
      model_name?: string;
      accuracy?: number;
      avg_confidence?: number;
      precision_fake?: number;
      recall_fake?: number;
      f1_fake?: number;
      dataset_rows?: number;
      train_rows?: number;
      validation_rows?: number;
      test_rows?: number;
      max_images_per_label?: number;
      best_epoch?: number;
      target_reached?: boolean;
      history?: Array<{
        epoch: number;
        train?: MetricSet;
        validation?: MetricSet;
      }>;
      test_metrics?: MetricSet;
    };
  };
};

const performanceChartConfig = {
  accuracy: { label: "Accuracy", color: "#22d3ee" },
  precision: { label: "Precision", color: "#fb7185" },
  recall: { label: "Recall", color: "#f59e0b" },
  f1: { label: "F1", color: "#34d399" },
  confidence: { label: "Confidence", color: "#a78bfa" },
};

const historyChartConfig = {
  validationAccuracy: { label: "Validation Accuracy", color: "#22d3ee" },
  validationF1: { label: "Validation F1", color: "#34d399" },
  validationConfidence: { label: "Validation Confidence", color: "#a78bfa" },
};

function formatPercent(value?: number) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

export function Deepfake() {
  const reduceMotion = useReducedMotion();
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "https://Harman823-fndetector-backend.hf.space";

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [status, setStatus] = useState<"idle" | "uploading" | "scanning" | "result">("idle");
  const [error, setError] = useState("");
  const [result, setResult] = useState<DeepfakePredictionResponse | null>(null);
  const [metrics, setMetrics] = useState<DeepfakeMetricsResponse | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    void fetch(`${apiBaseUrl}/api/deepfake/metrics`)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error("Unable to reach the deepfake detector backend.");
        }
        return response.json();
      })
      .then((payload: DeepfakeMetricsResponse) => {
        setMetrics(payload);
      })
      .catch(() => undefined);
  }, [apiBaseUrl]);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0];
    if (!nextFile) {
      return;
    }

    if (!nextFile.type.startsWith("image/")) {
      setError("This detector currently accepts image files only. Upload a JPG, PNG, or WEBP file.");
      setStatus("idle");
      setFile(null);
      setResult(null);
      return;
    }

    setError("");
    setFile(nextFile);
    setResult(null);
    setStatus("uploading");

    try {
      const formData = new FormData();
      formData.append("file", nextFile);

      setStatus("scanning");
      const response = await fetch(`${apiBaseUrl}/api/deepfake/predict`, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json();
      if (!response.ok || !payload.prediction) {
        throw new Error(payload.detail ?? "Deepfake prediction failed.");
      }

      setResult(payload.prediction as DeepfakePredictionResponse);
      setStatus("result");
    } catch (requestError) {
      const message =
        requestError instanceof Error ? requestError.message : "Deepfake prediction failed.";
      setError(message);
      setStatus("idle");
    }
  };

  const reset = () => {
    setStatus("idle");
    setFile(null);
    setError("");
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const fakePercent = Math.round((result?.scores.fake ?? 0) * 100);
  const realPercent = Math.round((result?.scores.real ?? 0) * 100);
  const isFake = result?.prediction === "fake";
  const verdictTitle = isFake ? "Synthetic Signals Detected" : "Looks More Natural";
  const verdictCopy = isFake
    ? "The detector found artifact patterns that lean toward manipulation. Treat this as a screening result, not a final forensic verdict."
    : "The uploaded image looks closer to the real class under this detector, though manual verification still matters for high-stakes use.";

  const evaluationMetrics = result?.model_metrics ?? metrics?.snapshot.summary.test_metrics ?? {
    accuracy: metrics?.snapshot.summary.accuracy,
    avg_confidence: metrics?.snapshot.summary.avg_confidence,
    precision_fake: metrics?.snapshot.summary.precision_fake,
    recall_fake: metrics?.snapshot.summary.recall_fake,
    f1_fake: metrics?.snapshot.summary.f1_fake,
  };
  const performanceData = evaluationMetrics
    ? [
        evaluationMetrics.accuracy !== undefined ? { key: "accuracy", label: "Accuracy", value: Number((evaluationMetrics.accuracy * 100).toFixed(1)) } : null,
        evaluationMetrics.precision_fake !== undefined ? { key: "precision", label: "Precision", value: Number((evaluationMetrics.precision_fake * 100).toFixed(1)) } : null,
        evaluationMetrics.recall_fake !== undefined ? { key: "recall", label: "Recall", value: Number((evaluationMetrics.recall_fake * 100).toFixed(1)) } : null,
        evaluationMetrics.f1_fake !== undefined ? { key: "f1", label: "F1", value: Number((evaluationMetrics.f1_fake * 100).toFixed(1)) } : null,
        evaluationMetrics.avg_confidence !== undefined ? { key: "confidence", label: "Confidence", value: Number((evaluationMetrics.avg_confidence * 100).toFixed(1)) } : null,
      ].filter(Boolean) as Array<{ key: string; label: string; value: number }>
    : [];
  const historyData = (metrics?.snapshot.summary.history ?? []).map((entry) => ({
    epoch: `E${entry.epoch}`,
    validationAccuracy: Number(((entry.validation?.accuracy ?? 0) * 100).toFixed(1)),
    validationF1: Number(((entry.validation?.f1_fake ?? 0) * 100).toFixed(1)),
    validationConfidence: Number(((entry.validation?.avg_confidence ?? 0) * 100).toFixed(1)),
  }));
  const confusionMatrix = evaluationMetrics?.confusion_matrix ?? [[0, 0], [0, 0]];

  return (
    <div className="min-h-screen pt-32 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto flex flex-col items-center">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: reduceMotion ? 0.2 : 0.4 }}
        className="text-center mb-16 w-full"
      >
        <ScanFace className="w-16 h-16 text-cyan-400 mx-auto mb-6" />
        <h1 className="text-4xl md:text-6xl font-black text-white mb-6 tracking-tight">
          Deepfake Detector
        </h1>
        <p className="text-xl text-neutral-400 max-w-3xl mx-auto">
          Upload an image, inspect the synthetic-vs-natural verdict, and review evaluation metrics and training curves from the deployed detector bundle.
        </p>
      </motion.div>

      <div className="grid grid-cols-1 xl:grid-cols-[1.1fr_0.9fr] gap-8 w-full">
        <AnimatePresence mode="wait">
          {status === "idle" && (
            <motion.div
              key="idle"
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: reduceMotion ? 0.2 : 0.35 }}
              className="w-full bg-neutral-900 border-2 border-dashed border-neutral-700 hover:border-cyan-500 rounded-3xl p-12 text-center cursor-pointer transition-colors group relative overflow-hidden min-h-[540px] flex flex-col justify-center"
              onClick={handleUploadClick}
            >
              <div className="absolute inset-0 bg-cyan-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />

              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept="image/*"
                onChange={handleFileChange}
              />

              <UploadCloud className="w-24 h-24 text-neutral-600 group-hover:text-cyan-400 transition-colors mx-auto mb-8" />

              <h3 className="text-3xl font-bold text-white mb-4">Drop in an image to analyze</h3>
              <p className="text-neutral-500 text-lg mb-8">or click to browse your files (JPG, PNG, WEBP)</p>

              <div className="flex justify-center gap-4 flex-wrap">
                <div className="flex items-center gap-2 text-neutral-400 bg-neutral-950 px-4 py-2 rounded-xl border border-neutral-800">
                  <ImageIcon className="w-5 h-5 text-cyan-400" /> Image upload
                </div>
                <div className="flex items-center gap-2 text-neutral-400 bg-neutral-950 px-4 py-2 rounded-xl border border-neutral-800">
                  <ShieldAlert className="w-5 h-5 text-amber-400" /> CPU-safe model
                </div>
              </div>

              {error ? (
                <div className="mt-8 rounded-2xl border border-red-500/30 bg-red-500/10 p-4 text-red-200 max-w-2xl mx-auto">
                  {error}
                </div>
              ) : null}
            </motion.div>
          )}

          {status === "uploading" && (
            <motion.div
              key="uploading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center w-full min-h-[540px] bg-neutral-900 rounded-3xl border border-neutral-800"
            >
              <div className="w-full max-w-md bg-neutral-900 h-4 rounded-full overflow-hidden mb-6 border border-neutral-800">
                <motion.div
                  initial={{ width: "0%" }}
                  animate={{ width: "100%" }}
                  transition={{ duration: reduceMotion ? 0.4 : 1.2, ease: "easeInOut" }}
                  className="h-full bg-cyan-500"
                />
              </div>
              <p className="text-xl text-cyan-400 font-medium animate-pulse">Uploading securely...</p>
            </motion.div>
          )}

          {status === "scanning" && (
            <motion.div
              key="scanning"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.02 }}
              className="w-full h-full relative bg-neutral-900 rounded-3xl border border-neutral-800 min-h-[540px] flex items-center justify-center p-8"
            >
              <div className="relative rounded-3xl overflow-hidden border border-neutral-800 mx-auto w-full max-w-2xl bg-neutral-950 aspect-square flex items-center justify-center">
                {previewUrl ? (
                  <img src={previewUrl} alt="Upload preview" className="absolute inset-0 w-full h-full object-cover opacity-40 blur-[2px]" />
                ) : null}

                <motion.div
                  initial={{ top: "-10%" }}
                  animate={{ top: "110%" }}
                  transition={{ duration: reduceMotion ? 1.2 : 2, repeat: Infinity, ease: "linear" }}
                  className="absolute left-0 right-0 h-1 bg-cyan-400 shadow-[0_0_20px_5px_rgba(34,211,238,0.5)] z-20"
                />

                <div className="absolute inset-0 z-10 grid grid-cols-6 grid-rows-4">
                  {[...Array(24)].map((_, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: [0, 0.45, 0] }}
                      transition={{ duration: 1, repeat: Infinity, delay: index * 0.03 }}
                      className="border border-cyan-500/20"
                    />
                  ))}
                </div>

                <div className="relative z-30 flex flex-col items-center text-center">
                  <LoaderCircle className="w-12 h-12 text-cyan-400 animate-spin mb-4" />
                  <h3 className="text-2xl font-bold text-white tracking-widest">ANALYZING ARTIFACTS</h3>
                  <p className="text-cyan-500 font-mono mt-2">Checking compression and frequency signals...</p>
                </div>
              </div>
            </motion.div>
          )}

          {status === "result" && result && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: reduceMotion ? 0.2 : 0.35 }}
              className="w-full flex flex-col gap-6"
            >
              <div className="relative rounded-3xl overflow-hidden border border-neutral-800 bg-neutral-900 group min-h-[380px]">
                {previewUrl ? (
                  <img src={previewUrl} alt={file?.name ?? "Uploaded image"} className="w-full h-full object-cover" />
                ) : null}
                <div className={`absolute inset-0 ${isFake ? "bg-gradient-to-tr from-red-950/40 via-red-500/20 to-transparent" : "bg-gradient-to-tr from-emerald-950/40 via-emerald-500/20 to-transparent"}`} />
                <div className="absolute left-4 bottom-4 right-4 rounded-2xl border border-white/10 bg-neutral-950/80 backdrop-blur p-4">
                  <p className="text-xs uppercase tracking-[0.25em] text-neutral-400 mb-2">Uploaded image</p>
                  <p className="text-white font-semibold break-all">{file?.name}</p>
                  <p className="text-sm text-neutral-400 mt-2">
                    {result.image_size.width} x {result.image_size.height}px
                  </p>
                </div>
              </div>

              <div className={`bg-neutral-900 border ${isFake ? "border-red-500/30" : "border-emerald-500/30"} rounded-3xl p-8 relative overflow-hidden`}>
                <div className={`absolute top-0 right-0 w-32 h-32 ${isFake ? "bg-red-500/5" : "bg-emerald-500/5"} rounded-bl-full`} />
                {isFake ? (
                  <Eye className="w-12 h-12 text-red-500 mb-6" />
                ) : (
                  <CheckCircle2 className="w-12 h-12 text-emerald-500 mb-6" />
                )}

                <h2 className="text-4xl font-black text-white mb-2 tracking-tight">{verdictTitle}</h2>
                <p className="text-xl text-neutral-400 mb-8">{verdictCopy}</p>

                <div className="space-y-6">
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-neutral-300 font-medium">Synthetic Probability</span>
                      <span className={`font-bold ${isFake ? "text-red-400" : "text-emerald-400"}`}>{fakePercent}%</span>
                    </div>
                    <div className="w-full bg-neutral-950 h-3 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${fakePercent}%` }}
                        transition={{ duration: reduceMotion ? 0.25 : 1.1, ease: "circOut" }}
                        className={`h-full ${isFake ? "bg-gradient-to-r from-amber-500 to-red-500" : "bg-gradient-to-r from-cyan-500 to-emerald-500"}`}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-neutral-300 font-medium">Natural Probability</span>
                      <span className="text-cyan-300 font-bold">{realPercent}%</span>
                    </div>
                    <div className="w-full bg-neutral-950 h-3 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${realPercent}%` }}
                        transition={{ duration: reduceMotion ? 0.25 : 1.1, delay: 0.1, ease: "circOut" }}
                        className="h-full bg-cyan-500"
                      />
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
                  <div className="rounded-2xl bg-neutral-950/80 border border-neutral-800 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">Confidence</p>
                    <p className="text-2xl font-black text-white">{Math.round(result.confidence * 100)}%</p>
                  </div>
                  <div className="rounded-2xl bg-neutral-950/80 border border-neutral-800 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">Model Accuracy</p>
                    <p className="text-2xl font-black text-white">{formatPercent(evaluationMetrics?.accuracy)}</p>
                  </div>
                  <div className="rounded-2xl bg-neutral-950/80 border border-neutral-800 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">Model</p>
                    <p className="text-lg font-bold text-white">{result.model_name}</p>
                  </div>
                  <div className="rounded-2xl bg-neutral-950/80 border border-neutral-800 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">Mode</p>
                    <p className="text-lg font-bold text-white capitalize">{result.model_status}</p>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 2xl:grid-cols-[0.95fr_1.05fr] gap-6">
                <div className="rounded-3xl border border-neutral-800 bg-neutral-900 p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <Target className="w-5 h-5 text-cyan-400" />
                    <h3 className="text-xl font-bold text-white">Model Statistics</h3>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                    {performanceData.map((item) => (
                      <div key={item.key} className="rounded-2xl border border-neutral-800 bg-neutral-950/80 p-4">
                        <p className="text-xs uppercase tracking-[0.2em] text-neutral-500">{item.label}</p>
                        <p className="text-2xl font-black text-white mt-2">{item.value}%</p>
                      </div>
                    ))}
                  </div>
                  <ChartContainer config={performanceChartConfig} className="mt-6 h-[280px] w-full">
                    <BarChart data={performanceData}>
                      <CartesianGrid vertical={false} strokeDasharray="3 3" />
                      <XAxis dataKey="label" tickLine={false} axisLine={false} />
                      <YAxis domain={[0, 100]} tickLine={false} axisLine={false} />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <Bar dataKey="value" radius={[12, 12, 0, 0]}>
                        {performanceData.map((item) => (
                          <Cell key={item.key} fill={`var(--color-${item.key})`} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ChartContainer>
                </div>

                <div className="rounded-3xl border border-neutral-800 bg-neutral-900 p-6">
                  <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-3">Training Trend</p>
                  <ChartContainer config={historyChartConfig} className="h-[320px] w-full">
                    <LineChart data={historyData}>
                      <CartesianGrid vertical={false} strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" tickLine={false} axisLine={false} />
                      <YAxis domain={[0, 100]} tickLine={false} axisLine={false} />
                      <ChartTooltip content={<ChartTooltipContent indicator="line" />} />
                      <Legend content={<ChartLegendContent />} />
                      <Line type="monotone" dataKey="validationAccuracy" stroke="var(--color-validationAccuracy)" strokeWidth={3} dot={false} />
                      <Line type="monotone" dataKey="validationF1" stroke="var(--color-validationF1)" strokeWidth={3} dot={false} />
                      <Line type="monotone" dataKey="validationConfidence" stroke="var(--color-validationConfidence)" strokeWidth={3} dot={false} />
                    </LineChart>
                  </ChartContainer>
                </div>
              </div>

              <div className="grid grid-cols-1 2xl:grid-cols-[0.95fr_1.05fr] gap-6">
                <div className="rounded-3xl border border-neutral-800 bg-neutral-900 p-6">
                  <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-3">Confusion Matrix</p>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-2xl border border-neutral-800 bg-neutral-950/80 p-4">
                      <p className="text-sm text-neutral-500">True Real -&gt; Pred Real</p>
                      <p className="text-3xl font-black text-white mt-2">{confusionMatrix[0]?.[0] ?? 0}</p>
                    </div>
                    <div className="rounded-2xl border border-neutral-800 bg-neutral-950/80 p-4">
                      <p className="text-sm text-neutral-500">True Real -&gt; Pred Fake</p>
                      <p className="text-3xl font-black text-white mt-2">{confusionMatrix[0]?.[1] ?? 0}</p>
                    </div>
                    <div className="rounded-2xl border border-neutral-800 bg-neutral-950/80 p-4">
                      <p className="text-sm text-neutral-500">True Fake -&gt; Pred Real</p>
                      <p className="text-3xl font-black text-white mt-2">{confusionMatrix[1]?.[0] ?? 0}</p>
                    </div>
                    <div className="rounded-2xl border border-neutral-800 bg-neutral-950/80 p-4">
                      <p className="text-sm text-neutral-500">True Fake -&gt; Pred Fake</p>
                      <p className="text-3xl font-black text-white mt-2">{confusionMatrix[1]?.[1] ?? 0}</p>
                    </div>
                  </div>
                </div>

                <div className="rounded-3xl border border-neutral-800 bg-neutral-900 p-6">
                  <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-3">Top Signals</p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {result.top_signals.map((signal) => (
                      <div key={signal.name} className="rounded-2xl border border-neutral-800 bg-neutral-950/70 p-4">
                        <p className="text-sm text-neutral-400">{signal.name.replaceAll("_", " ")}</p>
                        <p className="text-xl font-bold text-white mt-1">{signal.value.toFixed(4)}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <motion.button
                whileHover={reduceMotion ? undefined : { scale: 1.03 }}
                whileTap={{ scale: 0.95 }}
                onClick={reset}
                className="w-fit py-4 px-6 rounded-xl bg-neutral-800 hover:bg-neutral-700 text-white font-bold transition-colors border border-neutral-700 flex items-center justify-center gap-2"
              >
                <Camera className="w-5 h-5" />
                Scan Another File
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>

        <motion.aside
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: reduceMotion ? 0.2 : 0.35 }}
          className="space-y-6"
        >
          <div className="bg-neutral-900 border border-neutral-800 rounded-3xl p-6">
            <h3 className="text-xl font-bold text-white mb-4">Detector Profile</h3>
            <p className="text-neutral-400 mb-5">
              This route now uses a real vision backbone and exposes held-out evaluation statistics instead of only a raw synthetic score.
            </p>
            <div className="space-y-3">
              <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">API</p>
                <p className="font-semibold text-white break-all">{apiBaseUrl}</p>
              </div>
              <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">Model</p>
                <p className="font-semibold text-white">{metrics?.snapshot.summary.model_name ?? "lightweight-deepfake-linear"}</p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">Features</p>
                  <p className="text-2xl font-black text-white">{metrics?.snapshot.feature_names.length ?? "--"}</p>
                </div>
                <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">Samples</p>
                  <p className="text-2xl font-black text-white">{metrics?.snapshot.summary.dataset_rows ?? "--"}</p>
                </div>
                <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">Best Epoch</p>
                  <p className="text-2xl font-black text-white">{metrics?.snapshot.summary.best_epoch ?? "--"}</p>
                </div>
                <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">Target Gate</p>
                  <p className={`text-sm font-bold ${metrics?.snapshot.summary.target_reached ? "text-emerald-400" : "text-amber-400"}`}>
                    {metrics?.snapshot.summary.target_reached ? "Reached" : "Not reached"}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-neutral-900 border border-neutral-800 rounded-3xl p-6">
            <h3 className="text-xl font-bold text-white mb-4">Laptop-Safe Settings</h3>
            <div className="space-y-3 text-neutral-300">
              <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4 flex items-start gap-3">
                <Sparkles className="w-5 h-5 text-cyan-400 mt-0.5" />
                <p>The current bundle stays CPU-friendly by using a frozen MobileNet image backbone with a lightweight trained classifier head.</p>
              </div>
              <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4 flex items-start gap-3">
                <CheckCircle2 className="w-5 h-5 text-emerald-400 mt-0.5" />
                <p>The UI now surfaces accuracy, precision, recall, F1, confidence, and validation history directly beside each scan.</p>
              </div>
              <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4 flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5" />
                <p>High metrics still depend on the dataset quality and class balance, so deployment should follow the refreshed Kaggle-backed bundle rather than the older heuristic artifact.</p>
              </div>
            </div>
          </div>

          <div className="bg-neutral-900 border border-neutral-800 rounded-3xl p-6">
            <h3 className="text-xl font-bold text-white mb-4">Sample Test Image</h3>
            <p className="text-neutral-400 mb-4">Use this bundled sample image to smoke-test the detector after deployment.</p>
            <a
              href="/samples/deepfake-sample.png"
              target="_blank"
              rel="noreferrer"
              className="block rounded-2xl overflow-hidden border border-neutral-800 bg-neutral-950 hover:border-cyan-500/60 transition-colors"
            >
              <img
                src="/samples/deepfake-sample.png"
                alt="Sample deepfake test"
                className="w-full h-56 object-cover"
              />
            </a>
          </div>
        </motion.aside>
      </div>
    </div>
  );
}
