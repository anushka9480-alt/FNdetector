import { useEffect, useState } from "react";
import { motion, AnimatePresence, useReducedMotion } from "motion/react";
import {
  AlertTriangle,
  BarChart3,
  BookOpenText,
  CheckCircle,
  Info,
  LoaderCircle,
  RefreshCw,
  Search,
  ShieldAlert,
  Target,
  XCircle,
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

type HistoryEntry = {
  epoch: number;
  train_loss: number;
  train?: MetricSet;
  validation?: MetricSet;
};

type MetricsResponse = {
  status: string;
  snapshot?: {
    train_rows?: number;
    validation_rows?: number;
    test_rows?: number;
  };
  report?: {
    best_epoch?: number;
    history?: HistoryEntry[];
    targets?: {
      accuracy?: number;
      confidence?: number;
      reached?: boolean;
    };
    test_metrics?: MetricSet;
    best_validation_metrics?: MetricSet;
  };
};

type PredictionResponse = {
  prediction: "real" | "fake" | "uncertain";
  confidence?: number;
  model_label?: "real" | "fake";
  calibration?: {
    temperature: number;
    threshold: number;
    uncertainty_margin: number;
  };
  fact_check_signals?: {
    source_signal: string;
    date_signal: string;
    entity_signal: string;
    risk_level: string;
    trusted_mentions: string[];
    low_trust_mentions: string[];
    date_mentions: string[];
    named_entities: string[];
    risk_cues: string[];
    recommended_checks: string[];
    retrieval_needed: boolean;
  };
  scores: {
    real: number;
    fake: number;
  };
};

type PredictApiPayload = {
  status?: string;
  prediction?: PredictionResponse;
  model_metrics?: MetricSet;
};

const performanceChartConfig = {
  accuracy: { label: "Accuracy", color: "#38bdf8" },
  precision: { label: "Precision", color: "#a78bfa" },
  recall: { label: "Recall", color: "#f59e0b" },
  f1: { label: "F1", color: "#34d399" },
  confidence: { label: "Confidence", color: "#f472b6" },
};

const historyChartConfig = {
  validationAccuracy: { label: "Validation Accuracy", color: "#38bdf8" },
  validationF1: { label: "Validation F1", color: "#34d399" },
  trainLoss: { label: "Train Loss", color: "#f97316" },
};

function formatPercent(value?: number) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

function getResultDetails(result: PredictionResponse | null) {
  if (!result) {
    return null;
  }

  if (result.prediction === "uncertain") {
    return {
      icon: <AlertTriangle className="w-16 h-16 text-amber-400" />,
      color: "text-amber-400",
      bg: "bg-amber-500/10 border-amber-500/30",
      title: "Needs Review",
      text: "The calibrated classifier could not separate real and fake strongly enough, so this sample needs manual review.",
      percent: Math.round(Math.max(result.scores.real, result.scores.fake) * 100),
    };
  }

  if (result.prediction === "fake") {
    const fakePercent = Math.round(result.scores.fake * 100);
    if (fakePercent > 75) {
      return {
        icon: <XCircle className="w-16 h-16 text-red-500" />,
        color: "text-red-500",
        bg: "bg-red-500/10 border-red-500/30",
        title: "High Risk of Disinformation",
        text: "The classifier strongly associates this sample with the fake-news class.",
        percent: fakePercent,
      };
    }

    return {
      icon: <AlertTriangle className="w-16 h-16 text-amber-500" />,
      color: "text-amber-500",
      bg: "bg-amber-500/10 border-amber-500/30",
      title: "Suspicious Signals Found",
      text: "The classifier leans fake, but the margin is not strong enough to treat as a final fact-check verdict.",
      percent: fakePercent,
    };
  }

  return {
    icon: <CheckCircle className="w-16 h-16 text-emerald-500" />,
    color: "text-emerald-500",
    bg: "bg-emerald-500/10 border-emerald-500/30",
    title: "Model Leaning Real",
    text: "The current checkpoint leans toward the real-news class for this sample. That still is not the same thing as verified truth.",
    percent: Math.round(result.scores.real * 100),
  };
}

export function FakeNews() {
  const reduceMotion = useReducedMotion();
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "https://Harman823-fndetector-backend.hf.space";

  const [input, setInput] = useState("");
  const [status, setStatus] = useState<"idle" | "scanning" | "result">("idle");
  const [error, setError] = useState("");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [modelMetrics, setModelMetrics] = useState<MetricSet | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);

  useEffect(() => {
    void fetch(`${apiBaseUrl}/api/metrics`)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error("Unable to reach the fake-news metrics endpoint.");
        }
        return response.json();
      })
      .then((payload: MetricsResponse) => {
        setMetrics(payload);
        setModelMetrics(payload.report?.test_metrics ?? null);
      })
      .catch(() => undefined);
  }, [apiBaseUrl]);

  async function handleScan() {
    const trimmed = input.trim();
    if (!trimmed) {
      setError("Enter article text or a headline before running detection.");
      return;
    }

    setError("");
    setStatus("scanning");
    setPrediction(null);

    try {
      const response = await fetch(`${apiBaseUrl}/api/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: trimmed }),
      });

      const payload: PredictApiPayload = await response.json();
      if (!response.ok || !payload.prediction) {
        throw new Error("Prediction failed.");
      }

      setPrediction(payload.prediction);
      setModelMetrics(payload.model_metrics ?? metrics?.report?.test_metrics ?? null);
      setStatus("result");
    } catch (requestError) {
      const message =
        requestError instanceof Error ? requestError.message : "Prediction failed.";
      setError(message);
      setStatus("idle");
    }
  }

  function reset() {
    setStatus("idle");
    setPrediction(null);
    setError("");
  }

  function fillSample() {
    setInput(
      "Officials denied the viral claim after fact checkers flagged the screenshot as manipulated and unsupported by any primary source."
    );
    setError("");
  }

  const details = getResultDetails(prediction);
  const factSignals = prediction?.fact_check_signals;
  const evaluationMetrics = modelMetrics ?? metrics?.report?.test_metrics ?? null;
  const performanceData = evaluationMetrics
    ? [
        evaluationMetrics.accuracy !== undefined ? { key: "accuracy", label: "Accuracy", value: Number((evaluationMetrics.accuracy * 100).toFixed(1)) } : null,
        evaluationMetrics.precision_fake !== undefined ? { key: "precision", label: "Precision", value: Number((evaluationMetrics.precision_fake * 100).toFixed(1)) } : null,
        evaluationMetrics.recall_fake !== undefined ? { key: "recall", label: "Recall", value: Number((evaluationMetrics.recall_fake * 100).toFixed(1)) } : null,
        evaluationMetrics.f1_fake !== undefined ? { key: "f1", label: "F1", value: Number((evaluationMetrics.f1_fake * 100).toFixed(1)) } : null,
        evaluationMetrics.avg_confidence !== undefined ? { key: "confidence", label: "Confidence", value: Number((evaluationMetrics.avg_confidence * 100).toFixed(1)) } : null,
      ].filter(Boolean) as Array<{ key: string; label: string; value: number }>
    : [];
  const historyData = (metrics?.report?.history ?? []).map((entry) => ({
    epoch: `E${entry.epoch}`,
    validationAccuracy: Number(((entry.validation?.accuracy ?? 0) * 100).toFixed(1)),
    validationF1: Number(((entry.validation?.f1_fake ?? 0) * 100).toFixed(1)),
    trainLoss: Number((entry.train_loss ?? 0).toFixed(3)),
  }));
  const confusionMatrix = evaluationMetrics?.confusion_matrix ?? [[0, 0], [0, 0]];

  return (
    <div className="min-h-screen pt-32 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto flex flex-col items-center">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: reduceMotion ? 0.2 : 0.4 }}
        className="text-center mb-12 w-full"
      >
        <ShieldAlert className="w-16 h-16 text-indigo-500 mx-auto mb-6" />
        <h1 className="text-4xl md:text-6xl font-black text-white mb-6 tracking-tight">
          Fact Check Engine
        </h1>
        <p className="text-xl text-neutral-400 max-w-3xl mx-auto">
          Run the text detector, then inspect the verdict together with held-out evaluation statistics, confidence, and training history.
        </p>
      </motion.div>

      <div className="grid grid-cols-1 xl:grid-cols-[1.3fr_0.7fr] gap-8 w-full">
        <AnimatePresence mode="wait">
          {status !== "result" ? (
            <motion.div
              key={status}
              initial={{ opacity: 0, scale: 0.97 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: reduceMotion ? 0.2 : 0.3 }}
              className="w-full bg-neutral-900 border border-neutral-800 rounded-3xl p-6 shadow-2xl relative"
            >
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="Paste article text or a headline here..."
                className="w-full h-56 bg-neutral-950 border border-neutral-800 rounded-2xl p-6 text-white placeholder:text-neutral-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none text-lg transition-shadow"
              />

              <div className="mt-6 flex flex-wrap gap-3 justify-between items-center px-1">
                <div className="flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={handleScan}
                    disabled={!input.trim() || status === "scanning"}
                    className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white px-8 py-3 rounded-xl font-bold flex items-center gap-2 transition-all shadow-[0_0_20px_-5px_rgba(79,70,229,0.5)]"
                  >
                    {status === "scanning" ? (
                      <LoaderCircle className="w-5 h-5 animate-spin" />
                    ) : (
                      <Search className="w-5 h-5" />
                    )}
                    {status === "scanning" ? "Analyzing" : "Analyze Now"}
                  </button>
                  <button
                    type="button"
                    onClick={fillSample}
                    className="px-6 py-3 rounded-xl bg-neutral-800 hover:bg-neutral-700 text-white font-semibold transition-colors border border-neutral-700"
                  >
                    Load Sample
                  </button>
                </div>
                <span className="text-sm text-neutral-500 font-medium">Text-only screening</span>
              </div>

              {status === "scanning" ? (
                <div className="mt-8 flex flex-col items-center justify-center py-12">
                  <div className="relative w-24 h-24 flex items-center justify-center">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1.8, repeat: Infinity, ease: "linear" }}
                      className="absolute inset-0 rounded-full border-t-4 border-l-4 border-indigo-500 border-r-4 border-transparent border-b-4 border-transparent opacity-80"
                    />
                    <Search className="w-8 h-8 text-white" />
                  </div>
                  <p className="mt-6 text-lg font-semibold text-neutral-300">
                    Running model inference against the active checkpoint...
                  </p>
                </div>
              ) : null}

              {error ? (
                <div className="mt-6 rounded-2xl border border-red-500/30 bg-red-500/10 p-4 text-red-200">
                  {error}
                </div>
              ) : null}
            </motion.div>
          ) : (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: reduceMotion ? 0.2 : 0.35 }}
              className="w-full space-y-6"
            >
              {details ? (
                <div className={`border ${details.bg} rounded-3xl p-10 flex flex-col items-center text-center shadow-xl backdrop-blur-sm`}>
                  {details.icon}
                  <h2 className="text-3xl font-black text-white mt-6 mb-2">{details.title}</h2>
                  <p className={`text-6xl font-black my-6 tracking-tighter ${details.color}`}>
                    {details.percent}
                    <span className="text-3xl opacity-50">%</span>
                  </p>
                  <div className="w-full bg-neutral-900 h-4 rounded-full overflow-hidden mb-6">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${details.percent}%` }}
                      transition={{ duration: reduceMotion ? 0.25 : 0.8, ease: "easeOut" }}
                      className={`h-full ${prediction?.prediction === "fake" ? "bg-amber-500" : prediction?.prediction === "uncertain" ? "bg-yellow-400" : "bg-emerald-500"}`}
                    />
                  </div>
                  <p className="text-lg text-neutral-300 mb-8 max-w-2xl mx-auto leading-relaxed">
                    {details.text}
                  </p>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full">
                    <div className="rounded-2xl bg-neutral-950/80 border border-neutral-800 p-4 text-left">
                      <p className="text-sm text-neutral-500 uppercase tracking-wide">Real Score</p>
                      <p className="text-3xl font-black text-white mt-2">{formatPercent(prediction?.scores.real)}</p>
                    </div>
                    <div className="rounded-2xl bg-neutral-950/80 border border-neutral-800 p-4 text-left">
                      <p className="text-sm text-neutral-500 uppercase tracking-wide">Fake Score</p>
                      <p className="text-3xl font-black text-white mt-2">{formatPercent(prediction?.scores.fake)}</p>
                    </div>
                    <div className="rounded-2xl bg-neutral-950/80 border border-neutral-800 p-4 text-left">
                      <p className="text-sm text-neutral-500 uppercase tracking-wide">Prediction Confidence</p>
                      <p className="text-3xl font-black text-white mt-2">{formatPercent(prediction?.confidence)}</p>
                    </div>
                    <div className="rounded-2xl bg-neutral-950/80 border border-neutral-800 p-4 text-left">
                      <p className="text-sm text-neutral-500 uppercase tracking-wide">Model Test Accuracy</p>
                      <p className="text-3xl font-black text-white mt-2">{formatPercent(evaluationMetrics?.accuracy)}</p>
                    </div>
                  </div>
                </div>
              ) : null}

              <div className="grid grid-cols-1 2xl:grid-cols-[0.95fr_1.05fr] gap-6">
                <div className="rounded-3xl border border-neutral-800 bg-neutral-900 p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <Target className="w-5 h-5 text-cyan-400" />
                    <h3 className="text-xl font-bold text-white">Prediction Statistics</h3>
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
                  <div className="flex items-center gap-3 mb-4">
                    <BarChart3 className="w-5 h-5 text-fuchsia-400" />
                    <h3 className="text-xl font-bold text-white">Training Trend</h3>
                  </div>
                  <ChartContainer config={historyChartConfig} className="h-[320px] w-full">
                    <LineChart data={historyData}>
                      <CartesianGrid vertical={false} strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" tickLine={false} axisLine={false} />
                      <YAxis yAxisId="score" domain={[0, 100]} tickLine={false} axisLine={false} />
                      <YAxis yAxisId="loss" orientation="right" tickLine={false} axisLine={false} />
                      <ChartTooltip content={<ChartTooltipContent indicator="line" />} />
                      <Legend content={<ChartLegendContent />} />
                      <Line yAxisId="score" type="monotone" dataKey="validationAccuracy" stroke="var(--color-validationAccuracy)" strokeWidth={3} dot={false} />
                      <Line yAxisId="score" type="monotone" dataKey="validationF1" stroke="var(--color-validationF1)" strokeWidth={3} dot={false} />
                      <Line yAxisId="loss" type="monotone" dataKey="trainLoss" stroke="var(--color-trainLoss)" strokeWidth={2} strokeDasharray="6 4" dot={false} />
                    </LineChart>
                  </ChartContainer>
                </div>
              </div>

              <div className="grid grid-cols-1 2xl:grid-cols-[0.9fr_1.1fr] gap-6">
                <div className="rounded-3xl border border-neutral-800 bg-neutral-900 p-6">
                  <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-3">Confusion Matrix</p>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-2xl border border-neutral-800 bg-neutral-950/80 p-4">
                      <p className="text-sm text-neutral-500">True Fake -&gt; Pred Fake</p>
                      <p className="text-3xl font-black text-white mt-2">{confusionMatrix[0]?.[0] ?? 0}</p>
                    </div>
                    <div className="rounded-2xl border border-neutral-800 bg-neutral-950/80 p-4">
                      <p className="text-sm text-neutral-500">True Fake -&gt; Pred Real</p>
                      <p className="text-3xl font-black text-white mt-2">{confusionMatrix[0]?.[1] ?? 0}</p>
                    </div>
                    <div className="rounded-2xl border border-neutral-800 bg-neutral-950/80 p-4">
                      <p className="text-sm text-neutral-500">True Real -&gt; Pred Fake</p>
                      <p className="text-3xl font-black text-white mt-2">{confusionMatrix[1]?.[0] ?? 0}</p>
                    </div>
                    <div className="rounded-2xl border border-neutral-800 bg-neutral-950/80 p-4">
                      <p className="text-sm text-neutral-500">True Real -&gt; Pred Real</p>
                      <p className="text-3xl font-black text-white mt-2">{confusionMatrix[1]?.[1] ?? 0}</p>
                    </div>
                  </div>
                </div>

                <div className="rounded-3xl border border-neutral-800 bg-neutral-900 p-6">
                  <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-3">Fact-Check Signals</p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm text-neutral-300">
                    <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-3">
                      <p className="text-neutral-500 mb-1">Source signal</p>
                      <p>{factSignals?.source_signal ?? "unknown"}</p>
                    </div>
                    <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-3">
                      <p className="text-neutral-500 mb-1">Date signal</p>
                      <p>{factSignals?.date_signal ?? "unknown"}</p>
                    </div>
                    <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-3">
                      <p className="text-neutral-500 mb-1">Entity signal</p>
                      <p>{factSignals?.entity_signal ?? "unknown"}</p>
                    </div>
                    <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-3">
                      <p className="text-neutral-500 mb-1">Risk level</p>
                      <p>{factSignals?.risk_level ?? "unknown"}</p>
                    </div>
                  </div>
                  {factSignals?.recommended_checks?.length ? (
                    <div className="mt-4 rounded-2xl border border-neutral-800 bg-neutral-950/70 p-4">
                      <p className="text-neutral-500 text-sm mb-2">Recommended checks</p>
                      <ul className="space-y-2 text-sm text-neutral-300">
                        {factSignals.recommended_checks.map((item) => (
                          <li key={item}>- {item}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                </div>
              </div>

              <motion.button
                whileHover={reduceMotion ? undefined : { scale: 1.03 }}
                whileTap={{ scale: 0.98 }}
                onClick={reset}
                className="flex items-center gap-2 px-6 py-3 rounded-full bg-neutral-800 hover:bg-neutral-700 text-white font-semibold transition-colors border border-neutral-700 hover:border-neutral-600 w-fit"
              >
                <RefreshCw className="w-5 h-5" />
                Scan Another Source
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
            <h3 className="text-xl font-bold text-white mb-4">Model Snapshot</h3>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">Train Rows</p>
                <p className="text-2xl font-black text-white">{metrics?.snapshot?.train_rows ?? "--"}</p>
              </div>
              <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">Validation Rows</p>
                <p className="text-2xl font-black text-white">{metrics?.snapshot?.validation_rows ?? "--"}</p>
              </div>
              <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">Test Rows</p>
                <p className="text-2xl font-black text-white">{metrics?.snapshot?.test_rows ?? "--"}</p>
              </div>
              <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">Best Epoch</p>
                <p className="text-2xl font-black text-white">{metrics?.report?.best_epoch ?? "--"}</p>
              </div>
            </div>
            <div className="mt-4 rounded-2xl bg-neutral-950 border border-neutral-800 p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">Target Gate</p>
              <p className="text-white font-semibold">
                Accuracy {formatPercent(metrics?.report?.targets?.accuracy)} / Confidence {formatPercent(metrics?.report?.targets?.confidence)}
              </p>
              <p className={`text-sm mt-2 ${metrics?.report?.targets?.reached ? "text-emerald-400" : "text-amber-400"}`}>
                {metrics?.report?.targets?.reached ? "Validation targets reached" : "Validation targets not yet reached"}
              </p>
            </div>
          </div>

          <div className="bg-neutral-900 border border-neutral-800 rounded-3xl p-6">
            <h3 className="text-xl font-bold text-white mb-4">What This Model Sees</h3>
            <div className="space-y-3 text-neutral-300">
              <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4 flex items-start gap-3">
                <BookOpenText className="w-5 h-5 text-indigo-400 mt-0.5" />
                <p>It looks only at the text you paste. It does not inspect linked evidence, publishers, or screenshots.</p>
              </div>
              <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4 flex items-start gap-3">
                <Info className="w-5 h-5 text-cyan-400 mt-0.5" />
                <p>A fabricated article can still score real if it mimics mainstream reporting style and avoids obvious clickbait language.</p>
              </div>
              <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4 flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5" />
                <p>Use this as a screening signal. The numbers shown here are dataset-level evaluation metrics, not proof that one article is true.</p>
              </div>
            </div>
          </div>
        </motion.aside>
      </div>
    </div>
  );
}
