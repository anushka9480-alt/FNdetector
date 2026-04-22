import { useEffect, useState } from "react";
import { motion, AnimatePresence, useReducedMotion } from "motion/react";
import {
  AlertTriangle,
  CheckCircle,
  LoaderCircle,
  RefreshCw,
  Search,
  ShieldAlert,
  XCircle,
} from "lucide-react";

type PredictionResponse = {
  prediction: "real" | "fake";
  scores: {
    real: number;
    fake: number;
  };
};

type PredictApiPayload = PredictionResponse | {
  status?: string;
  prediction?: PredictionResponse;
};

type WorkflowPreset = {
  key: string;
  label: string;
  description: string;
  available: boolean;
  default_epochs: number;
};

type WorkflowResponse = {
  recommended_sequence: string[];
  presets: WorkflowPreset[];
};

type HealthResponse = {
  status: string;
  active_model: string | null;
  running_jobs: number;
};

type ModelResponse = {
  available: boolean;
  active_model: null | {
    path: string;
    preset: string | null;
    metrics: {
      accuracy?: number;
      f1_fake?: number;
    } | null;
  };
};

type TrainJob = {
  job_id: string;
  status: string;
  preset: string;
  output_dir: string;
  error: string | null;
};

function getResultDetails(result: PredictionResponse | null) {
  if (!result) {
    return null;
  }

  if (result.prediction === "fake") {
    const fakePercent = Math.round(result.scores.fake * 100);
    if (fakePercent > 75) {
      return {
        icon: <XCircle className="w-16 h-16 text-red-500" />,
        color: "text-red-500",
        bg: "bg-red-500/10 border-red-500/30",
        title: "High Risk of Disinformation",
        text: "The model strongly associates this text with the fake-news class.",
        percent: fakePercent,
      };
    }

    return {
      icon: <AlertTriangle className="w-16 h-16 text-amber-500" />,
      color: "text-amber-500",
      bg: "bg-amber-500/10 border-amber-500/30",
      title: "Suspicious Signals Found",
      text: "The model leans fake, but confidence is moderate enough that human review still matters.",
      percent: fakePercent,
    };
  }

  return {
    icon: <CheckCircle className="w-16 h-16 text-emerald-500" />,
    color: "text-emerald-500",
    bg: "bg-emerald-500/10 border-emerald-500/30",
    title: "More Likely Credible",
    text: "The model currently leans toward the real-news class for this sample.",
    percent: Math.round(result.scores.real * 100),
  };
}

function normalizePredictionResponse(payload: PredictApiPayload): PredictionResponse | null {
  if ("scores" in payload && payload.scores) {
    return payload as PredictionResponse;
  }

  if ("prediction" in payload && payload.prediction?.scores) {
    return payload.prediction;
  }

  return null;
}

export function FakeNews() {
  const reduceMotion = useReducedMotion();
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

  const [input, setInput] = useState("");
  const [status, setStatus] = useState<"idle" | "scanning" | "result">("idle");
  const [error, setError] = useState("");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [workflow, setWorkflow] = useState<WorkflowResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [model, setModel] = useState<ModelResponse | null>(null);
  const [trainJob, setTrainJob] = useState<TrainJob | null>(null);
  const [isTraining, setIsTraining] = useState(false);

  async function loadBackendState() {
    try {
      const [workflowResponse, healthResponse, modelResponse] = await Promise.all([
        fetch(`${apiBaseUrl}/api/workflow`),
        fetch(`${apiBaseUrl}/api/health`),
        fetch(`${apiBaseUrl}/api/model`),
      ]);

      if (workflowResponse.ok) {
        setWorkflow(await workflowResponse.json());
      }
      if (healthResponse.ok) {
        setHealth(await healthResponse.json());
      }
      if (modelResponse.ok) {
        setModel(await modelResponse.json());
      }
    } catch {
      setError("Backend is not reachable yet. Start the API server to enable prediction and training.");
    }
  }

  useEffect(() => {
    void loadBackendState();
  }, []);

  useEffect(() => {
    if (!trainJob || trainJob.status === "completed" || trainJob.status === "failed") {
      return;
    }

    const timer = window.setInterval(() => {
      void fetch(`${apiBaseUrl}/api/train/${trainJob.job_id}`)
        .then((response) => response.json())
        .then((payload: TrainJob) => {
          setTrainJob(payload);
          if (payload.status === "completed" || payload.status === "failed") {
            setIsTraining(false);
            void loadBackendState();
          }
        })
        .catch(() => undefined);
    }, 3000);

    return () => window.clearInterval(timer);
  }, [apiBaseUrl, trainJob]);

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
      if (!response.ok) {
        throw new Error((payload as { detail?: string; error?: string }).detail ?? (payload as { detail?: string; error?: string }).error ?? "Prediction failed.");
      }

      const normalizedPrediction = normalizePredictionResponse(payload);
      if (!normalizedPrediction) {
        throw new Error("Prediction response was missing score data.");
      }

      setPrediction(normalizedPrediction);
      setStatus("result");
    } catch (requestError) {
      const message =
        requestError instanceof Error ? requestError.message : "Prediction failed.";
      setError(message);
      setStatus("idle");
    }
  }

  async function handleTrain(preset: string) {
    setIsTraining(true);
    setError("");

    try {
      const response = await fetch(`${apiBaseUrl}/api/train`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ preset }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "Training request failed.");
      }

      setTrainJob(payload);
    } catch (requestError) {
      const message =
        requestError instanceof Error ? requestError.message : "Training request failed.";
      setError(message);
      setIsTraining(false);
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
  const activeModel = model?.active_model;

  return (
    <div className="min-h-screen pt-32 pb-20 px-4 sm:px-6 lg:px-8 max-w-6xl mx-auto flex flex-col items-center">
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
          This screen is connected to the local backend workflow. Run live predictions, inspect the
          active model, and trigger smoke, quick, or full training presets without leaving the app.
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
                <span className="text-sm text-neutral-500 font-medium">
                  Backend: {health?.status ?? "offline"}
                </span>
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
              className="w-full"
            >
              {details ? (
                <div className={`border ${details.bg} rounded-3xl p-10 flex flex-col items-center text-center shadow-xl backdrop-blur-sm`}>
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: reduceMotion ? 0.2 : 0.3 }}
                  >
                    {details.icon}
                  </motion.div>
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
                      className={`h-full ${prediction?.prediction === "fake" ? "bg-amber-500" : "bg-emerald-500"}`}
                    />
                  </div>
                  <p className="text-lg text-neutral-300 mb-4 max-w-xl mx-auto leading-relaxed">
                    {details.text}
                  </p>
                  <div className="grid grid-cols-2 gap-4 w-full max-w-xl mb-8">
                    <div className="rounded-2xl bg-neutral-950/80 border border-neutral-800 p-4">
                      <p className="text-sm text-neutral-500 uppercase tracking-wide">Real Score</p>
                      <p className="text-3xl font-black text-white mt-2">
                        {Math.round((prediction?.scores.real ?? 0) * 100)}%
                      </p>
                    </div>
                    <div className="rounded-2xl bg-neutral-950/80 border border-neutral-800 p-4">
                      <p className="text-sm text-neutral-500 uppercase tracking-wide">Fake Score</p>
                      <p className="text-3xl font-black text-white mt-2">
                        {Math.round((prediction?.scores.fake ?? 0) * 100)}%
                      </p>
                    </div>
                  </div>
                  <motion.button
                    whileHover={reduceMotion ? undefined : { scale: 1.03 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={reset}
                    className="flex items-center gap-2 px-6 py-3 rounded-full bg-neutral-800 hover:bg-neutral-700 text-white font-semibold transition-colors border border-neutral-700 hover:border-neutral-600"
                  >
                    <RefreshCw className="w-5 h-5" />
                    Scan Another Source
                  </motion.button>
                </div>
              ) : null}
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
            <h3 className="text-xl font-bold text-white mb-4">Active Backend State</h3>
            <div className="space-y-4">
              <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">API</p>
                <p className="font-semibold text-white">{apiBaseUrl}</p>
              </div>
              <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">Active Model</p>
                <p className="font-semibold text-white break-all">{activeModel?.path ?? health?.active_model ?? "none"}</p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">Accuracy</p>
                  <p className="text-2xl font-black text-white">
                    {activeModel?.metrics?.accuracy !== undefined
                      ? activeModel.metrics.accuracy.toFixed(2)
                      : "--"}
                  </p>
                </div>
                <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">F1 Fake</p>
                  <p className="text-2xl font-black text-white">
                    {activeModel?.metrics?.f1_fake !== undefined
                      ? activeModel.metrics.f1_fake.toFixed(2)
                      : "--"}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-neutral-900 border border-neutral-800 rounded-3xl p-6">
            <h3 className="text-xl font-bold text-white mb-4">Training Workflow</h3>
            <p className="text-neutral-400 mb-5">
              The backend exposes smoke, quick, and full training presets and updates the active
              model when a job completes.
            </p>
            <div className="space-y-3">
              {workflow?.presets.map((preset) => (
                <button
                  key={preset.key}
                  type="button"
                  onClick={() => handleTrain(preset.key)}
                  disabled={!preset.available || isTraining}
                  className="w-full text-left rounded-2xl border border-neutral-800 bg-neutral-950 px-4 py-4 hover:border-indigo-500/40 hover:bg-neutral-900 transition-colors disabled:opacity-50"
                >
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <p className="font-bold text-white">{preset.label}</p>
                      <p className="text-sm text-neutral-500 mt-1">{preset.description}</p>
                    </div>
                    <span className="text-xs uppercase tracking-[0.2em] text-indigo-400">
                      {preset.key}
                    </span>
                  </div>
                </button>
              ))}
            </div>
            {trainJob ? (
              <div className="mt-5 rounded-2xl border border-neutral-800 bg-neutral-950 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">Latest Training Job</p>
                <p className="font-semibold text-white">{trainJob.job_id}</p>
                <p className="text-neutral-400 mt-2">
                  {trainJob.preset} {"->"} {trainJob.status}
                </p>
                {trainJob.error ? <p className="text-red-300 mt-2">{trainJob.error}</p> : null}
              </div>
            ) : null}
          </div>
        </motion.aside>
      </div>
    </div>
  );
}
