import { useState } from "react";
import { motion, AnimatePresence, useReducedMotion } from "motion/react";
import {
  AlertTriangle,
  BookOpenText,
  CheckCircle,
  Info,
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
        text: "The text-only classifier strongly associates this sample with the fake-news class.",
        percent: fakePercent,
      };
    }

    return {
      icon: <AlertTriangle className="w-16 h-16 text-amber-500" />,
      color: "text-amber-500",
      bg: "bg-amber-500/10 border-amber-500/30",
      title: "Suspicious Signals Found",
      text: "The classifier leans fake, but confidence is moderate enough that human review still matters.",
      percent: fakePercent,
    };
  }

  return {
    icon: <CheckCircle className="w-16 h-16 text-emerald-500" />,
    color: "text-emerald-500",
    bg: "bg-emerald-500/10 border-emerald-500/30",
    title: "Model Leaning Real",
    text: "The current text-only checkpoint leans toward the real-news class for this sample. That is not the same thing as verified truth.",
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
          This screen uses the deployed text-only detector for quick screening. It can help flag suspicious language patterns, but it does not verify sources, dates, or real-world claims on its own.
        </p>
      </motion.div>

      <div className="grid grid-cols-1 xl:grid-cols-[1.35fr_0.65fr] gap-8 w-full">
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
                  <div className="mb-6 w-full max-w-xl rounded-2xl border border-neutral-800 bg-neutral-950/70 p-4 text-left">
                    <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-2">Important</p>
                    <p className="text-sm text-neutral-300">
                      If you know this story is fabricated but the model still leans real, that means the current checkpoint is over-trusting the writing style of your sample. It is a text classifier, not a live fact-checking system.
                    </p>
                  </div>
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
            <h3 className="text-xl font-bold text-white mb-4">What This Model Sees</h3>
            <div className="space-y-3 text-neutral-300">
              <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4 flex items-start gap-3">
                <BookOpenText className="w-5 h-5 text-indigo-400 mt-0.5" />
                <p>It looks only at the text you paste. It does not inspect links, publishers, screenshots, or outside evidence.</p>
              </div>
              <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4 flex items-start gap-3">
                <Info className="w-5 h-5 text-cyan-400 mt-0.5" />
                <p>A fabricated article can still score “real” if it uses calm wording, resembles mainstream reporting, or avoids obvious clickbait cues.</p>
              </div>
              <div className="rounded-2xl bg-neutral-950 border border-neutral-800 p-4 flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5" />
                <p>Use this as a screening signal only. For stronger results, the fake-news model needs retraining or a retrieval-based fact-checking layer.</p>
              </div>
            </div>
          </div>
        </motion.aside>
      </div>
    </div>
  );
}
