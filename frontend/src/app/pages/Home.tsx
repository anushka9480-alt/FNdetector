import { motion, useScroll, useTransform } from "motion/react";
import { ArrowRight, ShieldCheck, Zap, Activity } from "lucide-react";
import { Link } from "react-router";
import { useRef } from "react";

const features = [
  {
    icon: <ShieldCheck className="w-8 h-8 text-indigo-400" />,
    title: "Fake News Detection",
    description: "Run the local transformer-backed fake-news detector and inspect the active deployment model.",
    link: "/fake-news",
  },
  {
    icon: <Zap className="w-8 h-8 text-blue-400" />,
    title: "Deepfake Recognition",
    description: "Scan images and videos to detect unnatural artifacts and facial inconsistencies instantly.",
    link: "/deepfake",
  },
  {
    icon: <Activity className="w-8 h-8 text-emerald-400" />,
    title: "Real-Time Tracking",
    description: "Track backend health, model metrics, and the recommended smoke-to-full training sequence.",
    link: "/how-it-works",
  },
];

export function Home() {
  const heroRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ["start start", "end start"],
  });

  const yBackground = useTransform(scrollYProgress, [0, 1], ["0%", "50%"]);
  const opacityBackground = useTransform(scrollYProgress, [0, 0.8], [1, 0]);
  const scaleText = useTransform(scrollYProgress, [0, 1], [1, 0.8]);
  const yText = useTransform(scrollYProgress, [0, 1], ["0%", "100%"]);

  return (
    <div className="relative overflow-hidden w-full">
      {/* Hero Section */}
      <section ref={heroRef} className="relative h-[90vh] flex items-center justify-center overflow-hidden">
        {/* Parallax Background */}
        <motion.div
          style={{ y: yBackground, opacity: opacityBackground }}
          className="absolute inset-0 z-0"
        >
          <div className="absolute inset-0 bg-neutral-950/80 z-10" />
          <img
            src="https://images.unsplash.com/photo-1626447837522-a62b9f75a5dd?auto=format&fit=crop&q=80"
            alt="Hero Background"
            className="w-full h-full object-cover object-center scale-110"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-neutral-950 via-neutral-900/50 to-transparent z-10" />
        </motion.div>

        {/* Hero Content */}
        <motion.div
          style={{ scale: scaleText, y: yText }}
          className="relative z-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center mt-[-10vh]"
        >
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, ease: "easeOut" }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 mb-8"
          >
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-indigo-500"></span>
            </span>
            Protecting truth in the digital era
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.2, ease: "easeOut" }}
            className="text-6xl md:text-8xl font-extrabold tracking-tight text-white mb-6"
          >
            Truth, <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400">Verified.</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.4, ease: "easeOut" }}
            className="text-xl md:text-2xl text-neutral-300 max-w-3xl mx-auto mb-10 leading-relaxed"
          >
            Advanced AI models designed to detect fake news before it spreads. The frontend now connects directly to the backend workflow for prediction, training, and model monitoring.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.6, ease: "easeOut" }}
            className="flex flex-col sm:flex-row gap-4 justify-center items-center"
          >
            <Link to="/fake-news">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-8 py-4 rounded-full bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-lg flex items-center gap-2 transition-colors shadow-[0_0_40px_-10px_rgba(79,70,229,0.5)]"
              >
                Start Detecting
                <ArrowRight className="w-5 h-5" />
              </motion.button>
            </Link>
            <Link to="/how-it-works">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-8 py-4 rounded-full bg-neutral-800 hover:bg-neutral-700 text-white font-semibold text-lg transition-colors border border-neutral-700 hover:border-neutral-600"
              >
                Learn How It Works
              </motion.button>
            </Link>
          </motion.div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section className="relative z-30 py-32 bg-neutral-950">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.8 }}
            className="text-center mb-20"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">Cutting-Edge Detection</h2>
            <p className="text-xl text-neutral-400 max-w-2xl mx-auto">
              Equip yourself with the tools to spot disinformation, altered media, and propaganda effortlessly.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {features.map((feature, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 50 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-100px" }}
                transition={{ duration: 0.6, delay: i * 0.2 }}
                whileHover={{ y: -10 }}
                className="bg-neutral-900 border border-neutral-800 rounded-3xl p-8 hover:bg-neutral-800/80 transition-colors group"
              >
                <div className="bg-neutral-950 w-16 h-16 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300 border border-neutral-800">
                  {feature.icon}
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">{feature.title}</h3>
                <p className="text-neutral-400 mb-8 leading-relaxed">
                  {feature.description}
                </p>
                <Link to={feature.link} className="inline-flex items-center text-indigo-400 font-medium hover:text-indigo-300 transition-colors group/link">
                  Explore tool 
                  <ArrowRight className="w-4 h-4 ml-2 group-hover/link:translate-x-1 transition-transform" />
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
      
      {/* Stats Parallax Section */}
      <StatsParallax />
    </div>
  );
}

function StatsParallax() {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start end", "end start"],
  });

  const y = useTransform(scrollYProgress, [0, 1], ["20%", "-20%"]);

  return (
    <section ref={ref} className="relative h-[60vh] overflow-hidden flex items-center justify-center">
      <motion.div style={{ y }} className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-indigo-900/40 mix-blend-multiply z-10" />
        <img
          src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&q=80"
          alt="Parallax background"
          className="w-full h-[150%] object-cover object-center scale-110"
        />
        <div className="absolute inset-0 bg-neutral-950/80 z-10" />
      </motion.div>

      <div className="relative z-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 text-center">
          {[
            { value: "99.8%", label: "Detection Accuracy" },
            { value: "10M+", label: "Articles Scanned Daily" },
            { value: "0.2s", label: "Average Response Time" },
          ].map((stat, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, scale: 0.5 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, delay: i * 0.2, type: "spring", bounce: 0.4 }}
              className="flex flex-col items-center justify-center"
            >
              <div className="text-5xl md:text-7xl font-black text-transparent bg-clip-text bg-gradient-to-br from-white to-indigo-300 mb-4 drop-shadow-lg">
                {stat.value}
              </div>
              <div className="text-xl text-indigo-200 font-medium tracking-wide uppercase">
                {stat.label}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
