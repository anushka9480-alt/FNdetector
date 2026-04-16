import { motion, useScroll, useTransform } from "motion/react";
import { Search, Database, Fingerprint, Shield, ExternalLink } from "lucide-react";
import { useRef } from "react";

const steps = [
  {
    icon: <Search className="w-10 h-10 text-cyan-400" />,
    title: "1. Data Intake & Processing",
    description:
      "When you submit an article, URL, image, or video, our ingestion engines instantly begin parsing the content. Text is broken down into semantic vectors, while images and videos are analyzed frame-by-frame for spatial abnormalities.",
    color: "from-cyan-500/20 to-transparent",
    image: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80",
  },
  {
    icon: <Database className="w-10 h-10 text-indigo-400" />,
    title: "2. Global Cross-Referencing",
    description:
      "Our AI cross-references claims against thousands of verified databases, academic journals, established media outlets, and real-time public records simultaneously to determine factuality.",
    color: "from-indigo-500/20 to-transparent",
    image: "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&q=80",
  },
  {
    icon: <Fingerprint className="w-10 h-10 text-pink-400" />,
    title: "3. Synthetic Artifact Detection",
    description:
      "For visual media, advanced neural networks scan for microscopic imperfections left behind by generative AI—such as unnatural lighting, pixel bleeding, and inconsistent facial geometries.",
    color: "from-pink-500/20 to-transparent",
    image: "https://images.unsplash.com/photo-1620808240212-32a88d61d1ea?auto=format&fit=crop&q=80",
  },
  {
    icon: <Shield className="w-10 h-10 text-emerald-400" />,
    title: "4. Final Verification Score",
    description:
      "Within seconds, you receive a definitive truth score. This transparent report highlights why a piece of content was flagged, allowing you to trace the original sources and draw your own informed conclusions.",
    color: "from-emerald-500/20 to-transparent",
    image: "https://images.unsplash.com/photo-1507146153580-69a1fe6d8aa1?auto=format&fit=crop&q=80",
  }
];

function StepCard({ step, index }: { step: typeof steps[0], index: number }) {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start end", "center center"],
  });

  const y = useTransform(scrollYProgress, [0, 1], [100, 0]);
  const opacity = useTransform(scrollYProgress, [0, 0.5, 1], [0, 1, 1]);
  const scale = useTransform(scrollYProgress, [0, 1], [0.8, 1]);

  const isEven = index % 2 === 0;

  return (
    <motion.div
      ref={ref}
      style={{ y, opacity }}
      className={`flex flex-col ${isEven ? 'md:flex-row' : 'md:flex-row-reverse'} items-center gap-12 py-24 relative z-10`}
    >
      <div className="flex-1 space-y-6">
        <motion.div
          style={{ scale }}
          className={`w-20 h-20 bg-gradient-to-br ${step.color} border border-neutral-700 rounded-3xl flex items-center justify-center shadow-lg backdrop-blur-sm`}
        >
          {step.icon}
        </motion.div>
        
        <h2 className="text-4xl md:text-5xl font-black text-white tracking-tight">
          {step.title}
        </h2>
        
        <p className="text-xl text-neutral-400 leading-relaxed max-w-xl">
          {step.description}
        </p>
        
        <button className="flex items-center gap-2 text-indigo-400 hover:text-indigo-300 font-medium mt-4 transition-colors group">
          Read Technical Whitepaper 
          <ExternalLink className="w-4 h-4 group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform" />
        </button>
      </div>

      <motion.div
        style={{ scale }}
        className="flex-1 w-full max-w-xl aspect-square md:aspect-[4/3] rounded-[2rem] overflow-hidden relative group"
      >
        <div className={`absolute inset-0 bg-gradient-to-br ${step.color} mix-blend-overlay z-10`} />
        <img
          src={step.image}
          alt={step.title}
          className="w-full h-full object-cover object-center group-hover:scale-105 transition-transform duration-700 ease-out"
        />
        <div className="absolute inset-0 border border-neutral-700/50 rounded-[2rem] z-20 pointer-events-none" />
      </motion.div>
    </motion.div>
  );
}

export function HowItWorks() {
  const containerRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"],
  });

  const lineHeight = useTransform(scrollYProgress, [0, 1], ["0%", "100%"]);

  return (
    <div className="min-h-screen pt-32 pb-32 overflow-hidden bg-neutral-950 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative" ref={containerRef}>
        
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="text-center mb-32 max-w-4xl mx-auto relative z-20"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 mb-8 font-medium tracking-wide">
            Under the Hood
          </div>
          <h1 className="text-5xl md:text-7xl font-black text-white mb-6 tracking-tighter">
            Anatomy of <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400">Verification</span>
          </h1>
          <p className="text-xl md:text-2xl text-neutral-400 leading-relaxed">
            Discover how our advanced multi-modal models dissect and analyze information to separate fact from highly sophisticated fiction.
          </p>
        </motion.div>

        {/* Timeline Line */}
        <div className="absolute left-4 md:left-1/2 top-96 bottom-32 w-0.5 bg-neutral-900 -translate-x-1/2 hidden md:block z-0">
          <motion.div
            style={{ height: lineHeight }}
            className="w-full bg-gradient-to-b from-cyan-500 via-indigo-500 to-pink-500"
          />
        </div>

        {/* Steps */}
        <div className="relative">
          {steps.map((step, index) => (
            <StepCard key={index} step={step} index={index} />
          ))}
        </div>
      </div>
    </div>
  );
}
