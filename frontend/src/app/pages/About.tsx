import { motion } from "motion/react";
import { Users, Crosshair, Globe2, BookOpen } from "lucide-react";

const teamStats = [
  { value: "50+", label: "AI Researchers" },
  { value: "10B+", label: "Sources Indexed" },
  { value: "190", label: "Countries Covered" },
  { value: "24/7", label: "Real-time Monitoring" },
];

export function About() {
  return (
    <div className="min-h-screen pt-32 pb-20 overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="text-center mb-24 max-w-4xl mx-auto"
        >
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", delay: 0.3 }}
            className="w-20 h-20 bg-indigo-500/10 rounded-3xl mx-auto mb-8 flex items-center justify-center rotate-3 border border-indigo-500/20"
          >
            <Users className="w-10 h-10 text-indigo-400 -rotate-3" />
          </motion.div>
          <h1 className="text-5xl md:text-7xl font-black text-white mb-8 tracking-tighter">
            Defending <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400">Truth</span>
          </h1>
          <p className="text-xl md:text-2xl text-neutral-400 leading-relaxed">
            VeritasAI was founded by independent journalists, AI ethicists, and cybersecurity experts with a unified mission: restoring integrity to the digital landscape.
          </p>
        </motion.div>

        {/* Vision & Mission Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 mb-32">
          {[
            {
              icon: <Crosshair className="w-8 h-8 text-cyan-400" />,
              title: "Our Mission",
              desc: "To equip everyday users and organizations with military-grade truth-verification tools. We believe that an informed public is the bedrock of a functioning society."
            },
            {
              icon: <Globe2 className="w-8 h-8 text-indigo-400" />,
              title: "Global Reach",
              desc: "Disinformation knows no borders. Our multi-lingual natural language models monitor thousands of networks simultaneously, translating and contextualizing threats instantly."
            }
          ].map((item, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: i % 2 === 0 ? -50 : 50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-100px" }}
              transition={{ duration: 0.8 }}
              className="bg-neutral-900/50 border border-neutral-800 p-10 rounded-[2rem] hover:bg-neutral-900 transition-colors"
            >
              <div className="bg-neutral-950 w-16 h-16 rounded-2xl flex items-center justify-center mb-8 border border-neutral-800">
                {item.icon}
              </div>
              <h3 className="text-3xl font-bold text-white mb-4">{item.title}</h3>
              <p className="text-lg text-neutral-400 leading-relaxed">{item.desc}</p>
            </motion.div>
          ))}
        </div>

        {/* Stats Section */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="bg-indigo-600 rounded-[3rem] p-12 md:p-20 relative overflow-hidden text-center shadow-2xl"
        >
          <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1557683316-973673baf926?auto=format&fit=crop&q=80')] mix-blend-overlay opacity-50 bg-cover bg-center" />
          <div className="relative z-10 grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-4">
            {teamStats.map((stat, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, scale: 0.5 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.1, type: "spring", bounce: 0.5 }}
              >
                <div className="text-4xl md:text-6xl font-black text-white mb-2 tracking-tighter drop-shadow-md">{stat.value}</div>
                <div className="text-indigo-200 font-semibold tracking-wide uppercase text-sm md:text-base">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </motion.div>
        
        {/* Transparency Section */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 1 }}
          className="mt-32 text-center max-w-3xl mx-auto"
        >
          <BookOpen className="w-12 h-12 text-neutral-600 mx-auto mb-6" />
          <h2 className="text-3xl md:text-5xl font-bold text-white mb-6 tracking-tight">Radical Transparency</h2>
          <p className="text-xl text-neutral-400 leading-relaxed">
            Our models are audited monthly by independent third parties. We open-source our core architecture because the tools of truth should belong to everyone.
          </p>
        </motion.div>
      </div>
    </div>
  );
}
