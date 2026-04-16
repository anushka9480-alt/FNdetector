import { motion } from "motion/react";
import { Mail, MessageSquare, Send, Phone } from "lucide-react";
import { useState } from "react";

export function Contact() {
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    setTimeout(() => setSubmitted(false), 3000);
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1, delayChildren: 0.2 },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, x: -20 },
    show: { opacity: 1, x: 0, transition: { type: "spring", stiffness: 100 } },
  };

  return (
    <div className="min-h-screen pt-32 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto flex flex-col md:flex-row gap-16 md:items-center">
      {/* Contact Info */}
      <motion.div
        initial="hidden"
        animate="show"
        variants={containerVariants}
        className="flex-1 space-y-8"
      >
        <motion.div variants={itemVariants}>
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 mb-6 font-medium">
            <Mail className="w-4 h-4" /> Get in Touch
          </div>
          <h1 className="text-5xl md:text-7xl font-black text-white tracking-tighter mb-6">
            Let's Start a Conversation.
          </h1>
          <p className="text-xl text-neutral-400 max-w-xl leading-relaxed">
            Whether you are interested in enterprise API access, media partnerships, or have found a bug, our team is ready to respond.
          </p>
        </motion.div>

        <motion.div variants={itemVariants} className="pt-8 grid gap-8">
          <div className="flex items-center gap-6">
            <div className="w-14 h-14 bg-neutral-900 border border-neutral-800 rounded-2xl flex items-center justify-center shrink-0">
              <Mail className="w-6 h-6 text-indigo-400" />
            </div>
            <div>
              <p className="text-neutral-500 font-medium mb-1">Email Us</p>
              <a href="mailto:hello@veritasai.com" className="text-xl font-bold text-white hover:text-indigo-400 transition-colors">
                hello@veritasai.com
              </a>
            </div>
          </div>
          
          <div className="flex items-center gap-6">
            <div className="w-14 h-14 bg-neutral-900 border border-neutral-800 rounded-2xl flex items-center justify-center shrink-0">
              <Phone className="w-6 h-6 text-indigo-400" />
            </div>
            <div>
              <p className="text-neutral-500 font-medium mb-1">Call Support</p>
              <p className="text-xl font-bold text-white tracking-wider">
                1-800-VERITAS
              </p>
            </div>
          </div>
        </motion.div>
      </motion.div>

      {/* Contact Form */}
      <motion.div
        initial={{ opacity: 0, x: 50 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.8, delay: 0.4, type: "spring", bounce: 0.3 }}
        className="flex-1 w-full max-w-lg mx-auto md:mx-0 bg-neutral-900/50 border border-neutral-800 p-8 md:p-12 rounded-[2rem] shadow-2xl backdrop-blur-xl"
      >
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-2">
            <label htmlFor="name" className="text-sm font-medium text-neutral-400 pl-2">Full Name</label>
            <input
              type="text"
              id="name"
              required
              className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-4 text-white placeholder:text-neutral-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
              placeholder="Jane Doe"
            />
          </div>
          
          <div className="space-y-2">
            <label htmlFor="email" className="text-sm font-medium text-neutral-400 pl-2">Email Address</label>
            <input
              type="email"
              id="email"
              required
              className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-4 text-white placeholder:text-neutral-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
              placeholder="jane@example.com"
            />
          </div>

          <div className="space-y-2">
            <label htmlFor="message" className="text-sm font-medium text-neutral-400 pl-2">Message</label>
            <textarea
              id="message"
              required
              rows={4}
              className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-4 text-white placeholder:text-neutral-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none transition-all"
              placeholder="How can we help you?"
            />
          </div>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            disabled={submitted}
            type="submit"
            className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-4 rounded-xl flex items-center justify-center gap-2 transition-colors disabled:opacity-50 disabled:hover:bg-indigo-600 shadow-[0_0_30px_-10px_rgba(79,70,229,0.5)]"
          >
            {submitted ? (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center gap-2"
              >
                Message Sent <MessageSquare className="w-5 h-5" />
              </motion.span>
            ) : (
              <span className="flex items-center gap-2">
                Send Message <Send className="w-5 h-5" />
              </span>
            )}
          </motion.button>
        </form>
      </motion.div>
    </div>
  );
}
