import { NavLink, Outlet, useLocation } from "react-router";
import { motion, AnimatePresence } from "motion/react";
import { ShieldAlert, Menu, X } from "lucide-react";
import { useEffect, useState } from "react";

const navLinks = [
  { name: "Home", path: "/" },
  { name: "Fake News Detector", path: "/fake-news" },
  { name: "Deepfake Detector", path: "/deepfake" },
  { name: "How It Works", path: "/how-it-works" },
  { name: "About", path: "/about" },
  { name: "Contact", path: "/contact" },
];

export function Layout() {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setIsOpen(false);
  }, [location.pathname]);

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 flex flex-col font-sans overflow-x-hidden">
      <motion.header
        className="fixed top-0 left-0 right-0 z-50 bg-neutral-950/80 backdrop-blur-md border-b border-neutral-800"
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            <NavLink to="/" className="flex items-center gap-3 group">
              <motion.div
                whileHover={{ rotate: 180 }}
                transition={{ duration: 0.6, ease: "backOut" }}
              >
                <ShieldAlert className="w-8 h-8 text-indigo-500" />
              </motion.div>
              <span className="text-xl font-bold tracking-tight text-white group-hover:text-indigo-400 transition-colors">
                FN Detector
              </span>
            </NavLink>

            <nav className="hidden md:flex gap-8">
              {navLinks.map((link) => (
                <NavLink
                  key={link.path}
                  to={link.path}
                  className={({ isActive }) =>
                    `relative text-sm font-medium transition-colors ${
                      isActive ? "text-indigo-400" : "text-neutral-400 hover:text-white"
                    }`
                  }
                >
                  {({ isActive }) => (
                    <>
                      {link.name}
                      {isActive ? (
                        <motion.div
                          layoutId="nav-indicator"
                          className="absolute -bottom-7 left-0 right-0 h-0.5 bg-indigo-500"
                          transition={{ type: "spring", stiffness: 300, damping: 30 }}
                        />
                      ) : null}
                    </>
                  )}
                </NavLink>
              ))}
            </nav>

            <button
              className="md:hidden text-neutral-400 hover:text-white transition-colors"
              onClick={() => setIsOpen(!isOpen)}
            >
              {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </motion.header>

      <AnimatePresence>
        {isOpen ? (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden fixed top-20 left-0 right-0 z-40 bg-neutral-900 border-b border-neutral-800 overflow-hidden"
          >
            <div className="px-4 py-6 flex flex-col gap-4">
              {navLinks.map((link) => (
                <NavLink
                  key={link.path}
                  to={link.path}
                  className={({ isActive }) =>
                    `text-lg font-medium px-4 py-2 rounded-lg transition-colors ${
                      isActive ? "bg-indigo-500/10 text-indigo-400" : "text-neutral-400 hover:text-white hover:bg-neutral-800"
                    }`
                  }
                >
                  {link.name}
                </NavLink>
              ))}
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>

      <main className="flex-1 mt-20 relative">
        <Outlet />
      </main>

      <footer className="bg-neutral-950 py-12 border-t border-neutral-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-neutral-500 text-sm">
          <div className="flex justify-center items-center gap-2 mb-4">
            <ShieldAlert className="w-5 h-5 text-neutral-600" />
            <span className="font-semibold text-neutral-400">FN Detector</span>
          </div>
          <p>&copy; {new Date().getFullYear()} FN Detector. Model-backed verification workflow for local deployment.</p>
        </div>
      </footer>
    </div>
  );
}
